#!/usr/bin/env python3
"""R4 (EEF pose estimators over 256 steps) and R6 (action scale vs the Wan training data). CPU only.

R6: per-dimension |action| percentiles of the Wan training split (cws_v1 manifest, recorded step_e actions) vs the
    actions pi0.5 executed in the L1 rollouts.
R4: estimators of the 8-d proprio state (eef pos, eef rotation, gripper qpos) along each L1 episode, all causal
    (only the keyframe's real state at reset, the executed actions, and Wan frames seen so far):
      read   state reader on the last Wan frame of each chunk (what the policy is given today)
      int    action integration from the reset state with a per-step linear model fitted on step_e train
             (pos / rotation increments linear in the current and previous action; gripper first-order lag)
      int_l1 same model fitted on L1 itself, 2-fold split by start keyframe (on-policy fit, for comparison)
      fuse_k int corrected towards read at every chunk end with gain k (complementary filter)
    Errors at chunk ends vs the real state: position (cm), rotation geodesic angle (deg), gripper (mean |qpos|).
    Per-step estimates for every episode are saved for the R3 retrieval replay.

  python r4_r6_pose_action.py --out DIR [--fit-episodes 600]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

R = "/mnt/cpfs/jieduan"
ROLL = f"{R}/data/cws_rollout"
CHUNK = 8


# ------------------------------------------------------------------ data
def step_e_episode(rec):
    """Recorded trajectory -> (actions [T,7], states [T+1, 3+4+2]: pos, quat xyzw, gripper qpos)."""
    j = json.load(open(os.path.join(ROLL, rec["json"])))
    # trajectory[t].observation is the state after trajectory[t].action. initial_observation is the root's start,
    # which for a counterfactual branch is not the state before its first transition, so it is not used.
    obs = [s["observation"] for s in j["trajectory"]]
    if len(obs) < 9:
        raise ValueError(f"only {len(obs)} transitions")
    st = np.array([np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"]]) for o in obs])
    return np.array([s["action"] for s in j["trajectory"][1:]], np.float64).reshape(-1, 7), st


def l1_episode(path):
    z = np.load(path)
    rs = z["real_state"].astype(np.float64)  # [1 + n, 8]: pos, axis-angle, gripper
    n = len(rs) - 1
    a = z["actions"].reshape(-1, 7)[:n].astype(np.float64)
    q = Rot.from_rotvec(rs[:, 3:6]).as_quat()
    st = np.concatenate([rs[:, :3], q, rs[:, 6:8]], 1)
    return dict(name=os.path.basename(path), kf=int(z["kf"]), a=a, st=st, read=z["wan_read"].astype(np.float64))


# ------------------------------------------------------------------ integrator
def features(a):
    prev = np.vstack([np.zeros((1, 7)), a[:-1]])
    return np.concatenate([a[:, :6], prev[:, :6], np.ones((len(a), 1))], 1)  # 13


def fit(episodes):
    """Least squares for pos / rotation increments (world frame) and the gripper lag, pooled over episodes."""
    X, Yp, Yr, G, Gy = [], [], [], [], []
    for a, st in episodes:
        X.append(features(a))
        Yp.append(np.diff(st[:, :3], axis=0))
        r = Rot.from_quat(st[:, 3:7])
        Yr.append((r[1:] * r[:-1].inv()).as_rotvec())  # R_{t+1} = Exp(w) R_t
        g = st[:, 7:9]
        G.append(np.concatenate([g[:-1], a[:, 6:7], np.ones((len(a), 1))], 1))
        Gy.append(np.diff(g, axis=0))
    X, Yp, Yr = np.concatenate(X), np.concatenate(Yp), np.concatenate(Yr)
    G, Gy = np.concatenate(G), np.concatenate(Gy)
    Wp = np.linalg.lstsq(X, Yp, rcond=None)[0]
    Wr = np.linalg.lstsq(X, Yr, rcond=None)[0]
    Wg = np.linalg.lstsq(G, Gy, rcond=None)[0]
    r2 = lambda Y, P: 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return dict(Wp=Wp, Wr=Wr, Wg=Wg, r2_pos=r2(Yp, X @ Wp).tolist(), r2_rot=r2(Yr, X @ Wr).tolist(),
                r2_grip=r2(Gy, G @ Wg).tolist(), n_steps=int(len(X)))


def integrate(model, st0, a, read=None, k=0.0):
    """Per-step estimate [T+1, 9]; with read (reader output after each step) and k > 0, the estimate is pulled
    towards the reading at every chunk end: x <- x + k (read - x)."""
    F = features(a)
    dp, dw = F @ model["Wp"], F @ model["Wr"]
    out = np.zeros((len(a) + 1, 9))
    out[0] = st0
    p, r, g = st0[:3].copy(), Rot.from_quat(st0[3:7]), st0[7:9].copy()
    for t in range(len(a)):
        p = p + dp[t]
        r = Rot.from_rotvec(dw[t]) * r
        g = g + np.concatenate([g, a[t, 6:7], [1.0]]) @ model["Wg"]
        if read is not None and k > 0 and (t + 1) % CHUNK == 0 and t < len(read):
            m = read[t]
            p = p + k * (m[:3] - p)
            rm = Rot.from_rotvec(m[3:6])
            r = Rot.from_rotvec(k * (rm * r.inv()).as_rotvec()) * r
            g = g + k * (m[6:8] - g)
        out[t + 1] = np.concatenate([p, r.as_quat(), g])
    return out


def read_track(ep):
    """Reader estimate held between chunk ends (the policy's view), as [T+1, 9]; step 0 = the real reset state."""
    T = len(ep["a"])
    out = np.zeros((T + 1, 9))
    out[0] = ep["st"][0]
    cur = ep["st"][0]
    for t in range(T):
        if (t + 1) % CHUNK == 0 and t < len(ep["read"]):
            m = ep["read"][t]
            cur = np.concatenate([m[:3], Rot.from_rotvec(m[3:6]).as_quat(), m[6:8]])
        out[t + 1] = cur
    return out


def errors(est, st):
    pos = np.linalg.norm(est[:, :3] - st[:, :3], axis=1) * 100
    rot = np.degrees((Rot.from_quat(est[:, 3:7]) * Rot.from_quat(st[:, 3:7]).inv()).magnitude())
    grip = np.abs(est[:, 7:9] - st[:, 7:9]).mean(1)
    return pos, rot, grip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-episodes", type=int, default=600)
    ap.add_argument("--l1", default=f"{R}/work/scgrpo/causal/l1[ab]/dump/arrays/*.npz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = [json.loads(l) for l in open(f"{ROLL}/index/cws_v1/manifest.jsonl")]
    train = [r for r in rows if r["split"] == "train"]
    pick = [train[i] for i in rng.choice(len(train), min(args.fit_episodes, len(train)), replace=False)]
    se = []
    for r in pick:
        try:
            se.append(step_e_episode(r))
        except Exception as exc:  # noqa: BLE001
            print("skip", r["idx"], exc)
    l1 = [l1_episode(f) for f in sorted(glob.glob(args.l1))]
    res = {"n_fit_step_e": len(se), "n_l1": len(l1), "n_l1_keyframes": len({e["kf"] for e in l1})}

    # ---- R6: action scale
    A_tr = np.concatenate([a for a, _ in se])
    A_l1 = np.concatenate([e["a"] for e in l1])
    q = [50, 90, 95, 99]
    res["r6"] = {"pct": q,
                 "train_abs": np.percentile(np.abs(A_tr[:, :6]), q, axis=0).round(4).tolist(),
                 "l1_abs": np.percentile(np.abs(A_l1[:, :6]), q, axis=0).round(4).tolist(),
                 "train_std": A_tr[:, :6].std(0).round(4).tolist(), "l1_std": A_l1[:, :6].std(0).round(4).tolist(),
                 "train_frac_pos_norm_lt_0.01": float((np.linalg.norm(A_tr[:, :3], axis=1) < 0.01).mean()),
                 "l1_frac_pos_norm_lt_0.01": float((np.linalg.norm(A_l1[:, :3], axis=1) < 0.01).mean()),
                 "train_frac_scaled_clip": float((np.abs(A_tr[:, :6] * 8) > 4).any(1).mean()),
                 "train_gripper_closed": float((A_tr[:, 6] > 0).mean()), "l1_gripper_closed": float((A_l1[:, 6] > 0).mean())}

    # ---- R4: estimators
    model = fit(se)
    res["fit_step_e"] = {k: v for k, v in model.items() if k.startswith(("r2", "n_"))}
    kfs = sorted({e["kf"] for e in l1})
    fold = {k: i % 2 for i, k in enumerate(rng.permutation(kfs))}
    m_l1 = [fit([(e["a"], e["st"]) for e in l1 if fold[e["kf"]] != f]) for f in (0, 1)]
    res["fit_l1_r2_pos"] = [m["r2_pos"] for m in m_l1]
    ks = (0.1, 0.25, 0.5)
    est = {"read": [], "int": [], "int_l1": [], **{f"fuse_{k}": [] for k in ks}}
    for e in l1:
        st0 = e["st"][0]
        est["read"].append(read_track(e))
        est["int"].append(integrate(model, st0, e["a"]))
        est["int_l1"].append(integrate(m_l1[fold[e["kf"]]], st0, e["a"]))
        for k in ks:
            est[f"fuse_{k}"].append(integrate(model, st0, e["a"], e["read"], k))
    steps = [8, 32, 64, 128, 192, 256]
    res["r4"] = {}
    for name, tracks in est.items():
        row = {}
        for s in steps:
            P, Rr, G = [], [], []
            for e, x in zip(l1, tracks):
                if s < len(x):
                    p, r, g = errors(x[s:s + 1], e["st"][s:s + 1])
                    P.append(p[0]); Rr.append(r[0]); G.append(g[0])
            row[s] = {"pos_cm_med": float(np.median(P)), "pos_cm_p90": float(np.percentile(P, 90)),
                      "rot_deg_med": float(np.median(Rr)), "grip_med": float(np.median(G)), "n": len(P)}
        res["r4"][name] = row
    np.savez_compressed(os.path.join(args.out, "r4_tracks.npz"), names=np.array([e["name"] for e in l1]),
                        **{k: np.array(v, dtype=object) for k, v in est.items()})
    json.dump(res, open(os.path.join(args.out, "r4_r6.json"), "w"), indent=1)

    r6 = res["r6"]
    print(f"R6  fit episodes {len(se)}  L1 episodes {len(l1)} ({res['n_l1_keyframes']} keyframes)")
    print("    |a| pose dims, percentiles", q)
    for nm in ("train_abs", "l1_abs"):
        print(f"    {nm:10s}", [[round(x, 4) for x in row] for row in r6[nm]])
    print(f"    std train {r6['train_std']}  l1 {r6['l1_std']}")
    print(f"    |pos a|<0.01: train {r6['train_frac_pos_norm_lt_0.01']:.2f}  l1 {r6['l1_frac_pos_norm_lt_0.01']:.2f};"
          f"  clip hit in train {r6['train_frac_scaled_clip']:.4f};  gripper closed train {r6['train_gripper_closed']:.2f}"
          f" l1 {r6['l1_gripper_closed']:.2f}")
    print(f"R4  integrator fit on step_e: R2 pos {np.round(model['r2_pos'], 3).tolist()} rot "
          f"{np.round(model['r2_rot'], 3).tolist()} grip {np.round(model['r2_grip'], 3).tolist()}")
    print("    estimator   " + "  ".join(f"step {s:3d} pos/p90/rot" for s in steps))
    for name, row in res["r4"].items():
        print(f"    {name:10s}  " + "  ".join(f"{row[s]['pos_cm_med']:5.1f}/{row[s]['pos_cm_p90']:5.1f}/"
                                              f"{row[s]['rot_deg_med']:4.1f}" for s in steps))


if __name__ == "__main__":
    main()
