#!/usr/bin/env python3
"""P1: paired Wan-vs-real drift under the same executed actions (training-free).

Reads the episode npz files written by SyncRealWorldModelEnv with ``side_render: True`` and ``dump_arrays: True``
(one closed-loop rollout of the frozen policy on Wan observations; the synchronised simulator executes the same
actions and renders every step). Frame alignment: Wan frame k of chunk d and real frame 1 + 8d + k both follow
action 8d + k (real frame 0 is the restored keyframe).

Per chunk depth d (1..32) it reports, with episode-cluster bootstrap CIs (clusters = start keyframes):
  mae_{agent,wrist}       Wan vs real pixel MAE (0-255)
  copy0_{agent,wrist}     real start frame vs real frame t (static-world baseline)
  eef_cm / grip           state reader on the Wan frame vs the real 8-d state (eef position, gripper qpos)
  eef_cm_floor            state reader on the real frame vs the real state (reader error floor)
  rm flips                per chunk, RM decision (max over rm_frames >= thr) on Wan vs on real frames:
                          f01 = P(Wan fires, real does not), f10 = the reverse, agree
Episode level: RM-once on Wan / on real / real success, and AUROC of the per-chunk RM max against the real goal.
Also fits e_{d+1} = L e_d + c per view on episode chunk-mean MAE (how close the error recursion is to L = 1).

  python p1_paired_drift.py --dump DIR [DIR ...] --out p1.json [--reader state_reg.pt] [--workers 32]
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

CHUNK = 8


def _dec(buf, off, i):
    return np.asarray(Image.open(io.BytesIO(buf[off[i]:off[i + 1]].tobytes())).convert("RGB"))


def load_episode(path):
    z = np.load(path)
    if "real_jpg" not in z.files:
        return None
    n_wan = len(z["wan_off"]) - 1
    d_max = n_wan // CHUNK
    wan = np.stack([_dec(z["wan_jpg"], z["wan_off"], i) for i in range(n_wan)])
    n_real = len(z["real_off"]) - 1
    real = np.stack([_dec(z["real_jpg"], z["real_off"], i) for i in range(n_real)])
    if n_real < n_wan + 1:  # a slot stopped early: compare only the executed prefix
        n_wan = n_real - 1
        d_max = n_wan // CHUNK
        wan = wan[: d_max * CHUNK]
    w, r = wan.astype(np.int16), real[1: 1 + d_max * CHUNK].astype(np.int16)
    r0 = real[:1].astype(np.int16)
    ag, wr = slice(0, 256), slice(256, 512)
    out = dict(
        path=os.path.basename(path), kf=int(z["kf"]), d_max=d_max,
        mae_agent=np.abs(w[:, ag] - r[:, ag]).mean((1, 2, 3)), mae_wrist=np.abs(w[:, wr] - r[:, wr]).mean((1, 2, 3)),
        copy0_agent=np.abs(r0[:, ag] - r[:, ag]).mean((1, 2, 3)),
        copy0_wrist=np.abs(r0[:, wr] - r[:, wr]).mean((1, 2, 3)),
        real_state=z["real_state"][1: 1 + d_max * CHUNK], wan_read=z["wan_read"][: d_max * CHUNK],
        rm_wan=z["rm_wan"][:d_max], rm_real=z["rm_real"][:d_max],
        goal=z["step_goal"][: d_max * CHUNK].astype(bool),
        real_last=real[1: 1 + d_max * CHUNK][CHUNK - 1::CHUNK],  # real frame at each chunk end (for the reader floor)
    )
    return out


def reader_floor(eps, reader_path, device):
    """State reader applied to the real chunk-end frames, minus the real state: the reader's own error."""
    import torch

    from rlinf.envs.sim.world_model.safe.safe_models import StateReader

    rd = StateReader(reader_path).to(device)
    for e in eps:
        x = torch.from_numpy(e.pop("real_last")).to(device)
        e["real_read_end"] = rd(x).float().cpu().numpy()


def boot_ci(values, clusters, n=1000, seed=0):
    """Median and 95% CI of the per-episode values, resampling start keyframes (clusters)."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if not len(values):
        return [np.nan, np.nan, np.nan]
    uniq = np.unique(clusters)
    groups = [values[clusters == u] for u in uniq]
    rng = np.random.default_rng(seed)
    stats = [np.median(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
             for _ in range(n)]
    return [float(np.median(values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def boot_mean(values, clusters, n=1000, seed=0):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [values[clusters == u] for u in uniq]
    rng = np.random.default_rng(seed)
    stats = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return [float(values.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks over ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_L(series):
    """Least squares e_{d+1} = L e_d + c over all episodes' chunk-mean errors."""
    x = np.concatenate([s[:-1] for s in series])
    y = np.concatenate([s[1:] for s in series])
    A = np.stack([x, np.ones_like(x)], 1)
    (L, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(L), float(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True, help="dump dirs (each has arrays/*.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reader", default="/mnt/cpfs/jieduan/data/scgrpo_bundle/scoring/state_reg_v1/state_reg.pt")
    ap.add_argument("--thr", type=float, default=0.877)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files = sorted(f for d in args.dump for f in glob.glob(os.path.join(d, "arrays", "*.npz")))
    with ProcessPoolExecutor(args.workers) as ex:
        eps = [e for e in ex.map(load_episode, files, chunksize=4) if e is not None]
    if not eps:
        raise SystemExit("no side_render episodes found")
    reader_floor(eps, args.reader, args.device)
    D = max(e["d_max"] for e in eps)
    res = {"n_episodes": len(eps), "n_keyframes": len({e["kf"] for e in eps}), "files": len(files), "by_depth": []}

    for d in range(D):
        sel = [e for e in eps if e["d_max"] > d]
        cl = [e["kf"] for e in sel]
        s = slice(d * CHUNK, (d + 1) * CHUNK)
        row = {"depth": d + 1, "step": (d + 1) * CHUNK, "n": len(sel)}
        for key in ("mae_agent", "mae_wrist", "copy0_agent", "copy0_wrist"):
            row[key] = boot_ci([e[key][s].mean() for e in sel], cl)
        eef = [np.linalg.norm(e["wan_read"][s, :3] - e["real_state"][s, :3], axis=1).mean() * 100 for e in sel]
        grip = [np.abs(e["wan_read"][s, 6:8] - e["real_state"][s, 6:8]).mean() for e in sel]
        floor = [np.linalg.norm(e["real_read_end"][d, :3] - e["real_state"][s][-1, :3]) * 100 for e in sel]
        row["eef_cm"], row["grip"], row["eef_cm_floor"] = boot_ci(eef, cl), boot_ci(grip, cl), boot_ci(floor, cl)
        dw = np.array([e["rm_wan"][d].max() >= args.thr for e in sel])
        dr = np.array([e["rm_real"][d].max() >= args.thr for e in sel])
        row["f01"], row["f10"] = boot_mean(dw & ~dr, cl), boot_mean(~dw & dr, cl)
        row["agree"] = boot_mean(dw == dr, cl)
        res["by_depth"].append(row)

    # episode level and per-chunk AUROC against the real goal, in depth bands
    bands = [(0, 8), (8, 16), (16, 32)]
    res["auroc"] = {}
    for lo, hi in bands:
        sw, sr, lab = [], [], []
        for e in eps:
            for d in range(lo, min(hi, e["d_max"])):
                sw.append(e["rm_wan"][d].max())
                sr.append(e["rm_real"][d].max())
                lab.append(e["goal"][d * CHUNK:(d + 1) * CHUNK].any())
        res["auroc"][f"chunks_{lo + 1}_{hi}"] = {"wan": auroc(sw, lab), "real": auroc(sr, lab), "n": len(lab),
                                                 "pos_rate": float(np.mean(lab)) if lab else float("nan")}
    once_w = np.array([(e["rm_wan"].max(1) >= args.thr).any() for e in eps])
    once_r = np.array([(e["rm_real"].max(1) >= args.thr).any() for e in eps])
    succ = np.array([e["goal"].any() for e in eps])
    res["episode"] = {"real_success": float(succ.mean()), "rm_once_wan": float(once_w.mean()),
                      "rm_once_real": float(once_r.mean()),
                      "fp_wan": float((once_w & ~succ).sum() / max(1, (~succ).sum())),
                      "fp_real": float((once_r & ~succ).sum() / max(1, (~succ).sum())),
                      "fn_wan": float((~once_w & succ).sum() / max(1, succ.sum())),
                      "fn_real": float((~once_r & succ).sum() / max(1, succ.sum()))}
    res["fit_L"] = {}
    for key in ("mae_agent", "mae_wrist"):
        series = [e[key][: e["d_max"] * CHUNK].reshape(-1, CHUNK).mean(1) for e in eps if e["d_max"] >= 2]
        res["fit_L"][key] = dict(zip(("L", "c"), fit_L(series)))
    json.dump(res, open(args.out, "w"), indent=1)

    print(f"episodes {res['n_episodes']}  keyframes {res['n_keyframes']}")
    print("depth step   n  agent  wrist  copy0a copy0w  eef_cm floor  grip   f01   f10  agree")
    for r in res["by_depth"]:
        print(f"{r['depth']:5d} {r['step']:4d} {r['n']:3d} {r['mae_agent'][0]:6.1f} {r['mae_wrist'][0]:6.1f} "
              f"{r['copy0_agent'][0]:6.1f} {r['copy0_wrist'][0]:6.1f} {r['eef_cm'][0]:7.2f} {r['eef_cm_floor'][0]:5.2f} "
              f"{r['grip'][0]:5.3f} {r['f01'][0]:5.3f} {r['f10'][0]:5.3f} {r['agree'][0]:5.3f}")
    print("auroc", json.dumps(res["auroc"]))
    print("episode", json.dumps(res["episode"]))
    print("fit_L", json.dumps(res["fit_L"]))


if __name__ == "__main__":
    main()
