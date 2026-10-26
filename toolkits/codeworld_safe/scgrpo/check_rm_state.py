"""Replay real recorded actions from keyframes through SafeWorldModelEnv (8 chunks) and compare
  RM success calls (threshold x consecutive frames) with the real success frame, and
  the proprio state read from generated frames (with the per-scene offset) with the real state.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
from smoke_env import CFG  # noqa: E402

from rlinf.envs.sim.world_model.safe.safe_env import SafeWorldModelEnv  # noqa: E402

KF = "/data_cf/cws_sam/v1/scgrpo/keyframes_v1"
EPS = "/data_cf/cws_sam/v1/labels/episodes.jsonl"


def states_of(e):
    sys.path.insert(0, "/mnt/nfs/fjj/Code/RLinf-Safety")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bk", "/mnt/nfs/fjj/Code/RLinf-Safety/toolkits/libero_safety/sam_alignment/scgrpo/build_keyframes.py")
    return None


def quat2axisangle(q):
    q = np.asarray(q, np.float64)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    return np.zeros(3) if den < 1e-8 else q[:3] * 2.0 * np.arccos(w) / den


def st(o):
    return np.concatenate([o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]), o["robot0_gripper_qpos"]])


def episode_states(e):
    rj = json.load(open(e["root_json"]))
    root = [st(rj["initial_observation"])] + [st(s["observation"]) for s in rj["trajectory"]]
    if e["kind"] == "root":
        s = root
    else:
        bj = json.load(open(e["json"]))
        s = root[: int(e["branch_timestep"])] + [st(x["observation"]) for x in bj["trajectory"]]
    return np.stack(s)[: e["num_frames"]]


def main():
    torch.cuda.set_device(0)
    from rlinf.scheduler import Worker
    Worker.torch_device_type = "cuda"
    cfg = CFG.copy()
    cfg.group_size = 1
    import os
    cfg.state_offset = os.environ.get("STATE_OFFSET", "1") == "1"
    meta = np.load(f"{KF}/meta.npz")
    eps = {}
    for l in open(EPS):
        e = json.loads(l)
        eps[e["uid"]] = e
    rng = np.random.default_rng(1)
    kind = meta["kind"]
    pick = np.concatenate([rng.choice(np.flatnonzero(np.isin(kind, (1, 3))), 48, replace=False),
                           rng.choice(np.flatnonzero(np.isin(kind, (0, 2))), 48, replace=False)])
    env = SafeWorldModelEnv(cfg, num_envs=16, seed_offset=0, total_num_processes=1)
    rec = []
    for b in range(0, len(pick), 16):
        kfs = pick[b:b + 16]
        env.reset(episode_indices=kfs)
        acts, gt_fs, true_st, t0s = [], [], [], []
        for k in kfs:
            e = eps[str(meta["uid"][k])]
            t0 = int(meta["t0"][k])
            with np.load(e["labels"]) as z:
                a, succ = z["action_env"].astype(np.float32), z["success"]
            s_idx = np.flatnonzero(succ)
            gt_fs.append(int(s_idx[0]) + 1 if len(s_idx) else None)
            idx = np.clip(np.arange(t0, t0 + 64), 0, len(a) - 1)
            acts.append(a[idx])
            sts = episode_states(e)
            true_st.append(sts[np.clip(np.arange(t0 + 8, t0 + 65, 8), 0, len(sts) - 1)])
            t0s.append(t0)
        acts = np.stack(acts)
        p_all, st_read = [], []
        for c in range(8):
            env.chunk_step(torch.from_numpy(acts[:, 8 * c:8 * c + 8]))
            p_all.append(env.last_p_succ.cpu().numpy())
            st_read.append(env.states.cpu().numpy())
        p_all = np.concatenate(p_all, 1)                 # [16, 64] per imagined frame
        st_read = np.stack(st_read, 1)                    # [16, 8, 8]
        for j in range(len(kfs)):
            rec.append({"t0": t0s[j], "gt_fs": gt_fs[j], "p": p_all[j], "st_err": np.abs(st_read[j] - true_st[j][:8])})
        print(f"batch {b // 16 + 1}/{len(pick) // 16}", flush=True)

    out = {}
    for thr in (0.2, 0.3, 0.5, 0.7):
        for cons in (1, 2, 3):
            tp = fp = fn = tn = 0
            lag = []
            for r in rec:
                hit = r["p"] >= thr
                for _ in range(1, cons):
                    hit = hit[1:] & hit[:-1]
                pred = int(np.argmax(hit)) + 1 if hit.any() else None   # imagined frame offset (1..64)
                gt = (r["gt_fs"] - r["t0"]) if r["gt_fs"] is not None and 0 < r["gt_fs"] - r["t0"] <= 64 else None
                if gt is not None and pred is not None:
                    tp += 1
                    lag.append(pred - gt)
                elif gt is None and pred is not None:
                    fp += 1
                elif gt is not None:
                    fn += 1
                else:
                    tn += 1
            out[f"thr{thr}_c{cons}"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                                        "median_lag_frames": float(np.median(lag)) if lag else None}
            print(f"thr {thr} cons {cons}: P {out[f'thr{thr}_c{cons}']['precision']:.2f} "
                  f"R {out[f'thr{thr}_c{cons}']['recall']:.2f} tp {tp} fp {fp} fn {fn} tn {tn} "
                  f"lag {out[f'thr{thr}_c{cons}']['median_lag_frames']}", flush=True)
    err = np.stack([r["st_err"] for r in rec])            # [n, 8 chunks, 8]
    out["state_err_by_chunk"] = {f"d{c + 1}": {"pos_m": float(err[:, c, :3].mean()), "rot_rad": float(err[:, c, 3:6].mean()),
                                               "grip_m": float(err[:, c, 6:8].mean())} for c in range(8)}
    print(json.dumps(out["state_err_by_chunk"], indent=None))
    json.dump(out, open(f"/data_cf/cws_sam/v1/scgrpo/check_rm_state_off{int(cfg.state_offset)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
