#!/usr/bin/env python3
"""Per-frame proprio of every cws_v1 episode, aligned with its stitched visual frames (T1 pose labels).

Frames are stitched from ``agent_segments`` ([npz, start, stop], one npz per audit json). Frame i of a segment is
the state after i transitions of that segment's json: i = 0 is the state before its first transition
(initial_observation for a root; for a counterfactual branch the root's state at the branch point, because a
branch json's initial_observation is the root's start), i >= 1 is trajectory[i - 1].observation.
Output: <out>/<idx>.npy float32 [num_frames, 9] = eef pos (3), eef quat xyzw (4), gripper qpos (2);
<out>/stats.json with train-split normalisation statistics. Checked against keyframes_v1 (states at t0).

  python build_pose_index.py --out DIR [--workers 64] [--check-only N]
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = "/mnt/cpfs/jieduan/data/cws_rollout"


def obs_vec(o):
    return np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"]]).astype(np.float32)


def segment_states(npz_path, first=None):
    """States of every frame of one visual segment npz (via its audit json); ``first`` replaces frame 0."""
    vis = os.path.join(ROOT, npz_path)
    jpath = os.path.join(os.path.dirname(os.path.dirname(vis)), os.path.basename(vis).replace(".npz", ".json"))
    j = json.load(open(jpath))
    traj = [obs_vec(s["observation"]) for s in j["trajectory"]]
    return np.stack([obs_vec(j["initial_observation"]) if first is None else first] + traj)


def episode_states(row):
    """Root: one segment. Branch: the root's prefix segment, then the branch segment, whose frame 0 is the root's
    state after branch_timestep - 1 transitions (its json's initial_observation is the root's start)."""
    segs = row["agent_segments"]
    if row["kind"] == "root" or len(segs) == 1:
        parts = [segment_states(p)[a:b] for p, a, b in segs]
    else:
        root = segment_states(segs[0][0])
        bstart = int(row["branch_timestep"]) - 1
        parts = [root[segs[0][1]:segs[0][2]]]
        for p, a, b in segs[1:]:
            parts.append(segment_states(p, first=root[min(bstart, len(root) - 1)])[a:b])
    st = np.concatenate(parts)[: row["num_frames"]]
    if len(st) != row["num_frames"]:
        raise ValueError(f"{row['idx']}: {len(st)} states vs {row['num_frames']} frames")
    return st


def work(args):
    row, out = args
    try:
        st = episode_states(row)
        np.save(os.path.join(out, f"{row['idx']}.npy"), st)
        return row["idx"], None
    except Exception as exc:  # noqa: BLE001
        return row["idx"], repr(exc)


def check(rows_by_idx, out, n):
    """keyframes_v1 states (pi0.5 proprio at t0, axis-angle) vs the index at frame t0 of the same episode."""
    from scipy.spatial.transform import Rotation as Rot

    meta = np.load("/mnt/cpfs/jieduan/data/scgrpo_bundle/keyframes_v1/meta.npz")
    uid2idx = {}
    for l in open("/mnt/cpfs/jieduan/data/labels/episodes.jsonl"):
        r = json.loads(l)
        uid2idx[r["uid"]] = r["uid"].split("/")[-1]
    sel = np.random.default_rng(0).permutation(len(meta["uid"]))
    pos, rot, kinds, used = [], [], [], 0
    for i in sel:
        idx = uid2idx.get(str(meta["uid"][i]))
        p = os.path.join(out, f"{idx}.npy")
        if idx not in rows_by_idx or not os.path.exists(p):
            continue
        st = np.load(p)
        t0 = int(meta["t0"][i])
        if t0 >= len(st):
            continue
        s = meta["states"][i]
        pos.append(np.linalg.norm(st[t0, :3] - s[:3]) * 100)
        rot.append(np.degrees((Rot.from_quat(st[t0, 3:7]) * Rot.from_rotvec(s[3:6]).inv()).magnitude()))
        kinds.append(rows_by_idx[idx]["kind"])
        used += 1
        if used >= n:
            break
    pos, rot, kinds = np.array(pos), np.array(rot), np.array(kinds)
    for k in np.unique(kinds):
        m = kinds == k
        print(f"check {k:22s} n={m.sum():4d}  pos cm med {np.median(pos[m]):.3f} max {pos[m].max():.3f}  "
              f"rot deg med {np.median(rot[m]):.3f} max {rot[m].max():.3f}")
    return pos, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--check-only", type=int, default=0, help="build only the episodes of N checked keyframes")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{ROOT}/index/cws_v1/manifest.jsonl")]
    by_idx = {r["idx"]: r for r in rows}
    todo = rows
    if args.check_only:
        meta = np.load("/mnt/cpfs/jieduan/data/scgrpo_bundle/keyframes_v1/meta.npz")
        want = {str(u).split("/")[-1] for u in meta["uid"][np.random.default_rng(0).permutation(len(meta["uid"]))[
            : 4 * args.check_only]]}
        todo = [r for r in rows if r["idx"] in want]
    cwd = os.getcwd()
    os.chdir(ROOT)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(work, [(r, args.out) for r in todo], chunksize=8))
    os.chdir(cwd)
    bad = [(i, e) for i, e in res if e]
    print(f"built {len(res) - len(bad)}/{len(res)}; failed {len(bad)}", bad[:3])
    check(by_idx, args.out, 400 if not args.check_only else args.check_only)
    if not args.check_only:
        tr = [np.load(os.path.join(args.out, f"{r['idx']}.npy")) for r in rows
              if r["split"] == "train" and r.get("dual") and os.path.exists(os.path.join(args.out, f"{r['idx']}.npy"))]
        X = np.concatenate(tr)
        json.dump({"mean": X.mean(0).tolist(), "std": X.std(0).tolist(), "n_frames": int(len(X)),
                   "n_episodes": len(tr), "layout": "pos3 quat_xyzw4 gripper_qpos2"},
                  open(os.path.join(args.out, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
