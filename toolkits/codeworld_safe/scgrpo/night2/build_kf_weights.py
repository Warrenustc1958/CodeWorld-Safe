"""Build keyframe_exclude / keyframe_weights from the reset scan and per-episode dumps.

  python build_kf_weights.py SCAN_NPZ OUT_DIR DUMP_DIR [DUMP_DIR ...]
- exclude: keyframes with a grader violation or a real success already at the keyframe.
- weight: keyframes are binned by (kind, quadrant, relative t0 quintile) - the outcome of a group depends on the
  source trajectory and the position in it, much more than on the task cell. Weight of a bin = its smoothed
  fraction of mixed groups (some but not all of the G rollouts succeed), (mixed + A * m0) / (groups + A) with the
  global mixed fraction m0 and A = 10, floored at 0.15 of the maximum so every bin stays reachable.
Prints the expected mixed-group fraction and the suite mix under uniform vs weighted sampling.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

KF = "/data_cf/cws_sam/v1/scgrpo/keyframes_v1"
A, FLOOR = 10.0, 0.15


def bins(meta):
    t0 = meta["t0"].astype(int)
    tmax = defaultdict(int)
    for u, t in zip(meta["uid"], t0):
        tmax[str(u)] = max(tmax[str(u)], t)
    rel = np.array([min(int(5 * t / max(tmax[str(u)], 1)), 4) for u, t in zip(meta["uid"], t0)])
    return np.array([f"k{k}|{q}|r{r}" for k, q, r in zip(meta["kind"], meta["quadrant"], rel)])


def main(scan, out, dumps):
    meta = np.load(f"{KF}/meta.npz")
    z = np.load(scan)
    bad = z["viol"].any(1) | z["success"]
    b = bins(meta)
    groups = defaultdict(list)
    for d in dumps:
        for f in glob.glob(f"{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                groups[(d, os.path.basename(f), r["reset"], r["group"])].append(r)
    gs = [g for g in groups.values() if not bad[g[0]["kf"]]]
    mixed = lambda g: 0 < sum(r["real_success"] for r in g) < len(g)
    m0 = np.mean([mixed(g) for g in gs])
    cnt, mix = defaultdict(int), defaultdict(int)
    for g in gs:
        k = b[g[0]["kf"]]
        cnt[k] += 1
        mix[k] += mixed(g)
    names = sorted(set(b))
    est = {k: (mix[k] + A * m0) / (cnt[k] + A) for k in names}
    floor = FLOOR * max(est.values())
    wb = {k: max(est[k], floor) for k in names}
    w = np.array([wb[k] for k in b])
    w[bad] = 0.0
    np.save(f"{out}/kf_exclude.npy", np.flatnonzero(bad))
    np.save(f"{out}/kf_weights.npy", w)
    kept = ~bad
    print(f"excluded {int(bad.sum())}/{len(bad)}; groups used {len(gs)}; global mixed fraction m0 {m0:.3f}")
    print("bin | groups | mixed | smoothed | weight | kept keyframes")
    for k in names:
        print(f"{k} | {cnt[k]} | {mix[k]} | {est[k]:.3f} | {wb[k]:.3f} | {int(((b == k) & kept).sum())}")
    e = np.array([est[k] for k in b])
    u = e[kept].mean()
    ww = w[kept] / w[kept].sum()
    print(f"expected mixed-group fraction: uniform {u:.3f} -> weighted {float((ww * e[kept]).sum()):.3f}")
    suites = np.array([str(s) for s in meta["suite"]])
    for s in sorted(set(suites)):
        m = suites[kept] == s
        print(f"  suite {s}: uniform {m.mean():.3f} -> weighted {ww[m].sum():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
