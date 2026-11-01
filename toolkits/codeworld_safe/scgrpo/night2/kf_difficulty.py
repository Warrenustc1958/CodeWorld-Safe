"""Which keyframe features predict all-success / all-fail / mixed groups? (dumps from sync_real_env dump_dir)

  python kf_difficulty.py SCAN_NPZ DUMP_DIR [DUMP_DIR ...]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

KF = "/data_cf/cws_sam/v1/scgrpo/keyframes_v1"


def main(scan, dumps):
    meta = np.load(f"{KF}/meta.npz")
    bad = np.load(scan)["viol"].any(1) | np.load(scan)["success"]
    t0 = meta["t0"].astype(int)
    # relative position of the keyframe in its source trajectory (t0 / max t0 seen for that uid)
    tmax = defaultdict(int)
    for u, t in zip(meta["uid"], t0):
        tmax[str(u)] = max(tmax[str(u)], t)
    rel = np.array([t / max(tmax[str(u)], 1) for u, t in zip(meta["uid"], t0)])
    groups = defaultdict(list)
    for d in dumps:
        for f in glob.glob(f"{d}/*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                groups[(d, os.path.basename(f), r["reset"], r["group"])].append(r)
    gs = [g for g in groups.values() if not bad[g[0]["kf"]]]
    print(f"{len(gs)} groups on kept keyframes")

    def table(name, key):
        agg = defaultdict(lambda: [0, 0, 0, 0])
        for g in gs:
            k = key(g[0]["kf"])
            m = sum(r["real_success"] for r in g)
            a = agg[k]
            a[0] += 1
            a[1] += m == 0
            a[2] += m == len(g)
            a[3] += 0 < m < len(g)
        print(f"\n## by {name}: groups | all-fail | all-success | mixed")
        for k in sorted(agg):
            a = agg[k]
            print(f"{k} | {a[0]} | {a[1] / a[0]:.2f} | {a[2] / a[0]:.2f} | {a[3] / a[0]:.2f}")

    table("quadrant", lambda i: str(meta["quadrant"][i]))
    table("kind", lambda i: int(meta["kind"][i]))
    table("relative t0", lambda i: f"{min(int(rel[i] * 5), 4) / 5:.1f}-{min(int(rel[i] * 5), 4) / 5 + .2:.1f}")
    table("suite", lambda i: str(meta["suite"][i]))
    # repeat draws of the same keyframe: how consistent is its outcome across groups?
    by_kf = defaultdict(list)
    for g in gs:
        by_kf[g[0]["kf"]].append(np.mean([r["real_success"] for r in g]))
    rep = [v for v in by_kf.values() if len(v) > 1]
    if rep:
        sd = np.mean([np.std(v) for v in rep])
        print(f"\nkeyframes drawn more than once: {len(rep)}; mean std of group success rate across draws {sd:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
