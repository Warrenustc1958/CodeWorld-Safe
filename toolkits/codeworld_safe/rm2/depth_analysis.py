"""RM accuracy vs episode depth on a long-horizon dump (M1: 32 chunks = 256 steps from the initial-state keyframes).

  python depth_analysis.py DUMP_DIR [THR]
Truncate every episode at H chunks (H = 4, 8, 16, 24, 32) and recompute, on the same rollouts:
real success within H (first real goal step < 8H), RM success within H (any of the first H chunk maxima >= THR),
m, FP / FN, spurious-signal groups and the advantage cosine RM vs real.
Also: for successful episodes, the chunk where the RM first fires minus the chunk of the real goal (timing);
for RM false positives, the chunk where they fire.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np


def adv(x):
    return (x - x.mean()) / max(x.std(), 0.1)


def main(d, thr):
    groups = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/episodes_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            groups[(os.path.basename(f), r["reset"], r["group"])].append(r)
    groups = [g for g in groups.values() if len(g) == 8]
    print(f"{len(groups)} groups, threshold {thr}")
    print("| H chunks (steps) | success | m | FP | FN | spurious | spurious/real | adv cosine |")
    print("|---|---|---|---|---|---|---|---|")
    for H in (4, 8, 16, 24, 32):
        A, R, spur, mixed = [], [], 0, 0
        fp = fn = nf = ns = 0
        for g in groups:
            real = np.array([float(0 <= r["real_success_step"] < 8 * H) for r in g])
            rm = np.array([float(any(x is not None and x >= thr for x in r["rm_chunk_max"][:H])) for r in g])
            mr, mp = 0 < real.sum() < 8, 0 < rm.sum() < 8
            mixed += mr
            spur += mp and not mr
            A.append(adv(rm))
            R.append(adv(real))
            fp += int(((rm == 1) & (real == 0)).sum())
            nf += int((real == 0).sum())
            fn += int(((rm == 0) & (real == 1)).sum())
            ns += int((real == 1).sum())
        A, R = np.concatenate(A), np.concatenate(R)
        cos = float(A @ R / (np.linalg.norm(A) * np.linalg.norm(R))) if np.linalg.norm(A) and np.linalg.norm(R) else 0.0
        G = len(groups)
        print(f"| {H} ({8 * H}) | {ns / (ns + nf):.3f} | {mixed / G:.3f} | {fp / max(nf, 1):.3f} | {fn / max(ns, 1):.3f} | "
              f"{spur / G:.3f} | {spur / max(mixed, 1):.3f} | {cos:.3f} |")
    lag, fp_chunk = Counter(), Counter()
    for g in groups:
        for r in g:
            fire = next((c for c, x in enumerate(r["rm_chunk_max"]) if x is not None and x >= thr), None)
            if r["real_success"] and r["real_success_step"] >= 0:
                gc = r["real_success_step"] // 8
                lag["never" if fire is None else max(-4, min(4, fire - gc))] += 1
            elif fire is not None:
                fp_chunk[min(fire // 4 * 4, 28)] += 1
    print("\nsuccess episodes, RM first-fire chunk minus real goal chunk (clipped to +-4; 'never' = missed):")
    print(dict(sorted(lag.items(), key=lambda x: (isinstance(x[0], str), x[0]))))
    print("false-positive episodes by the chunk where the RM first fires (bins of 4 chunks):", dict(sorted(fp_chunk.items())))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.877)
