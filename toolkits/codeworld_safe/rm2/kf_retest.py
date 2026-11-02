"""Test-retest of per-keyframe group outcomes under the same policy, and the m reachable by selecting on draw 1.

  python kf_retest.py DUMP_DIR [DUMP_DIR ...]
Groups are keyed by their keyframe. For keyframes drawn >= 2 times: draw 1 selects, a later draw evaluates (held out).
Selection rules (all use only draw 1):
  real p1 in [lo, hi]   : real group success rate (oracle, needs labels)
  rm   q1 in [lo, hi]   : mean R-A decision of the group (no labels; R-A threshold .877 on chunk mid/end frames)
  real mixed1           : draw-1 group had mixed real outcomes
Reports, on the evaluation draw: m (mixed share), success, kept share of keyframes; plus test-retest correlation.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main(dirs):
    draws = defaultdict(list)  # kf -> [(p_real, q_rm, mixed)] in time order
    for d in dirs:
        g = defaultdict(list)
        for f in sorted(glob.glob(f"{d}/episodes_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                g[(d, os.path.basename(f), r["reset"], r["group"])].append(r)
        for k in sorted(g, key=lambda x: (x[0], x[2], x[1], x[3])):
            v = g[k]
            if len(v) != 8:
                continue
            s = sum(x["real_success"] for x in v)
            draws[v[0]["kf"]].append((s / 8, sum(x["rm_success_once"] for x in v) / 8, 0 < s < 8))
    rep = {k: v for k, v in draws.items() if len(v) >= 2}
    print(f"{sum(len(v) for v in draws.values())} groups, {len(draws)} keyframes, {len(rep)} drawn >= 2 times")
    a = np.array([v[0] for v in rep.values()])
    b = np.array([v[1] for v in rep.values()])
    print(f"test-retest: corr(p1, p2) {np.corrcoef(a[:, 0], b[:, 0])[0, 1]:.3f}; "
          f"P(mixed2 | mixed1) {b[a[:, 2] == 1, 2].mean():.3f} (n {int(a[:, 2].sum())}) vs "
          f"P(mixed2 | not mixed1) {b[a[:, 2] == 0, 2].mean():.3f}; corr(rm q1, real p2) {np.corrcoef(a[:, 1], b[:, 0])[0, 1]:.3f}")
    print(f"per-keyframe p1: all-fail {np.mean(a[:, 0] == 0):.3f}, all-success {np.mean(a[:, 0] == 1):.3f}, in between {np.mean((a[:, 0] > 0) & (a[:, 0] < 1)):.3f}")
    print("\n| selection on draw 1 | kept keyframes | m on draw 2 | success on draw 2 |")
    print("|---|---|---|---|")
    rules = [("none (uniform)", np.ones(len(a), bool)), ("real mixed1", a[:, 2] == 1)]
    for lo, hi in ((0.05, 0.95), (0.2, 0.8), (0.3, 0.7)):
        rules.append((f"real p1 in [{lo}, {hi}]", (a[:, 0] >= lo) & (a[:, 0] <= hi)))
    for lo, hi in ((0.05, 0.95), (0.1, 0.9), (0.2, 0.8)):
        rules.append((f"rm q1 in [{lo}, {hi}]", (a[:, 1] >= lo) & (a[:, 1] <= hi)))
    rules.append(("rm q1 > 0 (RM fired at least once)", a[:, 1] > 0))
    for name, m in rules:
        if m.sum() == 0:
            print(f"| {name} | 0 | - | - |")
            continue
        print(f"| {name} | {m.mean():.3f} ({int(m.sum())}) | {b[m, 2].mean():.3f} | {b[m, 0].mean():.3f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
