"""Online RM accuracy during GRPO from per-episode dumps (reward_source rm, synchronised real simulator labels).

  python rm_online_dump.py DUMP_DIR [DUMP_DIR ...]
Per run: episode false-positive rate (RM fired, real failure), false-negative rate (real success, RM silent),
groups whose real outcomes are all failures and get >= 1 RM false positive, groups with a real learning signal
(mixed real outcomes) vs groups with an RM signal (mixed paid outcomes), and agreement of the GRPO advantage sign
(paid vs real) per episode inside mixed groups. Also by training step (4 rollout epochs per step) and by suite.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load(d):
    groups = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/episodes_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            groups[(os.path.basename(f), r["reset"], r["group"])].append(r)
    return groups


def stats(gs):
    eps = [r for g in gs for r in g]
    fail = [r for r in eps if not r["real_success"]]
    succ = [r for r in eps if r["real_success"]]
    fp = sum(r["paid_success"] for r in fail)
    fn = sum(not r["paid_success"] for r in succ)
    allfail = [g for g in gs if not any(r["real_success"] for r in g)]
    allsucc = [g for g in gs if all(r["real_success"] for r in g)]
    mixed_real = [g for g in gs if 0 < sum(r["real_success"] for r in g) < len(g)]
    mixed_paid = [g for g in gs if 0 < sum(r["paid_success"] for r in g) < len(g)]
    spurious = [g for g in mixed_paid if g not in mixed_real]  # RM signal where reality has none
    lost = [g for g in mixed_real if g not in mixed_paid]      # real signal the RM erased
    agree = tot = 0
    for g in mixed_real:
        mr = np.mean([r["real_success"] for r in g])
        mp = np.mean([r["paid_success"] for r in g])
        for r in g:
            ar, ap = r["real_success"] - mr, r["paid_success"] - mp
            if ar != 0:
                tot += 1
                agree += np.sign(ar) == np.sign(ap)
    G = max(len(gs), 1)
    return {"episodes": len(eps), "groups": len(gs), "real_success": len(succ) / max(len(eps), 1),
            "paid_success": sum(r["paid_success"] for r in eps) / max(len(eps), 1),
            "fp_rate": fp / max(len(fail), 1), "fn_rate": fn / max(len(succ), 1),
            "allfail_groups": len(allfail) / G,
            "allfail_with_fp": (sum(any(r["paid_success"] for r in g) for g in allfail) / len(allfail)) if allfail else None,
            "allsucc_with_fn": (sum(not all(r["paid_success"] for r in g) for g in allsucc) / len(allsucc)) if allsucc else None,
            "mixed_real": len(mixed_real) / G, "mixed_paid": len(mixed_paid) / G,
            "spurious_signal_groups": len(spurious) / G, "lost_signal_groups": len(lost) / G,
            "adv_sign_agree": agree / tot if tot else None}


def fmt(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main(dirs):
    cols = ["episodes", "real_success", "paid_success", "fp_rate", "fn_rate", "allfail_groups", "allfail_with_fp",
            "allsucc_with_fn", "mixed_real", "mixed_paid", "spurious_signal_groups", "lost_signal_groups",
            "adv_sign_agree"]
    for d in dirs:
        groups = load(d)
        print(f"\n# {d}")
        print("| slice | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        s = stats(list(groups.values()))
        print("| all | " + " | ".join(fmt(s[c]) for c in cols) + " |")
        by_step = defaultdict(list)
        for k, g in groups.items():
            by_step[(k[1] - 1) // 4].append(g)
        for st in sorted(by_step):
            s = stats(by_step[st])
            print(f"| step {st} | " + " | ".join(fmt(s[c]) for c in cols) + " |")
        by_suite = defaultdict(list)
        for g in groups.values():
            by_suite[g[0]["suite"]].append(g)
        for su in sorted(by_suite, key=str):
            s = stats(by_suite[su])
            print(f"| {su} | " + " | ".join(fmt(s[c]) for c in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
