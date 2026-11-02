"""Counterfactual threshold sweep on GRPO dumps: recompute the RM decision from the stored per-chunk RM max.

  python rm_thr_sweep.py DUMP_DIR [DUMP_DIR ...]
E15 runs use rm_frames [3, 7], success_consecutive 1, rule "once", fixed 64-step horizon: paid = any chunk max >= thr.
Per threshold: episode FP / FN, spurious-signal groups (RM mixed, real not), lost-signal groups (real mixed, RM not),
advantage-sign agreement in real-mixed groups, and the cosine between GRPO advantages from the RM and from the real
outcome over all episodes (A = (r - group mean) / max(group std, 0.1), as in training; 1.0 = real reward).
The rollouts stay those of the run (the threshold would also change the policy), so this is an estimate.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]


def load(d):
    groups = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/episodes_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            vals = [x for x in r["rm_chunk_max"] if x is not None]
            groups[(os.path.basename(f), r["reset"], r["group"])].append(
                (bool(r["real_success"]), max(vals) if vals else 0.0, bool(r["paid_success"])))
    return list(groups.values())


def adv(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / max(x.std(), 0.1)


def evaluate(groups, thr):
    fp = fn = nf = ns = spur = lost = agree = tot = 0
    dot = nr = npd = 0.0
    for g in groups:
        real = [e[0] for e in g]
        paid = [e[2] if thr is None else e[1] >= thr for e in g]
        for r, p in zip(real, paid):
            if r:
                ns += 1
                fn += not p
            else:
                nf += 1
                fp += p
        mr, mp = 0 < sum(real) < len(g), 0 < sum(paid) < len(g)
        spur += mp and not mr
        lost += mr and not mp
        ar, ap = adv(real), adv(paid)
        dot += float(ar @ ap)
        nr += float(ar @ ar)
        npd += float(ap @ ap)
        if mr:
            for a, b in zip(ar, ap):
                tot += 1
                agree += np.sign(a) == np.sign(b)
    G = len(groups)
    return {"fp": fp / max(nf, 1), "fn": fn / max(ns, 1), "spurious": spur / G, "lost": lost / G,
            "sign_agree": agree / tot if tot else None,
            "adv_cos": dot / np.sqrt(nr * npd) if nr and npd else 0.0}


def main(dirs):
    for d in dirs:
        groups = load(d)
        real_mixed = np.mean([0 < sum(e[0] for e in g) < len(g) for g in groups])
        print(f"\n# {d}: {len(groups)} groups, real-signal groups {real_mixed:.3f}")
        print("| threshold | FP | FN | spurious groups | lost groups | sign agree | adv cosine |")
        print("|---|---|---|---|---|---|---|")
        rows = [("as run", evaluate(groups, None))] + [(f"{t:.2f}", evaluate(groups, t)) for t in THRS]
        best = max(rows[1:], key=lambda x: x[1]["adv_cos"])[0]
        for name, s in rows:
            mark = " **best**" if name == best else ""
            agree = "-" if s["sign_agree"] is None else f"{s['sign_agree']:.3f}"
            print(f"| {name}{mark} | {s['fp']:.3f} | {s['fn']:.3f} | {s['spurious']:.3f} | {s['lost']:.3f} | "
                  f"{agree} | {s['adv_cos']:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
