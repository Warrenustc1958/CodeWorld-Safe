"""Second gate of the pre-registered rule on keyframes selected by draw 1 (evaluated on a later draw, held out).

  python kf_retest_gate.py DUMP_DIR [DUMP_DIR ...]
Per selection rule: on the evaluation draw, m, success, R-A FP / FN, spurious-signal groups / real-signal groups,
and the advantage cosine R-A vs real at threshold .877 and the best of a sweep (A = (r - mean) / max(std, 0.1)).
Gate (pre-registered): m >= .40, cosine >= .60, spurious / real <= .5.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.877, 0.9, 0.95]


def adv(x):
    return (x - x.mean()) / max(x.std(), 0.1)


def load(dirs):
    draws = defaultdict(list)  # kf -> [array (8, 3): real, prob, rm]
    for d in dirs:
        g = defaultdict(list)
        for f in sorted(glob.glob(f"{d}/episodes_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                vals = [x for x in r["rm_chunk_max"] if x is not None]
                g[(d, os.path.basename(f), r["reset"], r["group"])].append(
                    (float(r["real_success"]), max(vals) if vals else 0.0, float(r["rm_success_once"]), r["kf"]))
        for k in sorted(g, key=lambda x: (x[0], x[2], x[1], x[3])):
            v = g[k]
            if len(v) == 8:
                draws[v[0][3]].append(np.array([x[:3] for x in v]))
    return {k: v for k, v in draws.items() if len(v) >= 2}


def gate(groups):
    def cos(thr):
        A = np.concatenate([adv(g[:, 2] if thr is None else (g[:, 1] >= thr).astype(float)) for g in groups])
        R = np.concatenate([adv(g[:, 0]) for g in groups])
        na, nr = np.linalg.norm(A), np.linalg.norm(R)
        return float(A @ R / (na * nr)) if na and nr else 0.0
    real = np.concatenate([g[:, 0] for g in groups])
    rm = np.concatenate([g[:, 2] for g in groups])
    mixed = [0 < g[:, 0].sum() < 8 for g in groups]
    spur = [0 < g[:, 2].sum() < 8 and not mx for g, mx in zip(groups, mixed)]
    sweep = {t: cos(t) for t in THRS}
    best = max(sweep, key=sweep.get)
    m = float(np.mean(mixed))
    return {"n": len(groups), "m": m, "success": float(real.mean()),
            "fp": float(rm[real == 0].mean()) if (real == 0).any() else 0.0,
            "fn": float(1 - rm[real == 1].mean()) if (real == 1).any() else 0.0,
            "spur_over_real": float(np.mean(spur)) / m if m else None,
            "cos_877": cos(None), "cos_best": sweep[best], "best_thr": best}


def main(dirs):
    rep = load(dirs)
    first = {k: v[0] for k, v in rep.items()}
    later = {k: v[1:] for k, v in rep.items()}
    q1 = {k: g[:, 2].mean() for k, g in first.items()}
    p1 = {k: g[:, 0].mean() for k, g in first.items()}
    rules = [("none (uniform)", lambda k: True),
             ("real mixed1 (oracle)", lambda k: 0 < p1[k] < 1),
             ("rm q1 in [.05, .95]", lambda k: 0.05 <= q1[k] <= 0.95),
             ("rm q1 in [.2, .8]", lambda k: 0.2 <= q1[k] <= 0.8),
             ("rm q1 > 0", lambda k: q1[k] > 0)]
    print(f"{len(rep)} keyframes drawn >= 2 times; evaluation uses every later draw")
    print("| selection on draw 1 | eval groups | m | success | FP | FN | spurious/real | cos .877 | cos best (thr) | gate |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, f in rules:
        gs = [g for k in rep if f(k) for g in later[k]]
        if not gs:
            continue
        s = gate(gs)
        cos = max(s["cos_877"], s["cos_best"])
        ok = s["m"] >= 0.40 and cos >= 0.60 and (s["spur_over_real"] if s["spur_over_real"] is not None else 9) <= 0.5
        sr = "-" if s["spur_over_real"] is None else f"{s['spur_over_real']:.3f}"
        print(f"| {name} | {s['n']} | {s['m']:.3f} | {s['success']:.3f} | {s['fp']:.3f} | {s['fn']:.3f} | {sr} | "
              f"{s['cos_877']:.3f} | {s['cos_best']:.3f} ({s['best_thr']}) | {'PASS' if ok else 'no'} |")


if __name__ == "__main__":
    main(sys.argv[1:])
