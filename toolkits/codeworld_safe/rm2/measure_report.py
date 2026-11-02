"""Report for the pre-registered within-group variance measurement (M0-M3).

  python measure_report.py DUMP_DIR [DUMP_DIR ...]
Reward is the real outcome; the R-A decision (threshold .877 on the chunk mid/end frames) is logged as rm_success_once
and its per-chunk max probability as rm_chunk_max. Per condition:
  success, m (groups with mixed real outcomes), within-group variance share;
  R-A online FP / FN over all episodes and inside all-fail / all-success groups;
  spurious-signal groups (R-A mixed, real not): measured and predicted
    pred = P(all fail)(1-(1-FP)^G) + P(all succ)(1-(1-FN)^G), with FP / FN the episode rates over all episodes
  spurious / real-signal ratio; advantage cosine R-A vs real at .877 and at the best threshold of a sweep.
Decision rule (pre-registered): m >= .40, success drop vs M0 <= .10, cosine >= .60, spurious/real <= .5.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.877, 0.9, 0.95]


def load(d):
    g = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/episodes_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            vals = [x for x in r["rm_chunk_max"] if x is not None]
            g[(os.path.basename(f), r["reset"], r["group"])].append(
                (float(r["real_success"]), max(vals) if vals else 0.0, float(r["rm_success_once"])))
    return [np.array(v) for v in g.values() if len(v) == 8]


def adv(x):
    return (x - x.mean()) / max(x.std(), 0.1)


def cosine(groups, col=None, thr=None):
    A, R = [], []
    for g in groups:
        rm = g[:, 2] if thr is None else (g[:, 1] >= thr).astype(float)
        A.append(adv(rm))
        R.append(adv(g[:, 0]))
    A, R = np.concatenate(A), np.concatenate(R)
    na, nr = np.linalg.norm(A), np.linalg.norm(R)
    return float(A @ R / (na * nr)) if na and nr else 0.0


def stats(groups):
    G = 8
    real = np.concatenate([g[:, 0] for g in groups])
    rm = np.concatenate([g[:, 2] for g in groups])
    p = np.array([g[:, 0].mean() for g in groups])
    tot = real.mean() * (1 - real.mean())
    allf = [g for g in groups if g[:, 0].sum() == 0]
    alls = [g for g in groups if g[:, 0].sum() == G]
    mixed = [g for g in groups if 0 < g[:, 0].sum() < G]
    rm_mixed = [g for g in groups if 0 < g[:, 2].sum() < G]
    fp = rm[real == 0].mean() if (real == 0).any() else 0.0
    fn = 1 - rm[real == 1].mean() if (real == 1).any() else 0.0
    spur = sum(1 for g in rm_mixed if not 0 < g[:, 0].sum() < G) / len(groups)
    pred = len(allf) / len(groups) * (1 - (1 - fp) ** G) + len(alls) / len(groups) * (1 - (1 - fn) ** G)
    m = len(mixed) / len(groups)
    sweep = {t: cosine(groups, thr=t) for t in THRS}
    best = max(sweep, key=sweep.get)
    return {"groups": len(groups), "success": float(real.mean()), "m": m,
            "within_share": float(np.mean(p * (1 - p)) / tot) if tot else 0.0,
            "fp": float(fp), "fn": float(fn),
            "fp_in_allfail": float(np.concatenate([g[:, 2] for g in allf]).mean()) if allf else None,
            "fn_in_allsucc": float(1 - np.concatenate([g[:, 2] for g in alls]).mean()) if alls else None,
            "spurious": spur, "spurious_pred": pred, "spurious_over_real": spur / m if m else None,
            "cos_877": cosine(groups), "cos_best": sweep[best], "best_thr": best}


def main(dirs):
    rows = {}
    for d in dirs:
        gs = load(d)
        if gs:
            rows[os.path.basename(os.path.dirname(d.rstrip("/")))] = stats(gs)
    keys = ["groups", "success", "m", "within_share", "fp", "fn", "fp_in_allfail", "fn_in_allsucc", "spurious",
            "spurious_pred", "spurious_over_real", "cos_877", "cos_best", "best_thr"]
    print("| condition | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    for k, s in rows.items():
        print(f"| {k} | " + " | ".join(f(s[c]) for c in keys) + " |")
    base = rows.get("m0", {}).get("success")
    print("\nPre-registered decision (m >= .40, success drop <= .10 vs M0, cosine >= .60, spurious/real <= .5):")
    for k, s in rows.items():
        drop = (base - s["success"]) if base is not None else 0.0
        cos = max(s["cos_877"], s["cos_best"])
        ok = s["m"] >= 0.40 and drop <= 0.10 and cos >= 0.60 and (s["spurious_over_real"] or 9) <= 0.5
        pe = abs(s["spurious"] - s["spurious_pred"]) / max(s["spurious"], 1e-6)
        print(f"- {k}: m {s['m']:.3f}, success drop {drop:+.3f}, cosine {cos:.3f}, spurious/real "
              f"{f(s['spurious_over_real'])} -> {'GO' if ok else 'no'}; prediction error {pe:.0%}"
              f"{' (> 50%: model needs revision)' if pe > 0.5 else ''}")
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.dirname(dirs[0].rstrip('/'))), "msnight",
                                      "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
