#!/usr/bin/env python3
"""Summarise p_replay.py output (all shards) into the P3-P6 tables; CIs resample start keyframes.

  python p_summary.py --rep out.jsonl --out summary.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json

import numpy as np


def boot(values, clusters, stat=np.mean, n=1000, seed=0):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    if not len(values):
        return [float("nan")] * 3
    groups = [values[clusters == u] for u in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    bs = [stat(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(n)]
    return [float(stat(values)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def fmt(ci, p=2):
    return f"{ci[0]:.{p}f} [{ci[1]:.{p}f},{ci[2]:.{p}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, help="prefix given to p_replay --out (shards .0 .1 ...)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for f in sorted(glob.glob(args.rep + ".*")) for l in open(f)]
    res = {"n_records": len(recs)}

    # per (episode, chunk): the free (online) error, the baseline every repair is compared with
    free = {(r["ep"], r["chunk"]): r for r in recs if r.get("mode") == "free"}
    res["n_episodes"] = len({k[0] for k in free})

    # ---- P4: oracle / single-view repair at depth d, error after k chunks relative to free
    rep = collections.defaultdict(list)  # (mode, d0, k) -> [(kf, ratio_a, ratio_w, eef_ratio, mae_a, mae_w)]
    for r in recs:
        m = r.get("mode", "")
        if "@" not in m or m.split("@")[0] not in ("oracle", "wrist", "agent"):
            continue
        f = free.get((r["ep"], r["chunk"]))
        if f is None:
            continue
        k = r["chunk"] - r["d0"]  # 1 = the chunk right after the repaired history
        rep[(m.split("@")[0], r["d0"], k)].append(
            (r["kf"], r["mae_agent"] / max(f["mae_agent"], 1e-6), r["mae_wrist"] / max(f["mae_wrist"], 1e-6),
             r["eef_cm"] / max(f["eef_cm"], 1e-6), r["mae_agent"], r["mae_wrist"], f["mae_agent"], f["mae_wrist"]))
    res["p4"] = []
    for (m, d0, k), v in sorted(rep.items()):
        v = np.array(v, float)
        cl = v[:, 0]
        res["p4"].append({"mode": m, "d0": d0, "k": k, "n": len(v),
                          "ratio_agent": boot(v[:, 1], cl, np.median), "ratio_wrist": boot(v[:, 2], cl, np.median),
                          "ratio_eef": boot(v[:, 3], cl, np.median),
                          "mae_agent": float(np.median(v[:, 4])), "mae_wrist": float(np.median(v[:, 5])),
                          "free_agent": float(np.median(v[:, 6])), "free_wrist": float(np.median(v[:, 7]))})

    # ---- P3 / P5: one chunk after a history swap, distance to the oracle chunk (same noise)
    p3 = collections.defaultdict(list)
    for r in recs:
        if "p3" in r:
            p3[(r["p3"], r["depth"])].append(r)
    res["p3"] = []
    for (m, d), v in sorted(p3.items()):
        cl = [x["kf"] for x in v]
        row = {"cmp": m, "depth": d, "n": len(v),
               "agent": boot([x["vs_oracle_agent"] for x in v], cl, np.median),
               "wrist": boot([x["vs_oracle_wrist"] for x in v], cl, np.median)}
        if m == "wanhist":  # P5: output deviation / input (history) deviation under the same noise
            row["gain_agent"] = boot([x["vs_oracle_agent"] / max(x["hist_err_agent"], 1e-6) for x in v], cl, np.median)
            row["gain_wrist"] = boot([x["vs_oracle_wrist"] / max(x["hist_err_wrist"], 1e-6) for x in v], cl, np.median)
            row["hist_agent"] = float(np.median([x["hist_err_agent"] for x in v]))
            row["hist_wrist"] = float(np.median([x["hist_err_wrist"] for x in v]))
        res["p3"].append(row)

    # ---- P6: pixel MSE bias^2 / variance over fresh-noise rollouts, per chunk
    bv = collections.defaultdict(list)
    for r in recs:
        if r.get("biasvar"):
            bv[r["chunk"]].append(r)
    res["p6"] = []
    for c, v in sorted(bv.items()):
        cl = [x["kf"] for x in v]
        row = {"chunk": c, "n": len(v), "n_seeds": int(np.median([x["n_seeds"] for x in v]))}
        for view in ("agent", "wrist"):
            b2 = np.array([x[f"bias2_{view}"] for x in v])
            var = np.array([x[f"var_{view}"] for x in v])
            row[f"var_share_{view}"] = boot(var / np.maximum(b2 + var, 1e-9), cl, np.mean)
            row[f"bias2_{view}"], row[f"var_{view}"] = float(b2.mean()), float(var.mean())
        res["p6"].append(row)

    # ---- repro numerics
    rp = [r for r in recs if r.get("mode") == "repro" and "vs_dump_jpg_agent" in r]
    sd = [r for r in recs if r.get("mode", "").startswith("seed") and r["chunk"] == 1 and "vs_dump_agent" in r]
    res["repro"] = {"repro_jpg_agent": float(np.median([r["vs_dump_jpg_agent"] for r in rp])) if rp else None,
                    "repro_jpg_wrist": float(np.median([r["vs_dump_jpg_wrist"] for r in rp])) if rp else None,
                    "seed_raw_agent": float(np.median([r["vs_dump_agent"] for r in sd])) if sd else None,
                    "seed_raw_wrist": float(np.median([r["vs_dump_wrist"] for r in sd])) if sd else None}
    json.dump(res, open(args.out, "w"), indent=1)

    print(f"records {res['n_records']}  episodes {res['n_episodes']}  repro {res['repro']}")
    print("\nP4 repair at chunk d0 -> error ratio vs free after k chunks (median [95% CI]); <1 = better than free")
    for r in res["p4"]:
        if r["k"] in (1, 2, 4, 8, 16) or r["k"] == max(x["k"] for x in res["p4"] if x["mode"] == r["mode"]
                                                         and x["d0"] == r["d0"]):
            print(f"  {r['mode']:6s} d0={r['d0']:2d} k={r['k']:2d} n={r['n']:4d}  agent {fmt(r['ratio_agent'])}  "
                  f"wrist {fmt(r['ratio_wrist'])}  eef {fmt(r['ratio_eef'])}  "
                  f"(mae {r['mae_agent']:.1f}/{r['mae_wrist']:.1f} vs free {r['free_agent']:.1f}/{r['free_wrist']:.1f})")
    print("\nP3/P5 one chunk after the history is changed: MAE to the oracle chunk (same noise)")
    for r in res["p3"]:
        extra = (f"  gain a {fmt(r['gain_agent'])} w {fmt(r['gain_wrist'])}  (hist err {r['hist_agent']:.1f}/"
                 f"{r['hist_wrist']:.1f})" if "gain_agent" in r else "")
        print(f"  {r['cmp']:8s} d={r['depth']:2d} n={r['n']:4d}  agent {fmt(r['agent'])}  wrist {fmt(r['wrist'])}{extra}")
    print("\nP6 variance share of pixel MSE over fresh-noise rollouts (same actions)")
    for r in res["p6"]:
        if r["chunk"] in (1, 2, 4, 8, 16, 24, 32):
            print(f"  chunk {r['chunk']:2d} n={r['n']:4d} seeds={r['n_seeds']}  agent {fmt(r['var_share_agent'], 3)}  "
                  f"wrist {fmt(r['var_share_wrist'], 3)}  (bias2 {r['bias2_agent']:.0f}/{r['bias2_wrist']:.0f}, "
                  f"var {r['var_agent']:.0f}/{r['var_wrist']:.0f})")


if __name__ == "__main__":
    main()
