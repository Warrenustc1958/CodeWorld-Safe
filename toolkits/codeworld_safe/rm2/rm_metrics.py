"""Shared offline RM metrics on a build_rm_dataset split (val / test): every executed frame of each episode.

score(rows, probs, thr) -> dict with frame AUROC / AP and episode-level numbers. An episode "fires" when any of its
frames has p >= thr (the env uses k consecutive frames over 8 per chunk; with the mid and end frame per chunk this is
an approximation). calibrate(rows, probs, max_fp) picks the lowest threshold whose failed-episode fire rate <= max_fp.
"""
from collections import defaultdict

import numpy as np


def auroc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):  # average ranks over ties
        j = i
        while j < len(sp) and sp[j] == sp[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    if not y.any():
        return None
    o = np.argsort(-p, kind="mergesort")
    tp = np.cumsum(y[o])
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y[o]).sum() / y.sum())


def _episodes(rows, probs):
    ep = defaultdict(lambda: {"p": 0.0, "succ": False, "grp": None})
    for r, p in zip(rows, probs):
        k = (r["col"], r["file"], r["reset"], r["slot"])
        e = ep[k]
        e["p"] = max(e["p"], float(p))
        e["succ"] = r["ep_success"]
        e["grp"] = (r["col"], r["file"], r["reset"], r["group"])
    return list(ep.values())


def score(rows, probs, thr):
    y = [r["label"] for r in rows]
    eps = _episodes(rows, probs)
    fail = [e for e in eps if not e["succ"]]
    succ = [e for e in eps if e["succ"]]
    groups = defaultdict(list)
    for e in eps:
        groups[e["grp"]].append(e)
    allfail = [g for g in groups.values() if not any(e["succ"] for e in g)]
    pairs = agree = 0
    for g in groups.values():
        for a in (e for e in g if e["succ"]):
            for b in (e for e in g if not e["succ"]):
                pairs += 1
                agree += (a["p"] > b["p"]) + 0.5 * (a["p"] == b["p"])
    hard = [i for i, r in enumerate(rows) if r["category"].startswith("hard")]
    return {
        "frames": len(rows), "frame_pos": int(sum(y)),
        "frame_auroc": auroc(y, probs), "frame_ap": average_precision(y, probs),
        "hard_neg_fire": float(np.mean([probs[i] >= thr for i in hard])) if hard else None,
        "episodes": len(eps), "thr": float(thr),
        "fire_on_failure": float(np.mean([e["p"] >= thr for e in fail])) if fail else None,
        "miss_on_success": float(np.mean([e["p"] < thr for e in succ])) if succ else None,
        "allfail_groups": len(allfail),
        "allfail_groups_with_fp": float(np.mean([any(e["p"] >= thr for e in g) for g in allfail])) if allfail else None,
        "group_rank_acc": agree / pairs if pairs else None,
    }


def calibrate(rows, probs, max_fp=0.02):
    eps = _episodes(rows, probs)
    fail = sorted(e["p"] for e in eps if not e["succ"])
    if not fail:
        return 0.5
    k = int(np.floor(max_fp * len(fail)))  # allow at most k firing failures
    thr = fail[len(fail) - 1 - k] + 1e-6 if k < len(fail) else 0.0
    return float(min(max(thr, 0.05), 0.999))
