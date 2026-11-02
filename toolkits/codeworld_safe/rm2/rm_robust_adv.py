"""Offline estimate of noise-robust GRPO advantage estimators on dumps (same rollouts, RM decision recomputed).

  python rm_robust_adv.py DUMP_DIR [DUMP_DIR ...]
Reference: the advantage training would get from the real outcome, A* = (r - mean) / max(std, 0.1) per group.
Methods (reward per episode -> group advantage):
  binary@thr floor f : r = 1[max chunk RM prob >= thr], std floor f       (f = 0.1 is the GRPO default)
  cont floor f       : r = max chunk RM prob (continuous), std floor f
  gate@thr band d    : binary@thr, but a group whose episodes have any prob in [thr - d, thr + d] gets A = 0
  cont+gate band d   : continuous, groups with any prob in [0.5 - d, 0.5 + d] get A = 0
Metrics over all episodes: cosine(A, A*); projection dot(A, A*)/|A*|^2 (share of the real direction, scale-dependent);
sign agreement in real-signal groups; share of groups that get a non-zero update.
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
            vals = [x for x in r["rm_chunk_max"] if x is not None]
            groups[(os.path.basename(f), r["reset"], r["group"])].append(
                (float(r["real_success"]), max(vals) if vals else 0.0, float(r["paid_success"])))
    return [np.array(g) for g in groups.values()]  # columns: real, prob, paid


def adv(x, floor):
    return (x - x.mean()) / max(x.std(), floor)


def evaluate(groups, method):
    A, R, keep, agree, tot = [], [], 0, 0, 0
    for g in groups:
        real, prob, paid = g[:, 0], g[:, 1], g[:, 2]
        a = method(real, prob, paid)
        if a is None:
            a = np.zeros(len(g))
        else:
            keep += bool(np.any(a != 0))
        ar = adv(real, 0.1)
        A.append(a)
        R.append(ar)
        if 0 < real.sum() < len(g):
            for x, y in zip(ar, a):
                tot += 1
                agree += np.sign(x) == np.sign(y)
    A, R = np.concatenate(A), np.concatenate(R)
    na, nr = np.linalg.norm(A), np.linalg.norm(R)
    return {"cos": float(A @ R / (na * nr)) if na and nr else 0.0, "proj": float(A @ R / nr ** 2),
            "agree": agree / tot if tot else 0.0, "kept": keep / len(groups)}


def methods(thr_run):
    m = {"real reward (reference)": lambda real, p, paid: adv(real, 0.1),
         "as run (paid, floor 0.1)": lambda real, p, paid: adv(paid, 0.1)}
    for thr in sorted({thr_run, 0.4, 0.5}):
        for f in (0.1, 0.25, 0.4, 0.5):
            m[f"binary@{thr:.2f} floor {f}"] = (lambda t, f: lambda real, p, paid: adv((p >= t).astype(float), f))(thr, f)
        for d in (0.1, 0.2, 0.3):
            def gate(real, p, paid, t=thr, d=d):
                if np.any((p >= t - d) & (p <= t + d)):
                    return None
                return adv((p >= t).astype(float), 0.1)
            m[f"gate@{thr:.2f} band {d}"] = gate
    for f in (0.1, 0.25, 0.4, 0.5):
        m[f"cont floor {f}"] = (lambda f: lambda real, p, paid: adv(p, f))(f)
    for d in (0.1, 0.2, 0.3):
        def cgate(real, p, paid, d=d):
            if np.any((p >= 0.5 - d) & (p <= 0.5 + d)):
                return None
            return adv(p, 0.25)
        m[f"cont+gate band {d} (floor 0.25)"] = cgate
    return m


def main(dirs):
    thr_run = {"e15a": 0.877, "e15q": 0.953}
    for d in dirs:
        groups = load(d)
        tag = os.path.basename(os.path.dirname(d.rstrip("/")))
        print(f"\n# {d}: {len(groups)} groups")
        print("| method | adv cosine | projection | sign agree (real-signal groups) | groups updated |")
        print("|---|---|---|---|---|")
        rows = [(k, evaluate(groups, fn)) for k, fn in methods(thr_run.get(tag, 0.85)).items()]
        for k, s in sorted(rows, key=lambda x: -x[1]["cos"]):
            print(f"| {k} | {s['cos']:.3f} | {s['proj']:.3f} | {s['agree']:.3f} | {s['kept']:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
