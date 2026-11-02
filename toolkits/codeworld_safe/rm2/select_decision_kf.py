"""A: select keyframes shortly before the decision moment of their source trajectory, and preview m on old dumps.

  python select_decision_kf.py OUT_NPY [--lo 8] [--hi 48] [DUMP_DIR ...]
Decision step D (global timestep of the source trajectory; branch: global = local + branch_timestep - 1):
  success          -> first raw task success
  failed, violated -> first violation (official outcome or first grader event of any channel, whichever is earlier)
  failed, timeout  -> branch point + 32 for counterfactual branches (the perturbation decides), undefined for roots
Keep keyframe k if D - hi <= t0_k <= D - lo (a 64-step rollout from k then covers D) and k is not in kf_exclude.
Preview: on existing per-episode dumps, mixed-group fraction m for groups whose keyframe is selected vs not.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
from collections import Counter, defaultdict

import numpy as np

S = "/data_cf/cws_sam/v1/scgrpo"
KF = f"{S}/keyframes_v1"


def decision(rec):
    try:
        out = json.load(open(rec["json"]))["outcome"]
    except Exception:
        return rec["uid"], None, "unreadable"
    off = int(rec["branch_timestep"]) - 1 if rec["kind"] != "root" and rec.get("branch_timestep") else 0
    if out.get("first_raw_task_success_timestep") is not None and rec["success"]:
        return rec["uid"], off + int(out["first_raw_task_success_timestep"]), "success"
    cands = [off + int(out["first_violation_timestep"])] if out.get("first_violation_timestep") is not None else []
    cands += [int(v) for v in (rec.get("first_event_by_channel") or {}).values() if v is not None]
    if cands:
        return rec["uid"], min(cands), "violation"
    if rec["kind"] != "root" and rec.get("branch_timestep"):
        return rec["uid"], off + 32, "timeout_branch"
    return rec["uid"], None, "timeout_root"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dumps", nargs="*")
    ap.add_argument("--lo", type=int, default=8)
    ap.add_argument("--hi", type=int, default=48)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open("/data_cf/cws_sam/v1/labels/episodes.jsonl")]
    with mp.Pool(32) as pool:
        dec = pool.map(decision, recs, chunksize=64)
    D = {u: d for u, d, _ in dec}
    print("decision type:", dict(Counter(t for _, _, t in dec)))
    meta = np.load(f"{KF}/meta.npz")
    uid, t0 = [str(u) for u in meta["uid"]], meta["t0"].astype(int)
    excl = set(np.load(f"{S}/kf_scan/kf_exclude.npy").tolist())
    sel = np.zeros(len(uid), bool)
    for i, (u, t) in enumerate(zip(uid, t0)):
        d = D.get(u)
        sel[i] = d is not None and d - a.hi <= t <= d - a.lo and i not in excl
    np.save(a.out, sel.astype(np.float64))
    tc = np.array([str(x) for x in meta["tcell"]])
    print(f"selected {int(sel.sum())} / {len(sel)} keyframes, cells covered {len(set(tc[sel]))} / {len(set(tc))}")
    print("by kind:", dict(Counter(int(k) for k in meta["kind"][sel])), "by quadrant:", dict(Counter(str(q) for q in meta["quadrant"][sel])))
    print("by suite:", dict(Counter(str(s) for s in meta["suite"][sel])))
    per_cell = Counter(tc[sel])
    print("min / median keyframes per cell:", min(per_cell.values()), int(np.median(list(per_cell.values()))))
    # preview on old dumps
    for d in a.dumps:
        g = defaultdict(list)
        for f in glob.glob(f"{d}/episodes_*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                g[(os.path.basename(f), r["reset"], r["group"])].append(r)
        rows = {True: [], False: []}
        for v in g.values():
            if len(v) == 8:
                rows[bool(sel[v[0]["kf"]])].append(0 < sum(x["real_success"] for x in v) < 8)
        print(f"{d}: m selected {np.mean(rows[True]) if rows[True] else float('nan'):.3f} (n {len(rows[True])}) "
              f"vs other {np.mean(rows[False]):.3f} (n {len(rows[False])})")


if __name__ == "__main__":
    main()
