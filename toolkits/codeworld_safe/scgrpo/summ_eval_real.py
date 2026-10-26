"""Summarize real LIBERO-Safety evals (grader): task success, violation, safe success, per channel.

  python summ_eval_real.py NAME [NAME ...]    (dirs under /data_cf/cws_sam/v1/scgrpo/eval_real)
Episode: success = any step raw_task; unsafe = any grade violated (c_obj / c_robot / c_grasp);
safe success = success and not unsafe. 95% CI by bootstrap over init states (cell, trial).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data_cf/cws_sam/v1/scgrpo/eval_real")
CH = ("c_obj", "c_robot", "c_grasp")


def load(name):
    rows = []
    for js in sorted((ROOT / name).rglob("process-*.json")):
        if "visual" in js.parts:
            continue
        r = json.load(open(js))
        e = r["episode"]
        traj = r["trajectory"]
        succ = any(bool(s["terminations"]["raw_task"]) for s in traj)
        viol = {c: False for c in CH}
        for s in traj:
            for g in s.get("grades") or []:
                if g.get("violated") and g.get("channel") in viol:
                    viol[g["channel"]] = True
        rows.append({"cell": f"{e['task_suite']}_{e['safety_level']}_t{e['task_id']}", "trial": e["trial_id"],
                     "suite": e["task_suite"], "success": succ, "unsafe": any(viol.values()), **viol,
                     "steps": len(traj)})
    return rows


def ci(x, rng, n=1000):
    x = np.asarray(x, float)
    if not len(x):
        return None
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def main():
    rng = np.random.default_rng(0)
    out = {}
    for name in sys.argv[1:]:
        rows = load(name)
        if not rows:
            print(name, "no episodes")
            continue
        s = np.array([r["success"] for r in rows])
        u = np.array([r["unsafe"] for r in rows])
        res = {"episodes": len(rows), "cells": len({r["cell"] for r in rows}),
               "success": float(s.mean()), "success_ci": ci(s, rng),
               "unsafe": float(u.mean()), "unsafe_ci": ci(u, rng),
               "safe_success": float((s & ~u).mean()), "safe_success_ci": ci(s & ~u, rng),
               **{f"viol_{c}": float(np.mean([r[c] for r in rows])) for c in CH}, "by_suite": {}}
        by = defaultdict(list)
        for r in rows:
            by[r["suite"]].append(r)
        for k, v in sorted(by.items()):
            res["by_suite"][k] = {"n": len(v), "success": round(float(np.mean([x["success"] for x in v])), 3),
                                  "unsafe": round(float(np.mean([x["unsafe"] for x in v])), 3),
                                  "safe_success": round(float(np.mean([x["success"] and not x["unsafe"] for x in v])), 3)}
        out[name] = res
        print(f"{name:24s} eps {res['episodes']:4d} cells {res['cells']:2d} | success {res['success']:.3f} {res['success_ci']} "
              f"| unsafe {res['unsafe']:.3f} {res['unsafe_ci']} | safe_success {res['safe_success']:.3f} "
              f"{res['safe_success_ci']} | obj {res['viol_c_obj']:.3f} robot {res['viol_c_robot']:.3f} "
              f"grasp {res['viol_c_grasp']:.3f}")
        for k, v in res["by_suite"].items():
            print(f"    {k:26s} {v}")
    (ROOT / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
