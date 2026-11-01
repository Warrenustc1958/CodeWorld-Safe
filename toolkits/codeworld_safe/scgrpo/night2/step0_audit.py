"""Step 0 offline analysis of real-eval audits (grader channels vs LIBERO predicates, violation timing).

  python step0_audit.py NAME [NAME ...]   (dirs under /data_cf/cws_sam/v1/scgrpo/eval_real) -> JSON on stdout
Per run and suite: unsafe per channel; for each channel's first violation: at reset (step <= 1), before the
first goal, after the first goal, after step 64 (training episode length); episodes the grader flags but no
predicate (official or aux) ever flags.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data_cf/cws_sam/v1/scgrpo/eval_real")
CH = ("c_obj", "c_robot", "c_grasp")
TRAIN_LEN = 64


def episode(js):
    r = json.load(open(js))
    traj = r["trajectory"]
    goal_t = next((i for i, s in enumerate(traj) if s["terminations"].get("raw_task")), None)
    first = dict.fromkeys(CH)
    pred_any = False
    for i, s in enumerate(traj):
        preds = s.get("predicates") or {}
        vals = preds.values() if isinstance(preds, dict) else preds
        if any(isinstance(p, dict) and p.get("source") in ("constraint", "aux_constraint") and p.get("value")
               for p in vals):
            pred_any = True
        for g in s.get("grades") or []:
            if g.get("violated") and g["channel"] in first and first[g["channel"]] is None:
                first[g["channel"]] = i
    return r["episode"]["task_suite"], goal_t, first, pred_any, len(traj)


def main(names):
    out = {}
    for name in names:
        agg = defaultdict(lambda: defaultdict(int))
        for js in sorted((ROOT / name).rglob("process-*.json")):
            if "visual" in js.parts:
                continue
            suite, goal_t, first, pred_any, n = episode(js)
            for key in (suite, "ALL"):
                a = agg[key]
                a["n"] += 1
                a["success"] += goal_t is not None
                a["unsafe"] += any(v is not None for v in first.values())
                a["grader_unsafe_no_predicate"] += any(v is not None for v in first.values()) and not pred_any
                for c, t in first.items():
                    if t is None:
                        continue
                    a[f"{c}"] += 1
                    a[f"{c}_at_reset"] += t <= 1
                    a[f"{c}_before_goal"] += goal_t is None or t <= goal_t
                    a[f"{c}_after_goal"] += goal_t is not None and t > goal_t
                    a[f"{c}_after_{TRAIN_LEN}"] += t >= TRAIN_LEN
        out[name] = {k: dict(v) for k, v in agg.items()}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
