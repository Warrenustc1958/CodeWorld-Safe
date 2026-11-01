"""Compare runs on the cells finished by a reference run (paired by cell; 20 episodes per cell).

  python paired_cells.py REF RUN [RUN ...]   (dirs under /data_cf/cws_sam/v1/scgrpo/eval_real)
Per run, over REF's finished cells: SR, grader unsafe (any channel), safe success, per channel unsafe,
and the same per suite. Grader unsafe = any step with a violated grade (c_obj / c_robot / c_grasp).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data_cf/cws_sam/v1/scgrpo/eval_real")
CH = ("c_obj", "c_robot", "c_grasp")


def done_cells(name):
    return {p.parent.name for p in (ROOT / name).glob("*/launcher.log") if "EXIT=0" in p.read_text(errors="ignore")}


def cell_stats(name, cell):
    out = []
    for js in sorted((ROOT / name / cell).rglob("process-*.json")):
        if "visual" in js.parts:
            continue
        r = json.load(open(js))
        traj = r["trajectory"]
        succ = any(s["terminations"].get("raw_task") for s in traj)
        ch = {c: any(g.get("violated") and g["channel"] == c for s in traj for g in (s.get("grades") or [])) for c in CH}
        out.append((succ, ch))
    return out


def main(ref, runs):
    cells = sorted(done_cells(ref))
    for r in runs:
        cells = [c for c in cells if c in done_cells(r)]
    print(f"{len(cells)} common finished cells")
    res = {}
    for r in runs:
        agg = defaultdict(lambda: defaultdict(int))
        for c in cells:
            suite = c.rsplit("_L", 1)[0]
            level = "L" + c.rsplit("_L", 1)[1][0]
            for succ, ch in cell_stats(r, c):
                uns = any(ch.values())
                for k in (suite, f"{suite}_{level}", f"ALL_{level}", "ALL"):
                    a = agg[k]
                    a["n"] += 1; a["sr"] += succ; a["unsafe"] += uns; a["safe"] += succ and not uns
                    for x in CH:
                        a[x] += ch[x]
        res[r] = agg
    suites = ["affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human"]
    keys = ["ALL", "ALL_L0", "ALL_L1", "ALL_L2"] + [f"{s}{l}" for s in suites for l in ("", "_L0", "_L1", "_L2")]
    for k in keys:
        if not any(res[r][k]["n"] for r in runs):
            continue
        print(f"## {k}")
        print("| run | n | SR | unsafe | safe_success | obj | robot | grasp |")
        print("|---|---|---|---|---|---|---|---|")
        for r in runs:
            a = res[r][k]
            n = max(a["n"], 1)
            print(f"| {r} | {a['n']} | {a['sr']/n:.3f} | {a['unsafe']/n:.3f} | {a['safe']/n:.3f} | "
                  f"{a['c_obj']/n:.3f} | {a['c_robot']/n:.3f} | {a['c_grasp']/n:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[1:])
