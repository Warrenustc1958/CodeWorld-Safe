"""Compare partial ablation evals with base on the SAME cells only (official SR, unsafe, gripper flips)."""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
from summ_eval_real import load  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path("/data_cf/cws_sam/v1/scgrpo/eval_real")


def flips(name):
    out = {}
    for js in (ROOT / name).rglob("process-*.json"):
        if "visual" in js.parts:
            continue
        r = json.load(open(js))
        g = np.sign([s["action"][6] for s in r["trajectory"]])
        e = r["episode"]
        out[(f"{e['task_suite']}_{e['safety_level']}_t{e['task_id']}", e["trial_id"])] = int((g[1:] != g[:-1]).sum())
    return out


base = load("off_base")
bf = flips("off_base")
for name in sys.argv[1:]:
    rows = load(name)
    cells = {r["cell"] for r in rows}
    cells = {c for c in cells if sum(1 for r in rows if r["cell"] == c) == 20}
    b = [r for r in base if r["cell"] in cells]
    a = [r for r in rows if r["cell"] in cells]
    af = flips(name)
    print(f"== {name}: {len(cells)} matched cells, {len(a)} eps")
    by = defaultdict(lambda: [[], []])
    for r in b:
        by[r["suite"]][0].append(r)
    for r in a:
        by[r["suite"]][1].append(r)
    for s, (rb, ra) in sorted(by.items()):
        fb = np.mean([bf.get((r["cell"], r["trial"]), 0) for r in rb])
        fa = np.mean([af.get((r["cell"], r["trial"]), 0) for r in ra])
        print(f"  {s:24s} n {len(ra):3d}  SR base {np.mean([r['official_success'] for r in rb]):.3f} -> {np.mean([r['official_success'] for r in ra]):.3f}"
              f"  unsafe {np.mean([r['unsafe'] for r in rb]):.3f} -> {np.mean([r['unsafe'] for r in ra]):.3f}  flips/ep {fb:.2f} -> {fa:.2f}")
    print(f"  {'ALL':24s} n {len(a):3d}  SR base {np.mean([r['official_success'] for r in b]):.3f} -> {np.mean([r['official_success'] for r in a]):.3f}"
          f"  unsafe {np.mean([r['unsafe'] for r in b]):.3f} -> {np.mean([r['unsafe'] for r in a]):.3f}")
