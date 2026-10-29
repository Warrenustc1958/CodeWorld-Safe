# Per suite x level paired comparison vs base on finished cells only (official SR, unsafe)
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo/diag")
sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
from matched_summ import load  # noqa: E402

base = load("off_base")
for name in sys.argv[1:]:
    rows = load(name)
    cells = {r["cell"] for r in rows}
    cells = {c for c in cells if sum(1 for r in rows if r["cell"] == c) == 20}
    by = defaultdict(lambda: [[], [], set()])
    for r in base:
        if r["cell"] in cells:
            by[(r["suite"], r["level"])][0].append(r)
    for r in rows:
        if r["cell"] in cells:
            by[(r["suite"], r["level"])][1].append(r)
            by[(r["suite"], r["level"])][2].add(r["cell"])
    print(f"== {name}: {len(cells)} cells")
    for (s, lv), (rb, ra, cs) in sorted(by.items()):
        sb, sa = np.mean([r["official_success"] for r in rb]), np.mean([r["official_success"] for r in ra])
        ub, ua = np.mean([r["unsafe"] for r in rb]), np.mean([r["unsafe"] for r in ra])
        print(f"  {s:24s} {lv}  tasks {len(cs)}/5  SR {sb:.3f} -> {sa:.3f} ({sa - sb:+.3f})  unsafe {ub:.3f} -> {ua:.3f}")
