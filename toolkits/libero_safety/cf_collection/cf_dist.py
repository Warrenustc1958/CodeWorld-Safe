"""Read-only Q1-Q4 distribution for every cell under COLLECT_ROOT (does not touch state)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/mnt/nfs/fjj/Code/rlinf-safety-runtime/scripts")
import cf_coverage as cc  # noqa: E402

root = Path(sys.argv[1])
cache = {}
for p in (root / "state").glob("*.index.json"):
    try:
        cache.update(json.loads(p.read_text()))
    except Exception:
        pass
recs = []
for path in (root / "runs").rglob("libero_safety_audits/process_*/*.json"):
    st = path.stat()
    hit = cache.get(str(path))
    r = hit[1] if hit and hit[0] == [st.st_mtime_ns, st.st_size] else cc._parse(path, root)
    if r:
        recs.append(r)

cells = defaultdict(list)
for r in recs:
    cells[(r["suite"], r["level"])].append(r)
order = ["human_safety", "obstacle_avoidance", "obstacle_avoidance_human", "affordance"]
for s, l in sorted(cells, key=lambda k: (order.index(k[0]) if k[0] in order else 9, k[1])):
    rs = cells[(s, l)]
    tasks = sorted({r["task"] for r in rs})
    summ = cc.summarize(rs, tasks)
    tot = defaultdict(int)
    print(f"== {s} {l}")
    for t in tasks:
        x = summ[t]
        n = sum(x[q] for q in ("Q1", "Q2", "Q3", "Q4"))
        for q in ("Q1", "Q2", "Q3", "Q4"):
            tot[q] += x[q]
        print(f"task{t}|{x['Q1']}|{x['Q2']}|{x['Q3']}|{x['Q4']}|{n}|{x['invalid']}|{x['Rall']}")
    print(f"SUM|{tot['Q1']}|{tot['Q2']}|{tot['Q3']}|{tot['Q4']}|{sum(tot.values())}")
