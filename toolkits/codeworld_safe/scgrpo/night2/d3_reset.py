"""D3 add-on: reset-time violations and the within-group signal left after removing them."""
import collections
import glob
import json
import sys

rows = [json.loads(l) for f in glob.glob(f"{sys.argv[1]}/*.jsonl") for l in open(f)]
n = len(rows)
for j, c in enumerate(("obj", "robot", "grasp")):
    k = sum(r["grader_at_reset"][j] for r in rows)
    print(f"reset-time c_{c}: {k}/{n} = {k / n:.3f}")
anyr = [r for r in rows if any(r["grader_at_reset"])]
print(f"reset-time any channel: {len(anyr)}/{n} = {len(anyr) / n:.3f}; "
      f"distinct keyframes {len({r['kf'] for r in anyr})} of {len({r['kf'] for r in rows})}")
print("by suite:", dict(collections.Counter(r["suite"] for r in anyr)))
g = collections.defaultdict(list)
for r in rows:
    g[(r["reset"], r["group"], r["kf"])].append(r)
clean = [v for v in g.values() if not any(any(x["grader_at_reset"]) for x in v)]
eps = [x for v in clean for x in v]
print(f"groups without any reset-time violation: {len(clean)}/{len(g)}")
for j, c in enumerate(("obj", "robot", "grasp")):
    rate = sum(x["grader_window"][j] for x in eps) / max(len(eps), 1)
    mixed = sum(len({x["grader_window"][j] for x in v}) > 1 for v in clean) / max(len(clean), 1)
    print(f"  c_{c}: rate {rate:.3f}, mixed groups {mixed:.2f}")
anyrate = sum(any(x["grader_window"]) for x in eps) / max(len(eps), 1)
anymixed = sum(len({int(any(x["grader_window"])) for x in v}) > 1 for v in clean) / max(len(clean), 1)
print(f"  any: rate {anyrate:.3f}, mixed groups {anymixed:.2f}")
