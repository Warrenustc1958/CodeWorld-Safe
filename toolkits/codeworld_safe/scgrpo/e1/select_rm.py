"""Pick the RM for E9 from runs/gate_d2g.json: among candidates whose best final-rule entry has recall >= .75,
the highest final-rule precision; it must beat RM v3 (stored scores) under the same criterion, else RM v3 is kept.
Prints: <name> <checkpoint path> <threshold> <consecutive> <views>
  python select_rm.py name=path [name=path ...]"""
import json
import sys

V3 = "/data_cf/cws_rm/runs/rm_v3_agent_w41k/best.pt"
rep = json.load(open("/data_cf/cws_rm/runs/gate_d2g.json"))
paths = dict(s.split("=", 1) for s in sys.argv[1:])


def best(name):
    r = rep[name]
    ok = [(v["final"]["P"], k) for k, v in r.items() if isinstance(v, dict) and "final" in v and v["final"]["R"] >= 0.75]
    return max(ok) if ok else (0.0, None)


p3, r3 = best("v3_stored")
choice = ("v3", V3, "0.8x4", 1, p3)  # if nothing beats v3, keep E8-b's exact RM and rule (E9 = E8-b replicate)
for name, path in paths.items():
    if name not in rep:
        continue
    p, rule = best(name)
    if rule and p > choice[4]:
        choice = (name, path, rule, rep[name]["views"], p)
name, path, rule, views, p = choice
thr, k = rule.split("x")
print(name, path, thr, k, views)
print(f"# selected {name} final-rule P {p:.3f} (v3 {p3:.3f} with {r3})", file=sys.stderr)
