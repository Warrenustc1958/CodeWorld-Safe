#!/usr/bin/env python3
"""Cross-check knowledge/tasks/*.json against LIBERO's official BDDL parser.

Run inside the LIBERO venv with LIBERO_CONFIG_PATH set:
  python crosscheck_knowledge.py --libero /mnt/nfs/fjj/Code/LIBERO-Safety --knowledge knowledge
Compares language, fixtures, objects, obj_of_interest, init, goal and constraints
after normalising case and stray commas. Exit code 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def norm(x):
    if isinstance(x, (list, tuple)):
        return tuple(norm(i) for i in x)
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).strip().rstrip(",").lower()


def as_set(items):
    return sorted(norm(i) for i in items)


def official_names(d):
    """fixtures/objects in the official parse are {type: [names]}."""
    if isinstance(d, dict):
        return sorted(norm(n) for names in d.values() for n in names)
    return sorted(norm(n) for n in d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", type=Path, required=True)
    ap.add_argument("--knowledge", type=Path, required=True)
    ap.add_argument("--show", nargs="*", default=[], help="task keys to print side by side")
    args = ap.parse_args()
    sys.path.insert(0, str(args.libero))
    from libero.libero.envs.bddl_utils import robosuite_parse_problem

    bad = 0
    files = sorted((args.knowledge / "tasks").glob("*.json"))
    for f in files:
        k = json.loads(f.read_text())
        o = robosuite_parse_problem(str(args.libero / k["bddl_file"]))
        checks = {
            "language": (norm(" ".join(o["language_instruction"]) if isinstance(o["language_instruction"], list)
                              else o["language_instruction"]), norm(k["instruction"])),
            "fixtures": (official_names(o["fixtures"]), sorted(norm(x["name"]) for x in k["fixtures"])),
            "objects": (official_names(o["objects"]), sorted(norm(x["name"]) for x in k["objects"])),
            "obj_of_interest": (as_set(o["obj_of_interest"]), as_set(k["obj_of_interest"])),
            "init": (as_set(o["initial_state"]), as_set(k["init"])),
            "goal": (as_set(o["goal_state"]), as_set(k["goal"])),
            "constraints": (as_set(o.get("constraints", [])), as_set(k["constraints"])),
        }
        diff = [name for name, (a, b) in checks.items() if a != b]
        bad += bool(diff)
        print(f"{k['key']:<34} {'OK' if not diff else 'MISMATCH ' + ','.join(diff)}")
        for name in diff:
            print(f"    {name}: official={checks[name][0]}\n    {' ' * len(name)}  ours    ={checks[name][1]}")
        if k["key"] in args.show:
            print(json.dumps({"official": {x: o.get(x) for x in ("goal_state", "constraints", "obj_of_interest")},
                              "hazards": k["hazards"], "fixture_state_changes": k["fixture_state_changes"],
                              "dynamic_objects": k["dynamic_objects"], "known_issues": k["known_issues"]},
                             indent=1, default=str))
    print(f"TOTAL {len(files)} tasks, {bad} mismatched")
    sys.exit(1 if bad or not files else 0)


if __name__ == "__main__":
    main()
