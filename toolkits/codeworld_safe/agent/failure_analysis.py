#!/usr/bin/env python3
"""Step A0 probe: give the LLM one task's knowledge pack and ask ONLY for a failure-mode
analysis (no DSL program). Output is archived for manual review.

  python failure_analysis.py --provider default --task obstacle_avoidance_human_L1_t9 \
      --knowledge knowledge --out /mnt/nfs/fjj/Code/rlinf-safety-runtime/tmp/cws_a0 [--blind]

--blind replaces the task JSON with the raw BDDL text and drops the derived `hazards`,
so the model has to find the hazard objects from :constraints itself.
Only knowledge-pack text is sent; no data, paths to data, or credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from llm_client import LLMClient

SYSTEM = (
    "You are the planner of a counterfactual-data generator for LIBERO-Safety, a MuJoCo "
    "manipulation benchmark with a Franka Panda arm. Read the project knowledge and ONE task, "
    "then write a failure-mode analysis. Do NOT write any program, JSON or code. Be concrete: "
    "name objects exactly as in the task file."
)

QUESTIONS = """Write the failure-mode analysis for this task under these headings:
1. Task and success condition: what must be true at the end (cite the goal predicates).
2. Hazards: every hazard object, which predicate flags it, and whether its label is reliable.
3. Violation mechanisms: concrete ways the policy could violate each constraint, in which task phase (reach / grasp / carry / place / fixture operation), and why.
4. Quadrant routes: how an episode would end in Q2 (success but unsafe) and Q4 (fail and unsafe); which is harder to produce and why.
5. Traps: dynamics, reset randomness, label quirks, or rendering constraints a data generator must respect for this task.
Keep it under 600 words."""


def build_user(kdir: Path, key: str, blind: bool, libero: Path | None) -> str:
    task = json.loads((kdir / "tasks" / f"{key}.json").read_text())
    parts = [f"# project_map.md\n{(kdir / 'project_map.md').read_text()}",
             f"# predicates.md\n{(kdir / 'predicates.md').read_text()}",
             f"# lessons.md\n{(kdir / 'lessons.md').read_text()}"]
    if blind:
        if libero is None:
            raise SystemExit("--blind needs --libero to read the raw BDDL")
        bddl = (libero / task["bddl_file"]).read_text()
        parts.append(f"# task {key} (raw BDDL, {task['bddl_file']})\n{bddl}")
    else:
        parts.append(f"# tasks/{key}.json\n{json.dumps(task, indent=1)}")
    parts.append(QUESTIONS)
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider")
    ap.add_argument("--task", required=True)
    ap.add_argument("--knowledge", type=Path, default=Path(__file__).with_name("knowledge"))
    ap.add_argument("--libero", type=Path)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    user = build_user(args.knowledge, args.task, args.blind, args.libero)
    client = LLMClient(args.provider)
    t = time.time()
    text = client.complete(SYSTEM, user, role="planner", max_tokens=2048)
    meta = {
        "task": args.task, "blind": args.blind, "provider": client.name,
        "model": client.model("planner"), "served_model": client.last_model,
        "usage": client.last_usage, "latency_s": round(time.time() - t, 1),
        "prompt_chars": len(SYSTEM) + len(user),
        "prompt_sha1": hashlib.sha1((SYSTEM + user).encode()).hexdigest(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}{'_blind' if args.blind else ''}_{client.name}"
    (args.out / f"{stem}.prompt.txt").write_text(SYSTEM + "\n\n" + user)
    (args.out / f"{stem}.md").write_text(f"<!-- {json.dumps(meta)} -->\n\n{text}\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
