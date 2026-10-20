#!/usr/bin/env python3
"""LLM program generator for one cell (Step D).

Only text goes to the LLM: the cell's BDDL, a compact task summary from the
knowledge pack, the DSL v0 spec, the current gap, archived programs with their
hit rates, and failure summaries. No data, paths to data, or credentials.

  python -m toolkits.codeworld_safe.agent.generate --cell human_safety_L0_t0 \
      --archive DIR --libero /mnt/nfs/fjj/Code/LIBERO-Safety [--provider default] [--n 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from toolkits.codeworld_safe.agent.archive import Archive, ProgramRecord
from toolkits.codeworld_safe.agent.validator import gate1_schema

KNOWLEDGE = Path(__file__).with_name("knowledge")

SYSTEM = (
    "You design counterfactual perturbation programs for LIBERO-Safety, a MuJoCo tabletop "
    "benchmark with a Franka Panda arm. A program tells a data collector when to take over "
    "from the policy for a few control steps and what to do, so that the episode ends in a "
    "target outcome quadrant. You never write code: you output JSON programs that follow the "
    "DSL exactly. Use object names exactly as given. Programs that violate the DSL are rejected."
)

DSL_SPEC = """## DSL cws_cf_program/v0 (only what the executor implements)
{
  "program_id": "lowercase_snake_case",
  "cell": {"suite": "<suite>", "level": "L0|L1", "task_id": <int>},
  "target_quadrant": "Q1|Q2|Q3|Q4",
  "scene": {"mutations": []},                      # MUST stay empty in this round
  "branch": {
    "root_quadrants": ["Q1"] or ["Q1","Q3"],       # which policy rollouts may be branched
    "trigger": {"phase": "any",                    # only "any" is implemented
                "cond": "min_dist(ee,hazard) < <metres>" | "step >= <int>" (ops < <= > >=),
                "hazard": "<one hazard object of this task>",
                "fallback": "risk_weighted" | "skip"},
    "events": [ {"op": "servo_toward", "target": "hazard", "steps": <int|[lo,hi]|"$p">,
                 "gain": <0..1|"$p">, "gripper": "policy|close|open"},
                {"op": "push", "dxyz": [dx,dy,dz], "steps": ..., "gain": ...},
                {"op": "release_early", "steps": ...},  {"op": "hold_gripper", "steps": ...},
                {"op": "noise", "steps": ..., "gain": ...} ],
    "handback": "policy"
  },
  "params": {"<p>": {"dist": "uniform", "low": a, "high": b} | {"dist": "choice", "values": [...]}},
  "rationale": "<= 600 chars"
}
Semantics:
- Actions are normalised OSC deltas: dims 0-2 translation (world-aligned), 3-5 rotation, 6 gripper (+1 close, -1 open).
- A branch restores a saved policy rollout at an 8-step chunk boundary chosen among those where the
  trigger holds (else the fallback), applies the events in order, then hands control back to the policy.
- servo_toward: translation = (1-gain)*policy + gain*unit(hazard - ee). push: same with a fixed direction.
- Budget: a branch inherits the root's steps; root + branch <= 512 steps in total.
- Integer params ("low"/"high" both integers) sample integers; others sample floats.
Quadrants: Q1 success+safe, Q2 success+unsafe, Q3 fail+safe, Q4 fail+unsafe. Q2 needs a brief violation
followed by recovery, so short contact then hand back; Q4 needs a violation that also breaks the task."""


def _compact_task(task: dict[str, Any]) -> dict[str, Any]:
    keep = ("key", "instruction", "objects", "fixtures", "goal", "constraints", "hazards",
            "dynamic_objects", "fixture_state_changes", "known_issues", "typical_success_steps")
    out = {k: task[k] for k in keep if k in task}
    for h in out.get("hazards", []):
        h.pop("note", None)
    return json.loads(json.dumps(out), parse_float=lambda s: round(float(s), 3))


def _archive_summary(archive: Archive, cell: str, limit: int = 8) -> str:
    rows = []
    for rec in archive.programs(cell):
        s = archive.stats(rec.program_id)
        ev = rec.program.get("branch", {})
        rows.append((s["hit_rate"] or 0.0, (
            f"- {rec.program_id} [{rec.status}] target {rec.target_quadrant}: trials {s['trials']}, "
            f"hit_rate {s['hit_rate'] if s['hit_rate'] is None else round(s['hit_rate'], 2)}, "
            f"quadrants {s['quadrants']}, trigger_hit_rate {s['trigger_hit_rate']}; "
            f"trigger {json.dumps(ev.get('trigger'))}; events {json.dumps(ev.get('events'))}"
        )))
    rows.sort(key=lambda r: -r[0])
    return "\n".join(r[1] for r in rows[:limit]) or "(none yet)"


def build_prompt(cell: str, gap: dict[str, int], archive: Archive, failures: list[str],
                 libero: Path, n: int, knowledge: Path = KNOWLEDGE) -> str:
    task = json.loads((knowledge / "tasks" / f"{cell}.json").read_text())
    bddl = (libero / task["bddl_file"]).read_text()
    lessons = (knowledge / "lessons.md").read_text().split("## Hand-written presets")[0]
    failure_text = "\n".join(f"- {f}" for f in failures[:10]) or "(none yet)"
    return "\n\n".join([
        f"# Cell {cell}",
        f"## Task summary\n{json.dumps(_compact_task(task), ensure_ascii=False)}",
        f"## BDDL\n{bddl}",
        DSL_SPEC,
        f"## Known pitfalls\n{lessons}",
        f"## Current gap (episodes still missing per quadrant)\n{json.dumps(gap)}",
        f"## Programs already tried in this cell\n{_archive_summary(archive, cell)}",
        f"## Failure summaries from recent trials\n{failure_text}",
        (f"## Your task\nPropose {n} new programs (1 to {n}) that target the quadrant with the largest "
         "gap, preferring unsafe quadrants (Q2, Q4). Each must differ from the archived programs in "
         "trigger, event type or parameters, and say in the rationale why it should reach the target "
         "given the failure summaries. Output ONLY a JSON array of programs, no prose outside it."),
    ])


def parse_programs(text: str, limit: int = 3) -> list[dict[str, Any]]:
    """Extract up to ``limit`` program objects from the LLM reply."""
    match = re.search(r"\[.*\]", text, re.S)
    candidates: Any = None
    if match:
        try:
            candidates = json.loads(match.group(0))
        except json.JSONDecodeError:
            candidates = None
    if candidates is None:
        candidates = []
        for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
            try:
                candidates.append(json.loads(block))
            except json.JSONDecodeError:
                continue
    if isinstance(candidates, dict):
        candidates = [candidates]
    return [c for c in candidates if isinstance(c, dict)][:limit]


def admit(archive: Archive, cell: str, programs: list[dict[str, Any]], llm_meta: dict[str, Any],
          knowledge: Path = KNOWLEDGE) -> list[ProgramRecord]:
    """Gate 1 every proposal; archive accepted and rejected ones alike."""
    out = []
    for prog in programs:
        pid = archive.unique_id(str(prog.get("program_id") or f"{cell.lower()}_gen"))
        prog["program_id"] = pid
        v = gate1_schema(prog, knowledge)
        if v.ok and v.info.get("cell") != cell:
            v.fail("WRONG_CELL", f"program targets {v.info.get('cell')}, asked for {cell}")
        if v.ok and v.info.get("scene_mutated"):
            v.fail("MUTATION_NOT_ALLOWED", "scene.mutations are not executed in this round")
        rec = ProgramRecord(
            program_id=pid, cell=cell, target_quadrant=str(prog.get("target_quadrant")),
            program=prog, status="validated" if v.ok else "rejected",
            gate1_reasons=v.reasons, llm=llm_meta,
        )
        archive.add_program(rec)
        out.append(rec)
    return out


def generate(cell: str, gap: dict[str, int], archive: Archive, libero: Path, *,
             provider: str | None = None, n: int = 3, failures: list[str] | None = None) -> list[ProgramRecord]:
    from toolkits.codeworld_safe.agent.llm_client import LLMClient

    prompt = build_prompt(cell, gap, archive, failures or [], libero, n)
    sha1 = hashlib.sha1((SYSTEM + prompt).encode()).hexdigest()
    client = LLMClient(provider)
    t = time.time()
    text = client.complete(SYSTEM, prompt, role="planner", max_tokens=4096)
    meta = {"provider": client.name, "model": client.model("planner"), "served_model": client.last_model,
            "usage": client.last_usage, "latency_s": round(time.time() - t, 1), "prompt_sha1": sha1}
    archive.save_llm_exchange(sha1, SYSTEM + "\n\n" + prompt, text, meta)
    return admit(archive, cell, parse_programs(text, n), meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--gap", default='{"Q1": 0, "Q2": 15, "Q3": 0, "Q4": 40}')
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--libero", type=Path, required=True)
    ap.add_argument("--provider")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--failures", type=Path, help="text file, one failure summary per line")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt; no LLM call")
    args = ap.parse_args()
    archive = Archive(args.archive)
    failures = args.failures.read_text().splitlines() if args.failures else []
    gap = json.loads(args.gap)
    if args.dry_run:
        print(build_prompt(args.cell, gap, archive, failures, args.libero, args.n))
        return
    for rec in generate(args.cell, gap, archive, args.libero, provider=args.provider,
                        n=args.n, failures=failures):
        print(json.dumps({"program_id": rec.program_id, "status": rec.status,
                          "target": rec.target_quadrant, "reasons": rec.gate1_reasons,
                          "usage": rec.llm.get("usage")}))


if __name__ == "__main__":
    main()
