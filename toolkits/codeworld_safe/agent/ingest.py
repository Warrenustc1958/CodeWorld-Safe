#!/usr/bin/env python3
"""Ingest one trial batch of program-driven audits into the archive (Step D).

  python -m toolkits.codeworld_safe.agent.ingest --audits DIR --archive DIR [--chunk 8]

For every branch episode carrying a ``program_id``: gate 4 (chunk-aligned,
sim-state sidecar required), trajectory feature, dedup against the archive,
outcome row. Prints per-program stats and writes ``failures_<cell>.txt``
(one line per program) for the next generation prompt.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from toolkits.codeworld_safe.agent.archive import Archive, trajectory_feature
from toolkits.codeworld_safe.agent.validator import gate4_episode, load_episodes


def _cell(meta: dict) -> str:
    return f"{meta.get('task_suite')}_{meta.get('safety_level')}_t{meta.get('task_id')}"


def _min_hazard_dist(record: dict) -> float | None:
    vals = [g["min_dist"] for s in record["trajectory"] for g in (s.get("grades") or [])]
    return float(min(vals)) if vals else None


def ingest(audits: Path, archive: Archive, chunk: int = 8) -> dict[str, dict]:
    lookup, failures = load_episodes(sorted(Path(audits).rglob("process-*.json")))
    per_program: dict[str, list[dict]] = defaultdict(list)
    for ep in lookup.values():
        meta = ep.record["episode"]
        pid = meta.get("program_id")
        if not pid:
            continue
        cell = _cell(meta)
        if pid not in {r.program_id for r in archive.programs()}:
            raise ValueError(f"{ep.audit_path}: program {pid} is not in the archive")
        v = gate4_episode(ep, lookup, chunk=chunk, require_sim_state=True)
        row = archive.add_outcome(
            program_id=pid, cell=cell, quadrant=ep.quadrant,
            feature=trajectory_feature(ep.record), gate4_ok=v.ok, gate4_codes=v.codes,
            audit_path=str(ep.audit_path),
            extra={
                "trigger_hit": bool((meta.get("program_branch") or {}).get("trigger_hit")),
                "end_reason": ep.record["outcome"].get("end_reason"),
                "min_hazard_dist": _min_hazard_dist(ep.record),
                "branch_timestep": meta.get("branch_timestep"),
            },
        )
        per_program[pid].append(row)
    return {"programs": {pid: archive.stats(pid) for pid in per_program},
            "schema_failures": failures,
            "rows": {pid: rows for pid, rows in per_program.items()}}


def failure_lines(result: dict) -> dict[str, list[str]]:
    by_cell: dict[str, list[str]] = defaultdict(list)
    for pid, rows in result["rows"].items():
        s = result["programs"][pid]
        dists = [r["min_hazard_dist"] for r in rows if r.get("min_hazard_dist") is not None]
        by_cell[s["cell"]].append(
            f"{pid} (target {s['target_quadrant']}): {s['trials']} branches, quadrants {s['quadrants']}, "
            f"trigger held at {sum(r['trigger_hit'] for r in rows)}/{len(rows)} branch points, "
            f"end reasons {dict(Counter(r['end_reason'] for r in rows))}, "
            f"gate4 rejects {dict(Counter(c for r in rows for c in r['gate4_codes']))}, "
            f"duplicates {s['duplicates']}"
            + (f", min ee-hazard distance median {np.median(dists):.3f} m" if dists else "")
        )
    return by_cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", type=Path, required=True)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()
    archive = Archive(args.archive)
    result = ingest(args.audits, archive, args.chunk)
    for cell, lines in failure_lines(result).items():
        (args.archive / f"failures_{cell}.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"programs": result["programs"], "schema_failures": len(result["schema_failures"])},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
