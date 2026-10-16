#!/usr/bin/env python3
"""Quota coverage, stall detection and resume state for collect_counterfactual_l0l1.sh.

Commands
--------
status      Scan audits of one suite/level, print a per-task coverage table,
            update stall state, and print ``PENDING=<ids>`` / ``ROUND=<n>``.
round-done  Mark one finished collection round for the given tasks.

Quotas per task (all counts are valid episodes: JSON + matching visual NPZ):
  Q1 >= q1_min with >= min_roots distinct roots
  Q3 >= q3_min with >= min_roots distinct roots
  Q2+Q4 >= unsafe_min with >= min_unsafe_roots distinct roots
  Q2 >= q2_soft (soft: only drives rounds until it stalls)

A requirement is infeasible once its (count, roots) did not grow over the last
``stall_rounds`` rounds of that task. A task is pending while any unmet
requirement is still feasible and the task has fewer than ``max_rounds`` rounds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "libero_safety_episode_audit/v1"
REQUIREMENTS = ("q1", "q3", "unsafe", "q2soft")


def _field(record: dict, key: str):
    value = record.get(key)
    if value is None:
        value = record.get("episode", {}).get(key)
    return value


def _parse(path: Path, collect_root: Path) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if record.get("schema_version") != SCHEMA:
        return None
    episode = record.get("episode", {})
    trajectory = record.get("trajectory", [])
    visual = record.get("visual_trajectory", {}) or {}
    rel = visual.get("path")
    valid = bool(
        trajectory
        and rel
        and int(visual.get("num_frames", -1)) == len(trajectory) + 1
        and (path.parent / rel).is_file()
    )
    # Episode ids are only unique within one process directory of one run.
    scope = os.path.relpath(path.parent, collect_root)
    root_id = _field(record, "root_episode_id") or episode.get("episode_id")
    return {
        "suite": str(episode.get("task_suite")),
        "level": str(episode.get("safety_level")),
        "task": int(episode.get("task_id", -1)),
        "q": record.get("outcome", {}).get("quadrant", record.get("quadrant")),
        "root": f"{scope}::{root_id}",
        "valid": valid,
    }


def scan(collect_root: Path, suite: str, level: str, cache_path: Path) -> list[dict]:
    """Return parsed records for one cell, reusing an mtime/size keyed cache."""
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    runs = collect_root / "runs"
    prefix = f"{suite}_{level}_"
    fresh, records = {}, []
    if runs.is_dir():
        for run_dir in sorted(runs.iterdir()):
            # obstacle_avoidance_L0_ must not match obstacle_avoidance_human_L0_.
            if not run_dir.name.startswith(prefix):
                continue
            for path in run_dir.rglob("libero_safety_audits/process_*/*.json"):
                stat = path.stat()
                key = str(path)
                stamp = [stat.st_mtime_ns, stat.st_size]
                hit = cache.get(key)
                record = hit[1] if hit and hit[0] == stamp else _parse(path, collect_root)
                if record is None:
                    continue
                fresh[key] = [stamp, record]
                records.append(record)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fresh), encoding="utf-8")
    tmp.replace(cache_path)
    return [r for r in records if r["suite"] == suite and r["level"] == level]


def summarize(records: list[dict], task_ids: list[int]) -> dict[int, dict]:
    counts = {t: defaultdict(int) for t in task_ids}
    roots = {t: defaultdict(set) for t in task_ids}
    invalid = defaultdict(int)
    for record in records:
        task = record["task"]
        if task not in counts:
            continue
        if not record["valid"]:
            invalid[task] += 1
            continue
        q = record["q"]
        if q not in ("Q1", "Q2", "Q3", "Q4"):
            continue
        counts[task][q] += 1
        roots[task][q].add(record["root"])
    out = {}
    for task in task_ids:
        c, r = counts[task], roots[task]
        out[task] = {
            "Q1": c["Q1"], "Q2": c["Q2"], "Q3": c["Q3"], "Q4": c["Q4"],
            "R1": len(r["Q1"]), "R2": len(r["Q2"]), "R3": len(r["Q3"]),
            "R4": len(r["Q4"]),
            "Ru": len(r["Q2"] | r["Q4"]),
            "Rall": len(r["Q1"] | r["Q2"] | r["Q3"] | r["Q4"]),
            "invalid": invalid[task],
        }
    return out


def requirement_state(snap: dict, args) -> dict[str, tuple[bool, tuple[int, int]]]:
    """Return {requirement: (met, progress)} for one coverage snapshot."""
    unsafe = snap["Q2"] + snap["Q4"]
    return {
        "q1": (snap["Q1"] >= args.q1_min and snap["R1"] >= args.min_roots,
               (snap["Q1"], snap["R1"])),
        "q3": (snap["Q3"] >= args.q3_min and snap["R3"] >= args.min_roots,
               (snap["Q3"], snap["R3"])),
        "unsafe": (unsafe >= args.unsafe_min and snap["Ru"] >= args.min_unsafe_roots,
                   (unsafe, snap["Ru"])),
        "q2soft": (snap["Q2"] >= args.q2_soft, (snap["Q2"], snap["R2"])),
    }


def cmd_status(args) -> int:
    collect_root = Path(args.collect_root)
    state_dir = collect_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    cell = f"{args.suite}_{args.level}"
    state_path = state_dir / f"{cell}.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"tasks": {}}
    records = scan(collect_root, args.suite, args.level, state_dir / f"{cell}.index.json")
    summary = summarize(records, args.tasks)

    pending, max_round = [], 0
    lines = []
    for task in args.tasks:
        entry = state["tasks"].setdefault(
            str(task), {"rounds": 0, "history": [], "infeasible": []}
        )
        snap = summary[task]
        # history[i] = coverage after i rounds; append once per completed round.
        if len(entry["history"]) < entry["rounds"] + 1:
            entry["history"].append(snap)
        else:
            entry["history"][-1] = snap
        reqs = requirement_state(snap, args)
        history = entry["history"]
        infeasible = set(entry["infeasible"])
        for name, (met, progress) in reqs.items():
            if met or name in infeasible:
                continue
            if len(history) > args.stall_rounds:
                old = requirement_state(history[-1 - args.stall_rounds], args)[name][1]
                if progress <= old:
                    infeasible.add(name)
        entry["infeasible"] = sorted(infeasible)
        unmet = [n for n, (met, _) in reqs.items() if not met]
        open_reqs = [n for n in unmet if n not in infeasible]
        exhausted = entry["rounds"] >= args.max_rounds
        if not unmet:
            status = "DONE"
        elif not open_reqs:
            status = "DONE-PARTIAL(infeasible:" + ",".join(unmet) + ")"
        elif exhausted:
            status = "MAX_ROUNDS(unmet:" + ",".join(unmet) + ")"
        else:
            status = "pending(" + ",".join(open_reqs) + ")"
            pending.append(task)
            max_round = max(max_round, entry["rounds"])
        lines.append(
            f"[coverage] {cell} task-{task}: "
            f"Q1={snap['Q1']}/{args.q1_min} Q2={snap['Q2']}/{args.q2_soft}(soft) "
            f"Q3={snap['Q3']}/{args.q3_min} Q4={snap['Q4']} "
            f"unsafe={snap['Q2'] + snap['Q4']}/{args.unsafe_min} | roots "
            f"R1={snap['R1']} R2={snap['R2']} R3={snap['R3']} R4={snap['R4']} "
            f"Ru={snap['Ru']} all={snap['Rall']} (min {args.min_roots}/"
            f"{args.min_unsafe_roots}) | invalid={snap['invalid']} "
            f"rounds={entry['rounds']} | {status}"
        )
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(state_path)
    for line in lines:
        print(line, file=sys.stderr)
    print("PENDING=" + ",".join(map(str, pending)))
    print(f"ROUND={max_round + 1}")
    return 0


def cmd_round_done(args) -> int:
    state_path = Path(args.collect_root) / "state" / f"{args.suite}_{args.level}.json"
    state = json.loads(state_path.read_text())
    for task in args.tasks:
        state["tasks"][str(task)]["rounds"] += 1
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(state_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "round-done"))
    parser.add_argument("--collect-root", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--tasks", type=lambda s: [int(x) for x in s.split(",") if x],
                        required=True)
    parser.add_argument("--q1-min", type=int, default=25)
    parser.add_argument("--q3-min", type=int, default=25)
    parser.add_argument("--unsafe-min", type=int, default=50)
    parser.add_argument("--q2-soft", type=int, default=10)
    parser.add_argument("--min-roots", type=int, default=15)
    parser.add_argument("--min-unsafe-roots", type=int, default=10)
    parser.add_argument("--stall-rounds", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=20)
    args = parser.parse_args()
    return cmd_status(args) if args.command == "status" else cmd_round_done(args)


if __name__ == "__main__":
    raise SystemExit(main())
