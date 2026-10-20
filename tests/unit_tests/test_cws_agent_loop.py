# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Archive, planner and generator (prompt/parse/gate-1) without any LLM call."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from test_libero_cf_program import PROGRAM, TASK
from toolkits.codeworld_safe.agent.archive import Archive, ProgramRecord, distance
from toolkits.codeworld_safe.agent.generate import admit, build_prompt, parse_programs
from toolkits.codeworld_safe.agent.planner import Planner, coverage_gaps, split_budget

CELL = TASK["key"]


def _knowledge(tmp_path: Path) -> Path:
    k = tmp_path / "knowledge"
    (k / "tasks").mkdir(parents=True)
    (k / "tasks" / f"{CELL}.json").write_text(json.dumps({**TASK, "bddl_file": "t.bddl"}))
    (k / "lessons.md").write_text("# Lessons\n- P2 keep branch points inside the root\n## Hand-written presets\nX")
    (tmp_path / "t.bddl").write_text("(define (problem p) (:objects toy_car_1 - toy_car))")
    return k


def _feature(offset: float) -> np.ndarray:
    return np.full(16 * 3, offset)


def _record(archive: Archive, pid: str, status="validated", target="Q2") -> None:
    prog = copy.deepcopy(PROGRAM)
    prog["program_id"] = pid
    archive.add_program(ProgramRecord(pid, CELL, target, prog, status=status))


def test_archive_dedup_novelty_and_stats(tmp_path):
    a = Archive(tmp_path / "arch")
    _record(a, "p1")
    _record(a, "p2")
    first = a.add_outcome(program_id="p1", cell=CELL, quadrant="Q2", feature=_feature(0.0),
                          gate4_ok=True, gate4_codes=[], audit_path="a")
    dup = a.add_outcome(program_id="p2", cell=CELL, quadrant="Q2", feature=_feature(0.005),
                        gate4_ok=True, gate4_codes=[], audit_path="b")
    far = a.add_outcome(program_id="p2", cell=CELL, quadrant="Q4", feature=_feature(0.04),
                        gate4_ok=True, gate4_codes=[], audit_path="c")
    bad = a.add_outcome(program_id="p1", cell=CELL, quadrant="Q4", feature=_feature(0.2),
                        gate4_ok=False, gate4_codes=["TOO_SHORT"], audit_path="d")
    assert first["admitted"] and not dup["admitted"] and dup["duplicate"]
    assert far["admitted"] and not bad["admitted"] and not bad["duplicate"]
    assert abs(distance(_feature(0.0), _feature(0.04)) - 0.04 * np.sqrt(3)) < 1e-9
    s1, s2 = a.stats("p1"), a.stats("p2")
    assert s1["trials"] == 2 and s1["hit_rate"] == 0.5 and s1["gate4_pass_rate"] == 0.5
    assert s2["duplicates"] == 1 and s2["admitted"] == 1
    assert abs(s2["novelty_m"] - 0.04 * np.sqrt(3)) < 1e-6  # nearest other-program outcome
    assert a.cell_counts(CELL) == {"Q1": 0, "Q2": 1, "Q3": 0, "Q4": 1}
    # Reload from disk: last status record wins.
    a.set_status("p1", "rejected")
    b = Archive(tmp_path / "arch")
    assert b.program("p1").status == "rejected" and len(b.outcomes()) == 4
    assert b.unique_id("p1") == "p1_v2"


def test_gaps_and_budget_split():
    counts = {"a": {"Q1": 50, "Q2": 0, "Q3": 40, "Q4": 0}, "b": {"Q1": 50, "Q2": 15, "Q3": 40, "Q4": 30},
              "done": {"Q1": 50, "Q2": 15, "Q3": 40, "Q4": 40}}
    gaps = coverage_gaps(counts)
    assert set(gaps) == {"a", "b"} and gaps["a"]["Q4"] == 40 and gaps["b"]["Q4"] == 10
    alloc = split_budget(gaps, 100)
    assert sum(alloc.values()) == 100 and alloc["a"] > alloc["b"]
    assert alloc["a"] == round(100 * 110 / 130) or abs(alloc["a"] - 100 * 110 / 130) < 1


def test_planner_exploits_validated_then_generates_and_tracks_stagnation(tmp_path):
    a = Archive(tmp_path / "arch")
    _record(a, "good", target="Q4")
    _record(a, "untried", target="Q2")
    _record(a, "bad", status="rejected", target="Q4")
    for i in range(5):
        a.add_outcome(program_id="good", cell=CELL, quadrant="Q4" if i < 3 else "Q3",
                      feature=_feature(0.1 * i), gate4_ok=True, gate4_codes=[], audit_path=str(i))
    planner = Planner(a, tmp_path / "state.json")
    counts = {CELL: a.cell_counts(CELL)}
    plan = planner.plan_round(counts, budget=60)
    runs = [p for p in plan if p.action == "run" and p.program_id]
    assert {p.program_id for p in runs} <= {"good", "untried"} and "bad" not in {p.program_id for p in runs}
    assert runs[0].program_id == "good"
    assert sum(p.episodes for p in plan if p.action == "run") == 60
    # The first round grows from nothing (baseline); then 2 stagnant rounds allow
    # mutation and a 3rd marks the cell infeasible.
    planner.end_round(counts)
    assert planner.state[CELL].stagnant == 0
    for _ in range(2):
        planner.end_round(counts)
    assert all(p.allow_mutation for p in planner.plan_round(counts, 10) if p.action == "generate")
    planner.end_round(counts)
    assert planner.state[CELL].infeasible
    assert planner.plan_round(counts, 10) == []
    assert Planner(a, tmp_path / "state.json").state[CELL].infeasible  # persisted


def test_prompt_contains_inputs_and_no_data_paths(tmp_path):
    k = _knowledge(tmp_path)
    a = Archive(tmp_path / "arch")
    _record(a, "p1")
    prompt = build_prompt(CELL, {"Q1": 0, "Q2": 15, "Q3": 0, "Q4": 40}, a,
                          ["p1: trigger held at 0/8 branch points"], tmp_path, 3, knowledge=k)
    for needle in ("## BDDL", "toy_car_1", "cws_cf_program/v0", '"Q4": 40', "p1 [validated]",
                   "trigger held at 0/8", "JSON array"):
        assert needle in prompt, needle
    assert "/mnt/" not in prompt and "audit" not in prompt.lower() and "api_key" not in prompt.lower()


def test_parse_and_gate1_admission(tmp_path):
    k = _knowledge(tmp_path)
    a = Archive(tmp_path / "arch")
    good = copy.deepcopy(PROGRAM)
    bad = copy.deepcopy(PROGRAM)
    bad["program_id"] = "bad_hazard"
    bad["branch"]["trigger"]["hazard"] = "plate_1"
    mutated = copy.deepcopy(PROGRAM)
    mutated["program_id"] = "mutated"
    mutated["scene"] = {"mutations": [{"op": "shift", "object": "akita_black_bowl_1", "dxyz": [0, 0.02, 0]}]}
    reply = "Here you go:\n" + json.dumps([good, bad, mutated, good]) + "\nThanks"
    progs = parse_programs(reply, limit=3)
    assert len(progs) == 3
    recs = admit(a, CELL, progs, {"provider": "test"}, knowledge=k)
    assert [r.status for r in recs] == ["validated", "rejected", "rejected"]
    assert recs[1].gate1_reasons[0]["code"] == "SCHEMA"
    assert recs[2].gate1_reasons[0]["code"] == "MUTATION_NOT_ALLOWED"
    fenced = "```json\n" + json.dumps(good) + "\n```"
    again = admit(a, CELL, parse_programs(fenced), {}, knowledge=k)
    assert again[0].program_id == "oa_l1_t5_graze_v2"
    assert parse_programs("no json here") == []
