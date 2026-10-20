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

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_libero_cf_dataset import _auditor, _branch_meta, _episode
from test_libero_cf_program import PROGRAM, TASK
from toolkits.codeworld_safe.agent.validator import (
    gate1_schema,
    gate2_reset,
    gate3_trial,
    gate4_episode,
    load_episodes,
)


def _knowledge(tmp_path: Path) -> Path:
    (tmp_path / "knowledge" / "tasks").mkdir(parents=True)
    (tmp_path / "knowledge" / "tasks" / f"{TASK['key']}.json").write_text(json.dumps(TASK))
    return tmp_path / "knowledge"


def test_gate1_accepts_valid_and_reports_schema_errors(tmp_path):
    kdir = _knowledge(tmp_path)
    assert gate1_schema(PROGRAM, kdir).ok
    bad = copy.deepcopy(PROGRAM)
    bad["branch"]["trigger"]["hazard"] = "plate_1"
    v = gate1_schema(bad, kdir)
    assert not v.ok and v.codes == ["SCHEMA"] and "not in task hazards" in v.reasons[0]["detail"]
    other = copy.deepcopy(PROGRAM)
    other["cell"]["task_id"] = 7
    assert gate1_schema(other, kdir).codes == ["UNKNOWN_CELL"]


def test_gate2_collects_all_reset_problems():
    grades = [
        {"name": "a", "violated": True, "min_dist": -0.001, "normal_force": 3.0},
        {"name": "b", "violated": False, "min_dist": 0.0, "normal_force": 0.5},
        {"name": "c", "violated": False, "min_dist": 0.01, "normal_force": 0.0},
        {"name": "d", "violated": False, "min_dist": 0.20, "normal_force": 0.0},
    ]
    muts = [{"object": "bowl", "dxyz": [0.0, 0.09, 0.0]}, {"object": "bowl", "dyaw_deg": 40}]
    v = gate2_reset(grades, muts, min_clearance_m=0.02)
    assert v.codes == ["INITIAL_VIOLATION", "INITIAL_CONTACT", "INITIAL_CLEARANCE",
                       "MUTATION_SHIFT", "MUTATION_ROTATE"]
    assert gate2_reset(grades[3:]).ok


@pytest.mark.parametrize(
    ("quadrants", "codes"),
    [
        (["Q2", "Q1", "Q3", "Q3", "Q4"], []),
        # Every failing reason is reported, not only the first.
        (["Q3", "Q4", "Q4", "Q3", "Q3"], ["UNSOLVABLE", "LOW_HIT_RATE"]),
        (["Q1", "Q4", "Q4", "Q3", "Q3"], ["LOW_HIT_RATE"]),
        (["Q1", "Q1", "Q1", "Q3", "Q3"], ["LOW_HIT_RATE"]),
        ([], ["NO_TRIALS"]),
    ],
)
def test_gate3_solvability_and_hit_rate(quadrants, codes):
    assert gate3_trial(quadrants, "Q2").codes == codes


def _lookup(root: Path):
    lookup, failures = load_episodes(sorted(root.rglob("process-*.json")))
    assert not failures
    by_id = {ep.source_episode_id: ep for ep in lookup.values()}
    return lookup, by_id


def test_gate4_reports_known_pitfalls(tmp_path):
    root = tmp_path / "audits"
    aud = _auditor(root)
    rid = _episode(aud, 20, 0, lambda t: t)
    ok_branch = _episode(aud, 20, 4, lambda t: 100 + t, violation_at=1, **_branch_meta(rid, 5))
    out_of_range = _episode(aud, 20, 4, lambda t: 100 + t, violation_at=1, **_branch_meta(rid, 50))
    bad_seam = _episode(aud, 20, 60, lambda t: 100 + t, violation_at=1, **_branch_meta(rid, 5))
    short = _episode(aud, 5, 0, lambda t: t)
    # Short out-of-range branch: only the range error, no false TOO_SHORT.
    short_oob = _episode(aud, 4, 4, lambda t: 100 + t, violation_at=1, **_branch_meta(rid, 50))
    orphan = _episode(aud, 20, 0, lambda t: t, violation_at=1, **_branch_meta("missing", 3))
    lookup, by_id = _lookup(root)

    def codes(eid, **kw):
        return gate4_episode(by_id[eid], lookup, **kw).codes

    assert codes(rid) == []
    assert codes(ok_branch) == []
    assert codes(out_of_range) == ["BRANCH_OUT_OF_RANGE"]
    assert codes(bad_seam) == ["SEAM_PIXEL_MISMATCH"]
    assert codes(short) == ["TOO_SHORT"]
    assert codes(short_oob) == ["BRANCH_OUT_OF_RANGE"]
    assert codes(orphan) == ["NO_PARENT"]
    # New-data requirements: chunk-aligned branch points and a sim-state sidecar.
    assert codes(ok_branch, chunk=8, require_sim_state=True) == [
        "BRANCH_NOT_CHUNK_ALIGNED", "NO_SIM_STATE"]


def test_gate4_flags_legacy_over_budget_branch(tmp_path):
    root = tmp_path / "audits"
    aud = _auditor(root)
    rid = _episode(aud, 300, 0, lambda t: t % 200)
    # Old collector: branch restarted with a full 512-step budget after step 300.
    long_branch = _episode(aud, 512, 299 % 200, lambda t: 100 + t % 100, violation_at=1,
                           **_branch_meta(rid, 300))
    lookup, by_id = _lookup(root)
    v = gate4_episode(by_id[long_branch], lookup)
    assert v.codes == ["OVER_BUDGET"] and v.info["stitched_frames"] == 299 + 512 + 1
