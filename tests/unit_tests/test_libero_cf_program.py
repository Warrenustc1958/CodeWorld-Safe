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

import copy

import numpy as np
import pytest

from rlinf.envs.sim.libero.cf_executor import MIN_BRANCH_STEPS, ProgramExecutor
from rlinf.envs.sim.libero.cf_program import ProgramError, load_program, sample_params
from rlinf.envs.sim.libero.counterfactual_branch_collector import (
    CounterfactualBranchCollector,
    RiskSnapshot,
)

HAZARD_KEY = "constraint:000|checkrobotcontact|toy_car_1"
HAZARD_POS = np.array([0.0, 0.0, 1.0])

TASK = {
    "key": "obstacle_avoidance_L1_t5",
    "suite": "obstacle_avoidance",
    "level": "L1",
    "task_id": 5,
    "hazards": [{"object": "toy_car_1"}],
    "objects": [{"name": "toy_car_1"}, {"name": "akita_black_bowl_1"}],
    "dynamic_objects": [{"name": "toy_car_1"}],
}

PROGRAM = {
    "program_id": "oa_l1_t5_graze",
    "cell": {"suite": "obstacle_avoidance", "level": "L1", "task_id": 5},
    "target_quadrant": "Q2",
    "branch": {
        "root_quadrants": ["Q1"],
        "trigger": {
            "phase": "any",
            "cond": "min_dist(ee,hazard) < 0.12",
            "hazard": "toy_car_1",
            "fallback": "skip",
        },
        "events": [
            {"op": "servo_toward", "target": "hazard", "steps": "$steps", "gain": "$gain"}
        ],
    },
    "params": {
        "steps": {"dist": "uniform", "low": 2, "high": 4},
        "gain": {"dist": "uniform", "low": 0.6, "high": 0.9},
    },
}


def _program(**edits):
    raw = copy.deepcopy(PROGRAM)
    for path, value in edits.items():
        node = raw
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[int(key)] if isinstance(node, list) else node[key]
        node[keys[-1]] = value
    return raw


def _executor(seed=0, **kwargs):
    program = load_program(PROGRAM, TASK)
    rng = np.random.default_rng(seed)
    return ProgramExecutor(program, sample_params(program, rng), rng, **kwargs)


def _eef(t):
    # Far from the hazard until step 17, then within 0.1 m.
    return HAZARD_POS + np.array([0.5 if t < 17 else 0.1, 0.0, 0.0])


def _snapshot(t):
    return RiskSnapshot(
        timestep=t,
        simulator_state={"t": t},
        action=np.zeros(7),
        hazards=((HAZARD_KEY, HAZARD_POS),),
        eef_pos=_eef(t),
    )


# ------------------------------------------------------------------ program


def test_load_program_validates_and_samples_params():
    program = load_program(PROGRAM, TASK)
    assert program.cell_key == "obstacle_avoidance_L1_t5"
    assert program.trigger.metric == "min_dist(ee,hazard)"
    assert program.trigger.holds(0.1) and not program.trigger.holds(0.2)
    params = sample_params(program, np.random.default_rng(3))
    assert isinstance(params["steps"], int) and 2 <= params["steps"] <= 4
    assert 0.6 <= params["gain"] <= 0.9
    assert sample_params(program, np.random.default_rng(3)) == params


@pytest.mark.parametrize(
    ("edits", "message"),
    [
        ({"branch.trigger.hazard": "plate_1"}, "not in task hazards"),
        ({"branch.trigger.phase": "carry"}, "not implemented in v0"),
        ({"branch.trigger.cond": "min_dist(held,hazard) < 0.1"}, "trigger metric"),
        ({"branch.events.0.steps": "$missing"}, "unknown param"),
        ({"branch.events.0.op": "teleport"}, "op must be one of"),
        ({"branch.root_quadrants": ["Q2"]}, "root_quadrants"),
        ({"cell.task_id": 6}, "does not match cell"),
        ({"program_id": "Bad-ID"}, "program_id"),
        (
            {"scene": {"mutations": [{"op": "shift", "object": "toy_car_1", "dxyz": [0, 0.01, 0]}]}},
            "mocap-driven",
        ),
        (
            {"scene": {"mutations": [{"op": "shift", "object": "akita_black_bowl_1", "dxyz": [0, 0.09, 0]}]}},
            "exceeds 0.08",
        ),
    ],
)
def test_load_program_rejects_out_of_contract_programs(edits, message):
    with pytest.raises(ProgramError, match=message):
        load_program(_program(**edits), TASK)


def test_scene_mutation_within_bounds_marks_program_mutated():
    raw = _program(
        scene={"mutations": [{"op": "shift", "object": "akita_black_bowl_1", "dxyz": [0.0, 0.05, 0.0]}]}
    )
    assert load_program(raw, TASK).scene_mutated


# ----------------------------------------------------------------- executor


def test_executor_budget_and_chunk_boundaries():
    ex = _executor()
    assert ex.remaining_budget(1) == 512
    assert ex.remaining_budget(17) == 496
    assert ex.is_chunk_boundary(1) and ex.is_chunk_boundary(17)
    assert not ex.is_chunk_boundary(16)
    # Not a chunk boundary, beyond the saved root, or too little budget left.
    assert not ex.eligible(_snapshot(16), saved_steps=100)
    assert not ex.eligible(_snapshot(25), saved_steps=23)
    assert ex.eligible(_snapshot(25), saved_steps=24)
    late = 512 - MIN_BRANCH_STEPS + 2 + 7  # first chunk boundary with < 16 steps left
    late -= (late - 1) % 8
    assert ex.remaining_budget(late) < MIN_BRANCH_STEPS
    assert not ex.eligible(_snapshot(late), saved_steps=600)
    with pytest.raises(ValueError, match="multiple of the action chunk"):
        _executor(chunk=8, max_total_steps=510)


def test_executor_picks_triggered_point_and_servos_toward_hazard():
    ex = _executor(seed=1)
    history = [_snapshot(t) for t in range(1, 41)]
    point = ex.choose_branch_point(history, saved_steps=40)
    assert point.timestep in (17, 25, 33)
    ex.start(point)
    steps = ex.total_steps
    assert 2 <= steps <= 4
    actions = []
    for _ in range(steps):
        actions.append(ex.action(np.zeros(7), _eef(point.timestep), ((HAZARD_KEY, HAZARD_POS),)))
    assert ex.action(np.zeros(7), _eef(point.timestep), ()) is None
    # Hazard is at -x from the end effector: translation must point to -x.
    for action in actions:
        assert action[0] < -0.5 and abs(action[1]) < 1e-9 and abs(action[2]) < 1e-9
        assert action[6] == 0.0  # gripper follows the policy
    meta = ex.branch_metadata()
    assert meta["program_id"] == "oa_l1_t5_graze" and meta["max_total_steps"] == 512
    assert meta["program_branch"]["trigger_hit"] is True


def test_executor_skip_fallback_returns_none_without_trigger():
    ex = _executor()
    far = [_snapshot(t) for t in range(1, 17)]
    assert ex.choose_branch_point(far, saved_steps=16) is None


# ---------------------------------------------------------------- collector


def _program_collector():
    collector = CounterfactualBranchCollector(num_envs=1, seed=5, process_id=0)
    collector.reset_root(0)
    collector.set_program(0, _executor(seed=2))
    collector.set_root_episode_id(0, "root-0")
    return collector


def _record(collector, t, saved_steps):
    collector.record_root_step(
        0,
        timestep=t,
        simulator_state={"t": t},
        action=np.zeros(7),
        hazards=((HAZARD_KEY, HAZARD_POS),),
        eef_pos=_eef(t),
        saved_steps=saved_steps,
    )


def test_collector_program_branch_keeps_chunk_snapshots_and_metadata():
    collector = _program_collector()
    for t in range(1, 41):
        _record(collector, t, saved_steps=t - 1)
    assert [s.timestep for s in collector._histories[0]] == [1, 9, 17, 25, 33]
    branch = collector.begin_search(0, root_quadrant="Q1", root_saved_steps=40)
    assert branch.root_timestep in (17, 25, 33)
    assert branch.simulator_state == {"t": branch.root_timestep}
    assert branch.perturbed_action[0] < -0.5
    meta = branch.metadata()
    assert meta["program_id"] == "oa_l1_t5_graze"
    assert meta["parent_episode_id"] == "root-0"
    assert meta["branch_timestep"] == branch.root_timestep
    assert meta["max_total_steps"] == 512
    # First action is pending; the remaining event steps follow the live policy.
    first = collector.consume_perturbed_action(0, policy_action=np.zeros(7))
    np.testing.assert_allclose(first, branch.perturbed_action)
    remaining = 0
    while collector.consume_perturbed_action(
        0, policy_action=np.zeros(7), eef_pos=_eef(30), hazards=((HAZARD_KEY, HAZARD_POS),)
    ) is not None:
        remaining += 1
    # perturb_steps counts every edited action, including the pending first one.
    assert remaining == branch.perturb_steps - 1


def test_collector_skips_snapshots_outside_saved_root_p2():
    collector = _program_collector()
    for t in range(1, 41):
        # The auditor stopped saving after 10 transitions.
        _record(collector, t, saved_steps=min(t - 1, 10))
    assert [s.timestep for s in collector._histories[0]] == [1, 9]
    # Only far snapshots survive and the program's fallback is "skip".
    assert collector.begin_search(0, root_quadrant="Q1", root_saved_steps=10) is None
    assert not collector.is_running_branch(0)


def test_begin_search_drops_snapshots_beyond_saved_root():
    collector = _program_collector()
    for t in range(1, 41):
        _record(collector, t, saved_steps=t - 1)
    # The root audit only kept 20 transitions: branching at 25 or 33 would be unstitchable.
    branch = collector.begin_search(0, root_quadrant="Q1", root_saved_steps=20)
    assert branch.root_timestep == 17
    assert all(s.timestep - 1 <= 20 for s in collector._histories[0])


def test_set_program_requires_fresh_root():
    collector = CounterfactualBranchCollector(num_envs=1, seed=5, process_id=0)
    collector.reset_root(0)
    collector.record_root_step(0, timestep=1, simulator_state={}, action=np.zeros(7))
    with pytest.raises(RuntimeError, match="right after reset_root"):
        collector.set_program(0, _executor())
