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

import numpy as np
import pytest

from rlinf.envs.sim.libero.counterfactual_branch_collector import (
    CounterfactualBranchCollector,
    hazards_from_predicates,
)
from rlinf.envs.sim.libero.counterfactual_strategy import BranchStrategy


def _collector(**overrides):
    kwargs = {
        "num_envs": 1,
        "seed": 7,
        "process_id": 2,
        "risk_window_steps": 2,
        "max_branches_per_root": 6,
        "perturbation_range": (0.25, 0.75),
        "action_dimensions": (0, 1, 2, 3, 4, 5),
    }
    kwargs.update(overrides)
    return CounterfactualBranchCollector(**kwargs)


def test_branch_search_samples_q1_tail_and_preserves_gripper():
    collector = _collector()
    for timestep in range(1, 4):
        collector.record_root_step(
            0,
            timestep=timestep,
            simulator_state={"qpos": np.array([timestep], dtype=np.float64)},
            action=np.array([0.1] * 6 + [-1.0]),
        )

    branch = collector.begin_search(0, root_episode_id="root-007")

    assert branch is not None
    assert branch.root_episode_id == "root-007"
    assert branch.branch_id == "root-007-branch-000"
    assert branch.root_timestep in {2, 3}
    assert 0.25 <= branch.alpha <= 0.75
    assert branch.perturbation_range == (0.25, 0.75)
    assert branch.perturbed_action[-1] == -1.0
    assert np.all(branch.perturbed_action >= -1.0)
    assert np.all(branch.perturbed_action <= 1.0)
    np.testing.assert_allclose(
        collector.consume_perturbed_action(0), branch.perturbed_action
    )
    assert collector.consume_perturbed_action(0) is None
    assert branch.metadata()["root_episode_id"] == "root-007"


def test_branch_search_retries_until_q2_and_q4_are_found():
    collector = _collector()
    collector.record_root_step(
        0,
        timestep=5,
        simulator_state={"qpos": np.array([1.0])},
        action=np.zeros(7),
    )
    first = collector.begin_search(0, root_episode_id="root")
    assert first is not None

    second = collector.complete_branch(0, quadrant="Q1")
    assert second is not None
    assert second.branch_id.endswith("branch-001")

    third = collector.complete_branch(0, quadrant="Q2")
    assert third is not None
    assert third.branch_id.endswith("branch-002")

    assert collector.complete_branch(0, quadrant="Q4") is None
    assert not collector.is_running_branch(0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"risk_window_steps": 0}, "risk_window_steps"),
        ({"max_branches_per_root": 0}, "max_branches_per_root"),
        ({"perturbation_range": (-0.1, 0.5)}, "perturbation_range"),
        ({"target_quadrants": frozenset({"Q1"})}, "target_quadrants"),
        ({"root_quadrants": frozenset({"Q2"})}, "root_quadrants"),
        ({"root_quadrants": frozenset()}, "root_quadrants"),
        (
            {
                "root_quadrants": frozenset({"Q1", "Q3"}),
                "target_quadrants": frozenset({"Q2"}),
            },
            "Q3 roots require Q4",
        ),
    ],
)
def test_branch_collector_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        _collector(**overrides)


def _record_one_step(collector, env_idx=0):
    collector.record_root_step(
        env_idx,
        timestep=5,
        simulator_state={"qpos": np.array([1.0])},
        action=np.zeros(7),
    )


def test_default_collector_accepts_only_q1_roots():
    collector = _collector()
    assert collector.accepts_root("Q1")
    assert not collector.accepts_root("Q3")
    _record_one_step(collector)
    with pytest.raises(ValueError, match="root quadrant"):
        collector.begin_search(0, root_episode_id="root", root_quadrant="Q3")


def test_q3_root_searches_only_for_q4():
    collector = _collector(root_quadrants=frozenset({"Q1", "Q3"}))
    assert collector.accepts_root("Q3")
    _record_one_step(collector)

    first = collector.begin_search(0, root_episode_id="root", root_quadrant="Q3")
    assert first is not None
    assert first.root_quadrant == "Q3"
    assert first.metadata()["root_quadrant"] == "Q3"

    # Q2 is unreachable from a failed root; a Q3 search must stop at Q4 alone.
    second = collector.complete_branch(0, quadrant="Q3")
    assert second is not None
    assert collector.complete_branch(0, quadrant="Q4") is None
    assert not collector.is_running_branch(0)


def test_q3_root_search_stops_at_max_branches():
    collector = _collector(
        root_quadrants=frozenset({"Q3"}), max_branches_per_root=2
    )
    _record_one_step(collector)
    assert collector.begin_search(0, root_episode_id="r", root_quadrant="Q3")
    assert collector.complete_branch(0, quadrant="Q3") is not None
    assert collector.complete_branch(0, quadrant="Q3") is None


def test_reset_root_restores_q1_search_targets():
    collector = _collector(root_quadrants=frozenset({"Q1", "Q3"}))
    _record_one_step(collector)
    collector.begin_search(0, root_episode_id="r3", root_quadrant="Q3")
    collector.reset_root(0)
    collector.set_root_episode_id(0, "r1")
    _record_one_step(collector)

    branch = collector.begin_search(0)
    assert branch.root_quadrant == "Q1"
    # Q1 roots still need both Q2 and Q4 before stopping early.
    assert collector.complete_branch(0, quadrant="Q4") is not None
    assert collector.complete_branch(0, quadrant="Q2") is None


def _legacy_reference(seed, process_id, history_len, action, perturbation_range):
    """Re-derive the pre-strategy collector's first branch from its RNG."""
    seed_sequence = np.random.SeedSequence([seed, process_id])
    rng = np.random.default_rng(seed_sequence.spawn(1)[0])
    index = int(rng.integers(0, history_len))
    alpha = float(rng.uniform(*perturbation_range))
    random_action = rng.uniform(-1.0, 1.0, size=action.shape)
    expected = action.copy()
    expected[:6] = (1.0 - alpha) * action[:6] + alpha * random_action[:6]
    return index, alpha, np.clip(expected, -1.0, 1.0)


def test_legacy_strategy_keeps_rng_sequence_and_single_step():
    collector = _collector()
    action = np.array([0.1] * 6 + [-1.0])
    for timestep in range(1, 4):
        collector.record_root_step(
            0,
            timestep=timestep,
            simulator_state={"t": timestep},
            action=action,
            hazards={"constraint:000|checkrobotcontact|bowl": np.zeros(3)},
            eef_pos=np.ones(3),
        )
    branch = collector.begin_search(0, root_episode_id="root")
    index, alpha, expected = _legacy_reference(7, 2, 2, action, (0.25, 0.75))
    assert branch.root_timestep == (2, 3)[index]
    assert branch.alpha == pytest.approx(alpha)
    np.testing.assert_allclose(branch.perturbed_action, expected)
    assert branch.perturb_steps == 1 and branch.directed is False
    collector.consume_perturbed_action(0, policy_action=action)
    assert collector.consume_perturbed_action(0, policy_action=action) is None


def test_directed_multi_step_strategy_pushes_toward_live_hazard():
    strategy = BranchStrategy(
        name="test_directed",
        branch_point_mode="risk_weighted",
        history_steps=8,
        perturb_steps=3,
        perturbation_range=(1.0, 1.0),
        directed_prob=1.0,
        directed_gain=1.0,
        gripper_mode="close",
        hazard_filters=("hand",),
    )
    collector = _collector()
    collector.set_strategy(0, strategy)
    hand_key = "constraint:000|checkrobotcontact|left_hand_1"
    far_key = "constraint:001|checkrobotcontact|bowl_1"
    for timestep in range(1, 9):
        collector.record_root_step(
            0,
            timestep=timestep,
            simulator_state={"t": timestep},
            action=np.array([0.0] * 6 + [-1.0]),
            hazards={hand_key: np.array([1.0, 0.0, 0.0]), far_key: np.zeros(3)},
            eef_pos=np.zeros(3),
        )
    branch = collector.begin_search(0, root_episode_id="root")
    assert branch.hazard_key == hand_key  # filtered: bowl is ignored
    assert branch.directed is True and branch.perturb_steps == 3
    np.testing.assert_allclose(branch.perturbed_action[:3], [1.0, 0.0, 0.0])
    assert branch.perturbed_action[-1] == 1.0  # gripper forced closed
    meta = branch.metadata()
    assert meta["branch_strategy"] == "test_directed"
    assert meta["perturb_steps"] == 3 and meta["directed"] is True

    policy = np.array([0.0] * 6 + [-1.0])
    first = collector.consume_perturbed_action(0, policy_action=policy)
    np.testing.assert_allclose(first, branch.perturbed_action)
    # The hand moved to +y: the second push re-aims at the live position.
    second = collector.consume_perturbed_action(
        0,
        policy_action=policy,
        eef_pos=np.zeros(3),
        hazards={hand_key: np.array([0.0, 2.0, 0.0])},
    )
    np.testing.assert_allclose(second[:3], [0.0, 1.0, 0.0])
    third = collector.consume_perturbed_action(
        0, policy_action=policy, eef_pos=np.zeros(3)
    )
    assert third is not None
    assert collector.consume_perturbed_action(0, policy_action=policy) is None


def test_risk_weighted_branch_points_prefer_near_hazard_steps():
    strategy = BranchStrategy(
        branch_point_mode="risk_weighted",
        history_steps=20,
        risk_distance_scale=0.01,
        risk_weight_floor=0.0,
    )
    collector = _collector(max_branches_per_root=50)
    collector.set_strategy(0, strategy)
    for timestep in range(1, 21):
        # Only timestep 7 is close to the hazard.
        eef = np.zeros(3) if timestep == 7 else np.array([1.0, 1.0, 1.0])
        collector.record_root_step(
            0,
            timestep=timestep,
            simulator_state={"t": timestep},
            action=np.zeros(7),
            hazards=[("constraint:000|checkcontact|bottle_1", np.zeros(3))],
            eef_pos=eef,
        )
    branch = collector.begin_search(0, root_episode_id="root")
    timesteps = {branch.root_timestep}
    for _ in range(10):
        branch = collector.complete_branch(0, quadrant="Q1")
        timesteps.add(branch.root_timestep)
    assert timesteps == {7}


def test_set_strategy_requires_fresh_root():
    collector = _collector()
    collector.record_root_step(
        0, timestep=1, simulator_state={}, action=np.zeros(7)
    )
    with pytest.raises(RuntimeError, match="set_strategy"):
        collector.set_strategy(0, BranchStrategy())
    collector.reset_root(0)
    collector.set_strategy(0, BranchStrategy(history_steps=5))
    assert collector._histories[0].maxlen == 5


def test_hazards_from_predicates_keeps_violation_positions_only():
    predicates = [
        {
            "instance_id": "constraint:000",
            "predicate": "checkrobotcontact",
            "semantics": "violation",
            "hazard_object": "left_hand_1",
            "hazard_pos": [0.1, 0.2, 0.3],
        },
        {"instance_id": "goal:000", "predicate": "on", "semantics": "goal"},
        {
            "instance_id": "constraint:001",
            "predicate": "checkcontact",
            "semantics": "violation",
        },
    ]
    hazards = hazards_from_predicates(predicates)
    assert [key for key, _ in hazards] == [
        "constraint:000|checkrobotcontact|left_hand_1"
    ]
    np.testing.assert_allclose(hazards[0][1], [0.1, 0.2, 0.3])
