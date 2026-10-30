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

"""Contracts for action-conditioned safe world-model episodes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch


class _RealEnv:
    """Real simulator stub: goal from step `goal_at` on, a violation at step `viol_at` (1-based)."""

    def __init__(self, goal_at=2, viol_at=None):
        self.steps, self.goal_at, self.viol_at = 0, goal_at, viol_at

    def step(self, action):
        self.steps += 1

    def check_success(self):
        return self.goal_at is not None and self.steps >= self.goal_at


def _make_env(monkeypatch, *, reward_source, termination_source, rm_probs, real_env, max_chunks=1,
              tail_chunks=1, terminate_on_real_violation=False, end_on_violation=False,
              outcome_order="legacy", reward_rule="once"):
    import rlinf.envs.sim.libero.safety_auditor as auditor
    from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv

    monkeypatch.setattr(
        auditor, "audit_predicate_instances",
        lambda env: [{"source": "constraint", "predicate": "v",
                      "value": env.viol_at is not None and env.steps >= env.viol_at}],
    )
    T = 8 * max_chunks
    env = SyncRealWorldModelEnv.__new__(SyncRealWorldModelEnv)
    env.num_envs, env.chunk, env.max_chunks, env.max_episode_steps = 1, 8, max_chunks, T
    env.device = "cpu"
    env.reward_source, env.termination_source = reward_source, termination_source
    env.end_on_violation, env.terminate_on_real_violation = end_on_violation, terminate_on_real_violation
    env.step_order, env.reward_rule = outcome_order == "step", reward_rule
    env.tail_chunks, env.record_steps = tail_chunks, True
    env.onload = lambda: None
    env._generate = lambda actions: (np.zeros((1, 8, 512, 256, 3), np.uint8), torch.zeros(1))
    probs = iter(rm_probs)  # one 8-frame probability vector per chunk_step call
    env.reward_model = SimpleNamespace(predict_rew=lambda frames, instructions: next(probs))
    env.sam = lambda latents, actions, task_ids: torch.zeros(1, 8, 3)
    env.state_reader = lambda frame: torch.zeros(1, 1)
    env._wrap_obs = lambda: None
    env.task_descriptions, env.task_ids = ["task"], torch.zeros(1, dtype=torch.long)
    env.success_threshold, env.success_consecutive = 0.8, 4
    env.chmask = torch.ones(1, 3, dtype=torch.bool)
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    env.ended = torch.zeros(1, dtype=torch.bool)
    env.step_costs, env.cost_by_depth = torch.zeros(1, T), torch.zeros(1, max_chunks)
    env.top_m, env.eta = 8, 0.0
    env.state_offset, env.ep_cost = torch.zeros(1, 1), torch.zeros(1)
    env.success_once = torch.zeros(1, dtype=torch.bool)
    env.returns, env._finished_costs = torch.zeros(1), []
    env._real = [real_env]
    env.real_succ = np.zeros(1, dtype=bool)
    env.real_viol = np.zeros(1, dtype=bool)
    env.real_done = np.zeros(1, dtype=bool)
    env.real_succ_step, env.tail_left = np.full(1, -1), np.full(1, -1)
    env.step_rec = np.zeros((1, T, 3), dtype=bool)
    env.rm_success_once = torch.zeros(1, dtype=torch.bool)
    env.last_goal = np.zeros(1, dtype=bool)
    return env


@pytest.mark.parametrize(
    ("reward_source", "rm_probability", "expected_reward"),
    [("real", 0.0, 1.0), ("rm", 0.0, 0.0), ("rm", 0.9, 1.0)],
)
def test_sync_real_fixed_horizon_separates_reward_from_termination(
    monkeypatch, reward_source, rm_probability, expected_reward
):
    env = _make_env(monkeypatch, reward_source=reward_source, termination_source="horizon",
                    rm_probs=[torch.full((8,), rm_probability)], real_env=_RealEnv(goal_at=2))

    _, rewards, terminations, truncations, _ = env.chunk_step(torch.zeros(1, 8, 7))

    assert env._real[0].steps == 8
    assert rewards[0, -1].item() == expected_reward
    assert not terminations[0, -1]
    assert truncations[0, -1]


# RM hit completes at step 7 (frames 4-7 >= 0.8); real violation at step 3 (in-chunk index 2)
_LATE_HIT = torch.tensor([0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9])


@pytest.mark.parametrize(("outcome_order", "expected_reward"), [("legacy", 1.0), ("step", 0.0)])
def test_rm_termination_violation_before_rm_hit(monkeypatch, outcome_order, expected_reward):
    env = _make_env(monkeypatch, reward_source="rm", termination_source="rm", rm_probs=[_LATE_HIT],
                    real_env=_RealEnv(goal_at=None, viol_at=3), terminate_on_real_violation=True,
                    end_on_violation=True, outcome_order=outcome_order)

    _, rewards, terminations, _, _ = env.chunk_step(torch.zeros(1, 8, 7))

    assert terminations[0, -1]
    assert rewards[0, -1].item() == expected_reward


def test_rm_termination_hit_before_violation_still_succeeds(monkeypatch):
    early_hit = torch.tensor([0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0])  # completes at index 3
    env = _make_env(monkeypatch, reward_source="rm", termination_source="rm", rm_probs=[early_hit],
                    real_env=_RealEnv(goal_at=None, viol_at=6), terminate_on_real_violation=True,
                    end_on_violation=True, outcome_order="step")

    _, rewards, _, _, _ = env.chunk_step(torch.zeros(1, 8, 7))

    assert rewards[0, -1].item() == 1.0


@pytest.mark.parametrize(("outcome_order", "expected_reward"), [("legacy", 1.0), ("step", 0.0)])
def test_real_termination_same_step_goal_and_violation(monkeypatch, outcome_order, expected_reward):
    env = _make_env(monkeypatch, reward_source="real", termination_source="real",
                    rm_probs=[torch.zeros(8)], real_env=_RealEnv(goal_at=3, viol_at=3), tail_chunks=0,
                    terminate_on_real_violation=True, end_on_violation=True, outcome_order=outcome_order)

    _, rewards, terminations, _, _ = env.chunk_step(torch.zeros(1, 8, 7))

    assert terminations[0, -1]
    assert rewards[0, -1].item() == expected_reward


@pytest.mark.parametrize(("reward_rule", "expected_reward"), [("once", 1.0), ("final", 0.0)])
def test_rm_reward_rule_transient_false_hit(monkeypatch, reward_rule, expected_reward):
    # RM fires in chunk 1 only; no real success; fixed 16-step horizon
    env = _make_env(monkeypatch, reward_source="rm", termination_source="horizon",
                    rm_probs=[torch.full((8,), 0.9), torch.zeros(8)], real_env=_RealEnv(goal_at=None),
                    max_chunks=2, tail_chunks=2, outcome_order="step", reward_rule=reward_rule)

    _, r1, _, t1, _ = env.chunk_step(torch.zeros(1, 8, 7))
    _, r2, _, t2, _ = env.chunk_step(torch.zeros(1, 8, 7))

    assert r1.sum().item() == 0.0 and not t1[0, -1]
    assert t2[0, -1]
    assert r2[0, -1].item() == expected_reward


@pytest.mark.parametrize(("reward_rule", "expected_reward"), [("once", 1.0), ("final", 0.0)])
def test_real_reward_rule_goal_lost_before_horizon(monkeypatch, reward_rule, expected_reward):
    class LoseGoal(_RealEnv):
        def check_success(self):
            return 2 <= self.steps <= 9  # reached in chunk 1, lost in chunk 2

    env = _make_env(monkeypatch, reward_source="real", termination_source="horizon",
                    rm_probs=[torch.zeros(8), torch.zeros(8)], real_env=LoseGoal(),
                    max_chunks=2, tail_chunks=2, outcome_order="step", reward_rule=reward_rule)

    env.chunk_step(torch.zeros(1, 8, 7))
    _, r2, _, t2, _ = env.chunk_step(torch.zeros(1, 8, 7))

    assert env._real[0].steps == 16
    assert t2[0, -1]
    assert r2[0, -1].item() == expected_reward
