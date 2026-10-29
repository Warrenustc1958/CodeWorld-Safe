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


@pytest.mark.parametrize(
    ("reward_source", "rm_probability", "expected_reward"),
    [("real", 0.0, 1.0), ("rm", 0.0, 0.0), ("rm", 0.9, 1.0)],
)
def test_sync_real_fixed_horizon_separates_reward_from_termination(
    monkeypatch, reward_source, rm_probability, expected_reward
):
    import rlinf.envs.sim.libero.safety_auditor as auditor
    from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv

    class RealEnv:
        def __init__(self):
            self.steps = 0

        def step(self, action):
            self.steps += 1

        def check_success(self):
            return self.steps >= 2

    monkeypatch.setattr(auditor, "audit_predicate_instances", lambda env: [])
    env = SyncRealWorldModelEnv.__new__(SyncRealWorldModelEnv)
    env.num_envs, env.chunk, env.max_chunks, env.max_episode_steps = 1, 8, 1, 8
    env.device = "cpu"
    env.reward_source, env.termination_source = reward_source, "horizon"
    env.end_on_violation = env.terminate_on_real_violation = False
    env.tail_chunks, env.record_steps = 1, True
    env.onload = lambda: None
    env._generate = lambda actions: (
        np.zeros((1, 8, 512, 256, 3), np.uint8), torch.zeros(1)
    )
    env.reward_model = SimpleNamespace(
        predict_rew=lambda frames, instructions: torch.full((8,), rm_probability)
    )
    env.sam = lambda latents, actions, task_ids: torch.zeros(1, 8, 3)
    env.state_reader = lambda frame: torch.zeros(1, 1)
    env._wrap_obs = lambda: None
    env.task_descriptions, env.task_ids = ["task"], torch.zeros(1, dtype=torch.long)
    env.success_threshold, env.success_consecutive = 0.8, 4
    env.chmask = torch.ones(1, 3, dtype=torch.bool)
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    env.ended = torch.zeros(1, dtype=torch.bool)
    env.step_costs, env.cost_by_depth = torch.zeros(1, 8), torch.zeros(1, 1)
    env.top_m, env.eta = 8, 0.0
    env.state_offset, env.ep_cost = torch.zeros(1, 1), torch.zeros(1)
    env.success_once = torch.zeros(1, dtype=torch.bool)
    env.returns, env._finished_costs = torch.zeros(1), []
    env._real = [RealEnv()]
    env.real_succ = np.zeros(1, dtype=bool)
    env.real_viol = np.zeros(1, dtype=bool)
    env.real_done = np.zeros(1, dtype=bool)
    env.real_succ_step, env.tail_left = np.full(1, -1), np.full(1, -1)
    env.step_rec = np.zeros((1, 8, 3), dtype=bool)
    env.rm_success_once = torch.zeros(1, dtype=torch.bool)

    _, rewards, terminations, truncations, _ = env.chunk_step(torch.zeros(1, 8, 7))

    assert env._real[0].steps == 8
    assert rewards[0, -1].item() == expected_reward
    assert not terminations[0, -1]
    assert truncations[0, -1]

