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

    def __init__(self, goal_at=2, viol_at=None, aux_at=None):
        self.steps, self.goal_at, self.viol_at, self.aux_at = 0, goal_at, viol_at, aux_at

    def step(self, action):
        self.steps += 1

    def check_success(self):
        return self.goal_at is not None and self.steps >= self.goal_at


def _make_env(monkeypatch, *, reward_source, termination_source, rm_probs, real_env, max_chunks=1,
              tail_chunks=1, terminate_on_real_violation=False, end_on_violation=False,
              outcome_order="legacy", reward_rule="once", cost_source="sam", eta=0.0):
    import rlinf.envs.sim.libero.safety_auditor as auditor
    from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv

    monkeypatch.setattr(
        auditor, "audit_predicate_instances",
        lambda env: [{"source": "constraint", "predicate": "v",
                      "value": env.viol_at is not None and env.steps >= env.viol_at},
                     {"source": "aux_constraint", "predicate": "wronggrasppart",
                      "value": getattr(env, "aux_at", None) is not None and env.steps >= env.aux_at}],
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
    env.top_m, env.eta = 8, eta
    env.cost_source = cost_source
    env.c_off, env.c_aux = np.zeros(1, dtype=bool), np.zeros(1, dtype=bool)
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


@pytest.mark.parametrize(("views", "rows"), [(1, 256), (2, 512)])
def test_reward_model_gets_agent_or_agent_plus_wrist(monkeypatch, views, rows):
    seen = {}

    def predict(frames, instructions):
        seen["shape"] = tuple(frames.shape)
        return torch.zeros(8)

    env = _make_env(monkeypatch, reward_source="rm", termination_source="horizon",
                    rm_probs=[torch.zeros(8)], real_env=_RealEnv(goal_at=None))
    env.reward_model = SimpleNamespace(views=views, predict_rew=predict)

    env.chunk_step(torch.zeros(1, 8, 7))

    assert seen["shape"] == (8, 3, rows, 256)


# ---------------------------------------------------------------- E11: real event cost + global eta
def _real_cost_run(monkeypatch, real_env, eta=1.0, cost_source="real"):
    env = _make_env(monkeypatch, reward_source="real", termination_source="horizon",
                    rm_probs=[torch.zeros(8)], real_env=real_env, tail_chunks=1,
                    outcome_order="step", cost_source=cost_source, eta=eta)
    _, rewards, _, truncations, info = env.chunk_step(torch.zeros(1, 8, 7))
    assert truncations[0, -1]
    return env, rewards[0, -1].item(), info[0]["episode"]


@pytest.mark.parametrize(
    ("goal_at", "viol_at", "aux_at", "cost", "official", "aux"),
    [
        (5, 3, None, 1.0, 1.0, 0.0),    # official violation before the goal
        (3, 5, None, 0.0, 0.0, 0.0),    # violation after the goal: outside the cost window
        (5, None, 3, 1.0, 0.0, 1.0),    # auxiliary (wrong grasp part) before the goal
        (3, None, 5, 0.0, 0.0, 0.0),    # auxiliary after the goal
        (3, 3, None, 1.0, 1.0, 0.0),    # same step as the goal: the goal step is inside the window
        (None, 4, 6, 1.0, 1.0, 1.0),    # no goal: whole horizon is the window, both kinds recorded
        (None, None, None, 0.0, 0.0, 0.0),
    ],
)
def test_real_event_cost_window(monkeypatch, goal_at, viol_at, aux_at, cost, official, aux):
    env, _, ep = _real_cost_run(monkeypatch, _RealEnv(goal_at=goal_at, viol_at=viol_at, aux_at=aux_at))
    assert ep["cost"][0].item() == cost
    assert ep["cost_official"][0].item() == official
    assert ep["cost_aux"][0].item() == aux
    assert env._finished_costs == [cost]


def test_real_event_cost_enters_reward(monkeypatch):
    _, r, ep = _real_cost_run(monkeypatch, _RealEnv(goal_at=5, viol_at=3), eta=0.7)
    # goal reached (once rule, horizon termination) minus eta * C with C = 1
    assert r == pytest.approx(1.0 - 0.7)
    assert ep["cost_sam"][0].item() == pytest.approx(0.5)  # SAM still logged (stub: sigmoid(0) = .5)


def test_sam_cost_source_unchanged(monkeypatch):
    _, r, ep = _real_cost_run(monkeypatch, _RealEnv(goal_at=5, viol_at=3), eta=0.7, cost_source="sam")
    assert ep["cost"][0].item() == pytest.approx(0.5)  # SAM stub: zero logits -> probability .5
    assert r == pytest.approx(1.0 - 0.7 * 0.5)
    assert ep["cost_official"][0].item() == 1.0  # still logged for diagnosis


def _init_sync_env(monkeypatch, **cfg):
    """Run SyncRealWorldModelEnv.__init__ with the parent (model loading) stubbed out, up to the
    option checks; returns the env or raises the option error."""
    from rlinf.envs.sim.world_model.safe import safe_env as se
    from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv

    def parent_init(self, *a, **k):
        self.max_chunks, self.release_coef = 8, 0.0

    monkeypatch.setattr(se.SafeWorldModelEnv, "__init__", parent_init)
    base = {"reward_source": "real", "termination_source": "horizon", "end_on_violation": False,
            "success_tail_chunks": 8, "episodes_jsonl": "/nonexistent", "keyframe_dir": "/nonexistent"}
    base.update(cfg)
    try:
        SyncRealWorldModelEnv(base)
    except (FileNotFoundError, OSError):
        return "passed option checks"  # reached the episode index loading after all option checks


@pytest.mark.parametrize(("aux_env", "ok"), [(None, False), ("0", False), ("1", True)])
def test_real_cost_requires_aux_labels(monkeypatch, aux_env, ok):
    if aux_env is None:
        monkeypatch.delenv("RLINF_LIBERO_SAFETY_AUX", raising=False)
    else:
        monkeypatch.setenv("RLINF_LIBERO_SAFETY_AUX", aux_env)
    if ok:
        assert _init_sync_env(monkeypatch, cost_source="real") == "passed option checks"
    else:
        with pytest.raises(ValueError, match="RLINF_LIBERO_SAFETY_AUX"):
            _init_sync_env(monkeypatch, cost_source="real")


def test_sam_cost_source_needs_no_aux_labels(monkeypatch):
    monkeypatch.delenv("RLINF_LIBERO_SAFETY_AUX", raising=False)
    assert _init_sync_env(monkeypatch) == "passed option checks"          # default cost_source=sam
    with pytest.raises(ValueError, match="cost_source"):
        _init_sync_env(monkeypatch, cost_source="bogus")


def _safe_env_stub(global_eta):
    from rlinf.envs.sim.world_model.safe.safe_env import SafeWorldModelEnv

    env = SafeWorldModelEnv.__new__(SafeWorldModelEnv)
    env.eta, env.eta_lr, env.eta_max, env.cost_budget = 0.5, 2.0, 5.0, 0.2
    env.global_eta = global_eta
    env._round_cost_sum, env._round_cost_n = 0.0, 0
    env._finished_costs = []
    return env


def test_local_eta_update_unchanged():
    env = _safe_env_stub(global_eta=False)
    env._finished_costs = [1.0, 0.0, 1.0, 0.0]
    env.update_reset_state_ids()
    assert env.eta == pytest.approx(0.5 + 2.0 * (0.5 - 0.2))
    assert env.pop_round_costs() == (0.0, 0)


def test_global_eta_mode_only_accumulates():
    env = _safe_env_stub(global_eta=True)
    env._finished_costs = [1.0, 0.0, 1.0]
    env.update_reset_state_ids()
    env._finished_costs = [1.0]          # a second rollout epoch of the same step
    env.update_reset_state_ids()
    assert env.eta == 0.5                # untouched locally
    assert env.pop_round_costs() == (3.0, 4)
    assert env.pop_round_costs() == (0.0, 0)
    env.set_eta(1.25)
    assert env.eta == 1.25


def test_global_eta_calibration_update_and_clip(tmp_path):
    from rlinf.algorithms.global_eta import GlobalEta

    g = GlobalEta(str(tmp_path / "g.json"), eta_init=0.0, eta_lr=2.0, eta_max=5.0, budget_factor=0.5)
    assert g.update(0, 40.0, 100) == pytest.approx(0.4)   # base .4 -> budget .2; 0 + 2*(.4-.2)
    assert g.state["base_cost"] == pytest.approx(0.4) and g.state["budget"] == pytest.approx(0.2)
    assert g.update(1, 10.0, 100) == pytest.approx(0.2)   # .4 + 2*(.1-.2)
    assert g.update(2, 0.0, 100) == 0.0                    # clipped at 0
    assert g.update(3, 0.0, 0) == 0.0                      # no finished episodes: unchanged, no history
    assert len(g.state["history"]) == 3
    for s in range(4, 20):
        g.update(s, 100.0, 100)
    assert g.eta == 5.0                                    # clipped at eta_max


def test_global_eta_resume(tmp_path):
    from rlinf.algorithms.global_eta import GlobalEta

    path = str(tmp_path / "g.json")
    g = GlobalEta(path, 0.0, 2.0, 5.0, 0.5)
    for s, c in enumerate([40.0, 60.0, 50.0, 30.0]):
        g.update(s, c, 100)
    # resume from the checkpoint written after 2 updates (global_step 2): steps 2 and 3 are dropped
    r = GlobalEta(path, 0.0, 2.0, 5.0, 0.5)
    r.load(2)
    assert [h["step"] for h in r.state["history"]] == [0, 1]
    assert r.eta == pytest.approx(g.state["history"][1]["eta_next"])
    assert r.state["budget"] == pytest.approx(0.2)
    # resume before any checkpoint: start over and re-calibrate
    z = GlobalEta(path, 0.0, 2.0, 5.0, 0.5)
    z.load(0)
    assert z.state["budget"] is None and z.eta == 0.0 and z.state["history"] == []


def test_env_worker_aggregates_and_broadcasts():
    from rlinf.workers.env.env_worker import EnvWorker

    class E:
        def __init__(self, s, n):
            self.s, self.n, self.eta = s, n, 0.0

        def pop_round_costs(self):
            out, (self.s, self.n) = (self.s, self.n), (0.0, 0)
            return out

        def set_eta(self, eta):
            self.eta = eta

    w = EnvWorker.__new__(EnvWorker)
    w.env_list = [E(2.0, 5), E(1.0, 3)]
    assert w.pop_round_costs() == (3.0, 8)
    w.set_global_eta(0.75)
    assert [e.eta for e in w.env_list] == [0.75, 0.75]


def test_runner_global_eta_round_trip(tmp_path):
    from rlinf.algorithms.global_eta import GlobalEta
    from rlinf.runners.embodied_runner import EmbodiedRunner

    class H:
        def __init__(self, v):
            self.v = v

        def wait(self):
            return self.v

    class Env:
        eta = None

        def pop_round_costs(self):
            return H([(30.0, 50), (10.0, 50)])     # two env workers

        def set_global_eta(self, eta):
            Env.eta = eta
            return H([None, None])

    r = EmbodiedRunner.__new__(EmbodiedRunner)
    r.env = Env()
    r.logger = SimpleNamespace(info=lambda *a, **k: None)
    r.global_eta = GlobalEta(str(tmp_path / "g.json"), 0.0, 2.0, 5.0, 0.5)
    r._update_global_eta(0)
    assert r.global_eta.state["base_cost"] == pytest.approx(0.4)
    assert Env.eta == pytest.approx(0.4)
