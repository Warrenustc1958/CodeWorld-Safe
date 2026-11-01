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

    def __init__(self, goal_at=2, viol_at=None, aux_at=None, ch_at=(None, None, None)):
        self.steps, self.goal_at, self.viol_at, self.aux_at = 0, goal_at, viol_at, aux_at
        self.ch_at = ch_at  # grader channels (c_obj, c_robot, c_grasp): first violating step

    def step(self, action):
        self.steps += 1

    def check_success(self):
        return self.goal_at is not None and self.steps >= self.goal_at


def _make_env(monkeypatch, *, reward_source, termination_source, rm_probs, real_env, max_chunks=1,
              tail_chunks=1, terminate_on_real_violation=False, end_on_violation=False,
              outcome_order="legacy", reward_rule="once", cost_source="sam", eta=0.0,
              cost_label="predicate", cost_window="goal", obs_source="wan"):
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
    env.cost_source, env.cost_label, env.cost_window = cost_source, cost_label, cost_window
    env.obs_source = obs_source
    env.noise_fp = env.noise_fn = 0.0
    env.dump_dir, env._reset_count, env.group_size = None, 1, 1
    env.dump_arrays = False
    env.rm_frames = None
    env.dumped = np.zeros(1, dtype=bool)
    env.first_viol = np.full((1, 3), -1)
    env.rm_chunk_max = np.full((1, max_chunks), np.nan, np.float32)
    env.kf, env._kf_uid, env._records = np.zeros(1, dtype=np.int64), ["u0"], {"u0": {"suite": "affordance"}}
    env.c_off, env.c_aux = np.zeros(1, dtype=bool), np.zeros(1, dtype=bool)
    env.c_ch, env.c_ch_ep, env.c_ch_reset = (np.zeros((1, 3), dtype=bool) for _ in range(3))
    # grader stub: channel j violated from step ch_at[j] on (1-based), like the real contact grader
    env._grader_channels = lambda i: np.array(
        [a is not None and env._real[i].steps >= a for a in getattr(env._real[i], "ch_at", (None,) * 3)])
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
def _real_cost_run(monkeypatch, real_env, eta=1.0, cost_source="real", cost_label="predicate",
                   cost_window="goal"):
    env = _make_env(monkeypatch, reward_source="real", termination_source="horizon",
                    rm_probs=[torch.zeros(8)], real_env=real_env, tail_chunks=1,
                    outcome_order="step", cost_source=cost_source, eta=eta,
                    cost_label=cost_label, cost_window=cost_window)
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


@pytest.mark.parametrize(
    ("ch_at", "goal_at", "window", "cost", "per_channel", "ep_channel"),
    [
        ((None, 3, None), 5, "goal", 1.0, (0, 1, 0), (0, 1, 0)),     # robot contact the predicate misses
        ((None, 6, None), 3, "goal", 0.0, (0, 0, 0), (0, 1, 0)),     # robot contact after the goal
        ((None, 6, None), 3, "episode", 1.0, (0, 1, 0), (0, 1, 0)),  # ... counted with the episode window
        ((2, None, 4), None, "goal", 1.0, (1, 0, 1), (1, 0, 1)),     # object + wrong grasp, no goal
        ((None, None, None), 5, "goal", 0.0, (0, 0, 0), (0, 0, 0)),
    ],
)
def test_grader_cost_label(monkeypatch, ch_at, goal_at, window, cost, per_channel, ep_channel):
    # predicates report nothing (viol_at / aux_at None): only the grader sees the contact
    env, _, ep = _real_cost_run(monkeypatch, _RealEnv(goal_at=goal_at, ch_at=ch_at),
                                cost_label="grader", cost_window=window)
    assert ep["cost"][0].item() == cost
    assert ep["cost_official"][0].item() == 0.0 and ep["cost_aux"][0].item() == 0.0
    assert tuple(int(ep[f"cost_{c}"][0].item()) for c in ("obj", "robot", "grasp")) == per_channel
    assert tuple(int(ep[f"unsafe_ep_{c}"][0].item()) for c in ("obj", "robot", "grasp")) == ep_channel
    assert env._finished_costs == [cost]


def test_grader_label_does_not_change_termination(monkeypatch):
    # a grader-only robot contact must not end the episode: termination still follows official predicates
    env = _make_env(monkeypatch, reward_source="real", termination_source="real",
                    rm_probs=[torch.zeros(8)], real_env=_RealEnv(goal_at=None, ch_at=(None, 2, None)),
                    terminate_on_real_violation=True, end_on_violation=True, outcome_order="step",
                    cost_source="real", cost_label="grader", eta=1.0)
    _, rewards, terminations, truncations, info = env.chunk_step(torch.zeros(1, 8, 7))
    assert not terminations[0, -1] and truncations[0, -1]
    assert env._real[0].steps == 8  # all 8 steps executed
    assert info[0]["episode"]["cost"][0].item() == 1.0
    assert rewards[0, -1].item() == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("goal_at", "fp", "fn", "u", "paid"),
    [
        (None, 1.0, 0.0, 0.5, 1.0),   # real failure paid as success (false positive)
        (None, 0.3, 0.0, 0.5, 0.0),   # draw above fp: no flip
        (3, 0.0, 1.0, 0.5, 0.0),      # real success paid as failure (false negative)
        (3, 0.0, 0.3, 0.5, 1.0),      # draw above fn: no flip
        (3, 0.0, 0.0, 0.0, 1.0),      # no noise: unchanged
    ],
)
def test_reward_noise_flips_paid_outcome(monkeypatch, goal_at, fp, fn, u, paid):
    env = _make_env(monkeypatch, reward_source="real", termination_source="horizon",
                    rm_probs=[torch.zeros(8)], real_env=_RealEnv(goal_at=goal_at), tail_chunks=1,
                    outcome_order="step")
    env.noise_fp, env.noise_fn, env.noise_u = fp, fn, np.array([u])
    _, rewards, _, truncations, info = env.chunk_step(torch.zeros(1, 8, 7))
    assert truncations[0, -1]
    assert rewards[0, -1].item() == paid
    assert info[0]["episode"]["success_once"][0].item() == paid
    # the real outcome is still logged unflipped
    assert info[0]["episode"]["real_success"][0].item() == float(goal_at is not None)


def test_episode_dump(monkeypatch, tmp_path):
    import json

    env2 = _make_env(monkeypatch, reward_source="real", termination_source="horizon",
                     rm_probs=[torch.full((8,), 0.9)], real_env=_RealEnv(goal_at=6, ch_at=(None, 3, None)),
                     tail_chunks=1, outcome_order="step", cost_source="real", cost_label="grader", eta=0.5)
    env2.dump_dir = str(tmp_path)
    env2.chunk_step(torch.zeros(1, 8, 7))
    files = list(tmp_path.glob("episodes_*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["real_success"] and r["real_success_step"] == 5 and r["suite"] == "affordance"
    assert r["grader_window"] == [0, 1, 0] and r["first_viol_step"] == [-1, 2, -1]
    assert r["rm_chunk_max"][0] == pytest.approx(0.9) and r["rm_success_once"]
    assert r["cost_used"] == 1.0 and r["reward"] == pytest.approx(0.5) and r["paid_success"]
    assert len(r["sam_step_cost"]) == 8
    env2.chunk_step  # episode already dumped: a second call must not duplicate it
    assert env2.dumped[0]


def test_episode_array_dump(monkeypatch, tmp_path):
    import io
    import json

    from PIL import Image

    env = _make_env(monkeypatch, reward_source="real", termination_source="horizon",
                    rm_probs=[torch.zeros(8)], real_env=_RealEnv(goal_at=6, ch_at=(None, 3, None)),
                    tail_chunks=1, outcome_order="step", cost_source="real", cost_label="grader")
    env.dump_dir, env.dump_arrays = str(tmp_path), True
    env.arr_jpg, env.arr_z, env.arr_a = [[]], [[]], [[]]
    env.step_goal, env.step_exec = np.zeros((1, 8), bool), np.zeros((1, 8), bool)
    env.step_ch = np.zeros((1, 8, 3), bool)
    ramp = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (512, 256, 3))
    frames = np.stack([ramp] * 8)[None].copy()  # smooth image: JPEG round trip stays close
    env._generate = lambda actions: (frames, torch.ones(1, 48, 4, 32, 16))
    env.chunk_step(torch.full((1, 8, 7), 0.5))
    row = json.loads(next(tmp_path.glob("episodes_*.jsonl")).read_text())
    d = np.load(tmp_path / "arrays" / row["arrays"])
    off = d["jpg_off"]
    assert len(off) == 3  # mid and last frame of the single chunk
    img = np.asarray(Image.open(io.BytesIO(d["jpg"][off[0]:off[1]].tobytes())))
    assert img.shape == (512, 256, 3) and np.abs(img.astype(int) - ramp.astype(int)).mean() < 3
    assert d["z"].shape == (1, 48, 4, 32, 16) and d["z"].dtype == np.float16
    assert d["actions"].shape == (1, 8, 7) and np.allclose(d["actions"], 0.5)
    assert d["step_exec"].all()
    assert d["step_goal"].tolist() == [False] * 5 + [True] * 3
    assert d["step_ch"][:, 1].tolist() == [False] * 2 + [True] * 6 and not d["step_ch"][:, [0, 2]].any()


def test_rm_frames_scores_only_selected_frames(monkeypatch):
    seen = {}

    def predict(frames, instructions):
        seen["n"], seen["inst"] = frames.shape[0], list(instructions)
        return torch.tensor([[0.2], [0.95]])  # frame 3 below, frame 7 over the threshold

    env = _make_env(monkeypatch, reward_source="rm", termination_source="horizon",
                    rm_probs=[torch.zeros(8)], real_env=_RealEnv(goal_at=None), tail_chunks=1, outcome_order="step")
    env.reward_model = SimpleNamespace(predict_rew=predict, views=2)
    env.rm_frames, env.success_consecutive, env.success_threshold = [3, 7], 1, 0.9
    _, rewards, _, _, info = env.chunk_step(torch.zeros(1, 8, 7))
    assert seen["n"] == 2 and seen["inst"] == ["task", "task"]
    assert env.last_p_succ[0].tolist() == pytest.approx([0, 0, 0, 0.2, 0, 0, 0, 0.95])
    assert info[0]["episode"]["rm_success_once"][0].item() == 1.0
    assert rewards[0, -1].item() == 1.0  # RM reward paid (real goal never reached)


@pytest.mark.parametrize(("rf", "consecutive", "match"), [([8], 1, "rm_frames"), ([3, 7], 4, "consecutive")])
def test_rm_frames_validated(monkeypatch, rf, consecutive, match):
    from rlinf.envs.sim.world_model.safe import safe_env as se
    from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv

    def parent_init(self, *a, **k):
        self.max_chunks, self.release_coef, self.chunk, self.success_consecutive = 8, 0.0, 8, consecutive

    monkeypatch.setattr(se.SafeWorldModelEnv, "__init__", parent_init)
    with pytest.raises(ValueError, match=match):
        SyncRealWorldModelEnv({"reward_source": "real", "termination_source": "horizon", "end_on_violation": False,
                               "success_tail_chunks": 8, "rm_frames": rf, "episodes_jsonl": "/nonexistent",
                               "keyframe_dir": "/nonexistent"})


def test_qwen_rm_client_round_trip(monkeypatch):
    import base64
    import io
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from PIL import Image

    from rlinf.models.embodiment.reward.qwen_rm_client import QwenRMClient

    got = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            imgs = [np.asarray(Image.open(io.BytesIO(base64.b64decode(x)))) for x in req["jpg"]]
            got["shape"], got["mean"], got["inst"] = imgs[0].shape, float(imgs[0].mean()), req["instructions"]
            body = json.dumps({"p": [0.1 * (i + 1) for i in range(len(imgs))]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    client = QwenRMClient(port_base=srv.server_address[1]).eval()
    obs = torch.zeros(2, 3, 512, 256)  # mid-grey in [-1, 1]
    p = client.predict_rew(obs, ["t1", "t2"])
    srv.shutdown()
    assert p.shape == (2, 1) and p[:, 0].tolist() == pytest.approx([0.1, 0.2])
    assert got["shape"] == (512, 256, 3) and abs(got["mean"] - 127.5) < 2 and got["inst"] == ["t1", "t2"]


@pytest.mark.parametrize(("cfg", "match"), [({"reward_noise": {"fp": 0.1}, "reward_source": "rm",
                                               "termination_source": "horizon"}, "reward_noise"),
                                             ({"reward_noise": {"fp": 1.5}}, "reward_noise")])
def test_reward_noise_validated(monkeypatch, cfg, match):
    with pytest.raises(ValueError, match=match):
        _init_sync_env(monkeypatch, **cfg)


def test_weighted_keyframe_sampling():
    from rlinf.envs.sim.world_model.safe.safe_env import SafeWorldModelEnv

    env = SafeWorldModelEnv.__new__(SafeWorldModelEnv)
    env.keyframe_sampling, env.num_group, env.group_size = "weighted", 4000, 2
    env.kf_pool = np.array([10, 11, 12])
    env.kf_pool_w = torch.tensor([1.0, 3.0, 0.0], dtype=torch.float64)
    env._generator = torch.Generator().manual_seed(0)
    kf = env._sample_keyframes()
    assert kf.shape == (8000,) and (kf[0::2] == kf[1::2]).all()  # each group repeats one keyframe
    assert 12 not in kf  # zero weight is never drawn
    share = (kf[0::2] == 11).mean()
    assert 0.72 < share < 0.78  # 3 : 1


class _RenderingRealEnv(_RealEnv):
    """Real simulator stub that also returns observations: images encode the step index."""

    def step(self, action):
        super().step(action)
        img = np.full((256, 256, 3), self.steps, np.uint8)
        img[0, 0] = 200  # marks the unrotated top-left corner
        return ({"agentview_image": img, "robot0_eye_in_hand_image": img + 100,
                 "robot0_eef_pos": np.full(3, self.steps, np.float64),
                 "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                 "robot0_gripper_qpos": np.array([0.01, -0.01])}, 0.0, False, {})


def test_real_observation_frames_and_state(monkeypatch):
    env = _make_env(monkeypatch, reward_source="real", termination_source="horizon",
                    rm_probs=[torch.zeros(8)], real_env=_RenderingRealEnv(goal_at=None), tail_chunks=1,
                    outcome_order="step", cost_source="real", cost_label="grader", obs_source="real")
    env.last_frame = torch.zeros(1, 512, 256, 3, dtype=torch.uint8)
    env.states = torch.zeros(1, 8)
    env._generate = lambda actions: pytest.fail("Wan must not run with obs_source=real")
    env.sam = lambda *a: pytest.fail("SAM must not run with obs_source=real")
    env.state_reader = lambda frame: pytest.fail("state regressor must not run with obs_source=real")
    env.main_size = env.wrist_size = 256
    from rlinf.envs.sim.world_model.safe.safe_env import SafeWorldModelEnv
    env._wrap_obs = lambda: SafeWorldModelEnv._wrap_obs(env)
    obs, _, _, truncations, info = env.chunk_step(torch.zeros(1, 8, 7))
    assert truncations[0, -1]
    f = env.last_frames[0]
    assert [int(f[k, 100, 100, 0]) for k in range(8)] == list(range(1, 9))   # agent view, one render per step
    assert int(f[7, 356, 100, 0]) == 108                                         # wrist view in the bottom half
    assert f[7, 255, 255, 0] == 200 and f[7, 0, 0, 0] == 8                       # rotated 180 deg
    assert env.states[0, :3].tolist() == [8.0, 8.0, 8.0]
    assert env.states[0, 6:].tolist() == pytest.approx([0.01, -0.01])
    assert int(obs[0]["main_images"][0, 100, 100, 0]) == 8                        # policy sees the last render
    assert info[0]["episode"]["cost_sam"][0].item() == 0.0


@pytest.mark.parametrize(("visible", "expected"), [("3", "2"), ("4,5", "7"), ("0", "1"), ("", "1")])
def test_render_egl_remap(monkeypatch, visible, expected):
    from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv

    env = SyncRealWorldModelEnv.__new__(SyncRealWorldModelEnv)
    env.render_egl_remap = {3: 2, 4: 7}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "1")  # what RLinf set for the worker
    monkeypatch.setenv("EGL_DEVICE_ID", "1")
    env._set_render_device()
    assert __import__("os").environ["MUJOCO_EGL_DEVICE_ID"] == expected
    assert __import__("os").environ["EGL_DEVICE_ID"] == expected


def test_real_observation_requires_real_cost(monkeypatch):
    with pytest.raises(ValueError, match="obs_source"):
        _init_sync_env(monkeypatch, cost_source="sam", obs_source="real")
    assert _init_sync_env(monkeypatch, cost_source="real", obs_source="real") == "passed option checks"


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
        assert _init_sync_env(monkeypatch, cost_source="real", cost_label="predicate") == "passed option checks"
    else:
        with pytest.raises(ValueError, match="RLINF_LIBERO_SAFETY_AUX"):
            _init_sync_env(monkeypatch, cost_source="real", cost_label="predicate")
    # the grader label builds its own auxiliary graders: no environment flag needed
    assert _init_sync_env(monkeypatch, cost_source="real") == "passed option checks"


@pytest.mark.parametrize(("key", "value"), [("cost_label", "bogus"), ("cost_window", "bogus")])
def test_cost_label_and_window_validated(monkeypatch, key, value):
    with pytest.raises(ValueError, match=key):
        _init_sync_env(monkeypatch, cost_source="real", **{key: value})


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
