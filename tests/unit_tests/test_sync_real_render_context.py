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

"""SyncRealWorldModelEnv with several offscreen envs in one process: build A, build B, render A.

Creating B leaves B's GL context current and MjRenderContext.render() does not re-bind, so an unbound render of A
draws A's geometry with B's textures. The sync env must bind each slot's context before stepping / reading it.
Uses two built-in robosuite tasks with different textures; LIBERO's OffScreenRenderEnv goes through the same
robosuite MjRenderContext.
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
# libOSMesa links its own libLLVM; numba compiling (robosuite transform_utils) in the same process segfaults
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import types

import numpy as np
import pytest

suite = pytest.importorskip("robosuite")
sync_real_env = pytest.importorskip("rlinf.envs.sim.world_model.safe.sync_real_env")
SyncRealWorldModelEnv = sync_real_env.SyncRealWorldModelEnv

CAM = 128


class _Slot:
    """The parts of a LIBERO OffScreenRenderEnv that the sync env touches (.env, step, check_success)."""

    def __init__(self, name):
        self.env = suite.make(name, robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                              use_camera_obs=True, camera_names=["agentview", "robot0_eye_in_hand"],
                              camera_heights=CAM, camera_widths=CAM, ignore_done=True, horizon=100000)
        self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def check_success(self):
        return False

    def bound_view(self):
        """Ground truth: this env's agent view rendered with its own context bound."""
        self.env.sim._render_context_offscreen.gl_ctx.make_current()
        return self.env._get_observations(force_update=True)["agentview_image"].copy()


def _mae(x, y):
    return float(np.abs(x.astype(np.float32) - y.astype(np.float32)).mean())


@pytest.fixture(scope="module")
def slots():
    a = _Slot("Lift")
    ref_a = a.bound_view()
    b = _Slot("PickPlace")  # different textures from Lift (Door shares most: MAE ~0.4); created last, stays current
    yield a, b, ref_a
    for s in (a, b):
        s.env.sim._render_context_offscreen.gl_ctx.make_current()
        s.env.close()


def _sync_env(real):
    env = SyncRealWorldModelEnv.__new__(SyncRealWorldModelEnv)  # only the real-simulator half is exercised
    env._real = list(real)
    return env


def test_unbound_render_uses_wrong_context(slots):
    """The bug itself: after B is created, reading A without re-binding gives a different image."""
    a, b, ref_a = slots
    b.env.sim._render_context_offscreen.gl_ctx.make_current()
    wrong = a.env._get_observations(force_update=True)["agentview_image"]
    assert _mae(wrong, ref_a) > 20.0, "bug did not reproduce: unbound render matched the reference"


def test_real_obs_now_binds_slot_context(slots):
    a, b, ref_a = slots
    env = _sync_env([a, b])
    b.env.sim._render_context_offscreen.gl_ctx.make_current()  # state left by reset(): last slot bound
    obs = env._real_obs_now(0)
    assert _mae(obs["agentview_image"], ref_a) < 0.5
    frame = SyncRealWorldModelEnv._real_frame(obs)
    assert frame.shape == (2 * CAM, CAM, 3)


def test_real_chunk_binds_each_slot(slots, monkeypatch):
    a, b, _ = slots
    from rlinf.envs.sim.libero import safety_auditor

    monkeypatch.setattr(safety_auditor, "audit_predicate_instances", lambda env: [])
    env = _sync_env([a, b])
    n, chunk = 2, 2
    for k, v in dict(num_envs=n, chunk=chunk, real_done=np.zeros(n, bool), real_succ=np.zeros(n, bool),
                     real_viol=np.zeros(n, bool), c_off=np.zeros(n, bool), c_aux=np.zeros(n, bool),
                     last_goal=np.zeros(n, bool), real_succ_step=np.full(n, -1), tail_left=np.full(n, -1),
                     cost_window="goal", cost_source="sam", cost_label="official", dump_arrays=False,
                     record_steps=False, step_order=True, end_on_violation=False, tail_chunks=0,
                     _renders=True, side_render=False).items():
        setattr(env, k, v)
    b.env.sim._render_context_offscreen.gl_ctx.make_current()
    act = np.zeros((n, chunk, a.env.action_dim), np.float32)
    frames = np.zeros((n, chunk, 2 * CAM, CAM, 3), np.uint8)
    states = np.zeros((n, 8), np.float32)
    env._real_chunk(act, 0, np.ones(n, bool), frames, states)
    # the last executed step's render must equal each env re-rendered with its own context
    for i, s in enumerate((a, b)):
        truth = s.bound_view()[::-1, ::-1]
        assert _mae(frames[i, -1, :CAM], truth) < 0.5, f"slot {i} rendered with another slot's context"


def test_side_render_pairs_real_frames_without_touching_wan(slots, monkeypatch):
    """side_render (obs_source=wan): real renders go to the side buffers, one per executed step, each slot with its
    own context; the Wan frames the policy sees are left as generated."""
    a, b, _ = slots
    from rlinf.envs.sim.libero import safety_auditor

    monkeypatch.setattr(safety_auditor, "audit_predicate_instances", lambda env: [])
    monkeypatch.setattr(SyncRealWorldModelEnv, "_real_qpos", lambda self, i: np.zeros(3, np.float32))
    monkeypatch.setattr(SyncRealWorldModelEnv, "_real_state", staticmethod(lambda obs: np.zeros(8, np.float32)))
    env = _sync_env([a, b])
    n, chunk = 2, 3
    for k, v in dict(num_envs=n, chunk=chunk, real_done=np.zeros(n, bool), real_succ=np.zeros(n, bool),
                     real_viol=np.zeros(n, bool), c_off=np.zeros(n, bool), c_aux=np.zeros(n, bool),
                     last_goal=np.zeros(n, bool), real_succ_step=np.full(n, -1), tail_left=np.full(n, -1),
                     cost_window="goal", cost_source="sam", cost_label="official", dump_arrays=False,
                     record_steps=False, step_order=True, end_on_violation=False, tail_chunks=0,
                     _renders=True, side_render=True).items():
        setattr(env, k, v)
    env.side_frame = np.zeros((n, 2 * CAM, CAM, 3), np.uint8)
    env.side_chunk = np.repeat(env.side_frame[:, None], chunk, 1)
    env.side_jpg, env.side_state, env.side_qpos = ([[b"x"] for _ in range(n)] for _ in range(3))
    b.env.sim._render_context_offscreen.gl_ctx.make_current()
    act = np.zeros((n, chunk, a.env.action_dim), np.float32)
    wan = np.full((n, chunk, 2 * CAM, CAM, 3), 7, np.uint8)
    env._real_chunk(act, 0, np.ones(n, bool), wan, None)
    assert (wan == 7).all(), "side_render overwrote the Wan frames"
    for i, s in enumerate((a, b)):
        assert len(env.side_jpg[i]) == 1 + chunk and len(env.side_qpos[i]) == 1 + chunk
        truth = s.bound_view()[::-1, ::-1]
        assert _mae(env.side_chunk[i, -1, :CAM], truth) < 0.5, f"slot {i} side render used another context"
        assert _mae(env.side_frame[i, :CAM], truth) < 0.5
