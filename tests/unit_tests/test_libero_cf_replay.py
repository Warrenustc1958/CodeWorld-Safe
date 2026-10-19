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

"""Per-chunk replay from saved simulator snapshots reproduces the trajectory (P6/P7).

Open-loop replay of the whole action sequence drifts after contact, so the
sidecar stores a snapshot every chunk; restoring snapshot k and replaying the
chunk's actions must land on snapshot k+1. Uses a task with a mocap-driven
obstacle so the motion-generator state is part of what must round-trip.
Requires a working LIBERO-Safety install (skipped otherwise).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

CHUNK = 8
N_CHUNKS = 4
TASK = ("obstacle_avoidance", "L1", "pick_up_the_akita_black_bowl_on_the_stove_and_place_it_on_the_plate")


@pytest.fixture(scope="module")
def env():
    pytest.importorskip("robosuite")
    try:
        import libero.libero as libero_root
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as error:  # noqa: BLE001 - any import failure means no LIBERO here
        pytest.skip(f"LIBERO-Safety not importable: {error}")
    os.environ.setdefault("LIBERO_TYPE", "safety")
    suite, level, name = TASK
    bddl = os.path.join(libero_root.get_libero_path("bddl_files"), suite, level, f"{name}.bddl")
    if not os.path.isfile(bddl):
        pytest.skip(f"missing BDDL {bddl}")
    instance = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=64, camera_widths=64, ignore_done=True
    )
    instance.seed(0)
    instance.reset()
    yield instance
    instance.close()


def _qpos_qvel(snapshot):
    # MjSimState.flatten() = [time, qpos..., qvel..., (act...)]; drop time.
    return np.asarray(snapshot["mujoco_state"], dtype=np.float64)[1:]


def test_chunk_replay_from_snapshots_matches_original(env):
    from rlinf.envs.sim.libero.venv import (
        _restore_simulator_state,
        _snapshot_simulator_state,
        _unwrap_task_env,
    )

    task_env = _unwrap_task_env(env)
    assert getattr(task_env, "mocap_motion_generators", {}), "task should have a mocap obstacle"
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.6, 0.6, size=(N_CHUNKS * CHUNK, 7))
    actions[:, 6] = np.where(np.arange(len(actions)) < 16, -1.0, 1.0)

    snapshots = []
    for t, action in enumerate(actions):
        if t % CHUNK == 0:
            snapshots.append(_snapshot_simulator_state(env))
        env.step(action)
    snapshots.append(_snapshot_simulator_state(env))

    drifts, mocap_drifts = [], []
    for k in range(N_CHUNKS):
        _restore_simulator_state(env, snapshots[k])
        for action in actions[k * CHUNK : (k + 1) * CHUNK]:
            env.step(action)
        replay = _snapshot_simulator_state(env)
        drifts.append(float(np.abs(_qpos_qvel(replay) - _qpos_qvel(snapshots[k + 1])).max()))
        mocap_drifts.append(
            float(np.abs(np.asarray(replay["mocap_pos"]) - np.asarray(snapshots[k + 1]["mocap_pos"])).max())
        )
    # The obstacle actually moved, so the mocap comparison is not vacuous.
    assert np.abs(np.asarray(snapshots[-1]["mocap_pos"]) - np.asarray(snapshots[0]["mocap_pos"])).max() > 0
    assert max(drifts) < 1e-6, f"per-chunk qpos/qvel drift {drifts}"
    assert max(mocap_drifts) < 1e-9, f"per-chunk mocap drift {mocap_drifts}"
