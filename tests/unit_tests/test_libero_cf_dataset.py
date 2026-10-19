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

"""Stitching, length gates and the sim-state sidecar for counterfactual audits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlinf.envs.sim.libero.safety_auditor import LiberoSafetyEpisodeAuditor
from toolkits.libero_safety.build_dataset import build_dataset


def _observation(value: int) -> dict:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) + value
    return {
        "agentview_image": image,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([-1.0, -1.0], dtype=np.float32),
    }


def _metadata(**extra) -> dict:
    return {
        "task_suite": "obstacle_avoidance",
        "safety_level": "L1",
        "task_id": 5,
        "trial_id": 2,
        "reset_state_id": 3,
        "instruction": "pick up the bowl",
        **extra,
    }


def _auditor(root: Path, **kwargs) -> LiberoSafetyEpisodeAuditor:
    return LiberoSafetyEpisodeAuditor(
        root, process_id=0, num_envs=1, mode="shadow_collect",
        save_visual_observations=True, **kwargs
    )


def _episode(auditor, n_steps, first_value, step_value, *, violation_at=None, **meta):
    """Record one episode; frame k shows _observation(step_value(k))."""
    auditor.start_episode(0, _metadata(**meta), _observation(first_value))
    path = None
    for t in range(1, n_steps + 1):
        violation = violation_at == t
        path = auditor.record_transition(
            0,
            timestep=t,
            action=np.full(7, 0.01 * t, dtype=np.float32),
            observation=_observation(step_value(t)),
            reward=float(t == n_steps),
            raw_task_termination=t == n_steps,
            safety_violation_termination=violation,
            severe_collision_termination=False,
            official_termination=violation or t == n_steps,
            emitted_termination=t == n_steps,
            truncated=False,
            predicates=[{
                "instance_id": "constraint:000", "source": "constraint",
                "predicate": "checkrobotcontact", "arguments": ["toy_car_1"],
                "semantics": "violation", "value": violation,
            }],
        )
    assert path is not None
    return json.loads(path.read_text())["episode"]["episode_id"]


def _branch_meta(parent_id, branch_timestep):
    return {
        "episode_kind": "counterfactual_branch",
        "root_episode_id": parent_id,
        "parent_episode_id": parent_id,
        "branch_id": f"{parent_id}-branch-000",
        "branch_timestep": branch_timestep,
    }


def _build(root, tmp_path, name, **kwargs):
    kwargs.setdefault("train_fraction", 1.0)
    kwargs.setdefault("val_fraction", 0.0)
    return build_dataset(audit_roots=[root], output_dir=tmp_path / name, **kwargs)


@pytest.mark.parametrize(("offset", "accepted"), [(0, True), (1, True), (3, False)])
def test_seam_pixel_difference_below_two_is_accepted(tmp_path, offset, accepted):
    root = tmp_path / "audits"
    auditor = _auditor(root)
    rid = _episode(auditor, 20, 0, lambda t: t)
    # Branch before root step 5 starts from root frame 4 (+ offset gray levels).
    _episode(auditor, 20, 4 + offset, lambda t: 100 + t, violation_at=1,
             **_branch_meta(rid, 5))
    if accepted:
        summary = _build(root, tmp_path, "ds")
        assert summary["branches_stitched"] == 1
        manifest = [json.loads(x) for x in (tmp_path / "ds" / "manifest.jsonl").read_text().splitlines()]
        branch = next(m for m in manifest if m["stitched"])
        assert branch["root_prefix_transitions"] == 4
        assert branch["num_frames"] == 4 + 20 + 1
    else:
        with pytest.raises(ValueError, match="root/branch pairing is inconsistent"):
            _build(root, tmp_path, "ds")


def test_branch_of_branch_stitches_recursively(tmp_path):
    root = tmp_path / "audits"
    auditor = _auditor(root)
    rid = _episode(auditor, 20, 0, lambda t: t)
    b1 = _episode(auditor, 20, 4, lambda t: 100 + t, **_branch_meta(rid, 5))
    # Second-level branch before b1's own step 3 starts from b1 frame 2.
    _episode(auditor, 20, 102, lambda t: 200 - t, violation_at=2, **_branch_meta(b1, 3))
    summary = _build(root, tmp_path, "ds")
    assert summary["branches_stitched"] == 2
    manifest = {
        m["episode_kind"] + str(m["root_prefix_transitions"]): m
        for m in map(json.loads, (tmp_path / "ds" / "manifest.jsonl").read_text().splitlines())
    }
    deep = manifest["counterfactual_branch6"]
    assert deep["num_frames"] == 6 + 20 + 1
    frames = np.load(tmp_path / "ds" / deep["wan_path"], allow_pickle=True)
    assert len(frames) == deep["num_frames"]


def test_out_of_range_branch_timestep_is_rejected(tmp_path):
    root = tmp_path / "audits"
    auditor = _auditor(root)
    rid = _episode(auditor, 20, 0, lambda t: t)
    _episode(auditor, 20, 4, lambda t: 100 + t, violation_at=1, **_branch_meta(rid, 50))
    with pytest.raises(ValueError, match="outside root"):
        _build(root, tmp_path, "ds")


def test_episodes_shorter_than_16_frames_are_rejected(tmp_path):
    root = tmp_path / "audits"
    auditor = _auditor(root)
    _episode(auditor, 20, 0, lambda t: t)
    short = _episode(auditor, 5, 0, lambda t: t)
    summary = _build(root, tmp_path, "ds", dry_run=True)
    assert summary["min_frames"] == 16
    assert summary["rejected_too_short"] == 1
    assert summary["episodes_selected"] == 1
    assert any(short in item for item in summary["rejected_too_short_examples"])
    # A 6-step branch (7 frames alone) after 10 root steps stitches to 17 frames: kept.
    rid = _episode(auditor, 20, 0, lambda t: t)
    _episode(auditor, 6, 10, lambda t: 50 + t, violation_at=1, **_branch_meta(rid, 11))
    summary = _build(root, tmp_path, "ds2", dry_run=True)
    assert summary["rejected_too_short"] == 1
    assert summary["branches_stitched"] == 1
    # The same branch from step 8 stitches to 7 + 6 + 1 = 14 frames: rejected.
    rid2 = _episode(auditor, 20, 0, lambda t: t)
    _episode(auditor, 6, 7, lambda t: 50 + t, violation_at=1, **_branch_meta(rid2, 8))
    summary = _build(root, tmp_path, "ds3", dry_run=True)
    assert summary["rejected_too_short"] == 2


def test_auditor_writes_chunk_sim_state_sidecar_and_mocap_start(tmp_path):
    root = tmp_path / "audits"
    auditor = _auditor(root, save_sim_state=True)

    def snap(t):
        return {
            "mujoco_state": np.full(5, t, dtype=np.float64),
            "mocap_pos": np.array([[0.1 * t, 0.0, 1.0]]),
            "mocap_quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
            "cur_time": 0.05 * t,
            "motion_generators": {"toy_car_1": {"phase": np.array([float(t)])}},
        }

    auditor.start_episode(0, _metadata(), _observation(0))
    for t in range(1, 17):
        if (t - 1) % 8 == 0:
            auditor.record_sim_state(0, t, snap(t))
        auditor.record_sim_state(0, t + 3, snap(t + 3))  # misaligned: ignored
        path = auditor.record_transition(
            0, timestep=t, action=np.zeros(7), observation=_observation(t), reward=0.0,
            raw_task_termination=t == 16, safety_violation_termination=False,
            severe_collision_termination=False, official_termination=t == 16,
            emitted_termination=t == 16, truncated=False, predicates=[],
            fixture_qpos={"microwave_1_joint": 0.1 * t},
        )
    record = json.loads(path.read_text())
    assert record["episode"]["alignment"] == "frame[t] + action[t] -> frame[t+1]"
    assert record["episode"]["camera_keys"] == ["agentview_image"]
    assert record["trajectory"][3]["fixture_qpos"] == {"microwave_1_joint": pytest.approx(0.4)}
    meta = record["sim_state_trajectory"]
    assert meta["timesteps"] == [1, 9]
    assert record["mocap_start"]["mocap_pos"] == [[pytest.approx(0.1), 0.0, 1.0]]
    with np.load(path.parent / meta["path"], allow_pickle=False) as state:
        np.testing.assert_array_equal(state["timesteps"], [1, 9])
        assert state["mujoco_state"].shape == (2, 5)
        assert json.loads(str(state["motion_generators"][1]))["toy_car_1"]["phase"] == [9.0]
    assert auditor.last_finalized_steps(0) == 16


def _state_episode(auditor, n_steps, first_value, step_value, state_of, **meta):
    """Like _episode, with a sim-state snapshot every 8 steps (state_of(t) -> vector)."""
    auditor.start_episode(0, _metadata(**meta), _observation(first_value))
    path = None
    for t in range(1, n_steps + 1):
        if (t - 1) % 8 == 0:
            auditor.record_sim_state(
                0, t, {"mujoco_state": np.r_[0.05 * t, state_of(t)], "cur_time": 0.05 * t}
            )
        path = auditor.record_transition(
            0, timestep=t, action=np.zeros(7), observation=_observation(step_value(t)),
            reward=float(t == n_steps), raw_task_termination=t == n_steps,
            safety_violation_termination=t == 1 and "root_episode_id" in meta,
            severe_collision_termination=False, official_termination=t == n_steps,
            emitted_termination=t == n_steps, truncated=False, predicates=[],
        )
    return json.loads(path.read_text())["episode"]["episode_id"]


@pytest.mark.parametrize(("state_shift", "accepted"), [(0.0, True), (1e-3, False)])
def test_sidecar_state_proves_seam_even_when_pixels_differ(tmp_path, state_shift, accepted):
    root = tmp_path / "audits"
    auditor = _auditor(root, save_sim_state=True)
    rid = _state_episode(auditor, 24, 0, lambda t: t, lambda t: np.full(4, float(t)))
    # Branch before root step 9: first frame is 5 gray levels off root frame 8
    # (fails the pixel heuristic), but its first snapshot equals root snapshot 9.
    _state_episode(
        auditor, 20, 8 + 5, lambda t: 100 + t,
        lambda t: np.full(4, 9.0 + state_shift if t == 1 else 50.0 + t),
        **_branch_meta(rid, 9),
    )
    if accepted:
        summary = _build(root, tmp_path, "ds")
        assert summary["branches_stitched"] == 1
    else:
        with pytest.raises(ValueError, match="simulator state differs"):
            _build(root, tmp_path, "ds")


def test_auditor_refuses_sim_state_episode_without_snapshots(tmp_path):
    auditor = _auditor(tmp_path / "audits", save_sim_state=True)
    auditor.start_episode(0, _metadata(), _observation(0))
    with pytest.raises(RuntimeError, match="has no snapshots"):
        auditor.record_transition(
            0, timestep=1, action=np.zeros(7), observation=_observation(1), reward=1.0,
            raw_task_termination=True, safety_violation_termination=False,
            severe_collision_termination=False, official_termination=True,
            emitted_termination=True, truncated=False, predicates=[],
        )
