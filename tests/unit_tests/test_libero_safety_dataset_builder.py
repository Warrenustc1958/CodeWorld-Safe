# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
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


def _predicate(value: bool) -> dict:
    return {
        "instance_id": "constraint:000",
        "source": "constraint",
        "predicate": "checkcontact",
        "arguments": ["mug", "obstacle"],
        "semantics": "violation",
        "value": value,
    }


def _metadata(**extra) -> dict:
    return {
        "task_suite": "obstacle_avoidance",
        "safety_level": "L0",
        "task_id": 1,
        "trial_id": 2,
        "reset_state_id": 3,
        "instruction": "put the mug in the basket",
        **extra,
    }


def _record(
    auditor: LiberoSafetyEpisodeAuditor,
    timestep: int,
    *,
    violation: bool,
    success: bool,
):
    return auditor.record_transition(
        0,
        timestep=timestep,
        action=np.full(7, timestep, dtype=np.float32),
        observation=_observation(timestep),
        reward=float(success),
        raw_task_termination=success,
        safety_violation_termination=violation,
        severe_collision_termination=False,
        official_termination=violation or success,
        emitted_termination=success,
        truncated=False,
        predicates=[_predicate(violation)],
    )


def test_builder_exports_aligned_wan_and_sam_data(tmp_path: Path):
    audit_root = tmp_path / "audits"
    auditor = LiberoSafetyEpisodeAuditor(
        audit_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )

    # Root: two safe steps, success on step 2.
    auditor.start_episode(0, _metadata(), _observation(0))
    assert _record(auditor, 1, violation=False, success=False) is None
    q1_path = _record(auditor, 2, violation=False, success=True)
    assert q1_path is not None
    q1_id = json.loads(q1_path.read_text())["episode"]["episode_id"]

    # Branch restored from the snapshot taken just before root step 2, so its
    # first frame is root frame 1 and the root prefix is one transition.
    auditor.start_episode(
        0,
        _metadata(
            episode_kind="counterfactual_branch",
            root_episode_id=q1_id,
            branch_id=f"{q1_id}-branch-000",
            branch_timestep=2,
            alpha=0.5,
            perturbation_range=[0.2, 0.8],
        ),
        _observation(1),
    )
    assert _record(auditor, 1, violation=True, success=False) is None
    q2_path = _record(auditor, 2, violation=False, success=True)
    assert q2_path is not None

    q1_record = json.loads(q1_path.read_text())
    visual_path = q1_path.parent / q1_record["visual_trajectory"]["path"]
    with np.load(visual_path, allow_pickle=False) as visual:
        assert visual["agentview_image"].shape == (3, 2, 3, 3)
        np.testing.assert_array_equal(
            visual["agentview_image"][0], _observation(0)["agentview_image"][::-1, ::-1]
        )

    output = tmp_path / "dataset"
    summary = build_dataset(
        audit_roots=[audit_root],
        output_dir=output,
        quadrants=("Q1", "Q2"),
        train_fraction=1.0,
        val_fraction=0.0,
        min_per_quadrant={"Q1": 1, "Q2": 1},
    )

    assert summary["quadrant_counts"] == {"Q1": 1, "Q2": 1, "Q3": 0, "Q4": 0}
    assert summary["ready_for_wan"] is True
    manifest = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(manifest) == 2
    assert {item["quadrant"] for item in manifest} == {"Q1", "Q2"}

    q1_manifest = next(item for item in manifest if item["quadrant"] == "Q1")
    q2_manifest = next(item for item in manifest if item["quadrant"] == "Q2")
    # Branch metadata is resolved and the branch shares its root's split group.
    assert q2_manifest["root_episode_id"] == q1_manifest["episode_id"]
    assert q2_manifest["alpha"] == 0.5
    assert q2_manifest["stitched"] is True
    assert q2_manifest["root_prefix_transitions"] == 1
    assert q2_manifest["num_transitions"] == 3
    assert q2_manifest["episode_num_transitions"] == 2
    assert q1_manifest["stitched"] is False
    assert summary["branches_stitched"] == 1
    assert summary["transitions_selected"] == 2 + 3
    assert summary["frames_selected"] == 3 + 4

    q1_trajectory = np.load(output / q1_manifest["wan_path"], allow_pickle=True)
    assert len(q1_trajectory) == 3  # T + 1, outcome frame kept
    assert q1_trajectory[-1]["is_terminal"] is True
    assert q1_trajectory[-1]["action_valid"] is False
    np.testing.assert_array_equal(q1_trajectory[-1]["delta_action"], np.zeros(7))

    trajectory = np.load(output / q2_manifest["wan_path"], allow_pickle=True)
    # root frame 0 + branch frames 0..2 (branch frame 0 replaces root frame 1)
    assert len(trajectory) == 4
    assert [frame["segment"] for frame in trajectory] == [
        "root_prefix",
        "episode",
        "episode",
        "episode",
    ]
    for index, value in enumerate((0, 1, 1, 2)):
        np.testing.assert_array_equal(
            trajectory[index]["image"],
            _observation(value)["agentview_image"][::-1, ::-1],
        )
    # Root step 1 action, then branch actions.
    assert [float(frame["delta_action"][0]) for frame in trajectory] == [
        1.0,
        1.0,
        2.0,
        0.0,
    ]
    assert trajectory[0]["delta_action"].shape == (7,)
    assert trajectory[0]["init_ee_pose"].shape == (8,)
    assert trajectory[0]["post_action_violation"] is False
    assert trajectory[1]["post_action_violation"] is True
    assert trajectory[2]["state_violation"] is True
    assert trajectory[-1]["is_terminal"] is True
    assert [frame["timestep"] for frame in trajectory] == [1, 2, 3, 4]

    # The output is directly consumable by RLinf's existing Wan reset dataset.
    dataset = NpyTrajectoryDatasetWrapper(str(output / "wan" / "train"))
    item = dataset[0]
    assert item["start_items"][0]["image"].shape == (3, 2, 3)
    assert item["start_items"][0]["action"].shape == (7,)

    sam_labels = [
        json.loads(line)
        for line in (output / "sam" / "train.jsonl").read_text().splitlines()
    ]
    q2_labels = [item for item in sam_labels if item["quadrant"] == "Q2"]
    assert [item["segment"] for item in q2_labels] == [
        "root_prefix",
        "episode",
        "episode",
    ]
    # The root prefix sees the branch's violation one step ahead.
    assert q2_labels[0]["post_action_violation"] is False
    assert q2_labels[0]["time_to_next_violation_steps"] == 1
    assert q2_labels[1]["post_action_violation"] is True
    assert q2_labels[1]["active_constraint_instance_ids"] == ["constraint:000"]
    assert q2_labels[1]["time_to_next_violation_steps"] == 0
    assert q2_labels[1]["source_timestep"] == 1
    assert q2_labels[2]["unsafe_so_far"] is True


def test_builder_rejects_mismatched_branch_seam_and_missing_root(tmp_path: Path):
    audit_root = tmp_path / "audits"
    auditor = LiberoSafetyEpisodeAuditor(
        audit_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )
    auditor.start_episode(0, _metadata(), _observation(0))
    q1_path = _record(auditor, 1, violation=False, success=True)
    q1_id = json.loads(q1_path.read_text())["episode"]["episode_id"]
    # Claims to branch before step 1 (root frame 0) but starts from frame 40.
    auditor.start_episode(
        0,
        _metadata(
            episode_kind="counterfactual_branch",
            root_episode_id=q1_id,
            branch_id=f"{q1_id}-branch-000",
            branch_timestep=1,
        ),
        _observation(40),
    )
    assert _record(auditor, 1, violation=True, success=True) is not None
    with pytest.raises(ValueError, match="root/branch pairing is inconsistent"):
        build_dataset(
            audit_roots=[audit_root],
            output_dir=tmp_path / "dataset",
            train_fraction=1.0,
            val_fraction=0.0,
        )

    # A branch whose root is not under the audit roots cannot be stitched.
    orphan_root = tmp_path / "orphan"
    orphan = LiberoSafetyEpisodeAuditor(
        orphan_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )
    orphan.start_episode(
        0,
        _metadata(
            episode_kind="counterfactual_branch",
            root_episode_id="missing-root",
            branch_id="missing-root-branch-000",
            branch_timestep=1,
        ),
        _observation(0),
    )
    assert _record(orphan, 1, violation=True, success=True) is not None
    with pytest.raises(ValueError, match="not found under the audit roots"):
        build_dataset(
            audit_roots=[orphan_root], output_dir=tmp_path / "d2", dry_run=True
        )
    summary = build_dataset(
        audit_roots=[orphan_root],
        output_dir=tmp_path / "d2",
        stitch_branches=False,
        dry_run=True,
    )
    assert summary["branches_stitched"] == 0


def test_builder_refuses_missing_visual_sidecar_and_quadrant_shortfall(
    tmp_path: Path,
):
    audit_root = tmp_path / "audits"
    auditor = LiberoSafetyEpisodeAuditor(
        audit_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
    )
    auditor.start_episode(0, _metadata(), _observation(0))
    assert _record(auditor, 1, violation=False, success=True) is not None

    with pytest.raises(ValueError, match="no visual_trajectory"):
        build_dataset(
            audit_roots=[audit_root],
            output_dir=tmp_path / "dataset",
            quadrants=("Q1",),
            dry_run=True,
        )

    visual_root = tmp_path / "multi_run_audits" / "seed_0"
    visual_auditor = LiberoSafetyEpisodeAuditor(
        visual_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )
    visual_auditor.start_episode(0, _metadata(), _observation(0))
    assert _record(visual_auditor, 1, violation=False, success=True) is not None
    with pytest.raises(ValueError, match="shortfalls"):
        build_dataset(
            audit_roots=[visual_root],
            output_dir=tmp_path / "dataset",
            quadrants=("Q1", "Q2"),
            min_per_quadrant={"Q2": 1},
            dry_run=True,
        )

    # Auditor episode IDs restart in each rollout directory. The builder must
    # namespace them so independent seeds can be accumulated safely.
    second_visual_root = tmp_path / "multi_run_audits" / "seed_1"
    second_auditor = LiberoSafetyEpisodeAuditor(
        second_visual_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )
    second_auditor.start_episode(0, _metadata(), _observation(0))
    assert _record(second_auditor, 1, violation=False, success=True) is not None
    summary = build_dataset(
        audit_roots=[tmp_path / "multi_run_audits"],
        output_dir=tmp_path / "dataset",
        quadrants=("Q1",),
        dry_run=True,
    )
    assert summary["episodes_selected"] == 2
