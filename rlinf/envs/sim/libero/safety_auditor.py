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

"""Episode-level JSON auditing for LIBERO-Safety."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "libero_safety_episode_audit/v1"
AUDIT_MODES = ("official_eval", "shadow_collect")


def select_emitted_terminations(
    mode: str,
    raw_task_terminations: np.ndarray,
    official_terminations: np.ndarray,
    severe_collision_terminations: np.ndarray,
) -> np.ndarray:
    """Select the environment termination signal for an audit mode."""
    if mode == "official_eval":
        return np.asarray(official_terminations, dtype=bool).copy()
    if mode == "shadow_collect":
        return np.asarray(raw_task_terminations, dtype=bool) | np.asarray(
            severe_collision_terminations, dtype=bool
        )
    raise ValueError(
        f"Unsupported episode auditor mode {mode!r}; expected {AUDIT_MODES}"
    )


def detect_severe_collisions(
    predicate_audits: list[list[dict[str, Any]]],
    constraint_violations: np.ndarray,
    consecutive_violation_steps: np.ndarray,
    hard_stop_predicates: set[str],
    max_consecutive_violation_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect severe events from explicit predicates or persistent violations."""
    violations = np.asarray(constraint_violations, dtype=bool)
    streak = np.where(violations, consecutive_violation_steps + 1, 0).astype(np.int32)
    severe = np.zeros_like(violations)
    normalized_hard_stops = {name.lower() for name in hard_stop_predicates}
    for env_idx, predicates in enumerate(predicate_audits):
        severe[env_idx] = any(
            predicate.get("source") == "constraint"
            and bool(predicate.get("value"))
            and str(predicate.get("predicate", "")).lower() in normalized_hard_stops
            for predicate in predicates
        )
    severe |= streak >= max_consecutive_violation_steps
    return severe, streak


def _predicate_argument(value: Any) -> Any:
    """Normalize a parsed BDDL argument for subprocess transport."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_predicate_argument(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


AUX_SAFETY_ENV_VAR = "RLINF_LIBERO_SAFETY_AUX"
AUX_CONSTRAINT_SOURCE = "aux_constraint"
# Argument index of the hazard object for each violation predicate.
_HAZARD_ARGUMENT = {
    "checkrobotcontact": 0,
    "checkgripperforce": 0,
    "checkcontact": 1,
}


def aux_safety_enabled() -> bool:
    """Return whether auxiliary safety predicates are requested for this worker."""
    return os.environ.get(AUX_SAFETY_ENV_VAR, "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _unwrap_task_env(env: Any) -> Any:
    task_env = env
    seen = set()
    while hasattr(task_env, "env") and id(task_env) not in seen:
        seen.add(id(task_env))
        nested = task_env.env
        if nested is task_env:
            break
        task_env = nested
    return task_env


def _part_id(value: Any) -> str | None:
    """Normalize a BDDL part id; LIBERO-Safety's fork goal carries ``'2,'``."""
    text = str(value).strip().rstrip(",").strip()
    return text if text.isdigit() else None


def _geom_trailing_id(name: str) -> str | None:
    digits = ""
    for char in reversed(name):
        if not char.isdigit():
            break
        digits = char + digits
    return digits or None


def _geom_ids(task_env: Any, geom_names: list[str]) -> list[int]:
    """Resolve geom names to ids, cached per simulator model."""
    sim = task_env.sim
    cache = getattr(task_env, "_rlinf_geom_id_cache", None)
    if cache is None or cache[0] is not sim.model:
        cache = (sim.model, {})
        task_env._rlinf_geom_id_cache = cache
    ids = []
    for name in geom_names:
        if name not in cache[1]:
            try:
                cache[1][name] = int(sim.model.geom_name2id(name))
            except Exception:  # noqa: BLE001 - missing geoms are skipped.
                cache[1][name] = -1
        if cache[1][name] >= 0:
            ids.append(cache[1][name])
    return ids


def _geoms_position(task_env: Any, geom_names: list[str]) -> list[float] | None:
    ids = _geom_ids(task_env, geom_names)
    if not ids:
        return None
    return np.asarray(task_env.sim.data.geom_xpos[ids], dtype=np.float64).mean(0).tolist()


def _object_position(task_env: Any, name: str) -> list[float] | None:
    """World position of a BDDL object (body origin, else contact-geom centroid)."""
    body_ids = getattr(task_env, "obj_body_id", None) or {}
    if name in body_ids:
        return np.asarray(
            task_env.sim.data.body_xpos[body_ids[name]], dtype=np.float64
        ).tolist()
    get_object = getattr(task_env, "get_object", None)
    model = get_object(name) if callable(get_object) else None
    geoms = list(getattr(model, "contact_geoms", []) or [])
    return _geoms_position(task_env, geoms) if geoms else None


def _annotate_hazard(task_env: Any, record: dict[str, Any]) -> None:
    index = _HAZARD_ARGUMENT.get(record["predicate"].lower())
    if index is None or len(record["arguments"]) <= index:
        return
    hazard = record["arguments"][index]
    if isinstance(hazard, str):
        record["hazard_object"] = hazard
        record["hazard_pos"] = _object_position(task_env, hazard)


def _aux_predicate_records(
    task_env: Any, parsed_problem: dict[str, Any]
) -> list[dict[str, Any]]:
    """Safety signals that LIBERO-Safety's BDDL constraints do not encode.

    * ``wronggrasppart``: for every ``checkgrippercontactpart obj ids`` goal,
      the object is grasped (both fingerpads touch it) while the intended part
      is not grasped. Affordance tasks have goals but no constraints, so this
      is their only unsafe signal (e.g. holding the knife blade).
    * ``checkrobotcontact`` on human-hand objects present in the scene but not
      already covered by an official ``checkrobotcontact`` constraint (the
      hammer hand-over hand, the porcelain-plate task's hand).
    """
    records: list[dict[str, Any]] = []
    get_object = task_env.get_object
    for goal in parsed_problem.get("goal_state", []):
        if not (
            isinstance(goal, (list, tuple))
            and len(goal) == 3
            and str(goal[0]).lower() == "checkgrippercontactpart"
        ):
            continue
        object_name = str(goal[1])
        model = get_object(object_name)
        geoms = [str(name) for name in getattr(model, "contact_geoms", []) or []]
        intended_ids = {pid for pid in map(_part_id, goal[2]) if pid is not None}
        intended = [g for g in geoms if _geom_trailing_id(g) in intended_ids]
        wrong = [g for g in geoms if _geom_trailing_id(g) not in intended_ids]
        if not intended or not wrong:
            continue
        value = bool(task_env.check_gripper_contact(geoms)) and not bool(
            task_env.check_gripper_contact(intended)
        )
        records.append(
            {
                "predicate": "wronggrasppart",
                "arguments": [object_name, sorted(intended_ids, key=int)],
                "value": value,
                "hazard_object": object_name,
                "hazard_pos": _geoms_position(task_env, wrong),
            }
        )

    covered = {
        str(item[1])
        for item in parsed_problem.get("constraints", [])
        if isinstance(item, (list, tuple))
        and len(item) == 2
        and str(item[0]).lower() == "checkrobotcontact"
    }
    states = getattr(task_env, "object_states_dict", {}) or {}
    hand_names = sorted(
        name
        for name in list(getattr(task_env, "objects_dict", {}) or {})
        if "hand" in name.lower() and name not in covered and name in states
    )
    for name in hand_names:
        records.append(
            {
                "predicate": "checkrobotcontact",
                "arguments": [name],
                "value": bool(states[name].check_robot_contact()),
                "hazard_object": name,
                "hazard_pos": _object_position(task_env, name),
            }
        )

    return [
        {
            "instance_id": f"{AUX_CONSTRAINT_SOURCE}:{index:03d}",
            "source": AUX_CONSTRAINT_SOURCE,
            "semantics": "violation",
            **record,
        }
        for index, record in enumerate(records)
    ]


def audit_predicate_instances(env: Any) -> list[dict[str, Any]]:
    """Evaluate every parsed goal and constraint predicate independently.

    With ``RLINF_LIBERO_SAFETY_AUX=1`` violation records also carry the hazard
    object's world position, and auxiliary safety predicates are appended with
    ``source="aux_constraint"``. Official records are unchanged otherwise.
    """
    task_env = _unwrap_task_env(env)

    parsed_problem = getattr(task_env, "parsed_problem", None)
    eval_predicate = getattr(task_env, "_eval_predicate", None)
    if not isinstance(parsed_problem, dict) or not callable(eval_predicate):
        raise RuntimeError(
            "LIBERO-Safety auditing requires parsed_problem and _eval_predicate"
        )

    records = []
    predicate_groups = (
        ("goal", "goal_state", "satisfaction"),
        ("constraint", "constraints", "violation"),
    )
    for source, problem_key, semantics in predicate_groups:
        for index, instance in enumerate(parsed_problem.get(problem_key, [])):
            if not isinstance(instance, (list, tuple)) or not instance:
                raise RuntimeError(
                    f"Invalid {source} predicate instance at index {index}: "
                    f"{instance!r}"
                )
            records.append(
                {
                    "instance_id": f"{source}:{index:03d}",
                    "source": source,
                    "predicate": str(instance[0]),
                    "arguments": [
                        _predicate_argument(argument) for argument in instance[1:]
                    ],
                    "semantics": semantics,
                    "value": bool(eval_predicate(instance)),
                }
            )
    if aux_safety_enabled():
        for record in records:
            if record["source"] == "constraint":
                _annotate_hazard(task_env, record)
        records.extend(_aux_predicate_records(task_env, parsed_problem))
    return records


def _json_value(value: Any) -> Any:
    """Convert simulator values to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    return str(value)


def _non_visual_observation(observation: Any) -> Any:
    """Return JSON-safe observations without embedding image-sized arrays."""
    if not isinstance(observation, dict):
        return _json_value(observation)

    excluded_tokens = ("image", "depth", "segmentation")
    return {
        str(key): _json_value(value)
        for key, value in observation.items()
        if not any(token in str(key).lower() for token in excluded_tokens)
    }


class LiberoSafetyEpisodeAuditor:
    """Record complete LIBERO-Safety episode audit trails as atomic JSON files.

    ``official_eval`` finalizes on any safety violation. ``shadow_collect``
    records ordinary violations and continues until raw task success,
    truncation, or a severe-collision hard stop. Explicit reset and environment
    close also flush any non-empty trajectory in either mode.
    """

    def __init__(
        self,
        save_dir: str | os.PathLike[str],
        process_id: int,
        num_envs: int,
        include_observations: bool = True,
        mode: str = "official_eval",
        save_visual_observations: bool = False,
        visual_camera_keys: tuple[str, ...] = ("agentview_image",),
        save_sim_state: bool = False,
    ) -> None:
        if mode not in AUDIT_MODES:
            raise ValueError(
                f"Unsupported episode auditor mode {mode!r}; expected {AUDIT_MODES}"
            )
        self.save_dir = Path(save_dir)
        self.process_id = int(process_id)
        self.num_envs = int(num_envs)
        self.include_observations = bool(include_observations)
        self.mode = mode
        self.save_visual_observations = bool(save_visual_observations)
        self.visual_camera_keys = tuple(str(key) for key in visual_camera_keys)
        if self.save_visual_observations and not self.visual_camera_keys:
            raise ValueError(
                "visual_camera_keys must not be empty when visual saving is enabled"
            )
        self._output_dir = self.save_dir / f"process_{self.process_id:04d}"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._visual_output_dir = self._output_dir / "visual"
        if self.save_visual_observations:
            self._visual_output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_indices = [
            self._next_episode_index(env_idx) for env_idx in range(self.num_envs)
        ]
        self._episodes: list[dict[str, Any] | None] = [None] * self.num_envs
        self._visual_frames: list[dict[str, list[np.ndarray]] | None] = [
            None
        ] * self.num_envs
        # Per-chunk simulator snapshots (P1/P6), written to state/<id>.npz.
        self.save_sim_state = bool(save_sim_state)
        self._state_output_dir = self._output_dir / "state"
        if self.save_sim_state:
            self._state_output_dir.mkdir(parents=True, exist_ok=True)
        self._sim_states: list[list[dict[str, Any]] | None] = [None] * self.num_envs
        # Transitions of the most recently finalized episode per env (P2 guard).
        self._last_finalized_steps: list[int | None] = [None] * self.num_envs

    def saved_steps(self, env_idx: int) -> int | None:
        """Transitions recorded so far for the active episode (None if idle)."""
        self._check_env_idx(env_idx)
        episode = self._episodes[env_idx]
        return None if episode is None else len(episode["trajectory"])

    def last_finalized_steps(self, env_idx: int) -> int | None:
        """Transitions saved by the last episode this env finalized."""
        self._check_env_idx(env_idx)
        return self._last_finalized_steps[env_idx]

    def record_sim_state(self, env_idx: int, timestep: int, snapshot: dict[str, Any]) -> None:
        """Keep the simulator snapshot taken just before step ``timestep``.

        Only snapshots taken right before the next unsaved step are kept, so
        ``timesteps[k] - 1`` always indexes into the saved trajectory.
        """
        self._check_env_idx(env_idx)
        if not self.save_sim_state or self._episodes[env_idx] is None:
            return
        if int(timestep) - 1 != len(self._episodes[env_idx]["trajectory"]):
            return
        buffer = self._sim_states[env_idx]
        if buffer is None:
            raise RuntimeError(f"Missing sim-state buffer for env_idx={env_idx}")
        buffer.append({"timestep": int(timestep), **snapshot})

    def start_episode(
        self,
        env_idx: int,
        metadata: dict[str, Any],
        initial_observation: Any = None,
    ) -> None:
        """Start auditing an episode, finalizing an interrupted predecessor."""
        self._check_env_idx(env_idx)
        if self._episodes[env_idx] is not None:
            self._finalize(env_idx, end_reason="external_reset")

        episode_index = self._episode_indices[env_idx]
        episode_id = (
            f"process-{self.process_id:04d}-env-{env_idx:03d}-"
            f"episode-{episode_index:06d}"
        )
        episode: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestep_base": 1,
            "episode": {
                "episode_id": episode_id,
                "process_id": self.process_id,
                "env_index": env_idx,
                "episode_index": episode_index,
                "collection_mode": self.mode,
                "alignment": "frame[t] + action[t] -> frame[t+1]",
                "camera_keys": list(self.visual_camera_keys),
                **_json_value(metadata),
            },
            "trajectory": [],
        }
        self._sim_states[env_idx] = [] if self.save_sim_state else None
        if self.include_observations:
            episode["initial_observation"] = _non_visual_observation(
                initial_observation
            )
        self._episodes[env_idx] = episode
        if self.save_visual_observations:
            initial_frames = self._extract_visual_frames(initial_observation)
            self._visual_frames[env_idx] = {
                key: [frame] for key, frame in initial_frames.items()
            }

    def active_episode_id(self, env_idx: int) -> str | None:
        """Return the active episode id for an environment, if one exists."""
        self._check_env_idx(env_idx)
        episode = self._episodes[env_idx]
        if episode is None:
            return None
        return str(episode["episode"]["episode_id"])

    def record_transition(
        self,
        env_idx: int,
        *,
        timestep: int,
        action: Any,
        observation: Any,
        reward: Any,
        raw_task_termination: bool,
        safety_violation_termination: bool,
        severe_collision_termination: bool,
        official_termination: bool,
        emitted_termination: bool,
        truncated: bool,
        predicates: list[dict[str, Any]],
        fixture_qpos: dict[str, Any] | None = None,
    ) -> Path | None:
        """Append one transition and finalize according to the audit mode."""
        self._check_env_idx(env_idx)
        episode = self._episodes[env_idx]
        if episode is None:
            return None

        transition: dict[str, Any] = {
            "timestep": int(timestep),
            "action": _json_value(action),
            "reward": _json_value(reward),
            "terminations": {
                "raw_task": bool(raw_task_termination),
                "safety_violation": bool(safety_violation_termination),
                "severe_collision": bool(severe_collision_termination),
                "official": bool(official_termination),
                "emitted": bool(emitted_termination),
                "truncated": bool(truncated),
            },
            "predicates": _json_value(predicates),
        }
        if fixture_qpos is not None:
            transition["fixture_qpos"] = _json_value(fixture_qpos)
        if self.include_observations:
            transition["observation"] = _non_visual_observation(observation)
        episode["trajectory"].append(transition)
        if self.save_visual_observations:
            visual_frames = self._visual_frames[env_idx]
            if visual_frames is None:
                raise RuntimeError(f"Missing visual buffer for env_idx={env_idx}")
            current_frames = self._extract_visual_frames(observation)
            for key, frame in current_frames.items():
                visual_frames[key].append(frame)

        if severe_collision_termination:
            return self._finalize(env_idx, end_reason="severe_collision")
        if (
            self.mode == "official_eval"
            and emitted_termination
            and safety_violation_termination
        ):
            return self._finalize(env_idx, end_reason="safety_violation")
        if raw_task_termination:
            return self._finalize(env_idx, end_reason="raw_task_success")
        if truncated:
            return self._finalize(env_idx, end_reason="horizon")
        return None

    def close(self) -> None:
        """Persist any episode still active when the environment closes."""
        for env_idx, episode in enumerate(self._episodes):
            if episode is not None and episode["trajectory"]:
                self._finalize(env_idx, end_reason="environment_close")

    def _finalize(self, env_idx: int, end_reason: str) -> Path | None:
        episode = self._episodes[env_idx]
        if episode is None:
            return None
        trajectory = episode["trajectory"]
        if not trajectory:
            self._episodes[env_idx] = None
            self._visual_frames[env_idx] = None
            self._sim_states[env_idx] = None
            self._last_finalized_steps[env_idx] = 0
            return None
        self._last_finalized_steps[env_idx] = len(trajectory)

        first_raw_success = next(
            (
                step["timestep"]
                for step in trajectory
                if step["terminations"]["raw_task"]
            ),
            None,
        )
        first_violation = next(
            (
                step["timestep"]
                for step in trajectory
                if step["terminations"]["safety_violation"]
            ),
            None,
        )
        first_severe_collision = next(
            (
                step["timestep"]
                for step in trajectory
                if step["terminations"]["severe_collision"]
            ),
            None,
        )
        raw_success = first_raw_success is not None
        safety_violation = first_violation is not None
        severe_collision = first_severe_collision is not None
        quadrant = {
            (True, False): "Q1",
            (True, True): "Q2",
            (False, False): "Q3",
            (False, True): "Q4",
        }[(raw_success, safety_violation)]
        episode["outcome"] = {
            "success": raw_success,
            "unsafe": safety_violation,
            "raw_task_success": raw_success,
            "safety_violation": safety_violation,
            "severe_collision": severe_collision,
            "safe_task_success": raw_success and not safety_violation,
            "quadrant": quadrant,
            "first_raw_task_success_timestep": first_raw_success,
            "first_violation_timestep": first_violation,
            "first_severe_collision_timestep": first_severe_collision,
            "end_reason": end_reason,
            "num_transitions": len(trajectory),
        }
        first_violation_index = next(
            (
                index
                for index, step in enumerate(trajectory)
                if step["terminations"]["safety_violation"]
            ),
            None,
        )
        episode["success"] = raw_success
        episode["unsafe"] = safety_violation
        episode["quadrant"] = quadrant
        episode["trajectory_after_violation"] = (
            trajectory[first_violation_index:]
            if first_violation_index is not None
            else []
        )
        for key in (
            "root_episode_id",
            "branch_id",
            "alpha",
            "perturbation_range",
        ):
            if key in episode["episode"]:
                episode[key] = episode["episode"][key]

        episode_id = episode["episode"]["episode_id"]
        if self.save_visual_observations:
            visual_frames = self._visual_frames[env_idx]
            if visual_frames is None:
                raise RuntimeError(f"Missing visual buffer for episode {episode_id}")
            expected_frames = len(trajectory) + 1
            invalid_lengths = {
                key: len(frames)
                for key, frames in visual_frames.items()
                if len(frames) != expected_frames
            }
            if invalid_lengths:
                raise RuntimeError(
                    f"Visual/action alignment failed for {episode_id}: expected "
                    f"{expected_frames} frames, got {invalid_lengths}"
                )
            visual_path = self._visual_output_dir / f"{episode_id}.npz"
            temporary_visual_path = visual_path.with_suffix(".npz.tmp")
            with temporary_visual_path.open("wb") as file:
                np.savez_compressed(
                    file,
                    **{
                        key: np.stack(frames, axis=0)
                        for key, frames in visual_frames.items()
                    },
                )
            os.replace(temporary_visual_path, visual_path)
            episode["visual_trajectory"] = {
                "format": "npz",
                "path": f"visual/{visual_path.name}",
                "camera_keys": list(visual_frames),
                "num_frames": expected_frames,
                "preprocessing": "rotate_180_to_rlinf_policy_view",
                "alignment": "frame[t] + action[t] -> frame[t+1]",
            }
        if self.save_sim_state:
            self._write_sim_states(env_idx, episode, episode_id)
        output_path = self._output_dir / f"{episode_id}.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(episode, file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, output_path)

        self._episodes[env_idx] = None
        self._visual_frames[env_idx] = None
        self._sim_states[env_idx] = None
        self._episode_indices[env_idx] += 1
        return output_path

    def _write_sim_states(self, env_idx: int, episode: dict[str, Any], episode_id: str) -> None:
        """Write the per-chunk snapshots as state/<id>.npz (no pickles)."""
        states = self._sim_states[env_idx] or []
        if not states:
            raise RuntimeError(
                f"save_sim_state is on but episode {episode_id} has no snapshots; "
                "the env must call record_sim_state at every chunk boundary"
            )
        arrays: dict[str, np.ndarray] = {
            "timesteps": np.asarray([s["timestep"] for s in states], dtype=np.int32),
            "cur_time": np.asarray([s.get("cur_time", np.nan) for s in states], dtype=np.float64),
            "motion_generators": np.asarray(
                [json.dumps(_json_value(s.get("motion_generators", {}))) for s in states]
            ),
            "controller_state": np.asarray(
                [json.dumps(_json_value(s.get("controller_state", []))) for s in states]
            ),
            "gripper_state": np.asarray(
                [json.dumps(_json_value(s.get("gripper_state", []))) for s in states]
            ),
        }
        for key in ("mujoco_state", "mocap_pos", "mocap_quat", "act", "ctrl",
                    "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
            if all(key in s for s in states):
                arrays[key] = np.stack([np.asarray(s[key]) for s in states], axis=0)
        path = self._state_output_dir / f"{episode_id}.npz"
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as file:
            np.savez_compressed(file, **arrays)
        os.replace(tmp, path)
        episode["sim_state_trajectory"] = {
            "format": "npz",
            "path": f"state/{path.name}",
            "num_snapshots": len(states),
            "timesteps": arrays["timesteps"].tolist(),
            "fields": sorted(arrays),
            "semantics": "snapshot k is the state right before step timesteps[k]",
        }
        first = states[0]
        if first["timestep"] == 1 and "mocap_pos" in first:
            # Dynamic-obstacle origin as actually re-sampled at reset (P7).
            episode["mocap_start"] = {
                "mocap_pos": _json_value(first["mocap_pos"]),
                "mocap_quat": _json_value(first.get("mocap_quat")),
                "motion_generators": _json_value(first.get("motion_generators", {})),
            }

    def _extract_visual_frames(self, observation: Any) -> dict[str, np.ndarray]:
        """Extract policy-view RGB frames and fail fast on incomplete data."""
        if not isinstance(observation, dict):
            raise RuntimeError(
                "Visual episode collection requires dictionary observations"
            )
        frames: dict[str, np.ndarray] = {}
        for key in self.visual_camera_keys:
            if key not in observation:
                raise RuntimeError(
                    f"Visual camera {key!r} is missing from observation; "
                    f"available keys: {sorted(observation)}"
                )
            frame = np.asarray(observation[key])
            if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                raise RuntimeError(
                    f"Visual camera {key!r} must be HWC RGB(A), got {frame.shape}"
                )
            # LIBERO camera buffers are upside-down relative to the policy input.
            frames[key] = np.ascontiguousarray(frame[::-1, ::-1, :3]).copy()
        return frames

    def _next_episode_index(self, env_idx: int) -> int:
        prefix = f"process-{self.process_id:04d}-env-{env_idx:03d}-episode-"
        existing_indices = []
        for path in self._output_dir.glob(f"{prefix}*.json"):
            raw_index = path.stem.removeprefix(prefix)
            if raw_index.isdigit():
                existing_indices.append(int(raw_index))
        return max(existing_indices, default=-1) + 1

    def _check_env_idx(self, env_idx: int) -> None:
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"env_idx={env_idx} is outside [0, {self.num_envs})")
