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

"""Counterfactual branch search for safe LIBERO-Safety episodes."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from rlinf.envs.sim.libero.counterfactual_strategy import (
    LEGACY_STRATEGY,
    BranchStrategy,
)

TARGET_QUADRANTS = frozenset({"Q2", "Q4"})
ROOT_QUADRANTS = frozenset({"Q1", "Q3"})
# A perturbation cannot turn a failed root into a success it never reached, so
# Q3 roots only search for unsafe failures.
Q3_ROOT_TARGETS = frozenset({"Q4"})

Hazards = tuple[tuple[str, np.ndarray], ...]


@dataclass(frozen=True)
class RiskSnapshot:
    """Simulator state and policy action immediately before one root step."""

    timestep: int
    simulator_state: Any
    action: np.ndarray
    # (hazard key, world position) of violation predicates; empty when the
    # auditor does not publish hazard positions.
    hazards: Hazards = ()
    eef_pos: np.ndarray | None = None


@dataclass(frozen=True)
class CounterfactualBranch:
    """One reproducible counterfactual branch specification."""

    root_episode_id: str
    branch_id: str
    root_timestep: int
    simulator_state: Any
    alpha: float
    perturbation_range: tuple[float, float]
    original_action: np.ndarray
    perturbed_action: np.ndarray
    perturbation: np.ndarray
    strategy: BranchStrategy = LEGACY_STRATEGY
    perturb_steps: int = 1
    directed: bool = False
    hazard_key: str | None = None
    hazard_pos: np.ndarray | None = None
    hazards: Hazards = ()
    root_quadrant: str = "Q1"

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible branch metadata for the episode auditor."""
        return {
            "root_episode_id": self.root_episode_id,
            "root_quadrant": self.root_quadrant,
            "branch_id": self.branch_id,
            "branch_timestep": self.root_timestep,
            "alpha": self.alpha,
            "perturbation_range": list(self.perturbation_range),
            "original_action": self.original_action.tolist(),
            "perturbed_action": self.perturbed_action.tolist(),
            "perturbation": self.perturbation.tolist(),
            "branch_strategy": self.strategy.name,
            "branch_strategy_params": self.strategy.metadata(),
            "perturb_steps": self.perturb_steps,
            "directed": self.directed,
            "hazard_key": self.hazard_key,
            "hazard_pos": None
            if self.hazard_pos is None
            else np.asarray(self.hazard_pos).tolist(),
        }


@dataclass
class _PerturbationPlan:
    """Progress of a multi-step branch perturbation."""

    branch: CounterfactualBranch
    steps_done: int = 1


class CounterfactualBranchCollector:
    """Search Q2 and Q4 branches from a Q1 (or optionally Q3) root trajectory.

    ``root_quadrants`` selects which root outcomes start a search. A Q3 root
    (horizon timeout, no success, no violation) can only produce Q3/Q4 under a
    perturbation, so its search targets ``target_quadrants & {Q4}``. This is
    what lets tasks without any Q1 root (e.g. the affordance fork task) still
    yield unsafe branches.

    The collector owns only branch selection and perturbation state. The caller
    remains responsible for saving/restoring the simulator, executing actions,
    rolling out the policy, and writing episode audit files.

    With the default (legacy) strategy a branch is one uniformly random action
    perturbation at a uniform point of the last ``risk_window_steps`` steps.
    Per-environment :class:`BranchStrategy` objects (``counterfactual_strategy``)
    enable risk-weighted branch points and multi-step perturbations aimed at
    the task's hazard.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        process_id: int,
        risk_window_steps: int = 32,
        max_branches_per_root: int = 16,
        perturbation_range: tuple[float, float] = (0.15, 1.0),
        action_dimensions: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        target_quadrants: frozenset[str] = TARGET_QUADRANTS,
        strategy: BranchStrategy = LEGACY_STRATEGY,
        root_quadrants: frozenset[str] = frozenset({"Q1"}),
    ) -> None:
        alpha_min, alpha_max = map(float, perturbation_range)
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if risk_window_steps < 1:
            raise ValueError("risk_window_steps must be at least 1")
        if max_branches_per_root < 1:
            raise ValueError("max_branches_per_root must be at least 1")
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError("perturbation_range must satisfy 0 <= min <= max <= 1")
        if not action_dimensions:
            raise ValueError("action_dimensions must not be empty")
        if not target_quadrants or not target_quadrants <= TARGET_QUADRANTS:
            raise ValueError("target_quadrants must be a non-empty subset of Q2/Q4")
        root_quadrants = frozenset(str(value).upper() for value in root_quadrants)
        if not root_quadrants or not root_quadrants <= ROOT_QUADRANTS:
            raise ValueError("root_quadrants must be a non-empty subset of Q1/Q3")
        if "Q3" in root_quadrants and not (target_quadrants & Q3_ROOT_TARGETS):
            raise ValueError("Q3 roots require Q4 in target_quadrants")

        self.num_envs = int(num_envs)
        self.risk_window_steps = int(risk_window_steps)
        self.max_branches_per_root = int(max_branches_per_root)
        self.perturbation_range = (alpha_min, alpha_max)
        self.action_dimensions = tuple(int(index) for index in action_dimensions)
        self.target_quadrants = frozenset(target_quadrants)
        self.root_quadrants = root_quadrants
        seed_sequence = np.random.SeedSequence([int(seed), int(process_id)])
        self._rngs = [
            np.random.default_rng(child) for child in seed_sequence.spawn(self.num_envs)
        ]
        self._strategies = [strategy] * self.num_envs
        self._histories = [
            deque(maxlen=self._history_length(strategy)) for _ in range(self.num_envs)
        ]
        self._phases = ["root"] * self.num_envs
        self._root_episode_ids: list[str | None] = [None] * self.num_envs
        self._branch_counts = np.zeros(self.num_envs, dtype=np.int32)
        self._found_quadrants = [set() for _ in range(self.num_envs)]
        self._active_branches: list[CounterfactualBranch | None] = [
            None
        ] * self.num_envs
        self._pending_perturbed_actions: list[np.ndarray | None] = [
            None
        ] * self.num_envs
        self._plans: list[_PerturbationPlan | None] = [None] * self.num_envs
        self._root_quadrants: list[str] = ["Q1"] * self.num_envs
        self._search_targets = [self.target_quadrants] * self.num_envs

    def accepts_root(self, quadrant: str) -> bool:
        """Return whether a root ending in ``quadrant`` starts a search."""
        return str(quadrant).upper() in self.root_quadrants

    def _history_length(self, strategy: BranchStrategy) -> int:
        if strategy.history_steps is None:
            return self.risk_window_steps
        return int(strategy.history_steps)

    def set_strategy(self, env_idx: int, strategy: BranchStrategy) -> None:
        """Bind a strategy to an environment right after ``reset_root``."""
        self._check_env_idx(env_idx)
        if not self.is_collecting_root(env_idx) or self._histories[env_idx]:
            raise RuntimeError("set_strategy must be called right after reset_root")
        self._strategies[env_idx] = strategy
        length = self._history_length(strategy)
        if self._histories[env_idx].maxlen != length:
            self._histories[env_idx] = deque(maxlen=length)

    def strategy(self, env_idx: int) -> BranchStrategy:
        """Return the strategy bound to an environment."""
        self._check_env_idx(env_idx)
        return self._strategies[env_idx]

    def reset_root(self, env_idx: int) -> None:
        """Start collecting snapshots for a new candidate root episode."""
        self._check_env_idx(env_idx)
        self._histories[env_idx].clear()
        self._phases[env_idx] = "root"
        self._root_episode_ids[env_idx] = None
        self._branch_counts[env_idx] = 0
        self._found_quadrants[env_idx].clear()
        self._active_branches[env_idx] = None
        self._pending_perturbed_actions[env_idx] = None
        self._plans[env_idx] = None
        self._root_quadrants[env_idx] = "Q1"
        self._search_targets[env_idx] = self.target_quadrants

    def set_root_episode_id(self, env_idx: int, episode_id: str) -> None:
        """Bind the audit record created at root reset to this search."""
        self._check_env_idx(env_idx)
        if not self.is_collecting_root(env_idx):
            raise RuntimeError(
                f"cannot set a root episode id while env {env_idx} is not in root phase"
            )
        if not episode_id:
            raise ValueError("root episode id must be non-empty")
        self._root_episode_ids[env_idx] = str(episode_id)

    def root_episode_id(self, env_idx: int) -> str | None:
        """Return the audit id bound when the current root episode began."""
        self._check_env_idx(env_idx)
        return self._root_episode_ids[env_idx]

    def is_collecting_root(self, env_idx: int) -> bool:
        """Return whether the environment is recording a root trajectory."""
        self._check_env_idx(env_idx)
        return self._phases[env_idx] == "root"

    def is_running_branch(self, env_idx: int) -> bool:
        """Return whether the environment is currently rolling out a branch."""
        self._check_env_idx(env_idx)
        return self._phases[env_idx] == "branch"

    def record_root_step(
        self,
        env_idx: int,
        *,
        timestep: int,
        simulator_state: Any,
        action: np.ndarray,
        hazards: Any = (),
        eef_pos: np.ndarray | None = None,
    ) -> None:
        """Record a bounded pre-action snapshot while the root is active."""
        self._check_env_idx(env_idx)
        if not self.is_collecting_root(env_idx):
            return
        self._histories[env_idx].append(
            RiskSnapshot(
                timestep=int(timestep),
                simulator_state=copy.deepcopy(simulator_state),
                action=np.asarray(action, dtype=np.float64).copy(),
                hazards=normalize_hazards(hazards),
                eef_pos=None
                if eef_pos is None
                else np.asarray(eef_pos, dtype=np.float64).copy(),
            )
        )

    def begin_search(
        self,
        env_idx: int,
        *,
        root_episode_id: str | None = None,
        root_quadrant: str = "Q1",
    ) -> CounterfactualBranch | None:
        """Begin branching after the recorded root ended in an accepted quadrant."""
        self._check_env_idx(env_idx)
        root_quadrant = str(root_quadrant).upper()
        if root_quadrant not in self.root_quadrants:
            raise ValueError(
                f"root quadrant {root_quadrant!r} is not in {sorted(self.root_quadrants)}"
            )
        self._root_quadrants[env_idx] = root_quadrant
        self._search_targets[env_idx] = (
            self.target_quadrants & Q3_ROOT_TARGETS
            if root_quadrant == "Q3"
            else self.target_quadrants
        )
        if not self._histories[env_idx]:
            self._phases[env_idx] = "finished"
            return None
        if root_episode_id is not None:
            if self._root_episode_ids[env_idx] not in (None, str(root_episode_id)):
                raise RuntimeError("root episode id changed during counterfactual search")
            self._root_episode_ids[env_idx] = str(root_episode_id)
        if self._root_episode_ids[env_idx] is None:
            raise RuntimeError("root episode id was not bound at root reset")
        self._phases[env_idx] = "branch"
        self._branch_counts[env_idx] = 0
        self._found_quadrants[env_idx].clear()
        return self._sample_branch(env_idx)

    def consume_perturbed_action(
        self,
        env_idx: int,
        *,
        policy_action: np.ndarray | None = None,
        eef_pos: np.ndarray | None = None,
        hazards: Any = None,
    ) -> np.ndarray | None:
        """Return the next perturbed action of the active branch, if any.

        The first call returns the branch's precomputed action. Multi-step
        strategies then keep perturbing the live ``policy_action`` until
        ``perturb_steps`` actions were replaced; directed branches re-aim at the
        hazard's current position (``hazards``) from ``eef_pos``.
        """
        self._check_env_idx(env_idx)
        action = self._pending_perturbed_actions[env_idx]
        self._pending_perturbed_actions[env_idx] = None
        if action is not None:
            return action.copy()
        plan = self._plans[env_idx]
        if plan is None or policy_action is None:
            return None
        branch = plan.branch
        hazard_pos = branch.hazard_pos
        if hazards is not None and branch.hazard_key is not None:
            hazard_pos = dict(normalize_hazards(hazards)).get(
                branch.hazard_key, hazard_pos
            )
        perturbed = self._perturb(
            env_idx,
            np.asarray(policy_action, dtype=np.float64),
            alpha=branch.alpha,
            strategy=branch.strategy,
            directed=branch.directed,
            hazard_pos=hazard_pos,
            eef_pos=eef_pos,
        )
        plan.steps_done += 1
        if plan.steps_done >= branch.perturb_steps:
            self._plans[env_idx] = None
        return perturbed

    def complete_branch(
        self, env_idx: int, *, quadrant: str
    ) -> CounterfactualBranch | None:
        """Record an outcome and return the next branch, if search continues."""
        self._check_env_idx(env_idx)
        if not self.is_running_branch(env_idx):
            raise RuntimeError(f"env {env_idx} has no active counterfactual search")
        if quadrant not in {"Q1", "Q2", "Q3", "Q4"}:
            raise ValueError(f"invalid branch quadrant {quadrant!r}")
        if quadrant in self.target_quadrants:
            self._found_quadrants[env_idx].add(quadrant)
        self._active_branches[env_idx] = None
        self._pending_perturbed_actions[env_idx] = None
        self._plans[env_idx] = None
        if (
            self._found_quadrants[env_idx] >= self._search_targets[env_idx]
            or self._branch_counts[env_idx] >= self.max_branches_per_root
        ):
            self._phases[env_idx] = "finished"
            return None
        return self._sample_branch(env_idx)

    def active_branch(self, env_idx: int) -> CounterfactualBranch | None:
        """Return the active branch specification without consuming it."""
        self._check_env_idx(env_idx)
        return self._active_branches[env_idx]

    @staticmethod
    def _branch_weights(
        history: deque, strategy: BranchStrategy
    ) -> np.ndarray | None:
        """Branch-point probabilities for ``risk_weighted`` (None = uniform)."""
        if strategy.branch_point_mode != "risk_weighted":
            return None
        distances = []
        for snapshot in history:
            hazards = filter_hazards(snapshot.hazards, strategy)
            if snapshot.eef_pos is None or not hazards:
                distances.append(np.inf)
                continue
            distances.append(
                min(
                    float(np.linalg.norm(position - snapshot.eef_pos))
                    for _, position in hazards
                )
            )
        distances = np.asarray(distances, dtype=np.float64)
        if not np.isfinite(distances).any():
            return None
        weights = np.where(
            np.isfinite(distances),
            np.exp(-distances / strategy.risk_distance_scale),
            0.0,
        )
        weights = weights + strategy.risk_weight_floor
        return weights / weights.sum()

    def _perturb(
        self,
        env_idx: int,
        original: np.ndarray,
        *,
        alpha: float,
        strategy: BranchStrategy,
        directed: bool,
        hazard_pos: np.ndarray | None,
        eef_pos: np.ndarray | None,
    ) -> np.ndarray:
        """Mix ``original`` with random or hazard-directed commands."""
        rng = self._rngs[env_idx]
        random_action = rng.uniform(-1.0, 1.0, size=original.shape)
        target = random_action.copy()
        if directed and hazard_pos is not None and eef_pos is not None:
            offset = np.asarray(hazard_pos, dtype=np.float64) - np.asarray(
                eef_pos, dtype=np.float64
            )
            norm = float(np.linalg.norm(offset))
            if norm > 1e-6:
                # OSC delta translation is in the robot base frame, which is
                # axis-aligned with the world frame in LIBERO scenes.
                gain = strategy.directed_gain
                target[:3] = gain * offset / norm + (1.0 - gain) * random_action[:3]
        perturbed = original.copy()
        for action_index in self.action_dimensions:
            if not 0 <= action_index < original.size:
                raise ValueError(
                    f"action dimension {action_index} outside action size {original.size}"
                )
            perturbed[action_index] = (1.0 - alpha) * original[
                action_index
            ] + alpha * target[action_index]
        if strategy.gripper_mode == "close":
            perturbed[-1] = 1.0
        elif strategy.gripper_mode == "open":
            perturbed[-1] = -1.0
        return np.clip(perturbed, -1.0, 1.0)

    def _sample_branch(self, env_idx: int) -> CounterfactualBranch:
        rng = self._rngs[env_idx]
        strategy = self._strategies[env_idx]
        history = self._histories[env_idx]
        weights = self._branch_weights(history, strategy)
        if weights is None:
            snapshot = history[int(rng.integers(0, len(history)))]
        else:
            snapshot = history[int(rng.choice(len(history), p=weights))]
        perturbation_range = strategy.perturbation_range or self.perturbation_range
        alpha = float(rng.uniform(*perturbation_range))
        hazard_key, hazard_pos = None, None
        hazards = filter_hazards(snapshot.hazards, strategy)
        if hazards and snapshot.eef_pos is not None:
            hazard_key, hazard_pos = min(
                hazards,
                key=lambda item: float(np.linalg.norm(item[1] - snapshot.eef_pos)),
            )
        directed = False
        perturb_steps = strategy.perturb_steps
        # The legacy strategy keeps the original RNG call sequence exactly.
        if strategy != LEGACY_STRATEGY:
            directed = hazard_pos is not None and bool(
                rng.random() < strategy.directed_prob
            )
            if strategy.perturb_steps_min is not None:
                perturb_steps = int(
                    rng.integers(strategy.perturb_steps_min, strategy.perturb_steps + 1)
                )
        original = snapshot.action.copy()
        perturbed = self._perturb(
            env_idx,
            original,
            alpha=alpha,
            strategy=strategy,
            directed=directed,
            hazard_pos=hazard_pos,
            eef_pos=snapshot.eef_pos,
        )
        branch_number = int(self._branch_counts[env_idx])
        self._branch_counts[env_idx] += 1
        root_episode_id = self._root_episode_ids[env_idx]
        if root_episode_id is None:
            raise RuntimeError("root episode id is missing")
        branch = CounterfactualBranch(
            root_episode_id=root_episode_id,
            branch_id=f"{root_episode_id}-branch-{branch_number:03d}",
            root_timestep=snapshot.timestep,
            simulator_state=copy.deepcopy(snapshot.simulator_state),
            alpha=alpha,
            perturbation_range=perturbation_range,
            original_action=original,
            perturbed_action=perturbed,
            perturbation=perturbed - original,
            strategy=strategy,
            perturb_steps=perturb_steps,
            directed=directed,
            hazard_key=hazard_key,
            hazard_pos=None if hazard_pos is None else hazard_pos.copy(),
            hazards=snapshot.hazards,
            root_quadrant=self._root_quadrants[env_idx],
        )
        self._active_branches[env_idx] = branch
        self._pending_perturbed_actions[env_idx] = perturbed.copy()
        self._plans[env_idx] = (
            _PerturbationPlan(branch=branch) if perturb_steps > 1 else None
        )
        return branch

    def _check_env_idx(self, env_idx: int) -> None:
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"env_idx={env_idx} is outside [0, {self.num_envs})")


def normalize_hazards(hazards: Any) -> Hazards:
    """Accept a mapping or (key, position) pairs; drop entries without positions."""
    if not hazards:
        return ()
    items = hazards.items() if isinstance(hazards, dict) else hazards
    normalized = []
    for key, position in items:
        if position is None:
            continue
        array = np.asarray(position, dtype=np.float64).reshape(-1)
        if array.size == 3 and np.isfinite(array).all():
            normalized.append((str(key), array.copy()))
    return tuple(normalized)


def filter_hazards(hazards: Hazards, strategy: BranchStrategy) -> Hazards:
    """Keep hazards whose key contains one of the strategy's filter tokens."""
    if not strategy.hazard_filters:
        return hazards
    tokens = tuple(token.lower() for token in strategy.hazard_filters)
    return tuple(
        (key, position)
        for key, position in hazards
        if any(token in key.lower() for token in tokens)
    )


def hazards_from_predicates(predicates: list[dict[str, Any]] | None) -> Hazards:
    """Extract ``("<instance_id>|<predicate>|<object>", hazard_pos)`` pairs."""
    hazards = []
    for predicate in predicates or []:
        if predicate.get("semantics") != "violation":
            continue
        position = predicate.get("hazard_pos")
        if position is None:
            continue
        key = "|".join(
            (
                str(predicate.get("instance_id", "")),
                str(predicate.get("predicate", "")),
                str(predicate.get("hazard_object", "")),
            )
        )
        hazards.append((key, position))
    return normalize_hazards(hazards)
