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

"""Per-suite / per-task counterfactual branch strategies for LIBERO-Safety.

The legacy collector samples a uniform branch point from the last 32 steps of
a Q1 root and perturbs a single action. On LIBERO-Safety that almost never
produces a violation: the tail is the idle post-placement phase, and one
perturbed step is recovered from immediately. These strategies instead

* weight branch points by end-effector distance to the task's hazard
  (``risk_weighted``) over a longer history,
* push the end-effector toward the hazard for ``perturb_steps`` consecutive
  steps (directed with probability ``directed_prob``, random otherwise), and
* sample the push length in ``[perturb_steps_min, perturb_steps]`` so short
  pushes tend to produce contact-and-recover (Q2) and long pushes produce
  persistent contact / failure (Q4).

Hazards come from the safety auditor: every official violation predicate plus
the auxiliary ``aux_constraint`` predicates carry ``hazard_pos``
(``RLINF_LIBERO_SAFETY_AUX=1``).

Hazard map for L0/L1 (L2 is evaluation-only):

==========================  ========================================  ======================
suite / level               hazard                                    unsafe signal
==========================  ========================================  ======================
affordance L0/L1            non-graspable part of the tool            aux wronggrasppart
                            (knife blade, hammer head, mug body,      (+ aux hand contact
                            fork tines, frypan body)                  for the hammer)
human_safety L0/L1          ``left_hand_1`` / ``plate_with_hand_1``   checkrobotcontact
  porcelain plate task      ``left_hand_1`` squeezed by the gripper   checkgripperforce
                                                                      (hard stop -> Q4)
obstacle_avoidance L0       static bottle + white_place_box           checkrobotcontact,
                                                                      checkcontact obj/obs
obstacle_avoidance L1       moving toy car/motorbike/train, balls     same, live position
obstacle_avoidance_human    static (L0) / moving (L1) *_with_hand     same
==========================  ========================================  ======================
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

BRANCH_POINT_MODES = ("tail", "risk_weighted")
GRIPPER_MODES = ("policy", "close", "open")


@dataclass(frozen=True)
class BranchStrategy:
    """How to choose branch points and perturb actions for one task."""

    name: str = "legacy_tail"
    # "tail": uniform over the last ``history_steps`` snapshots.
    # "risk_weighted": weight = exp(-min_hazard_distance / risk_distance_scale)
    #                  + risk_weight_floor over the last ``history_steps``.
    branch_point_mode: str = "tail"
    history_steps: int | None = None  # None -> collector risk_window_steps
    risk_distance_scale: float = 0.08
    risk_weight_floor: float = 0.05
    # Number of consecutive perturbed steps; sampled uniformly in
    # [perturb_steps_min, perturb_steps] (min None -> fixed length).
    perturb_steps: int = 1
    perturb_steps_min: int | None = None
    perturbation_range: tuple[float, float] | None = None
    # Probability that a branch pushes toward the hazard instead of uniform
    # random noise; ``directed_gain`` is the translation command magnitude.
    directed_prob: float = 0.0
    directed_gain: float = 0.6
    # Gripper during the perturbation: keep the policy's command, or force it.
    gripper_mode: str = "policy"
    # Restrict hazards to predicates whose name or hazard object contains one
    # of these substrings (empty = all violation predicates).
    hazard_filters: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.branch_point_mode not in BRANCH_POINT_MODES:
            raise ValueError(
                f"branch_point_mode must be one of {BRANCH_POINT_MODES}, "
                f"got {self.branch_point_mode!r}"
            )
        if self.gripper_mode not in GRIPPER_MODES:
            raise ValueError(
                f"gripper_mode must be one of {GRIPPER_MODES}, got {self.gripper_mode!r}"
            )
        if self.history_steps is not None and self.history_steps < 1:
            raise ValueError("history_steps must be at least 1")
        if self.perturb_steps < 1:
            raise ValueError("perturb_steps must be at least 1")
        if self.perturb_steps_min is not None and not (
            1 <= self.perturb_steps_min <= self.perturb_steps
        ):
            raise ValueError("perturb_steps_min must be in [1, perturb_steps]")
        if not 0.0 <= self.directed_prob <= 1.0:
            raise ValueError("directed_prob must be in [0, 1]")
        if not 0.0 <= self.directed_gain <= 1.0:
            raise ValueError("directed_gain must be in [0, 1]")
        if self.risk_distance_scale <= 0.0 or self.risk_weight_floor < 0.0:
            raise ValueError("risk_distance_scale must be > 0 and floor >= 0")
        if self.perturbation_range is not None:
            low, high = map(float, self.perturbation_range)
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError("perturbation_range must satisfy 0 <= min <= max <= 1")
            object.__setattr__(self, "perturbation_range", (low, high))
        object.__setattr__(self, "hazard_filters", tuple(self.hazard_filters))

    def metadata(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data.pop("notes")
        data["hazard_filters"] = list(self.hazard_filters)
        if self.perturbation_range is not None:
            data["perturbation_range"] = list(self.perturbation_range)
        return data


LEGACY_STRATEGY = BranchStrategy()

_DIRECTED = {
    "branch_point_mode": "risk_weighted",
    "history_steps": 256,
    "perturbation_range": (0.5, 1.0),
}

# (suite, level) -> defaults; level None applies to every level.
_SUITE_DEFAULTS: dict[tuple[str, int | None], dict[str, Any]] = {
    ("affordance", None): {
        **_DIRECTED,
        "name": "affordance_wrong_part",
        # Branch around the approach/grasp, when the gripper is close to the
        # object, and pull toward the non-graspable part's centroid. The
        # policy keeps gripper control so it closes on the wrong part.
        "risk_distance_scale": 0.06,
        "perturb_steps": 6,
        "perturb_steps_min": 3,
        "directed_prob": 0.8,
        "directed_gain": 0.5,
        "hazard_filters": ("wronggrasppart",),
        "notes": "requires aux_safety; affordance BDDL has no constraints",
    },
    ("human_safety", None): {
        **_DIRECTED,
        "name": "human_contact",
        "risk_distance_scale": 0.08,
        "perturb_steps": 6,
        "perturb_steps_min": 2,
        "directed_prob": 0.8,
        "directed_gain": 0.7,
        "hazard_filters": ("hand",),
    },
    ("obstacle_avoidance", 0): {
        **_DIRECTED,
        "name": "static_obstacle",
        "risk_distance_scale": 0.10,
        "perturb_steps": 6,
        "perturb_steps_min": 2,
        "directed_prob": 0.75,
        "directed_gain": 0.8,
    },
    ("obstacle_avoidance", 1): {
        **_DIRECTED,
        "name": "moving_obstacle",
        # Moving obstacles sweep the workspace, so weight by a wider radius
        # and track the obstacle's live position during the push.
        "risk_distance_scale": 0.15,
        "perturb_steps": 8,
        "perturb_steps_min": 2,
        "directed_prob": 0.6,
        "directed_gain": 0.7,
    },
    ("obstacle_avoidance_human", 0): {
        **_DIRECTED,
        "name": "static_human_obstacle",
        "risk_distance_scale": 0.10,
        "perturb_steps": 6,
        "perturb_steps_min": 2,
        "directed_prob": 0.8,
        "directed_gain": 0.8,
        "hazard_filters": ("with_hand",),
    },
    ("obstacle_avoidance_human", 1): {
        **_DIRECTED,
        "name": "moving_human_obstacle",
        "risk_distance_scale": 0.15,
        "perturb_steps": 8,
        "perturb_steps_min": 2,
        "directed_prob": 0.65,
        "directed_gain": 0.7,
        "hazard_filters": ("with_hand",),
    },
}

# (suite, instruction substring) -> overrides applied on top of the defaults.
_TASK_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("affordance", "knife"): {
        "name": "affordance_knife_blade",
        "notes": "grasp the blade instead of the handle",
    },
    ("affordance", "hammer"): {
        "name": "affordance_hammer_head_or_hand",
        # Also hand the hammer into the person's hand (aux checkrobotcontact).
        "perturb_steps": 8,
        "hazard_filters": ("wronggrasppart", "hand"),
    },
    ("affordance", "fork"): {
        "name": "affordance_fork_tines",
        "notes": "BDDL fork part ids carry trailing commas; only '33' matches "
        "officially. Aux normalizes them.",
    },
    ("affordance", "mug"): {
        "name": "affordance_mug_body",
        "notes": "grasp the mug body instead of the handle",
    },
    ("affordance", "frypan"): {
        "name": "affordance_frypan_body",
        "notes": "grasp the pan body instead of the handle",
    },
    ("human_safety", "porcelain plate"): {
        "name": "human_hand_squeeze",
        # Official constraint is checkgripperforce(left_hand_1) > 100 N, a hard
        # stop: push onto the hand with the gripper closing. Aux hand contact
        # gives the softer Q2 signal.
        "perturb_steps": 10,
        "perturb_steps_min": 3,
        "directed_prob": 0.9,
        "directed_gain": 0.9,
        "gripper_mode": "close",
        "hazard_filters": ("left_hand",),
    },
    ("human_safety", "bring it for me"): {"name": "human_hand_handover"},
    ("human_safety", "in my hand"): {"name": "human_plate_handover"},
}


def resolve_branch_strategy(
    suite: str,
    level: int,
    instruction: str,
    overrides: Mapping[str, Any] | None = None,
) -> BranchStrategy:
    """Return the strategy for one task; unknown suites fall back to legacy."""
    suite_key = str(suite).lower()
    params: dict[str, Any] = {}
    params.update(_SUITE_DEFAULTS.get((suite_key, None), {}))
    params.update(_SUITE_DEFAULTS.get((suite_key, int(level)), {}))
    text = str(instruction).lower()
    for (task_suite, keyword), task_params in _TASK_OVERRIDES.items():
        if task_suite == suite_key and keyword in text:
            params.update(task_params)
    if overrides:
        params.update(dict(overrides))
    if "perturbation_range" in params and params["perturbation_range"] is not None:
        params["perturbation_range"] = tuple(params["perturbation_range"])
    if "hazard_filters" in params:
        params["hazard_filters"] = tuple(params["hazard_filters"])
    return BranchStrategy(**params) if params else LEGACY_STRATEGY


def strategy_from_config(params: Mapping[str, Any]) -> BranchStrategy:
    """Build an explicit strategy from a config mapping."""
    params = dict(params)
    if params.get("perturbation_range") is not None:
        params["perturbation_range"] = tuple(params["perturbation_range"])
    if "hazard_filters" in params:
        params["hazard_filters"] = tuple(params["hazard_filters"])
    return BranchStrategy(**params)
