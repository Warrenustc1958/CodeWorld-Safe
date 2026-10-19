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

"""Constrained counterfactual programs (``cws_cf_program/v0``) for LIBERO-Safety.

A Code Agent writes one JSON program per counterfactual idea; this module only
parses and validates it, and samples its parameters. Nothing here executes
code from the program. The executor (``cf_executor``) interprets the result.

v0 scope (anything else is rejected loudly, never ignored):

* ``branch.trigger.phase`` must be ``any``; ``cond`` is ``<metric> <op> <num>``
  with metric ``min_dist(ee,hazard)`` or ``step``.
* ``events`` ops: ``servo_toward``, ``push``, ``release_early``,
  ``hold_gripper``, ``noise``; control then returns to the policy.
* ``scene.mutations`` ops ``shift`` / ``rotate`` on free-joint objects, bounded
  by 8 cm and 30 deg.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

SCHEMA_ID = "cws_cf_program/v0"
SUITES = (
    "affordance",
    "human_safety",
    "obstacle_avoidance",
    "obstacle_avoidance_human",
)
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
ROOT_QUADRANTS = ("Q1", "Q3")
EVENT_OPS = ("servo_toward", "push", "release_early", "hold_gripper", "noise")
GRIPPER_MODES = ("policy", "close", "open")
TRIGGER_METRICS = ("min_dist(ee,hazard)", "step")
TRIGGER_OPS = ("<", "<=", ">", ">=")
MAX_SHIFT_M = 0.08
MAX_ROTATE_DEG = 30.0
_COND_RE = re.compile(r"^\s*([a-z_]+(?:\([a-z_,\s]+\))?)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
_PROGRAM_ID_RE = re.compile(r"^[a-z0-9_]+$")


class ProgramError(ValueError):
    """A program that does not satisfy the v0 contract."""


@dataclass(frozen=True)
class ParamDist:
    dist: str
    low: float | None = None
    high: float | None = None
    values: tuple[Any, ...] = ()
    integer: bool = False

    def sample(self, rng: np.random.Generator) -> Any:
        if self.dist == "choice":
            return self.values[int(rng.integers(0, len(self.values)))]
        if self.integer:
            return int(rng.integers(int(self.low), int(self.high) + 1))
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Trigger:
    metric: str
    op: str
    threshold: float
    hazard: str | None
    fallback: str  # "risk_weighted" | "skip"

    def holds(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        return {
            "<": value < self.threshold,
            "<=": value <= self.threshold,
            ">": value > self.threshold,
            ">=": value >= self.threshold,
        }[self.op]


@dataclass(frozen=True)
class Event:
    op: str
    # Literal value, a [low, high] integer range, or "$name" into params.
    steps: Any = 1
    gain: Any = 0.8
    target: str = "hazard"
    dxyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper: str = "policy"


@dataclass(frozen=True)
class Mutation:
    op: str
    object: str
    dxyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dyaw_deg: float = 0.0


@dataclass(frozen=True)
class CFProgram:
    program_id: str
    suite: str
    level: str
    task_id: int
    target_quadrant: str
    root_quadrants: frozenset[str]
    trigger: Trigger
    events: tuple[Event, ...]
    mutations: tuple[Mutation, ...] = ()
    params: Mapping[str, ParamDist] = field(default_factory=dict)
    extra_predicates: tuple[str, ...] = ()
    rationale: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def cell_key(self) -> str:
        return f"{self.suite}_{self.level}_t{self.task_id}"

    @property
    def scene_mutated(self) -> bool:
        return bool(self.mutations)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ProgramError(msg)


def _vec3(value: Any, what: str) -> tuple[float, float, float]:
    _require(
        isinstance(value, (list, tuple)) and len(value) == 3,
        f"{what} must be a list of 3 numbers",
    )
    out = tuple(float(v) for v in value)
    _require(all(math.isfinite(v) for v in out), f"{what} must be finite")
    return out  # type: ignore[return-value]


def _parse_params(raw: Mapping[str, Any]) -> dict[str, ParamDist]:
    params: dict[str, ParamDist] = {}
    for name, spec in raw.items():
        _require(isinstance(spec, Mapping), f"params.{name} must be an object")
        dist = spec.get("dist")
        if dist == "uniform":
            low, high = spec.get("low"), spec.get("high")
            _require(
                isinstance(low, (int, float)) and isinstance(high, (int, float)) and low <= high,
                f"params.{name}: uniform needs numeric low <= high",
            )
            integer = isinstance(low, int) and isinstance(high, int)
            params[name] = ParamDist("uniform", float(low), float(high), integer=integer)
        elif dist == "choice":
            values = spec.get("values")
            _require(
                isinstance(values, list) and values, f"params.{name}: choice needs values"
            )
            params[name] = ParamDist("choice", values=tuple(values))
        else:
            raise ProgramError(f"params.{name}: dist must be uniform or choice, got {dist!r}")
    return params


def _check_ref(value: Any, params: Mapping[str, ParamDist], what: str) -> None:
    if isinstance(value, str):
        _require(value.startswith("$") and value[1:] in params, f"{what}: unknown param {value!r}")


def _hazard_objects(task: Mapping[str, Any]) -> set[str]:
    return {str(h["object"]) for h in task.get("hazards", []) if h.get("object")}


def load_program(obj: Mapping[str, Any], task: Mapping[str, Any]) -> CFProgram:
    """Validate a program against v0 and one task knowledge file (tasks/<key>.json)."""
    _require(isinstance(obj, Mapping), "program must be a JSON object")
    pid = obj.get("program_id")
    _require(isinstance(pid, str) and bool(_PROGRAM_ID_RE.match(pid)), "program_id must match ^[a-z0-9_]+$")

    cell = obj.get("cell")
    _require(isinstance(cell, Mapping), "cell is required")
    suite, level, task_id = cell.get("suite"), cell.get("level"), cell.get("task_id")
    _require(suite in SUITES, f"cell.suite must be one of {SUITES}")
    _require(level in ("L0", "L1"), "cell.level must be L0 or L1")
    _require(isinstance(task_id, int) and 0 <= task_id <= 9, "cell.task_id must be 0-9")
    _require(
        (task.get("suite"), task.get("level"), task.get("task_id")) == (suite, level, task_id),
        f"task knowledge {task.get('key')} does not match cell {suite}_{level}_t{task_id}",
    )

    tq = obj.get("target_quadrant")
    _require(tq in QUADRANTS, f"target_quadrant must be one of {QUADRANTS}")
    params = _parse_params(obj.get("params") or {})

    branch = obj.get("branch")
    _require(isinstance(branch, Mapping), "branch is required")
    roots = branch.get("root_quadrants") or []
    _require(
        isinstance(roots, list) and roots and all(q in ROOT_QUADRANTS for q in roots),
        "branch.root_quadrants must be a non-empty subset of [Q1, Q3]",
    )
    _require(branch.get("handback", "policy") == "policy", "branch.handback must be policy")

    hazards = _hazard_objects(task)
    trig = branch.get("trigger") or {}
    _require(isinstance(trig, Mapping), "branch.trigger must be an object")
    phase = trig.get("phase", "any")
    _require(phase == "any", f"trigger.phase {phase!r} is not implemented in v0 (use 'any')")
    match = _COND_RE.match(str(trig.get("cond", "")))
    _require(match is not None, "trigger.cond must be '<metric> <op> <number>'")
    metric = match.group(1).replace(" ", "")
    _require(metric in TRIGGER_METRICS, f"trigger metric {metric!r} not in {TRIGGER_METRICS}")
    hazard = trig.get("hazard")
    if metric.endswith("hazard)"):
        _require(hazard in hazards, f"trigger.hazard {hazard!r} not in task hazards {sorted(hazards)}")
    fallback = trig.get("fallback", "skip")
    _require(fallback in ("risk_weighted", "skip"), "trigger.fallback must be risk_weighted or skip")
    trigger = Trigger(metric, match.group(2), float(match.group(3)), hazard, fallback)

    raw_events = branch.get("events") or []
    _require(isinstance(raw_events, list) and raw_events, "branch.events must be a non-empty list")
    events = []
    for i, ev in enumerate(raw_events):
        what = f"events[{i}]"
        _require(isinstance(ev, Mapping), f"{what} must be an object")
        op = ev.get("op")
        _require(op in EVENT_OPS, f"{what}.op must be one of {EVENT_OPS}")
        steps = ev.get("steps", 1)
        _check_ref(steps, params, f"{what}.steps")
        if isinstance(steps, list):
            _require(
                len(steps) == 2 and all(isinstance(s, int) for s in steps) and 1 <= steps[0] <= steps[1],
                f"{what}.steps range must be [lo, hi] with 1 <= lo <= hi",
            )
            steps = tuple(steps)
        elif isinstance(steps, int):
            _require(steps >= 1, f"{what}.steps must be >= 1")
        gain = ev.get("gain", 0.8)
        _check_ref(gain, params, f"{what}.gain")
        if isinstance(gain, (int, float)):
            _require(0.0 <= float(gain) <= 1.0, f"{what}.gain must be in [0, 1]")
        target = ev.get("target", "hazard")
        if op == "servo_toward":
            _require(
                target == "hazard" or target in hazards,
                f"{what}.target {target!r} must be 'hazard' or a task hazard",
            )
        gripper = ev.get("gripper", "policy")
        _require(gripper in GRIPPER_MODES, f"{what}.gripper must be one of {GRIPPER_MODES}")
        dxyz = _vec3(ev["dxyz"], f"{what}.dxyz") if "dxyz" in ev else (0.0, 0.0, 0.0)
        if op == "push":
            _require(np.linalg.norm(dxyz) > 0, f"{what}: push needs a non-zero dxyz")
        events.append(Event(op, steps, gain, str(target), dxyz, gripper))

    scene = obj.get("scene") or {}
    raw_mut = scene.get("mutations") or [] if isinstance(scene, Mapping) else None
    _require(isinstance(raw_mut, list), "scene.mutations must be a list")
    dynamic = {str(d["name"]) for d in task.get("dynamic_objects", [])}
    objects = {str(o["name"]) for o in task.get("objects", [])}
    mutations = []
    for i, m in enumerate(raw_mut):
        what = f"scene.mutations[{i}]"
        _require(isinstance(m, Mapping), f"{what} must be an object")
        op, name = m.get("op"), m.get("object")
        _require(op in ("shift", "rotate"), f"{what}.op must be shift or rotate")
        _require(name in objects, f"{what}.object {name!r} is not a task object")
        _require(name not in dynamic, f"{what}: {name} is mocap-driven; v0 cannot mutate it")
        dxyz = _vec3(m.get("dxyz", [0, 0, 0]), f"{what}.dxyz")
        _require(np.linalg.norm(dxyz) <= MAX_SHIFT_M + 1e-9, f"{what}: shift exceeds {MAX_SHIFT_M} m")
        dyaw = float(m.get("dyaw_deg", 0.0))
        _require(abs(dyaw) <= MAX_ROTATE_DEG, f"{what}: rotation exceeds {MAX_ROTATE_DEG} deg")
        mutations.append(Mutation(op, name, dxyz, dyaw))

    extra = tuple(obj.get("extra_predicates") or ())
    return CFProgram(
        program_id=pid,
        suite=suite,
        level=level,
        task_id=task_id,
        target_quadrant=tq,
        root_quadrants=frozenset(roots),
        trigger=trigger,
        events=tuple(events),
        mutations=tuple(mutations),
        params=params,
        extra_predicates=extra,
        rationale=str(obj.get("rationale", ""))[:600],
        raw=dict(obj),
    )


def sample_params(program: CFProgram, rng: np.random.Generator) -> dict[str, Any]:
    """Draw one value per declared parameter (sorted order, reproducible)."""
    return {name: program.params[name].sample(rng) for name in sorted(program.params)}


def resolve(value: Any, params: Mapping[str, Any], rng: np.random.Generator | None = None) -> Any:
    """Resolve a literal, a ``$param`` reference, or an integer ``(lo, hi)`` range."""
    if isinstance(value, str) and value.startswith("$"):
        return params[value[1:]]
    if isinstance(value, tuple):
        if rng is None:
            raise ProgramError("a range value needs an rng to resolve")
        return int(rng.integers(value[0], value[1] + 1))
    return value
