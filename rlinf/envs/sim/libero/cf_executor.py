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

"""Interpret a ``cws_cf_program/v0`` program inside the branch collector.

The executor owns branch-point choice and the per-step action edits of one
branch; the collector still owns snapshots, search bookkeeping and the root /
branch phases. Branch points are restricted to action-chunk boundaries
(``(timestep - 1) % chunk == 0``) so a chunked policy (OpenVLA-OFT, chunk 8)
replans from the restored observation, and to the part of the root that the
auditor actually saved (P2). A branch inherits the root's elapsed steps, so
its remaining budget is ``max_total_steps - (branch_timestep - 1)`` (P8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from rlinf.envs.sim.libero.cf_program import CFProgram, Event, resolve

# Branches that cannot run at least this many steps are not worth starting:
# the validator rejects episodes shorter than 16 frames (P9).
MIN_BRANCH_STEPS = 16
RISK_DISTANCE_SCALE = 0.08
RISK_WEIGHT_FLOOR = 0.05


def hazard_position(hazards: Sequence[tuple[str, np.ndarray]], name: str | None):
    """Position of the hazard whose key names ``name`` (any hazard if None)."""
    for key, pos in hazards:
        if name is None or key.split("|")[-1] == name:
            return np.asarray(pos, dtype=np.float64)
    return None


def _min_hazard_distance(snapshot, name: str | None) -> float:
    if snapshot.eef_pos is None:
        return float("inf")
    dists = [
        float(np.linalg.norm(np.asarray(pos) - snapshot.eef_pos))
        for key, pos in snapshot.hazards
        if name is None or key.split("|")[-1] == name
    ]
    return min(dists) if dists else float("inf")


@dataclass
class _ResolvedEvent:
    op: str
    steps: int
    gain: float
    target: str
    dxyz: np.ndarray
    gripper: str


class ProgramExecutor:
    """Branch-point selection and action edits for one program on one env."""

    def __init__(
        self,
        program: CFProgram,
        params: dict[str, Any],
        rng: np.random.Generator,
        *,
        chunk: int = 8,
        max_total_steps: int = 512,
    ) -> None:
        if chunk < 1 or max_total_steps < 1:
            raise ValueError("chunk and max_total_steps must be positive")
        if max_total_steps % chunk:
            raise ValueError("max_total_steps must be a multiple of the action chunk (P14)")
        self.program = program
        self.params = dict(params)
        self.rng = rng
        self.chunk = int(chunk)
        self.max_total_steps = int(max_total_steps)
        self._queue: list[_ResolvedEvent] = []
        self._hazard_name: str | None = None
        self._branch_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ budget
    def remaining_budget(self, branch_timestep: int) -> int:
        return self.max_total_steps - (int(branch_timestep) - 1)

    def is_chunk_boundary(self, timestep: int) -> bool:
        return (int(timestep) - 1) % self.chunk == 0

    # ------------------------------------------------------------ branch point
    def trigger_value(self, snapshot) -> float:
        trig = self.program.trigger
        if trig.metric == "step":
            return float(snapshot.timestep)
        return _min_hazard_distance(snapshot, trig.hazard)

    def eligible(self, snapshot, saved_steps: int) -> bool:
        t = int(snapshot.timestep)
        return (
            self.is_chunk_boundary(t)
            and t - 1 <= saved_steps
            and self.remaining_budget(t) >= MIN_BRANCH_STEPS
        )

    def choose_branch_point(self, history: Sequence[Any], saved_steps: int):
        """A snapshot satisfying the trigger, else the fallback, else None."""
        pool = [s for s in history if self.eligible(s, saved_steps)]
        if not pool:
            return None
        hits = [s for s in pool if self.program.trigger.holds(self.trigger_value(s))]
        if hits:
            return hits[int(self.rng.integers(0, len(hits)))]
        if self.program.trigger.fallback == "skip":
            return None
        dist = np.asarray([_min_hazard_distance(s, self.program.trigger.hazard) for s in pool])
        weights = np.where(np.isfinite(dist), np.exp(-dist / RISK_DISTANCE_SCALE), 0.0)
        weights = weights + RISK_WEIGHT_FLOOR
        return pool[int(self.rng.choice(len(pool), p=weights / weights.sum()))]

    # ---------------------------------------------------------------- execution
    def start(self, snapshot) -> None:
        """Resolve the event schedule for a new branch from ``snapshot``."""
        self._queue = []
        for ev in self.program.events:
            self._queue.append(self._resolve_event(ev))
        self._hazard_name = self.program.trigger.hazard
        self._branch_info = {
            "trigger_value": self.trigger_value(snapshot),
            "trigger_hit": bool(self.program.trigger.holds(self.trigger_value(snapshot))),
            "events": [
                {"op": e.op, "steps": e.steps, "gain": e.gain, "target": e.target,
                 "dxyz": e.dxyz.tolist(), "gripper": e.gripper}
                for e in self._queue
            ],
        }

    def _resolve_event(self, ev: Event) -> _ResolvedEvent:
        steps = int(resolve(ev.steps, self.params, self.rng))
        gain = float(resolve(ev.gain, self.params, self.rng))
        if steps < 1 or not 0.0 <= gain <= 1.0:
            raise ValueError(f"resolved event {ev.op} has steps={steps} gain={gain}")
        return _ResolvedEvent(ev.op, steps, gain, ev.target, np.asarray(ev.dxyz, float), ev.gripper)

    @property
    def total_steps(self) -> int:
        return sum(e.steps for e in self._queue)

    @property
    def active(self) -> bool:
        return bool(self._queue)

    def branch_metadata(self) -> dict[str, Any]:
        return {
            "program_id": self.program.program_id,
            "program_params": dict(self.params),
            "program_target_quadrant": self.program.target_quadrant,
            "program_branch": dict(self._branch_info),
            "scene_mutated": self.program.scene_mutated,
            "max_total_steps": self.max_total_steps,
        }

    def action(self, policy_action: np.ndarray, eef_pos, hazards) -> np.ndarray | None:
        """Edit one policy action; None once every event has run."""
        if not self._queue:
            return None
        ev = self._queue[0]
        out = np.asarray(policy_action, dtype=np.float64).copy()
        if ev.op in ("servo_toward", "push"):
            if ev.op == "push":
                direction = ev.dxyz
            else:
                name = None if ev.target == "hazard" else ev.target
                target = hazard_position(hazards or (), name or self._hazard_name)
                direction = (
                    None if target is None or eef_pos is None
                    else target - np.asarray(eef_pos, dtype=np.float64)
                )
            norm = 0.0 if direction is None else float(np.linalg.norm(direction))
            if norm > 1e-6:
                # OSC delta translation is axis-aligned with the world in LIBERO scenes.
                out[:3] = (1.0 - ev.gain) * out[:3] + ev.gain * direction / norm
        elif ev.op == "noise":
            noise = self.rng.uniform(-1.0, 1.0, size=6)
            out[:6] = (1.0 - ev.gain) * out[:6] + ev.gain * noise
        elif ev.op == "release_early":
            out[-1] = -1.0
        elif ev.op == "hold_gripper":
            out[-1] = 1.0
        if ev.gripper == "close":
            out[-1] = 1.0
        elif ev.gripper == "open":
            out[-1] = -1.0
        ev.steps -= 1
        if ev.steps <= 0:
            self._queue.pop(0)
        return np.clip(out, -1.0, 1.0)
