#!/usr/bin/env python3
"""Four-gate validator for Code-Agent counterfactual programs and their episodes.

Every gate returns a ``Verdict`` with *all* failure reasons (stable codes plus a
human-readable detail), never only the first. Nothing that fails a gate is
admitted to the dataset.

  gate 1  program schema + task knowledge          (cf_program.load_program)
  gate 2  reset sanity: no initial hazard contact, mutation bounds
  gate 3  trial batch: solvable (>=1 success in N seeds), target hit rate
  gate 4  per-episode admission: audit/visual alignment, stitched length,
          branch point inside the saved parent, seam (sim state or pixels),
          total budget, chunk alignment, sim-state sidecar presence

Gates 1, 3 and 4 are pure functions of JSON / on-disk audits; gate 2 takes the
grader readings an env produced right after reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rlinf.envs.sim.libero.cf_program import MAX_ROTATE_DEG, MAX_SHIFT_M, ProgramError, load_program
from toolkits.libero_safety.build_dataset import (
    BRANCH_FRAME_MATCH_TOLERANCE,
    BRANCH_STATE_MATCH_TOLERANCE,
    ValidatedEpisode,
    _is_branch,
    _record_field,
    _root_key,
    _sim_state_at,
    validate_episode,
)

MIN_FRAMES = 16
MAX_TOTAL_STEPS = 512
MIN_HIT_RATE = 0.2
LEGACY_SKIP_FRAMES = 13  # Wan trainer skipped sequences shorter than this


@dataclass
class Verdict:
    gate: int
    ok: bool = True
    reasons: list[dict[str, str]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def fail(self, code: str, detail: str) -> None:
        self.ok = False
        self.reasons.append({"code": code, "detail": detail})

    @property
    def codes(self) -> list[str]:
        return [r["code"] for r in self.reasons]


# ------------------------------------------------------------------ gate 1


def gate1_schema(program_json: Mapping[str, Any], knowledge_dir: Path) -> Verdict:
    v = Verdict(gate=1)
    cell = program_json.get("cell") if isinstance(program_json, Mapping) else None
    if not isinstance(cell, Mapping):
        v.fail("SCHEMA", "program has no cell object")
        return v
    task_file = Path(knowledge_dir) / "tasks" / f"{cell.get('suite')}_{cell.get('level')}_t{cell.get('task_id')}.json"
    if not task_file.is_file():
        v.fail("UNKNOWN_CELL", f"no task knowledge file {task_file.name}")
        return v
    import json

    try:
        program = load_program(program_json, json.loads(task_file.read_text()))
    except ProgramError as error:
        v.fail("SCHEMA", str(error))
        return v
    v.info.update(program_id=program.program_id, cell=program.cell_key,
                  scene_mutated=program.scene_mutated)
    return v


# ------------------------------------------------------------------ gate 2


def gate2_reset(reset_grades: Sequence[Mapping[str, Any]],
                applied_mutations: Sequence[Mapping[str, Any]] = (),
                min_clearance_m: float = 0.0) -> Verdict:
    """``reset_grades``: safety_grader.grade_step() output right after reset settling.

    ``applied_mutations``: records from apply_scene_mutations with the realised
    ``dxyz`` / ``dyaw_deg`` (the program limits apply to what actually moved).
    """
    v = Verdict(gate=2)
    for g in reset_grades:
        if g.get("violated"):
            v.fail("INITIAL_VIOLATION", f"{g['name']} already violated at reset")
        elif float(g.get("normal_force", 0.0)) > 0.0:
            v.fail("INITIAL_CONTACT", f"{g['name']} in contact at reset ({g['normal_force']:.3g} N)")
        elif float(g.get("min_dist", np.inf)) < min_clearance_m:
            v.fail("INITIAL_CLEARANCE", f"{g['name']} min_dist {g['min_dist']:.3f} m < {min_clearance_m}")
    for m in applied_mutations:
        shift = float(np.linalg.norm(m.get("dxyz", (0.0, 0.0, 0.0))))
        if shift > MAX_SHIFT_M + 1e-9:
            v.fail("MUTATION_SHIFT", f"{m.get('object')} moved {shift:.3f} m > {MAX_SHIFT_M}")
        if abs(float(m.get("dyaw_deg", 0.0))) > MAX_ROTATE_DEG:
            v.fail("MUTATION_ROTATE", f"{m.get('object')} rotated {m['dyaw_deg']} deg > {MAX_ROTATE_DEG}")
    return v


# ------------------------------------------------------------------ gate 3


def gate3_trial(quadrants: Sequence[str], target_quadrant: str,
                min_hit_rate: float = MIN_HIT_RATE) -> Verdict:
    """``quadrants``: outcomes of a small trial batch (one per seed / branch)."""
    v = Verdict(gate=3)
    n = len(quadrants)
    successes = sum(q in ("Q1", "Q2") for q in quadrants)
    hits = sum(q == target_quadrant for q in quadrants)
    rate = hits / n if n else 0.0
    v.info.update(trials=n, successes=successes, hits=hits, hit_rate=rate)
    if n == 0:
        v.fail("NO_TRIALS", "trial batch is empty")
        return v
    if successes == 0:
        v.fail("UNSOLVABLE", f"policy succeeded in 0 of {n} trials")
    if rate < min_hit_rate:
        v.fail("LOW_HIT_RATE", f"{target_quadrant} hit {hits}/{n} = {rate:.2f} < {min_hit_rate}")
    return v


# ------------------------------------------------------------------ gate 4


def _load_frame(episode: ValidatedEpisode, index: int) -> np.ndarray:
    with np.load(episode.visual_path, allow_pickle=False) as sidecar:
        return np.asarray(sidecar[episode.camera_key][index])


def gate4_episode(
    episode: ValidatedEpisode,
    lookup: Mapping[tuple[str, str], ValidatedEpisode],
    *,
    min_frames: int = MIN_FRAMES,
    max_total_steps: int = MAX_TOTAL_STEPS,
    chunk: int | None = None,
    require_sim_state: bool = False,
    check_seam: bool = True,
) -> Verdict:
    """Admission check of one validated audit against its parent chain.

    ``chunk``: require branch points on chunk boundaries (new data, P14).
    ``require_sim_state``: require the per-chunk sidecar (new data, P1/P6).
    """
    v = Verdict(gate=4)
    own = episode.num_transitions
    stitched_steps = own
    item = episode
    depth = 0
    while _is_branch(item):
        parent = lookup.get(_root_key(item))
        bt = _record_field(item.record, "branch_timestep")
        if parent is None:
            v.fail("NO_PARENT", f"parent {_root_key(item)[1]} not found")
            break
        if bt is None:
            v.fail("NO_BRANCH_TIMESTEP", "branch has no branch_timestep")
            break
        bt = int(bt)
        if not 0 <= bt - 1 <= parent.num_transitions:
            v.fail("BRANCH_OUT_OF_RANGE",
                   f"branch_timestep={bt} but parent saved {parent.num_transitions} transitions (P2)")
            break
        if chunk and (bt - 1) % chunk:
            v.fail("BRANCH_NOT_CHUNK_ALIGNED", f"branch_timestep={bt} not on a {chunk}-step boundary (P14)")
        if check_seam and item is episode:
            parent_state = _sim_state_at(parent, bt)
            own_state = _sim_state_at(item, 1)
            if parent_state is not None and own_state is not None:
                diff = float(np.abs(parent_state - own_state).max())
                v.info["seam_state_diff"] = diff
                if diff > BRANCH_STATE_MATCH_TOLERANCE:
                    v.fail("SEAM_STATE_MISMATCH", f"branch start state differs by {diff:.3g}")
            else:
                drift = float(np.abs(_load_frame(parent, bt - 1).astype(np.int16)
                                     - _load_frame(item, 0).astype(np.int16)).mean())
                v.info["seam_pixel_diff"] = drift
                if drift > BRANCH_FRAME_MATCH_TOLERANCE:
                    v.fail("SEAM_PIXEL_MISMATCH", f"first frame differs from parent frame {bt - 1} by {drift:.3f}")
        stitched_steps += bt - 1
        item = parent
        depth += 1
        if depth > 8:
            v.fail("CHAIN_TOO_DEEP", "parent chain deeper than 8 or cyclic")
            break
    frames = stitched_steps + 1
    v.info.update(own_transitions=own, chain_depth=depth)
    if any(c in ("NO_PARENT", "NO_BRANCH_TIMESTEP", "BRANCH_OUT_OF_RANGE", "CHAIN_TOO_DEEP")
           for c in v.codes):
        # The prefix could not be resolved, so the stitched length is unknown;
        # judging own length alone would add a false TOO_SHORT.
        return v
    v.info["stitched_frames"] = frames
    if frames < min_frames:
        v.fail("TOO_SHORT", f"stitched sequence has {frames} frames < {min_frames} (P9)")
    if stitched_steps > max_total_steps:
        v.fail("OVER_BUDGET", f"stitched sequence has {stitched_steps} steps > {max_total_steps} (P8)")
    if require_sim_state and not isinstance(episode.record.get("sim_state_trajectory"), dict):
        v.fail("NO_SIM_STATE", "episode has no per-chunk sim-state sidecar (P1/P6)")
    return v


def load_episodes(paths: Iterable[Path], camera_key: str = "agentview_image",
                  collection_of=None) -> tuple[dict, list[dict[str, str]]]:
    """Validate audits (gate 4a); returns {(collection, source_id): episode} and schema failures."""
    from toolkits.libero_safety.build_dataset import _audit_collection_root, _collection_id

    lookup: dict[tuple[str, str], ValidatedEpisode] = {}
    failures = []
    for path in paths:
        path = Path(path)
        collection = collection_of(path) if collection_of else _collection_id(_audit_collection_root(path))
        try:
            ep = validate_episode(path, camera_key, collection)
        except (KeyError, TypeError, ValueError, OSError) as error:
            failures.append({"path": str(path), "code": "AUDIT_INVALID", "detail": str(error)[:300]})
            continue
        lookup[(ep.collection_id, ep.source_episode_id)] = ep
    return lookup, failures
