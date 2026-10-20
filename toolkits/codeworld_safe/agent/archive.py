#!/usr/bin/env python3
"""Program archive for the Code-Agent counterfactual generator.

Layout (one directory per campaign):
  programs.jsonl   one ProgramRecord per program (append-only; last record wins)
  outcomes.jsonl   one row per branch episode produced by a program
  llm/             prompts and raw responses, named by prompt sha1

Features: the branch's end-effector path resampled to ``N_POINTS`` points plus
the final positions of the task's objects. ``distance`` is the RMS 3D distance
per point (metres), so ``DEDUP_EPS_M`` (1 cm) is the same scale cf_refine used.
Novelty of an episode = distance to the nearest outcome of *another* program in
the same cell; novelty of a program = mean over its admitted outcomes.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

N_POINTS = 16
DEDUP_EPS_M = 0.01
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


@dataclass
class ProgramRecord:
    program_id: str
    cell: str
    target_quadrant: str
    program: dict[str, Any]
    status: str = "proposed"  # proposed | validated | rejected
    gate1_reasons: list[dict[str, str]] = field(default_factory=list)
    llm: dict[str, Any] = field(default_factory=dict)  # provider, model, usage, prompt_sha1
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


def trajectory_feature(record: dict[str, Any]) -> np.ndarray:
    """Feature vector of one audit record (its own, post-branch trajectory)."""
    traj = record["trajectory"]
    eef = np.asarray([s["observation"]["robot0_eef_pos"] for s in traj], dtype=np.float64)
    if len(eef) == 0:
        raise ValueError("empty trajectory")
    idx = np.linspace(0, len(eef) - 1, N_POINTS)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(eef) - 1)
    w = (idx - lo)[:, None]
    path = (1 - w) * eef[lo] + w * eef[hi]
    final = traj[-1]["observation"]
    objects = [
        np.asarray(final[k], dtype=np.float64)[:3]
        for k in sorted(final)
        if k.endswith("_pos") and not k.startswith("robot0_") and np.asarray(final[k]).size >= 3
    ]
    return np.concatenate([path.reshape(-1)] + objects) if objects else path.reshape(-1)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS per-point 3D distance (m); features of different length are incomparable."""
    if a.shape != b.shape:
        return float("inf")
    points = a.size // 3
    return float(np.linalg.norm(a - b) / np.sqrt(max(points, 1)))


class Archive:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "llm").mkdir(exist_ok=True)
        self._programs: dict[str, ProgramRecord] = {}
        self._outcomes: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------ storage
    def _load(self) -> None:
        p = self.root / "programs.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = ProgramRecord(**json.loads(line))
                    self._programs[rec.program_id] = rec
        o = self.root / "outcomes.jsonl"
        if o.exists():
            self._outcomes = [json.loads(line) for line in o.read_text().splitlines() if line.strip()]

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with (self.root / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ----------------------------------------------------------- programs
    def unique_id(self, program_id: str) -> str:
        if program_id not in self._programs:
            return program_id
        n = 2
        while f"{program_id}_v{n}" in self._programs:
            n += 1
        return f"{program_id}_v{n}"

    def add_program(self, rec: ProgramRecord) -> None:
        self._programs[rec.program_id] = rec
        self._append("programs.jsonl", asdict(rec))

    def set_status(self, program_id: str, status: str) -> None:
        rec = self._programs[program_id]
        rec.status = status
        self._append("programs.jsonl", asdict(rec))

    def program(self, program_id: str) -> ProgramRecord:
        return self._programs[program_id]

    def programs(self, cell: str | None = None, status: str | None = None) -> list[ProgramRecord]:
        return [r for r in self._programs.values()
                if (cell is None or r.cell == cell) and (status is None or r.status == status)]

    # ----------------------------------------------------------- outcomes
    def outcomes(self, program_id: str | None = None, cell: str | None = None) -> list[dict[str, Any]]:
        return [o for o in self._outcomes
                if (program_id is None or o["program_id"] == program_id)
                and (cell is None or o["cell"] == cell)]

    def novelty(self, feature: np.ndarray, cell: str, exclude_program: str | None = None) -> float:
        best = float("inf")
        for o in self._outcomes:
            if o["cell"] != cell or not o.get("admitted") or o["program_id"] == exclude_program:
                continue
            best = min(best, distance(feature, np.asarray(o["feature"])))
        return best

    def nearest_admitted(self, feature: np.ndarray, cell: str) -> float:
        """Distance to the nearest admitted outcome of any program (dedup check)."""
        return self.novelty(feature, cell, exclude_program=None)

    def add_outcome(self, *, program_id: str, cell: str, quadrant: str, feature: np.ndarray,
                    gate4_ok: bool, gate4_codes: Iterable[str], audit_path: str,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record one branch; it is admitted only if gate 4 passed and it is not a duplicate."""
        nearest = self.nearest_admitted(feature, cell)
        duplicate = nearest < DEDUP_EPS_M
        row = {
            "program_id": program_id,
            "cell": cell,
            "quadrant": quadrant,
            "gate4_ok": bool(gate4_ok),
            "gate4_codes": list(gate4_codes),
            "duplicate": bool(duplicate),
            "nearest_m": None if not np.isfinite(nearest) else nearest,
            "admitted": bool(gate4_ok and not duplicate),
            "feature": np.asarray(feature, dtype=np.float64).round(5).tolist(),
            "audit_path": audit_path,
            **(extra or {}),
        }
        self._outcomes.append(row)
        self._append("outcomes.jsonl", row)
        return row

    # -------------------------------------------------------------- stats
    def stats(self, program_id: str) -> dict[str, Any]:
        rec = self._programs[program_id]
        rows = self.outcomes(program_id)
        admitted = [r for r in rows if r["admitted"]]
        quadrants = Counter(r["quadrant"] for r in rows)
        hits = sum(r["quadrant"] == rec.target_quadrant for r in rows)
        novelty = [
            self.novelty(np.asarray(r["feature"]), rec.cell, exclude_program=program_id)
            for r in admitted
        ]
        finite = [n for n in novelty if np.isfinite(n)]
        return {
            "program_id": program_id,
            "cell": rec.cell,
            "status": rec.status,
            "target_quadrant": rec.target_quadrant,
            "trials": len(rows),
            "quadrants": {q: quadrants.get(q, 0) for q in QUADRANTS},
            "hit_rate": hits / len(rows) if rows else None,
            "gate4_pass_rate": sum(r["gate4_ok"] for r in rows) / len(rows) if rows else None,
            "admitted": len(admitted),
            "duplicates": sum(r["duplicate"] for r in rows),
            "novelty_m": (float(np.mean(finite)) if finite else (float("inf") if admitted else None)),
            "trigger_hit_rate": (
                sum(bool(r.get("trigger_hit")) for r in rows) / len(rows) if rows else None
            ),
        }

    def cell_counts(self, cell: str) -> dict[str, int]:
        c = Counter(o["quadrant"] for o in self._outcomes if o["cell"] == cell and o["admitted"])
        return {q: c.get(q, 0) for q in QUADRANTS}

    def save_llm_exchange(self, sha1: str, prompt: str, response: str, meta: dict) -> Path:
        path = self.root / "llm" / f"{sha1[:12]}.json"
        path.write_text(json.dumps({"meta": meta, "prompt": prompt, "response": response},
                                   ensure_ascii=False, indent=1))
        return path
