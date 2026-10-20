#!/usr/bin/env python3
"""Round planner: pick programs per gap cell and split the episode budget by gap.

Per round, for every cell with an unmet quota:
  * budget (branch episodes) is split across cells in proportion to the unsafe
    gap (Q2 + Q4 weigh double), then to the remaining gap;
  * inside a cell, validated programs are scored
        score = gap[target] * hit_rate_est * novelty_factor
    (untried programs get an optimistic prior hit rate) and exploited;
  * whatever the best program cannot cover becomes "generate" slots: up to
    ``max_llm_calls`` new programs from the LLM for the quadrant with the
    largest gap.
State (rounds without growth) decides when scene mutations are allowed
(2 stagnant rounds, task book section 6 rule 2) and when a cell is infeasible (3).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from toolkits.codeworld_safe.agent.archive import QUADRANTS, Archive

QUOTA = {"Q1": 50, "Q2": 15, "Q3": 40, "Q4": 40}
UNSAFE = ("Q2", "Q4")
PRIOR_HIT_RATE = 0.3  # optimistic for validated but untried programs
MUTATION_AFTER_STAGNANT = 2
INFEASIBLE_AFTER_STAGNANT = 3


@dataclass
class Assignment:
    cell: str
    action: str  # "run" | "generate"
    target_quadrant: str
    episodes: int = 0
    program_id: str | None = None
    score: float | None = None
    allow_mutation: bool = False
    reason: str = ""


@dataclass
class CellState:
    rounds: int = 0
    stagnant: int = 0
    last_counts: dict[str, int] = field(default_factory=dict)
    infeasible: bool = False


def coverage_gaps(counts: dict[str, dict[str, int]], quota: dict[str, int] = QUOTA) -> dict[str, dict[str, int]]:
    """{cell: {quadrant: missing}} for cells that still miss any quota."""
    out = {}
    for cell, c in counts.items():
        gap = {q: max(0, quota[q] - int(c.get(q, 0))) for q in QUADRANTS}
        if any(gap.values()):
            out[cell] = gap
    return out


def _weight(gap: dict[str, int]) -> float:
    return 2.0 * sum(gap[q] for q in UNSAFE) + sum(gap[q] for q in QUADRANTS if q not in UNSAFE)


def split_budget(gaps: dict[str, dict[str, int]], budget: int) -> dict[str, int]:
    """Integer split proportional to gap weight (largest remainders)."""
    weights = {c: _weight(g) for c, g in gaps.items()}
    total = sum(weights.values())
    if total <= 0 or budget <= 0:
        return {c: 0 for c in gaps}
    raw = {c: budget * w / total for c, w in weights.items()}
    alloc = {c: int(math.floor(v)) for c, v in raw.items()}
    rest = budget - sum(alloc.values())
    for c in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[:rest]:
        alloc[c] += 1
    return alloc


class Planner:
    def __init__(self, archive: Archive, state_path: Path, *, max_llm_calls: int = 3,
                 quota: dict[str, int] = QUOTA) -> None:
        self.archive = archive
        self.state_path = Path(state_path)
        self.max_llm_calls = max_llm_calls
        self.quota = quota
        self.state: dict[str, CellState] = {}
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text())
            self.state = {c: CellState(**s) for c, s in raw.items()}

    def save(self) -> None:
        self.state_path.write_text(json.dumps({c: asdict(s) for c, s in self.state.items()}, indent=1))

    def end_round(self, counts: dict[str, dict[str, int]]) -> None:
        """Update stagnation after a round's episodes were ingested."""
        for cell, c in counts.items():
            st = self.state.setdefault(cell, CellState())
            st.rounds += 1
            grew = any(c.get(q, 0) > st.last_counts.get(q, 0) for q in UNSAFE)
            st.stagnant = 0 if grew else st.stagnant + 1
            st.last_counts = dict(c)
            gap = {q: max(0, self.quota[q] - c.get(q, 0)) for q in UNSAFE}
            st.infeasible = st.stagnant >= INFEASIBLE_AFTER_STAGNANT and any(gap.values())
        self.save()

    def _score(self, program_id: str, gap: dict[str, int]) -> float:
        s = self.archive.stats(program_id)
        hit = s["hit_rate"] if s["trials"] else PRIOR_HIT_RATE
        nov = s["novelty_m"]
        novelty_factor = 1.0 if nov is None or not math.isfinite(nov) else min(1.0, nov / 0.05)
        return gap[s["target_quadrant"]] * max(hit, 0.0) * max(novelty_factor, 0.1)

    def plan_round(self, counts: dict[str, dict[str, int]], budget: int) -> list[Assignment]:
        gaps = coverage_gaps(counts, self.quota)
        gaps = {c: g for c, g in gaps.items() if not self.state.get(c, CellState()).infeasible}
        alloc = split_budget(gaps, budget)
        plan: list[Assignment] = []
        for cell, gap in sorted(gaps.items()):
            st = self.state.get(cell, CellState())
            mutate = st.stagnant >= MUTATION_AFTER_STAGNANT
            left = alloc.get(cell, 0)
            target = max(QUADRANTS, key=lambda q: (gap[q] * (2 if q in UNSAFE else 1), q in UNSAFE))
            scored = sorted(
                ((self._score(r.program_id, gap), r.program_id)
                 for r in self.archive.programs(cell, status="validated")),
                reverse=True,
            )
            for score, pid in scored:
                if left <= 0 or score <= 0:
                    break
                s = self.archive.stats(pid)
                hit = s["hit_rate"] if s["trials"] else PRIOR_HIT_RATE
                want = gap[s["target_quadrant"]]
                n = min(left, max(1, math.ceil(want / max(hit, 0.05))))
                plan.append(Assignment(cell, "run", s["target_quadrant"], n, pid, round(score, 3),
                                       reason=f"exploit: hit={hit:.2f} gap={want}"))
                left -= n
            good = any(sc >= 1.0 for sc, _ in scored)
            if not good or left > 0:
                plan.append(Assignment(cell, "generate", target, 0, None, None, mutate,
                                       reason=f"explore: best score {scored[0][0] if scored else 0:.2f}, "
                                              f"{left} episodes uncovered, gap {gap}"))
                if left > 0:
                    plan.append(Assignment(cell, "run", target, left, None, None, mutate,
                                           reason="episodes for programs generated this round"))
        return plan
