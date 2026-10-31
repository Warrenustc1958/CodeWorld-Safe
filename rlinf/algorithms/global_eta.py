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

"""One Lagrange multiplier for the whole rollout (all env workers), with a budget calibrated on the first
rollout (the unchanged initial policy): budget = cost_budget_factor * mean cost of that rollout.
State (eta, budget, history) is kept in a json next to the checkpoints so a resumed run continues it."""

from __future__ import annotations

import json
import os


class GlobalEta:
    def __init__(self, path: str, eta_init: float, eta_lr: float, eta_max: float, budget_factor: float,
                 budget: float | None = None):
        self.path = path
        self.eta_lr, self.eta_max, self.factor = float(eta_lr), float(eta_max), float(budget_factor)
        self.state = {"eta": float(eta_init), "budget": budget, "base_cost": None, "history": []}

    def load(self, global_step: int) -> None:
        """Restore the state as it was after `global_step` updates (drops later, unsaved history)."""
        if not os.path.exists(self.path):
            return
        st = json.load(open(self.path))
        hist = [h for h in st.get("history", []) if h["step"] < global_step]
        if not hist:  # nothing checkpointed yet: start over (the budget is re-calibrated on the first rollout)
            self.state["history"] = []
            return
        self.state = {"eta": float(hist[-1]["eta_next"]), "budget": st.get("budget"),
                      "base_cost": st.get("base_cost"), "history": hist}

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.path)

    @property
    def eta(self) -> float:
        return float(self.state["eta"])

    def update(self, step: int, cost_sum: float, cost_n: int) -> float:
        """Update with the costs of the rollout of `step`; returns the eta for the next rollout."""
        if cost_n <= 0:
            return self.eta
        mean = cost_sum / cost_n
        if self.state["budget"] is None:  # first rollout = unchanged initial policy
            self.state["base_cost"] = mean
            self.state["budget"] = self.factor * mean
        budget = float(self.state["budget"])
        eta_next = min(self.eta_max, max(0.0, self.eta + self.eta_lr * (mean - budget)))
        self.state["history"].append({"step": step, "mean_cost": mean, "n": cost_n, "budget": budget,
                                      "eta": self.eta, "eta_next": eta_next})
        self.state["eta"] = eta_next
        self.save()
        return eta_next
