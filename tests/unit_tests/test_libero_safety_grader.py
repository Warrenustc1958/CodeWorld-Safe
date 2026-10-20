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

"""Graders agree with the audited predicates and give finite continuous values.

Requires LIBERO-Safety (skipped otherwise). Tasks cover checkcontact +
checkrobotcontact (obstacle_avoidance), aux wronggrasppart (affordance knife)
and aux hand contact (human_safety porcelain plate).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

TASKS = [
    ("obstacle_avoidance", "L1", "pick_up_the_akita_black_bowl_on_the_stove_and_place_it_on_the_plate"),
    ("affordance", "L0", "grab_the_knife_and_cut_the_lemon_on_the_cutting_board"),
    ("human_safety", "L0", "put_the_banana_on_the_porcelain_plate"),
]


def _make(task):
    pytest.importorskip("robosuite")
    try:
        import libero.libero as libero_root
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"LIBERO-Safety not importable: {error}")
    suite, level, name = task
    bddl = os.path.join(libero_root.get_libero_path("bddl_files"), suite, level, f"{name}.bddl")
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=32, camera_widths=32, ignore_done=True)
    env.seed(0)
    env.reset()
    return env


@pytest.mark.parametrize("task", TASKS, ids=[t[0] for t in TASKS])
def test_graders_match_audited_predicates(task, monkeypatch):
    monkeypatch.setenv("RLINF_LIBERO_SAFETY_AUX", "1")
    from rlinf.envs.sim.libero.safety_auditor import audit_predicate_instances
    from rlinf.envs.sim.libero.safety_grader import build_graders, grade_step
    from rlinf.envs.sim.libero.venv import _unwrap_task_env

    env = _make(task)
    try:
        task_env = _unwrap_task_env(env)
        graders = build_graders(task_env, task_env.parsed_problem, aux=True)
        assert graders, "every L0/L1 task has at least one safety grader"
        rng = np.random.default_rng(0)
        compared = 0
        for _ in range(40):
            env.step(rng.uniform(-1.0, 1.0, size=7))
            grades = {g["name"].split("|")[0]: g for g in grade_step(graders, task_env)}
            for rec in audit_predicate_instances(env):
                if rec["semantics"] != "violation" or rec["predicate"].lower() == "checkgripperforce":
                    continue
                g = grades.get(rec["instance_id"])
                assert g is not None, f"no grader for {rec['instance_id']} {rec['predicate']}"
                if rec["predicate"].lower() == "checkrobotcontact":
                    # LIBERO's check_robot_contact compares geom names with integer ids
                    # and never fires; the grader may only add detections, never drop one.
                    assert g["violated"] or not rec["value"], (rec, g)
                else:
                    assert g["violated"] == bool(rec["value"]), (rec, g)
                assert np.isfinite(g["min_dist"]) and g["normal_force"] >= 0.0
                if g["violated"] and g["channel"] != "c_grasp":
                    assert g["min_dist"] <= 1e-3 and not g["near_miss"]
                compared += 1
        assert compared >= 40
    finally:
        env.close()
