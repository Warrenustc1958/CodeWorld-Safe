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

"""Integration: training cost labels on a real LIBERO-Safety MuJoCo contact (no mocks).

Moves the human hand of human_safety task 0 onto the gripper, so robot and hand geoms really touch.
The grader-based cost (``cost_label: grader``) must see it; the LIBERO ``check_robot_contact``
predicate does not (it matches geom names against geom ids), which is why E11 had no robot cost.
Needs the LIBERO-Safety runtime (ImageMagick, LIBERO_CONFIG_PATH); skipped otherwise.
"""

import numpy as np
import pytest

try:
    from libero.libero.envs.env_wrapper import ControlEnv
except Exception as exc:  # pragma: no cover - runtime without LIBERO
    pytest.skip(f"LIBERO-Safety runtime not available: {exc}", allow_module_level=True)

from rlinf.envs.sim.libero.safety_grader import CHANNELS, build_graders, violated_channels
from rlinf.envs.sim.libero.utils import get_benchmark_overridden
from rlinf.envs.sim.libero.venv import _unwrap_task_env
from rlinf.envs.sim.world_model.safe.sync_real_env import SyncRealWorldModelEnv

HAND = "left_hand_1"


@pytest.fixture(scope="module")
def env():
    suite = get_benchmark_overridden("human_safety")()
    task = suite.get_task(0)
    e = ControlEnv(bddl_file_name=suite.get_task_bddl_file_path(task.level, task.level_id),
                   has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, horizon=100000)
    e.seed(0)
    e.reset()
    yield e
    e.close()


def _robot_hand_contacts(te):
    from rlinf.envs.sim.libero.safety_grader import object_geom_ids, robot_geom_ids

    a, b = set(robot_geom_ids(te)), set(object_geom_ids(te, HAND))
    d = te.sim.data
    return sum((c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a)
               for c in d.contact[: d.ncon])


def _move_hand_to_gripper(te):
    m, d = te.sim.model, te.sim.data
    bid = m.body_name2id(te.objects_dict[HAND].root_body)
    qadr = m.jnt_qposadr[m.body_jntadr[bid]]
    d.qpos[qadr:qadr + 3] = d.site_xpos[m.site_name2id("gripper0_grip_site")]
    te.sim.forward()


def _channels_via_env(e):
    stub = SyncRealWorldModelEnv.__new__(SyncRealWorldModelEnv)
    stub._real = [e]
    return dict(zip(CHANNELS, SyncRealWorldModelEnv._grader_channels(stub, 0)))


def test_real_robot_contact_reaches_training_cost(env):
    te = _unwrap_task_env(env)
    assert te.parsed_problem["constraints"] == [["checkrobotcontact", HAND]]

    # at reset: no contact, all labels clean
    assert _robot_hand_contacts(te) == 0
    assert not any(_channels_via_env(env).values())

    _move_hand_to_gripper(te)
    assert _robot_hand_contacts(te) > 0, "setup did not create a robot/hand contact"

    # the cost label used for training (through SyncRealWorldModelEnv and its grader cache) sees it
    ch = _channels_via_env(env)
    assert ch["c_robot"] and not ch["c_obj"] and not ch["c_grasp"]
    # ... and agrees with the evaluation grader's boolean (same implementation)
    grades = [g.value(te) for g in build_graders(te, te.parsed_problem, aux=True)]
    assert any(g["violated"] for g in grades if g["channel"] == "c_robot")
    assert violated_channels(build_graders(te, te.parsed_problem, aux=True), te)["c_robot"]

    # the LIBERO predicate misses the same contact (the E11 cost_label=predicate path)
    assert not bool(te.object_states_dict[HAND].check_robot_contact())


def test_grader_cache_rebuilt_after_model_change(env):
    te = _unwrap_task_env(env)
    _channels_via_env(env)
    model, graders = te._rlinf_cost_graders
    assert model is te.sim.model and graders
    te._rlinf_cost_graders = (object(), graders)  # stale cache key -> rebuilt
    _channels_via_env(env)
    assert te._rlinf_cost_graders[0] is te.sim.model
    assert np.all([g._a for g in te._rlinf_cost_graders[1]])
