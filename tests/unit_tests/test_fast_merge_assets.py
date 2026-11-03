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

"""sync_real_env._fast_merge_assets builds byte-identical LIBERO-Safety scene XML to robosuite's merge_assets.

Needs LIBERO-Safety with its assets; skipped otherwise. Physics only (no renderer).
"""

import os
import time

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import pytest

ET = pytest.importorskip("xml.etree.ElementTree")
pytest.importorskip("robosuite")
sync_real_env = pytest.importorskip("rlinf.envs.sim.world_model.safe.sync_real_env")

CELLS = [("affordance", 0), ("human_safety", 5), ("obstacle_avoidance_human", 10)]


def _build_xml(suite, task):
    from libero.libero.envs.env_wrapper import ControlEnv
    from rlinf.envs.sim.libero.utils import get_benchmark_overridden

    s = get_benchmark_overridden(suite)()
    t = s.get_task(task)
    t0 = time.perf_counter()
    env = ControlEnv(bddl_file_name=s.get_task_bddl_file_path(t.level, t.level_id), has_offscreen_renderer=False,
                     use_camera_obs=False, ignore_done=True, horizon=100000)
    dt = time.perf_counter() - t0
    xml = env.env.model.get_xml()  # robosuite scene before compilation (MuJoCo's compiled XML differs build to build)
    env.close()
    return xml, dt


def test_fast_merge_assets_identical_scene():
    from robosuite.models.base import MujocoXML
    from robosuite.models.objects.objects import MujocoObject

    originals = {cls: cls.__dict__["merge_assets"] for cls in (MujocoXML, MujocoObject)}
    try:
        for cls, fn in originals.items():
            cls.merge_assets = fn
        ref = [_build_xml(*c) for c in CELLS]
        sync_real_env._patch_merge_assets()
        assert MujocoXML.merge_assets is sync_real_env._fast_merge_assets
        new = [_build_xml(*c) for c in CELLS]
    finally:
        for cls, fn in originals.items():
            cls.merge_assets = fn
    for (cell, (x0, t0), (x1, t1)) in zip(CELLS, ref, new):
        assert x0 == x1, f"scene XML differs for {cell}"
        print(f"{cell}: build {t0:.1f}s -> {t1:.1f}s")
