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

"""SyncRealWorldModelEnv idle real-env pool: at most idle_env_cap idle envs, least recently used task closed first."""

import pytest

sync_real_env = pytest.importorskip("rlinf.envs.sim.world_model.safe.sync_real_env")


class _Env:
    closed = 0

    def close(self):
        _Env.closed += 1


def _env(cap, pool, used):
    env = sync_real_env.SyncRealWorldModelEnv.__new__(sync_real_env.SyncRealWorldModelEnv)
    env.idle_env_cap = cap
    env._pool = {k: [_Env() for _ in range(n)] for k, n in pool.items()}
    env._pool_used = dict(used)
    return env


def test_under_cap_keeps_everything():
    _Env.closed = 0
    env = _env(10, {("a", 0): 4, ("a", 1): 4}, {("a", 0): 1, ("a", 1): 2})
    env._trim_pool()
    assert _Env.closed == 0 and sum(map(len, env._pool.values())) == 8


def test_over_cap_closes_least_recently_used_first():
    _Env.closed = 0
    pool = {("a", 0): 8, ("a", 1): 8, ("b", 2): 8}
    env = _env(12, pool, {("a", 0): 5, ("a", 1): 1, ("b", 2): 3})  # ("a", 1) oldest, then ("b", 2)
    env._trim_pool()
    assert _Env.closed == 12
    assert {k: len(v) for k, v in env._pool.items()} == {("a", 0): 8, ("a", 1): 0, ("b", 2): 4}


def test_zero_cap_is_unbounded():
    _Env.closed = 0
    env = _env(0, {("a", 0): 50}, {})
    env._trim_pool()
    assert _Env.closed == 0
