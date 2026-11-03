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

"""algorithm.adv_norm for GRPO: group (default, unchanged), batch, none."""

import pytest
import torch

from rlinf.algorithms.advantages import compute_grpo_advantages

G = 8


def _adv(rewards, **kw):
    r = torch.tensor(rewards, dtype=torch.float32).view(1, -1)
    adv, _ = compute_grpo_advantages(r, torch.ones_like(r), G, adv_std_floor=0.1, **kw)
    return adv.view(-1, G)


def test_group_is_default_and_unchanged():
    r = [1.0, 0, 0, 0, 0, 0, 0, 0] + [0.5] * 8
    a = _adv(r)
    assert torch.equal(a, _adv(r, adv_norm="group"))
    x = torch.tensor(r[:8], dtype=torch.float32)
    assert torch.allclose(a[0], (x - x.mean()) / x.std())
    assert torch.equal(a[1], torch.zeros(8))


def test_batch_keeps_near_constant_group_small():
    # soft RM scores: a truly all-fail group with tiny RM noise next to a real mixed group
    noisy = [0.02, 0.03, 0.01, 0.02, 0.04, 0.02, 0.03, 0.01]
    mixed = [0.9, 0.85, 0.1, 0.05, 0.9, 0.1, 0.05, 0.1]
    group = _adv(noisy + mixed, adv_norm="group")
    batch = _adv(noisy + mixed, adv_norm="batch")
    # group: the floor (.1) stops the blow-up, but the noise group still gets a sizeable share of the update
    assert group[0].abs().max() / group[1].abs().max() > 0.1
    # batch: one std for all groups keeps the noise ~25x smaller than the real signal
    assert batch[0].abs().max() / batch[1].abs().max() < 0.05
    c = torch.tensor(noisy + mixed).view(2, G)
    c = c - c.mean(1, keepdim=True)
    assert torch.allclose(batch, c / c.std())


def test_none_is_centered_only():
    r = [1.0, 0, 0, 0, 0, 0, 0, 1]
    x = torch.tensor(r, dtype=torch.float32)
    assert torch.allclose(_adv(r, adv_norm="none")[0], x - x.mean())


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        _adv([0] * 8, adv_norm="global")
