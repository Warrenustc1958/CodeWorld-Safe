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

"""Frozen helpers for the safety-constrained world-model env (SC-GRPO).

Vendored from RLinf-Safety (toolkits/libero_safety/sam_alignment and wan_ac_sft) so this
repo does not import across forks; keep them in sync with the frozen checkpoints:
  SAMNet    sam_alignment/model.py (cost head on Wan latents, cost_all_d32)
  StateNet  sam_alignment/scgrpo/train_state_reg.py (dual frame -> pi0.5 proprio)
  enable_dual / scale_pose   wan_ac_sft/dual_view.py, wan_ac_sft/sampler.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CHUNK = 8
K = 3  # c_obj, c_robot, c_grasp
NULL_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
VARIANT_SLOTS = {"B0": (), "B1": (0, 1), "B2": (0, 1, 2, 3)}


def scale_pose(actions: np.ndarray, pose_scale: float = 8.0, pose_clip: float = 4.0) -> np.ndarray:
    """The Wan action transform: pose dims x pose_scale, capped at +-pose_clip; gripper unchanged."""
    out = np.array(actions, np.float32, copy=True)
    out[..., :6] *= pose_scale
    if pose_clip > 0:
        out[..., :6] = np.clip(out[..., :6], -pose_clip, pose_clip)
    return out


# ---------------------------------------------------------------- dual-view Wan patches
_VAE_PATCHED = False


def _patch_vae_dual() -> None:
    global _VAE_PATCHED
    if _VAE_PATCHED:
        return
    from diffsynth.models import wan_video_vae as wv

    for cls in {c for c in vars(wv).values() if isinstance(c, type) and hasattr(c, "single_encode")}:
        enc, dec = cls.single_encode, cls.single_decode

        def single_encode(self, video, device, _enc=enc):
            h, w = video.shape[-2:]
            if h == 2 * w:
                return torch.cat([_enc(self, video[..., : h // 2, :], device),
                                  _enc(self, video[..., h // 2:, :], device)], dim=-2)
            return _enc(self, video, device)

        def single_decode(self, hidden_state, device, _dec=dec):
            h, w = hidden_state.shape[-2:]
            if h == 2 * w:
                return torch.cat([_dec(self, hidden_state[..., : h // 2, :], device),
                                  _dec(self, hidden_state[..., h // 2:, :], device)], dim=-2)
            return _dec(self, hidden_state, device)

        cls.single_encode, cls.single_decode = single_encode, single_decode
    _VAE_PATCHED = True


def enable_dual(pipe, views: int = 2) -> None:
    """Agent over wrist stacked 512x256 frames: per-view VAE, action embedding repeated per view."""
    _patch_vae_dual()
    m = pipe.dit.action_mlp2
    if getattr(m, "_views", None) == views:
        return
    orig = type(m).forward

    def forward(x, _m=m):
        y = orig(_m, x)
        return y.repeat_interleave(_m._views, dim=-2) if _m._views > 1 else y

    m._views = views
    m.forward = forward


# ---------------------------------------------------------------- SAM cost head
class SAMNet(nn.Module):
    def __init__(self, variant="B2", views=2, num_tasks=15, dim=256, heads=8, layers=2,
                 latent_mean=None, latent_std=None, action_mean=None, action_std=None, dropout=0.1):
        super().__init__()
        self.variant, self.views, self.num_tasks, self.dim = variant, views, num_tasks, dim
        self.slots = list(VARIANT_SLOTS[variant])
        self.register_buffer("lat_mu", torch.tensor(latent_mean or [0.0] * 48).view(1, 48, 1, 1, 1))
        self.register_buffer("lat_sd", torch.tensor(latent_std or [1.0] * 48).view(1, 48, 1, 1, 1))
        self.register_buffer("act_mu", torch.tensor(action_mean or [0.0] * 7))
        self.register_buffer("act_sd", torch.tensor(action_std or [1.0] * 7))
        if self.slots:
            self.stem = nn.Sequential(nn.Conv2d(48, 128, 3, 2, 1), nn.GELU(), nn.Conv2d(128, dim, 3, 1, 1))
            self.view_emb = nn.Parameter(torch.zeros(2, dim))
            self.slot_emb = nn.Parameter(torch.zeros(4, dim))
            self.pos_emb = nn.Parameter(torch.zeros(64, dim))
        self.act = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
        self.step_emb = nn.Parameter(torch.zeros(CHUNK, dim))
        self.task_emb = nn.Embedding(num_tasks, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.cost_head = nn.Linear(dim, K)
        self.prox_head = nn.Linear(dim, 2)

    def visual_tokens(self, z):
        z = (z - self.lat_mu) / self.lat_sd
        b, s = z.shape[0], len(self.slots)
        out = []
        for v in range(self.views):
            zv = z[:, :, self.slots, 16 * v:16 * (v + 1)]
            x = self.stem(zv.permute(0, 2, 1, 3, 4).reshape(b * s, 48, 16, 16))
            x = x.flatten(2).transpose(1, 2).reshape(b, s, 64, self.dim)
            x = x + self.pos_emb + self.slot_emb[self.slots][None, :, None] + self.view_emb[v]
            out.append(x.reshape(b, s * 64, self.dim))
        return torch.cat(out, 1)

    def forward(self, z, action, task_id):
        """z [B,48,4,32,16], action [B,8,7] env units, task_id [B] -> cost logits [B,8,3]."""
        tokens = [self.act((action - self.act_mu) / self.act_sd) + self.step_emb, self.task_emb(task_id)[:, None]]
        if self.slots:
            tokens.append(self.visual_tokens(z))
        h = self.norm(self.encoder(torch.cat(tokens, 1)))[:, :CHUNK]
        return self.cost_head(h)

    @classmethod
    def from_checkpoint(cls, path: str) -> "SAMNet":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ck["config"])
        m.load_state_dict(ck["model_state_dict"], strict=False)
        return m.eval()


# ---------------------------------------------------------------- proprio state from frames
class StateNet(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.GELU())
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256), blk(256, 256),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 8))

    def forward(self, x):  # x uint8 [B, 256, 128, 3]
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1)
        return self.h(self.f(x))


class StateReader(nn.Module):
    """Dual 512x256 uint8 frames -> pi0.5 proprio state (8) in env units."""

    def __init__(self, path: str):
        super().__init__()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = StateNet()
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.register_buffer("mu", torch.as_tensor(ck["mu"], dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(ck["sd"], dtype=torch.float32))

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, h, w, c = frames.shape
        x = frames.float().reshape(b, h // 2, 2, w // 2, 2, c).mean((2, 4)).round().to(torch.uint8)
        return self.net(x) * self.sd + self.mu
