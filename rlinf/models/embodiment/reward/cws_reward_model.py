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

"""Task-conditioned ResNet success classifier for CodeWorld-Safe world-model rollouts.

The visual encoder is the one of diffsynth's ``TaskEmbedResnetRewModel`` (same
module layout, so the WoVR ``resnet_rm.pth`` encoder loads unchanged). Differences:

* ``views``: 1 = agentview only, 2 = agentview + wrist. The dual Wan stacks the two
  256x256 views vertically (agent on top). Each half goes through the shared
  encoder and the two 512-d features are concatenated before the head. An
  agent-only model given a dual frame scores the top half, so the same checkpoint
  works with single- and dual-view world models.
* The task vocabulary (instruction strings) lives in the checkpoint, so the world
  model backend needs no suite-specific lookup table.
* ``forward`` returns logits; ``predict_rew`` returns sigmoid probabilities (or
  rounded ones with ``binarize``, matching the diffsynth reward models).

Inputs are RGB in [-1, 1], ``[B, 3, H, W]`` with H = W (single) or H = 2W (dual).
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = (
            nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
            if use_1x1conv
            else None
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return F.relu(y + x)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class CWSRewardModel(nn.Module):
    def __init__(
        self,
        views: int = 1,
        instructions: Sequence[str] = (),
        task_embed_dim: int = 64,
        init: str | None = None,
        binarize: bool = False,
    ) -> None:
        super().__init__()
        if views not in (1, 2):
            raise ValueError(f"views must be 1 or 2, got {views}")
        if not instructions:
            raise ValueError("instructions (task vocabulary) must not be empty")
        self.views = views
        self.instructions = [str(s).strip() for s in instructions]
        self.task_embed_dim = task_embed_dim
        self.binarize = binarize
        self._task_ids = {s: i for i, s in enumerate(self.instructions)}

        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.visual_encoder = nn.Sequential(
            b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.task_embedding = nn.Embedding(len(self.instructions), task_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(512 * views + task_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        if init is not None:
            self.load_encoder(init)

    def config(self) -> dict:
        return {
            "views": self.views,
            "instructions": self.instructions,
            "task_embed_dim": self.task_embed_dim,
        }

    @classmethod
    def from_checkpoint(cls, path: str, binarize: bool = False) -> "CWSRewardModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**ckpt["config"], binarize=binarize)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def load_encoder(self, path: str) -> None:
        """Load the visual encoder of a diffsynth Resnet/TaskEmbed reward checkpoint."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        encoder = {}
        for key, value in state.items():
            for prefix in ("net.", "visual_encoder."):
                if key.startswith(prefix):
                    rest = key[len(prefix) :]
                    if int(rest.split(".")[0]) < 5:  # b1..b5; the head is dropped
                        encoder[rest] = value
        self.visual_encoder.load_state_dict(encoder, strict=True)

    def task_ids(self, instructions: Union[str, Sequence]) -> torch.Tensor:
        if isinstance(instructions, (str, bytes)):
            instructions = [instructions]
        ids = []
        for inst in instructions:
            if hasattr(inst, "item"):
                inst = inst.item()
            if isinstance(inst, bytes):
                inst = inst.decode("utf-8")
            inst = str(inst).strip()
            if inst not in self._task_ids:
                raise ValueError(
                    f"Unknown instruction {inst!r}; known: {self.instructions}"
                )
            ids.append(self._task_ids[inst])
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, obs: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """obs [B, 3, H, W] in [-1, 1]; returns success logits [B]."""
        h, w = obs.shape[-2:]
        if self.views == 2:
            if h != 2 * w:
                raise ValueError(f"dual-view model needs stacked 2:1 frames, got {h}x{w}")
            feats = self.visual_encoder(torch.cat([obs[..., :w, :], obs[..., w:, :]], 0))
            feats = torch.cat(feats.chunk(2, 0), 1)
        else:
            if h == 2 * w:
                obs = obs[..., :w, :]
            feats = self.visual_encoder(obs)
        return self.head(torch.cat([feats, self.task_embedding(task_id)], 1)).squeeze(-1)

    @torch.no_grad()
    def predict_rew(self, obs: torch.Tensor, instruction) -> torch.Tensor:
        """World-model env interface: [B, 3, H, W] in [-1, 1] -> rewards [B, 1]."""
        param = next(self.parameters())
        obs = obs.to(device=param.device, dtype=param.dtype).clamp(-1.0, 1.0)
        reward = torch.sigmoid(self.forward(obs, self.task_ids(instruction).to(param.device)))
        if self.binarize:
            reward = torch.round(reward)
        return reward.unsqueeze(-1)
