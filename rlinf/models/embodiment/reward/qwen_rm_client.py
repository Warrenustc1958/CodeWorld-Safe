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

"""Client for a Qwen3-VL success RM served over local HTTP (toolkits/codeworld_safe/rm2/qwen_rm_server.py).

Qwen3-VL needs a newer transformers than the openpi runtime, so the model runs in its own process (one per GPU) and
the env worker sends JPEG frames. Same interface as CWSRewardModel: predict_rew(obs [B, 3, H, W] in [-1, 1],
instructions) -> [B, 1] success probabilities. The server port is port_base + the first CUDA_VISIBLE_DEVICES entry.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import numpy as np
import torch
from PIL import Image


class QwenRMClient(torch.nn.Module):
    def __init__(self, port_base: int, views: int = 2, host: str = "127.0.0.1", timeout: float = 600.0,
                 quality: int = 90):
        super().__init__()
        self.port_base, self.views, self.host, self.timeout, self.quality = int(port_base), int(views), host, \
            float(timeout), int(quality)
        # a parameter so .to() / next(parameters()) behave like the other reward models
        self.register_parameter("_anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

    def url(self) -> str:
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        return f"http://{self.host}:{self.port_base + int(gpu)}/score"

    def _jpegs(self, obs: torch.Tensor) -> list[str]:
        x = ((obs.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        x = x.permute(0, 2, 3, 1).cpu().numpy()
        out = []
        for img in x:
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=self.quality)
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out

    @torch.no_grad()
    def predict_rew(self, obs: torch.Tensor, instruction) -> torch.Tensor:
        inst = [instruction] * obs.shape[0] if isinstance(instruction, str) else [str(s) for s in instruction]
        body = json.dumps({"jpg": self._jpegs(obs), "instructions": inst}).encode()
        req = urllib.request.Request(self.url(), data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        if "p" not in out or len(out["p"]) != obs.shape[0]:
            raise RuntimeError(f"Qwen RM server error: {out.get('error', out)}")
        return torch.tensor(out["p"], dtype=torch.float32, device=obs.device).unsqueeze(-1)
