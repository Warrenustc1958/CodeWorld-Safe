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

"""Frame dataset over the JPEG shards written by ``build_frames.py``.

Any frame set in the same format (e.g. world-model generated frames labelled from
the paired real trajectory) can be mixed in with ``ConcatDataset``, as long as it
shares ``tasks.json``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VIEWS = ("agent", "wrist")


class RMFrames(torch.utils.data.Dataset):
    """Item: (uint8 [V*256, 256, 3] with agent on top, label, task_id, index)."""

    def __init__(self, root, split: str, views: int):
        self.root = Path(root)
        self.split = split
        self.views = VIEWS[:views]
        with np.load(self.root / f"{split}_meta.npz") as m:
            self.meta = {k: m[k] for k in m.files}
        self.offsets = {v: np.load(self.root / f"{split}_{v}.off.npy") for v in self.views}
        self.episodes = [json.loads(l) for l in (self.root / f"{split}_episodes.jsonl").read_text().splitlines()]
        self._bins = None  # opened lazily, once per dataloader worker

    def __len__(self):
        return len(self.meta["label"])

    def _decode(self, view: str, i: int) -> np.ndarray:
        if self._bins is None:
            self._bins = {
                v: np.memmap(self.root / f"{self.split}_{v}.bin", dtype=np.uint8, mode="r")
                for v in self.views
            }
        off = self.offsets[view]
        blob = self._bins[view][off[i] : off[i + 1]].tobytes()
        return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))

    def __getitem__(self, i):
        frame = np.concatenate([self._decode(v, i) for v in self.views], axis=0)
        return (
            torch.from_numpy(frame),
            torch.tensor(float(self.meta["label"][i])),
            int(self.meta["task_id"][i]),
            i,
        )
