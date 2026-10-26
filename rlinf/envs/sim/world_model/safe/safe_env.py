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

"""Safety-constrained world-model env for SC-GRPO (dual-view Wan + RM + SAM cost).

Each episode starts from a real keyframe (anchor frame 0, memory frames t0-3..t0 with their
actions, real proprio state at t0) and imagines at most ``max_chunks`` chunks of 8 steps with
the frozen dual-view Wan. Per chunk:
  frames  Wan generates 8 dual frames (agent over wrist, 512x256); the latent before VAE
          decoding is kept for the cost head
  reward  RM (agent view) success probability; success if max over the chunk >= threshold
  cost    SAM on the generated latent + executed actions: per step, max over the task's
          applicable channels of sigmoid(logit) (continuous probability)
  state   pi0.5 proprio read from the last generated frame by a frozen CNN regressor
An episode ends at the first success or after ``max_chunks``. Its reward, placed on the last
step of the ending chunk, is  1{success} - eta * C  with C = mean of the top-M step costs of the
episode (SafeDojo). eta is a Lagrange multiplier updated once per rollout:
  eta <- max(0, eta + eta_lr * (mean C over finished episodes - cost_budget)).
Chunk depth 1 is covered by the Step 8 calibration; depths 2..max_chunks are not (heuristic).
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rlinf.scheduler import Worker, WorkerInfo

from .safe_models import NULL_ACTION, SAMNet, StateReader, enable_dual, scale_pose

__all__ = ["SafeWorldModelEnv"]


class SafeWorldModelEnv:
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, record_metrics=True,
                 worker_info: Optional[WorkerInfo] = None):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = int(cfg.seed) + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics
        self.device = torch.device(Worker.torch_device_type or "cpu")
        if Worker.torch_device_type == "cuda":
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if cfg.get("auto_reset", False):
            raise ValueError("SafeWorldModelEnv runs fixed-length episodes; set auto_reset: False")
        self.auto_reset = False
        self.ignore_terminations = cfg.get("ignore_terminations", False)
        self.group_size = int(cfg.group_size)
        self.num_group = num_envs // self.group_size
        self.chunk = 8
        self.max_chunks = int(cfg.max_chunks)
        self.max_episode_steps = self.chunk * self.max_chunks
        self.success_threshold = float(cfg.success_threshold)
        self.success_consecutive = int(cfg.get("success_consecutive", 1))
        self.use_state_offset = bool(cfg.get("state_offset", True))
        self.top_m = int(cfg.cost_top_m)
        self.cost_budget = float(cfg.cost_budget)
        self.eta_lr = float(cfg.eta_lr)
        self.eta = float(cfg.get("eta_init", 0.0))
        self.pose_scale, self.pose_clip = float(cfg.pose_scale), float(cfg.pose_clip)
        self.main_size, self.wrist_size = int(cfg.main_image_size), int(cfg.wrist_image_size)
        self.video_cfg = cfg.get("video_cfg", None)

        # keyframe pool (memmapped) and per-cell applicable cost channels
        kf = cfg.keyframe_dir
        self.kf_anchor = np.load(f"{kf}/anchor.npy", mmap_mode="r")
        self.kf_memory = np.load(f"{kf}/memory.npy", mmap_mode="r")
        meta = np.load(f"{kf}/meta.npz")
        self.kf_mem_actions = meta["mem_actions"].astype(np.float32)
        self.kf_states = meta["states"].astype(np.float32)
        self.kf_task = meta["task_id"].astype(np.int64)
        self.kf_inst = [str(s) for s in meta["instruction"]]
        self.kf_tcell = [str(s) for s in meta["tcell"]]
        if cfg.get("keyframe_suites", None):
            keep = np.isin(meta["suite"], list(cfg.keyframe_suites))
            self.kf_pool = np.flatnonzero(keep)
        else:
            self.kf_pool = np.arange(len(self.kf_task))
        cmask = json.load(open(cfg.channel_mask))
        self.kf_chmask = np.array([cmask[t] for t in self.kf_tcell], bool)

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self._batch_counter = 0
        self._is_start = True
        self._is_offloaded = False
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.long)
        self._finished_costs: list[float] = []
        self.update_reset_state_ids()

        self._build_models()
        self.task_descriptions = [""] * num_envs

    # ------------------------------------------------------------------ models
    def _build_models(self):
        from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=str(self.device),
            model_configs=[ModelConfig(path=self.cfg.model_path, offload_device="cpu"),
                           ModelConfig(path=self.cfg.VAE_path, offload_device="cpu")])
        pipe.dit.to(self.device)
        pipe.vae.to(self.device)
        enable_dual(pipe)
        self._latent = {}
        orig = pipe.vae.decode

        def decode(hidden_states, *a, **k):  # keep the generated latent the pipeline decodes
            self._latent["z"] = hidden_states.detach()
            return orig(hidden_states, *a, **k)

        pipe.vae.decode = decode
        self.pipe = pipe
        self.sam = SAMNet.from_checkpoint(self.cfg.sam_path).to(self.device)
        self.state_reader = StateReader(self.cfg.state_reader_path).to(self.device)
        from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel

        self.reward_model = CWSRewardModel.from_checkpoint(self.cfg.reward_model.from_pretrained).eval().to(self.device)

    # ------------------------------------------------------------------ gym-ish properties
    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @elapsed_steps.setter
    def elapsed_steps(self, value):
        self._elapsed_steps = value

    # ------------------------------------------------------------------ reset
    def update_reset_state_ids(self):
        """Called once per rollout (finish_rollout): Lagrange multiplier update."""
        if self._finished_costs:
            mean_c = float(np.mean(self._finished_costs))
            self.eta = max(0.0, self.eta + self.eta_lr * (mean_c - self.cost_budget))
            self._last_mean_cost = mean_c
            self._finished_costs = []

    def _sample_keyframes(self):
        g = torch.randint(0, len(self.kf_pool), (self.num_group,), generator=self._generator).numpy()
        return np.repeat(self.kf_pool[g], self.group_size)

    @torch.no_grad()
    def reset(self, *, seed=None, options=None, episode_indices=None, env_idx=None):
        if env_idx is not None:
            raise NotImplementedError("SafeWorldModelEnv resets all slots at once")
        self.onload()
        kf = self._sample_keyframes() if episode_indices is None else np.asarray(episode_indices)
        order = np.argsort(kf)
        anchor = np.empty((self.num_envs,) + self.kf_anchor.shape[1:], np.uint8)
        memory = np.empty((self.num_envs,) + self.kf_memory.shape[1:], np.uint8)
        anchor[order] = self.kf_anchor[kf[order]]
        memory[order] = self.kf_memory[kf[order]]
        self.kf = kf
        self.anchor = [Image.fromarray(a) for a in anchor]
        self.memory = [[Image.fromarray(m) for m in mem] for mem in memory]
        self.mem_actions = self.kf_mem_actions[kf].copy()             # [B, 4, 7] env units
        self.task_ids = torch.from_numpy(self.kf_task[kf]).to(self.device)
        self.chmask = torch.from_numpy(self.kf_chmask[kf]).to(self.device)
        self.task_descriptions = [self.kf_inst[i] for i in kf]
        self.states = torch.from_numpy(self.kf_states[kf]).to(self.device)
        self.last_frame = torch.from_numpy(memory[:, -1]).to(self.device)  # [B, 512, 256, 3] uint8
        # per-scene regressor bias: real keyframe state minus the reading of the same frame
        if self.use_state_offset:
            self.state_offset = self.states - self.state_reader(self.last_frame)
        else:
            self.state_offset = torch.zeros_like(self.states)
        self.step_costs = torch.zeros(self.num_envs, self.max_episode_steps, device=self.device)
        self.ended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.ep_cost = torch.zeros(self.num_envs, device=self.device)
        self.cost_by_depth = torch.zeros(self.num_envs, self.max_chunks, device=self.device)
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._is_start = False
        return self._wrap_obs(), {}

    # ------------------------------------------------------------------ observation
    def _wrap_obs(self):
        top, bottom = self.last_frame[:, :256], self.last_frame[:, 256:]

        def rs(x, s):
            if x.shape[1] == s and x.shape[2] == s:
                return x
            y = F.interpolate(x.permute(0, 3, 1, 2).float(), size=(s, s), mode="bilinear", align_corners=False)
            return y.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        return {"main_images": rs(top, self.main_size), "wrist_images": rs(bottom, self.wrist_size),
                "states": self.states.float(), "task_descriptions": list(self.task_descriptions)}

    # ------------------------------------------------------------------ dynamics
    def _generate(self, actions: np.ndarray):
        """actions [B, 8, 7] env units -> new frames uint8 [B, 8, 512, 256, 3], latent [B,48,4,32,16]."""
        window = np.concatenate([np.repeat(NULL_ACTION[None, None], self.num_envs, 0), self.mem_actions, actions], 1)
        window = torch.from_numpy(scale_pose(window, self.pose_scale, self.pose_clip)).to(self.device)
        self._batch_counter += 1
        seed = (self.seed * 1_000_003 + self._batch_counter) % (2 ** 31)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.pipe(seed=seed, tiled=False, input_image=self.anchor, input_image4=self.memory,
                            action=window, height=512, width=256, num_frames=13,
                            num_inference_steps=int(self.cfg.num_inference_steps), cfg_scale=1.0,
                            progress_bar_cmd=lambda x: x, batch_size=self.num_envs)
        frames = np.stack([np.stack([np.asarray(im, np.uint8) for im in o[5:13]]) for o in out])
        self.memory = [list(o[9:13]) for o in out]
        self.mem_actions = actions[:, -4:].copy()
        return frames, self._latent["z"].float()

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        self.onload()
        a = policy_output_action
        a = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, np.float32)
        a = a.reshape(self.num_envs, self.chunk, -1)[..., :7].astype(np.float32)
        frames, z = self._generate(a)
        ft = torch.from_numpy(frames).to(self.device)                   # [B, 8, 512, 256, 3]
        depth = int(self._elapsed_steps[0].item()) // self.chunk           # 0-based chunk index

        # reward: RM on the agent view of the 8 new frames
        agent = ft[:, :, :256].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        inst = [s for s in self.task_descriptions for _ in range(self.chunk)]
        p_succ = self.reward_model.predict_rew(agent, inst).view(self.num_envs, self.chunk)
        self.last_p_succ = p_succ
        hit = p_succ >= self.success_threshold
        for k in range(1, self.success_consecutive):  # k consecutive frames over the threshold
            hit = hit[:, 1:] & hit[:, :-1]
        success = hit.any(1)

        # cost: SAM on the generated latent, continuous probability, task-applicable channels
        logits = self.sam(z.to(self.device), torch.from_numpy(a).to(self.device), self.task_ids).float()
        prob = torch.sigmoid(logits).masked_fill(~self.chmask[:, None, :], 0.0)
        step_cost = prob.max(-1).values                                    # [B, 8]
        live = ~self.ended
        s0 = depth * self.chunk
        self.step_costs[live, s0:s0 + self.chunk] = step_cost[live]
        self.cost_by_depth[live, depth] = step_cost[live].mean(1)

        # state: read from the last generated frame
        self.last_frame = ft[:, -1]
        self.states = self.state_reader(self.last_frame) + self.state_offset
        self._elapsed_steps = self._elapsed_steps + self.chunk

        trunc_now = live & ~success & (self._elapsed_steps >= self.max_episode_steps)
        succ_now = live & success
        end_now = succ_now | trunc_now
        n_steps = (self._elapsed_steps.float()).clamp(min=1)
        m = min(self.top_m, s0 + self.chunk)
        C = self.step_costs[:, : s0 + self.chunk].topk(m, dim=1).values.mean(1)
        rewards = torch.zeros(self.num_envs, self.chunk, device=self.device)
        rewards[end_now, -1] = succ_now[end_now].float() - self.eta * C[end_now]
        self.ep_cost = torch.where(end_now, C, self.ep_cost)
        self.success_once |= succ_now
        self.returns += rewards.sum(1)
        self.ended |= end_now
        self._finished_costs.extend(C[end_now].tolist())

        terminations = torch.zeros(self.num_envs, self.chunk, dtype=torch.bool, device=self.device)
        truncations = torch.zeros_like(terminations)
        terminations[:, -1] = succ_now
        truncations[:, -1] = trunc_now
        info = {"episode": {"success_once": self.success_once.clone(), "return": self.returns.clone(),
                            "episode_len": n_steps, "reward": self.returns / n_steps,
                            "cost": self.ep_cost.clone(), "eta": torch.full_like(self.returns, self.eta),
                            **{f"cost_d{d + 1}": self.cost_by_depth[:, d].clone() for d in range(self.max_chunks)}}}
        return [self._wrap_obs()], rewards, terminations, truncations, [info]

    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("use chunk_step")

    # ------------------------------------------------------------------ device management
    def offload(self):
        if self._is_offloaded:
            return
        for m in (self.pipe.dit, self.pipe.vae, self.sam, self.state_reader, self.reward_model):
            m.to("cpu")
        torch.cuda.empty_cache()
        self._is_offloaded = True

    def onload(self):
        if not self._is_offloaded:
            return
        self.pipe.dit.to(self.device)
        self.pipe.vae.to(self.device)
        for m in (self.sam, self.state_reader, self.reward_model):
            m.to(self.device)
        self._is_offloaded = False
