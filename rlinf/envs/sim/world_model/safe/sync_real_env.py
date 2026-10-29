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

"""SafeWorldModelEnv with a synchronised real LIBERO simulator (physics only, no rendering).

At reset every slot restores a real simulator to the keyframe's saved snapshot; every action chunk
sent to Wan is also executed in that simulator. With ``reward_source: real`` the episode outcome
comes from the real simulator (check_success, official constraint violation), not from the reward
model: used for the R1 causal test and to label Wan frames for reward-model training. RM and SAM are
still computed and logged (``rm_success_once``) so the two can be compared.
Episodes end at the first real success, at an official violation, or after ``max_chunks`` chunks.
"""

from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import torch

from .safe_env import SafeWorldModelEnv

__all__ = ["SyncRealWorldModelEnv"]

EPISODES = "/data_cf/cws_sam/v1/labels/episodes.jsonl"
_SNAP_JSON = ("motion_generators", "controller_state", "gripper_state")


def _restore_types(value, template):
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype)
    if isinstance(value, dict) and isinstance(template, dict):
        return {k: _restore_types(v, template.get(k)) for k, v in value.items()}
    if isinstance(value, list) and isinstance(template, (tuple, list)):
        items = [_restore_types(v, template[i] if i < len(template) else None) for i, v in enumerate(value)]
        return tuple(items) if isinstance(template, tuple) else items
    return value


class SyncRealWorldModelEnv(SafeWorldModelEnv):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.reward_source = str(cfg.get("reward_source", "real"))  # real | rm
        self.end_on_violation = bool(cfg.get("end_on_violation", True))
        # data collection only: keep stepping this many chunks after the first real success and
        # record the per-step real goal state (frames can then be labelled by the actual state);
        # the success time and the outcome are unchanged. 0 = end at the first success (training).
        self.tail_chunks = int(cfg.get("success_tail_chunks", 0))
        # diagnostics only: per real step, record goal / wrong-grasp-part (aux) / official violation
        self.record_steps = bool(cfg.get("record_steps", False))
        if self.release_coef > 0:
            raise ValueError("SyncRealWorldModelEnv does not implement the release penalty")
        self._records = {}
        for line in open(cfg.get("episodes_jsonl", EPISODES)):
            r = json.loads(line)
            self._records[r["uid"]] = r
        meta = np.load(f"{cfg.keyframe_dir}/meta.npz")
        self._kf_uid = [str(u) for u in meta["uid"]]
        self._kf_t0 = meta["t0"].astype(np.int64)
        self._suites = {}
        self._pool = {}                       # (suite, task) -> idle real envs
        self._real = [None] * self.num_envs   # env used by each slot this episode
        self._real_key = [None] * self.num_envs
        self._raw_snaps = {}

    # ------------------------------------------------------------------ real simulator
    def _suite(self, name):
        if name not in self._suites:
            from rlinf.envs.sim.libero.utils import get_benchmark_overridden

            self._suites[name] = get_benchmark_overridden(name)()
        return self._suites[name]

    def _make_env(self, suite, task_id):
        from libero.libero.envs.env_wrapper import ControlEnv

        s = self._suite(suite)
        task = s.get_task(task_id)
        env = ControlEnv(bddl_file_name=s.get_task_bddl_file_path(task.level, task.level_id),
                         has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, horizon=100000)
        if hasattr(env, "seed"):
            env.seed(self.seed)
        env.reset()
        return env

    def _raw_snapshot(self, index):
        if index in self._raw_snaps:
            return self._raw_snaps[index]
        rec = self._records[self._kf_uid[index]]
        t0 = int(self._kf_t0[index])
        src, local = pathlib.Path(rec["json"]), t0
        if rec["kind"] != "root":
            branch_start = int(rec["branch_timestep"]) - 1
            if t0 < branch_start:
                src = pathlib.Path(rec["root_json"])
            else:
                local = t0 - branch_start
        source = json.loads(src.read_text())
        with np.load(src.parent / source["sim_state_trajectory"]["path"]) as saved:
            hit = np.flatnonzero(saved["timesteps"] == local + 1)
            if len(hit) != 1:
                raise RuntimeError(f"keyframe {index}: no unique snapshot for local step {local}")
            i = int(hit[0])
            snap = {n: saved[n][i].copy() for n in saved.files if n not in ("timesteps",) + _SNAP_JSON}
            for n in _SNAP_JSON:
                snap[n] = json.loads(str(saved[n][i]))
        for controller in snap["controller_state"]:
            for k, v in controller.items():
                if isinstance(v, list):
                    controller[k] = np.asarray(v)
        for grippers in snap["gripper_state"]:
            for gripper in grippers:
                for k, v in gripper.items():
                    if isinstance(v, list):
                        gripper[k] = np.asarray(v)
        self._raw_snaps[index] = snap
        return snap

    def _restore(self, slot, index):
        from rlinf.envs.sim.libero.venv import _restore_simulator_state, _unwrap_task_env

        rec = self._records[self._kf_uid[index]]
        key = (rec["suite"], int(rec["task"]))
        pool = self._pool.setdefault(key, [])
        env = pool.pop() if pool else self._make_env(*key)
        snap = copy.deepcopy(self._raw_snapshot(index))
        generators = getattr(_unwrap_task_env(env), "mocap_motion_generators", {})
        for name, state in snap["motion_generators"].items():
            snap["motion_generators"][name] = _restore_types(state, vars(generators[name]))
        _restore_simulator_state(env, snap)
        self._real[slot], self._real_key[slot] = env, key

    # ------------------------------------------------------------------ reset
    @torch.no_grad()
    def reset(self, *, seed=None, options=None, episode_indices=None, env_idx=None):
        obs, info = super().reset(seed=seed, options=options, episode_indices=episode_indices, env_idx=env_idx)
        for slot in range(self.num_envs):  # return last episode's envs to their task pools
            if self._real[slot] is not None:
                self._pool[self._real_key[slot]].append(self._real[slot])
                self._real[slot] = None
        for slot, index in enumerate(self.kf):
            self._restore(slot, int(index))
        n = self.num_envs
        self.real_succ = np.array([bool(e.check_success()) for e in self._real])
        self.real_success_at_reset = int(self.real_succ.sum())  # should be 0 for valid keyframes
        self.real_viol = np.zeros(n, bool)
        self.real_done = self.real_succ.copy()
        self.real_succ_step = np.where(self.real_succ, 0, -1)
        self.tail_left = np.full(n, -1)  # chunks still to run after the first real success
        self.step_rec = np.zeros((n, self.max_episode_steps, 3), bool)  # goal, wrong part, official violation
        self.rm_success_once = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_real_goal = np.zeros((n, self.chunk), bool)
        return obs, info

    # ------------------------------------------------------------------ dynamics
    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        from rlinf.envs.sim.libero.safety_auditor import audit_predicate_instances

        self.onload()
        a = policy_output_action
        a = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, np.float32)
        a = a.reshape(self.num_envs, self.chunk, -1)[..., :7].astype(np.float32)
        frames, z = self._generate(a)
        self.last_frames = frames                                           # [B, 8, 512, 256, 3]
        ft = torch.from_numpy(frames).to(self.device)
        depth = int(self._elapsed_steps[0].item()) // self.chunk
        s0 = depth * self.chunk
        live = ~self.ended
        live_np = live.cpu().numpy()

        # reward model on the agent view (logged; used only with reward_source == "rm")
        agent = ft[:, :, :256].reshape(-1, 256, 256, 3).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        inst = [s for s in self.task_descriptions for _ in range(self.chunk)]
        p_succ = self.reward_model.predict_rew(agent, inst).view(self.num_envs, self.chunk)
        self.last_p_succ = p_succ
        hit = p_succ >= self.success_threshold
        for _ in range(1, self.success_consecutive):
            hit = hit[:, 1:] & hit[:, :-1]
        rm_success = hit.any(1)
        self.rm_success_once |= live & rm_success

        # real simulator: same actions, stop at success / official violation
        goal = np.zeros((self.num_envs, self.chunk), bool)
        viol_now = np.zeros(self.num_envs, bool)
        succ_end = np.zeros(self.num_envs, bool)  # outcome decided as success in this chunk
        for i in np.flatnonzero(live_np & ~self.real_done):
            env = self._real[i]
            was_succ = bool(self.real_succ[i])
            for k in range(self.chunk):
                env.step(a[i, k])
                if self.record_steps:
                    preds = audit_predicate_instances(env)
                    self.step_rec[i, s0 + k] = (
                        bool(env.check_success()),
                        any(p["source"] == "aux_constraint" and p["predicate"] == "wronggrasppart" and p["value"]
                            for p in preds),
                        any(p["source"] == "constraint" and p["value"] for p in preds))
                if env.check_success():
                    goal[i, k] = True
                    if not self.real_succ[i]:
                        self.real_succ[i], self.real_succ_step[i] = True, s0 + k
                        self.tail_left[i] = self.tail_chunks
                        if self.tail_chunks == 0:
                            break
                    continue
                if (not self.real_succ[i] and self.end_on_violation
                        and any(p["source"] == "constraint" and p["value"] for p in audit_predicate_instances(env))):
                    viol_now[i] = True
                    self.real_viol[i], self.real_done[i] = True, True
                    break
            if self.real_succ[i]:
                if was_succ:
                    self.tail_left[i] -= 1
                if self.tail_left[i] <= 0:
                    succ_end[i], self.real_done[i] = True, True
        self.last_real_goal = goal

        # cost (logged, eta-weighted as in the parent) and state read-out
        logits = self.sam(z.to(self.device), torch.from_numpy(a).to(self.device), self.task_ids).float()
        prob = torch.sigmoid(logits).masked_fill(~self.chmask[:, None, :], 0.0)
        step_cost = prob.max(-1).values
        self.step_costs[live, s0:s0 + self.chunk] = step_cost[live]
        self.cost_by_depth[live, depth] = step_cost[live].mean(1)
        self.last_frame = ft[:, -1]
        self.states = self.state_reader(self.last_frame) + self.state_offset
        self._elapsed_steps = self._elapsed_steps + self.chunk

        if self.reward_source == "rm":
            success = rm_success
            failure = torch.zeros_like(rm_success)
        else:
            if self.tail_chunks:  # a tail cut by the horizon still counts as success
                succ_end |= live_np & self.real_succ & (self._elapsed_steps.cpu().numpy() >= self.max_episode_steps)
            success = torch.from_numpy(succ_end).to(self.device)
            failure = torch.from_numpy(viol_now).to(self.device)
        succ_now = live & success
        fail_now = live & failure & ~success
        trunc_now = live & ~success & ~failure & (self._elapsed_steps >= self.max_episode_steps)
        end_now = succ_now | fail_now | trunc_now
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
        terminations[:, -1] = succ_now | fail_now
        truncations[:, -1] = trunc_now
        n_steps = self._elapsed_steps.float().clamp(min=1)
        dev = self.device
        info = {"episode": {
            "success_once": self.success_once.clone(), "return": self.returns.clone(),
            "episode_len": n_steps, "reward": self.returns / n_steps,
            "cost": self.ep_cost.clone(), "eta": torch.full_like(self.returns, self.eta),
            "real_success": torch.from_numpy(self.real_succ.copy()).to(dev).float(),
            "real_violation": torch.from_numpy(self.real_viol.copy()).to(dev).float(),
            "rm_success_once": self.rm_success_once.float(),
            "rm_false_success": (self.rm_success_once & ~torch.from_numpy(self.real_succ.copy()).to(dev)).float(),
            **{f"cost_d{d + 1}": self.cost_by_depth[:, d].clone() for d in range(self.max_chunks)}}}
        return [self._wrap_obs()], rewards, terminations, truncations, [info]
