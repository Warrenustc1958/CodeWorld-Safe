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
Reward, success termination, and violation termination are independently configurable.
``outcome_order`` (legacy | step) and ``success_reward_rule`` (once | final) default to the
original behavior.
"""

from __future__ import annotations

import copy
import json
import os
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
        self.termination_source = str(cfg.get("termination_source", self.reward_source))  # real | rm | horizon
        self.end_on_violation = bool(cfg.get("end_on_violation", True))
        self.terminate_on_real_violation = bool(
            cfg.get("terminate_on_real_violation", self.reward_source == "real")
        )
        if self.reward_source not in ("real", "rm"):
            raise ValueError(f"Unknown reward_source: {self.reward_source}")
        if self.termination_source not in ("real", "rm", "horizon"):
            raise ValueError(f"Unknown termination_source: {self.termination_source}")
        # Continue physics after success so fixed-horizon runs retain paired labels.
        # Zero preserves the original early-stop behavior.
        self.tail_chunks = int(cfg.get("success_tail_chunks", 0))
        if self.termination_source == "horizon" and (
            self.end_on_violation or self.tail_chunks < self.max_chunks
        ):
            raise ValueError(
                "horizon termination requires end_on_violation=false and "
                "success_tail_chunks >= max_chunks"
            )
        # outcome_order: legacy = original chunk-level rule (a success anywhere in the chunk beats a
        # violation in the same chunk); step = decide by step order inside the chunk, a tie counts as
        # the violation (official protocol: success = goal reached without a prior violation).
        self.step_order = str(cfg.get("outcome_order", "legacy")) == "step"
        if str(cfg.get("outcome_order", "legacy")) not in ("legacy", "step"):
            raise ValueError(f"Unknown outcome_order: {cfg.get('outcome_order')}")
        # success_reward_rule: once = success at any time in the episode (original); final = success
        # still holds at the end of the episode (real: goal at the last executed step; rm: RM hit in
        # the episode's final chunk), so a transient false RM hit is not paid at the horizon.
        self.reward_rule = str(cfg.get("success_reward_rule", "once"))
        if self.reward_rule not in ("once", "final"):
            raise ValueError(f"Unknown success_reward_rule: {self.reward_rule}")
        # cost_source: sam = top-M mean of SAM step costs (original); real = 0/1 event cost from the synchronised
        # simulator (label and window below). SAM is still computed and logged as cost_sam. Termination still
        # follows the official predicates, so the benchmark protocol is unchanged.
        self.cost_source = str(cfg.get("cost_source", "sam"))
        if self.cost_source not in ("sam", "real"):
            raise ValueError(f"Unknown cost_source: {self.cost_source}")
        # cost_label (cost_source=real): grader = binary ``violated`` of safety_grader on all three channels
        # (c_obj / c_robot / c_grasp), the same implementation the real-eval grader uses; predicate = the
        # LIBERO official + auxiliary predicates (E11, legacy: check_robot_contact never fires, so c_robot is 0).
        self.cost_label = str(cfg.get("cost_label", "grader"))
        if self.cost_label not in ("grader", "predicate"):
            raise ValueError(f"Unknown cost_label: {self.cost_label}")
        # cost_window: goal = from the start up to and including the first real goal step; episode = all executed steps
        self.cost_window = str(cfg.get("cost_window", "goal"))
        if self.cost_window not in ("goal", "episode"):
            raise ValueError(f"Unknown cost_window: {self.cost_window}")
        if (self.cost_source == "real" and self.cost_label == "predicate"
                and os.environ.get("RLINF_LIBERO_SAFETY_AUX") != "1"):
            raise ValueError("cost_label=predicate needs RLINF_LIBERO_SAFETY_AUX=1 (auxiliary violations)")
        # obs_source: wan = policy sees Wan-generated frames (original); real = policy sees the synchronised simulator's
        # renders (agent + wrist, 256 px, rotated 180 deg like keyframes_v1) and its robot state. Wan is not called, so
        # SAM costs are unavailable: obs_source=real needs cost_source=real.
        self.obs_source = str(cfg.get("obs_source", "wan"))
        if self.obs_source not in ("wan", "real"):
            raise ValueError(f"Unknown obs_source: {self.obs_source}")
        if self.obs_source == "real" and self.cost_source != "real":
            raise ValueError("obs_source=real needs cost_source=real (SAM needs Wan latents)")
        # render_egl_remap (obs_source=real with MUJOCO_GL=egl): {worker CUDA device: EGL device index} so a worker
        # renders on another GPU (e.g. off GPUs with EGL faults). EGL indices follow eglQueryDevicesEXT, not CUDA ids.
        self.render_egl_remap = {int(k): int(v) for k, v in (cfg.get("render_egl_remap") or {}).items()}
        # diagnostics only: per real step, record goal / wrong-grasp-part (aux) / official violation
        self.record_steps = bool(cfg.get("record_steps", False))
        # reward_noise {fp, fn} (diagnostic, reward_source=real): per episode and independently of the trajectory, a real
        # failure is paid as success with probability fp and a real success as failure with probability fn
        noise = cfg.get("reward_noise") or {}
        self.noise_fp, self.noise_fn = float(noise.get("fp", 0.0)), float(noise.get("fn", 0.0))
        if not (0.0 <= self.noise_fp <= 1.0 and 0.0 <= self.noise_fn <= 1.0):
            raise ValueError(f"reward_noise rates must be in [0, 1], got {noise}")
        if (self.noise_fp or self.noise_fn) and self.reward_source != "real":
            raise ValueError("reward_noise needs reward_source=real")
        self._noise_rng = None
        # dump_dir: append one JSON line per finished episode (ids, real outcome, RM / SAM / grader signals, paid reward)
        self.dump_dir = cfg.get("dump_dir")
        self._reset_count = 0
        # dump_arrays: also write dump_dir/arrays/<pid>_<reset>_<slot>.npz per episode: JPEGs of the mid and last frame of
        # every chunk (agent over wrist), Wan latents, actions, and per-step goal / grader channels / executed masks
        self.dump_arrays = bool(cfg.get("dump_arrays", False))
        if self.dump_arrays and not self.dump_dir:
            raise ValueError("dump_arrays needs dump_dir")
        # rm_frames: in-chunk frame indices the RM scores (others count as 0), e.g. [3, 7] = the mid and end frame that
        # the phase-2 RM dataset uses; needs success_consecutive 1. None = all frames (original).
        rf = cfg.get("rm_frames", None)
        self.rm_frames = None if rf is None else sorted(int(x) for x in rf)
        if self.rm_frames is not None:
            if not self.rm_frames or self.rm_frames[0] < 0 or self.rm_frames[-1] >= self.chunk:
                raise ValueError(f"rm_frames must be in [0, {self.chunk}), got {rf}")
            if self.success_consecutive != 1:
                raise ValueError("rm_frames needs success_consecutive 1")
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
        if self.obs_source == "real":
            self._set_render_device()  # before LIBERO imports robosuite
        from libero.libero.envs.env_wrapper import ControlEnv

        s = self._suite(suite)
        task = s.get_task(task_id)
        if self.obs_source == "real":
            from libero.libero.envs.env_wrapper import OffScreenRenderEnv

            env = OffScreenRenderEnv(bddl_file_name=s.get_task_bddl_file_path(task.level, task.level_id),
                                     camera_heights=256, camera_widths=256, ignore_done=True, horizon=100000)
        else:
            env = ControlEnv(bddl_file_name=s.get_task_bddl_file_path(task.level, task.level_id),
                             has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, horizon=100000)
        if hasattr(env, "seed"):
            env.seed(self.seed)
        env.reset()
        return env

    def _set_render_device(self):
        """Apply render_egl_remap for this worker's GPU (first CUDA_VISIBLE_DEVICES entry) before a renderer exists."""
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
        target = os.environ.get("MUJOCO_EGL_DEVICE_ID")  # RLinf: EGL index of the worker's GPU
        if gpu.isdigit() and int(gpu) in self.render_egl_remap:
            target = str(self.render_egl_remap[int(gpu)])
        # robosuite's import-time check wants MUJOCO_EGL_DEVICE_ID to be a CUDA id in CUDA_VISIBLE_DEVICES, which an
        # EGL index generally is not; import it without the variable, the EGL context reads it again at creation.
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        import robosuite.utils.binding_utils  # noqa: F401
        if target is not None:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["EGL_DEVICE_ID"] = target

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
        from rlinf.envs.sim.libero.venv import (
            _restore_simulator_state,
            _unwrap_task_env,
        )

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

    def _grader_channels(self, slot):
        """[c_obj, c_robot, c_grasp] binary violations of the slot's real simulator (safety_grader, aux on)."""
        from rlinf.envs.sim.libero.safety_grader import CHANNELS, build_graders, violated_channels
        from rlinf.envs.sim.libero.venv import _unwrap_task_env

        task_env = _unwrap_task_env(self._real[slot])
        cache = getattr(task_env, "_rlinf_cost_graders", None)
        if cache is None or cache[0] is not task_env.sim.model:
            cache = (task_env.sim.model, build_graders(task_env, task_env.parsed_problem, aux=True))
            task_env._rlinf_cost_graders = cache
        v = violated_channels(cache[1], task_env)
        return np.array([v[c] for c in CHANNELS], bool)

    @staticmethod
    def _real_frame(obs):
        """Agent over wrist, uint8 [512, 256, 3], rotated 180 deg like the keyframes_v1 / Wan frames."""
        return np.concatenate([obs["agentview_image"][::-1, ::-1], obs["robot0_eye_in_hand_image"][::-1, ::-1]], 0)

    @staticmethod
    def _real_state(obs):
        """8-d robot state as in keyframes_v1: eef pos, eef axis-angle, gripper qpos."""
        from robosuite.utils.transform_utils import quat2axisangle

        return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                               obs["robot0_gripper_qpos"]]).astype(np.float32)

    def _real_obs_now(self, slot):
        return self._real[slot].env._get_observations(force_update=True)

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
        self.c_off = np.zeros(n, bool)  # official violation up to and including the first goal step
        self.c_aux = np.zeros(n, bool)  # auxiliary violation, same window
        # grader labels per channel (c_obj, c_robot, c_grasp): in the cost window / over the whole episode / at reset
        self.c_ch = np.zeros((n, 3), bool)
        self.c_ch_ep = np.zeros((n, 3), bool)
        self.c_ch_reset = np.zeros((n, 3), bool)
        if self.cost_source == "real" and self.cost_label == "grader":
            for i in range(n):
                self.c_ch_reset[i] = self._grader_channels(i)
        self.real_done = self.real_succ.copy()
        self.real_succ_step = np.where(self.real_succ, 0, -1)
        self.tail_left = np.full(n, -1)  # chunks still to run after the first real success
        self.step_rec = np.zeros((n, self.max_episode_steps, 3), bool)  # goal, wrong part, official violation
        self.rm_success_once = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.last_real_goal = np.zeros((n, self.chunk), bool)
        self.last_goal = self.real_succ.copy()  # goal predicate at the last executed real step
        self._reset_count += 1
        if self.noise_fp or self.noise_fn:  # one uniform draw per episode decides its flip
            if self._noise_rng is None:  # seeded per worker process so workers draw different flips
                self._noise_rng = np.random.default_rng([int(self.seed), os.getpid()])
            self.noise_u = self._noise_rng.random(n)
        self.dumped = np.zeros(n, bool)
        self.first_viol = np.full((n, 3), -1)  # first step of each grader channel violation (-1 = none)
        self.rm_chunk_max = np.full((n, self.max_chunks), np.nan, np.float32)
        if self.dump_arrays:
            self.arr_jpg, self.arr_z, self.arr_a = ([[] for _ in range(n)] for _ in range(3))
            self.step_goal = np.zeros((n, self.max_episode_steps), bool)
            self.step_ch = np.zeros((n, self.max_episode_steps, 3), bool)
            self.step_exec = np.zeros((n, self.max_episode_steps), bool)
        if self.obs_source == "real":
            o = [self._real_obs_now(i) for i in range(n)]
            self.last_frame = torch.from_numpy(np.stack([self._real_frame(x) for x in o])).to(self.device)
            self.states = torch.from_numpy(np.stack([self._real_state(x) for x in o])).to(self.device)
            obs = self._wrap_obs()
        return obs, info

    def _dump(self, slots, paid, C, C_sam, rewards):
        """Append one JSON line per finished episode to dump_dir/episodes_<pid>.jsonl (group = consecutive slots)."""
        if len(slots) == 0:
            return
        os.makedirs(os.path.join(self.dump_dir, "arrays") if self.dump_arrays else self.dump_dir, exist_ok=True)
        g = int(getattr(self, "group_size", 1) or 1)
        with open(os.path.join(self.dump_dir, f"episodes_{os.getpid()}.jsonl"), "a") as f:
            for i in slots:
                kf = int(self.kf[i])
                uid = self._kf_uid[kf]
                row = {
                    "reset": self._reset_count, "slot": int(i), "group": int(i) // g, "kf": kf, "uid": uid,
                    "suite": self._records.get(uid, {}).get("suite"),
                    "real_success": bool(self.real_succ[i]), "real_success_step": int(self.real_succ_step[i]),
                    "official_violation": bool(self.real_viol[i]),
                    "rm_success_once": bool(self.rm_success_once[i]),
                    "rm_chunk_max": [None if np.isnan(x) else round(float(x), 4) for x in self.rm_chunk_max[i]],
                    "cost_sam": round(float(C_sam[i]), 4), "cost_used": round(float(C[i]), 4),
                    "grader_window": self.c_ch[i].astype(int).tolist(),
                    "grader_episode": self.c_ch_ep[i].astype(int).tolist(),
                    "grader_at_reset": self.c_ch_reset[i].astype(int).tolist(),
                    "first_viol_step": [int(x) for x in self.first_viol[i]],
                    "sam_step_cost": [round(float(x), 3) for x in self.step_costs[i].tolist()],
                    "paid_success": bool(paid[i]), "reward": round(float(rewards[i, -1]), 4),
                    "eta": float(self.eta)}
                if self.dump_arrays:
                    name = f"{os.getpid()}_{self._reset_count}_{int(i)}.npz"
                    jpgs = self.arr_jpg[i]
                    np.savez(os.path.join(self.dump_dir, "arrays", name),
                             jpg=np.frombuffer(b"".join(jpgs), np.uint8),
                             jpg_off=np.cumsum([0] + [len(x) for x in jpgs]).astype(np.int64),
                             z=np.stack(self.arr_z[i]) if self.arr_z[i] else np.zeros(0, np.float16),
                             actions=np.stack(self.arr_a[i]), step_goal=self.step_goal[i],
                             step_ch=self.step_ch[i], step_exec=self.step_exec[i])
                    row["arrays"] = name
                f.write(json.dumps(row) + "\n")
                self.dumped[i] = True

    def _record_arrays(self, live_np, frames, z, a):
        """Keep, per live slot: JPEGs of the chunk's mid and last frame (agent over wrist), the Wan latent, the actions."""
        import io

        from PIL import Image

        zc = None if z is None else z.detach().to(torch.float16).cpu().numpy()
        for i in np.flatnonzero(live_np):
            for k in (self.chunk // 2 - 1, self.chunk - 1):
                buf = io.BytesIO()
                Image.fromarray(np.ascontiguousarray(frames[i, k])).save(buf, format="JPEG", quality=90)
                self.arr_jpg[i].append(buf.getvalue())
            if zc is not None:
                self.arr_z[i].append(zc[i])
            self.arr_a[i].append(np.asarray(a[i], np.float32))

    # ------------------------------------------------------------------ dynamics
    def _real_chunk(self, a, s0, live_np, frames, states_np=None):
        """Run the chunk in the real simulators (same actions); stop a slot at success / official violation.

        With ``states_np`` (obs_source=real) also writes each executed step's render into ``frames`` (later steps
        of a stopped slot repeat its last render) and the robot state into ``states_np``.
        Returns (goal [B, chunk], viol_now [B], viol_step [B], succ_end [B]).
        """
        from rlinf.envs.sim.libero.safety_auditor import audit_predicate_instances

        goal = np.zeros((self.num_envs, self.chunk), bool)
        viol_now = np.zeros(self.num_envs, bool)
        viol_step = np.full(self.num_envs, self.chunk)  # in-chunk step of the first new violation
        succ_end = np.zeros(self.num_envs, bool)  # outcome decided as success in this chunk
        for i in np.flatnonzero(live_np & ~self.real_done):
            env = self._real[i]
            was_succ = bool(self.real_succ[i])
            for k in range(self.chunk):
                step_out = env.step(a[i, k])
                if states_np is not None:  # obs_source=real: this step's render and robot state
                    step_obs = step_out[0]
                    frames[i, k:] = self._real_frame(step_obs)
                    states_np[i] = self._real_state(step_obs)
                preds = audit_predicate_instances(env)
                official_viol = any(
                    p["source"] == "constraint" and p["value"] for p in preds
                )
                is_goal = bool(env.check_success())
                new_violation = official_viol and not self.real_viol[i]
                self.real_viol[i] |= official_viol
                in_window = self.cost_window == "episode" or not self.real_succ[i]
                if in_window:  # goal window: before the first goal, the goal step included
                    self.c_off[i] |= official_viol
                    self.c_aux[i] |= any(p["source"] == "aux_constraint" and p["value"] for p in preds)
                ch = None
                if self.cost_source == "real" and self.cost_label == "grader":
                    ch = self._grader_channels(i)
                    self.first_viol[i][ch & (self.first_viol[i] < 0)] = s0 + k
                    self.c_ch_ep[i] |= ch
                    if in_window:
                        self.c_ch[i] |= ch
                self.last_goal[i] = is_goal
                if self.dump_arrays:
                    self.step_exec[i, s0 + k], self.step_goal[i, s0 + k] = True, is_goal
                    if ch is not None:
                        self.step_ch[i, s0 + k] = ch
                if self.record_steps:
                    self.step_rec[i, s0 + k] = (
                        is_goal,
                        any(p["source"] == "aux_constraint" and p["predicate"] == "wronggrasppart" and p["value"]
                            for p in preds),
                        official_viol)
                if self.step_order and new_violation and not self.real_succ[i]:
                    # a violating step cannot also be the success step
                    viol_now[i], viol_step[i] = True, min(viol_step[i], k)
                    if self.end_on_violation:
                        self.real_done[i] = True
                        break
                    continue
                if is_goal:
                    goal[i, k] = True
                    if not self.real_succ[i]:
                        self.real_succ[i], self.real_succ_step[i] = True, s0 + k
                        self.tail_left[i] = self.tail_chunks
                        if self.tail_chunks == 0:
                            break
                    continue
                if not self.step_order and not self.real_succ[i] and new_violation:
                    viol_now[i], viol_step[i] = True, min(viol_step[i], k)
                    if self.end_on_violation:
                        self.real_done[i] = True
                        break
            if self.real_succ[i]:
                if was_succ:
                    self.tail_left[i] -= 1
                if self.tail_left[i] <= 0:
                    succ_end[i], self.real_done[i] = True, True
        return goal, viol_now, viol_step, succ_end

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        self.onload()
        a = policy_output_action
        a = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, np.float32)
        a = a.reshape(self.num_envs, self.chunk, -1)[..., :7].astype(np.float32)
        if self.obs_source == "real":  # frames come from the simulator renders below; an ended slot repeats its last frame
            frames, z = np.repeat(self.last_frame.cpu().numpy()[:, None], self.chunk, 1), None
            states_np = self.states.cpu().numpy().copy()
        else:
            frames, z = self._generate(a)
        depth = int(self._elapsed_steps[0].item()) // self.chunk
        s0 = depth * self.chunk
        live = ~self.ended
        live_np = live.cpu().numpy()

        # real simulator: same actions, stop at success / official violation
        goal, viol_now, viol_step, succ_end = self._real_chunk(a, s0, live_np, frames,
                                                               states_np if self.obs_source == "real" else None)
        self.last_frames = frames                                           # [B, 8, 512, 256, 3]
        ft = torch.from_numpy(frames).to(self.device)
        if self.dump_arrays:
            self._record_arrays(live_np, frames, z, a)

        # reward model on the agent view (logged; used only with reward_source == "rm")
        # single-view RM: agent view (top 256 rows); dual-view RM: agent + wrist stacked (all 512 rows)
        rows = 256 * int(getattr(self.reward_model, "views", 1))
        fidx = self.rm_frames if self.rm_frames is not None else list(range(self.chunk))
        sub = ft[:, fidx, :rows]
        agent = sub.reshape(-1, rows, 256, 3).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        inst = [s for s in self.task_descriptions for _ in range(len(fidx))]
        p_sel = self.reward_model.predict_rew(agent, inst).view(self.num_envs, len(fidx))
        p_succ = torch.zeros(self.num_envs, self.chunk, device=p_sel.device, dtype=p_sel.dtype)
        p_succ[:, fidx] = p_sel  # unscored frames count as 0
        self.last_p_succ = p_succ
        self.rm_chunk_max[:, depth] = p_succ.max(1).values.float().cpu().numpy()
        hit = p_succ >= self.success_threshold
        for _ in range(1, self.success_consecutive):
            hit = hit[:, 1:] & hit[:, :-1]
        rm_success = hit.any(1)
        # in-chunk step at which the RM rule first fires (chunk = never)
        rm_step = torch.where(rm_success, hit.float().argmax(1) + self.success_consecutive - 1,
                              torch.full_like(rm_success, self.chunk, dtype=torch.long))
        self.rm_success_once |= live & rm_success

        self.last_real_goal = goal

        # cost (logged, eta-weighted as in the parent) and state read-out
        if z is not None:  # SAM needs the Wan latent (obs_source=wan); with real observations cost_sam stays 0
            logits = self.sam(z.to(self.device), torch.from_numpy(a).to(self.device), self.task_ids).float()
            prob = torch.sigmoid(logits).masked_fill(~self.chmask[:, None, :], 0.0)
            step_cost = prob.max(-1).values
            self.step_costs[live, s0:s0 + self.chunk] = step_cost[live]
            self.cost_by_depth[live, depth] = step_cost[live].mean(1)
        self.last_frame = ft[:, -1]
        if self.obs_source == "real":
            self.states = torch.from_numpy(states_np).to(self.device)
        else:
            self.states = self.state_reader(self.last_frame) + self.state_offset
        self._elapsed_steps = self._elapsed_steps + self.chunk

        if self.tail_chunks:  # a tail cut by the horizon still counts as success
            succ_end |= live_np & self.real_succ & (
                self._elapsed_steps.cpu().numpy() >= self.max_episode_steps
            )
        if self.termination_source == "real":
            terminate_success = torch.from_numpy(succ_end).to(self.device)
        elif self.termination_source == "rm":
            terminate_success = rm_success
        else:
            terminate_success = torch.zeros_like(rm_success)
        terminate_failure = (
            torch.from_numpy(viol_now).to(self.device)
            if self.terminate_on_real_violation
            else torch.zeros_like(rm_success)
        )
        if self.step_order and self.termination_source == "rm":
            # RM success counts only if it fires strictly before the first violation of the chunk
            rm_first = rm_step < torch.from_numpy(viol_step).to(self.device)
            terminate_success = terminate_success & (rm_first | ~terminate_failure)
        succ_now = live & terminate_success
        fail_now = live & terminate_failure & ~terminate_success
        trunc_now = live & ~succ_now & ~fail_now & (
            self._elapsed_steps >= self.max_episode_steps
        )
        end_now = succ_now | fail_now | trunc_now
        if self.reward_rule == "final":
            reward_success = (
                torch.from_numpy(self.last_goal.copy()).to(self.device)
                if self.reward_source == "real"
                else rm_success
            )
        else:
            reward_success = (
                torch.from_numpy(self.real_succ.copy()).to(self.device)
                if self.reward_source == "real"
                else self.rm_success_once
            )
        if self.noise_fp or self.noise_fn:  # diagnostic label noise, independent of the trajectory
            u = torch.from_numpy(self.noise_u).to(self.device)
            reward_success = torch.where(reward_success.bool(), u >= self.noise_fn, u < self.noise_fp)
        outcome_success = end_now & reward_success & ~fail_now
        m = min(self.top_m, s0 + self.chunk)
        C = self.step_costs[:, : s0 + self.chunk].topk(m, dim=1).values.mean(1)
        C_sam = C
        if self.cost_source == "real":
            event = self.c_ch.any(1) if self.cost_label == "grader" else self.c_off | self.c_aux
            C = torch.from_numpy(event).to(self.device).float()
        rewards = torch.zeros(self.num_envs, self.chunk, device=self.device)
        rewards[end_now, -1] = outcome_success[end_now].float() - self.eta * C[end_now]
        self.ep_cost = torch.where(end_now, C, self.ep_cost)
        self.success_once |= outcome_success
        self.returns += rewards.sum(1)
        self.ended |= end_now
        self._finished_costs.extend(C[end_now].tolist())
        if self.dump_dir:
            self._dump(np.flatnonzero(end_now.cpu().numpy() & ~self.dumped), outcome_success, C, C_sam, rewards)

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
            "cost_sam": C_sam.clone(),
            "cost_official": torch.from_numpy(self.c_off.copy()).to(dev).float(),
            "cost_aux": torch.from_numpy(self.c_aux.copy()).to(dev).float(),
            **{f"cost_{c}": torch.from_numpy(self.c_ch[:, j].copy()).to(dev).float()
               for j, c in enumerate(("obj", "robot", "grasp"))},
            **{f"unsafe_ep_{c}": torch.from_numpy(self.c_ch_ep[:, j].copy()).to(dev).float()
               for j, c in enumerate(("obj", "robot", "grasp"))},
            "viol_at_reset": torch.from_numpy(self.c_ch_reset.any(1)).to(dev).float(),
            "rm_false_success": (self.rm_success_once & ~torch.from_numpy(self.real_succ.copy()).to(dev)).float(),
            **{f"cost_d{d + 1}": self.cost_by_depth[:, d].clone() for d in range(self.max_chunks)}}}
        return [self._wrap_obs()], rewards, terminations, truncations, [info]
