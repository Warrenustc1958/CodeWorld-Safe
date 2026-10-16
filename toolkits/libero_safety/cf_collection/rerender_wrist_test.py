"""Kinematic re-render test: write saved per-step poses into MuJoCo qpos and render.

Rendering depends only on qpos, so if the saved poses are complete, the
re-rendered agentview should match the saved NPZ up to the render residual seen
at reset (~2 mean abs diff). If it does, the same pass yields wrist frames.
"""
import json
import os
import sys

import numpy as np

from rlinf.envs.sim.libero.utils import get_benchmark_overridden
from libero.libero.envs import OffScreenRenderEnv

ep_json = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/rerender_wrist_test"
os.makedirs(out_dir, exist_ok=True)

j = json.load(open(ep_json))
frames = np.load(os.path.join(os.path.dirname(ep_json), j["visual_trajectory"]["path"]))["agentview_image"]
ep = j["episode"]
suite = get_benchmark_overridden(ep["task_suite"])()
task = suite.get_task(ep["task_id"])
bddl = suite.get_task_bddl_file_path(task.level, task.level_id)
init_states = suite.get_task_init_states(task.level, task.level_id)

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, ignore_done=True)
env.seed(0)
env.reset()
env.set_init_state(init_states[ep["trial_id"]])
inner = env.env
sim = inner.sim
robot = inner.robots[0]

# Map "<name>_pos"/"<name>_quat" observation keys to free-joint qpos addresses.
free_joints = {}
for name in list(getattr(inner, "objects_dict", {}).keys()) + list(getattr(inner, "fixtures_dict", {}).keys()):
    obj = inner.objects_dict.get(name) if name in getattr(inner, "objects_dict", {}) else inner.fixtures_dict[name]
    for jn in getattr(obj, "joints", []) or []:
        jid = sim.model.joint_name2id(jn)
        if sim.model.jnt_type[jid] == 0:  # free joint
            free_joints[name] = sim.model.jnt_qposadr[jid]
print("free-joint objects:", sorted(free_joints))


def apply_obs(o):
    qpos = sim.data.qpos.copy()
    qpos[robot._ref_joint_pos_indexes] = np.asarray(o["robot0_joint_pos"], dtype=float)
    qpos[robot._ref_gripper_joint_pos_indexes] = np.asarray(o["robot0_gripper_qpos"], dtype=float)
    missing = []
    for name, adr in free_joints.items():
        if f"{name}_pos" not in o:
            missing.append(name)
            continue
        x, y, z, w = np.asarray(o[f"{name}_quat"], dtype=float)  # robosuite obs quat is xyzw
        qpos[adr:adr + 3] = np.asarray(o[f"{name}_pos"], dtype=float)
        qpos[adr + 3:adr + 7] = [w, x, y, z]
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = 0
    sim.forward()
    inner._update_observables(force=True)
    return inner._get_observations(), missing


def rot(im):
    return np.ascontiguousarray(im[::-1, ::-1, :3])


obs_list = [j["initial_observation"]] + [s["observation"] for s in j["trajectory"]]
diff_a, diff_b, wrist, missing_any = [], [], [], set()
for t, o in enumerate(obs_list):
    obs, missing = apply_obs(o)
    missing_any.update(missing)
    av = rot(obs["agentview_image"]).astype(int)
    wrist.append(rot(obs["robot0_eye_in_hand_image"]))
    if t < len(frames):
        diff_a.append(np.abs(av - frames[t].astype(int)).mean())  # obs[t] <-> frame[t]
    if 0 < t and t - 1 < len(frames):
        diff_b.append(np.abs(av - frames[t - 1].astype(int)).mean())  # obs[t] <-> frame[t-1]
env.close()

a, b = np.array(diff_a), np.array(diff_b)
print(f"episode={ep['episode_id']} kind={ep.get('episode_kind','root')} n_obs={len(obs_list)} n_frames={len(frames)}")
print(f"missing pose keys: {sorted(missing_any)}")
print(f"align obs[t]<->frame[t]:   median={np.median(a):.2f} p95={np.percentile(a,95):.2f} max={a.max():.2f}")
print(f"align obs[t]<->frame[t-1]: median={np.median(b):.2f} p95={np.percentile(b,95):.2f} max={b.max():.2f}")
np.savez_compressed(os.path.join(out_dir, ep["episode_id"] + "_wrist.npz"), wrist=np.stack(wrist))
