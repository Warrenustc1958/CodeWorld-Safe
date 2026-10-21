"""Offline check of an affordance near-miss definition (plan B), CPU only, read-only.

For each safe (not unsafe) affordance episode already written by Step E, restore every
saved sim_state snapshot, replay the recorded actions up to the next snapshot, and at
every step measure (drift_max = max |eef - recorded eef| checks the replay):

  grasp_ok : both fingerpads touch the intended part (LIBERO check_gripper_contact)
  d_fp     : min distance from fingerpad geoms to wrong-part geoms
  d_all    : min distance from all robot geoms to wrong-part geoms (current c_grasp min_dist)

Episode-level: d_fp_grasp = min d_fp over snapshots with grasp_ok (None if never grasped).
Plan-B near-miss = d_fp_grasp < 0.02 and the episode is safe.

    python toolkits/codeworld_safe/agent/verify_grasp_nm.py --out <dir> [--max-eps 80] [--workers 10]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
KNOWLEDGE = REPO / "toolkits/codeworld_safe/agent/knowledge/tasks"
LIBERO = Path(os.environ.get("LIBERO_SAFETY_PATH", "/mnt/nfs/fjj/Code/LIBERO-Safety"))
RUNS = Path(os.environ.get("CWS_STEP_E", "/data_cf/cws/step_e")) / "runs"
NM = 0.02


def _fingerpad_ids(task_env) -> list[int]:
    g = task_env.robots[0].gripper
    names = list(g.important_geoms["left_fingerpad"]) + list(g.important_geoms["right_fingerpad"])
    return [task_env.sim.model.geom_name2id(n) for n in names]


def _arrays(v):
    """JSON snapshot fields back to numpy: numeric lists -> arrays, recursively."""
    if isinstance(v, dict):
        return {k: _arrays(x) for k, x in v.items()}
    if isinstance(v, list):
        try:
            a = np.asarray(v, dtype=np.float64)
            if a.dtype != object:
                return a
        except (TypeError, ValueError):
            pass
        return [_arrays(x) for x in v]
    return v


def _restore(env, snap):
    from rlinf.envs.sim.libero.venv import _restore_simulator_state

    try:
        _restore_simulator_state(env, snap)
    except Exception as e:  # noqa: BLE001 - render context is absent offline; physics is already set
        if "render" not in str(e).lower() and "context" not in str(e).lower():
            raise


def run_cell(args):
    key, max_eps = args
    import torch
    from libero.libero.envs.env_wrapper import ControlEnv
    from rlinf.envs.sim.libero.safety_grader import WrongGraspPartGrader, _min_distance, build_graders
    from rlinf.envs.sim.libero.venv import _unwrap_task_env

    info = json.loads((KNOWLEDGE / f"{key}.json").read_text())
    env = ControlEnv(bddl_file_name=str(LIBERO / info["bddl_file"]), has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    inits = torch.load(str(LIBERO / info["init_states_file"]), weights_only=False)
    env.reset()
    env.set_init_state(inits[0])
    task_env = _unwrap_task_env(env)
    graders = [g for g in build_graders(task_env, task_env.parsed_problem, aux=True)
               if isinstance(g, WrongGraspPartGrader)]
    if not graders:
        return {"key": key, "error": "no wronggrasppart grader"}
    gr = graders[0]
    fp = _fingerpad_ids(task_env)
    sim = task_env.sim
    site = sim.model.site_name2id("gripper0_grip_site")

    eps = []
    files = sorted(glob.glob(str(RUNS / key / "r*/**/process-*.json"), recursive=True))
    for f in files:
        if len(eps) >= max_eps:
            break
        try:
            m = json.loads(Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        if m.get("unsafe"):
            continue
        ss = m.get("sim_state_trajectory") or {}
        npz = Path(f).parent / ss.get("path", "")
        if not npz.is_file():
            continue
        z = np.load(npz, allow_pickle=False)
        traj = {s["timestep"]: s for s in m["trajectory"]}
        last_t = max(traj)
        rows, drift, replay_end = [], [], None
        ts = [int(t) for t in z["timesteps"]]
        for k, t0 in enumerate(ts):
            # Restore the snapshot taken right before step t0, then replay the recorded
            # actions up to the next snapshot (snapshots are only every 8 steps, which
            # misses the grasp: success ends the episode on the first correct contact).
            snap = {fk: z[fk][k] for fk in ("mujoco_state", "mocap_pos", "mocap_quat", "act",
                                            "ctrl", "qacc_warmstart", "qfrc_applied",
                                            "xfrc_applied") if fk in z.files}
            snap["cur_time"] = float(z["cur_time"][k])
            for fk in ("motion_generators", "controller_state", "gripper_state"):
                snap[fk] = _arrays(json.loads(str(z[fk][k])))
            _restore(env, snap)
            if replay_end is not None:
                # Replay check: where the replay of the previous chunk ended vs this snapshot.
                drift.append(float(np.linalg.norm(sim.data.site_xpos[site] - replay_end)))
                replay_end = None
            t1 = ts[k + 1] if k + 1 < len(ts) else last_t + 1
            for t in range(t0, t1):
                if t not in traj:
                    break
                env.step(np.asarray(traj[t]["action"], dtype=np.float64))
                rows.append((t, bool(task_env.check_gripper_contact(gr._intended_names)),
                             _min_distance(task_env, fp, gr._b), _min_distance(task_env, gr._a, gr._b)))
            if k + 1 < len(ts):
                replay_end = np.array(sim.data.site_xpos[site])
        if not rows:
            continue
        g = [r[2] for r in rows if r[1]]
        eps.append({"file": f, "success": bool(m.get("success")), "n_steps": len(rows),
                    "n_grasp": len(g), "d_fp_grasp": min(g) if g else None,
                    "d_fp": min(r[2] for r in rows), "d_all": min(r[3] for r in rows),
                    "drift_max": max(drift) if drift else None})
    env.close()
    return {"key": key, "eps": eps}


def _q(v, p):
    return round(float(np.quantile(v, p)), 4) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-eps", type=int, default=80)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.name for p in RUNS.iterdir() if p.name.startswith("affordance_"))
    with get_context("spawn").Pool(min(a.workers, len(keys))) as pool:
        res = pool.map(run_cell, [(k, a.max_eps) for k in keys])
    lines = ["cell\tsafe_eps\tgrasped_eps\td_fp_grasp p10/p50/p90\tNM_B(<2cm)\td_all p50\tNM_old(<2cm)"
             "\tdrift_max p50/max"]
    for r in res:
        (out / f"{r['key']}.json").write_text(json.dumps(r, indent=1))
        if "error" in r:
            lines.append(f"{r['key']}\tERROR {r['error']}")
            continue
        e = r["eps"]
        dg = [x["d_fp_grasp"] for x in e if x["d_fp_grasp"] is not None]
        da = [x["d_all"] for x in e]
        dr = [x["drift_max"] for x in e if x["drift_max"] is not None]
        lines.append(f"{r['key']}\t{len(e)}\t{len(dg)}\t{_q(dg, .1)}/{_q(dg, .5)}/{_q(dg, .9)}\t"
                     f"{sum(d < NM for d in dg)}\t{_q(da, .5)}\t{sum(d < NM for d in da)}\t"
                     f"{_q(dr, .5)}/{_q(dr, 1.0)}")
    (out / "summary.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
