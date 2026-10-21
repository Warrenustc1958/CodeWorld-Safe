#!/usr/bin/env python3
"""Kinematic regrade of recorded LIBERO-Safety audits (no physics replay).

Physics replay of legacy pi0.5 audits is not reproducible (unseeded RNG in settle),
so each recorded step is rebuilt from its logged observation instead: robot joint and
gripper qpos, every object pose (free joint or mocap) -> mj_forward -> safety_grader.
Contacts / distances are exact for the logged poses; normal_force is not (no dynamics).
Fixture joints (drawers, knobs) are not in legacy observations and stay at init.

  --validate GLOB   new Step E audits with recorded grades: compare rebuilt vs recorded
  --kept FILE       legacy audits: new quadrant / near-miss labels
    python -m toolkits.codeworld_safe.agent.kin_regrade (--validate G | --kept F) --out DIR
        [--per-cell N] [--workers 32]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
import traceback
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from toolkits.codeworld_safe.agent.regrade_legacy import KNOWLEDGE, LIBERO, NEAR_MISS_M, _cell

_QUAD = {(True, False): "Q1", (True, True): "Q2", (False, False): "Q3", (False, True): "Q4"}


def _env(cell, cache):
    if cache.get("cell") == cell:
        return cache["env"]
    import torch
    from libero.libero.envs.env_wrapper import ControlEnv

    if cache.get("env") is not None:
        cache["env"].close()
    info = json.loads((KNOWLEDGE / f"{cell}.json").read_text())
    env = ControlEnv(bddl_file_name=str(LIBERO / info["bddl_file"]), has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    cache.update(cell=cell, env=env,
                 inits=torch.load(str(LIBERO / info["init_states_file"]), weights_only=False))
    return env


class Rebuilder:
    """Writes one logged observation into sim qpos / mocap and runs mj_forward."""

    def __init__(self, task_env):
        sim, m = task_env.sim, task_env.sim.model
        robot = task_env.robots[0]
        self.task_env = task_env
        self.arm = list(robot._ref_joint_pos_indexes)
        self.grip = list(robot._ref_gripper_joint_pos_indexes)
        self.bodies, self.skipped = {}, []
        for name, bid in task_env.obj_body_id.items():
            jadr, jnum = int(m.body_jntadr[bid]), int(m.body_jntnum[bid])
            if jnum >= 1 and int(m.jnt_type[jadr]) == 0:  # free joint
                self.bodies[name] = ("free", int(m.jnt_qposadr[jadr]))
            elif int(m.body_mocapid[bid]) >= 0:
                self.bodies[name] = ("mocap", int(m.body_mocapid[bid]))
            else:
                self.skipped.append(name)

    def apply(self, obs) -> float:
        """Set state from obs; returns max |xpos - logged pos| over objects (sanity)."""
        sim = self.task_env.sim
        q = sim.data.qpos
        q[self.arm] = np.asarray(obs["robot0_joint_pos"], float)
        q[self.grip] = np.asarray(obs["robot0_gripper_qpos"], float)
        for name, (kind, adr) in self.bodies.items():
            if f"{name}_pos" not in obs:
                continue
            pos = np.asarray(obs[f"{name}_pos"], float)
            x, y, z, w = np.asarray(obs[f"{name}_quat"], float)  # robosuite obs quat is xyzw
            quat = np.array([w, x, y, z])
            if kind == "free":
                q[adr:adr + 3], q[adr + 3:adr + 7] = pos, quat
            else:
                sim.data.mocap_pos[adr], sim.data.mocap_quat[adr] = pos, quat
        sim.forward()
        err = 0.0
        for name in self.bodies:
            if f"{name}_pos" in obs:
                bid = self.task_env.obj_body_id[name]
                err = max(err, float(np.abs(sim.data.body_xpos[bid] - np.asarray(obs[f"{name}_pos"])).max()))
        return err


def _grade_episode(path, cache, validate):
    from rlinf.envs.sim.libero.safety_grader import build_graders, grade_step
    from rlinf.envs.sim.libero.venv import _unwrap_task_env

    rec = json.load(open(path, encoding="utf-8"))
    cell = _cell(rec)
    env = _env(cell, cache)
    env.reset()
    env.set_init_state(cache["inits"][int(rec["episode"]["trial_id"])])
    task_env = _unwrap_task_env(env)
    graders = build_graders(task_env, task_env.parsed_problem, aux=True)
    rb = Rebuilder(task_env)
    pose_err, unsafe_steps, chan, md = 0.0, 0, Counter(), None
    agree, total, dist_err, rec_viol_steps = 0, 0, [], 0
    first_viol = None
    for step in rec["trajectory"]:
        pose_err = max(pose_err, rb.apply(step["observation"]))
        grades = grade_step(graders, task_env)
        g_viol = [g for g in grades if g["violated"]]
        for g in g_viol:
            chan[g["channel"]] += 1
        for g in grades:
            if g["channel"] in ("c_obj", "c_robot"):
                md = g["min_dist"] if md is None else min(md, g["min_dist"])
        logged = bool(step["terminations"]["safety_violation"])
        rec_viol_steps += logged
        unsafe = logged or bool(g_viol)
        unsafe_steps += unsafe
        if unsafe and first_viol is None:
            first_viol = step["timestep"]
        if validate and step.get("grades"):
            ref = {g["name"]: g for g in step["grades"]}
            for g in grades:
                r = ref.get(g["name"])
                if r is None:
                    continue
                total += 1
                agree += r["violated"] == g["violated"]
                if r["min_dist"] < 0.1 or g["min_dist"] < 0.1:
                    dist_err.append(abs(r["min_dist"] - g["min_dist"]))
    success = bool(rec["success"])
    q = _QUAD[(success, unsafe_steps > 0)]
    row = {"path": path, "cell": cell, "kind": "branch" if rec.get("root_episode_id") else "root",
           "steps": len(rec["trajectory"]), "pose_err": pose_err, "skipped_bodies": rb.skipped,
           "old_q": rec["quadrant"], "new_q": q, "logged_unsafe_steps": rec_viol_steps,
           "kin_unsafe_steps": unsafe_steps, "first_viol": first_viol, "channels": dict(chan),
           "min_hazard_dist": md, "near_miss": q in ("Q1", "Q3") and md is not None and md < NEAR_MISS_M}
    if validate:
        row.update(grade_agree=agree, grade_total=total,
                   dist_err_p95=float(np.percentile(dist_err, 95)) if dist_err else None,
                   dist_err_max=float(max(dist_err)) if dist_err else None)
    return row


def _work(args):
    paths, validate = args
    cache, rows = {}, []
    for p in paths:
        t0 = time.time()
        try:
            row = _grade_episode(p, cache, validate)
        except Exception:  # noqa: BLE001
            row = {"path": p, "error": traceback.format_exc(limit=4)}
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
    if cache.get("env") is not None:
        cache["env"].close()
    return rows


def _cell_of_path(p):
    with open(p, encoding="utf-8") as f:
        return _cell(json.load(f))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate")
    g.add_argument("--kept")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(a.validate, recursive=True)) if a.validate else \
        [p.strip() for p in open(a.kept) if p.strip()]
    with get_context("spawn").Pool(a.workers) as pool:
        cells = pool.map(_cell_of_path, paths, chunksize=64)
    by_cell = defaultdict(list)
    for p, c in zip(paths, cells):
        by_cell[c].append(p)
    rng = random.Random(a.seed)
    batches = []
    for c in sorted(by_cell):
        ps = by_cell[c]
        if a.per_cell:
            ps = rng.sample(ps, min(a.per_cell, len(ps)))
        batches += [(ps[i:i + 16], bool(a.validate)) for i in range(0, len(ps), 16)]
    total = sum(len(b[0]) for b in batches)
    print(f"{len(paths)} audits, {total} jobs, {len(batches)} batches", flush=True)
    t0, n = time.time(), 0
    with open(out / "rows.jsonl", "w") as f, get_context("spawn").Pool(a.workers) as pool:
        for rows in pool.imap_unordered(_work, batches):
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
            n += len(rows)
            print(f"[{n}/{total}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
