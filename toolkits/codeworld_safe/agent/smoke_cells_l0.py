"""Layer-0 smoke over every LIBERO-Safety L0/L1 cell (CPU only, no policy).

For each cell (suite x level x task) it checks, without any learned policy:

1. reset:      official / grader violations and goal satisfaction right after
               set_init_state + settle, over N init states (gate-2 behaviour).
2. reach:      every c_robot grader -> scripted OSC servo of the end effector
               onto the hazard (above-then-descend). Records whether the grader
               and the official predicate fire, first step, peak normal force.
3. drop:       every c_obj grader -> teleport the carried object above the
               hazard and let it fall; same records.
4. determinism: snapshot -> 16 random actions -> restore -> replay; max |dqpos|.

Writes <out>/cells/<key>.json and <out>/summary.tsv with one verdict row per cell.

    python toolkits/codeworld_safe/agent/smoke_cells_l0.py --out <dir> [--workers 20]
        [--inits 10] [--push-inits 3] [--only human_safety_L0_t0,...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
KNOWLEDGE = REPO / "toolkits/codeworld_safe/agent/knowledge/tasks"
LIBERO = Path(os.environ.get("LIBERO_SAFETY_PATH", "/mnt/nfs/fjj/Code/LIBERO-Safety"))
SETTLE_STEPS = 5
REACH_STEPS = 160
DROP_STEPS = 60
DET_STEPS = 16
DET_TOL = 1e-6


def _make_env(bddl: str):
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False,
                      use_camera_obs=False, ignore_done=True)


def _reset(env, init_state):
    env.reset()
    obs = env.set_init_state(init_state)
    act = np.zeros(7)
    act[-1] = -1.0
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(act)
    return obs


def _official(env) -> dict[str, dict]:
    from rlinf.envs.sim.libero.safety_auditor import audit_predicate_instances

    return {r["instance_id"]: r for r in audit_predicate_instances(env)}


def _grades(graders, task_env) -> dict[str, dict]:
    from rlinf.envs.sim.libero.safety_grader import grade_step

    return {g["name"].split("|")[0]: g for g in grade_step(graders, task_env)}


def _body_pos(task_env, name):
    return np.array(task_env.sim.data.body_xpos[task_env.obj_body_id[name]], dtype=float)


def _track(env, task_env, graders, iid, step_fn, n_steps):
    """Run step_fn for n_steps, recording grader iid and its official counterpart."""
    rec = {"grader_fired": False, "official_fired": False, "first_grader_step": None,
           "first_official_step": None, "max_force": 0.0, "min_dist": float("inf")}
    for t in range(n_steps):
        step_fn(t)
        g = _grades(graders, task_env)[iid]
        o = _official(env).get(iid)
        rec["max_force"] = max(rec["max_force"], float(g["normal_force"]))
        rec["min_dist"] = min(rec["min_dist"], float(g["min_dist"]))
        if g["violated"] and not rec["grader_fired"]:
            rec["grader_fired"], rec["first_grader_step"] = True, t
        if o is not None and bool(o["value"]) and not rec["official_fired"]:
            rec["official_fired"], rec["first_official_step"] = True, t
        if rec["grader_fired"] and t - rec["first_grader_step"] >= 10:
            break
    rec["official_exists"] = iid in _official(env)
    return rec


def _reach_test(env, task_env, graders, grader, init_state):
    obs = _reset(env, init_state)
    state = {"obs": obs}
    hazard = grader.hazard

    def step(t):
        eef = np.asarray(state["obs"]["robot0_eef_pos"], dtype=float)
        target = _body_pos(task_env, hazard)
        above = target + np.array([0.0, 0.0, 0.12])
        goal = above if np.linalg.norm((eef - above)[:2]) > 0.02 and t < REACH_STEPS // 2 else target
        act = np.zeros(7)
        act[:3] = np.clip((goal - eef) * 20.0, -1.0, 1.0)
        act[-1] = -1.0
        state["obs"], *_ = env.step(act)

    return _track(env, task_env, graders, grader.name.split("|")[0], step, REACH_STEPS)


def _drop_test(env, task_env, graders, grader, init_state):
    _reset(env, init_state)
    obj = task_env.get_object(grader.obj)
    joints = list(getattr(obj, "joints", []) or [])
    if not joints:
        return {"skipped": "no_free_joint"}
    target = _body_pos(task_env, grader.hazard) + np.array([0.0, 0.0, 0.15])
    quat = np.array(task_env.sim.data.get_joint_qpos(joints[0])[3:7], dtype=float)
    task_env.sim.data.set_joint_qpos(joints[0], np.concatenate([target, quat]))
    task_env.sim.forward()
    act = np.zeros(7)
    act[-1] = -1.0
    return _track(env, task_env, graders, grader.name.split("|")[0],
                  lambda t: env.step(act), DROP_STEPS)


def _det_test(env, init_state, seed=0):
    from rlinf.envs.sim.libero.venv import (_restore_simulator_state, _snapshot_simulator_state,
                                            _unwrap_task_env)

    _reset(env, init_state)
    snap = _snapshot_simulator_state(env)
    rng = np.random.default_rng(seed)
    acts = rng.uniform(-1.0, 1.0, size=(DET_STEPS, 7))
    sim = _unwrap_task_env(env).sim
    first = []
    for a in acts:
        env.step(a)
        first.append(np.array(sim.data.qpos, dtype=float))
    _restore_simulator_state(env, snap)
    diff = 0.0
    for a, q in zip(acts, first):
        env.step(a)
        diff = max(diff, float(np.max(np.abs(np.array(sim.data.qpos) - q))))
    return diff


def run_cell(args):
    key, out_dir, n_inits, n_push = args
    import torch
    from rlinf.envs.sim.libero.safety_grader import (ObjectContactGrader, RobotContactGrader,
                                                     build_graders)
    from rlinf.envs.sim.libero.venv import _unwrap_task_env

    t0 = time.time()
    info = json.loads((KNOWLEDGE / f"{key}.json").read_text())
    res = {"key": key, "errors": []}
    try:
        env = _make_env(str(LIBERO / info["bddl_file"]))
        inits = torch.load(str(LIBERO / info["init_states_file"]), weights_only=False)
        task_env = _unwrap_task_env(env)
        _reset(env, inits[0])
        graders = build_graders(task_env, task_env.parsed_problem, aux=True)
        res["graders"] = [g.name for g in graders]
        res["n_init_states"] = int(len(inits))

        reset_rows = []
        for i in range(min(n_inits, len(inits))):
            _reset(env, inits[i])
            g = _grades(graders, task_env)
            o = _official(env)
            reset_rows.append({
                "init": i,
                "grader_viol": [k for k, v in g.items() if v["violated"]],
                "near_miss": [k for k, v in g.items() if v["near_miss"]],
                "min_dist": {k: round(float(v["min_dist"]), 4) for k, v in g.items()},
                "official_viol": [k for k, v in o.items() if v["semantics"] == "violation" and v["value"]],
                "goal_at_reset": bool(task_env._check_success()),
            })
        res["reset"] = reset_rows

        res["reach"], res["drop"] = {}, {}
        for gr in graders:
            iid = gr.name
            rows = []
            for i in range(min(n_push, len(inits))):
                try:
                    if isinstance(gr, RobotContactGrader):
                        rows.append(_reach_test(env, task_env, graders, gr, inits[i]))
                    elif isinstance(gr, ObjectContactGrader):
                        rows.append(_drop_test(env, task_env, graders, gr, inits[i]))
                except Exception as e:  # noqa: BLE001
                    rows.append({"error": f"{type(e).__name__}: {e}"})
            if isinstance(gr, RobotContactGrader):
                res["reach"][iid] = rows
            elif isinstance(gr, ObjectContactGrader):
                res["drop"][iid] = rows
            else:
                res.setdefault("untested", []).append(iid)

        res["det_max_dqpos"] = [_det_test(env, inits[i], seed=i) for i in range(min(2, len(inits)))]
        env.close()
    except Exception as e:  # noqa: BLE001
        res["errors"].append(traceback.format_exc(limit=6))
    res["seconds"] = round(time.time() - t0, 1)
    res["verdict"] = verdict(res)
    (Path(out_dir) / "cells" / f"{key}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def verdict(res) -> dict:
    issues = []
    if res.get("errors"):
        issues.append("error")
    reset = res.get("reset", [])
    if reset:
        if any(r["grader_viol"] for r in reset):
            issues.append(f"reset_grader_viol={sum(bool(r['grader_viol']) for r in reset)}/{len(reset)}")
        if any(r["official_viol"] for r in reset):
            issues.append(f"reset_official_viol={sum(bool(r['official_viol']) for r in reset)}/{len(reset)}")
        if any(r["goal_at_reset"] for r in reset):
            issues.append("goal_at_reset")
    for kind in ("reach", "drop"):
        for iid, rows in res.get(kind, {}).items():
            ok = [r for r in rows if "grader_fired" in r]
            short = iid.split("|", 1)[1] if "|" in iid else iid
            if not ok:
                issues.append(f"{kind}_untested:{short}")
                continue
            g = sum(r["grader_fired"] for r in ok)
            o = sum(r["official_fired"] for r in ok)
            if g == 0:
                issues.append(f"{kind}_unreached:{short}")
            if iid.startswith("constraint") and g > 0 and o < g:
                issues.append(f"official_misses:{short}({o}/{g})")
    det = res.get("det_max_dqpos", [])
    if det and max(det) > DET_TOL:
        issues.append(f"nondeterministic({max(det):.1e})")
    return {"issues": issues, "status": "ok" if not issues else "check"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--inits", type=int, default=10)
    ap.add_argument("--push-inits", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    keys = sorted(p.stem for p in KNOWLEDGE.glob("*.json"))
    if a.only:
        want = set(a.only.split(","))
        keys = [k for k in keys if k in want]
    jobs = [(k, str(out), a.inits, a.push_inits) for k in keys]
    with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        results = []
        for r in pool.imap_unordered(run_cell, jobs):
            results.append(r)
            print(f"[{len(results)}/{len(jobs)}] {r['key']} {r['seconds']}s {r['verdict']['status']} "
                  f"{';'.join(r['verdict']['issues'])}", flush=True)
    results.sort(key=lambda r: r["key"])
    with open(out / "summary.tsv", "w") as f:
        f.write("cell\tstatus\tissues\tgraders\tseconds\n")
        for r in results:
            f.write(f"{r['key']}\t{r['verdict']['status']}\t{';'.join(r['verdict']['issues'])}\t"
                    f"{len(r.get('graders', []))}\t{r['seconds']}\n")
    print(f"wrote {out/'summary.tsv'}")


if __name__ == "__main__":
    sys.exit(main())
