#!/usr/bin/env python3
"""Sample fresh LIBERO-Safety initial states for calib_fresh / test_fresh.

The shipped .pruned_init files hold 50 states per task and all of them were already
collected in Step E (cws_v1 / cws_test_new), so they cannot serve as unseen
calibration or test groups. Here each task env is reset with new seeds: the BDDL
placement regions and samplers are unchanged, so the states come from the same
distribution as the shipped ones but were never used by Wan, RM or SAM.

Output mirrors the LIBERO layout, so libero_env can read it via ``init_states_root``:
  <out>/<problem_folder>/L<level>/<init_states_file>   torch float32 [N, D]
  <out>/manifest.json                                  seeds, shapes, distance checks

A state is kept only if it is not a near copy of any shipped state or of an
already kept one (min L2 over the full flattened state > --min_dist).

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=3 python gen_fresh_init_states.py --out <dir> --n 20
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

SUITES = ("affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20, help="fresh states per task")
    ap.add_argument("--levels", default="0,1")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--tasks", default="", help="optional comma list of level_ids (smoke test)")
    ap.add_argument("--seed_base", type=int, default=20261025)
    ap.add_argument("--min_dist", type=float, default=1e-3)
    ap.add_argument("--max_tries", type=int, default=400)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",")]
    only = {int(x) for x in args.tasks.split(",")} if args.tasks else None
    manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    shipped_root = Path(get_libero_path("init_states"))

    for suite_name in args.suites.split(","):
        suite = benchmark.get_benchmark(suite_name)()
        for level in levels:
            for task in suite.get_tasks_by_level(level):
                if only is not None and task.level_id not in only:
                    continue
                key = f"{suite_name}_L{level}_t{task.level_id}"
                dst = out / task.problem_folder / f"L{level}" / task.init_states_file
                if dst.exists() and manifest.get(key, {}).get("n") == args.n:
                    print(f"[skip] {key} exists", flush=True)
                    continue
                shipped = suite.get_task_init_states(level, task.level_id)
                shipped = np.asarray(shipped, dtype=np.float64)
                bddl = suite.get_task_bddl_file_path(level, task.level_id)
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
                t0, kept, seeds, dmin_shipped = time.time(), [], [], []
                task_seed = args.seed_base + 100000 * SUITES.index(suite_name) + 10000 * level + 1000 * task.level_id
                for k in range(args.max_tries):
                    if len(kept) >= args.n:
                        break
                    seed = task_seed + k
                    env.seed(seed)
                    env.reset()
                    # shipped states are post-settle snapshots (sim time ~= shipped[:, 0]); reset() returns t=0
                    # with objects still in the air, so hold the arm (all-zero action) until that time
                    target_t = float(np.median(shipped[:, 0]))
                    hold = np.zeros(7)
                    while env.env.sim.data.time < target_t - 1e-6:
                        env.step(hold)
                    s = np.asarray(env.get_sim_state(), dtype=np.float64)
                    if s.shape[0] != shipped.shape[1]:
                        raise RuntimeError(f"{key}: state dim {s.shape[0]} != shipped {shipped.shape[1]}")
                    d_ship = float(np.linalg.norm(shipped - s, axis=1).min())
                    d_kept = float(min(np.linalg.norm(np.stack(kept) - s, axis=1).min(), 1e9)) if kept else 1e9
                    if d_ship <= args.min_dist or d_kept <= args.min_dist:
                        continue
                    kept.append(s)
                    seeds.append(seed)
                    dmin_shipped.append(d_ship)
                env.close()
                if len(kept) < args.n:
                    raise RuntimeError(f"{key}: only {len(kept)} distinct states in {args.max_tries} tries")
                arr = torch.tensor(np.stack(kept), dtype=torch.float32)
                dst.parent.mkdir(parents=True, exist_ok=True)
                torch.save(arr, dst)
                pair = np.linalg.norm(shipped[:, None] - shipped[None], axis=2)
                manifest[key] = {
                    "path": str(dst.relative_to(out)), "n": len(kept), "dim": int(arr.shape[1]),
                    "seeds": seeds, "shipped_file": str(shipped_root / task.problem_folder / f"L{level}" / task.init_states_file),
                    "min_dist_to_shipped": {"min": min(dmin_shipped), "median": float(np.median(dmin_shipped))},
                    "shipped_pairwise_median": float(np.median(pair[np.triu_indices(len(shipped), 1)])),
                    "seconds": round(time.time() - t0, 1),
                }
                (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
                m = manifest[key]
                print(f"[ok] {key}: {m['n']} states dim {m['dim']} min_d_shipped {m['min_dist_to_shipped']['min']:.4f} "
                      f"(shipped pairwise median {m['shipped_pairwise_median']:.4f}) {m['seconds']}s", flush=True)
    print(f"done: {len(manifest)} tasks in {out}", flush=True)


if __name__ == "__main__":
    main()
