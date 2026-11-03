#!/usr/bin/env python3
"""R7: does dual-41k respond to pi0.5-sized pose actions? (training-free, teacher-forced)

At chunk d of an L1 episode, Wan gets the REAL 4-frame history and the online noise of that chunk; only the
chunk's 8 actions change:
  x1        recorded actions (reference)
  pose0     pose dims (0-5) zeroed, gripper kept
  pose3     pose dims x3
  pose25    pose dims x25: pi0.5's pose std is ~25x below the Wan training data's (R6)
  gripflip  gripper sign flipped, pose kept
  seed      recorded actions, another noise (the scale of a change that carries no action information)
Per variant: pixel MAE to the x1 output (agent / wrist), and the eef displacement over the chunk read from the
output (reader at the chunk end minus the real state at the chunk start) vs the real displacement.
A pose variant whose output moves no more than "seed" means the pose input is effectively not used at that scale.

  python r7_action_sensitivity.py --dump DIR [...] --out r7.json [--episodes 48]
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_replay import CHUNK, R, Episode, Job, build_pipe, generate, mae2  # noqa: E402

VARIANTS = ("x1", "pose0", "pose3", "pose25", "gripflip", "seed")


def variant_actions(a, v):
    a = a.copy()
    if v == "pose0":
        a[:, :6] = 0
    elif v == "pose3":
        a[:, :6] *= 3
    elif v == "pose25":
        a[:, :6] *= 25
    elif v == "gripflip":
        a[:, 6] = -a[:, 6]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--depths", default="4,8,16,24")
    ap.add_argument("--batch", type=int, default=12)
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    files = sorted(f for d in args.dump for f in glob.glob(os.path.join(d, "arrays", "*.npz")))
    files = [files[i] for i in np.random.default_rng(1).permutation(len(files))[: args.episodes]]
    kfd = f"{R}/data/scgrpo_bundle/keyframes_v1"
    kf_anchor = np.load(f"{kfd}/anchor.npy", mmap_mode="r")
    kf_memory = np.load(f"{kfd}/memory.npy", mmap_mode="r")
    meta = np.load(f"{kfd}/meta.npz")
    kf_inst = [str(s) for s in meta["instruction"]]
    mem_a, states = meta["mem_actions"].astype(np.float32), meta["states"].astype(np.float32)

    device = "cuda:0"
    pipe = build_pipe(device)
    from rlinf.envs.sim.world_model.safe.safe_models import StateReader

    reader = StateReader(f"{R}/data/scgrpo_bundle/scoring/state_reg_v1/state_reg.pt").to(device)

    @torch.no_grad()
    def read_end(fr, ep):
        off = ep.state0 - reader(torch.from_numpy(np.ascontiguousarray(ep.kf_mem[-1:])).to(device)).float().cpu().numpy()[0]
        return reader(torch.from_numpy(np.ascontiguousarray(fr[-1:])).to(device)).float().cpu().numpy()[0] + off

    recs = []
    for f in files:
        ep = Episode(f, kf_anchor, kf_memory, mem_a, kf_inst, states)
        jobs = []
        for d in depths:
            if d >= ep.D:
                continue
            for v in VARIANTS:
                e2 = copy.copy(ep)
                e2.actions = ep.actions.copy()
                e2.actions[d] = variant_actions(ep.actions[d], v)
                jobs.append(Job(e2, v, d, 1, ep.history(d, "real"), ("seed", 7) if v == "seed" else ("orig",)))
        outs = []
        for i in range(0, len(jobs), args.batch):
            outs += generate(pipe, jobs[i:i + args.batch], 5)
        res = {(j.mode, j.d0): o for j, o in zip(jobs, outs)}
        for d in depths:
            if ("x1", d) not in res:
                continue
            ref = res[("x1", d)]
            s0, s1 = ep.real_state[d * CHUNK], ep.real_state[(d + 1) * CHUNK]
            dreal = s1[:3] - s0[:3]
            for v in VARIANTS:
                fr = res[(v, d)]
                a, w = mae2(fr, ref)
                ra, rw = mae2(fr, ep.real_chunk(d))
                dw = read_end(fr, ep)[:3] - s0[:3]
                recs.append({"ep": ep.name, "kf": ep.kf, "d": d, "v": v, "vs_x1_agent": a, "vs_x1_wrist": w,
                             "vs_real_agent": ra, "vs_real_wrist": rw, "disp_cm": float(np.linalg.norm(dw) * 100),
                             "disp_real_cm": float(np.linalg.norm(dreal) * 100),
                             "disp_err_cm": float(np.linalg.norm(dw - dreal) * 100),
                             "pose_energy": float(np.abs(ep.actions[d, :, :6]).mean()),
                             "grip_energy": float(np.abs(ep.actions[d, :, 6]).mean())})
        print(f"{len(recs)} records", flush=True)

    summ = {}
    print("variant   | MAE to x1 agent / wrist | MAE to real agent / wrist | eef disp cm (real) | disp err cm")
    for v in VARIANTS:
        x = [r for r in recs if r["v"] == v]
        summ[v] = {k: float(np.median([r[k] for r in x])) for k in
                   ("vs_x1_agent", "vs_x1_wrist", "vs_real_agent", "vs_real_wrist", "disp_cm", "disp_real_cm",
                    "disp_err_cm")}
        s = summ[v]
        print(f"{v:9s} | {s['vs_x1_agent']:5.2f} / {s['vs_x1_wrist']:5.2f}        | {s['vs_real_agent']:5.2f} / "
              f"{s['vs_real_wrist']:5.2f}          | {s['disp_cm']:5.2f} ({s['disp_real_cm']:4.2f})      | "
              f"{s['disp_err_cm']:5.2f}")
    pe = np.median([r["pose_energy"] for r in recs if r["v"] == "x1"])
    ge = np.median([r["grip_energy"] for r in recs if r["v"] == "x1"])
    summ["energy"] = {"pose_mean_abs": float(pe), "grip_mean_abs": float(ge), "pose_scaled_x8": float(pe * 8)}
    print(f"input energy per step: pose |a| {pe:.4f} (x8 = {pe * 8:.3f}) vs gripper |a| {ge:.3f}")
    json.dump({"summary": summ, "records": recs}, open(args.out, "w"))


if __name__ == "__main__":
    main()
