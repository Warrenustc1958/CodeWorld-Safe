#!/usr/bin/env python3
"""Stage 1 (training-free) history-repair replays R1-R3 on L1 side_render dumps.

Every mode rolls out all chunks from the keyframe with the recorded actions and the online noise of each chunk
(common random numbers with the online rollout, see p_replay.py), and differs from the online rollout only in how
the 4 history frames are formed before every chunk d >= 1 (gen = this rollout's own last 4 frames, real = the real
renders at the same steps):
  R1  blendpx{a}   history = a * real + (1 - a) * gen in pixels, every chunk          (a = 1: per-chunk oracle)
      blendlat{a}  same mix on the VAE latent of the history (slot 1 of first_frame_latents)
  R2  period{N}    history = real every N-th chunk, gen otherwise
  R3  wristcont    wrist half = real, agent half = gen, every chunk (oracle for a wrist-only repair)
      retr_{src}   wrist half = real wrist frames retrieved by EEF pose from the other episodes of the same start
                   keyframe, agent half = gen; pose from src = real (oracle state), int (action integration, R4 int_l1),
                   read (state reader on the generated history: the online proprio)
Also writes the online rollout itself ("free") with the same metrics.

  python p_stage1.py --dump DIR [...] --tracks r4_tracks.npz --out out.jsonl --shard i --nshard n
  python p_stage1.py --summary out.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_replay import CHUNK, NULL_ACTION, R, Episode, build_pipe, dec, jpeg_roundtrip, mae2, orig_noise, scale_pose  # noqa: E402

ALPHAS_PX = (0.25, 0.5, 0.75, 1.0)
ALPHAS_LAT = (0.25, 0.5, 0.75)
PERIODS = (2, 4, 8)
SOURCES = ("real", "int", "read")


def pose_dist(q, S):
    """q [8] vs S [N, 8] (pos, axis-angle, gripper qpos): cm + 0.5 * deg + 1000 * |grip| (heuristic weights)."""
    pos = np.linalg.norm(S[:, :3] - q[:3], axis=1) * 100
    rot = np.degrees((Rot.from_rotvec(S[:, 3:6]) * Rot.from_rotvec(q[3:6]).inv()).magnitude())
    grip = np.abs(S[:, 6:8] - q[6:8]).mean(1)
    return pos + 0.5 * rot + 1000 * grip


class Pool:
    """Real frames of all episodes of one start keyframe, with their real 8-d state, decoded on demand."""

    def __init__(self, files):
        self.names, self.bufs, states, owner, frame = [], [], [], [], []
        for f in files:
            z = np.load(f)
            self.names.append(os.path.basename(f))
            self.bufs.append((z["real_jpg"], z["real_off"]))
            rs = z["real_state"]
            states.append(rs)
            owner.append(np.full(len(rs), len(self.names) - 1))
            frame.append(np.arange(len(rs)))
        self.state, self.owner, self.frame = np.concatenate(states), np.concatenate(owner), np.concatenate(frame)

    def wrist(self, q, exclude):
        d = pose_dist(q, self.state)
        d[self.owner == self.names.index(exclude)] = np.inf
        i = int(np.argmin(d))
        buf, off = self.bufs[self.owner[i]]
        return dec(buf, off, int(self.frame[i]))[256:], float(d[i])


class Job:
    def __init__(self, ep, mode, kind, param=None):
        self.ep, self.mode, self.kind, self.param = ep, mode, kind, param
        self.d, self.left = 0, ep.D
        self.mem, self.mem_a, self.blend = ep.kf_mem, ep.kf_mem_a, None


def add_latent_blend(pipe):
    """Wrap vae.encode: when pipe._blend = (alpha [B], real image lists) is set, the history encode returns
    z_gen + alpha (z_real - z_gen). Consumed by the first 5-frame (anchor + 4 history) encode of the call."""
    orig = pipe.vae.encode
    pipe._blend = None

    def encode(videos, *a, **k):
        z = orig(videos, *a, **k)
        bl = pipe._blend
        if bl is not None and getattr(videos, "shape", (0, 0, 0))[2] == 5:
            pipe._blend = None
            alpha, real5 = bl
            zr = orig(pipe.preprocess_video(real5), *a, **k)
            z = z + alpha.view(-1, 1, 1, 1, 1).to(z.device, z.dtype) * (zr.to(z.dtype) - z)
        return z

    pipe.vae.encode = encode


def generate(pipe, jobs):
    anchors = [Image.fromarray(j.ep.anchor) for j in jobs]
    mems = [[Image.fromarray(np.ascontiguousarray(m)) for m in j.mem] for j in jobs]
    win = np.stack([np.concatenate([NULL_ACTION[None], j.mem_a, j.ep.actions[j.d]], 0) for j in jobs])
    win = torch.from_numpy(scale_pose(win)).to(next(pipe.dit.parameters()).device)
    pipe._fixed_noise = torch.stack([orig_noise(j.ep, j.d) for j in jobs])
    alpha = torch.tensor([j.blend[0] if j.blend else 0.0 for j in jobs])
    if (alpha > 0).any():
        real5 = [[a] + ([Image.fromarray(np.ascontiguousarray(m)) for m in j.blend[1]] if j.blend else mm)
                 for a, mm, j in zip(anchors, mems, jobs)]
        pipe._blend = (alpha, real5)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pipe(seed=0, tiled=False, input_image=anchors, input_image4=mems, action=win, height=512, width=256,
                   num_frames=13, num_inference_steps=5, cfg_scale=1.0, progress_bar_cmd=lambda x: x,
                   batch_size=len(jobs))
    pipe._fixed_noise, pipe._blend = None, None
    return [np.stack([np.asarray(im, np.uint8) for im in o[5:13]]) for o in out]


def run(args):
    files = sorted(f for d in args.dump for f in glob.glob(os.path.join(d, "arrays", "*.npz")))
    by_kf = collections.defaultdict(list)
    for f in files:
        by_kf[int(np.load(f)["kf"])].append(f)
    kfs = sorted(by_kf)
    mine = [k for i, k in enumerate(kfs) if i % args.nshard == args.shard]
    if args.max_kf:
        mine = mine[: args.max_kf]
    kfd = f"{R}/data/scgrpo_bundle/keyframes_v1"
    kf_anchor = np.load(f"{kfd}/anchor.npy", mmap_mode="r")
    kf_memory = np.load(f"{kfd}/memory.npy", mmap_mode="r")
    meta = np.load(f"{kfd}/meta.npz")
    kf_inst = [str(s) for s in meta["instruction"]]
    mem_a, states = meta["mem_actions"].astype(np.float32), meta["states"].astype(np.float32)
    tr = np.load(args.tracks, allow_pickle=True)
    tracks = {str(n): np.asarray(t, np.float64) for n, t in zip(tr["names"], tr["int_l1"])}  # stored as objects

    device = "cuda:0"
    pipe = build_pipe(device)
    add_latent_blend(pipe)
    from rlinf.envs.sim.world_model.safe.safe_models import StateReader
    from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel

    reader = StateReader(f"{R}/data/scgrpo_bundle/scoring/state_reg_v1/state_reg.pt").to(device)
    rm = CWSRewardModel.from_checkpoint(args.rm).eval().to(device)
    rows = 256 * int(getattr(rm, "views", 1))

    @torch.no_grad()
    def read_state(frames, ep):
        off = ep.state0 - reader(torch.from_numpy(np.ascontiguousarray(ep.kf_mem[-1:])).to(device)).float().cpu().numpy()[0]
        return reader(torch.from_numpy(np.ascontiguousarray(frames)).to(device)).float().cpu().numpy() + off

    @torch.no_grad()
    def rm_max(frames, ep):
        x = torch.from_numpy(frames[[3, 7], :rows]).to(device).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        return float(rm.predict_rew(x, [ep.inst] * 2).float().max())

    def metrics(fr, ep, d):
        ma, mw = mae2(fr, ep.real_chunk(d))
        st = read_state(fr, ep)
        rs = ep.real_state[1 + d * CHUNK: 1 + (d + 1) * CHUNK]
        eef = float(np.linalg.norm(st[:, :3] - rs[:, :3], axis=1).mean() * 100)
        rot = float(np.degrees((Rot.from_rotvec(st[:, 3:6]) * Rot.from_rotvec(rs[:, 3:6]).inv()).magnitude()).mean())
        return {"mae_agent": ma, "mae_wrist": mw, "eef_cm": eef, "rot_deg": rot, "rm_max": rm_max(fr, ep),
                "rm_real": float(np.max(ep.rm_real[d]))}

    def history(j, d, gen4):
        ep = j.ep
        real4 = ep.history(d, "real")
        j.blend = None
        if j.kind == "blendpx":
            return np.clip(np.round(j.param * real4.astype(np.float32) + (1 - j.param) * gen4), 0, 255).astype(np.uint8)
        if j.kind == "blendlat":
            j.blend = (j.param, real4)
            return gen4
        if j.kind == "period":
            return real4 if d % j.param == 0 else gen4
        if j.kind == "wristcont":
            return np.concatenate([gen4[:, :256], real4[:, 256:]], 1)
        if j.kind == "retr":
            steps = range(d * CHUNK - 3, d * CHUNK + 1)
            if j.param == "real":
                poses = ep.real_state[list(steps)]
            elif j.param == "int":
                t = tracks[ep.name][list(steps)]
                poses = np.concatenate([t[:, :3], Rot.from_quat(t[:, 3:7]).as_rotvec(), t[:, 7:9]], 1)
            else:
                poses = read_state(gen4, ep)
            wr = []
            for q in poses:
                w, dist = ep.pool.wrist(q, ep.name)
                wr.append(w)
                j.match.append(dist)
            return np.concatenate([gen4[:, :256], np.stack(wr)], 1)
        raise ValueError(j.kind)

    out = open(f"{args.out}.{args.shard}", "w")
    for kf in mine:
        pool = Pool(by_kf[kf])
        sel = [by_kf[kf][i] for i in sorted(np.random.default_rng(kf).permutation(len(by_kf[kf]))[: args.per_kf])]
        eps = []
        for f in sel:
            ep = Episode(f, kf_anchor, kf_memory, mem_a, kf_inst, states)
            ep.D = min(ep.D, args.max_chunks)
            ep.rm_real = np.load(f)["rm_real"][: ep.D]
            ep.pool = pool
            eps.append(ep)
        jobs = []
        for ep in eps:
            for d in range(ep.D):  # the online rollout, same metrics
                fr = ep.wan[d * CHUNK:(d + 1) * CHUNK]
                out.write(json.dumps({"ep": ep.name, "kf": kf, "mode": "free", "chunk": d + 1, **metrics(fr, ep, d)}) + "\n")
            jobs += [Job(ep, f"blendpx{a}", "blendpx", a) for a in ALPHAS_PX]
            jobs += [Job(ep, f"blendlat{a}", "blendlat", a) for a in ALPHAS_LAT]
            jobs += [Job(ep, f"period{n}", "period", n) for n in PERIODS]
            jobs += [Job(ep, "wristcont", "wristcont")]
            jobs += [Job(ep, f"retr_{s}", "retr", s) for s in SOURCES]
        for j in jobs:
            j.match = []
        while jobs:
            batch, jobs = jobs[: args.batch], jobs[args.batch:]
            for j, fr in zip(batch, generate(pipe, batch)):
                ep, d = j.ep, j.d
                rec = {"ep": ep.name, "kf": kf, "mode": j.mode, "chunk": d + 1,
                       **metrics(np.stack([jpeg_roundtrip(x) for x in fr]), ep, d)}
                if j.match:
                    rec["match"] = float(np.mean(j.match[-4:]))
                out.write(json.dumps(rec) + "\n")
                j.left -= 1
                j.d += 1
                if j.left > 0:
                    j.mem, j.mem_a = history(j, j.d, fr[-4:]), ep.actions[d, -4:]
                    jobs.append(j)
        out.flush()
        print(f"shard {args.shard}: kf {kf} done ({len(eps)} episodes)", flush=True)
    out.close()


def summary(path, thr=0.877):
    recs = [json.loads(l) for f in sorted(glob.glob(path + ".[0-9]*")) for l in open(f)]  # shards only
    free = {(r["ep"], r["chunk"]): r for r in recs if r["mode"] == "free"}
    by = collections.defaultdict(list)
    for r in recs:
        by[r["mode"]].append(r)
    order = (["free"] + [f"blendpx{a}" for a in ALPHAS_PX] + [f"blendlat{a}" for a in ALPHAS_LAT]
             + [f"period{n}" for n in PERIODS] + ["wristcont"] + [f"retr_{s}" for s in SOURCES])
    res = {"n_episodes": len({r["ep"] for r in recs}), "n_keyframes": len({r["kf"] for r in recs}), "modes": {}}
    print(f"episodes {res['n_episodes']}  keyframes {res['n_keyframes']}")
    print("mode         | wrist MAE @ step 64/128/192/256 | agent MAE @ 256 | eef cm / rot deg @ 256 | "
          "RM agree chunks 17-32 | wrist ratio vs free @256")
    for m in order:
        v = by.get(m)
        if not v:
            continue
        row = {}
        for c in (8, 16, 24, 32):
            x = [r for r in v if r["chunk"] == c]
            row[c] = {k: float(np.median([r[k] for r in x])) for k in ("mae_agent", "mae_wrist", "eef_cm", "rot_deg")}
            row[c]["ratio_wrist"] = float(np.median([r["mae_wrist"] / free[(r["ep"], c)]["mae_wrist"] for r in x
                                                     if (r["ep"], c) in free]))
        late = [r for r in v if r["chunk"] >= 17]
        agree = float(np.mean([(r["rm_max"] >= thr) == (r["rm_real"] >= thr) for r in late]))
        f01 = float(np.mean([(r["rm_max"] >= thr) and (r["rm_real"] < thr) for r in late]))
        match = [r["match"] for r in v if "match" in r]
        res["modes"][m] = {"by_chunk": row, "agree_17_32": agree, "f01_17_32": f01,
                           "match": float(np.median(match)) if match else None}
        print(f"{m:12s} | {row[8]['mae_wrist']:5.1f} {row[16]['mae_wrist']:5.1f} {row[24]['mae_wrist']:5.1f} "
              f"{row[32]['mae_wrist']:5.1f} | {row[32]['mae_agent']:5.1f} | {row[32]['eef_cm']:5.1f} / "
              f"{row[32]['rot_deg']:5.1f} | {agree:.3f} (f01 {f01:.3f}) | {row[32]['ratio_wrist']:.2f}"
              + (f"  match {res['modes'][m]['match']:.1f}" if match else ""))
    json.dump(res, open(path + ".summary.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+")
    ap.add_argument("--tracks")
    ap.add_argument("--out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--per-kf", type=int, default=4, help="episodes replayed per start keyframe")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-chunks", type=int, default=32, help="smaller for a quick test")
    ap.add_argument("--max-kf", type=int, default=0, help="0 = all keyframes of this shard")
    ap.add_argument("--rm", default=f"{R}/data/scgrpo_bundle/scgrpo_rm2_20261102/rm_models/R-A_resnet_rm2/best.pt")
    ap.add_argument("--summary")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
    else:
        run(args)


if __name__ == "__main__":
    main()
