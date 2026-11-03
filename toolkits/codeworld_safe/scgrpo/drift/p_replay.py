#!/usr/bin/env python3
"""Offline Wan replays on side_render dumps: history interventions under fixed actions (training-free).

Every episode npz (SyncRealWorldModelEnv, side_render + dump_arrays) holds the executed actions, every real frame,
the Wan frames the policy saw, and each chunk's Wan seed. Only Wan is re-run here (no policy, no simulator): the
actions are the recorded ones, so the real frames remain the ground truth for every replay.

Noise: "orig" rebuilds exactly the noise this slot got online (same seed, same batch shape, sliced at the slot), so
replays that only change the history are common-random-numbers comparisons with the online rollout. "seed" draws
fresh noise. Latent slots 0-1 are overwritten by the history every step, so only the noise of slots 2-3 matters.

Modes (D = --depths, chunk index d = 0-based, history = the 4 frames before chunk d):
  repro          d=0, orig noise, 2 chunks        does the replay reproduce the dumped Wan frames (numerics check)
  oracle@D       real history at d, orig noise, to the end        P4: recovery after a perfect history repair
  wrist@D/agent@D  only that view of the history is real            which view's history matters
  tf_s1@D        real history at d, 1 chunk, another seed           noise-only spread at the same history (P3 scale)
  anchor@D       history = 4 copies of the anchor, 1 chunk          P3: no history
  swap@D         history = real frames of another start state, 1 chunk, own actions   P3: wrong history
  seed{k}        d=0, fresh noise k, to the end                     P6: bias / variance; P5 gain via oracle vs free
Output: one JSON line per (episode, mode, chunk) with per-view MAE vs real (and vs the oracle chunk for P3 modes),
state-reader eef error, RM max; plus one line per (episode, chunk) with the pixel MSE bias/variance split over seeds.

  python p_replay.py --dump DIR [...] --out out.jsonl --shard i --nshard n [--max-episodes N] [--seeds 4]
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os

import numpy as np
import torch
from PIL import Image

CHUNK = 8
NULL_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
R = "/mnt/cpfs/jieduan"


def dec(buf, off, i):
    return np.asarray(Image.open(io.BytesIO(buf[off[i]:off[i + 1]].tobytes())).convert("RGB"))


def jpeg_roundtrip(frame, quality=90):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame)).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def scale_pose(a, s=8.0, c=4.0):
    out = np.array(a, np.float32, copy=True)
    out[..., :6] = np.clip(out[..., :6] * s, -c, c)
    return out


class Episode:
    def __init__(self, path, kf_anchor, kf_memory, kf_mem_actions, kf_inst, kf_states):
        z = np.load(path)
        self.name = os.path.basename(path)
        self.kf = int(z["kf"])
        self.slot, self.bsz = int(z["slot_in_batch"]), int(z["batch_size"])
        self.seeds = z["wan_seed"].astype(np.int64)
        nr = len(z["real_off"]) - 1
        self.D = min(len(self.seeds), (nr - 1) // CHUNK, len(z["actions"]))
        self.actions = z["actions"][: self.D].astype(np.float32)  # [D, 8, 7] env units
        self.real = np.stack([dec(z["real_jpg"], z["real_off"], i) for i in range(1 + self.D * CHUNK)])
        self.wan = np.stack([dec(z["wan_jpg"], z["wan_off"], i) for i in range(self.D * CHUNK)])
        self.real_state = z["real_state"][: 1 + self.D * CHUNK]
        self.anchor = np.asarray(kf_anchor[self.kf])
        self.kf_mem = np.asarray(kf_memory[self.kf])
        self.kf_mem_a = kf_mem_actions[self.kf]
        self.inst = kf_inst[self.kf]
        self.state0 = kf_states[self.kf]

    def real_chunk(self, d):
        return self.real[1 + d * CHUNK: 1 + (d + 1) * CHUNK]

    def mem_actions(self, d):
        return self.kf_mem_a if d == 0 else self.actions[d - 1, -4:]

    def history(self, d, source):
        """The 4 history frames before chunk d: 'wan' = what the online rollout used, 'real' = real renders."""
        if d == 0:
            return self.kf_mem
        if source == "wan":
            return self.wan[d * CHUNK - 4: d * CHUNK]
        return self.real[d * CHUNK - 3: d * CHUNK + 1]


class Job:
    def __init__(self, ep, mode, d0, n_chunks, mem, noise, keep_first=False):
        self.ep, self.mode, self.d, self.left = ep, mode, d0, n_chunks
        self.mem, self.mem_a = mem, ep.mem_actions(d0)
        self.noise = noise  # ("orig",) or ("seed", k)
        self.keep_first, self.first = keep_first, None
        self.d0 = d0


def build_pipe(device):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    from rlinf.envs.sim.world_model.safe.safe_models import enable_dual

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=f"{R}/data/wan_dual_41000/ema-step-41000.safetensors", offload_device="cpu"),
                       ModelConfig(path=f"{R}/models/wan_vae/Wan2.2_VAE.pth", offload_device="cpu")])
    pipe.dit.to(device)
    pipe.vae.to(device)
    enable_dual(pipe)
    pipe._fixed_noise = None
    orig = pipe.generate_noise

    def generate_noise(shape, seed=None, **kw):  # explicit per-sample noise when set
        if pipe._fixed_noise is not None:
            assert tuple(pipe._fixed_noise.shape) == tuple(shape), (pipe._fixed_noise.shape, shape)
            return pipe._fixed_noise.to(dtype=pipe.torch_dtype, device=pipe.device)
        return orig(shape, seed=seed, **kw)

    pipe.generate_noise = generate_noise
    return pipe


_NOISE_SHAPE = (48, 4, 32, 16)  # z_dim, latent T for 13 frames, 512/16, 256/16


def orig_noise(ep, d):
    """The noise slot ep.slot received online for chunk d (batch generated on CPU from one seed)."""
    g = torch.Generator("cpu").manual_seed(int(ep.seeds[d]))
    full = torch.randn((ep.bsz,) + _NOISE_SHAPE, generator=g, device="cpu", dtype=torch.float32)  # as generate_noise
    return full[ep.slot]


def seed_noise(ep, d, k):
    g = torch.Generator("cpu").manual_seed((ep.kf * 7919 + d * 104729 + k * 15485863) % (2 ** 31))
    return torch.randn(_NOISE_SHAPE, generator=g, device="cpu", dtype=torch.float32)


def generate(pipe, jobs, steps):
    anchors = [Image.fromarray(j.ep.anchor) for j in jobs]
    mems = [[Image.fromarray(np.ascontiguousarray(m)) for m in j.mem] for j in jobs]
    win = np.stack([np.concatenate([NULL_ACTION[None], j.mem_a, j.ep.actions[j.d]], 0) for j in jobs])
    win = torch.from_numpy(scale_pose(win)).to(next(pipe.dit.parameters()).device)
    noise = [orig_noise(j.ep, j.d) if j.noise[0] == "orig" else seed_noise(j.ep, j.d, j.noise[1]) for j in jobs]
    pipe._fixed_noise = torch.stack(noise)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pipe(seed=0, tiled=False, input_image=anchors, input_image4=mems, action=win, height=512, width=256,
                   num_frames=13, num_inference_steps=5, cfg_scale=1.0, progress_bar_cmd=lambda x: x,
                   batch_size=len(jobs))
    pipe._fixed_noise = None
    return [np.stack([np.asarray(im, np.uint8) for im in o[5:13]]) for o in out]


def mae2(x, y):
    d = np.abs(x.astype(np.int16) - y.astype(np.int16))
    return float(d[:, :256].mean()), float(d[:, 256:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all; taken after a fixed shuffle")
    ap.add_argument("--depths", default="4,8,16,24")
    ap.add_argument("--seeds", type=int, default=4, help="fresh-noise full rollouts per episode (P6)")
    ap.add_argument("--modes", default="repro,oracle,wrist,agent,tf_s1,anchor,swap,seed")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.877)
    ap.add_argument("--rm", default=f"{R}/data/scgrpo_bundle/scgrpo_rm2_20261102/rm_models/R-A_resnet_rm2/best.pt")
    args = ap.parse_args()
    modes = set(args.modes.split(","))
    depths = [int(x) for x in args.depths.split(",")]
    device = "cuda:0"

    files = sorted(f for d in args.dump for f in glob.glob(os.path.join(d, "arrays", "*.npz")))
    files = [files[i] for i in np.random.default_rng(0).permutation(len(files))]
    if args.max_episodes:
        files = files[: args.max_episodes]
    mine = files[args.shard::args.nshard]
    kfd = f"{R}/data/scgrpo_bundle/keyframes_v1"
    kf_anchor = np.load(f"{kfd}/anchor.npy", mmap_mode="r")
    kf_memory = np.load(f"{kfd}/memory.npy", mmap_mode="r")
    meta = np.load(f"{kfd}/meta.npz")
    kf_inst = [str(s) for s in meta["instruction"]]

    pipe = build_pipe(device)
    from rlinf.envs.sim.world_model.safe.safe_models import StateReader
    from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel

    reader = StateReader(f"{R}/data/scgrpo_bundle/scoring/state_reg_v1/state_reg.pt").to(device)
    rm = CWSRewardModel.from_checkpoint(args.rm).eval().to(device)
    rows = 256 * int(getattr(rm, "views", 1))

    @torch.no_grad()
    def score(frames, ep):
        ft = torch.from_numpy(frames).to(device)
        read = reader(ft).float().cpu().numpy()
        # per-scene reader offset as in the env: real keyframe state minus the reading of its last memory frame
        off = ep.state0 - reader(torch.from_numpy(ep.kf_mem[-1:]).to(device)).float().cpu().numpy()[0]
        x = ft[[3, 7], :rows].permute(0, 3, 1, 2).float().div(127.5).sub(1)
        p = rm.predict_rew(x, [ep.inst] * 2).float().cpu().numpy()
        return read + off, float(p.max())

    out = open(f"{args.out}.{args.shard}", "w")
    for ei in range(0, len(mine), 4):  # 4 episodes at a time keep the batches full
        eps = []
        for f in mine[ei: ei + 4]:
            try:
                eps.append(Episode(f, kf_anchor, kf_memory, meta["mem_actions"].astype(np.float32), kf_inst,
                                   meta["states"].astype(np.float32)))
            except Exception as exc:  # noqa: BLE001
                print("skip", f, exc, flush=True)
        jobs = []
        for e_i, ep in enumerate(eps):
            if "repro" in modes:
                jobs.append(Job(ep, "repro", 0, min(2, ep.D), ep.kf_mem, ("orig",)))
            if "seed" in modes:
                jobs += [Job(ep, f"seed{k}", 0, ep.D, ep.kf_mem, ("seed", k)) for k in range(args.seeds)]
            other = [o for o in eps if o.kf != ep.kf]
            for d in depths:
                if d >= ep.D:
                    continue
                real, wan = ep.history(d, "real"), ep.history(d, "wan")
                if "oracle" in modes:
                    jobs.append(Job(ep, f"oracle@{d}", d, ep.D - d, real, ("orig",), keep_first=True))
                if "wrist" in modes:
                    jobs.append(Job(ep, f"wrist@{d}", d, ep.D - d, np.concatenate([wan[:, :256], real[:, 256:]], 1),
                                    ("orig",)))
                if "agent" in modes:
                    jobs.append(Job(ep, f"agent@{d}", d, ep.D - d, np.concatenate([real[:, :256], wan[:, 256:]], 1),
                                    ("orig",)))
                if "tf_s1" in modes:
                    jobs.append(Job(ep, f"tf_s1@{d}", d, 1, real, ("seed", 99), keep_first=True))
                if "anchor" in modes:
                    jobs.append(Job(ep, f"anchor@{d}", d, 1, np.repeat(ep.anchor[None], 4, 0), ("orig",),
                                    keep_first=True))
                if "swap" in modes and other:
                    o = other[(e_i + d) % len(other)]
                    if d < o.D:
                        jobs.append(Job(ep, f"swap@{d}", d, 1, o.history(d, "real"), ("orig",), keep_first=True))
        for ep in eps:  # the online rollout itself (dumped Wan frames), same metrics: the no-repair baseline
            for d in range(ep.D):
                fr = ep.wan[d * CHUNK:(d + 1) * CHUNK]
                ma, mw = mae2(fr, ep.real_chunk(d))
                st, pmax = score(fr, ep)
                eef = float(np.linalg.norm(st[:, :3] - ep.real_state[1 + d * CHUNK: 1 + (d + 1) * CHUNK, :3],
                                           axis=1).mean() * 100)
                out.write(json.dumps({"ep": ep.name, "kf": ep.kf, "mode": "free", "d0": 0, "chunk": d + 1,
                                      "mae_agent": ma, "mae_wrist": mw, "eef_cm": eef, "rm_max": pmax}) + "\n")
        # running pixel sums over seeds for the MSE bias / variance split
        acc = {}
        firsts = {}
        while jobs:
            batch, jobs = jobs[: args.batch], jobs[args.batch:]
            outs = generate(pipe, batch, 5)
            for j, fr in zip(batch, outs):
                ep, d = j.ep, j.d
                real = ep.real_chunk(d)
                fj = np.stack([jpeg_roundtrip(x) for x in fr])  # same codec as the dumped frames ("free")
                ma, mw = mae2(fj, real)
                st, pmax = score(fj, ep)
                eef = float(np.linalg.norm(st[:, :3] - ep.real_state[1 + d * CHUNK: 1 + (d + 1) * CHUNK, :3],
                                           axis=1).mean() * 100)
                rec = {"ep": ep.name, "kf": ep.kf, "mode": j.mode, "d0": j.d0, "chunk": d + 1, "mae_agent": ma,
                       "mae_wrist": mw, "eef_cm": eef, "rm_max": pmax}
                if j.mode == "repro" or (j.mode.startswith("seed") and d == 0):
                    # chunk 1 has the same (keyframe) history as online: repro differs only numerically, seed{k}
                    # only by its noise, so seed{k} is the scale against which the repro residual is judged
                    rec["vs_dump_agent"], rec["vs_dump_wrist"] = mae2(fr, ep.wan[d * CHUNK:(d + 1) * CHUNK])
                    # the dump stores JPEG q90: compare after the same encoding to separate codec from numerics
                    rec["vs_dump_jpg_agent"], rec["vs_dump_jpg_wrist"] = mae2(fj, ep.wan[d * CHUNK:(d + 1) * CHUNK])
                if j.keep_first and j.first is None:  # JPEG'd like the dumped Wan chunk it is compared with
                    j.first = fj
                    firsts[(ep.name, j.mode)] = fj
                if j.mode.startswith("seed"):
                    s, s2, n = acc.get((ep.name, d), (0.0, 0.0, 0))
                    f32 = fj.astype(np.float32)
                    acc[(ep.name, d)] = (s + f32, s2 + f32 * f32, n + 1)
                out.write(json.dumps(rec) + "\n")
                j.left -= 1
                j.d += 1
                if j.left > 0:
                    j.mem, j.mem_a = fr[-4:], ep.actions[d, -4:]
                    jobs.append(j)
        # P3: same history depth, compare one chunk against the oracle chunk (same orig noise) and the seed spread
        for ep in eps:
            for d in depths:
                ref = firsts.get((ep.name, f"oracle@{d}"))
                if ref is None:
                    continue
                for m in ("anchor", "swap", "tf_s1"):
                    x = firsts.get((ep.name, f"{m}@{d}"))
                    if x is not None:
                        a, w = mae2(x, ref)
                        out.write(json.dumps({"ep": ep.name, "kf": ep.kf, "p3": m, "depth": d,
                                              "vs_oracle_agent": a, "vs_oracle_wrist": w}) + "\n")
                # the online Wan chunk at d: same orig noise, generated history instead of the real one (P5 numerator)
                a, w = mae2(ep.wan[d * CHUNK:(d + 1) * CHUNK], ref)
                ha, hw = mae2(ep.history(d, "wan"), ep.history(d, "real"))
                out.write(json.dumps({"ep": ep.name, "kf": ep.kf, "p3": "wanhist", "depth": d,
                                      "vs_oracle_agent": a, "vs_oracle_wrist": w,
                                      "hist_err_agent": ha, "hist_err_wrist": hw}) + "\n")
        for (name, d), (s, s2, n) in acc.items():
            if n < 2:
                continue
            ep = next(e for e in eps if e.name == name)
            real = ep.real_chunk(d).astype(np.float32)
            mean = s / n
            var = (s2 / n - mean * mean) * n / (n - 1)
            rec = {"ep": name, "kf": ep.kf, "biasvar": True, "chunk": d + 1, "n_seeds": n}
            for v, sl in (("agent", slice(0, 256)), ("wrist", slice(256, 512))):
                rec[f"bias2_{v}"] = float(((mean[:, sl] - real[:, sl]) ** 2).mean())
                rec[f"var_{v}"] = float(var[:, sl].mean())
            out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"shard {args.shard}: {min(ei + 4, len(mine))}/{len(mine)} episodes", flush=True)
    out.close()


if __name__ == "__main__":
    main()
