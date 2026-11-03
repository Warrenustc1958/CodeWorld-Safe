#!/usr/bin/env python3
"""T2: per-chunk history repair adapter on Wan's history latent (slot 1 of first_frame_latents), Wan frozen.

The WM backend VAE-encodes [anchor, 4 history frames] into 2 latent slots before every chunk: slot 0 the anchor,
slot 1 the history. Wan's open-loop error is mostly inherited through slot 1 (WAN_DRIFT_PLAN 1.2, gain ~0.9-1).
The adapter maps the generated slot 1 towards the slot 1 of the real history, per view (agent / wrist halves of
the dual latent), from information available at inference only:
  z_gen   slot 1 of [anchor, generated history]   (what Wan would be given)
  z_anc   slot 0 (anchor)
  pose    the 4 history frames' proprio: training uses the measured state + noise of the size of the causal
          estimator's error (R4: ~3 cm / 3 deg at 256 steps); inference uses that estimator
  depth   chunk index
  out     z_gen + delta, delta from a zero-initialised last layer (step 0 = no repair)
Loss: MSE to slot 1 of [anchor, real history]. kappa = MSE(after) / MSE(before) on held-out episodes is the
latent analogue of R1's "pull the history back by 1 - kappa".

  python t2_adapter.py cache --t0 DIR --split train|val|test --out CACHE --shard i --nshard n [--episodes N]
  python t2_adapter.py train --cache CACHE --out RUN [--steps 20000]
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = "/mnt/cpfs/jieduan/data/cws_rollout"
POSE_DIR = "/mnt/cpfs/jieduan/work/scgrpo/drift/pose_index"
CHUNK = 8
ZC = 48  # latent channels


# ------------------------------------------------------------------ cache
def _dec(buf, off, i):
    return np.asarray(Image.open(io.BytesIO(buf[off[i]:off[i + 1]].tobytes())).convert("RGB"))


def cache(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from t0_gen import load_frames  # noqa: E402

    device = args.device
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    from rlinf.envs.sim.world_model.safe.safe_models import enable_dual

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path="/mnt/cpfs/jieduan/data/wan_dual_41000/ema-step-41000.safetensors",
                                   offload_device="cpu"),
                       ModelConfig(path="/mnt/cpfs/jieduan/models/wan_vae/Wan2.2_VAE.pth", offload_device="cpu")])
    pipe.vae.to(device)
    pipe.dit.to("cpu")  # only the VAE is needed
    enable_dual(pipe)
    rows = {json.loads(l)["idx"]: json.loads(l) for l in open(f"{ROOT}/index/cws_v1/manifest.jsonl")}
    files = sorted(glob.glob(os.path.join(args.t0, args.split, "*.npz")))
    if args.episodes:
        files = files[: args.episodes]
    files = files[args.shard::args.nshard]
    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(ROOT)

    @torch.no_grad()
    def encode(videos):  # list of [5 PIL] -> [B, 48, 2, 32, 16] (dual: agent rows 0-15, wrist 16-31)
        z = pipe.vae.encode(pipe.preprocess_video(videos), device=device, tiled=False)
        return z.float().cpu()

    for f in files:
        idx = os.path.basename(f)[:-4]
        dst = os.path.join(out_dir, f"{idx}.npz")
        if os.path.exists(dst) or idx not in rows:
            continue
        z = np.load(f)
        D = int(z["n_chunks"])
        if D < 2:
            continue
        row = rows[idx]
        n = 1 + CHUNK * D
        agent = load_frames(row["agent_segments"], n, "agentview_image")
        wrist = load_frames(row["wrist_segments"], n, "robot0_eye_in_hand_image")
        real = np.concatenate([agent, wrist], 1)
        pose = np.load(os.path.join(POSE_DIR, f"{idx}.npy"))
        anc = Image.fromarray(real[0])
        zg, zr, ps = [], [], []
        ds = list(range(1, D))
        for b in range(0, len(ds), args.batch):
            sub = ds[b:b + args.batch]
            gen5 = [[anc] + [Image.fromarray(_dec(z["hist_jpg"], z["hist_off"], 4 * (d - 1) + k)) for k in range(4)]
                    for d in sub]
            real5 = [[anc] + [Image.fromarray(real[CHUNK * d - 3 + k]) for k in range(4)] for d in sub]
            eg, er = encode(gen5), encode(real5)
            zg.append(eg[:, :, 1]); zr.append(er[:, :, 1])
            if b == 0:
                za = er[0, :, 0]
            ps += [pose[CHUNK * d - 3: CHUNK * d + 1] for d in sub]
        np.savez(dst, z_gen=torch.cat(zg).half().numpy(), z_real=torch.cat(zr).half().numpy(),
                 z_anc=za.half().numpy(), pose=np.stack(ps).astype(np.float32), d=np.asarray(ds, np.int16))
        print(f"cached {idx} ({len(ds)} chunks)", flush=True)


# ------------------------------------------------------------------ model
def pose_feat(p, stats):
    """[..., 4, 9] -> [..., 44]: per frame normalised pos, 6-D rotation, normalised gripper (as pose_cond.py)."""
    mu = torch.as_tensor(stats["mean"], dtype=p.dtype, device=p.device)
    sd = torch.as_tensor(stats["std"], dtype=p.dtype, device=p.device)
    pos = (p[..., :3] - mu[:3]) / sd[:3]
    q = F.normalize(p[..., 3:7], dim=-1)
    x, y, zq, w = q.unbind(-1)
    c0 = torch.stack([1 - 2 * (y * y + zq * zq), 2 * (x * y + zq * w), 2 * (x * zq - y * w)], -1)
    c1 = torch.stack([2 * (x * y - zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq + x * w)], -1)
    grip = (p[..., 7:9] - mu[7:9]) / sd[7:9].clamp_min(1e-6)
    return torch.cat([pos, c0, c1, grip], -1).flatten(-2)


class FiLMBlock(nn.Module):
    def __init__(self, c, cond):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, 1, 1)
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, 1, 1)
        self.film = nn.Linear(cond, 2 * c)

    def forward(self, x, cond):
        s, b = self.film(cond)[:, :, None, None].chunk(2, 1)
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h) * (1 + s) + b
        return x + self.c2(F.silu(h))


class Adapter(nn.Module):
    """One per view. Input latents [B, 48, 16, 16] (one view half), cond = pose (44) + depth embedding."""

    def __init__(self, width=256, blocks=6, cond=256, max_depth=64):
        super().__init__()
        self.inp = nn.Conv2d(2 * ZC, width, 3, 1, 1)
        self.cond = nn.Sequential(nn.Linear(44, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.depth = nn.Embedding(max_depth + 1, cond)
        self.blocks = nn.ModuleList([FiLMBlock(width, cond) for _ in range(blocks)])
        self.out = nn.Conv2d(width, ZC, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_gen, z_anc, pose44, depth):
        c = self.cond(pose44) + self.depth(depth.clamp(max=self.depth.num_embeddings - 1))
        h = self.inp(torch.cat([z_gen, z_anc], 1))
        for b in self.blocks:
            h = b(h, c)
        return z_gen + self.out(F.silu(h))


class Repair(nn.Module):
    """Both views; dual latent rows 0-15 = agent, 16-31 = wrist (per-view VAE, stacked)."""

    def __init__(self, **kw):
        super().__init__()
        self.agent, self.wrist = Adapter(**kw), Adapter(**kw)

    def forward(self, z_gen, z_anc, pose44, depth):
        h = z_gen.shape[-2] // 2
        a = self.agent(z_gen[..., :h, :], z_anc[..., :h, :], pose44, depth)
        w = self.wrist(z_gen[..., h:, :], z_anc[..., h:, :], pose44, depth)
        return torch.cat([a, w], -2)


# ------------------------------------------------------------------ train
def load_cache(d):
    G, Rr, A, P, D, E = [], [], [], [], [], []
    for i, f in enumerate(sorted(glob.glob(os.path.join(d, "*.npz")))):
        z = np.load(f)
        n = len(z["d"])
        G.append(z["z_gen"]); Rr.append(z["z_real"]); P.append(z["pose"]); D.append(z["d"])
        A.append(np.repeat(z["z_anc"][None], n, 0)); E.append(np.full(n, i))
    return {k: np.concatenate(v) for k, v in zip(("gen", "real", "anc", "pose", "d", "ep"), (G, Rr, A, P, D, E))}


def noisy_pose(p, pos_cm, rot_deg, g):
    """Measured state + estimator-sized noise (same draw for the 4 frames of a history: a smooth estimator)."""
    from scipy.spatial.transform import Rotation as Rot

    out = p.copy()
    n = len(p)
    out[:, :, :3] += (g.normal(0, pos_cm / 100 / math.sqrt(3), (n, 1, 3))).astype(np.float32)
    rv = g.normal(0, math.radians(rot_deg) / math.sqrt(3), (n, 3))
    r = Rot.from_rotvec(rv)
    for k in range(p.shape[1]):
        out[:, k, 3:7] = (r * Rot.from_quat(p[:, k, 3:7])).as_quat()
    return out


def train(args):
    dev = args.device
    stats = json.load(open(os.path.join(POSE_DIR, "stats.json")))
    tr, va = load_cache(os.path.join(args.cache, "train")), load_cache(os.path.join(args.cache, "val"))
    print(f"train {len(tr['d'])} histories ({len(np.unique(tr['ep']))} episodes), val {len(va['d'])}", flush=True)
    model = Repair(width=args.width, blocks=args.blocks).to(dev)
    print(f"adapter params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    g = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    def batch(data, ix, noise=True):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        pose = noisy_pose(data["pose"][ix], args.pose_noise_cm, args.pose_noise_deg, g) if noise else data["pose"][ix]
        return (t(data["gen"][ix]).float(), t(data["anc"][ix]).float(), t(data["real"][ix]).float(),
                pose_feat(t(pose), stats), t(data["d"][ix].astype(np.int64)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        before = np.zeros(2); after = np.zeros(2)
        for i in range(0, len(va["d"]), 512):
            ix = np.arange(i, min(i + 512, len(va["d"])))
            zg, za, zr, pf, d = batch(va, ix)
            h = zg.shape[-2] // 2
            out = model(zg, za, pf, d)
            for v, sl in enumerate((slice(0, h), slice(h, None))):
                before[v] += ((zg[..., sl, :] - zr[..., sl, :]) ** 2).mean((1, 2, 3)).sum().item()
                after[v] += ((out[..., sl, :] - zr[..., sl, :]) ** 2).mean((1, 2, 3)).sum().item()
        model.train()
        return {"kappa_agent": after[0] / before[0], "kappa_wrist": after[1] / before[1],
                "mse_before": (before / len(va["d"])).tolist(), "mse_after": (after / len(va["d"])).tolist()}

    print("step 0", json.dumps(evaluate()), flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        ix = g.integers(0, len(tr["d"]), args.batch)
        zg, za, zr, pf, d = batch(tr, ix)
        loss = F.mse_loss(model(zg, za, pf, d), zr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = {"step": step, "loss": loss.item(), "sec": time.time() - t0, **evaluate()}
            print(json.dumps(ev), flush=True)
            log.write(json.dumps(ev) + "\n"); log.flush()
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, os.path.join(args.out, "adapter.pt"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--t0", required=True); c.add_argument("--split", required=True); c.add_argument("--out", required=True)
    c.add_argument("--shard", type=int, default=0); c.add_argument("--nshard", type=int, default=1)
    c.add_argument("--episodes", type=int, default=0); c.add_argument("--batch", type=int, default=16)
    c.add_argument("--device", default="cuda:0")
    t = sub.add_parser("train")
    t.add_argument("--cache", required=True); t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20000); t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--width", type=int, default=256)
    t.add_argument("--blocks", type=int, default=6); t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--pose-noise-cm", type=float, default=3.0); t.add_argument("--pose-noise-deg", type=float, default=3.0)
    t.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cache(args) if args.cmd == "cache" else train(args)


if __name__ == "__main__":
    main()
