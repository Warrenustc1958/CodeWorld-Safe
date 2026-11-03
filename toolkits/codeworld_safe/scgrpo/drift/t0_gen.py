#!/usr/bin/env python3
"""T0: open-loop Wan histories on recorded step_e episodes (training pairs for T2/T3, offline eval set).

For each dual-view episode of the cws_v1 index, dual-41k rolls out chunk by chunk from frame 0 exactly as the WM
backend does (anchor = frame 0, history = the last 4 generated frames, frame 0 x4 for the first chunk; actions
in the backend convention: action[f] = recorded action that produced frame f, frame 0 = null), with the recorded
actions. No simulator is needed: the recorded frames are the real future of those actions.
Saved per episode: the generated history before every chunk d >= 1 (4 dual frames, JPEG q90), the noise seed of
every chunk, and per-chunk MAE (agent / wrist) of the generated chunk vs the recorded one. A training pair is
(generated history before chunk d, actions of chunk d, recorded frames 8d+1..8d+8).

  python t0_gen.py --split train --episodes 3000 --out DIR --shard i --nshard n [--batch 32] [--max-chunks 64]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import queue

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_replay import NULL_ACTION, build_pipe, scale_pose  # noqa: E402

ROOT = "/mnt/cpfs/jieduan/data/cws_rollout"
CHUNK = 8


def load_frames(segments, n, key):
    """Frames 0..n-1 stitched from [npz, start, stop] segments (paths relative to ROOT)."""
    out = []
    for path, a, b in segments:
        with np.load(os.path.join(ROOT, path)) as z:
            out.append(z[key][a:b])
        if sum(len(x) for x in out) >= n:
            break
    return np.concatenate(out)[:n]


def load_episode(row, max_chunks):
    n = min(row["num_frames"], 1 + CHUNK * max_chunks)
    D = (n - 1) // CHUNK
    n = 1 + CHUNK * D
    agent = load_frames(row["agent_segments"], n, "agentview_image")
    wrist = load_frames(row["wrist_segments"], n, "robot0_eye_in_hand_image")
    if len(agent) < n or len(wrist) < n:
        raise ValueError(f"{row['idx']}: {len(agent)}/{len(wrist)} frames < {n}")
    with np.load(os.path.join(ROOT, row["labels"])) as lab:
        acts = lab["actions"].astype(np.float32)
    a = np.stack([NULL_ACTION] + [acts[f - 1] for f in range(1, n)])  # a[f] produced frame f
    return {"idx": row["idx"], "frames": np.concatenate([agent, wrist], 1), "a": a, "D": D}


def jpg(x):
    b = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(x)).save(b, format="JPEG", quality=90)
    return b.getvalue()


class Slot:
    def __init__(self, ep):
        self.ep, self.d = ep, 0
        self.mem = np.repeat(ep["frames"][:1], 4, 0)
        self.mem_a = np.repeat(NULL_ACTION[None], 4, 0)
        self.hist, self.seeds, self.mae = [], [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=0, help="0 = all dual episodes of the split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-chunks", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{ROOT}/index/cws_v1/manifest.jsonl")]
    rows = [r for r in rows if r["split"] == args.split and r["dual"] and r["num_frames"] >= 1 + CHUNK]
    rows.sort(key=lambda r: r["idx"])
    if args.episodes:
        rows = [rows[i] for i in sorted(np.random.default_rng(0).permutation(len(rows))[: args.episodes])]
    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    mine = [r for i, r in enumerate(rows) if i % args.nshard == args.shard
            and not os.path.exists(os.path.join(out_dir, f"{r['idx']}.npz"))]

    q: queue.Queue = queue.Queue(maxsize=2 * args.batch)

    def loader():
        for r in mine:
            try:
                q.put(load_episode(r, args.max_chunks))
            except Exception as exc:  # noqa: BLE001
                print("skip", r["idx"], repr(exc), flush=True)
        q.put(None)

    threading.Thread(target=loader, daemon=True).start()
    pipe = build_pipe("cuda:0")
    dev = next(pipe.dit.parameters()).device
    slots, exhausted, done = [], False, 0
    while True:
        while len(slots) < args.batch and not exhausted:
            ep = q.get()
            if ep is None:
                exhausted = True
            else:
                slots.append(Slot(ep))
        if not slots:
            break
        win = np.stack([np.concatenate([NULL_ACTION[None], s.mem_a, s.ep["a"][1 + CHUNK * s.d: 1 + CHUNK * (s.d + 1)]])
                        for s in slots])
        noise = []
        for s in slots:
            seed = int(s.ep["idx"][1:]) * 1009 + s.d  # deterministic across processes (idx = n#####)
            s.seeds.append(seed)
            g = torch.Generator("cpu").manual_seed(seed)
            noise.append(torch.randn((48, 4, 32, 16), generator=g, dtype=torch.float32))
        pipe._fixed_noise = torch.stack(noise)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = pipe(seed=0, tiled=False, input_image=[Image.fromarray(s.ep["frames"][0]) for s in slots],
                        input_image4=[[Image.fromarray(np.ascontiguousarray(m)) for m in s.mem] for s in slots],
                        action=torch.from_numpy(scale_pose(win)).to(dev), height=512, width=256, num_frames=13,
                        num_inference_steps=5, cfg_scale=1.0, progress_bar_cmd=lambda x: x, batch_size=len(slots))
        pipe._fixed_noise = None
        keep = []
        for s, o in zip(slots, outs):
            fr = np.stack([np.asarray(im, np.uint8) for im in o[5:13]])
            real = s.ep["frames"][1 + CHUNK * s.d: 1 + CHUNK * (s.d + 1)].astype(np.int16)
            diff = np.abs(fr.astype(np.int16) - real)
            s.mae.append((float(diff[:, :256].mean()), float(diff[:, 256:].mean())))
            s.d += 1
            if s.d < s.ep["D"]:
                s.mem, s.mem_a = fr[-4:], s.ep["a"][1 + CHUNK * s.d - 4: 1 + CHUNK * s.d]
                s.hist.append([jpg(x) for x in fr[-4:]])  # history before chunk s.d
                keep.append(s)
            else:
                flat = [b for h in s.hist for b in h]
                np.savez(os.path.join(out_dir, f"{s.ep['idx']}.npz"),
                         hist_jpg=np.frombuffer(b"".join(flat), np.uint8) if flat else np.zeros(0, np.uint8),
                         hist_off=np.cumsum([0] + [len(b) for b in flat]).astype(np.int64),
                         seeds=np.asarray(s.seeds, np.int64), mae=np.asarray(s.mae, np.float32),
                         n_chunks=s.ep["D"])
                done += 1
                if done % 20 == 0:
                    print(f"shard {args.shard}: {done}/{len(mine)} episodes", flush=True)
        slots = keep
    print(f"shard {args.shard}: finished {done}/{len(mine)}", flush=True)


if __name__ == "__main__":
    main()
