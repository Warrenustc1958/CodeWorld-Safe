# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the CodeWorld-Safe success reward model on cached frames.

SafeDojo recipe (App. J): init from the WoVR/LIBERO RM, 20 epochs, batch 256,
lr 1e-4, BCEWithLogits with pos_weight = neg/pos. Model selection on the mean
frame AP over the validation splits; the best checkpoint is then scored on the
test splits. Episode metrics mirror how the world-model env uses the RM: an
episode counts as successful once a frame reaches the success threshold
(``success_reward_threshold``, default 0.9).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel
from toolkits.codeworld_safe.rm.data import RMFrames

QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


class _Indexed(torch.utils.data.Dataset):
    """Replace the per-set index returned by RMFrames with the global ConcatDataset index."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, t, _ = self.ds[i]
        return x, y, t, i


def augment(x: torch.Tensor, shift_px: int = 4, bright: float = 0.1, contrast: float = 0.1):
    """Per-image integer shift (border padding) and brightness/contrast jitter, x in [-1, 1]."""
    n, _, h, w = x.shape
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    shifts = torch.randint(-shift_px, shift_px + 1, (n, 2), device=x.device).float()
    theta[:, 0, 2] = shifts[:, 0] * 2 / w
    theta[:, 1, 2] = shifts[:, 1] * 2 / h
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="nearest", padding_mode="border", align_corners=False)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    c = 1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * contrast
    b = (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * bright * 2  # 0.1 of the [0, 1] range
    return ((x - mean) * c + mean + b).clamp(-1, 1)


def to_input(x: torch.Tensor, views: int, aug: bool) -> torch.Tensor:
    """uint8 [B, V*256, 256, 3] -> float [B, 3, V*256, 256] in [-1, 1]; views augmented independently."""
    x = x.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
    if aug:
        b, c, h, w = x.shape
        v = x.reshape(b, c, views, h // views, w).transpose(1, 2).reshape(b * views, c, h // views, w)
        x = augment(v).reshape(b, views, c, h // views, w).transpose(1, 2).reshape(b, c, h, w)
    return x.contiguous(memory_format=torch.channels_last)


def _rank(s: np.ndarray) -> np.ndarray:
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]  # average ranks for ties


def frame_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = y.astype(bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return {"pos": pos, "neg": neg}
    auroc = (_rank(p)[y].sum() - pos * (pos + 1) / 2) / (pos * neg)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / pos
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    j = int(f1.argmax())
    out = {"pos": pos, "neg": neg, "auroc": float(auroc), "ap": float(prec[ys].mean()),
           "best_f1": float(f1[j]), "best_thr": float(p[order][j])}
    for thr in (0.5, 0.9):
        pred = p >= thr
        out[f"f1@{thr}"] = float(2 * (pred & y).sum() / max(pred.sum() + pos, 1))
    return out


def episode_metrics(meta: dict, episodes: list, p: np.ndarray) -> dict:
    ep = meta["episode"]
    n = len(episodes)
    pmax = np.full(n, -1.0)
    np.maximum.at(pmax, ep, p)
    early = meta["dist_to_end"] > 8  # outside the last 8-step chunk before the end
    pearly = np.zeros(n)
    np.maximum.at(pearly, ep[early], p[early])
    succ = np.array([e["success"] for e in episodes])
    quad = np.array([e["quadrant"] for e in episodes])
    out = {"episodes": n, "success_episodes": int(succ.sum())}
    for thr in (0.5, 0.9):
        det = pmax >= thr
        out[f"thr{thr}"] = {
            "acc": float((det == succ).mean()),
            "miss_on_success": float((~det[succ]).mean()) if succ.any() else None,
            "fire_on_failure": float(det[~succ].mean()) if (~succ).any() else None,
            "premature_on_success": float((pearly[succ] >= thr).mean()) if succ.any() else None,
            "acc_by_quadrant": {q: float((det[quad == q] == succ[quad == q]).mean())
                                for q in QUADRANTS if (quad == q).any()},
        }
    return out


@torch.no_grad()
def evaluate(model, ds: RMFrames, args) -> dict:
    loader = DataLoader(ds, batch_size=args.eval_bs, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    model.eval()
    probs = []
    for x, _, t, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(to_input(x, args.views, False), t.cuda(non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu())
    model.train()
    p = torch.cat(probs).numpy()
    return {"frame": frame_metrics(ds.meta["label"], p), "episode": episode_metrics(ds.meta, ds.episodes, p)}


def save(model, path: Path, **extra):
    torch.save({"model_state_dict": model.state_dict(), "config": model.config(), **extra}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", required=True, help="build_frames.py output dir")
    ap.add_argument("--extra_train", nargs="*", default=[],
                    help="extra frame sets as DIR:SPLIT (e.g. world-model frames), same tasks.json")
    ap.add_argument("--views", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="WoVR/diffsynth reward checkpoint for the encoder")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--eval_bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--pos_weight_cap", type=float, default=0.0, help="0 = uncapped neg/pos")
    ap.add_argument("--extra_frac", type=float, default=0.0,
                    help="sample --extra_train frames as this fraction of each epoch (0 = plain shuffle by size)")
    ap.add_argument("--domain_pos_weight", action="store_true",
                    help="pos_weight = neg/pos computed per train set instead of over the union")
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--val_splits", nargs="+", default=["val", "newval"])
    ap.add_argument("--test_splits", nargs="+", default=["test", "newtest"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem_gb", type=float, default=30.0, help="per-process CUDA memory cap (shared GPUs)")
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--eval_only", default=None, help="checkpoint to score on val+test splits")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = Path(args.frames)
    tasks = json.loads((frames / "tasks.json").read_text())
    total_mem = torch.cuda.get_device_properties(0).total_memory
    if args.mem_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / total_mem))

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(out / "log.txt", "a") as f:
            f.write(line + "\n")

    if args.eval_only:
        model = CWSRewardModel.from_checkpoint(args.eval_only).cuda().to(memory_format=torch.channels_last)
        res = {s: evaluate(model, RMFrames(frames, s, args.views), args)
               for s in args.val_splits + args.test_splits}
        (out / "eval_only.json").write_text(json.dumps(res, indent=1))
        log(json.dumps({s: r["frame"] for s, r in res.items()}))
        return

    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    model = CWSRewardModel(views=args.views, instructions=tasks["instructions"], init=args.init)
    model = model.cuda().to(memory_format=torch.channels_last)

    train_sets = [RMFrames(frames, "train", args.views)]
    for spec in args.extra_train:
        root, split = spec.rsplit(":", 1)
        if json.loads((Path(root) / "tasks.json").read_text())["instructions"] != tasks["instructions"]:
            raise ValueError(f"{root} has a different task vocabulary")
        train_sets.append(RMFrames(root, split, args.views))
    labels = np.concatenate([d.meta["label"] for d in train_sets])
    domain = np.concatenate([np.full(len(d), k, np.int64) for k, d in enumerate(train_sets)])
    pos, neg = int(labels.sum()), int((~labels).sum())

    def capped(n_neg, n_pos):
        w = n_neg / max(n_pos, 1)
        return min(w, args.pos_weight_cap) if args.pos_weight_cap > 0 else w

    if args.domain_pos_weight:
        dom_pw = [capped(int((~d.meta["label"]).sum()), int(d.meta["label"].sum())) for d in train_sets]
    else:
        dom_pw = [capped(neg, pos)] * len(train_sets)
    val_sets = {s: RMFrames(frames, s, args.views) for s in args.val_splits}
    log(f"views={args.views} train frames={len(labels)} pos={pos} neg={neg} "
        f"per-set frames/pos={[(len(d), int(d.meta['label'].sum())) for d in train_sets]} "
        f"pos_weight={[round(w, 1) for w in dom_pw]} extra_frac={args.extra_frac} "
        f"val={ {s: len(d) for s, d in val_sets.items()} }")

    g = torch.Generator().manual_seed(args.seed)
    train_ds = _Indexed(ConcatDataset(train_sets))
    if args.extra_frac > 0 and len(train_sets) > 1:
        # real set gets 1 - extra_frac of the draws, extra sets share extra_frac by size; epoch length unchanged
        n_real, n_extra = len(train_sets[0]), len(labels) - len(train_sets[0])
        w = np.where(domain == 0, (1 - args.extra_frac) / n_real, args.extra_frac / n_extra)
        sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(labels), replacement=True,
                                        generator=g)
        loader = DataLoader(train_ds, batch_size=args.bs, sampler=sampler, drop_last=True,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    else:
        loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4, generator=g)
    total = args.epochs * len(loader)
    if args.max_steps:
        total = min(total, args.max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    pw_of = torch.tensor(dom_pw, dtype=torch.float32)[torch.from_numpy(domain)]  # per-frame pos_weight

    step, best, t0 = 0, -1.0, time.time()
    for epoch in range(args.epochs):
        run_loss, run_n = 0.0, 0
        for x, y, t, gi in loader:
            xb = to_input(x, args.views, not args.no_aug)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb, t.cuda(non_blocking=True))
            y = y.cuda(non_blocking=True)
            w = torch.where(y > 0, pw_of[gi].cuda(non_blocking=True), torch.ones_like(y))
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_n += 1
            if step % 100 == 0:
                log(f"ep {epoch} step {step}/{total} loss {run_loss / run_n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{step / (time.time() - t0):.2f} it/s maxmem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                run_loss, run_n = 0.0, 0
            if step >= total:
                break
        res = {s: evaluate(model, d, args) for s, d in val_sets.items()}
        aps = [r["frame"].get("ap", 0.0) for r in res.values()]
        score = float(np.mean(aps))
        with open(out / "val_metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, "score": score, **res}) + "\n")
        log(f"ep {epoch} val " + " ".join(
            f"{s}: ap {r['frame'].get('ap', 0):.3f} auroc {r['frame'].get('auroc', 0):.3f} "
            f"ep_acc@0.9 {r['episode']['thr0.9']['acc']:.3f}" for s, r in res.items()))
        save(model, out / "last.pt", epoch=epoch, step=step, val=res)
        if score > best:
            best = score
            save(model, out / "best.pt", epoch=epoch, step=step, val=res)
            log(f"new best {best:.4f} at epoch {epoch}")
        if step >= total:
            break

    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    res = {s: evaluate(model, RMFrames(frames, s, args.views), args) for s in args.test_splits}
    final = {"best_epoch": ckpt["epoch"], "val": ckpt["val"], "test": res,
             "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "hours": (time.time() - t0) / 3600}
    (out / "final.json").write_text(json.dumps(final, indent=1))
    log("test " + json.dumps({s: r["frame"] for s, r in res.items()}))


if __name__ == "__main__":
    main()
