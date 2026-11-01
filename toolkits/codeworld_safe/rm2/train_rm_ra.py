"""R-A: retrain the dual-view CWSRewardModel on a build_rm_dataset directory.

  python train_rm_ra.py DS_DIR OUT_DIR --init /data_cf/cws_rm/runs/rm_v8_dual/best.pt [--epochs 6]
Loss = BCE (positives weighted to balance the 1:2 mix, hard negatives x2) + within-group ranking: a positive frame of
a success episode vs a hard negative of a failed episode from the same rollout group (softplus margin loss).
Model selection on val frame AP; the threshold is calibrated on val to <= 1% failed-episode fire rate (--max_fp).
Writes best.pt (CWSRewardModel checkpoint + "threshold"), final.json with val and test metrics.
"""
import argparse
import io
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel

import rm_metrics


class Split(torch.utils.data.Dataset):
    def __init__(self, d, split):
        self.rows = [json.loads(l) for l in open(f"{d}/{split}.jsonl")]
        self.off = np.load(f"{d}/{split}.off.npy")
        self.path = f"{d}/{split}.jpg.bin"
        self._buf = None

    def __len__(self):
        return len(self.rows)

    def image(self, i):
        if self._buf is None:
            self._buf = np.memmap(self.path, np.uint8, mode="r")
        img = np.asarray(Image.open(io.BytesIO(self._buf[self.off[i]:self.off[i + 1]].tobytes())).convert("RGB"))
        return torch.from_numpy(img).permute(2, 0, 1).float().div(127.5).sub(1.0)

    def __getitem__(self, i):
        r = self.rows[i]
        w = 2.0 if r["label"] else (2.0 if r["category"].startswith("hard") else 1.0)
        return self.image(i), float(r["label"]), r["instruction"], w


class Pairs(torch.utils.data.Dataset):
    """(positive of a success episode, hard negative of a failed episode) from the same rollout group."""

    def __init__(self, ds):
        self.ds = ds
        pos, neg = defaultdict(list), defaultdict(list)
        for i, r in enumerate(ds.rows):
            g = (r["col"], r["file"], r["reset"], r["group"])
            if r["label"] and r["ep_success"]:
                pos[g].append(i)
            elif not r["label"] and not r["ep_success"] and r["category"].startswith("hard"):
                neg[g].append(i)
        self.pairs = [(a, b) for g in pos for a in pos[g] for b in neg.get(g, [])]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        a, b = self.pairs[k]
        return self.ds.image(a), self.ds.image(b), self.ds.rows[a]["instruction"]


@torch.no_grad()
def predict(model, ds, dev, bs=256, workers=8):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers)
    out = []
    for x, _, inst, _ in dl:
        out.append(torch.sigmoid(model(x.to(dev), model.task_ids(list(inst)).to(dev))).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("out")
    ap.add_argument("--init", default="/data_cf/cws_rm/runs/rm_v8_dual/best.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--pair_bs", type=int, default=32)
    ap.add_argument("--rank_w", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--max_fp", type=float, default=0.01, help="failed-episode fire rate for the threshold")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    model = CWSRewardModel.from_checkpoint(a.init).to(dev)
    assert model.views == 2, "R-A expects the dual-view model"
    train, val = Split(a.ds, "train"), Split(a.ds, "val")
    test = Split(a.ds, "test") if os.path.getsize(f"{a.ds}/test.jsonl") else None
    missing = {r["instruction"] for d in (train, val) for r in d.rows} - set(model.instructions)
    if missing:
        raise SystemExit(f"instructions not in the RM vocabulary: {sorted(missing)[:5]}")
    pairs = Pairs(train)
    print(f"train {len(train)} val {len(val)} test {len(test) if test else 0} pairs {len(pairs)}", flush=True)
    dl = torch.utils.data.DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    pl = (torch.utils.data.DataLoader(pairs, batch_size=a.pair_bs, shuffle=True, num_workers=a.workers // 2,
                                      drop_last=True) if len(pairs) >= a.pair_bs else None)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(total, 1), pct_start=0.05)
    best, step, log = -1.0, 0, []
    for ep in range(a.epochs):
        pit = iter(pl) if pl else None
        t0 = time.time()
        for x, y, inst, w in dl:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            logit = model(x, model.task_ids(list(inst)).to(dev))
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()
            if pit is not None:
                try:
                    xp, xn, pinst = next(pit)
                except StopIteration:
                    pit = iter(pl)
                    xp, xn, pinst = next(pit)
                tid = model.task_ids(list(pinst)).to(dev)
                sp, sn = model(xp.to(dev), tid), model(xn.to(dev), tid)
                loss = loss + a.rank_w * F.softplus(-(sp - sn) + 1.0).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        pv = predict(model, val, dev)
        thr = rm_metrics.calibrate(val.rows, pv, a.max_fp)
        res = rm_metrics.score(val.rows, pv, thr)
        log.append({"epoch": ep, "val": res})
        print(f"epoch {ep} val {json.dumps(res)}", flush=True)
        if (res["frame_ap"] or 0) > best:
            best = res["frame_ap"] or 0
            torch.save({"model_state_dict": model.state_dict(), "config": model.config(), "epoch": ep,
                        "threshold": thr, "val": res}, f"{a.out}/best.pt")
        if a.max_steps and step >= a.max_steps:
            break
    ck = torch.load(f"{a.out}/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    final = {"best_epoch": ck["epoch"], "threshold": ck["threshold"], "val": ck["val"], "log": log}
    if test:
        pt = predict(model, test, dev)
        final["test"] = rm_metrics.score(test.rows, pt, ck["threshold"])
    json.dump(final, open(f"{a.out}/final.json", "w"), indent=1)
    print("FINAL", json.dumps({k: v for k, v in final.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()
