"""R-C: LoRA fine-tuning of Qwen3-VL-4B-Instruct as a yes/no success RM on a build_rm_dataset directory.

  torchrun --nproc_per_node N train_rm_rc.py DS_DIR OUT_DIR [--epochs 1] [--lr 1e-4]
LoRA r16 / alpha 16 / dropout 0.05 on the language-model attention and MLP projections; the vision tower is frozen
(RLinf's Qwen3-VL reward SFT recipe also uses LoRA r16). Loss = weighted binary CE on log p(yes) - log p(no)
(positives and hard negatives x2, as R-A). Saves the adapter of the best val frame AP; the threshold is calibrated on
val to <= 1% failed-episode fire rate. Writes OUT_DIR/best (adapter), final.json.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import qwen_rm
import rm_metrics


class Split(torch.utils.data.Dataset):
    def __init__(self, d, split):
        self.rows = [json.loads(l) for l in open(f"{d}/{split}.jsonl")]
        self.off = np.load(f"{d}/{split}.off.npy")
        self.path = f"{d}/{split}.jpg.bin"
        self._buf = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._buf is None:
            self._buf = np.memmap(self.path, np.uint8, mode="r")
        r = self.rows[i]
        w = 2.0 if r["label"] or r["category"].startswith("hard") else 1.0
        return qwen_rm.decode(self._buf[self.off[i]:self.off[i + 1]].tobytes()), r["instruction"], float(r["label"]), w


def collate(b):
    return [x[0] for x in b], [x[1] for x in b], torch.tensor([x[2] for x in b]), torch.tensor([x[3] for x in b])


@torch.no_grad()
def eval_split(model, proc, ds, rank, world, bs):
    model.eval()
    idx = list(range(rank, len(ds), world))
    imgs, insts = [], []
    for i in idx:
        im, inst, _, _ = ds[i]
        imgs.append(im)
        insts.append(inst)
    p = qwen_rm.score_images(model, proc, imgs, insts, bs)
    g = [None] * world
    dist.all_gather_object(g, (idx, p.tolist()))
    probs = np.zeros(len(ds))
    for gi, gp in g:
        probs[gi] = gp
    model.train()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8, help="per GPU")
    ap.add_argument("--evals", type=int, default=3, help="val evaluations over the run")
    ap.add_argument("--max_fp", type=float, default=0.01)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(a.seed)
    random.seed(a.seed + rank)
    from peft import LoraConfig, get_peft_model

    model, proc = qwen_rm.load()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    if rank == 0:
        model.print_trainable_parameters()
    model.train()
    yes, no = qwen_rm.answer_ids(proc.tokenizer)
    train, val = Split(a.ds, "train"), Split(a.ds, "val")
    sampler = torch.utils.data.DistributedSampler(train, world, rank, shuffle=True, seed=a.seed)
    dl = torch.utils.data.DataLoader(train, batch_size=a.bs, sampler=sampler, num_workers=6, collate_fn=collate,
                                     drop_last=True)
    steps = a.max_steps or int(math.ceil(a.epochs * len(dl)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    warm = min(50, steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    eval_at = {int(steps * (k + 1) / a.evals) for k in range(a.evals)}
    dev = torch.device("cuda")
    best, step, log, t0 = -1.0, 0, [], time.time()
    ep = 0
    while step < steps:
        sampler.set_epoch(ep)
        for imgs, insts, y, w in dl:
            batch = qwen_rm.build(proc, imgs, insts).to(dev)
            d = qwen_rm.yes_logit(model, batch, yes, no)
            loss = (F.binary_cross_entropy_with_logits(d, y.to(dev), reduction="none") * w.to(dev)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in params:  # DDP-free gradient all-reduce (only the LoRA parameters)
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if rank == 0 and step % 20 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step in eval_at:
                probs = eval_split(model, proc, val, rank, world, 32)
                if rank == 0:
                    thr = rm_metrics.calibrate(val.rows, probs, a.max_fp)
                    res = rm_metrics.score(val.rows, probs, thr)
                    log.append({"step": step, "val": res})
                    print(f"eval step {step} {json.dumps(res)}", flush=True)
                    if (res["frame_ap"] or 0) > best:
                        best = res["frame_ap"] or 0
                        model.save_pretrained(f"{a.out}/best")
                        json.dump({"step": step, "threshold": thr, "val": res}, open(f"{a.out}/best/meta.json", "w"))
                dist.barrier()
            if step >= steps:
                break
        ep += 1
    if rank == 0:
        meta = json.load(open(f"{a.out}/best/meta.json"))
        json.dump({**meta, "log": log}, open(f"{a.out}/final.json", "w"), indent=1)
        print("FINAL", json.dumps(meta), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
