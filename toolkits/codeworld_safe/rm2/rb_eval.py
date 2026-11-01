"""R-B / R-C offline evaluation of the Qwen RM on build_rm_dataset val and test splits.

  torchrun --nproc_per_node N rb_eval.py DS_DIR OUT_JSON [--lora ADAPTER_DIR] [--max_fp 0.01]
Each rank scores a shard; rank 0 calibrates the threshold on val (failed-episode fire rate <= max_fp) and scores both.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist

import qwen_rm
import rm_metrics


def shard_scores(model, proc, d, split, rank, world, bs):
    rows = [json.loads(l) for l in open(f"{d}/{split}.jsonl")]
    off = np.load(f"{d}/{split}.off.npy")
    buf = np.memmap(f"{d}/{split}.jpg.bin", np.uint8, mode="r") if rows else None
    idx = list(range(rank, len(rows), world))
    imgs = [qwen_rm.decode(buf[off[i]:off[i + 1]].tobytes()) for i in idx]
    p = qwen_rm.score_images(model, proc, imgs, [rows[i]["instruction"] for i in idx], bs)
    return rows, idx, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("out")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--max_fp", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    model, proc = qwen_rm.load(a.lora)
    res = {}
    for split in ("val", "test"):
        if not os.path.getsize(f"{a.ds}/{split}.jsonl"):
            continue
        rows, idx, p = shard_scores(model, proc, a.ds, split, rank, world, a.bs)
        gathered = [None] * world
        dist.all_gather_object(gathered, (idx, p.tolist()))
        probs = np.zeros(len(rows))
        for gi, gp in gathered:
            probs[gi] = gp
        res[split] = (rows, probs)
    if rank == 0:
        rows, probs = res["val"]
        thr = rm_metrics.calibrate(rows, probs, a.max_fp)
        out = {"lora": a.lora, "threshold": thr, "val": rm_metrics.score(rows, probs, thr),
               "val_at_0.5": rm_metrics.score(rows, probs, 0.5)}
        if "test" in res:
            out["test"] = rm_metrics.score(*res["test"], thr)
        json.dump(out, open(a.out, "w"), indent=1)
        print("FINAL", json.dumps(out), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
