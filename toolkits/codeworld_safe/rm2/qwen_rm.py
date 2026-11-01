"""Qwen3-VL success RM: p(task completed | dual-view frame, instruction) from the yes/no next-token logits.

Used by rb_eval.py (zero-shot, R-B), train_rm_rc.py (LoRA, R-C) and qwen_rm_server.py (GRPO). Runs in /data_cf/qwen/venv.
"""
import io

import numpy as np
import torch
from PIL import Image

MODEL = "/data_cf/qwen/models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots"
PROMPT = ("The image shows a robot workspace: a third-person camera view on top and the robot's wrist camera view "
          "below. Task: \"{inst}\". Has the task been completed in this image? Answer yes or no.")


def model_dir():
    import glob

    return sorted(glob.glob(f"{MODEL}/*"))[-1]


def load(lora=None, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    path = model_dir()
    proc = AutoProcessor.from_pretrained(path)
    proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
    model.to(device).eval()
    return model, proc


def answer_ids(tok):
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    return ids(["yes", "Yes"]), ids(["no", "No"])


def build(proc, images, instructions):
    texts = []
    for inst in instructions:
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(inst=inst)}]}]
        texts.append(proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    return proc(text=texts, images=list(images), padding=True, return_tensors="pt")


def yes_logit(model, batch, yes, no):
    """log p(yes) - log p(no) at the first answer token (left padding: the last position)."""
    out = model(**batch)
    last = out.logits[:, -1, :].float()
    return torch.logsumexp(last[:, yes], -1) - torch.logsumexp(last[:, no], -1)


def decode(jpg_bytes):
    return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")


@torch.no_grad()
def score_images(model, proc, images, instructions, bs=32):
    yes, no = answer_ids(proc.tokenizer)
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(images), bs):
        b = build(proc, images[i:i + bs], instructions[i:i + bs]).to(dev)
        out.append(torch.sigmoid(yes_logit(model, b, yes, no)).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)
