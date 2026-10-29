"""E4 gate: score reward models on the held-out D2 split (d2test, 16 affordance origins never trained on)
under the training rule (per 8-frame chunk, p >= thr on k consecutive frames). Episode labels from the
real simulator over the fixed 64-step window. Reports precision / recall / wrong-part false positives.
  python gate_rm_d2.py name=path [name=path ...]"""
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/rm")
from data import RMFrames  # noqa: E402
from train_rm import to_input  # noqa: E402

from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel  # noqa: E402

ROOT = "/data_cf/cws_rm/frames_d2"
RULES = [(t, k) for t in (0.2, 0.5, 0.8, 0.9, 0.95) for k in (2, 3, 4)]


@torch.no_grad()
def score(path, ds):
    m = CWSRewardModel.from_checkpoint(path).cuda().eval().to(memory_format=torch.channels_last)
    out = []
    for x, _, t, _ in DataLoader(ds, batch_size=512, shuffle=False, num_workers=16, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(m(to_input(x, 1, False), t.cuda()).float()).cpu())
    return torch.cat(out).numpy()


def detect(p, thr, k):
    for c in range(len(p) // 8):
        run = 0
        for s in range(c * 8, c * 8 + 8):
            run = run + 1 if p[s] >= thr else 0
            if run >= k:
                return True
    return False


def main():
    ds = RMFrames(ROOT, "d2test", 1)
    meta = np.load(f"{ROOT}/d2test_meta.npz")
    eps = [json.loads(l) for l in open(f"{ROOT}/d2test_episodes.jsonl")]
    ep_of = meta["episode"]
    models = {"v3_stored": None}
    for spec in sys.argv[1:]:
        n, p = spec.split("=", 1)
        models[n] = p
    out = {}
    for name, path in models.items():
        p = meta["rm_v3_p"] if path is None else score(path, ds)
        res = {}
        for thr, k in RULES:
            tp = fp = fn = fpw = 0
            for e in eps:
                pe = p[ep_of == e["episode"]]
                d, y = detect(pe, thr, k), e["success"]
                tp += d and y
                fp += d and not y
                fn += (not d) and y
                fpw += d and not y and e["wrong_part"]
            res[f"{thr}x{k}"] = {"P": round(tp / max(tp + fp, 1), 3), "R": round(tp / max(tp + fn, 1), 3),
                                 "TP": tp, "FP": fp, "FP_wrong_part": fpw, "FN": fn}
        best = max(res.items(), key=lambda kv: (kv[1]["P"] >= 0.93 and kv[1]["R"] >= 0.75, kv[1]["P"] + kv[1]["R"]))
        out[name] = {"rules": res, "best": best[0]}
        print(f"== {name}  ({len(eps)} eps, {sum(e['success'] for e in eps)} real successes,"
              f" {sum(e['wrong_part'] and not e['success'] for e in eps)} wrong-part failures)")
        for r in ("0.2x2", "0.5x3", "0.8x4", "0.9x4", "0.95x4"):
            print(f"  {r:7s} {res[r]}")
        print(f"  best: {best[0]} {best[1]}  PASS={best[1]['P'] >= 0.93 and best[1]['R'] >= 0.75}", flush=True)
    json.dump(out, open("/data_cf/cws_rm/runs/gate_d2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
