"""Score CWSRewardModel checkpoints on a build_rm_dataset directory with the R-A / R-C protocol.

  python eval_old_rm.py DS_DIR OUT_JSON NAME=CKPT [NAME=CKPT ...] [--max_fp 0.01]
Threshold calibrated on val (failed-episode fire rate <= max_fp), then val and test scored with rm_metrics.
"""
import argparse
import json

import torch

from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel

import rm_metrics
from train_rm_ra import Split, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("out")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--max_fp", type=float, default=0.01)
    a = ap.parse_args()
    val, test = Split(a.ds, "val"), Split(a.ds, "test")
    res = {}
    for item in a.ckpts:
        name, path = item.split("=", 1)
        model = CWSRewardModel.from_checkpoint(path).cuda().eval()
        missing = {r["instruction"] for d in (val, test) for r in d.rows} - set(model.instructions)
        if missing:
            res[name] = {"error": f"{len(missing)} instructions not in vocabulary"}
            print(name, res[name], flush=True)
            continue
        pv, pt = predict(model, val, "cuda"), predict(model, test, "cuda")
        thr = rm_metrics.calibrate(val.rows, pv, a.max_fp)
        res[name] = {"views": model.views, "threshold": thr, "val": rm_metrics.score(val.rows, pv, thr),
                     "test": rm_metrics.score(test.rows, pt, thr),
                     "test_at_0.5": rm_metrics.score(test.rows, pt, 0.5)}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "val"}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
