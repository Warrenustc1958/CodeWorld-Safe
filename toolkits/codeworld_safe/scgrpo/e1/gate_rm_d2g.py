"""R0 gate on frames_d2g/d2gtest (10 source groups in no RM training set). Each model is fed its own number of
views (agent only, or agent + wrist). Rule: per 8-frame chunk, p >= thr on k consecutive frames.
  dec_chunk: precision/recall at the FIRST RM decision (TP if the real goal is reached by the end of that chunk)
  final    : RM fires in the last chunk vs real goal at the last frame (success_reward_rule=final)
Also false positives that are wrong-part grasps.
  python gate_rm_d2g.py name=path [name=path ...]"""
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/rm")
from data import RMFrames  # noqa: E402
from train_rm import to_input  # noqa: E402

from rlinf.models.embodiment.reward.cws_reward_model import CWSRewardModel  # noqa: E402

ROOT, SPLIT = "/data_cf/cws_rm/frames_d2g", "d2gtest"
RULES = [(t, k) for t in (0.2, 0.5, 0.8, 0.9) for k in (2, 4)]


@torch.no_grad()
def score(path):
    m = CWSRewardModel.from_checkpoint(path).cuda().eval().to(memory_format=torch.channels_last)
    ds = RMFrames(ROOT, SPLIT, m.views)
    out = []
    for x, _, t, _ in DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(m(to_input(x, m.views, False), t.cuda()).float()).cpu())
    return torch.cat(out).numpy(), m.views


def first(p, thr, k, chunks):
    for c in chunks:
        run = 0
        for s in range(c * 8, c * 8 + 8):
            run = run + 1 if p[s] >= thr else 0
            if run >= k:
                return s
    return -1


def main():
    meta = np.load(f"{ROOT}/{SPLIT}_meta.npz")
    eps = [json.loads(l) for l in open(f"{ROOT}/{SPLIT}_episodes.jsonl")]
    ep_of, label = meta["episode"], meta["label"].astype(bool)
    models = {"v3_stored": None}
    for spec in sys.argv[1:]:
        n, p = spec.split("=", 1)
        models[n] = p
    report = {}
    print(f"{SPLIT}: {len(eps)} eps, {sum(e['success'] for e in eps)} real successes, "
          f"{sum(e['goal_last'] for e in eps)} goal at last frame, {sum(e['wrong_part'] for e in eps)} wrong-part")
    for name, path in models.items():
        p, views = (meta["rm_v3_p"], 1) if path is None else score(path)
        report[name] = {"views": views}
        for thr, k in RULES:
            c = {"dec": [0, 0, 0, 0], "final": [0, 0, 0, 0]}  # TP FP FN FP_wrong
            for e in eps:
                m = ep_of == e["episode"]
                pe, ye = p[m], label[m]
                nc = len(pe) // 8
                d = first(pe, thr, k, range(nc))
                if d >= 0:
                    ok = bool(ye[: (d // 8) * 8 + 8].any())
                    c["dec"][0 if ok else 1] += 1
                    c["dec"][3] += int(not ok and e["wrong_part"])
                elif ye.any():
                    c["dec"][2] += 1
                df, yl = first(pe, thr, k, [nc - 1]), bool(ye[-1])
                if df >= 0:
                    c["final"][0 if yl else 1] += 1
                    c["final"][3] += int(not yl and e["wrong_part"])
                elif yl:
                    c["final"][2] += 1
            report[name][f"{thr}x{k}"] = {
                key: {"P": round(tp / max(tp + fp, 1), 3), "R": round(tp / max(tp + fn, 1), 3),
                      "FP": fp, "FP_wrong": fw, "FN": fn}
                for key, (tp, fp, fn, fw) in c.items()}
        best = max(((r, v["final"]) for r, v in report[name].items() if r != "views"),
                   key=lambda kv: (kv[1]["R"] >= 0.75, kv[1]["P"]))
        report[name]["best_final"] = best[0]
        print(f"== {name} (views={views})")
        for r in ("0.5x2", "0.8x4", "0.9x4"):
            d, f = report[name][r]["dec"], report[name][r]["final"]
            print(f"  {r:6s} first-decision P {d['P']:.3f} R {d['R']:.3f} FP {d['FP']} (wrong {d['FP_wrong']}) | "
                  f"final P {f['P']:.3f} R {f['R']:.3f} FP {f['FP']} (wrong {f['FP_wrong']})")
        print(f"  best final rule with R>=.75: {best[0]} {best[1]}", flush=True)
    json.dump(report, open("/data_cf/cws_rm/runs/gate_d2g.json", "w"), indent=1)


if __name__ == "__main__":
    main()
