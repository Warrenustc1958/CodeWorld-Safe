"""E4 gate v2 on held-out D2 (frames_d2/d2test), scored at the RM's FIRST decision, as RM-terminated
training uses it. Rule: per 8-frame chunk, p >= thr on k consecutive frames; decision frame = frame at
which the rule first completes (chunk c = frame // 8).
  dec_chunk : TP if the real goal was reached by the end of the decision chunk (the env steps the whole
              chunk before deciding), else FP
  dec_strict: TP only if the real goal was reached by the decision frame itself
  window    : old v1 definition (RM fires anywhere, real success anywhere in the 64-step window)
  final     : RM fires in the last chunk vs real goal at the last frame (success_reward_rule=final)
Subsets: all, and clean = source-trajectory groups not in d2train nor in frames_v5mix/poltrain.
  python gate_rm_d2_v2.py name=path [name=path ...]   (stored RM v3 scores are always included)"""
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
RULES = [(t, k) for t in (0.2, 0.5, 0.8, 0.9) for k in (2, 4)]


@torch.no_grad()
def score(path, ds):
    m = CWSRewardModel.from_checkpoint(path).cuda().eval().to(memory_format=torch.channels_last)
    out = []
    for x, _, t, _ in DataLoader(ds, batch_size=512, shuffle=False, num_workers=16, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(m(to_input(x, 1, False), t.cuda()).float()).cpu())
    return torch.cat(out).numpy()


def first_decision(p, thr, k, chunks=None):
    """frame at which the rule first completes, -1 if never (optionally only in the given chunks)"""
    for c in range(len(p) // 8) if chunks is None else chunks:
        run = 0
        for s in range(c * 8, c * 8 + 8):
            run = run + 1 if p[s] >= thr else 0
            if run >= k:
                return s
    return -1


def metrics(rows):
    out = {}
    for key in ("dec_chunk", "dec_strict", "window", "final"):
        tp = sum(r[key][0] for r in rows)
        fp = sum(r[key][1] for r in rows)
        fn = sum(r[key][2] for r in rows)
        fpw = sum(r[key][1] and r["wrong"] for r in rows)
        out[key] = {"P": round(tp / max(tp + fp, 1), 3), "R": round(tp / max(tp + fn, 1), 3),
                    "TP": tp, "FP": fp, "FP_wrong": fpw, "FN": fn}
    return out


def main():
    ds = RMFrames(ROOT, "d2test", 1)
    meta = np.load(f"{ROOT}/d2test_meta.npz")
    eps = [json.loads(l) for l in open(f"{ROOT}/d2test_episodes.jsonl")]
    train_groups = {json.loads(l)["group"] for l in open(f"{ROOT}/d2train_episodes.jsonl")}
    v5_groups = {json.loads(l).get("group") for l in open("/data_cf/cws_rm/frames_v5mix/poltrain_episodes.jsonl")}
    leaked = train_groups | v5_groups
    ep_of, label = meta["episode"], meta["label"].astype(bool)
    models = {"v3_stored": None}
    for spec in sys.argv[1:]:
        n, p = spec.split("=", 1)
        models[n] = p
    report = {"n_eps": len(eps), "n_clean": sum(e["group"] not in leaked for e in eps),
              "leaked_groups": sorted({e["group"] for e in eps} & leaked)}
    print(f"d2test {report['n_eps']} eps, clean {report['n_clean']}; leaked groups {report['leaked_groups']}")
    for name, path in models.items():
        p = meta["rm_v3_p"] if path is None else score(path, ds)
        report[name] = {}
        for thr, k in RULES:
            rows = {"all": [], "clean": []}
            for e in eps:
                m = ep_of == e["episode"]
                pe, ye = p[m], label[m]
                T = len(pe)
                real_any = bool(ye.any())
                d = first_decision(pe, thr, k)
                if d >= 0:
                    c_end = (d // 8) * 8 + 7
                    dc = bool(ye[: c_end + 1].any())
                    ds_ = bool(ye[: d + 1].any())
                    dec_chunk = (int(dc), int(not dc), 0)
                    dec_strict = (int(ds_), int(not ds_), 0)
                else:
                    dec_chunk = dec_strict = (0, 0, int(real_any))
                window = (int(d >= 0 and real_any), int(d >= 0 and not real_any), int(d < 0 and real_any))
                df = first_decision(pe, thr, k, chunks=[T // 8 - 1])
                yf = bool(ye[-1])
                final = (int(df >= 0 and yf), int(df >= 0 and not yf), int(df < 0 and yf))
                r = {"dec_chunk": dec_chunk, "dec_strict": dec_strict, "window": window, "final": final,
                     "wrong": bool(e["wrong_part"])}
                rows["all"].append(r)
                if e["group"] not in leaked:
                    rows["clean"].append(r)
            report[name][f"{thr}x{k}"] = {s: metrics(v) for s, v in rows.items()}
        print(f"== {name}")
        for rule in ("0.2x2", "0.5x2", "0.8x4", "0.9x4"):
            a, c = report[name][rule]["all"], report[name][rule]["clean"]
            print(f"  {rule:6s} all: dec_chunk P {a['dec_chunk']['P']:.3f} R {a['dec_chunk']['R']:.3f} | "
                  f"dec_strict P {a['dec_strict']['P']:.3f} | window P {a['window']['P']:.3f} R {a['window']['R']:.3f} | "
                  f"final P {a['final']['P']:.3f} R {a['final']['R']:.3f}   "
                  f"clean: dec_chunk P {c['dec_chunk']['P']:.3f} R {c['dec_chunk']['R']:.3f} final P {c['final']['P']:.3f}",
                  flush=True)
    json.dump(report, open("/data_cf/cws_rm/runs/gate_d2_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
