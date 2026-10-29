"""RM v6 data: D2 on-policy affordance rollouts (Wan frames, fixed 64-step window) as RMFrames splits.
Per-frame label = real simulator goal (checkgrippercontactpart) at that step, so wrong-part grasps are
negatives. Split by origin: origin position % 3 == 2 -> d2test (16 origins, gate only), else d2train.
  python build_frames_d2.py /data_cf/cws_sam/v1/scgrpo/diag/d2 /data_cf/cws_rm/frames_d2"""
import glob
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

D2, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S = Path("/data_cf/cws_sam/v1/scgrpo")
SUITES = ("affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human")
CKPT = {"base": 0, "r0b8_5": 1, "r0b8_10": 2, "r1_10": 3}
OUT.mkdir(parents=True, exist_ok=True)
if any(OUT.glob("*.bin")) and any(p.stat().st_size for p in OUT.glob("*.bin")):
    raise SystemExit(f"{OUT} already has frame data")
shutil.copy("/data_cf/cws_rm/frames_v3/tasks.json", OUT / "tasks.json")
tasks = json.loads((OUT / "tasks.json").read_text())
meta = np.load(S / "keyframes_v1/meta.npz")
origins = json.loads((D2 / "origins.json").read_text())["origins"]
split_of_origin = {o: ("d2test" if i % 3 == 2 else "d2train") for i, o in enumerate(origins)}


class Writer:
    KEYS = ("episode", "frame", "label", "dist_to_end", "task_id", "quadrant", "suite", "checkpoint", "rm_v3_p")

    def __init__(self, split):
        self.split = split
        self.bin = open(OUT / f"{split}_agent.bin", "wb")
        self.off = [0]
        self.meta = {k: [] for k in self.KEYS}
        self.eps = open(OUT / f"{split}_episodes.jsonl", "w")
        self.n_ep = 0

    def add(self, jpgs, labels, p, first_succ, info, extra):
        e, n = self.n_ep, len(jpgs)
        ref = first_succ if first_succ is not None else n - 1
        for f in range(n):
            self.bin.write(jpgs[f])  # already JPEG (quality 90) from the D2 collector
            self.off.append(self.off[-1] + len(jpgs[f]))
            for k, v in (("episode", e), ("frame", f), ("label", bool(labels[f])), ("dist_to_end", ref - f),
                         ("task_id", info["task_id"]), ("quadrant", info["quadrant"]), ("suite", info["suite"]),
                         ("checkpoint", info["ckpt_id"]), ("rm_v3_p", float(p[f]))):
                self.meta[k].append(v)
        self.eps.write(json.dumps({"episode": e, "success": first_succ is not None, "first_success_frame": first_succ,
                                   "num_frames": n, "num_sampled": n, "num_positive": int(np.sum(labels)),
                                   **{k: v for k, v in info.items() if k != "ckpt_id"}, **extra}) + "\n")
        self.n_ep += 1

    def close(self):
        self.bin.close()
        self.eps.close()
        np.save(OUT / f"{self.split}_agent.off.npy", np.asarray(self.off, np.int64))
        dt = {"episode": np.int32, "frame": np.int32, "label": bool, "dist_to_end": np.int32, "task_id": np.int16,
              "quadrant": np.int8, "suite": np.int8, "checkpoint": np.int8, "rm_v3_p": np.float32}
        np.savez(OUT / f"{self.split}_meta.npz", **{k: np.asarray(v, dt[k]) for k, v in self.meta.items()})


writers = {s: Writer(s) for s in ("d2train", "d2test")}
stats = {}
for pdir in sorted(p for p in D2.iterdir() if p.is_dir()):
    ck = pdir.name
    if ck not in CKPT:
        continue
    for f in sorted(pdir.glob("batch_*.pkl")):
        r = pickle.load(open(f, "rb"))
        for i in range(len(r["keyframes"])):
            if r["at_reset_success"][i]:
                continue
            k = int(r["keyframes"][i])
            tcell = str(meta["tcell"][k])
            goal = r["step_rec"][i, :, 0].astype(bool)
            wrong = bool(r["step_rec"][i, :, 1].any())
            fs = int(np.argmax(goal)) if goal.any() else None
            info = {"ckpt_id": CKPT[ck], "checkpoint": ck, "keyframe": k, "tcell": tcell,
                    "group": str(meta["group"][k]), "t0": int(meta["t0"][k]),
                    "task_id": tasks["instructions"].index(tasks["tcell_instruction"][tcell]),
                    "quadrant": int(["Q1", "Q2", "Q3", "Q4"].index(str(meta["quadrant"][k]))),
                    "suite": SUITES.index(str(meta["suite"][k]))}
            split = split_of_origin[k]
            writers[split].add(r["agent_jpg"][i], goal, r["rm_v3_p"][i], fs, info,
                               {"wrong_part": wrong, "origin": k})
            st = stats.setdefault(f"{split}/{ck}", {"eps": 0, "succ": 0, "wrong_part": 0, "pos_frames": 0})
            st["eps"] += 1
            st["succ"] += int(fs is not None)
            st["wrong_part"] += int(wrong)
            st["pos_frames"] += int(goal.sum())
for w in writers.values():
    w.close()
(OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
