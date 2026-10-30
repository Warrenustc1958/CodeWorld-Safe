"""R0/R1: D2 on-policy affordance rollouts as RMFrames with BOTH views (agent + wrist), split by SOURCE
TRAJECTORY GROUP. Test groups appear in no RM training set: groups also present in frames_v5mix/poltrain
always go to d2gtrain; of the remaining groups every 3rd (sorted) goes to d2gtest.
Per-frame label = real simulator goal at that step (wrong-part grasps are negatives).
  python build_frames_d2g.py /data_cf/cws_sam/v1/scgrpo/diag/d2 /data_cf/cws_rm/frames_d2g"""
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
if OUT.exists():
    raise SystemExit(f"{OUT} exists")
OUT.mkdir(parents=True)
shutil.copy("/data_cf/cws_rm/frames_v3/tasks.json", OUT / "tasks.json")
tasks = json.loads((OUT / "tasks.json").read_text())
meta = np.load(S / "keyframes_v1/meta.npz")
v5 = {json.loads(l).get("group") for l in open("/data_cf/cws_rm/frames_v5mix/poltrain_episodes.jsonl")}
groups = set()
for pdir in D2.iterdir():
    if pdir.is_dir() and pdir.name in CKPT:
        for f in pdir.glob("batch_*.pkl"):
            for k in pickle.load(open(f, "rb"))["keyframes"]:
                groups.add(str(meta["group"][int(k)]))
clean = sorted(groups - v5)
test_groups = set(clean[2::3])
split_of = {g: ("d2gtest" if g in test_groups else "d2gtrain") for g in groups}


class Writer:
    KEYS = ("episode", "frame", "label", "dist_to_end", "task_id", "quadrant", "suite", "checkpoint", "rm_v3_p")

    def __init__(self, split):
        self.split = split
        self.bin = {v: open(OUT / f"{split}_{v}.bin", "wb") for v in ("agent", "wrist")}
        self.off = {v: [0] for v in ("agent", "wrist")}
        self.meta = {k: [] for k in self.KEYS}
        self.eps = open(OUT / f"{split}_episodes.jsonl", "w")
        self.n_ep = 0

    def add(self, agent, wrist, labels, p, first_succ, info, extra):
        e, n = self.n_ep, len(agent)
        ref = first_succ if first_succ is not None else n - 1
        for f in range(n):
            for v, jp in (("agent", agent[f]), ("wrist", wrist[f])):
                self.bin[v].write(jp)
                self.off[v].append(self.off[v][-1] + len(jp))
            for k, v in (("episode", e), ("frame", f), ("label", bool(labels[f])), ("dist_to_end", ref - f),
                         ("task_id", info["task_id"]), ("quadrant", info["quadrant"]), ("suite", info["suite"]),
                         ("checkpoint", info["ckpt_id"]), ("rm_v3_p", float(p[f]))):
                self.meta[k].append(v)
        self.eps.write(json.dumps({"episode": e, "success": first_succ is not None, "first_success_frame": first_succ,
                                   "num_frames": n, "num_sampled": n, "num_positive": int(np.sum(labels)),
                                   **{k: v for k, v in info.items() if k != "ckpt_id"}, **extra}) + "\n")
        self.n_ep += 1

    def close(self):
        for v in ("agent", "wrist"):
            self.bin[v].close()
            np.save(OUT / f"{self.split}_{v}.off.npy", np.asarray(self.off[v], np.int64))
        self.eps.close()
        dt = {"episode": np.int32, "frame": np.int32, "label": bool, "dist_to_end": np.int32, "task_id": np.int16,
              "quadrant": np.int8, "suite": np.int8, "checkpoint": np.int8, "rm_v3_p": np.float32}
        np.savez(OUT / f"{self.split}_meta.npz", **{k: np.asarray(v, dt[k]) for k, v in self.meta.items()})


writers = {s: Writer(s) for s in ("d2gtrain", "d2gtest")}
stats = {}
for pdir in sorted(p for p in D2.iterdir() if p.is_dir() and p.name in CKPT):
    ck = pdir.name
    for f in sorted(pdir.glob("batch_*.pkl")):
        r = pickle.load(open(f, "rb"))
        for i in range(len(r["keyframes"])):
            if r["at_reset_success"][i]:
                continue
            k = int(r["keyframes"][i])
            tcell, group = str(meta["tcell"][k]), str(meta["group"][k])
            goal = r["step_rec"][i, :, 0].astype(bool)
            wrong = bool(r["step_rec"][i, :, 1].any())
            fs = int(np.argmax(goal)) if goal.any() else None
            info = {"ckpt_id": CKPT[ck], "checkpoint": ck, "keyframe": k, "tcell": tcell, "group": group,
                    "t0": int(meta["t0"][k]),
                    "task_id": tasks["instructions"].index(tasks["tcell_instruction"][tcell]),
                    "quadrant": int(["Q1", "Q2", "Q3", "Q4"].index(str(meta["quadrant"][k]))),
                    "suite": SUITES.index(str(meta["suite"][k]))}
            split = split_of[group]
            writers[split].add(r["agent_jpg"][i], r["wrist_jpg"][i], goal, r["rm_v3_p"][i], fs, info,
                               {"wrong_part": wrong, "origin": k, "goal_last": bool(goal[-1])})
            st = stats.setdefault(split, {"eps": 0, "succ": 0, "succ_last": 0, "wrong_part": 0, "groups": set()})
            st["eps"] += 1
            st["succ"] += int(fs is not None)
            st["succ_last"] += int(goal[-1])
            st["wrong_part"] += int(wrong)
            st["groups"].add(group)
for w in writers.values():
    w.close()
for st in stats.values():
    st["groups"] = len(st["groups"])
stats["n_groups"], stats["v5_overlap_groups"], stats["test_groups"] = len(groups), len(groups & v5), sorted(test_groups)
(OUT / "build_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
