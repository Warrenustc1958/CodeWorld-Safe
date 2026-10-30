"""Build RMFrames (agent + wrist views) from one or more D2-style collector outputs (<dir>/<policy>/batch_*.pkl).
Splits by SOURCE TRAJECTORY GROUP with the same held-out groups as frames_d2g (its build_stats.json):
  <prefix>train  episodes of non-test groups
  <prefix>test   episodes of the frames_d2g test groups
  <prefix>wrong  the <prefix>train episodes that contain a wrong-part grasp (for oversampling)
Unreadable (partially written) batch files are skipped. Writes to OUT.tmp and renames when complete.
  python build_frames_mix.py OUT PREFIX D2DIR [D2DIR ...]"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

OUT, PREFIX, DIRS = Path(sys.argv[1]), sys.argv[2], [Path(d) for d in sys.argv[3:]]
S = Path("/data_cf/cws_sam/v1/scgrpo")
SUITES = ("affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human")
if OUT.exists():
    raise SystemExit(f"{OUT} exists")
TMP = OUT.with_name(OUT.name + ".tmp")
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)
shutil.copy("/data_cf/cws_rm/frames_v3/tasks.json", TMP / "tasks.json")
tasks = json.loads((TMP / "tasks.json").read_text())
meta = np.load(S / "keyframes_v1/meta.npz")
test_groups = set(json.loads(Path("/data_cf/cws_rm/frames_d2g/build_stats.json").read_text())["test_groups"])
policies = sorted({p.name for d in DIRS for p in d.iterdir() if p.is_dir()})
CKPT = {n: i for i, n in enumerate(policies)}


class Writer:
    KEYS = ("episode", "frame", "label", "dist_to_end", "task_id", "quadrant", "suite", "checkpoint", "rm_v3_p")

    def __init__(self, split):
        self.split = split
        self.bin = {v: open(TMP / f"{split}_{v}.bin", "wb") for v in ("agent", "wrist")}
        self.off = {v: [0] for v in ("agent", "wrist")}
        self.meta = {k: [] for k in self.KEYS}
        self.eps = open(TMP / f"{split}_episodes.jsonl", "w")
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
            np.save(TMP / f"{self.split}_{v}.off.npy", np.asarray(self.off[v], np.int64))
        self.eps.close()
        dt = {"episode": np.int32, "frame": np.int32, "label": bool, "dist_to_end": np.int32, "task_id": np.int16,
              "quadrant": np.int8, "suite": np.int8, "checkpoint": np.int8, "rm_v3_p": np.float32}
        np.savez(TMP / f"{self.split}_meta.npz", **{k: np.asarray(v, dt[k]) for k, v in self.meta.items()})


writers = {s: Writer(PREFIX + s) for s in ("train", "test", "wrong")}
stats = {s: {"eps": 0, "succ": 0, "wrong_part": 0} for s in writers}
stats["skipped_files"] = []
for d in DIRS:
    for pdir in sorted(p for p in d.iterdir() if p.is_dir()):
        for f in sorted(pdir.glob("batch_*.pkl")):
            try:
                r = pickle.load(open(f, "rb"))
            except Exception:
                stats["skipped_files"].append(str(f))
                continue
            for i in range(len(r["keyframes"])):
                if r["at_reset_success"][i]:
                    continue
                k = int(r["keyframes"][i])
                tcell, group = str(meta["tcell"][k]), str(meta["group"][k])
                goal = r["step_rec"][i, :, 0].astype(bool)
                wrong = bool(r["step_rec"][i, :, 1].any())
                fs = int(np.argmax(goal)) if goal.any() else None
                info = {"ckpt_id": CKPT[pdir.name], "checkpoint": pdir.name, "keyframe": k, "tcell": tcell,
                        "group": group, "t0": int(meta["t0"][k]),
                        "task_id": tasks["instructions"].index(tasks["tcell_instruction"][tcell]),
                        "quadrant": int(["Q1", "Q2", "Q3", "Q4"].index(str(meta["quadrant"][k]))),
                        "suite": SUITES.index(str(meta["suite"][k]))}
                extra = {"wrong_part": wrong, "origin": k, "goal_last": bool(goal[-1])}
                splits = ["test"] if group in test_groups else (["train", "wrong"] if wrong else ["train"])
                for s in splits:
                    writers[s].add(r["agent_jpg"][i], r["wrist_jpg"][i], goal, r["rm_v3_p"][i], fs, info, extra)
                    stats[s]["eps"] += 1
                    stats[s]["succ"] += int(fs is not None)
                    stats[s]["wrong_part"] += int(wrong)
for w in writers.values():
    w.close()
stats["policies"] = CKPT
(TMP / "build_stats.json").write_text(json.dumps(stats, indent=1))
TMP.rename(OUT)
print(json.dumps(stats, indent=1))
