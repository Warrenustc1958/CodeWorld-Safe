"""RM online accuracy per training step from TensorBoard (episode means of the synchronised real simulator).

  python rm_online.py RUN [RUN ...]   (dirs under /data_cf/cws_sam/v1/scgrpo/causal) -> JSON on stdout
precision = (rm_once - rm_false) / rm_once ; recall = (rm_once - rm_false) / real_success.
Restarted runs have several event files: per step the value with the latest wall time wins.
"""
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/data_cf/cws_sam/v1/scgrpo/causal")
TAGS = ("env/rm_success_once", "env/rm_false_success", "env/real_success", "env/success_once")


def series(run):
    best = {t: {} for t in TAGS}
    for f in sorted((ROOT / run).rglob("events.out.tfevents*")):
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        have = set(acc.Tags()["scalars"])
        for t in TAGS:
            if t not in have:
                continue
            for ev in acc.Scalars(t):
                old = best[t].get(ev.step)
                if old is None or ev.wall_time > old[0]:
                    best[t][ev.step] = (ev.wall_time, ev.value)
    steps = sorted(set.intersection(*(set(best[t]) for t in TAGS[:3]))) if all(best[t] for t in TAGS[:3]) else []
    rows = []
    for s in steps:
        rm, fp, real = (best[t][s][1] for t in TAGS[:3])
        paid = best["env/success_once"].get(s, (0, float("nan")))[1]
        tp = rm - fp
        rows.append({"step": s, "real_success": round(real, 4), "rm_success": round(rm, 4),
                     "rm_false": round(fp, 4), "paid_success": round(paid, 4),
                     "precision": round(tp / rm, 4) if rm > 0 else None,
                     "recall": round(tp / real, 4) if real > 0 else None})
    return rows


if __name__ == "__main__":
    json.dump({r: series(r) for r in sys.argv[1:]}, sys.stdout, indent=1)
