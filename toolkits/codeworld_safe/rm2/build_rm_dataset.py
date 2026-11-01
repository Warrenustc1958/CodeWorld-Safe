"""Build the phase-2 RM dataset from phase-1 collection dumps (sync_real_env dump_dir with dump_arrays).

  python build_rm_dataset.py OUT_DIR --train colbase col10a ... --test col12r [--val_pct 15] [--seed 0]

Sample = one dual-view frame (agent over wrist, 512x256 JPEG) at the middle or the end of an executed chunk.
Label = the real goal predicate at that step: the RM judges task completion only; safety violations are a separate
field (for a cost head), never a negative label.

train (quota + balance, from the train collections on train uids):
  per episode: positives = the first goal frame + one random later goal frame;
  hard negatives (<= 4): goal within 4 steps (transient / boundary), old RM chunk max >= 0.8, the 16 steps before
  the first goal (success episodes), the last 2 chunks (failed episodes); easy negatives: 1 random.
  globally: negatives = 2x positives, hard negatives >= 70% of them when available; per-cell positive cap.
val  = all executed frames of train-collection episodes on val uids (for episode-level metrics and thresholds).
test = all executed frames of test-collection episodes on val uids: unseen policy and unseen source trajectories.
Split by source-trajectory uid hash. Writes {split}.jpg.bin, {split}.off.npy, {split}.jsonl and report.json.
"""
import argparse
import glob
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

S = "/data_cf/cws_sam/v1/scgrpo"
KF = f"{S}/keyframes_v1"
CH = 8


def is_val(uid, pct):
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100 < pct


def episodes(col):
    d = f"{S}/causal/{col}/dump"
    for f in sorted(glob.glob(f"{d}/episodes_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "arrays" in r:
                r["_npz"], r["_col"], r["_file"] = f"{d}/arrays/{r['arrays']}", col, os.path.basename(f)
                yield r


def frames_of(r):
    """[(step, chunk, jpg_index, label)] for the executed mid/end frames, plus the arrays."""
    z = np.load(r["_npz"])
    goal, ex, off, jpg = z["step_goal"], z["step_exec"], z["jpg_off"], z["jpg"]
    n_chunks = (len(off) - 1) // 2
    out = []
    for c in range(n_chunks):
        for j, k in enumerate((CH // 2 - 1, CH - 1)):
            s = c * CH + k
            if s < len(goal) and ex[s]:
                out.append((s, c, 2 * c + j, bool(goal[s])))
    viol = z["step_ch"]
    return out, goal, viol, off, jpg


def categorize(r, fr, goal, rng):
    first = int(np.argmax(goal)) if goal.any() else -1
    last_c = max((c for _, c, _, _ in fr), default=-1)
    pos, hard, easy = [], [], []
    for s, c, ji, lab in fr:
        if lab:
            pos.append((s, c, ji, "pos"))
            continue
        rm = r["rm_chunk_max"][c] if c < len(r["rm_chunk_max"]) else None
        if goal[max(0, s - 4):s + 5].any():
            cat = "hard_transient"
        elif rm is not None and rm >= 0.8:
            cat = "hard_rmfp"
        elif first >= 0 and first - 16 <= s < first:
            cat = "hard_pre_goal"
        elif first < 0 and c >= last_c - 1:
            cat = "hard_fail_tail"
        else:
            cat = "easy"
        (easy if cat == "easy" else hard).append((s, c, ji, cat))
    return pos, hard, easy


def sample_row(r, meta, s, c, cat, lab, viol, b):
    kf = r["kf"]
    return {"col": r["_col"], "file": r["_file"], "reset": r["reset"], "group": r["group"], "slot": r["slot"],
            "kf": kf, "uid": r["uid"], "suite": r["suite"], "tcell": str(meta["tcell"][kf]),
            "instruction": str(meta["instruction"][kf]), "step": int(s), "chunk": int(c), "category": cat,
            "label": int(lab), "violation": [int(x) for x in viol[s]], "ep_success": bool(r["real_success"]),
            "rm_old_chunk_max": r["rm_chunk_max"][c], "_jpg": b}


def write(out, split, rows):
    with open(f"{out}/{split}.jpg.bin", "wb") as fb, open(f"{out}/{split}.jsonl", "w") as fj:
        offs = [0]
        for x in rows:
            fb.write(x["_jpg"])
            offs.append(offs[-1] + len(x["_jpg"]))
            fj.write(json.dumps({k: v for k, v in x.items() if k != "_jpg"}) + "\n")
    np.save(f"{out}/{split}.off.npy", np.asarray(offs, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--val_pct", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    meta = np.load(f"{KF}/meta.npz")
    P, H, E, val, test = [], [], [], [], []
    ep_count, mismatch = Counter(), 0
    for col in a.train + a.test:
        is_test_col = col in a.test
        for r in episodes(col):
            v = is_val(r["uid"], a.val_pct)
            if is_test_col and not v:
                continue
            fr, goal, viol, off, jpg = frames_of(r)
            mismatch += bool(goal.any()) != bool(r["real_success"])
            get = lambda ji: jpg[off[ji]:off[ji + 1]].tobytes()
            if v:  # val / test: every executed frame, natural composition
                pos, hard, easy = categorize(r, fr, goal, rng)
                cats = {ji: cat for _, _, ji, cat in pos + hard + easy}
                dst = test if is_test_col else val
                for s, c, ji, lab in fr:
                    dst.append(sample_row(r, meta, s, c, cats[ji], lab, viol, get(ji)))
                ep_count["test" if is_test_col else "val"] += 1
                continue
            ep_count["train"] += 1
            pos, hard, easy = categorize(r, fr, goal, rng)
            sel = pos[:1] + ([rng.choice(pos[1:])] if len(pos) > 1 else [])
            rng.shuffle(hard)
            for s, c, ji, cat in sel:
                P.append(sample_row(r, meta, s, c, cat, True, viol, get(ji)))
            for s, c, ji, cat in hard[:4]:
                H.append(sample_row(r, meta, s, c, cat, False, viol, get(ji)))
            if easy:
                s, c, ji, cat = rng.choice(easy)
                E.append(sample_row(r, meta, s, c, cat, False, viol, get(ji)))
    # per-cell positive cap
    by_cell = defaultdict(list)
    for x in P:
        by_cell[x["tcell"]].append(x)
    med = float(np.median([len(v) for v in by_cell.values()])) if by_cell else 0
    cap = max(150, int(3 * med))
    P = [x for v in by_cell.values() for x in (rng.sample(v, cap) if len(v) > cap else v)]
    target = 2 * len(P)
    rng.shuffle(H)
    rng.shuffle(E)
    n_hard = min(len(H), int(0.7 * target))
    n_easy = min(len(E), target - n_hard)
    n_hard = min(len(H), target - n_easy)
    train = P + H[:n_hard] + E[:n_easy]
    rng.shuffle(train)
    for split, rows in (("train", train), ("val", val), ("test", test)):
        write(a.out, split, rows)
    # report and checks
    rep = {"episodes": dict(ep_count), "goal_vs_real_success_mismatch": mismatch, "per_cell_positive_cap": cap}
    uids = {sp: {x["uid"] for x in rows} for sp, rows in (("train", train), ("val", val), ("test", test))}
    rep["uid_overlap"] = {"train_val": len(uids["train"] & uids["val"]), "train_test": len(uids["train"] & uids["test"])}
    for split, rows in (("train", train), ("val", val), ("test", test)):
        rep[split] = {"samples": len(rows), "positive": sum(x["label"] for x in rows),
                      "by_category": dict(Counter(x["category"] for x in rows)),
                      "positive_by_suite": dict(Counter(x["suite"] for x in rows if x["label"])),
                      "negative_by_suite": dict(Counter(x["suite"] for x in rows if not x["label"])),
                      "with_violation": sum(any(x["violation"]) for x in rows)}
    json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
