# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cache reward-model frames from the Wan CWS indexes as JPEG shards.

Only dual rows are used, so the agent-only and dual-view runs see the same
episodes. Per episode the frames are every ``stride``-th frame plus the last
``tail`` frames before the episode end (success or horizon), so near-success
negatives are dense. Frame f is labelled 1 iff f >= first_success_frame. Episodes
stop at success, so a successful episode has exactly one positive frame (per-step
success labels, as in SafeDojo App. J).

Output per split: {split}_{agent,wrist}.bin (JPEG bytes), {split}_{agent,wrist}.off.npy
(int64 offsets, N+1), {split}_meta.npz, {split}_episodes.jsonl; plus tasks.json
(instruction vocabulary shared by all splits and later world-model frame sets).
"""

from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

INDEX_ROOT = Path("/data_cf/wan_ac_sft/index")
SPLITS = {
    "train": ("cws_v1", "train"),
    "val": ("cws_v1", "val"),
    "test": ("cws_v1", "test"),
    "newval": ("cws_test_new", "val"),
    "newtest": ("cws_test_new", "test"),
}
VIEW_KEYS = {
    "agent": ("agent_segments", "agentview_image"),
    "wrist": ("wrist_segments", "robot0_eye_in_hand_image"),
}
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
SUITES = ("affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human")


def last_frame(row: dict) -> int:
    end = row["num_frames"] - 1
    fs = row["first_success_frame"]
    return end if fs is None else min(fs, end)


def pick_frames(row: dict, stride: int, tail: int) -> list[int]:
    end = last_frame(row)
    ids = set(range(0, end + 1, stride)) | set(range(max(0, end - tail + 1), end + 1))
    return sorted(ids)


def gather(segments: list, ids: list[int], key: str) -> list[np.ndarray]:
    """Frames ``ids`` of a stitched ``[npz, start, stop]`` sequence, one load per segment."""
    bounds = np.cumsum([0] + [b - a for _, a, b in segments])
    out = {}
    for s, (path, start, _) in enumerate(segments):
        local = [f for f in ids if bounds[s] <= f < bounds[s + 1]]
        if not local:
            continue
        with np.load(path) as z:
            arr = z[key]
        for f in local:
            out[f] = arr[start + f - int(bounds[s])]
    return [out[f] for f in ids]


def jpeg(frame: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def work(job):
    row, stride, tail, quality = job
    ids = pick_frames(row, stride, tail)
    blobs = {
        view: [jpeg(f, quality) for f in gather(row[seg_key], ids, npz_key)]
        for view, (seg_key, npz_key) in VIEW_KEYS.items()
    }
    return ids, blobs


def build_tasks(manifests: dict) -> dict:
    tcell_inst = {}
    for rows in manifests.values():
        for r in rows:
            if r["tcell"] not in tcell_inst:
                with open(r["json"]) as f:
                    tcell_inst[r["tcell"]] = json.load(f)["episode"]["instruction"].strip()
    return {"instructions": sorted(set(tcell_inst.values())), "tcell_instruction": tcell_inst}


def build_split(split, rows, out, tasks, args):
    inst_id = {s: i for i, s in enumerate(tasks["instructions"])}
    stride = args.stride_train if split == "train" else args.stride_eval
    files = {v: open(out / f"{split}_{v}.bin", "wb") for v in VIEW_KEYS}
    offsets = {v: [0] for v in VIEW_KEYS}
    meta = {k: [] for k in ("episode", "frame", "label", "dist_to_end", "task_id", "quadrant", "suite")}
    t0 = time.time()
    jobs = [(r, stride, args.tail, args.quality) for r in rows]
    with ProcessPoolExecutor(args.workers) as pool, open(out / f"{split}_episodes.jsonl", "w") as eps:
        for e, (row, (ids, blobs)) in enumerate(zip(rows, pool.map(work, jobs, chunksize=2))):
            for v in VIEW_KEYS:
                for b in blobs[v]:
                    files[v].write(b)
                    offsets[v].append(offsets[v][-1] + len(b))
            fs, end = row["first_success_frame"], last_frame(row)
            task = inst_id[tasks["tcell_instruction"][row["tcell"]]]
            for f in ids:
                meta["episode"].append(e)
                meta["frame"].append(f)
                meta["label"].append(fs is not None and f >= fs)
                meta["dist_to_end"].append(end - f)
                meta["task_id"].append(task)
                meta["quadrant"].append(QUADRANTS.index(row["quadrant"]))
                meta["suite"].append(SUITES.index(row["suite"]))
            eps.write(json.dumps({
                "episode": e, "idx": row["idx"], "tcell": row["tcell"], "task_id": task,
                "quadrant": row["quadrant"], "suite": row["suite"], "level": row["level"],
                "success": fs is not None and fs <= row["num_frames"] - 1,
                "num_frames": row["num_frames"], "num_sampled": len(ids),
            }) + "\n")
            if (e + 1) % 500 == 0:
                print(f"[{split}] {e + 1}/{len(rows)} eps, {len(meta['label'])} frames, "
                      f"{time.time() - t0:.0f}s", flush=True)
    for v in VIEW_KEYS:
        files[v].close()
        np.save(out / f"{split}_{v}.off.npy", np.asarray(offsets[v], np.int64))
    dtypes = {"episode": np.int32, "frame": np.int32, "label": bool, "dist_to_end": np.int32,
              "task_id": np.int16, "quadrant": np.int8, "suite": np.int8}
    np.savez(out / f"{split}_meta.npz", **{k: np.asarray(v, dtypes[k]) for k, v in meta.items()})
    labels = np.asarray(meta["label"])
    size = sum(offsets[v][-1] for v in VIEW_KEYS) / 2**30
    print(f"[{split}] done: {len(rows)} eps, {len(labels)} frames, {int(labels.sum())} pos, "
          f"{size:.1f} GB, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--stride_train", type=int, default=8)
    ap.add_argument("--stride_eval", type=int, default=4)
    ap.add_argument("--tail", type=int, default=16)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N episodes per split (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifests = {
        name: [json.loads(l) for l in (INDEX_ROOT / name / "manifest.jsonl").read_text().splitlines()]
        for name in {n for n, _ in SPLITS.values()}
    }
    tasks_path = out / "tasks.json"
    if tasks_path.exists():
        tasks = json.loads(tasks_path.read_text())
    else:
        tasks = build_tasks(manifests)
        tasks_path.write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks['instructions'])} instructions over {len(tasks['tcell_instruction'])} task cells")
    (out / "build_args.json").write_text(json.dumps(vars(args), indent=1))

    for split in args.splits:
        name, sp = SPLITS[split]
        rows = [r for r in manifests[name] if r["split"] == sp and r["dual"]]
        if args.limit:
            rows = rows[: args.limit]
        build_split(split, rows, out, tasks, args)


if __name__ == "__main__":
    main()
