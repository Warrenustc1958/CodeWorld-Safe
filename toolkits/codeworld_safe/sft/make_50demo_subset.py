"""Build the 50-demos-per-task-instance subset of the LIBERO-Safety demonstrations.

The LeRobot release groups episodes by instruction (15 instructions), while the
benchmark has 40 L0/L1 (suite, level, task) instances; one instruction appears
in 2 or 4 instances (verified against the BDDL files). To get ~50 demos per
instance we sample 100 episodes for 2-instance instructions and 200 for
4-instance ones (2000 total), with a fixed seed.

The subset keeps the full-set meta/stats.json, so the 50-demo and full SFT runs
share one action normalization. data/ and videos/ are symlinks to the full set.

Usage: python make_50demo_subset.py <full_dataset_dir> <out_dir> [--per-instance 50] [--seed 0]
"""
import argparse
import collections
import json
import os
import random
import shutil

# Instructions that occur in obstacle_avoidance and obstacle_avoidance_human at
# L0 and L1 (4 instances). Every other instruction has 2 instances.
FOUR_INSTANCE = {
    "put both the alphabet soup and the cream cheese box in the basket",
    "pick up the akita black bowl on the stove and place it on the plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put both moka pots on the stove",
    "put the white yellow mug in the microwave and close it",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--per-instance", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.src, "meta", "episodes.jsonl"))]
    by_instruction = collections.defaultdict(list)
    for r in rows:
        by_instruction[r["tasks"][0].strip().lower()].append(r)

    rng = random.Random(args.seed)
    keep = []
    for instruction in sorted(by_instruction):
        n_inst = 4 if instruction in FOUR_INSTANCE else 2
        keep += rng.sample(by_instruction[instruction], args.per_instance * n_inst)
    keep.sort(key=lambda r: r["episode_index"])

    os.makedirs(os.path.join(args.dst, "meta"), exist_ok=True)
    for f in os.listdir(os.path.join(args.src, "meta")):
        if f.endswith(".json") or f == "tasks.jsonl":
            shutil.copy2(os.path.join(args.src, "meta", f), os.path.join(args.dst, "meta", f))
    with open(os.path.join(args.dst, "meta", "episodes.jsonl"), "w") as w:
        for r in keep:
            w.write(json.dumps(r) + "\n")
    for d in ("data", "videos"):
        link = os.path.join(args.dst, d)
        if not os.path.lexists(link):
            os.symlink(os.path.join(args.src, d), link)

    counts = collections.Counter(r["tasks"][0].strip().lower() for r in keep)
    print(f"episodes={len(keep)} frames={sum(r['length'] for r in keep)} "
          f"per-instruction={sorted(set(counts.values()))}")


if __name__ == "__main__":
    main()
