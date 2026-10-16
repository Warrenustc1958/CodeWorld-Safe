"""Keep a representative four-quadrant subset of one finished cell; delete the rest.

Selection per task (Rollout_Data_Implementation_detail.md):
  * Q2 / Q4 (unsafe): keep every valid episode -- the scarce signal.
  * Q1 / Q3: keep up to --cap episodes, chosen round-robin over distinct roots
    so root diversity is maximised (one episode per root per pass; the root
    episode itself is preferred over its branches).
  * Branch episodes store only the post-branch suffix, so every kept branch
    also keeps its root chain (JSON + NPZ) for prefix reconstruction.
Invalid episodes (missing / mismatched NPZ) are deleted unless needed as a root.

Usage:
  python cf_prune.py <collect_root> <suite> <level> [--cap 50] [--apply]
Without --apply only the plan is printed and the manifest written.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

SCHEMA = "libero_safety_episode_audit/v1"


def load(collect_root: Path, suite: str, level: str):
    eps = {}
    prefix = f"{suite}_{level}_"
    for run_dir in sorted((collect_root / "runs").iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        for path in run_dir.rglob("libero_safety_audits/process_*/*.json"):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                rec = None
            if not rec or rec.get("schema_version") != SCHEMA:
                eps[(path.parent, path.stem)] = {"json": path, "npz": None, "valid": False,
                                                 "task": None, "q": None, "root": None,
                                                 "kind": None}
                continue
            ep = rec.get("episode", {})
            vis = rec.get("visual_trajectory") or {}
            npz = path.parent / vis["path"] if vis.get("path") else None
            valid = bool(rec.get("trajectory") and npz and npz.is_file()
                         and int(vis.get("num_frames", -1)) == len(rec["trajectory"]) + 1)
            eid = ep.get("episode_id") or path.stem
            eps[(path.parent, eid)] = {
                "json": path, "npz": npz, "valid": valid,
                "task": int(ep.get("task_id", -1)),
                "q": rec.get("outcome", {}).get("quadrant", rec.get("quadrant")),
                "root": ep.get("root_episode_id") or rec.get("root_episode_id") or eid,
                "kind": ep.get("episode_kind"),
                "suite": ep.get("task_suite"), "level": ep.get("safety_level"),
            }
    return eps


def root_chain(eps, key):
    """Keys of all ancestors of `key` (excluding itself), nearest first."""
    chain, seen = [], {key}
    parent_dir, eid = key
    while True:
        rid = eps[(parent_dir, eid)]["root"]
        if rid is None or rid == eid:
            return chain
        nxt = (parent_dir, rid)
        if nxt in seen or nxt not in eps:
            return chain
        chain.append(nxt)
        seen.add(nxt)
        eid = rid


def top_root(eps, key):
    chain = root_chain(eps, key)
    return chain[-1] if chain else key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collect_root")
    ap.add_argument("suite")
    ap.add_argument("level")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(args.collect_root)
    eps = load(root, args.suite, args.level)

    keep = set()
    by_task_q = defaultdict(lambda: defaultdict(list))
    for key, e in eps.items():
        if e["valid"] and e["suite"] == args.suite and e["level"] == args.level \
                and e["q"] in ("Q1", "Q2", "Q3", "Q4"):
            by_task_q[e["task"]][e["q"]].append(key)

    for task, qs in by_task_q.items():
        for q in ("Q2", "Q4"):
            keep.update(qs.get(q, []))
        for q in ("Q1", "Q3"):
            groups = defaultdict(list)
            for key in sorted(qs.get(q, []), key=lambda k: (str(k[0]), k[1])):
                groups[top_root(eps, key)].append(key)
            for g in groups.values():
                # Root episode first, then branches in id order.
                g.sort(key=lambda k: (eps[k]["kind"] == "counterfactual_branch", str(k[0]), k[1]))
            order = sorted(groups, key=lambda r: (str(r[0]), r[1]))
            picked, depth = [], 0
            while len(picked) < args.cap and any(len(groups[r]) > depth for r in order):
                for r in order:
                    if len(groups[r]) > depth and len(picked) < args.cap:
                        picked.append(groups[r][depth])
                depth += 1
            keep.update(picked)

    # Closure: every kept branch keeps its root chain.
    for key in list(keep):
        keep.update(k for k in root_chain(eps, key) if k in eps)

    delete = [k for k in eps if k not in keep]
    files_del = []
    for k in delete:
        e = eps[k]
        files_del.append(e["json"])
        if e["npz"] is not None and e["npz"].is_file():
            files_del.append(e["npz"])
    bytes_del = sum(p.stat().st_size for p in files_del if p.is_file())

    # Report: kept per task/quadrant with distinct roots.
    stats = defaultdict(lambda: defaultdict(lambda: [0, set()]))
    before = defaultdict(lambda: defaultdict(int))
    for key, e in eps.items():
        if e["valid"] and e["q"] in ("Q1", "Q2", "Q3", "Q4") and e["suite"] == args.suite:
            before[e["task"]][e["q"]] += 1
            if key in keep:
                s = stats[e["task"]][e["q"]]
                s[0] += 1
                s[1].add(top_root(eps, key))
    cell = f"{args.suite}_{args.level}"
    print(f"== {cell} cap={args.cap} episodes={len(eps)} keep={len(keep)} delete={len(delete)}"
          f" files_del={len(files_del)} GB_del={bytes_del / 1e9:.1f}")
    print("task | Q1 kept/before(roots) | Q2 | Q3 | Q4")
    for t in sorted(before):
        cols = []
        for q in ("Q1", "Q2", "Q3", "Q4"):
            n, r = stats[t][q]
            cols.append(f"{n}/{before[t][q]}({len(r)})")
        print(f"task{t} | " + " | ".join(cols))

    ts = time.strftime("%Y%m%d_%H%M%S")
    mdir = root / "state" / f"prune_{cell}_{ts}"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "kept.txt").write_text("\n".join(sorted(str(eps[k]["json"]) for k in keep)) + "\n")
    (mdir / "deleted.txt").write_text("\n".join(sorted(str(p) for p in files_del)) + "\n")
    (mdir / "params.json").write_text(json.dumps({"cap": args.cap, "apply": args.apply}))
    print(f"manifest={mdir}")
    if args.apply:
        for p in files_del:
            p.unlink(missing_ok=True)
        print(f"DELETED {len(files_del)} files")


if __name__ == "__main__":
    main()
