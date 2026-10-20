#!/usr/bin/env python3
"""Back-test validator gate 4 on existing audits (e.g. the pi0.5 5068-episode v2 set).

  python -m toolkits.codeworld_safe.agent.backtest_gate4 --kept kept.txt --out DIR [--workers 24]

Episodes are grouped by root so each worker validates a root with its branches
and checks the seams locally. Legacy pi0.5 data used chunk 1 and no sim-state
sidecars, so chunk alignment and the sidecar are not required here.
Writes DIR/verdicts.jsonl (one row per audit, all reason codes) and DIR/summary.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from toolkits.codeworld_safe.agent.validator import (
    LEGACY_SKIP_FRAMES,
    MIN_FRAMES,
    gate4_episode,
    load_episodes,
)
from toolkits.libero_safety.build_dataset import _audit_collection_root, _collection_id


def _head(path: str) -> tuple[str, str, str, str | None]:
    """(path, collection, episode_id, root_id) from one audit JSON."""
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    ep = rec.get("episode", {})
    root = ep.get("root_episode_id") or rec.get("root_episode_id")
    return path, _collection_id(_audit_collection_root(Path(path))), str(ep.get("episode_id")), root


def _check_group(paths: list[str]) -> list[dict]:
    lookup, failures = load_episodes([Path(p) for p in paths])
    rows = [{**f, "codes": [f["code"]], "ok": False} for f in failures]
    for (collection, source_id), ep in sorted(lookup.items()):
        meta = ep.record["episode"]
        try:
            v = gate4_episode(ep, lookup, chunk=None, require_sim_state=False)
        except (KeyError, TypeError, ValueError, OSError) as error:
            rows.append({"path": str(ep.audit_path), "ok": False, "codes": ["CHECK_ERROR"],
                         "reasons": [{"code": "CHECK_ERROR", "detail": str(error)[:300]}]})
            continue
        rows.append({
            "path": str(ep.audit_path),
            "episode_id": source_id,
            "cell": f"{meta.get('task_suite')}_{meta.get('safety_level')}_t{meta.get('task_id')}",
            "kind": meta.get("episode_kind", "root"),
            "quadrant": ep.quadrant,
            "branch_timestep": meta.get("branch_timestep"),
            "ok": v.ok,
            "codes": v.codes,
            "reasons": v.reasons,
            **v.info,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    paths = [p for p in args.kept.read_text().split() if p]
    args.out.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.workers) as ex:
        heads = list(ex.map(_head, paths, chunksize=32))
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path, collection, eid, root in heads:
        groups[(collection, root or eid)].append(path)

    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        for result in ex.map(_check_group, list(groups.values()), chunksize=4):
            rows.extend(result)
    with (args.out / "verdicts.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    code_counts = Counter(c for r in rows for c in r["codes"])
    rejected = [r for r in rows if not r["ok"]]
    frames = [r.get("stitched_frames") for r in rows if r.get("stitched_frames") is not None]
    by_code_kind = Counter((c, r.get("kind")) for r in rejected for c in r["codes"])
    by_code_quadrant = Counter((c, r.get("quadrant")) for r in rejected for c in r["codes"])
    by_code_cell = defaultdict(Counter)
    for r in rejected:
        for c in r["codes"]:
            by_code_cell[c][r.get("cell")] += 1
    summary = {
        "audits": len(paths),
        "rows": len(rows),
        "accepted": len(rows) - len(rejected),
        "rejected": len(rejected),
        "reject_rate": len(rejected) / max(len(rows), 1),
        "groups": len(groups),
        "codes": dict(code_counts.most_common()),
        "codes_by_kind": {f"{c}|{k}": n for (c, k), n in sorted(by_code_kind.items())},
        "codes_by_quadrant": {f"{c}|{q}": n for (c, q), n in sorted(by_code_quadrant.items())},
        "codes_by_cell": {c: dict(v.most_common(8)) for c, v in by_code_cell.items()},
        f"stitched_frames_lt_{LEGACY_SKIP_FRAMES}": sum(f < LEGACY_SKIP_FRAMES for f in frames),
        f"stitched_frames_lt_{MIN_FRAMES}": sum(f < MIN_FRAMES for f in frames),
        "multi_code_rows": sum(len(r["codes"]) > 1 for r in rejected),
        "examples": {c: [r["path"] for r in rejected if c in r["codes"]][:3] for c in code_counts},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("audits", "accepted", "rejected", "reject_rate", "codes")}, indent=1))


if __name__ == "__main__":
    main()
