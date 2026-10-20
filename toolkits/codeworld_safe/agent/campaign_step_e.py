#!/usr/bin/env python3
"""Step E campaign: pi0.5 counterfactual collection over every L0/L1 cell.

One round = one run per active cell (pi0.5, chunk 8, 512 steps, agentview + wrist,
GPUs 4-5). Each run is ingested at once: gate 4, wrist present, stitched length
<= 512, dedup (< 1 cm to any admitted episode of the cell) -> admitted or not.
No post-hoc refinement.

Quota per cell: Q1>=50, Q2>=20, Q3>=50, Q4>=20, near-miss>=20 (safe episode whose
grader min_dist < 2 cm). A cell stops when all are met; it is "infeasible" after
3 consecutive runs without growth of any unmet quota; "needs_program" when only
Q2/Q4 are missing and the archive has no validated program for it.
Finished cells are uploaded (upload_cell.py) when MS_TOKEN_FILE exists, else
marked pending_upload. Nothing is deleted here.

  python -m toolkits.codeworld_safe.agent.campaign_step_e --root /data_cf/cws/step_e [--rounds 20]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from toolkits.codeworld_safe.agent.archive import Archive, trajectory_feature
from toolkits.codeworld_safe.agent.validator import gate4_episode, load_episodes

REPO = Path(__file__).resolve().parents[3]
KNOW = REPO / "toolkits/codeworld_safe/agent/knowledge/tasks"
TRIAL = REPO / "toolkits/codeworld_safe/cf/trial_step_e.sh"
UPLOAD = REPO / "toolkits/codeworld_safe/agent/upload_cell.py"
QUOTA = {"Q1": 50, "Q2": 20, "Q3": 50, "Q4": 20, "NM": 20}
UNSAFE = ("Q2", "Q4")
NEAR_MISS_M = 0.02
STAGNANT_LIMIT = 3
WRIST = "robot0_eye_in_hand_image"
PRIOR = 0.3


def all_cells() -> list[str]:
    return sorted(p.stem for p in KNOW.glob("*.json"))


def split_cell(cell: str) -> tuple[str, str, int]:
    suite, level, t = cell.rsplit("_", 2)
    return suite, level, int(t[1:])


def counts(archive: Archive, cell: str) -> dict[str, int]:
    rows = [o for o in archive.outcomes(cell=cell) if o["admitted"]]
    c = Counter(o["quadrant"] for o in rows)
    out = {q: c.get(q, 0) for q in ("Q1", "Q2", "Q3", "Q4")}
    out["NM"] = sum(bool(o.get("near_miss")) for o in rows)
    return out


def gaps(c: dict[str, int]) -> dict[str, int]:
    return {q: max(0, QUOTA[q] - c.get(q, 0)) for q in QUOTA}


def _has_wrist(visual_path: Path) -> bool:
    try:
        with zipfile.ZipFile(visual_path) as z:
            return any(n.startswith(WRIST) for n in z.namelist())
    except Exception:  # noqa: BLE001
        return False


def ingest_run(out: Path, archive: Archive, cell: str, run_id: str) -> dict:
    lookup, failures = load_episodes(sorted(out.rglob("process-*.json")))
    stats = Counter()
    for ep in lookup.values():
        rec, meta = ep.record, ep.record["episode"]
        if f"{meta['task_suite']}_{meta['safety_level']}_t{meta['task_id']}" != cell:
            continue
        traj = rec["trajectory"]
        kind = "branch" if rec.get("root_episode_id") else "root"
        pid = meta.get("program_id") or f"__root__:{cell}"
        try:
            v = gate4_episode(ep, lookup, chunk=8, require_sim_state=True)
            ok, codes = bool(v.ok), list(v.codes)
        except Exception as e:  # noqa: BLE001
            ok, codes = False, [f"GATE4_ERROR:{type(e).__name__}"]
        bt = meta.get("branch_timestep")
        stitched = (int(bt) - 1 + len(traj)) if (kind == "branch" and bt) else len(traj)
        if stitched > 512:
            ok, codes = False, codes + ["OVER_512"]
        if not _has_wrist(Path(ep.visual_path)):
            ok, codes = False, codes + ["NO_WRIST"]
        # c_grasp is the gripper-to-wrong-part distance, ~0 during any normal grasp,
        # so it says nothing about hazard proximity: only c_obj / c_robot count.
        dists = [g["min_dist"] for s in traj for g in (s.get("grades") or [])
                 if g.get("channel") in ("c_obj", "c_robot")]
        md = float(min(dists)) if dists else None
        nm = ep.quadrant in ("Q1", "Q3") and md is not None and md < NEAR_MISS_M
        row = archive.add_outcome(
            program_id=pid, cell=cell, quadrant=ep.quadrant, feature=trajectory_feature(rec),
            gate4_ok=ok, gate4_codes=codes, audit_path=str(ep.audit_path),
            extra={"kind": kind, "run": run_id, "near_miss": bool(nm), "min_hazard_dist": md,
                   "steps": len(traj), "stitched_len": stitched, "branch_timestep": bt,
                   "visual_path": str(ep.visual_path),
                   "trigger_hit": bool((meta.get("program_branch") or {}).get("trigger_hit"))},
        )
        stats["episodes"] += 1
        stats["steps"] += len(traj)
        stats[f"{kind}_{ep.quadrant}"] += 1
        stats["admitted"] += row["admitted"]
        stats["duplicates"] += row["duplicate"]
        stats["rejected"] += not ok
        stats["unsafe_admitted"] += row["admitted"] and ep.quadrant in UNSAFE
    stats["schema_failures"] = len(failures)
    return dict(stats)


def pick_program(archive: Archive, prior: Archive | None, cell: str, gap: dict[str, int]) -> str | None:
    if not any(gap[q] for q in UNSAFE):
        return None
    best, best_score = None, 0.0
    for rec in archive.programs(cell, status="validated"):
        s = archive.stats(rec.program_id)
        if not s["trials"] and prior is not None and rec.program_id in prior._programs:
            s = prior.stats(rec.program_id)
        if s["trials"]:
            q = s["quadrants"]
            rate = {u: q.get(u, 0) / s["trials"] for u in UNSAFE}
            keep = (s["admitted"] / s["trials"]) if s["trials"] else 1.0
        else:
            rate = {u: (PRIOR if rec.target_quadrant == u else 0.05) for u in UNSAFE}
            keep = 1.0
        score = sum(rate[u] for u in UNSAFE if gap[u]) * max(keep, 0.1)
        if score > best_score:
            best, best_score = rec.program_id, score
    return best


def infeasible_reason(archive: Archive, cell: str, c: dict[str, int]) -> str:
    rows = archive.outcomes(cell=cell)
    roots = [o for o in rows if o.get("kind") == "root"]
    br = [o for o in rows if o.get("kind") == "branch"]
    succ = sum(o["quadrant"] in ("Q1", "Q2") for o in roots) / len(roots) if roots else 0.0
    md = [o["min_hazard_dist"] for o in roots if o.get("min_hazard_dist") is not None]
    trig = sum(bool(o.get("trigger_hit")) for o in br) / len(br) if br else 0.0
    dup = sum(o["duplicate"] for o in rows) / len(rows) if rows else 0.0
    progs = len(archive.programs(cell, status="validated"))
    unmet = {k: v for k, v in gaps(c).items() if v}
    return (f"unmet {unmet}; root success {succ:.2f} over {len(roots)} roots; "
            f"median root min hazard dist {np.median(md) if md else float('nan'):.3f} m; "
            f"{len(br)} branches, trigger hit {trig:.2f}; duplicate share {dup:.2f}; {progs} validated programs")


def write_manifest(archive: Archive, cell: str, cell_dir: Path) -> Path:
    path = cell_dir / "manifest.jsonl"
    with path.open("w") as f:
        for o in archive.outcomes(cell=cell):
            if not o["admitted"]:
                continue
            f.write(json.dumps({k: o.get(k) for k in (
                "quadrant", "kind", "near_miss", "program_id", "run", "audit_path", "visual_path",
                "min_hazard_dist", "stitched_len", "branch_timestep")}) + "\n")
    return path


def report(root: Path, archive: Archive, state: dict) -> dict:
    rows, steps, unsafe = {}, 0, 0
    for cell in all_cells():
        c = counts(archive, cell)
        steps_c = sum(int(o.get("steps", 0)) for o in archive.outcomes(cell=cell))
        steps += steps_c
        unsafe += c["Q2"] + c["Q4"]
        rate = float(np.mean([min(1.0, c[q] / QUOTA[q]) for q in QUOTA]))
        rows[cell] = {**c, "quota_rate": round(rate, 3), "steps": steps_c,
                      "status": state.get(cell, {}).get("status", "active"),
                      "reason": state.get(cell, {}).get("reason", "")}
    rep = {"updated": time.strftime("%F %T"), "cells_done": sum(r["status"] in ("done", "uploaded") for r in rows.values()),
           "mean_quota_rate": round(float(np.mean([r["quota_rate"] for r in rows.values()])), 3),
           "total_sim_steps": steps, "unsafe_admitted": unsafe,
           "unsafe_per_10k_steps": round(1e4 * unsafe / steps, 2) if steps else None, "cells": rows}
    (root / "report.json").write_text(json.dumps(rep, indent=1))
    gp = {cell: {q: g for q, g in gaps(counts(archive, cell)).items() if q in ("Q1", "Q2", "Q3", "Q4")}
          for cell in all_cells() if state.get(cell, {}).get("status") in ("needs_program", "active")}
    (root / "gaps.json").write_text(json.dumps(gp, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data_cf/cws/step_e"))
    ap.add_argument("--prior-archive", type=Path, default=Path("/data_cf/cws/archive/cws_archive_d1"))
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--gpus", default="5", help="env (EGL rendering) GPUs; gpu006 GPU 4 faults in EGL")
    ap.add_argument("--rollout-gpus", default="4", help="policy GPUs ('' = same as --gpus)")
    ap.add_argument("--envs", type=int, default=50)
    ap.add_argument("--max-branches", type=int, default=3)
    ap.add_argument("--cells", default="")
    ap.add_argument("--timeout-s", type=int, default=3 * 3600)
    a = ap.parse_args()
    root = a.root
    (root / "runs").mkdir(parents=True, exist_ok=True)
    adir = root / "archive"
    if not (adir / "programs.jsonl").exists():
        adir.mkdir(parents=True, exist_ok=True)
        src = a.prior_archive / "programs.jsonl"
        (adir / "programs.jsonl").write_text(src.read_text() if src.exists() else "")
        if (a.prior_archive / "programs").is_dir():
            subprocess.run(["cp", "-a", str(a.prior_archive / "programs"), str(adir / "programs")], check=False)
    prior = Archive(a.prior_archive) if a.prior_archive.exists() else None
    spath = root / "state.json"
    state = json.loads(spath.read_text()) if spath.exists() else {}
    cells = a.cells.split(",") if a.cells else all_cells()
    log = open(root / "campaign.log", "a", buffering=1)

    def say(msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    start = max((s.get("rounds", 0) for s in state.values()), default=0)
    for rnd in range(start, a.rounds):
        active = 0
        for idx, cell in enumerate(cells):
            st = state.setdefault(cell, {"rounds": 0, "stagnant": 0, "status": "active", "reason": "", "runs": []})
            if st["status"] in ("done", "uploaded", "pending_upload", "infeasible"):
                continue
            archive = Archive(adir)  # reload: programs may be added by run_gen_round meanwhile
            c = counts(archive, cell)
            gap = gaps(c)
            pid = pick_program(archive, prior, cell, gap)
            if pid is None and not any(gap[q] for q in ("Q1", "Q3", "NM")):
                st.update(status="needs_program", reason=f"only {[q for q in UNSAFE if gap[q]]} missing, no validated program")
                continue
            st["status"] = "active"
            active += 1
            suite, level, task = split_cell(cell)
            run_id = f"r{rnd:02d}_{pid or 'roots'}_{time.strftime('%m%d_%H%M')}"
            out = root / "runs" / cell / run_id
            env = dict(os.environ, SUITE=suite, LEVEL=level, TASK=str(task), OUT=str(out), GPUS=a.gpus,
                       ROLLOUT_GPUS=a.rollout_gpus,
                       ENVS=str(a.envs), SEED=str(1000 * rnd + idx), MAX_BRANCHES=str(a.max_branches))
            if pid:
                ppath = adir / "programs" / f"{pid}.json"
                if not ppath.exists():
                    ppath.parent.mkdir(parents=True, exist_ok=True)
                    ppath.write_text(json.dumps(archive.program(pid).program, indent=1))
                env["PROGRAM"] = str(ppath)
            say(f"round {rnd} {cell} program={pid} gap={gap} -> {out}")
            t0 = time.time()
            try:
                rc = subprocess.run(["bash", str(TRIAL)], env=env, timeout=a.timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            stats = ingest_run(out, archive, cell, run_id) if out.exists() else {}
            c2 = counts(archive, cell)
            grew = any(min(c2[q], QUOTA[q]) > min(c[q], QUOTA[q]) for q in QUOTA)
            st["rounds"] = rnd + 1
            st["stagnant"] = 0 if grew else st["stagnant"] + 1
            st["runs"].append({"run": run_id, "program": pid, "rc": rc, "seconds": round(time.time() - t0),
                               "stats": stats, "counts": c2})
            say(f"  rc={rc} {round(time.time() - t0)}s stats={stats} counts={c2}")
            if not any(gaps(c2).values()):
                write_manifest(archive, cell, root / "runs" / cell)
                st["status"] = "pending_upload"
                tok = os.environ.get("MS_TOKEN_FILE", "")
                if tok and Path(tok).exists():
                    ur = subprocess.run(["/usr/bin/python3", str(UPLOAD), str(root / "runs" / cell), cell],
                                        env=dict(os.environ, MODELSCOPE_CREDENTIALS_PATH="/mnt/nfs/fjj/.modelscope_fjj"))
                    st["status"] = "uploaded" if ur.returncode == 0 else "pending_upload"
                    say(f"  upload {cell} rc={ur.returncode}")
            elif st["stagnant"] >= STAGNANT_LIMIT:
                st.update(status="infeasible", reason=infeasible_reason(archive, cell, c2))
                say(f"  INFEASIBLE {cell}: {st['reason']}")
            spath.write_text(json.dumps(state, indent=1))
            rep = report(root, archive, state)
            say(f"  report: done {rep['cells_done']}/40 quota {rep['mean_quota_rate']} "
                f"steps {rep['total_sim_steps']} unsafe/10k {rep['unsafe_per_10k_steps']}")
        if active == 0:
            say("no active cells left")
            break
    log.close()


if __name__ == "__main__":
    main()
