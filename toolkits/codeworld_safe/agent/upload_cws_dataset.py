#!/usr/bin/env python3
"""Upload the CWS Wan training dataset to ModelScope and verify every file.

Repo: Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout (dataset). Layout
  README.md                 dataset card
  index/cws_v1/             manifest.jsonl (repo-relative paths), stats, selection
                            report, labels/, motion/
  index/v1/labels/          labels of the legacy rows the manifest uses
  step_e/runs/<cell>/       all Step E audits (json + agentview / wrist npz)
  step_e/archive/           outcomes.jsonl (admission records), state/report/gaps
  legacy/...                legacy episodes the manifest uses (fill + old val/test,
                            with the root prefixes branches need), relative paths kept
  regrade/kin_legacy_5k.jsonl   new-grader regrade of the legacy 5k set
Files are staged as per-file symlinks under STAGE (upload_folder lists files with
glob + is_file, which follows file links but not directory links), so nothing is
copied and no upload cache is written into the source trees. Units are re-uploaded
on later passes while the Step E campaign keeps adding files. Never deletes remote
or source files.

  MS_TOKEN_FILE=<600 file> /usr/bin/python3 upload_cws_dataset.py [--once]
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = os.environ.get("MS_REPO", "Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout")
STAGE = Path(os.environ.get("MS_STAGE", "/data_cf/cws_ms_stage"))
STEP_E = Path("/data_cf/cws/step_e")
CWS = Path("/data_cf/wan_ac_sft/index/cws_v1")
V1 = Path("/mnt/nfs/fjj/wan_ac_sft/index/v1")
LEGACY = Path("/mnt/nfs/fjj/Code/rlinf-safety-runtime")
KIN = Path("/data_cf/cws/kin_legacy_5k/rows.jsonl")
PATH_MAP = [(str(STEP_E) + "/", "step_e/"), (str(LEGACY) + "/", "legacy/"),
            (str(CWS) + "/", "index/cws_v1/"), (str(V1) + "/", "index/v1/")]
SKIP = {".ms_upload_cache", ".ms_upload_progress", "upload_verify.json"}
LOG = STAGE / "upload.log"


def say(msg):
    line = f"[ms {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def portable(s):
    for a, b in PATH_MAP:
        if isinstance(s, str) and s.startswith(a):
            return b + s[len(a):]
    return s


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.is_symlink() and os.readlink(dst) == str(src):
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def mirror(src_dir: Path, dst_dir: Path):
    for p in src_dir.rglob("*"):
        if p.is_file() and not (SKIP & set(p.parts)) and p.name not in SKIP:
            link(dst_dir / p.relative_to(src_dir), p)


def write(dst: Path, text: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, dst)


def stage_static():
    """index / legacy / regrade / README; the manifest is a fixed snapshot."""
    rows = [json.loads(l) for l in (CWS / "manifest.jsonl").read_text().splitlines()]
    out, legacy_files, v1_labels = [], set(), set()
    for r in rows:
        for k in ("agent_segments", "wrist_segments"):
            for seg in r.get(k) or []:
                if seg[0].startswith(str(LEGACY)):
                    legacy_files.add(seg[0])
        if r.get("json", "").startswith(str(LEGACY)):
            legacy_files.add(r["json"])
        if r.get("labels", "").startswith(str(V1)):
            v1_labels.add(r["labels"])
        q = dict(r)
        for k in ("json", "labels"):
            q[k] = portable(q.get(k))
        for k in ("agent_segments", "wrist_segments"):
            if q.get(k):
                q[k] = [[portable(s[0]), s[1], s[2]] for s in q[k]]
        out.append(q)
    idx = STAGE / "index"
    write(idx / "cws_v1" / "manifest.jsonl", "".join(json.dumps(q) + "\n" for q in out))
    for name in ("stats.jsonl", "selection_report.json"):
        link(idx / "cws_v1" / name, CWS / name)
    mirror(CWS / "labels", idx / "cws_v1" / "labels")
    for p in (CWS / "motion").iterdir():
        if p.name.endswith(".npy"):
            link(idx / "cws_v1" / "motion" / p.name, p)
    for f in v1_labels:
        link(idx / "v1" / Path(f).relative_to(V1), Path(f))
    for f in legacy_files:
        link(STAGE / "legacy" / Path(f).relative_to(LEGACY), Path(f))
    kin = [json.loads(l) for l in KIN.read_text().splitlines()]
    for k in kin:
        k["path"] = portable(k["path"])
    write(STAGE / "regrade" / "kin_legacy_5k.jsonl", "".join(json.dumps(k) + "\n" for k in kin))
    say(f"staged index ({len(out)} rows), legacy ({len(legacy_files)} files), v1 labels ({len(v1_labels)})")


def stage_step_e():
    """Only runs the collector finished (listed in outcomes.jsonl); interrupted rounds are skipped."""
    known = set()
    for l in (STEP_E / "archive" / "outcomes.jsonl").read_text().splitlines():
        o = json.loads(l)
        known.add((o["cell"], o["run"]))
    cells = sorted(p.name for p in (STEP_E / "runs").iterdir() if p.is_dir())
    skipped = []
    for c in cells:
        for run in sorted(p for p in (STEP_E / "runs" / c).iterdir() if p.is_dir()):
            if (c, run.name) in known:
                mirror(run, STAGE / "step_e" / "runs" / c / run.name)
            else:
                skipped.append(f"{c}/{run.name}")
    if skipped:
        say(f"skipped {len(skipped)} runs not in outcomes.jsonl: {skipped[:5]}")
    for name in ("outcomes.jsonl",):
        link(STAGE / "step_e" / "archive" / name, STEP_E / "archive" / name)
    for name in ("state.json", "report.json", "gaps.json", "campaign.log"):
        if (STEP_E / name).exists():
            link(STAGE / "step_e" / "archive" / name, STEP_E / name)
    return cells


def units(cells):
    u = [("README.md", None), ("regrade", "regrade"), ("index", "index"), ("step_e/archive", "step_e/archive"),
         ("legacy", "legacy")]
    u += [(f"step_e/runs/{c}", f"step_e/runs/{c}") for c in cells]
    return u


def remote_files(api, root):
    out, page = {}, 1
    while True:
        fs = api.get_dataset_files(REPO, root_path=root, recursive=True, page_number=page, page_size=500)
        for f in fs:
            if f.get("Type", "blob") != "tree":
                out[f["Path"]] = int(f.get("Size", -1))
        if len(fs) < 500:
            return out
        page += 1


def local_files(stage_dir: Path, prefix: str):
    return {f"{prefix}/{p.relative_to(stage_dir).as_posix()}": p.stat().st_size
            for p in stage_dir.rglob("*") if p.is_file() and p.name not in SKIP and not p.name.endswith(".tmp")}


def upload_unit(api, tok, name, prefix, state):
    if prefix is None:  # README
        local = {"README.md": (STAGE / "README.md").stat().st_size}
        remote = remote_files(api, "")
        if remote.get("README.md") != local["README.md"]:
            api.upload_file(path_or_fileobj=str(STAGE / "README.md"), path_in_repo="README.md", repo_id=REPO,
                            repo_type="dataset", commit_message="Dataset card", token=tok)
            remote = remote_files(api, "")
    else:
        sdir = STAGE / prefix
        local = local_files(sdir, prefix)
        remote = remote_files(api, prefix)
        todo = [k for k, s in local.items() if remote.get(k) != s]
        if todo:
            say(f"upload {name}: {len(todo)} of {len(local)} files new/changed")
            api.upload_folder(REPO, folder_path=str(sdir), path_in_repo=prefix, repo_type="dataset",
                              commit_message=f"Upload {name}", token=tok)
            remote = remote_files(api, prefix)
    bad = [k for k, s in local.items() if remote.get(k) != s]
    ok = len(local) > 0 and not bad
    state[name] = {"files": len(local), "bytes": sum(local.values()), "bad": len(bad), "ok": ok,
                   "checked": time.strftime("%F %T")}
    say(f"{'verified' if ok else 'MISMATCH'} {name}: {len(local)} files, {len(bad)} bad")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=1800)
    a = ap.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    readme = Path(__file__).with_name("cws_dataset_README.md")
    if readme.exists():
        shutil.copyfile(readme, STAGE / "README.md")
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    state_f = STAGE / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    stage_static()
    while True:
        cells = stage_step_e()
        for name, prefix in units(cells):
            try:
                upload_unit(api, tok, name, prefix, state)
            except Exception as e:  # network, rate limit: retry next pass
                state[name] = {"ok": False, "error": repr(e)[:300], "checked": time.strftime("%F %T")}
                say(f"FAILED {name}: {e!r}"[:400])
            write(state_f, json.dumps(state, indent=1))
        n_ok = sum(1 for v in state.values() if v.get("ok"))
        tb = sum(v.get("bytes", 0) for v in state.values() if v.get("ok")) / 2**30
        say(f"pass done: {n_ok}/{len(state)} units verified, {tb:.1f} GB")
        done = "no active cells left" in (STEP_E / "campaign.log").read_text(errors="ignore")
        if a.once or (done and n_ok == len(state)):
            say("finished")
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
