#!/usr/bin/env python3
"""Upload the Wan drift T1 / T2 bundle (staged under STAGE) to a private ModelScope dataset repo.

Units are the top-level directories of STAGE (e.g. t1, code, t2), uploaded with upload_folder and verified by
remote file size; README.md and SHA256SUMS last. Never deletes remote or local files; verified units are skipped
on rerun, so rerunning after staging more files only uploads the new / changed ones.

  MS_TOKEN_FILE=<600 file> python upload_wan_drift.py [--stage DIR] [--repo NS/NAME] [--units t1,code]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi


def say(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def remote_files(api, repo, root):
    out, page = {}, 1
    while True:
        try:
            fs = api.get_dataset_files(repo, root_path=root, recursive=True, page_number=page, page_size=500)
        except Exception:
            return out
        for f in fs:
            if f.get("Type", "blob") != "tree":
                out[f["Path"]] = int(f.get("Size", -1))
        if len(fs) < 500:
            return out
        page += 1


def upload_unit(api, tok, repo, stage, prefix):
    sdir = stage / prefix
    # hidden files (the uploader writes its own .ms_upload_cache into the folder) are not part of the bundle
    local = {f"{prefix}/{p.relative_to(sdir).as_posix()}": p.stat().st_size for p in sdir.rglob("*")
             if p.is_file() and not p.name.startswith(".")}
    remote = remote_files(api, repo, prefix)
    todo = [k for k, s in local.items() if remote.get(k) != s]
    if todo:
        say(f"upload {prefix}: {len(todo)} of {len(local)} files, {sum(local[k] for k in todo) / 1e9:.2f} GB")
        api.upload_folder(repo, folder_path=str(sdir), path_in_repo=prefix, repo_type="dataset",
                          commit_message=f"Upload {prefix}", token=tok)
        remote = remote_files(api, repo, prefix)
    bad = [k for k, s in local.items() if remote.get(k) != s]
    say(f"{'verified' if not bad else 'MISMATCH'} {prefix}: {len(local)} files, "
        f"{sum(local.values()) / 1e9:.2f} GB, {len(bad)} bad {bad[:3]}")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="/mnt/cpfs/jieduan/work/scgrpo/ms_stage")
    ap.add_argument("--repo", default="Warrenustc1958/CodeWorld-Safe-Wan-drift-T1T2")
    ap.add_argument("--units", default="", help="comma list; default = every top-level directory of the stage")
    args = ap.parse_args()
    stage = Path(args.stage)
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    if not api.repo_exists(args.repo, repo_type="dataset", token=tok):
        api.create_repo(args.repo, token=tok, visibility="private", repo_type="dataset", exist_ok=True)
        say(f"created private dataset repo {args.repo}")
    units = args.units.split(",") if args.units else sorted(p.name for p in stage.iterdir() if p.is_dir())
    done = set()
    for attempt in range(1, 6):
        for u in units:
            if u in done:
                continue
            try:
                if upload_unit(api, tok, args.repo, stage, u):
                    done.add(u)
            except Exception as e:  # network / rate limit: retry next pass
                say(f"FAILED {u}: {e!r}"[:300])
        if len(done) == len(units):
            for name in ("README.md", "SHA256SUMS"):
                if (stage / name).exists():
                    api.upload_file(path_or_fileobj=str(stage / name), path_in_repo=name, repo_id=args.repo,
                                    repo_type="dataset", commit_message=name, token=tok)
            say(f"ALL VERIFIED: {', '.join(units)}")
            return 0
        time.sleep(120)
    say("gave up after retries; rerun to continue")
    return 1


if __name__ == "__main__":
    sys.exit(main())
