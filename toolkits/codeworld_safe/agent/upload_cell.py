#!/usr/bin/env python3
"""Upload one finished Step E cell to ModelScope and verify every file (path + size).

  MS_TOKEN_FILE=<600 file> /usr/bin/python3 upload_cell.py <cell_dir> <cell>
Exit 0 only if every local file is present remotely with the same size;
writes <cell_dir>/upload_verify.json. Never deletes anything.
"""
import json
import os
import sys
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = os.environ.get("MS_REPO", "Warrenustc1958/CodeWorld-Safe-counterfactual-pi05")
PREFIX = os.environ.get("MS_PREFIX", "step_e")


def remote_files(api, root_path):
    out, page = {}, 1
    while True:
        files = api.get_dataset_files(REPO, root_path=root_path, recursive=True,
                                      page_number=page, page_size=500)
        for f in files:
            if f.get("Type", "blob") != "tree":
                out[f["Path"]] = int(f.get("Size", -1))
        if len(files) < 500:
            return out
        page += 1


def main():
    cell_dir, cell = Path(sys.argv[1]), sys.argv[2]
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    dest = f"{PREFIX}/{cell}"
    api.upload_folder(REPO, folder_path=str(cell_dir), path_in_repo=dest,
                      commit_message=f"Step E {cell}", repo_type="dataset", token=tok)
    remote = remote_files(api, dest)
    n = miss = diff = 0
    bad = []
    for p in cell_dir.rglob("*"):
        if not p.is_file() or ".ms_upload_cache" in p.parts or p.name == "upload_verify.json":
            continue
        n += 1
        key = f"{dest}/{p.relative_to(cell_dir).as_posix()}"
        if key not in remote:
            miss += 1
            bad.append(key)
        elif remote[key] != p.stat().st_size:
            diff += 1
            bad.append(key)
    res = {"repo": REPO, "dest": dest, "local_files": n, "missing": miss, "size_mismatch": diff,
           "bad": bad[:50], "ok": n > 0 and miss == 0 and diff == 0}
    (cell_dir / "upload_verify.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("dest", "local_files", "missing", "size_mismatch", "ok")}))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
