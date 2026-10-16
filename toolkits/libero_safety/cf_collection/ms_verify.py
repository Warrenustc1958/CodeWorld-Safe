"""Compare local round dirs with the ModelScope dataset repo (path + size)."""
import json, os, sys
from pathlib import Path
from modelscope.hub.api import HubApi
REPO = "Warrenustc1958/LIBERO-Safety-Pi05-rollout"
root = Path(sys.argv[1]); names = sys.argv[2:]
api = HubApi()
remote = {}
for name in names:
    page = 1
    while True:
        files = api.get_dataset_files(REPO, root_path=f"cf_l0l1_g006/runs/{name}", recursive=True,
                                      page_number=page, page_size=500)
        if page == 1 and files: print("sample entry:", {k: files[0].get(k) for k in list(files[0])[:8]}, file=sys.stderr)
        for f in files:
            if f.get("Type", "blob") != "tree":
                remote[f["Path"]] = int(f.get("Size", -1))
        if len(files) < 500: break
        page += 1
bad = 0
for name in names:
    d = root / "runs" / name
    n = miss = diff = 0
    for p in d.rglob("*"):
        if not p.is_file() or ".ms_upload_cache" in p.parts: continue
        n += 1
        key = f"cf_l0l1_g006/runs/{name}/{p.relative_to(d).as_posix()}"
        if key not in remote: miss += 1
        elif remote[key] != p.stat().st_size: diff += 1
    bad += miss + diff
    print(f"{name}: local={n} missing={miss} size_mismatch={diff}")
print("TOTAL_BAD", bad)
