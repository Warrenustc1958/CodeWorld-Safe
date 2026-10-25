#!/usr/bin/env python3
"""Upload /data_cf/cws_rm and /data_cf/cws_sam to the CWS ModelScope dataset repo and verify every file.

Repo: Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout (dataset). New top-level folders:
  rm/   reward-model frame sets (frames_v1 real, frames_wan30k / frames_wan41k Wan-generated) + runs + smoke
  sam/  SAM alignment v1: contract, labels, splits, real latent cache, real/gen pairs, checkpoints, reports
frames_v2 / frames_v3 are directories of symlinks into frames_v1 + frames_wan41k; they are not
re-uploaded, their link targets are written to rm/frame_set_links.json instead.
Files are staged as per-file symlinks (nothing copied). Never deletes remote or source files.
Does not touch the root README.md that upload_cws_dataset.py maintains.

  MS_TOKEN_FILE=<600 file> /usr/bin/python3 upload_rm_sam.py
"""
import json
import os
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = "Warrenustc1958/CodeWorld-Safe-pi05-counterfactual-rollout"
STAGE = Path("/data_cf/cws_ms_stage_rmsam")
RM = Path("/data_cf/cws_rm")
SAM = Path("/data_cf/cws_sam")
SKIP = {".ms_upload_cache", ".ms_upload_progress", "__pycache__"}
LINK_DIRS = ("frames_v2", "frames_v3")
LOG = STAGE / "upload.log"
MAX_TRIES = 5

RM_README = """# rm/ - CodeWorld-Safe reward-model frame sets

Built from the Wan CWS indexes (index/cws_v1, cws_test_new); dual-view episodes only.

- frames_v1/: real frames (CodeWorld-Safe toolkits/codeworld_safe/rm/build_frames.py). Train stride 8,
  eval stride 4, plus the last 16 frames of every episode. Label = frame >= first success frame
  (episodes stop at success, so one positive per successful episode).
  train 559572 frames / 4089 pos; val 24055 / 96; test 37223 / 128; newval 52285 / 214; newtest 80290 / 314.
- frames_wan41k/ (and older frames_wan30k/): Wan-generated frames (RLinf-Safety
  toolkits/libero_safety/wan_ac_sft/build_wan_rm_frames.py, dual Wan ema-step-41000). Replay the last 48
  frames with the recorded actions, then 16 hold frames; stride 2. Hold frames inherit the episode outcome.
  wantrain 70387 / 18122 pos; wannewval 11000 / 2822; wannewtest 17600 / 4522.
- frame_set_links.json: frames_v2 / frames_v3 are symlink views over the two sets above.
- Per split: <split>_{agent,wrist}.bin (JPEG bytes), .off.npy (offsets), _meta.npz, _episodes.jsonl.
- runs/: RM checkpoints and metrics (rm_v1/v2/v3 variants); smoke_*: small smoke-test sets.
"""

SAM_README = """# sam/ - SAM alignment v1 (safety cost model on Wan latents)

Code: RLinf-Safety toolkits/libero_safety/sam_alignment/. Frozen dual Wan s3_dual_b10/ema-step-41000.

- v1/contract/: experiment contract (model hashes, window, action transform).
- v1/labels/: per-step 3-channel labels (c_obj / c_robot / c_grasp) rebuilt from grader grades.
- v1/splits/split_registry.json: init-state-group split (cost_train / h_train / dev / diag_*).
- v1/cache/real_v1/: real VAE latents per role, fp16 [N,48,4,32,16] + meta.npz.
- v1/pairs/: real vs generated latent pairs. pilot_v1 (dev, Step 5), cost_train_v1 (34573 rows),
  h_train_v1 (7562 rows). gen_z.npy is captured before VAE decoding; labels = real executed outcome.
- v1/checkpoints/: real-domain baselines B0 / B1 / B2 / B2_agent. v1/reports/: alignment + Step 5 pilot.
"""


def say(msg):
    line = f"[ms {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


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
        if p.is_file() and not p.is_symlink() and not (SKIP & set(p.parts)):
            link(dst_dir / p.relative_to(src_dir), p)


def write(dst: Path, text: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    dst.write_text(text)


def stage():
    links = {}
    for d in LINK_DIRS:
        links[d] = {p.name: os.readlink(p).replace(str(RM) + "/", "rm/")
                    for p in sorted((RM / d).iterdir()) if p.is_symlink()}
    write(STAGE / "rm" / "frame_set_links.json", json.dumps(links, indent=1))
    write(STAGE / "rm" / "README.md", RM_README)
    write(STAGE / "sam" / "README.md", SAM_README)
    units = [("rm/meta", None)]
    for p in sorted(RM.iterdir()):
        if p.name in LINK_DIRS:
            continue
        if p.is_dir():
            mirror(p, STAGE / "rm" / p.name)
            units.append((f"rm/{p.name}", f"rm/{p.name}"))
        elif p.is_file():
            link(STAGE / "rm" / p.name, p)
    for p in sorted((SAM / "v1").iterdir()):
        if p.is_dir():
            mirror(p, STAGE / "sam" / "v1" / p.name)
            units.append((f"sam/v1/{p.name}", f"sam/v1/{p.name}"))
        elif p.is_file():
            link(STAGE / "sam" / "v1" / p.name, p)
    units.append(("sam/meta", None))
    return units


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


def loose_files(top):
    """Loose files directly under rm/, or under sam/ and sam/v1/."""
    out = {}
    dirs = [STAGE / top] + ([STAGE / "sam" / "v1"] if top == "sam" else [])
    for d in dirs:
        for p in d.iterdir():
            if p.is_file():
                out[p.relative_to(STAGE).as_posix()] = p
    return out


def remote_for(api, keys):
    remote = {}
    for root in {str(Path(k).parent) for k in keys}:
        remote.update({k: v for k, v in remote_files(api, root).items() if str(Path(k).parent) == root})
    return remote


def upload_unit(api, tok, name, prefix, state):
    if prefix is None:
        files = loose_files(name.split("/")[0])
        local = {k: p.stat().st_size for k, p in files.items()}
        remote = remote_for(api, local)
        for k, p in files.items():
            if remote.get(k) != local[k]:
                api.upload_file(path_or_fileobj=str(p), path_in_repo=k, repo_id=REPO, repo_type="dataset",
                                commit_message=f"Upload {k}", token=tok)
        remote = remote_for(api, local)
    else:
        sdir = STAGE / prefix
        local = {f"{prefix}/{p.relative_to(sdir).as_posix()}": p.stat().st_size
                 for p in sdir.rglob("*") if p.is_file()}
        remote = remote_files(api, prefix)
        todo = [k for k, s in local.items() if remote.get(k) != s]
        if todo:
            say(f"upload {name}: {len(todo)} of {len(local)} files, "
                f"{sum(local[k] for k in todo) / 1e9:.2f} GB new/changed")
            api.upload_folder(REPO, folder_path=str(sdir), path_in_repo=prefix, repo_type="dataset",
                              commit_message=f"Upload {name}", token=tok)
            remote = remote_files(api, prefix)
    bad = [k for k, s in local.items() if remote.get(k) != s]
    ok = len(local) > 0 and not bad
    state[name] = {"files": len(local), "bytes": sum(local.values()), "bad": len(bad), "ok": ok,
                   "bad_examples": bad[:5], "checked": time.strftime("%F %T")}
    say(f"{'verified' if ok else 'MISMATCH'} {name}: {len(local)} files, "
        f"{sum(local.values()) / 1e9:.2f} GB, {len(bad)} bad")
    return ok


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    state_f = STAGE / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    units = stage()
    say(f"staged {len(units)} units")
    for attempt in range(1, MAX_TRIES + 1):
        for name, prefix in units:
            if state.get(name, {}).get("ok"):
                continue
            try:
                upload_unit(api, tok, name, prefix, state)
            except Exception as e:  # network / rate limit: retry next pass
                state[name] = {"ok": False, "error": repr(e)[:300], "checked": time.strftime("%F %T")}
                say(f"FAILED {name}: {e!r}"[:400])
            state_f.write_text(json.dumps(state, indent=1))
        n_ok = sum(1 for n, _ in units if state.get(n, {}).get("ok"))
        gb = sum(state[n].get("bytes", 0) for n, _ in units if state.get(n, {}).get("ok")) / 1e9
        say(f"pass {attempt} done: {n_ok}/{len(units)} units verified, {gb:.1f} GB")
        if n_ok == len(units):
            say("ALL VERIFIED")
            return 0
        time.sleep(300)
    say("gave up after retries; rerun to continue")
    return 1


if __name__ == "__main__":
    sys.exit(main())
