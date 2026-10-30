"""Upload the priority-1 and priority-2 results of the SC-GRPO root-cause / causal study to ModelScope
(dataset Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout, under causal_r1_rm_v5/) and verify
every file by size against the remote listing.

Staging (/data_cf/cws_ms_stage_causal, same filesystem as the sources): files are hard-linked; files larger
than 4 GB (checkpoint weights) are split into 4 GB parts with a SHA256SUMS of the original
(restore: cat X.part_* > X; sha256sum -c SHA256SUMS). Upload is idempotent: files already on the remote with
the same size are skipped, so rerunning continues where it stopped.
  MS_TOKEN_FILE=/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws /usr/bin/python3 upload_causal_ms.py [--dry-run] [--only P1|P2]
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout"
ROOT = "causal_r1_rm_v5"
STAGE = Path("/data_cf/cws_ms_stage_causal")
LOG = STAGE / "upload.log"
SPLIT = 4 * 1024 ** 3
S = Path("/data_cf/cws_sam/v1/scgrpo")
RM = Path("/data_cf/cws_rm")
CK = S / "causal/r1/scgrpo_causal_r1/checkpoints"

UNITS = [  # (priority, path in repo under ROOT, local source)
    ("P1", "r1_checkpoints/global_step_10", CK / "global_step_10"),
    ("P1", "r1_checkpoints/global_step_20", CK / "global_step_20"),
    ("P1", "rm_data/frames_pol_v5", RM / "frames_pol_v5"),
    ("P1", "rm_data/frames_pol_v4", RM / "frames_pol_v4"),
    ("P1", "rm_models/rm_v4_onpolicy", RM / "runs/rm_v4_onpolicy"),
    ("P1", "rm_models/rm_v5_tail", RM / "runs/rm_v5_tail"),
    ("P2", "eval_real/r1_step_10", S / "eval_real/r1_step_10"),
    ("P2", "eval_real/r1_step_20", S / "eval_real/r1_step_20"),
    ("P2", "eval_real/off_base", S / "eval_real/off_base"),
    ("P2", "eval_real/aba_step_40", S / "eval_real/aba_step_40"),
    ("P2", "eval_real/abb_step_40", S / "eval_real/abb_step_40"),
    ("P2", "eval_real/abc_step_40", S / "eval_real/abc_step_40"),
    ("P2", "diagnostics/frozen_full_v4", S / "diagnostics/rootcause_20261027_v1/frozen_full_v4"),
    ("P2", "diagnostics/diag_scripts", S / "diag"),
    ("P2", "tensorboard/r1", S / "causal/r1/tensorboard"),
    ("P2", "tensorboard/r1_gpu34", S / "causal/r1_gpu34_1028/tensorboard"),
    ("P2", "tensorboard/r0b", S / "causal/r0b/tensorboard"),
]
SKIP = (".ms_upload_cache", "__pycache__")

README = """# SC-GRPO root cause and causal test (2026-10-27/28)

- r1_checkpoints/: R1 = SC-GRPO with the reward from a synchronised real LIBERO simulator (real goal,
  eta = 0, affordance keyframes only), global_step_10 / 20 (actor full_weights.pt + dcp_checkpoint).
  Weights larger than 4 GB are split: `cat X.part_* > X; sha256sum -c SHA256SUMS`.
- rm_data/frames_pol_v4|v5: on-policy Wan frames of 5 pi0.5 checkpoints labelled by the synchronised real
  simulator (RMFrames format). v5 keeps a 16-step success tail with per-frame real goal labels and has
  poltrain / polval (model selection) / poltest (final gate) splits by keyframe group.
- rm_models/: RM v4 (on-policy data) and RM v5 (tail data) checkpoints; rm_v5_tail/rule_sweep has the
  detection-rule sweep (RM v3 with p>=0.8 on 4 consecutive frames passes the poltest gate).
- eval_real/: real LIBERO-Safety grader audits (60 cells x 20, official protocol) of the base policy, R1
  step 10 / 20 (partial) and ablations a/b/c step 40.
- diagnostics/frozen_full_v4: frozen-policy observation study (real vs Wan image x real vs read-out state).
- tensorboard/: training curves of R1 and R0b.
Code: github.com/Warrenustc1958/CodeWorld-Safe (branch codeworld-safe -> main).
"""


def say(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sha256(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def stage_unit(prefix, src):
    dst = STAGE / ROOT / prefix
    sums = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or any(s in p.parts for s in SKIP):
            continue
        rel = p.relative_to(src)
        q = dst / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.stat().st_size > SPLIT:
            if not list(q.parent.glob(q.name + ".part_*")):
                subprocess.run(["split", "-b", "4G", "-d", "-a", "2", str(p), str(q) + ".part_"], check=True)
            sums.append((sha256(p), rel.as_posix()))
        elif not q.exists():
            try:
                os.link(p, q)
            except OSError:
                shutil.copy2(p, q)
    if sums:
        (dst / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for h, r in sums))
    files = [f for f in dst.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def remote_files(api, root):
    out, page = {}, 1
    while True:
        try:
            fs = api.get_dataset_files(REPO, root_path=root, recursive=True, page_number=page, page_size=500)
        except Exception:
            return out
        for f in fs:
            if f.get("Type", "blob") != "tree":
                out[f["Path"]] = int(f.get("Size", -1))
        if len(fs) < 500:
            return out
        page += 1


def upload_unit(api, tok, prefix):
    sdir = STAGE / ROOT / prefix
    rp = f"{ROOT}/{prefix}"
    local = {f"{rp}/{p.relative_to(sdir).as_posix()}": p.stat().st_size for p in sdir.rglob("*") if p.is_file()}
    remote = remote_files(api, rp)
    todo = [k for k, s in local.items() if remote.get(k) != s]
    if todo:
        say(f"upload {rp}: {len(todo)} of {len(local)} files, {sum(local[k] for k in todo) / 1e9:.2f} GB")
        api.upload_folder(REPO, folder_path=str(sdir), path_in_repo=rp, repo_type="dataset",
                          commit_message=f"Upload {rp}", token=tok)
        remote = remote_files(api, rp)
    bad = [k for k, s in local.items() if remote.get(k) != s]
    say(f"{'verified' if not bad else 'MISMATCH'} {rp}: {len(local)} files, {sum(local.values()) / 1e9:.2f} GB, "
        f"{len(bad)} bad {bad[:3]}")
    return not bad


def main():
    dry = "--dry-run" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    units = [(pri, pre, src) for pri, pre, src in UNITS if only is None or pri == only]
    STAGE.mkdir(parents=True, exist_ok=True)
    missing = [str(src) for _, _, src in units if not src.exists()]
    if missing:
        say(f"missing sources (skipped): {missing}")
    units = [u for u in units if u[2].exists()]
    total = 0
    for pri, pre, src in units:
        n, b = stage_unit(pre, src)
        total += b
        say(f"staged {pri} {pre}: {n} files, {b / 1e9:.2f} GB")
    (STAGE / ROOT / "README.md").write_text(README)
    say(f"staged {len(units)} units, {total / 1e9:.1f} GB")
    if dry:
        say("dry run: nothing uploaded")
        return 0
    from modelscope.hub.api import HubApi
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    done = set()
    for attempt in range(1, 6):
        for _, pre, _ in units:
            if pre in done:
                continue
            try:
                if upload_unit(api, tok, pre):
                    done.add(pre)
            except Exception as e:  # network / rate limit: retry next pass
                say(f"FAILED {pre}: {e!r}"[:300])
        if len(done) == len(units):
            api.upload_file(path_or_fileobj=str(STAGE / ROOT / "README.md"), path_in_repo=f"{ROOT}/README.md",
                            repo_id=REPO, repo_type="dataset", commit_message=f"{ROOT} README", token=tok)
            say(f"ALL VERIFIED ({len(units)} units)")
            return 0
        say(f"pass {attempt}: {len(done)}/{len(units)} verified, retrying in 5 min")
        time.sleep(300)
    say(f"gave up with {len(done)}/{len(units)} verified; rerun to continue (already uploaded files are skipped)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
