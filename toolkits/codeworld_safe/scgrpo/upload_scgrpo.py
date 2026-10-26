#!/usr/bin/env python3
"""Upload the SC-GRPO training bundle to Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout.

Layout in the dataset repo:
  keyframes_v1/      GRPO initial states: anchor.npy / memory.npy split into 4 GB parts (*.part_NN),
                     meta.npz, report.json, SHA256SUMS (of the reassembled files)
  scoring/           channel_mask.json, state_reg_v1/state_reg.pt, sam_cost_all_d32/best.pt,
                     rm_v3_agent_w41k/{best.pt,args.json,final.json}
  eval/fresh_init_v1 real-env evaluation init states (unseen by policy and Wan)
  config/            libero_safety_scgrpo_openpi_pi05.yaml, run_scgrpo.sh, versions.json
  README.md
Reassemble:  cat memory.npy.part_* > memory.npy  (same for anchor.npy), then sha256sum -c SHA256SUMS.
Never deletes remote or source files; verified units are skipped on rerun.

  MS_TOKEN_FILE=<600 file> /usr/bin/python3 upload_scgrpo.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from modelscope.hub.api import HubApi

REPO = "Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout"
STAGE = Path("/dev/shm/scgrpo_ms_stage")
S = Path("/data_cf/cws_sam/v1/scgrpo")
KF = S / "keyframes_v1"
PART = "4G"
LOG = STAGE / "upload.log"

README = """# CodeWorld-Safe SC-GRPO bundle (LIBERO-Safety, pi0.5, dual-view Wan 41k)

Everything needed besides the three large model checkpoints (see "Also needed") to run safety-constrained
GRPO of pi0.5 inside the frozen action-conditioned Wan world model. Code: github.com/Warrenustc1958/CodeWorld-Safe
(`rlinf/envs/sim/world_model/safe/`, `examples/embodiment/config/libero_safety_scgrpo_openpi_pi05.yaml`,
`examples/embodiment/run_scgrpo.sh`, `toolkits/codeworld_safe/scgrpo/`).

## keyframes_v1 (GRPO initial states)
47,658 real keyframes from 8,867 Wan-seen train episodes (SAM roles cost_train / h_train / dev; no diag or fresh
data), chunk boundaries t0 (multiple of 8) with a success or safety event within 64 steps, plus 5 % random.
- anchor.npy  uint8 [N, 512, 256, 3]     frame 0, agentview over wrist, policy (180-rotated) view
- memory.npy  uint8 [N, 4, 512, 256, 3]  frames t0-3..t0
- meta.npz    mem_actions [N,4,7] (env actions that produced the memory frames), states [N,8] (pi0.5 proprio
              eef_pos + axis-angle + gripper_qpos at t0), uid, group, tcell, suite, task_id, instruction, t0,
              kind (0 rest / 1 success / 2 event / 3 both), quadrant, role
Large arrays are split: `cat anchor.npy.part_* > anchor.npy; cat memory.npy.part_* > memory.npy;
sha256sum -c SHA256SUMS`.

## scoring
- sam_cost_all_d32/best.pt   SAM safety cost head on Wan latents (3 channels c_obj / c_robot / c_grasp)
- channel_mask.json          per task cell, which cost channels have grader rules
- state_reg_v1/state_reg.pt  CNN dual frame -> pi0.5 proprio (used for imagined steps)
- rm_v3_agent_w41k/best.pt   success reward model (agent view), success = 2 consecutive frames >= 0.2

## eval/fresh_init_v1
20 fresh init states per task (40 tasks), unseen by the policy and Wan; real LIBERO-Safety evaluation with
the grader (`toolkits/codeworld_safe/scgrpo/eval_real.sh`, loaded via `env.eval.init_states_root`).

## Also needed (not in this repo)
- Wan dual-view DiT ema-step-41000: ModelScope Warrenustc1958/Wan-CodeWorld-Safe-pi05-dual-41000
- Wan2.2 VAE (Wan2.2_VAE.pth): HuggingFace RLinf/RLinf-Wan-LIBERO-Spatial
- pi0.5 LIBERO-Safety RLinf checkpoint (model.safetensors + assets/lerobot/norm_stats.json)
- Python env: RLinf openpi venv + diffsynth-studio on PYTHONPATH, plus modelscope==1.20.0, ftfy==6.3.1,
  seaborn==0.13.2; LIBERO-Safety (for real-env evaluation only)
"""


def say(msg):
    line = f"[ms {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sha256(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(bs):
            h.update(b)
    return h.hexdigest()


def link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def stage():
    k = STAGE / "keyframes_v1"
    k.mkdir(parents=True, exist_ok=True)
    sums = k / "SHA256SUMS"
    if not sums.exists():
        lines = []
        for name in ("anchor.npy", "memory.npy"):
            if not list(k.glob(f"{name}.part_*")):
                say(f"splitting {name}")
                subprocess.run(["split", "-b", PART, "-d", "-a", "2", str(KF / name), str(k / f"{name}.part_")],
                               check=True)
            lines.append(f"{sha256(KF / name)}  {name}")
            say(f"sha256 {name} done")
        sums.write_text("\n".join(lines) + "\n")
    for name in ("meta.npz", "report.json", "state_model_W.npy"):
        link(k / name, KF / name)
    sc = STAGE / "scoring"
    link(sc / "channel_mask.json", S / "channel_mask.json")
    link(sc / "state_reg_v1" / "state_reg.pt", S / "state_reg_v1" / "state_reg.pt")
    link(sc / "state_reg_v1" / "report.json", S / "state_reg_v1" / "report.json")
    link(sc / "sam_cost_all_d32" / "best.pt", Path("/data_cf/cws_sam/v1/checkpoints/cost_all_d32/best.pt"))
    link(sc / "sam_cost_all_d32" / "freeze.json", Path("/data_cf/cws_sam/v1/contract/freeze_cost_20261026b.json"))
    for n in ("best.pt", "args.json", "final.json"):
        link(sc / "rm_v3_agent_w41k" / n, Path("/data_cf/cws_rm/runs/rm_v3_agent_w41k") / n)
    fi = Path("/data_cf/cws/fresh_init/v1")
    for p in fi.rglob("*"):
        if p.is_file():
            link(STAGE / "eval" / "fresh_init_v1" / p.relative_to(fi), p)
    cfg = STAGE / "config"
    cw = Path("/mnt/nfs/fjj/Code/CodeWorld-Safe")
    link(cfg / "libero_safety_scgrpo_openpi_pi05.yaml", cw / "examples/embodiment/config/libero_safety_scgrpo_openpi_pi05.yaml")
    link(cfg / "run_scgrpo.sh", cw / "examples/embodiment/run_scgrpo.sh")
    commit = subprocess.run(["git", "-C", str(cw), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (cfg / "versions.json").write_text(json.dumps({
        "codeworld_safe_commit": commit, "wan": "s3_dual_b10/ema-step-41000",
        "sam": "cost_all_d32", "rm": "rm_v3_agent_w41k", "policy": "pi05_libero_safety_rlinf",
        "cost_budget": 0.13, "max_chunks": 8}, indent=1))
    (STAGE / "README.md").write_text(README)
    return ["config", "scoring", "eval", "keyframes_v1"]


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
    sdir = STAGE / prefix
    local = {f"{prefix}/{p.relative_to(sdir).as_posix()}": p.stat().st_size for p in sdir.rglob("*") if p.is_file()}
    remote = remote_files(api, prefix)
    todo = [k for k, s in local.items() if remote.get(k) != s]
    if todo:
        say(f"upload {prefix}: {len(todo)} of {len(local)} files, {sum(local[k] for k in todo) / 1e9:.1f} GB")
        api.upload_folder(REPO, folder_path=str(sdir), path_in_repo=prefix, repo_type="dataset",
                          commit_message=f"Upload {prefix}", token=tok)
        remote = remote_files(api, prefix)
    bad = [k for k, s in local.items() if remote.get(k) != s]
    say(f"{'verified' if not bad else 'MISMATCH'} {prefix}: {len(local)} files, "
        f"{sum(local.values()) / 1e9:.1f} GB, {len(bad)} bad {bad[:3]}")
    return not bad


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    units = stage()
    say("staged")
    done = set()
    for attempt in range(1, 6):
        for u in units:
            if u in done:
                continue
            try:
                if upload_unit(api, tok, u):
                    done.add(u)
            except Exception as e:  # network / rate limit: retry next pass
                say(f"FAILED {u}: {e!r}"[:300])
        if len(done) == len(units):
            api.upload_file(path_or_fileobj=str(STAGE / "README.md"), path_in_repo="README.md", repo_id=REPO,
                            repo_type="dataset", commit_message="README", token=tok)
            say("ALL VERIFIED")
            return 0
        time.sleep(300)
    say("gave up after retries; rerun to continue")
    return 1


if __name__ == "__main__":
    sys.exit(main())
