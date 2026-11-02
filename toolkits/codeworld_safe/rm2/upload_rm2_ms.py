"""Upload the 11-01/11-02 RM work, key policy checkpoints and datasets to ModelScope
(dataset Warrenustc1958/CodeWorld-Safe-LIBERO-Safety-pi05-GRPO-rollout, under scgrpo_rm2_20261102/).

Reuses upload_causal_ms.py (hard-link staging, files > 4 GB split into 4 GB parts + SHA256SUMS, size-verified,
idempotent). Unlike its main(), units are processed one at a time: stage -> upload -> verify -> delete the staged
copy, so split checkpoint parts never pile up on /data_cf. Policy checkpoints: actor model_state_dict only.
  MS_TOKEN_FILE=/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws /usr/bin/python3 upload_rm2_ms.py [--dry-run] [--only P1|P2]
"""
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
import upload_causal_ms as U  # noqa: E402

S, RM = U.S, U.RM
C = S / "causal"
MSD = "actor/model_state_dict"


def ck(run, step):
    return C / run / f"scgrpo_causal_{run}" / "checkpoints" / f"global_step_{step}" / MSD


U.ROOT = "scgrpo_rm2_20261102"
U.STAGE = Path("/data_cf/cws_ms_stage_rm2")
U.LOG = U.STAGE / "upload.log"
OLD_RM = ["rm_v1_agent", "rm_v1_dual", "rm_v2_agent", "rm_v2_agent_w41k", "rm_v3_agent_w41k", "rm_v3b_agent_w41k",
          "rm_v3c_agent_w41k", "rm_v7a_agent", "rm_v7_dual", "rm_v7w_dual", "rm_v8_dual"]  # v4/v5/v6 already uploaded
U.UNITS = (
    [("P1", f"rm_models/{r}", RM / "runs" / r) for r in OLD_RM]
    + [("P1", "rm_models/R-A_resnet_rm2", S / "rm2/ra"),
       ("P1", "rm_models/R-C_qwen3vl4b_lora", S / "rm2/rc"),
       ("P1", "rm_eval", S / "rm2/rm_eval_json"),  # old_rm_eval / rb_eval / rc_eval / ra final, thr sweeps
       ("P1", "rm_data/ds_v1", S / "rm2/ds_v1"),
       ("P1", "keyframe_scan", S / "kf_scan")]
    + [("P1", f"checkpoints/{run}_step_{st}", ck(run, st)) for run, st in
       [("e10o", 10), ("e10o", 20), ("e10o", 40), ("e11sc", 20), ("e11sam", 20), ("e12r", 10), ("e13", 10),
        ("e15a", 10), ("e15q", 10)]]
    + [("P2", f"collection/{t}", C / t / "dump") for t in ("colbase", "col10a", "col10b", "col10c", "col12r", "col13")]
    + [("P2", f"dumps/{t}", C / t / "dump") for t in ("e13d", "e14", "e15a", "e15q", "m0", "m1", "m2", "m3", "m1r")]
    + [("P2", f"eval_real/{n}", S / "eval_real" / n) for n in
       ("e10o_step_10", "e10o_step_20", "e10o_step_40", "e11sc_step_10", "e11sc_step_20", "e11sam_step_10",
        "e11sam_step_20", "e12r_step_10", "e13_step_10", "e14_step_10", "e15a_step_10", "e15q_step_10")]
    + [("P2", "reports", S / "causal/msnight")]
)
U.README = """# SC-GRPO 11-01/11-02: RM retraining, key checkpoints and datasets
Code: github.com/Warrenustc1958/CodeWorld-Safe (toolkits/codeworld_safe/rm2, scgrpo/night2), docs in
toolkits/codeworld_safe/scgrpo/{E12_E14_RM_PROGRESS,RM2_PLAN,RM_GRPO_THEORY,VARIANCE_MEASUREMENT}.md.
- rm_models/: CWSRewardModel checkpoints rm_v1..v3c, v7*, v8 (v4/v5 in causal_r1_rm_v5/, v6 in
  scgrpo_e1_e8_20261030/); R-A = dual-view ResNet retrained on rm_data/ds_v1 (best.pt has "threshold");
  R-C = Qwen3-VL-4B-Instruct LoRA r16 adapter (best/), base model Qwen/Qwen3-VL-4B-Instruct.
- rm_eval/: offline metrics of all RMs on ds_v1 val/test with the same protocol (threshold for <= 1% failed-episode fire).
- rm_data/ds_v1: build_rm_dataset.py output (train quota-selected, val/test all frames; split by source uid; test =
  unseen policy E12-R). Files: {split}.jpg.bin + {split}.off.npy + {split}.jsonl.
- keyframe_scan/: reset-time grader scan of all 47,658 keyframes (kf_reset.npz), kf_exclude / kf_weights / kf_t0_8 /
  kf_decision_8_48 sampling files for keyframes_v1.
- checkpoints/<run>_step_<n>/: actor model_state_dict of E10-O (real reward, Wan obs), E11-SC/SAM, E12-R (real
  obs), E13 (grader cost eta .5), E15-A (R-A reward), E15-Q (R-C reward). Weights > 4 GB are split:
  `cat X.part_* > X; sha256sum -c SHA256SUMS`.
- collection/: phase-1 on-policy collection (per-episode JSON + arrays: chunk mid/end JPEGs, Wan latents, actions,
  per-step goal / grader channels) of base, E10-O s10/s20/s40, E12-R, E13 policies.
- dumps/: per-episode GRPO / measurement dumps (D3, E14 noise, E15, M0-M3 variance measurement, M1R real obs, partial).
- eval_real/: official LIBERO-Safety grader audits, 60 cells x 20.
- reports/: pre-registered variance measurement report (report.md / report.json).
"""


def main():
    dry = "--dry-run" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    units = [u for u in U.UNITS if (only is None or u[0] == only)]
    U.STAGE.mkdir(parents=True, exist_ok=True)
    missing = [str(s) for _, _, s in units if not s.exists()]
    if missing:
        U.say(f"missing sources (skipped): {missing}")
    units = [u for u in units if u[2].exists()]
    U.say(f"{len(units)} units, {sum(sum(f.stat().st_size for f in (s.rglob('*') if s.is_dir() else [s]) if f.is_file()) for _, _, s in units) / 1e9:.1f} GB")
    if dry:
        return 0
    from modelscope.hub.api import HubApi
    tok = Path(os.environ["MS_TOKEN_FILE"]).read_text().strip()
    api = HubApi()
    api.login(tok)
    done = set()
    for attempt in range(1, 6):
        for pri, pre, src in units:
            if pre in done:
                continue
            try:
                n, b = U.stage_unit(pre, src)
                U.say(f"staged {pri} {pre}: {n} files, {b / 1e9:.2f} GB")
                if U.upload_unit(api, tok, pre):
                    done.add(pre)
                    shutil.rmtree(U.STAGE / U.ROOT / pre, ignore_errors=True)  # staged copy only
            except Exception as e:  # network / rate limit: retry next pass
                U.say(f"FAILED {pre}: {e!r}"[:300])
        if len(done) == len(units):
            (U.STAGE / U.ROOT).mkdir(parents=True, exist_ok=True)
            (U.STAGE / U.ROOT / "README.md").write_text(U.README)
            api.upload_file(path_or_fileobj=str(U.STAGE / U.ROOT / "README.md"), path_in_repo=f"{U.ROOT}/README.md",
                            repo_id=U.REPO, repo_type="dataset", commit_message=f"{U.ROOT} README", token=tok)
            U.say(f"ALL VERIFIED ({len(units)} units)")
            return 0
        U.say(f"pass {attempt}: {len(done)}/{len(units)} verified, retrying in 5 min")
        time.sleep(300)
    U.say(f"gave up with {len(done)}/{len(units)} verified; rerun to continue (already uploaded files are skipped)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
