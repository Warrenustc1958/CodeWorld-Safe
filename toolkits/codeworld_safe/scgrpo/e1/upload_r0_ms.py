"""Add R0 (group-isolated dual-view D2 data + gate), E7 step-20 weights and the E7 configs/curves to the same
ModelScope folder (scgrpo_e1_e8_20261030/). Already-uploaded files are skipped by size.
  MS_TOKEN_FILE=/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws /usr/bin/python3 upload_r0_ms.py [--dry-run]
"""
import sys

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
import upload_e1e8_ms as E  # noqa: E402  (sets ROOT / STAGE / README on upload_causal_ms)

U, C, S, RM = E.U, E.C, E.S, E.RM
U.UNITS = [
    ("P1", "rm_data/frames_d2g", RM / "frames_d2g"),
    ("P1", "checkpoints/e7_step_20", E.ck("e7", 20)),
    ("P1", "configs", C / "configs_e1e8"),
    ("P2", "gates", RM / "runs/gates_e4"),
    ("P2", "tensorboard/e7", C / "e7/tensorboard"),
    ("P2", "killtrace", C / "killtrace_upload"),
]
U.README = U.README + """
## 10-30 evening additions
- rm_data/frames_d2g: D2 rollouts with agent + wrist views, split by SOURCE TRAJECTORY GROUP (d2gtest = 10 groups
  that appear in no RM training set). Gate: gates/gate_d2g.json (first-decision and final-rule precision/recall).
- checkpoints/e7_step_20: E7 = E8-b config (RM v3 0.8x4, final rule, fixed 64 steps) on 8 GPUs, 80 steps (in progress).
- killtrace/: kernel signal trace of the external SIGKILLs that repeatedly killed training on gpu006.
"""

if __name__ == "__main__":
    sys.exit(U.main())
