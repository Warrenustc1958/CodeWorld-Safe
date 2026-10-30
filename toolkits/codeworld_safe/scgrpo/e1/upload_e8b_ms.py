"""Add E8-b (step-10 weights, real eval, tensorboard) and the refreshed summary/configs to the same ModelScope
folder as upload_e1e8_ms.py (scgrpo_e1_e8_20261030/); already-uploaded files are skipped by size.
  MS_TOKEN_FILE=/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws /usr/bin/python3 upload_e8b_ms.py [--dry-run]
"""
import sys

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
import upload_e1e8_ms as E  # noqa: E402  (sets ROOT / STAGE / README on upload_causal_ms)

U, C, S = E.U, E.C, E.S
U.UNITS = [
    ("P1", "checkpoints/e8b_step_10", E.ck("e8b", 10)),
    ("P1", "configs", C / "configs_e1e8"),
    ("P2", "eval_real/e8b_step_10", S / "eval_real/e8b_step_10"),
    ("P2", "tensorboard/e8b", C / "e8b/tensorboard"),
    ("P2", "summaries", C / "summaries_e1e8"),
]
U.README = U.README.replace(
    "| E8-a | RM v3 0.8x4 (once) | fixed 64 steps | logged only |",
    "| E8-a | RM v3 0.8x4 (once) | fixed 64 steps | logged only |\n"
    "| E8-b | RM v3 0.8x4 (final: RM hit in the last chunk) | fixed 64 steps | logged only |",
).replace("E8-O s10 .732; E8-a s10 .664.", "E8-O s10 .732; E8-a s10 .664; E8-b s10 .692 (affordance .513 vs E8-a .427).")

if __name__ == "__main__":
    sys.exit(U.main())
