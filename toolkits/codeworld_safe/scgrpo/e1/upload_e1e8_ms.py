"""Upload the E1/E2/E4/E8 results (10-29/30) to ModelScope, reusing upload_causal_ms.py (hard-link staging,
4 GB split + SHA256SUMS, idempotent size-verified upload). Checkpoints: actor model_state_dict only
(enough for eval / deployment; the dcp_checkpoint optimizer state is not uploaded).
  MS_TOKEN_FILE=/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws /usr/bin/python3 upload_e1e8_ms.py [--dry-run] [--only P1|P2]
"""
import sys
from pathlib import Path

sys.path.insert(0, "/data_cf/cws_sam/v1/scgrpo")
import upload_causal_ms as U  # noqa: E402

S, RM = U.S, U.RM
C = S / "causal"
MSD = "actor/model_state_dict"


def ck(run, step):
    return C / run / f"scgrpo_causal_{run}" / "checkpoints" / f"global_step_{step}" / MSD


U.ROOT = "scgrpo_e1_e8_20261030"
U.STAGE = Path("/data_cf/cws_ms_stage_e1e8")
U.LOG = U.STAGE / "upload.log"
U.UNITS = [
    ("P1", "checkpoints/r1all_step_10", ck("r1all", 10)),
    ("P1", "checkpoints/r1all_step_20", ck("r1all", 20)),
    ("P1", "checkpoints/e2_step_10", ck("e2", 10)),
    ("P1", "checkpoints/e2_step_20", ck("e2", 20)),
    ("P1", "checkpoints/e8o_step_10", ck("e8o", 10)),
    ("P1", "checkpoints/e8a_step_10", ck("e8a", 10)),
    ("P1", "rm_models/rm_v6_d2", RM / "runs/rm_v6_d2"),
    ("P1", "rm_data/frames_d2", RM / "frames_d2"),
    ("P1", "configs", C / "configs_e1e8"),
] + [
    ("P2", f"eval_real/{n}", S / "eval_real" / n)
    for n in ("off_base", "r1all_step_10", "r1all_step_20", "rmall_step_10", "rmall_step_20", "rmall_step_40",
              "e2_step_10", "e2_step_20", "e8o_step_10", "e8a_step_10")
] + [
    ("P2", f"tensorboard/{n}", C / n / "tensorboard") for n in ("r1all", "rmall", "e2", "e8o", "e8a")
] + [
    ("P2", "summaries", C / "summaries_e1e8"),
    ("P2", "gates", RM / "runs/gates_e4"),
]
U.README = """# SC-GRPO E1/E2/E4/E8 (2026-10-29/30)

All runs: pi0.5 SC-GRPO in the frozen Wan 41k world model, all four LIBERO-Safety suites, uniform keyframe
sampling, eta = 0, 64-step episodes, SigLIP kept in eval mode. Real eval = official LIBERO-Safety grader,
60 cells x 20 episodes (eval_real/<run>/<cell>/...).

| run | reward | episode end | violation |
|---|---|---|---|
| R1-all | real goal | real success / violation / 64 | ends episode |
| RM-all | RM v3 0.8x4 | RM success / violation / 64 | ends episode |
| E2 | RM v3 0.8x4 | real success / violation / 64 | ends episode |
| E8-O | real goal (once) | fixed 64 steps | logged only |
| E8-a | RM v3 0.8x4 (once) | fixed 64 steps | logged only |

Overall official SR (base .703): R1-all s10 .724 / s20 .712; RM-all s10 .696 / s20 .642 / s40 .609;
E2 s10 .722 / s20 .711; E8-O s10 .732; E8-a s10 .664. See summaries/.

- checkpoints/: actor model_state_dict (full_weights.pt split in 4 GB parts:
  `cat full_weights.pt.part_* > full_weights.pt; sha256sum -c SHA256SUMS`). RM-all checkpoints were deleted.
- rm_models/rm_v6_d2: RM v6 (v5 recipe + D2 affordance hard negatives); did not pass the E4 gate.
- rm_data/frames_d2: D2 on-policy affordance rollouts as RMFrames (d2train / d2test by origin; note: 1 source
  group shared between d2train and d2test, 5 d2test groups also in frames_v5mix/poltrain).
- gates/: gate_d2.json (window definition) and gate_d2_v2.json (precision at the first RM decision, clean subset).
- configs/: the training configs (libero_safety_scgrpo_causal_*.yaml).
Code: github.com/Warrenustc1958/CodeWorld-Safe main (toolkits/codeworld_safe/scgrpo/DIAGNOSIS.md).
"""

if __name__ == "__main__":
    sys.exit(U.main())
