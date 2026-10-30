#!/bin/bash
# Delete checkpoints of runs whose real-env eval is clearly below base (user-approved 10-30).
# Kept: R1 10/20, R1-all 5/10/15/20, E2 10/20, R0b8 5/10. Eval results (eval_real/) are untouched.
S=/data_cf/cws_sam/v1/scgrpo
df -h /data_cf | tail -1
for d in \
  $S/runs/scgrpo_pi05_wan41k_main/checkpoints/global_step_40 \
  $S/runs/scgrpo_pi05_wan41k_main/checkpoints/global_step_280 \
  $S/ablate/a/scgrpo_ablate_a/checkpoints/global_step_40 \
  $S/ablate/b/scgrpo_ablate_b/checkpoints/global_step_40 \
  $S/ablate/c/scgrpo_ablate_c/checkpoints/global_step_40 \
  $S/ablate/smoke/scgrpo_ablate_b/checkpoints/global_step_1 \
  $S/causal/rmall/scgrpo_causal_rmall/checkpoints/global_step_5 \
  $S/causal/rmall/scgrpo_causal_rmall/checkpoints/global_step_10 \
  $S/causal/rmall/scgrpo_causal_rmall/checkpoints/global_step_15 \
  $S/causal/rmall/scgrpo_causal_rmall/checkpoints/global_step_20 \
  $S/causal/rmall/scgrpo_causal_rmall/checkpoints/global_step_40; do
  case "$d" in */checkpoints/global_step_[0-9]*) ;; *) echo "refuse $d"; continue ;; esac
  if [ -d "$d/actor" ]; then rm -rf -- "$d" && echo "deleted $d"; else echo "skip (missing) $d"; fi
done
df -h /data_cf | tail -1
