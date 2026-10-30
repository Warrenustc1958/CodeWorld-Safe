#!/bin/bash
# User-approved (10-30 19:1x): delete E8-b step 5/6/8 and the unevaluated step-5 checkpoints of E8-a, E8-O,
# R1-all; refresh the E8 summary; upload E8-b (step-10 weights, eval, tensorboard, summary) to ModelScope.
# Kept: R0b8 step 5 (has real-env eval results).
S=/data_cf/cws_sam/v1/scgrpo
C=$S/causal
cd $S
df -h /data_cf | tail -1
for d in \
  $C/e8b/scgrpo_causal_e8b/checkpoints/global_step_5 \
  $C/e8b/scgrpo_causal_e8b/checkpoints/global_step_6 \
  $C/e8b/scgrpo_causal_e8b/checkpoints/global_step_8 \
  $C/e8a/scgrpo_causal_e8a/checkpoints/global_step_5 \
  $C/e8o/scgrpo_causal_e8o/checkpoints/global_step_5 \
  $C/r1all/scgrpo_causal_r1all/checkpoints/global_step_5; do
  case "$d" in */checkpoints/global_step_[0-9]*) ;; *) echo "refuse $d"; continue ;; esac
  if [ -d "$d/actor" ]; then rm -rf -- "$d" && echo "deleted $d"; else echo "skip (missing) $d"; fi
done
df -h /data_cf | tail -1
cd $S/diag
P=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python
{ CUDA_VISIBLE_DEVICES="" $P matched_summ.py e8o_step_10 e8a_step_10 e8b_step_10 r1all_step_10 e2_step_10 rmall_step_10
  CUDA_VISIBLE_DEVICES="" $P level_summ.py e8o_step_10 e8a_step_10 e8b_step_10; } 2>&1 | grep -vi warn > $C/summaries_e1e8/e8_step10_summary.txt
cp /mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config/libero_safety_scgrpo_causal_e8b*.yaml $C/configs_e1e8/
cd $S
MS_TOKEN_FILE=/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws nohup /usr/bin/python3 upload_e8b_ms.py > causal/upload_e8b.out 2>&1 < /dev/null &
echo "upload started pid $!"
