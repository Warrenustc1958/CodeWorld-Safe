#!/bin/bash
# E1: train R1-all on GPUs 0-3, then eval step 10 (GPUs 0-1) and step 20 (GPUs 2-3) in parallel
S=/data_cf/cws_sam/v1/scgrpo
cd $S
TAG=r1all PORT=6501 bash run_causal.sh > causal/train_r1all.log 2>&1
echo "TRAIN $(tail -1 causal/train_r1all.log)"
CK=$S/causal/r1all/scgrpo_causal_r1all/checkpoints
W=actor/model_state_dict/full_weights.pt
( [ -f $CK/global_step_10/$W ] && NAME=r1all_step_10 CKPT=$CK/global_step_10/$W GPUS=0-1 bash eval_real.sh > causal/e1_eval_step10.log 2>&1 ) &
( [ -f $CK/global_step_20/$W ] && NAME=r1all_step_20 CKPT=$CK/global_step_20/$W GPUS=2-3 bash eval_real_ev2.sh > causal/e1_eval_step20.log 2>&1 ) &
wait
cd diag && /mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python matched_summ.py r1all_step_10 r1all_step_20 > $S/causal/e1_summary.txt 2>&1
echo E1_DONE
