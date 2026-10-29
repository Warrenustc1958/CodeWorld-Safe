#!/bin/bash
# E1 RM arm: train RM-all on GPUs 4-7, then eval step 10 (GPUs 4-5) and step 20 (GPUs 6-7) in parallel
S=/data_cf/cws_sam/v1/scgrpo
cd $S
TAG=rmall PORT=6511 bash run_causal.sh > causal/train_rmall.log 2>&1
echo "TRAIN $(tail -1 causal/train_rmall.log)"
CK=$S/causal/rmall/scgrpo_causal_rmall/checkpoints
W=actor/model_state_dict/full_weights.pt
( [ -f $CK/global_step_10/$W ] && NAME=rmall_step_10 CKPT=$CK/global_step_10/$W GPUS=4-5 bash eval_real_ev3.sh > causal/e1rm_eval_step10.log 2>&1 ) &
( [ -f $CK/global_step_20/$W ] && NAME=rmall_step_20 CKPT=$CK/global_step_20/$W GPUS=6-7 bash eval_real_ev4.sh > causal/e1rm_eval_step20.log 2>&1 ) &
wait
cd diag && /mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python matched_summ.py rmall_step_10 rmall_step_20 > $S/causal/e1rm_summary.txt 2>&1
echo E1RM_DONE
