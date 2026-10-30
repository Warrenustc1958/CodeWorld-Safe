#!/bin/bash
# E8-b eval: if training finishes (EXIT=0, step 10 saved) eval step 10; if the training process disappears
# without EXIT=0 (killed), eval the latest saved checkpoint (step 8) instead. GPUs 0/1/2, chains p20-p22.
S=/data_cf/cws_sam/v1/scgrpo
cd $S
CK=$S/causal/e8b/scgrpo_causal_e8b/checkpoints
W=actor/model_state_dict/full_weights.pt
alive() {
  for q in $(ps -eo pid,args | grep "[b]ash run_causal.sh" | awk '{print $1}'); do
    tr '\0' '\n' < /proc/$q/environ 2>/dev/null | grep -qx "TAG=e8bres" && return 0
  done
  return 1
}
while true; do
  if grep -q "^EXIT=0" causal/train_e8b.log && [ -f $CK/global_step_10/$W ]; then STEP=10; break; fi
  if ! alive; then
    sleep 20; grep -q "^EXIT=0" causal/train_e8b.log && [ -f $CK/global_step_10/$W ] && { STEP=10; break; }
    STEP=$(ls -d $CK/global_step_* | grep -oE "[0-9]+$" | sort -n | while read s; do [ -f $CK/global_step_$s/$W ] && echo $s; done | tail -1)
    echo "[$(date +%H:%M:%S)] training gone without EXIT=0 ($(grep -a "^EXIT" causal/train_e8b.log)), latest checkpoint step $STEP"
    break
  fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] evaluating e8b step $STEP"
TAG=e8bres bash stop_causal_head.sh
sleep 10
JOBS="e8b_step_${STEP}|$CK/global_step_${STEP}/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
echo "[$(date +%H:%M:%S)] E8-b step $STEP EVAL_LAUNCHED"
