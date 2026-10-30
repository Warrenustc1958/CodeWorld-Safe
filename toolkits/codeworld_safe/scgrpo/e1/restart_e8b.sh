#!/bin/bash
# E8-b died in step 2 (EnvWorker on GPU 3, SYSTEM_ERROR, no checkpoint yet): stop its Ray head, keep the
# failed log, rerun the same config from scratch, then eval step 10 on GPUs 0/1/2 (chains p20-p22).
S=/data_cf/cws_sam/v1/scgrpo
W=actor/model_state_dict/full_weights.pt
cd $S
TAG=e8b bash stop_causal_head.sh  # no-op if already stopped
sleep 10
TS=$(date +%H%M)
[ -f causal/train_e8b.log ] && mv causal/train_e8b.log causal/train_e8b_failed_$TS.log
[ -d causal/e8b ] && mv causal/e8b causal/e8b_failed_$TS
busy=$(ss -ltn | awk '{print $4}' | grep -oE '[0-9]+$' | awk '$1>=56000 && $1<=57999' | wc -l)
echo "ports 56000-57999 in use: $busy"
[ "$busy" -lt 50 ] || { echo "worker port range busy"; exit 3; }  # a few ephemeral ports from other jobs are fine
for p in 6651 9651; do ss -ltn | grep -q ":$p " && { echo "port $p busy"; exit 3; }; done
(
  RAY_EXTRA="--min-worker-port=56000 --max-worker-port=57999" TAG=e8b PORT=6651 bash run_causal.sh > causal/train_e8b.log 2>&1
  rc=$?; echo "[$(date +%H:%M:%S)] E8-b (retry) TRAIN rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  CK=$S/causal/e8b/scgrpo_causal_e8b/checkpoints
  JOBS="e8b_step_10|$CK/global_step_10/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
  echo "[$(date +%H:%M:%S)] E8-b EVAL_LAUNCHED"
) > causal/e8b_retry.log 2>&1 < /dev/null &
sleep 1; echo E8B_RESTARTED
