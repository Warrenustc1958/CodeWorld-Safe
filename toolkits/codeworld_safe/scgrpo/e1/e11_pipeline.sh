#!/bin/bash
# E11 (run by systemd cws-e11.service, Restart=on-failure, KillMode=process; markers in causal/e11night/):
#   wait for E10-O training (TRAIN_DONE e10o) ->
#   E11-SC  train 20 steps (real event cost, global eta), 8 GPUs, keep steps 10/20 + latest ->
#   E11-SAM train 20 steps (SAM cost, global eta), same settings ->
#   evals of steps 10/20: E11-SC after E10-O's eval (chains p26-p31), E11-SAM after E11-SC's eval (p20-p25),
#   both on GPUs 0,1,2,5,6,7 (GPUs 3/4 have EGL faults). Every stage resumes from the latest checkpoint.
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/e11night
mkdir -p $N
cd $S
GPUS6="0 1 2 5 6 7"
ck() { echo $S/causal/$1/scgrpo_causal_$1/checkpoints; }
train() {  # TAG PORT WORKER_RANGE
  local tag=$1
  has ${tag}_train_done && return 0
  if grep -qs "TRAIN_DONE $tag" causal/${tag}_driver.log; then mark ${tag}_train_done; return 0; fi
  alive "launch_train.sh $tag" && return 0
  log "start/resume $tag training"
  ( TAG=$tag PORT=$2 WR=$3 KEEP="10 20" KEEP_LATEST=1 nohup setsid bash launch_train.sh $tag \
      >> causal/${tag}_driver.log 2>&1 < /dev/null & )
  sleep 30
}
evalrun() {  # TAG CHAINS
  local tag=$1 c; local K=$(ck $tag) J=""
  for s in 10 20; do [ -f $K/global_step_$s/$W ] && J="$J ${tag}_step_$s|$K/global_step_$s/$W"; done
  [ -n "$J" ] || { log "$tag: no checkpoints to evaluate"; mark ${tag}_eval_done; return 0; }
  ensure_eval ${tag}_eval $(( $(echo $J | wc -w) * 60 )) "$2" "$GPUS6" $J
}
log "e11 pipeline start pid $$"
while true; do
  if ! grep -qs "TRAIN_DONE e10o" causal/e10o_driver.log; then sleep 60; continue; fi
  train e11sc 6701 58000-59999
  has e11sc_train_done && train e11sam 6711 56000-57999
  has e11sc_train_done && [ -f $S/causal/e10night/e10o_eval_done ] && evalrun e11sc "26 27 28 29 30 31"
  has e11sam_train_done && has e11sc_eval_done && evalrun e11sam "20 21 22 23 24 25"
  if has e11sam_eval_done; then log "E11 done (SC + SAM, train + eval)"; exit 0; fi
  sleep 60
done
