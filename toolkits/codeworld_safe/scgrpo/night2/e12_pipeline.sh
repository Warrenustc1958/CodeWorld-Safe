#!/bin/bash
# E12-R (run by systemd cws-e12.service, Restart=on-failure, KillMode=process; markers in causal/e12night/):
#   E12-R train 10 steps (E10-O with real observations, osmesa rendering), 8 GPUs, keep step 10 + latest ->
#   eval of step 10 on GPUs 0,1,2,5,6,7 (chains p26-p31). Every stage resumes (latest checkpoint / done cells).
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/e12night
mkdir -p $N
cd $S
GPUS6="0 1 2 5 6 7"
TAG=e12r
K=$S/causal/$TAG/scgrpo_causal_$TAG/checkpoints
launcher_alive() {  # a launch_train.sh started with TAG=e12r (in its environment, not its arguments)
  local d
  for d in /proc/[0-9]*; do
    tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q "launch_train.sh" || continue
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "TAG=$TAG" && return 0
  done
  return 1
}
log "e12 pipeline start pid $$"
while true; do
  if ! has ${TAG}_train_done; then
    if grep -qs "TRAIN_DONE $TAG" causal/${TAG}_driver.log; then mark ${TAG}_train_done
    elif ! launcher_alive; then
      log "start/resume $TAG training"
      ( TAG=$TAG PORT=6791 WR=56000-57999 KEEP="10" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
          >> causal/${TAG}_driver.log 2>&1 < /dev/null & )
      sleep 30
    fi
  fi
  if has ${TAG}_train_done && ! has ${TAG}_eval_done; then
    J=""
    for s in 10; do [ -f $K/global_step_$s/$W ] && J="$J ${TAG}_step_$s|$K/global_step_$s/$W"; done
    if [ -z "$J" ]; then log "$TAG: no checkpoints to evaluate"; mark ${TAG}_eval_done
    else ensure_eval ${TAG}_eval $(( $(echo $J | wc -w) * 60 )) "26 27 28 29 30 31" "$GPUS6" $J; fi
  fi
  if has ${TAG}_eval_done; then log "E12-R done (train + eval)"; exit 0; fi
  sleep 60
done
