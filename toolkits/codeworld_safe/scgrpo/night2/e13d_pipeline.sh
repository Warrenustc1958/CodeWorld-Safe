#!/bin/bash
# E13-D / D3 (run by systemd cws-e13d.service; train only, Restart=on-failure, KillMode=process; markers in causal/e13dnight/):
#   E13 train 10 steps = E10-O (Wan observations, real goal reward) + grader cost labels, fixed eta 0.5,
#   8 GPUs, keep step 10 + latest -> eval of step 10 on GPUs 0,1,2,5,6,7 (chains p20-p25),
#   Every stage resumes. (E12-R evaluates after E13.)
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/e13dnight
mkdir -p $N
cd $S
GPUS6="0 1 2 5 6 7"
TAG=e13d
K=$S/causal/$TAG/scgrpo_causal_$TAG/checkpoints
launcher_alive() {  # a launch_train.sh started with TAG=e13 (in its environment, not its arguments)
  local d
  for d in /proc/[0-9]*; do
    tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q "launch_train.sh" || continue
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "TAG=$TAG" && return 0
  done 2>/dev/null
  return 1
}
log "e13d pipeline start pid $$"
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
  if has ${TAG}_train_done && ! has ${TAG}_eval_done; then mark ${TAG}_eval_done; fi
  if false; then
    J=""
    for s in 10; do [ -f $K/global_step_$s/$W ] && J="$J ${TAG}_step_$s|$K/global_step_$s/$W"; done
    if [ -z "$J" ]; then log "$TAG: no checkpoints to evaluate"; mark ${TAG}_eval_done
    else ensure_eval ${TAG}_eval $(( $(echo $J | wc -w) * 60 )) "20 21 22 23 24 25" "$GPUS6" $J; fi
  fi
  if has ${TAG}_eval_done; then log "E13-D done (train)"; exit 0; fi
  sleep 60
done
