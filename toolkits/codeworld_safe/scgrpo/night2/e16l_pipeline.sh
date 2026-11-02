#!/bin/bash
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/e16lnight; cd $S
TAG=e16l; K=$S/causal/$TAG/scgrpo_causal_$TAG/checkpoints
log "e16l pipeline start pid $$"
while true; do
  if ! has ${TAG}_train_done; then
    if grep -qs "TRAIN_DONE $TAG" causal/${TAG}_driver.log; then mark ${TAG}_train_done
    else
      alive=0
      for d in /proc/[0-9]*; do
        tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q launch_train.sh || continue
        tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx TAG=$TAG && { alive=1; break; }
      done 2>/dev/null
      if [ $alive = 0 ]; then
        log "start/resume $TAG"
        ( TAG=$TAG PORT=6871 WR=56000-57999 KEEP="10" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
            >> causal/${TAG}_driver.log 2>&1 < /dev/null & )
        sleep 60
      fi
    fi
  fi
  if has ${TAG}_train_done && ! has ${TAG}_eval_done; then
    ensure_eval ${TAG}_eval 60 "20 21 22 23 24 25" "0 1 2 5 6 7" "${TAG}_step_10|$K/global_step_10/$W"
  fi
  if has ${TAG}_eval_done; then log "E16-L done (train + eval)"; exit 0; fi
  sleep 60
done
