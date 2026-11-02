#!/bin/bash
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/m1rnight; cd $S
has m1r_done && exit 0
log "m1r pipeline start pid $$"
until grep -qs "TRAIN_DONE m1r" causal/m1r_driver.log; do
  alive=0
  for d in /proc/[0-9]*; do
    tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q launch_train.sh || continue
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx TAG=m1r && { alive=1; break; }
  done 2>/dev/null
  if [ $alive = 0 ]; then
    log "start/resume m1r"
    ( TAG=m1r PORT=6851 WR=58000-59999 KEEP="" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
        >> causal/m1r_driver.log 2>&1 < /dev/null & )
    sleep 60
  fi
  sleep 60
done
for d in $S/causal/m1r/scgrpo_causal_m1r/checkpoints/global_step_*; do  # lr-0 final save = copy of base
  [ -d "$d" ] && [ ! -L "$d" ] || continue
  case "$d" in */causal/m1r/scgrpo_causal_m1r/checkpoints/global_step_[0-9]*) rm -rf -- "$d" && log "removed $d" ;; esac
done
log "m1r: $(cat causal/m1r/dump/*.jsonl 2>/dev/null | wc -l) episodes"
mark m1r_done
