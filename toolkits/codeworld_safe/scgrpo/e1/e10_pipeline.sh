#!/bin/bash
# E10-O (real goal reward, eta=0, E7 parallel scale), 40 steps, run by systemd cws-e10.service
# (Restart=on-failure, KillMode=process). Order: train all 40 steps first (save every 2; keep steps 10/20/40 +
# latest), then real eval of steps 10, 20, 40 together (180 cells, chains p20-p25 on GPUs 0,1,2,5,6,7).
# Markers in causal/e10night/.
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/e10night
mkdir -p $N
cd $S
CK=$S/causal/e10o/scgrpo_causal_e10o/checkpoints
log "e10 pipeline start pid $$"
while true; do
  if ! has train_done; then
    if grep -qs "TRAIN_DONE e10o" causal/e10o_driver.log; then mark train_done
    elif ! alive "launch_train.sh e10o"; then
      log "start/resume E10-O training"
      ( TAG=e10o PORT=6691 WR=56000-57999 KEEP="10 20 40" KEEP_LATEST=1 nohup setsid bash launch_train.sh e10o \
          >> causal/e10o_driver.log 2>&1 < /dev/null & )
      sleep 30
    fi
  else
    J=""; for s in 10 20 40; do [ -f $CK/global_step_$s/$W ] && J="$J e10o_step_$s|$CK/global_step_$s/$W"; done
    ensure_eval e10o_eval $(( $(echo $J | wc -w) * 60 )) "20 21 22 23 24 25" "0 1 2 5 6 7" $J
    if has e10o_eval_done; then log "E10-O done (train + eval)"; exit 0; fi
  fi
  sleep 60
done
