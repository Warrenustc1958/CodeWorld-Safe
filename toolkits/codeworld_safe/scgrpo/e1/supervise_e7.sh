#!/bin/bash
# Keep E7 running until training finishes (user 10-30): if the E7 driver itself is gone, start launch_e7b.sh,
# which resumes from the latest complete checkpoint (config causal_e7.yaml is reused, not regenerated).
# Log: causal/supervise_e7.log
S=/data_cf/cws_sam/v1/scgrpo
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }
while true; do
  if grep -q "E7 EVAL_LAUNCHED" causal/e7_driver.log 2>/dev/null; then log "E7 done (eval launched)"; exit 0; fi
  if ! ps -eo args | grep -qE "^bash launch_e7b?\.sh"; then
    if grep -q "training finished (EXIT=0)" causal/e7_driver.log 2>/dev/null; then
      log "training finished but driver gone before eval; starting eval-only driver"
    else
      log "E7 driver not running; restarting from the latest checkpoint"
    fi
    nohup bash launch_e7b.sh >> causal/e7_driver.log 2>&1 < /dev/null &
    sleep 120
  fi
  sleep 60
done
