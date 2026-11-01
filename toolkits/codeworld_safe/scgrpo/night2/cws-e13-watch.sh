#!/bin/bash
# Run every 5 min by cws-e13-watch.timer: if E13 is not finished and cws-e13.service is not running
# (killed, failed, or stopped), start it again. It resumes from the latest checkpoint / skips finished eval cells.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/e13night/e13_eval_done ] && { echo "E13 finished, nothing to do"; exit 0; }
if ! systemctl is-active --quiet cws-e13.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-e13 not running ($(systemctl is-active cws-e13.service)), starting it" \
    >> $S/causal/e13night/pipeline.log
  systemctl start cws-e13.service
fi
