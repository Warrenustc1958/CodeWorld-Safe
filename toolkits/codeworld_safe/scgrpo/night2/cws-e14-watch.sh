#!/bin/bash
# Run every 5 min by cws-e14-watch.timer: if e14 is not finished and cws-e14.service is not running
# (killed, failed, or stopped), start it again. It resumes from the latest checkpoint / skips finished eval cells.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/e14night/e14_eval_done ] && { echo "e14 finished, nothing to do"; exit 0; }
if ! systemctl is-active --quiet cws-e14.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-e14 not running ($(systemctl is-active cws-e14.service)), starting it" \
    >> $S/causal/e14night/pipeline.log
  systemctl start cws-e14.service
fi
