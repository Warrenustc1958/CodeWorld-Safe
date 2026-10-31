#!/bin/bash
# Run every 5 min by cws-e11-watch.timer: if E11 is not finished and cws-e11.service is not running
# (killed, failed, or stopped), start it again. It resumes from the latest checkpoint / skips finished eval cells.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/e11night/e11sam_eval_done ] && { echo "E11 finished, nothing to do"; exit 0; }
if ! systemctl is-active --quiet cws-e11.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-e11 not running ($(systemctl is-active cws-e11.service)), starting it" \
    >> $S/causal/e11night/pipeline.log
  systemctl start cws-e11.service
fi
