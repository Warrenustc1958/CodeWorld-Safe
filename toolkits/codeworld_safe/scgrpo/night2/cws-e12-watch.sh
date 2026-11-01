#!/bin/bash
# Run every 5 min by cws-e12-watch.timer: if E12-R is not finished and cws-e12.service is not running
# (killed, failed, or stopped), start it again. It resumes from the latest checkpoint / skips finished eval cells.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/e12night/e12r_eval_done ] && { echo "E12-R finished, nothing to do"; exit 0; }
if ! systemctl is-active --quiet cws-e12.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-e12 not running ($(systemctl is-active cws-e12.service)), starting it" \
    >> $S/causal/e12night/pipeline.log
  systemctl start cws-e12.service
fi
