#!/bin/bash
# Run every 5 min by cws-e10-watch.timer: if E10-O is not finished and cws-e10.service is not running
# (killed, failed, or stopped), start it again. It resumes from the latest checkpoint / skips finished eval cells.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/e10night/e10o_eval_done ] && { echo "E10-O finished, nothing to do"; exit 0; }
if ! systemctl is-active --quiet cws-e10.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-e10 not running ($(systemctl is-active cws-e10.service)), starting it" \
    >> $S/causal/e10night/pipeline.log
  systemctl start cws-e10.service
fi
