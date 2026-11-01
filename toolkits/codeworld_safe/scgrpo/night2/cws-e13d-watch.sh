#!/bin/bash
# Run every 5 min by cws-e13d-watch.timer: if e13d is not finished and cws-e13d.service is not running
# (killed, failed, or stopped), start it again. It resumes from the latest checkpoint / skips finished eval cells.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/e13dnight/e13d_eval_done ] && { echo "e13d finished, nothing to do"; exit 0; }
if ! systemctl is-active --quiet cws-e13d.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-e13d not running ($(systemctl is-active cws-e13d.service)), starting it" \
    >> $S/causal/e13dnight/pipeline.log
  systemctl start cws-e13d.service
fi
