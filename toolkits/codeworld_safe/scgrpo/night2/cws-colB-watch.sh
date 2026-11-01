#!/bin/bash
# Run every 5 min by cws-colB-watch.timer: restart lane B if unfinished and its service is not running.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/colnight/laneB_done ] && { echo "lane B finished"; exit 0; }
if ! systemctl is-active --quiet cws-colB.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-colB not running, starting it" >> $S/causal/colnight/laneB.log
  systemctl start cws-colB.service
fi
