#!/bin/bash
# Run every 5 min by cws-colA-watch.timer: restart lane A if unfinished and its service is not running.
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/colnight/laneA_done ] && { echo "lane A finished"; exit 0; }
if ! systemctl is-active --quiet cws-colA.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-colA not running, starting it" >> $S/causal/colnight/laneA.log
  systemctl start cws-colA.service
fi
