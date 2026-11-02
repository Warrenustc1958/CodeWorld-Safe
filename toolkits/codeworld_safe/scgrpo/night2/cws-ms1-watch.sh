#!/bin/bash
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/msnight/lane1_done ] && { echo "lane 1 finished"; exit 0; }
if ! systemctl is-active --quiet cws-ms1.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-ms1 not running, starting it" >> $S/causal/msnight/lane1.log
  systemctl start cws-ms1.service
fi
