#!/bin/bash
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/msnight/lane2_done ] && { echo "lane 2 finished"; exit 0; }
if ! systemctl is-active --quiet cws-ms2.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-ms2 not running, starting it" >> $S/causal/msnight/lane2.log
  systemctl start cws-ms2.service
fi
