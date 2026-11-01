#!/bin/bash
S=/data_cf/cws_sam/v1/scgrpo
[ -f $S/causal/rm2night/rm2_done ] && { echo "rm2 finished"; exit 0; }
if ! systemctl is-active --quiet cws-rm2.service; then
  echo "[$(date '+%m-%d %H:%M:%S')] cws-rm2 not running, starting it" >> $S/causal/rm2night/pipeline.log
  systemctl start cws-rm2.service
fi
