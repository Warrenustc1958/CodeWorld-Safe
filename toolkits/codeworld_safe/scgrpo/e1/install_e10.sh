#!/bin/bash
# Install the E10-O config and start cws-e10.service.
# uninstall: systemctl stop cws-e10; rm /etc/systemd/system/cws-e10.service; systemctl daemon-reload
set -e
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
[ -e $C/libero_safety_scgrpo_causal_e10o.yaml ] || cp $S/libero_safety_scgrpo_causal_e10o.yaml $C/
mkdir -p $S/causal/e10night
cp $S/cws-e10.service /etc/systemd/system/cws-e10.service
systemctl daemon-reload
systemctl start cws-e10.service
sleep 5
systemctl --no-pager --lines=0 status cws-e10.service | head -4
tail -3 $S/causal/e10night/pipeline.log
