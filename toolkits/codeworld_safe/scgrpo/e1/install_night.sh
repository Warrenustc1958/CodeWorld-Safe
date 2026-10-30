#!/bin/bash
# Install + start cws-night.service. Stops only our old bash supervisor (supervise_e7.sh); the running E7 driver
# launch_e7b.sh and its training keep running and are adopted by the pipeline.
set -e
S=/data_cf/cws_sam/v1/scgrpo
mkdir -p $S/causal/night
for p in $(ps -eo pid,args | grep "[b]ash supervise_e7.sh" | awk '{print $1}'); do kill $p && echo "stopped old supervisor $p"; done
cp $S/cws-night.service /etc/systemd/system/cws-night.service
systemctl daemon-reload
systemctl start cws-night.service
sleep 5
systemctl --no-pager --lines=0 status cws-night.service | head -5
tail -5 $S/causal/night/pipeline.log
# uninstall: systemctl stop cws-night; rm /etc/systemd/system/cws-night.service; systemctl daemon-reload
