#!/bin/bash
# Install and start the E12-R pipeline (auto-resume training, then eval 10/20) and its 5-minute watchdog.
S=/data_cf/cws_sam/v1/scgrpo
mkdir -p $S/causal/e12night
cp $S/cws-e12.service $S/cws-e12-watch.service $S/cws-e12-watch.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now cws-e12.service cws-e12-watch.timer
sleep 3
echo "e12 $(systemctl is-active cws-e12.service), timer $(systemctl is-active cws-e12-watch.timer)"
tail -2 $S/causal/e12night/pipeline.log
