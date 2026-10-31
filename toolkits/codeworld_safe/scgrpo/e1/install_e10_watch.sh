#!/bin/bash
# Install the 5-minute watchdog timer for E10-O and set the service's own restart delay to 5 minutes.
# uninstall: systemctl disable --now cws-e10-watch.timer; rm /etc/systemd/system/cws-e10-watch.*; systemctl daemon-reload
set -e
S=/data_cf/cws_sam/v1/scgrpo
sed -i 's|^RestartSec=.*|RestartSec=300|' $S/cws-e10.service
cp $S/cws-e10.service $S/cws-e10-watch.service $S/cws-e10-watch.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now cws-e10-watch.timer
systemctl start cws-e10-watch.service      # first check now (starts cws-e10 if it is not running)
sleep 5
systemctl --no-pager list-timers cws-e10-watch.timer | head -3
systemctl is-active cws-e10.service
grep -n "RestartSec" /etc/systemd/system/cws-e10.service
tail -3 $S/causal/e10night/pipeline.log
