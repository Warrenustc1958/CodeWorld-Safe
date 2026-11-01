#!/bin/bash
# Install and start the phase 2/3/5 pipeline (cws-rm2.service) and its 5-minute watchdog.
S=/data_cf/cws_sam/v1/scgrpo
cd $S
for f in rm2/*.py rm2_pipeline.sh; do sed -i 's/\r$//' $f; done
bash -n rm2_pipeline.sh || { echo "rm2_pipeline.sh syntax error"; exit 1; }
mkdir -p causal/rm2night
sed -e "s#E13 grader cost, fixed eta 0.5 (10 steps, then eval 10)#phase 2/3/5 RM retraining and GRPO validation#" \
    -e "s#ExecStart=.*#ExecStart=/bin/bash $S/rm2_pipeline.sh#; s#e13night#rm2night#g" cws-e13.service > cws-rm2.service
cat > cws-rm2-watch.sh <<EOF
#!/bin/bash
S=$S
[ -f \$S/causal/rm2night/rm2_done ] && { echo "rm2 finished"; exit 0; }
if ! systemctl is-active --quiet cws-rm2.service; then
  echo "[\$(date '+%m-%d %H:%M:%S')] cws-rm2 not running, starting it" >> \$S/causal/rm2night/pipeline.log
  systemctl start cws-rm2.service
fi
EOF
sed "s#e13#rm2#g; s#E13#rm2#g" cws-e13-watch.service > cws-rm2-watch.service
sed "s#e13#rm2#g; s#E13#rm2#g" cws-e13-watch.timer > cws-rm2-watch.timer
grep -H "ExecStart\|StandardOutput" cws-rm2.service cws-rm2-watch.service
cp cws-rm2.service cws-rm2-watch.service cws-rm2-watch.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now cws-rm2.service cws-rm2-watch.timer
sleep 5
echo "rm2 $(systemctl is-active cws-rm2.service) timer $(systemctl is-active cws-rm2-watch.timer)"
tail -2 causal/rm2night/pipeline.log
