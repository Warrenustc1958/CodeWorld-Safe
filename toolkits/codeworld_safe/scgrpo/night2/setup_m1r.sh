#!/bin/bash
# Item 2: M1 with REAL rendered observations (M1R). Same start keyframes (t0 = 8), 256 steps, seed 301, noise 0.5,
# policy frozen, R-A logged; arrays dumped so frames can be re-scored offline by other RMs. 2 steps = 1024 episodes.
# One systemd service, no watchdog timer.
set -u
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
F=$C/libero_safety_scgrpo_causal_m1r.yaml
cp $C/libero_safety_scgrpo_causal_m1.yaml $F
sed -i -e "s#causal/m1#causal/m1r#g; s#scgrpo_causal_m1\"#scgrpo_causal_m1r\"#" \
  -e "s#^  max_steps: 3#  max_steps: 2#" \
  -e "s#^\# Variance measurement m1 #\# Item 2 m1r: M1 with real rendered observations (osmesa); #" \
  -e "/^    dump_dir: /a\\    obs_source: real\\n    dump_arrays: True" $F
grep -nE "log_path|experiment_name|max_steps:|seed: 301|noise_level|max_episode_steps|obs_source|dump_arrays|dump_dir|keyframe_weights|cost_source|from_pretrained|lr: 0.0" $F
grep -c "obs_source" $F
mkdir -p causal/m1r/dump causal/m1rnight
cat > m1r_pipeline.sh <<'EOF'
#!/bin/bash
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/m1rnight; cd $S
has m1r_done && exit 0
log "m1r pipeline start pid $$"
until grep -qs "TRAIN_DONE m1r" causal/m1r_driver.log; do
  alive=0
  for d in /proc/[0-9]*; do
    tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q launch_train.sh || continue
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx TAG=m1r && { alive=1; break; }
  done 2>/dev/null
  if [ $alive = 0 ]; then
    log "start/resume m1r"
    ( TAG=m1r PORT=6851 WR=58000-59999 KEEP="" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
        >> causal/m1r_driver.log 2>&1 < /dev/null & )
    sleep 60
  fi
  sleep 60
done
for d in $S/causal/m1r/scgrpo_causal_m1r/checkpoints/global_step_*; do  # lr-0 final save = copy of base
  [ -d "$d" ] && [ ! -L "$d" ] || continue
  case "$d" in */causal/m1r/scgrpo_causal_m1r/checkpoints/global_step_[0-9]*) rm -rf -- "$d" && log "removed $d" ;; esac
done
log "m1r: $(cat causal/m1r/dump/*.jsonl 2>/dev/null | wc -l) episodes"
mark m1r_done
EOF
sed -i 's/\r$//' m1r_pipeline.sh; bash -n m1r_pipeline.sh || { echo syntax; exit 1; }
cat > /etc/systemd/system/cws-m1r.service <<EOF
[Unit]
Description=CodeWorld-Safe item 2: M1 with real rendered observations - $S/m1r_pipeline.sh
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$S
Environment=HOME=/root
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/bin/bash $S/m1r_pipeline.sh
Restart=on-failure
RestartSec=300
KillMode=process
StandardOutput=append:$S/causal/m1rnight/pipeline.log
StandardError=append:$S/causal/m1rnight/pipeline.log
EOF
cp /etc/systemd/system/cws-m1r.service $S/cws-m1r.service
systemctl daemon-reload
systemctl start cws-m1r.service
sleep 20
echo "m1r $(systemctl is-active cws-m1r.service)"
grep -v /proc/ causal/m1rnight/pipeline.log | tail -2
df -h /data_cf | tail -1
