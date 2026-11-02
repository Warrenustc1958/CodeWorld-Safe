#!/bin/bash
# C = E16-L: E10-O training (Wan observations, real goal reward, eta 0) from the initial-state keyframes (t0 = 8),
# 256-step episodes, 10 steps, then the official eval of step 10 on GPUs 0,1,2,5,6,7 (chains p20-p25).
# One systemd service (Restart=on-failure), no watchdog timer. Resumes from the latest checkpoint.
set -u
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
F=$C/libero_safety_scgrpo_causal_e16l.yaml
cp $C/libero_safety_scgrpo_causal_m1.yaml $F
sed -i -e "s#causal/m1\"#causal/e16l\"#; s#causal/m1/dump#causal/e16l/dump#; s#scgrpo_causal_m1\"#scgrpo_causal_e16l\"#" \
  -e "s#^  max_steps: 3#  max_steps: 10#; s#^  save_interval: 1000#  save_interval: 2#" \
  -e "s#^    lr: 0.0   \# measurement: policy frozen#    lr: 5.0e-6#; s#^    seed: 301#    seed: 0#" \
  -e "s#^\# Variance measurement m1 .*#\# E16-L (C): E10-O training from the initial-state keyframes (t0 = 8), 256-step episodes, real reward.#" $F
grep -nE "^#|log_path|experiment_name|max_steps:|save_interval|lr: 5|^    seed:|max_chunks|max_episode_steps|success_tail|reward_source|keyframe_weights|noise_level|dump_dir|obs_source" $F
mkdir -p causal/e16l/dump causal/e16lnight
cat > e16l_pipeline.sh <<'EOF'
#!/bin/bash
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/e16lnight; cd $S
TAG=e16l; K=$S/causal/$TAG/scgrpo_causal_$TAG/checkpoints
log "e16l pipeline start pid $$"
while true; do
  if ! has ${TAG}_train_done; then
    if grep -qs "TRAIN_DONE $TAG" causal/${TAG}_driver.log; then mark ${TAG}_train_done
    else
      alive=0
      for d in /proc/[0-9]*; do
        tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q launch_train.sh || continue
        tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx TAG=$TAG && { alive=1; break; }
      done 2>/dev/null
      if [ $alive = 0 ]; then
        log "start/resume $TAG"
        ( TAG=$TAG PORT=6871 WR=56000-57999 KEEP="10" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
            >> causal/${TAG}_driver.log 2>&1 < /dev/null & )
        sleep 60
      fi
    fi
  fi
  if has ${TAG}_train_done && ! has ${TAG}_eval_done; then
    ensure_eval ${TAG}_eval 60 "20 21 22 23 24 25" "0 1 2 5 6 7" "${TAG}_step_10|$K/global_step_10/$W"
  fi
  if has ${TAG}_eval_done; then log "E16-L done (train + eval)"; exit 0; fi
  sleep 60
done
EOF
sed -i 's/\r$//' e16l_pipeline.sh; bash -n e16l_pipeline.sh || { echo syntax; exit 1; }
cat > /etc/systemd/system/cws-e16l.service <<EOF
[Unit]
Description=CodeWorld-Safe C / E16-L: initial-state 256-step real-reward GRPO, 10 steps + eval - $S/e16l_pipeline.sh
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$S
Environment=HOME=/root
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/bin/bash $S/e16l_pipeline.sh
Restart=on-failure
RestartSec=300
KillMode=process
StandardOutput=append:$S/causal/e16lnight/pipeline.log
StandardError=append:$S/causal/e16lnight/pipeline.log
EOF
cp /etc/systemd/system/cws-e16l.service $S/cws-e16l.service
systemctl daemon-reload
systemctl start cws-e16l.service
sleep 20
echo "e16l $(systemctl is-active cws-e16l.service)"
grep -v /proc/ causal/e16lnight/pipeline.log | tail -2
