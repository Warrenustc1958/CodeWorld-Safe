#!/bin/bash
# Create the M0-M3 configs, the M1 initial-state keyframe weights, systemd units for lanes 1/2, and start them.
set -u
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
for f in measure_pipeline.sh rm2/measure_report.py; do sed -i 's/\r$//' $f; done
bash -n measure_pipeline.sh || { echo "measure_pipeline.sh syntax error"; exit 1; }
/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python -c "
import numpy as np
t0 = np.load('$S/keyframes_v1/meta.npz')['t0']
w = (t0 == 8).astype(np.float64); np.save('$S/kf_scan/kf_t0_8.npy', w); print('t0==8 keyframes', int(w.sum()))"
# tag | steps | noise | initial-state (1/0)
for j in "m0|4|0.5|0" "m1|3|0.5|1" "m2|4|0.8|0" "m3|4|1.0|0"; do
  IFS='|' read tag steps noise init <<< "$j"
  F=$C/libero_safety_scgrpo_causal_$tag.yaml
  cp $C/libero_safety_scgrpo_causal_e10o.yaml $F
  sed -i -e "s#causal/e10o#causal/$tag#; s#scgrpo_causal_e10o#scgrpo_causal_$tag#" \
    -e "s#^  max_steps: 40#  max_steps: $steps#; s#^  save_interval: 2#  save_interval: 1000#" \
    -e "s#^  resume_dir: .*#  resume_dir: null#; s#^    lr: 5.0e-6#    lr: 0.0   \# measurement: policy frozen#" \
    -e "s#^    seed: 0#    seed: 301#; s#^      noise_level: 0.5#      noise_level: $noise#" \
    -e "s#^      from_pretrained: .*#      from_pretrained: $S/rm2/ra/best.pt#" \
    -e "s#^    success_threshold: .*#    success_threshold: 0.877#; s#^    success_consecutive: .*#    success_consecutive: 1#" \
    -e "s#^\# Real goal reward; Wan observations and state reader retained.#\# Variance measurement $tag (pre-registered): base pi0.5 frozen, noise $noise, initial-state $init; R-A logged only.#" \
    -e "/^    success_reward_rule: once/a\\    rm_frames: [3, 7]\\n    cost_source: real\\n    cost_label: grader\\n    dump_dir: $S/causal/$tag/dump" $F
  if [ "$init" = 1 ]; then
    sed -i -e "s#^    max_chunks: 8#    max_chunks: 32#; s#^    max_episode_steps: 64#    max_episode_steps: 256#" \
      -e "s#^    max_steps_per_rollout_epoch: 64#    max_steps_per_rollout_epoch: 256#; s#^    success_tail_chunks: 8#    success_tail_chunks: 32#" \
      -e "/^    dump_dir: /a\\    keyframe_sampling: weighted\\n    keyframe_weights: $S/kf_scan/kf_t0_8.npy" $F
  fi
  mkdir -p causal/$tag/dump
  echo "== $tag"; grep -nE "max_steps:|save_interval|lr: 0.0|^    seed:|noise_level|from_pretrained|success_threshold|success_consecutive|max_chunks|max_episode_steps|max_steps_per_rollout|success_tail|rm_frames|keyframe_|dump_dir|reward_source" $F
done
mkdir -p causal/msnight
for L in 1 2; do
  sed -e "s#E13 grader cost, fixed eta 0.5 (10 steps, then eval 10)#variance measurement lane $L#" \
      -e "s#ExecStart=.*#ExecStart=/bin/bash $S/measure_pipeline.sh $L#; s#e13night#msnight#g; s#pipeline.log#lane$L.log#g" \
      cws-e13.service > cws-ms$L.service
  cat > cws-ms$L-watch.sh <<EOF
#!/bin/bash
S=$S
[ -f \$S/causal/msnight/lane${L}_done ] && { echo "lane $L finished"; exit 0; }
if ! systemctl is-active --quiet cws-ms$L.service; then
  echo "[\$(date '+%m-%d %H:%M:%S')] cws-ms$L not running, starting it" >> \$S/causal/msnight/lane$L.log
  systemctl start cws-ms$L.service
fi
EOF
  sed "s#e13#ms$L#g; s#E13#ms$L#g" cws-e13-watch.service > cws-ms$L-watch.service
  sed "s#e13#ms$L#g; s#E13#ms$L#g" cws-e13-watch.timer > cws-ms$L-watch.timer
  cp cws-ms$L.service cws-ms$L-watch.service cws-ms$L-watch.timer /etc/systemd/system/
done
systemctl daemon-reload
systemctl enable --now cws-ms1.service cws-ms1-watch.timer cws-ms2.service cws-ms2-watch.timer
sleep 40
echo "ms1 $(systemctl is-active cws-ms1.service) ms2 $(systemctl is-active cws-ms2.service)"
tail -2 causal/msnight/lane1.log causal/msnight/lane2.log
df -h /data_cf | tail -1
