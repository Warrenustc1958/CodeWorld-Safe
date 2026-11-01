#!/bin/bash
# Phase 1: create the 7 collection configs, symlink their start checkpoints, install systemd units, start lanes A/B.
set -u
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
sed -i 's/\r$//' collect_pipeline.sh
bash -n collect_pipeline.sh || { echo "collect_pipeline.sh syntax error"; exit 1; }
# tag | source run | source step (0 = base policy) | env seed
JOBS="colbase|-|0|101 col10a|e10o|10|102 col10b|e10o|20|103 col10c|e10o|40|104 col12r|e12r|10|105 col13|e13|10|106 col11m|e11sam|20|107"
for j in $JOBS; do
  IFS='|' read tag src step seed <<< "$j"
  F=$C/libero_safety_scgrpo_causal_$tag.yaml
  cp $C/libero_safety_scgrpo_causal_e10o.yaml $F
  sed -i -e "s#causal/e10o#causal/$tag#; s#scgrpo_causal_e10o#scgrpo_causal_$tag#" \
    -e "s#^  max_steps: 40#  max_steps: $((step + 6))#; s#^  save_interval: 2#  save_interval: 1000#" \
    -e "s#^  resume_dir: .*#  resume_dir: null#; s#^    lr: 5.0e-6#    lr: 0.0   \# collection: policy frozen#" \
    -e "s#^    seed: 0#    seed: $seed#" \
    -e "s#^\# Real goal reward; Wan observations and state reader retained.#\# Phase-1 collection $tag: policy of $src step $step frozen (lr 0), 3072 episodes, dump with arrays.#" \
    -e "/^    success_reward_rule: once/a\\    cost_source: real\\n    cost_label: grader\\n    keyframe_sampling: weighted\\n    keyframe_weights: $S/kf_scan/kf_weights.npy\\n    keyframe_exclude: $S/kf_scan/kf_exclude.npy\\n    dump_dir: $S/causal/$tag/dump\\n    dump_arrays: True" $F
  mkdir -p causal/$tag/scgrpo_causal_$tag/checkpoints causal/$tag/dump
  if [ "$src" != "-" ]; then
    ln -sfn $S/causal/$src/scgrpo_causal_$src/checkpoints/global_step_$step \
      causal/$tag/scgrpo_causal_$tag/checkpoints/global_step_$step
  fi
  echo "$tag: $(grep -cE "lr: 0.0|seed: $seed|dump_arrays: True|keyframe_sampling: weighted|max_steps: $((step + 6))" $F)/5 edits," \
       "start $(ls causal/$tag/scgrpo_causal_$tag/checkpoints/ | tr '\n' ' ')"
done
mkdir -p causal/colnight
for L in A B; do
  sed -e "s#E13 grader cost, fixed eta 0.5 (10 steps, then eval 10)#phase-1 collection lane $L#" \
      -e "s#ExecStart=.*#ExecStart=/bin/bash $S/collect_pipeline.sh $L#; s#e13night#colnight#g" \
      -e "s#pipeline.log#lane$L.log#g" cws-e13.service > cws-col$L.service
  cat > cws-col$L-watch.sh <<EOF
#!/bin/bash
# Run every 5 min by cws-col$L-watch.timer: restart lane $L if unfinished and its service is not running.
S=$S
[ -f \$S/causal/colnight/lane${L}_done ] && { echo "lane $L finished"; exit 0; }
if ! systemctl is-active --quiet cws-col$L.service; then
  echo "[\$(date '+%m-%d %H:%M:%S')] cws-col$L not running, starting it" >> \$S/causal/colnight/lane$L.log
  systemctl start cws-col$L.service
fi
EOF
  sed "s#e13#col$L#g; s#E13#col$L#g" cws-e13-watch.service > cws-col$L-watch.service
  sed "s#e13#col$L#g; s#E13#col$L#g" cws-e13-watch.timer > cws-col$L-watch.timer
  cp cws-col$L.service cws-col$L-watch.service cws-col$L-watch.timer /etc/systemd/system/
done
grep -H "ExecStart\|StandardOutput" cws-colA.service cws-colB.service cws-colA-watch.service
systemctl daemon-reload
systemctl enable --now cws-colA.service cws-colA-watch.timer cws-colB.service cws-colB-watch.timer
sleep 45
echo "colA $(systemctl is-active cws-colA.service) colB $(systemctl is-active cws-colB.service)"
grep -vh "/proc/" causal/colnight/laneA.log causal/colnight/laneB.log | tail -4
