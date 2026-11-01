#!/bin/bash
# D2 = E14: E10-O + random reward-label noise (fp 0.05, fn 0.14), grader labels logged (eta 0), 10 steps, eval step 10.
# D3 = E13-D: continue E13 from step 10 for 2 steps with the per-episode dump (no eval).
# Both dump one JSON line per episode under causal/<tag>/dump/. Creates configs, pipelines, systemd units; starts both.
set -u
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
# ---- configs
F=$C/libero_safety_scgrpo_causal_e14.yaml
cp $C/libero_safety_scgrpo_causal_e10o.yaml $F
sed -i -e "s#causal/e10o#causal/e14#; s#scgrpo_causal_e10o#scgrpo_causal_e14#" \
  -e "s#^  max_steps: 40#  max_steps: 10#; s#^  resume_dir: .*#  resume_dir: null#" \
  -e "s#^\# Real goal reward; Wan observations and state reader retained.#\# E14 (D2): E10-O + random reward-label noise fp 0.05 / fn 0.14; grader labels logged only (eta 0).#" \
  -e "/^    success_reward_rule: once/a\\    cost_source: real\\n    cost_label: grader\\n    reward_noise: {fp: 0.05, fn: 0.14}\\n    dump_dir: $S/causal/e14/dump" $F
F=$C/libero_safety_scgrpo_causal_e13d.yaml
cp $C/libero_safety_scgrpo_causal_e13.yaml $F
sed -i -e "s#causal/e13\"#causal/e13d\"#; s#scgrpo_causal_e13\"#scgrpo_causal_e13d\"#" \
  -e "s#^  max_steps: 10#  max_steps: 12#; s#^  save_interval: 2#  save_interval: 100#; s#^  resume_dir: .*#  resume_dir: null#" \
  -e "s#^\# E13: #\# E13-D (D3): E13 continued from step 10 for 2 steps with the per-episode dump. #" \
  -e "/^    cost_window: goal/a\\    dump_dir: $S/causal/e13d/dump" $F
grep -n "log_path\|experiment_name\|max_steps\|save_interval\|reward_noise\|dump_dir\|cost_label\|eta_init" \
  $C/libero_safety_scgrpo_causal_e14.yaml $C/libero_safety_scgrpo_causal_e13d.yaml
# ---- E13-D starts from E13's step-10 checkpoint
mkdir -p causal/e13d/scgrpo_causal_e13d/checkpoints causal/e14/dump causal/e13d/dump
ln -sfn $S/causal/e13/scgrpo_causal_e13/checkpoints/global_step_10 \
  causal/e13d/scgrpo_causal_e13d/checkpoints/global_step_10
ls -la causal/e13d/scgrpo_causal_e13d/checkpoints/
# ---- pipelines from the E13 one
sed -e "s#^TAG=e13#TAG=e14#; s#PORT=6801 WR=58000-59999#PORT=6801 WR=58000-59999#" \
    -e "s#e13night#e14night#g; s#E13 done#E14 done#; s#e13 pipeline start#e14 pipeline start#" \
    -e "s#^\# E13 (run by systemd cws-e13.service#\# E14 / D2 (run by systemd cws-e14.service#" e13_pipeline.sh > e14_pipeline.sh
sed -e "s#^TAG=e13#TAG=e13d#; s#PORT=6801 WR=58000-59999#PORT=6791 WR=56000-57999#" \
    -e "s#e13night#e13dnight#g; s#E13 done (train + eval)#E13-D done (train)#; s#e13 pipeline start#e13d pipeline start#" \
    -e "s#^\# E13 (run by systemd cws-e13.service#\# E13-D / D3 (run by systemd cws-e13d.service; train only#" \
    -e "s#^  if has \${TAG}_train_done \&\& ! has \${TAG}_eval_done; then#  if has \${TAG}_train_done \&\& ! has \${TAG}_eval_done; then mark \${TAG}_eval_done; fi\n  if false; then#" \
    e13_pipeline.sh > e13d_pipeline.sh
bash -n e14_pipeline.sh && bash -n e13d_pipeline.sh && echo PIPELINES_OK
grep -n "^TAG=\|PORT=\|night\|eval_done\|if false" e14_pipeline.sh e13d_pipeline.sh | head -20
# ---- systemd units from the E13 ones
for t in e14 e13d; do
  sed -e "s#E13 grader cost, fixed eta 0.5 (10 steps, then eval 10)#$t (D2/D3 diagnostics)#" \
      -e "s#e13_pipeline#${t}_pipeline#; s#e13night#${t}night#g" cws-e13.service > cws-$t.service
  sed "s#e13#$t#g; s#E13#$t#g" cws-e13-watch.service > cws-$t-watch.service
  sed "s#e13#$t#g; s#E13#$t#g" cws-e13-watch.timer > cws-$t-watch.timer
  sed -e "s#e13night/e13_eval_done#${t}night/${t}_eval_done#; s#e13#$t#g; s#E13#$t#g" cws-e13-watch.sh > cws-$t-watch.sh
  mkdir -p causal/${t}night
  cp cws-$t.service cws-$t-watch.service cws-$t-watch.timer /etc/systemd/system/
done
grep -H "ExecStart" cws-e14.service cws-e13d.service cws-e14-watch.service
grep -H "done" cws-e14-watch.sh cws-e13d-watch.sh
systemctl daemon-reload
systemctl enable --now cws-e14.service cws-e14-watch.timer cws-e13d.service cws-e13d-watch.timer
sleep 50
echo "e14 $(systemctl is-active cws-e14.service) e13d $(systemctl is-active cws-e13d.service)"
tail -2 causal/e14night/pipeline.log causal/e13dnight/pipeline.log
