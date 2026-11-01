#!/bin/bash
# Create the E13 config (E10-O + grader cost labels, fixed eta 0.5, 10 steps) and its systemd units, then start.
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
sed -i 's/\r$//' e13_pipeline.sh
bash -n e13_pipeline.sh || { echo "e13_pipeline.sh syntax error"; exit 1; }
F=$C/libero_safety_scgrpo_causal_e13.yaml
cp $C/libero_safety_scgrpo_causal_e10o.yaml $F
sed -i -e "s#causal/e10o#causal/e13#; s#scgrpo_causal_e10o#scgrpo_causal_e13#" \
  -e "s#^  max_steps: 40#  max_steps: 10#; s#^  resume_dir: .*#  resume_dir: null#" \
  -e "s#^    eta_init: 0.0#    eta_init: 0.5   \# fixed (eta_lr 0): real grader cost enters the reward as -0.5 C#" \
  -e "s#^\# Real goal reward; Wan observations and state reader retained.#\# E13: E10-O + real grader cost labels (c_obj/c_robot/c_grasp, goal window), fixed eta 0.5.#" \
  -e "/^    success_reward_rule: once/a\\    cost_source: real\\n    cost_label: grader\\n    cost_window: goal" $F
diff $C/libero_safety_scgrpo_causal_e10o.yaml $F
mkdir -p causal/e13night
sed -e "s#E12-R real observations (20 steps, then eval 10/20)#E13 grader cost, fixed eta 0.5 (10 steps, then eval 10)#" \
    -e "s#e12_pipeline#e13_pipeline#; s#e12night#e13night#g" cws-e12.service > cws-e13.service
sed "s#e12#e13#g; s#E12-R#E13#g; s#E12#E13#g" cws-e12-watch.service > cws-e13-watch.service
sed "s#e12#e13#g; s#E12-R#E13#g; s#E12#E13#g" cws-e12-watch.timer > cws-e13-watch.timer
sed -e "s#e12night/e12r_eval_done#e13night/e13_eval_done#; s#e12#e13#g; s#E12-R#E13#g" cws-e12-watch.sh > cws-e13-watch.sh
grep -H "ExecStart\|Description" cws-e13.service cws-e13-watch.service cws-e13-watch.timer
grep -n "done\|systemctl" cws-e13-watch.sh
cp cws-e13.service cws-e13-watch.service cws-e13-watch.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now cws-e13.service cws-e13-watch.timer
sleep 45
echo "e13 $(systemctl is-active cws-e13.service), timer $(systemctl is-active cws-e13-watch.timer)"
tail -3 causal/e13night/pipeline.log
