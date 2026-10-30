#!/bin/bash
# E8 (uniform sampling, same as E1/E2), 10 steps, save every 5, eval step 10:
#   E8-O  GPUs 0-3: real reward, fixed 64-step horizon, violations logged only
#   E8-a  GPUs 4-7: RM v3 0.8x4 reward (once), otherwise identical, no real-sim info in the reward
# Both: outcome_order=step, success_reward_rule=once, success_tail_chunks=8.
# Stops the finished training heads (e2, rmall, rmall40) whose workers hold ports 56000-59999.
set -uo pipefail
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
for t in e2 rmall rmall40; do TAG=$t bash stop_causal_head.sh; done
sleep 5
busy=$(ss -ltn | awk '{print $4}' | grep -oE '[0-9]+$' | awk '$1>=56000 && $1<=59999' | wc -l)
echo "ports 56000-59999 in use: $busy"
[ "$busy" -eq 0 ] || { echo "worker port range still busy"; exit 3; }
for p in 6611 6631 9611 9631; do ss -ltn | grep -q ":$p " && { echo "port $p busy"; exit 3; }; done
mk() {  # $1 tag  $2 placement  $3 reward_source
  sed -e "s|actor,env,rollout: 4-7|actor,env,rollout: $2|" \
      -e "s|causal/rmall\"|causal/$1\"|" -e "s|\"scgrpo_causal_rmall\"|\"scgrpo_causal_$1\"|" \
      -e 's|max_steps: 20|max_steps: 10|' \
      -e "s|reward_source: rm|reward_source: $3|" \
      -e 's|termination_source: rm .*|termination_source: horizon        # fixed 64 steps, no early end|' \
      -e 's|terminate_on_real_violation: True .*|terminate_on_real_violation: False # violations logged only|' \
      -e 's|end_on_violation: True|end_on_violation: False\n    success_tail_chunks: 8\n    outcome_order: step\n    success_reward_rule: once|' \
      -e "s|^# E1 RM-all: .*|# E8 $1: fixed 64-step horizon, reward_source $3, uniform sampling, GPUs $2|" \
      $C/libero_safety_scgrpo_causal_rmall.yaml > $C/libero_safety_scgrpo_causal_$1.yaml
  echo "== $1"; diff $C/libero_safety_scgrpo_causal_rmall.yaml $C/libero_safety_scgrpo_causal_$1.yaml
  grep -q "termination_source: horizon" $C/libero_safety_scgrpo_causal_$1.yaml || { echo "$1 config failed"; exit 4; }
}
mk e8o 0-3 real
mk e8a 4-7 rm
W=actor/model_state_dict/full_weights.pt
run() {  # $1 tag  $2 port  $3 worker range  $4 eval GPUs  $5 eval chains
  cd $S
  RAY_EXTRA="--min-worker-port=${3%-*} --max-worker-port=${3#*-}" TAG=$1 PORT=$2 bash run_causal.sh > causal/train_$1.log 2>&1
  rc=$?
  echo "[$(date +%H:%M)] $1 TRAIN rc=$rc"
  [ $rc -eq 0 ] || return $rc
  CK=$S/causal/$1/scgrpo_causal_$1/checkpoints
  JOBS="${1}_step_10|$CK/global_step_10/$W" GPUS_LIST="$4" CHAINS="$5" bash run_multi.sh
  echo "[$(date +%H:%M)] $1 EVAL_LAUNCHED"
}
( run e8o 6611 56000-57999 "0 1 2" "20 21 22" ) > causal/e8o.log 2>&1 < /dev/null &
sleep 60
( run e8a 6631 58000-59999 "5 6 7" "26 27 28" ) > causal/e8a.log 2>&1 < /dev/null &
sleep 1; echo E8_LAUNCHED
