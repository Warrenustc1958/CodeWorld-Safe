#!/bin/bash
# After E8-O training ends: start E8-b on GPUs 0-3 right away; move E8-O's eval to GPUs 5/6/7 after E8-a's
# eval (GPUs 0-3 are busy with E8-b training; training ~64 GB + eval ~20 GB would not fit on one card).
#   E8-b = E8-a with success_reward_rule: final (RM v3 0.8x4, fixed 64-step horizon), 10 steps, save 5,
#          eval step 10 on GPUs 0/1/2 (chains p20-p22) after its training.
#   E8-O eval: chains p29-p31 on GPUs 5/6/7 once E8-a's chains p26-p28 are done.
# The original E8-O wrapper subshell (which would eval on GPUs 0-2) is stopped; E8-O training itself keeps
# running. Log: causal/auto_e8b.log
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
W=actor/model_state_dict/full_weights.pt
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }
# 1) stop only the e8o wrapper subshell (parent of run_causal.sh TAG=e8o), not the training
for q in $(ps -eo pid,args | grep "[b]ash run_causal.sh" | awk '{print $1}'); do
  tr '\0' '\n' < /proc/$q/environ 2>/dev/null | grep -qx "TAG=e8o" || continue
  pp=$(ps -o ppid= -p $q | tr -d ' ')
  ps -o args= -p $pp | grep -q "launch_e8.sh" && { kill -KILL $pp && log "stopped e8o wrapper $pp (training $q keeps running)"; }
done
# 2) E8-b config
sed -e 's|actor,env,rollout: 4-7|actor,env,rollout: 0-3|' \
    -e 's|causal/e8a"|causal/e8b"|' -e 's|"scgrpo_causal_e8a"|"scgrpo_causal_e8b"|' \
    -e 's|success_reward_rule: once|success_reward_rule: final|' \
    -e 's|^# E8 e8a: .*|# E8 e8b: RM v3 0.8x4 reward with success_reward_rule final, fixed 64-step horizon, GPUs 0-3|' \
    $C/libero_safety_scgrpo_causal_e8a.yaml > $C/libero_safety_scgrpo_causal_e8b.yaml
diff $C/libero_safety_scgrpo_causal_e8a.yaml $C/libero_safety_scgrpo_causal_e8b.yaml
grep -q "success_reward_rule: final" $C/libero_safety_scgrpo_causal_e8b.yaml || { log "e8b config failed"; exit 4; }
# 3) wait for E8-O training, free its head, start E8-b (reuses E8-O's worker-port range)
log "waiting for E8-O training"
until grep -q "^EXIT=" causal/train_e8o.log 2>/dev/null; do sleep 60; done
log "E8-O train $(grep "^EXIT=" causal/train_e8o.log)"
TAG=e8o bash stop_causal_head.sh
sleep 10
for p in 6651 9651; do ss -ltn | grep -q ":$p " && { log "port $p busy"; exit 3; }; done
(
  RAY_EXTRA="--min-worker-port=56000 --max-worker-port=57999" TAG=e8b PORT=6651 bash run_causal.sh > causal/train_e8b.log 2>&1
  rc=$?; log "E8-b TRAIN rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  CK=$S/causal/e8b/scgrpo_causal_e8b/checkpoints
  JOBS="e8b_step_10|$CK/global_step_10/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
  log "E8-b EVAL_LAUNCHED"
) &
log "E8-b training started"
# 4) E8-O eval on GPUs 5/6/7 after E8-a's eval
if grep -q "^EXIT=0" causal/train_e8o.log; then
  until grep -q "e8a EVAL_LAUNCHED" causal/e8a.log 2>/dev/null; do
    grep -q "e8a TRAIN rc=[1-9]" causal/e8a.log 2>/dev/null && break
    sleep 60
  done
  if grep -q "e8a EVAL_LAUNCHED" causal/e8a.log; then
    until [ $(cat causal/multi_p2{6,7,8}.log 2>/dev/null | grep -c CHAIN_DONE) -eq 3 ]; do sleep 60; done
  fi
  CK=$S/causal/e8o/scgrpo_causal_e8o/checkpoints
  JOBS="e8o_step_10|$CK/global_step_10/$W" GPUS_LIST="5 6 7" CHAINS="29 30 31" bash run_multi.sh
  log "E8-O EVAL_LAUNCHED"
fi
wait
