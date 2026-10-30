#!/bin/bash
# E7: E8-b config (RM v3 0.8x4, success_reward_rule final, fixed 64-step horizon, violations logged only,
# outcome_order step, uniform sampling) on 8 GPUs, 80 steps. 128 envs x 4 rollout epochs = 512 trajectories
# per step (same as E8-b's 64 x 8). Save every 2 steps; keep only every 20th checkpoint + the 2 latest.
# Auto-resume from the latest complete checkpoint when the training is killed. After training: real eval of
# steps 20/40/60/80 on GPUs 0,1,2,5,6,7 (chains p20-p25). Log: causal/e7_driver.log
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
CK=$S/causal/e7/scgrpo_causal_e7/checkpoints
W=actor/model_state_dict/full_weights.pt
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }
sed -e 's|actor,env,rollout: 0-3|actor,env,rollout: 0-7|' \
    -e 's|causal/e8b"|causal/e7"|' -e 's|"scgrpo_causal_e8b"|"scgrpo_causal_e7"|' \
    -e 's|max_steps: 10|max_steps: 80|' -e 's|save_interval: 5|save_interval: 2|' \
    -e 's|total_num_envs: 64|total_num_envs: 128|' -e 's|rollout_epoch: 8|rollout_epoch: 4|' \
    -e 's|^# E8 e8b: .*|# E7: E8-b config on 8 GPUs, 80 steps, save every 2 (pruned to every 20th + latest)|' \
    $C/libero_safety_scgrpo_causal_e8b.yaml > $C/libero_safety_scgrpo_causal_e7.yaml
diff $C/libero_safety_scgrpo_causal_e8b.yaml $C/libero_safety_scgrpo_causal_e7.yaml
grep -q "max_steps: 80" $C/libero_safety_scgrpo_causal_e7.yaml && grep -q "rollout: 0-7" $C/libero_safety_scgrpo_causal_e7.yaml \
  || { log "config edit failed"; exit 4; }

complete() { [ -f $CK/global_step_$1/actor/dcp_checkpoint/.metadata ] && [ -f $CK/global_step_$1/$W ]; }
steps() { ls -d $CK/global_step_* 2>/dev/null | grep -oE "[0-9]+$" | sort -n; }
latest() { for s in $(steps); do complete $s && echo $s; done | tail -1; }
prune() {  # keep every 20th and the 2 latest complete checkpoints
  local keep="$(for s in $(steps); do complete $s && echo $s; done | tail -2)"
  for s in $(steps); do
    [ $((s % 20)) -eq 0 ] && continue
    echo "$keep" | grep -qx "$s" && continue
    complete $s || continue          # never touch one that may still be writing
    d=$CK/global_step_$s
    case "$d" in */checkpoints/global_step_[0-9]*) rm -rf -- "$d" && log "pruned step $s" ;; esac
  done
}
cleanup() {  # our own e7 Ray leftovers only
  local all=""
  for d in /proc/[0-9]*; do
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/rp_ce7" && all="$all ${d#/proc/}"
  done
  for q in $(ps -eo pid,args | grep "[/]tmp/rp_ce7/" | awk '{print $1}'); do all="$all $q"; done
  all=$(echo $all | tr ' ' '\n' | grep -vx "$$" | sort -u)
  [ -n "$all" ] && { kill -TERM $all 2>/dev/null; sleep 8; for q in $all; do kill -0 $q 2>/dev/null && kill -KILL $q 2>/dev/null; done; }
}
n=0
while true; do
  last=$(latest)
  if [ -n "$last" ]; then
    sed -i "s|resume_dir: .*|resume_dir: $CK/global_step_$last|" $C/libero_safety_scgrpo_causal_e7.yaml
  else
    sed -i "s|resume_dir: .*|resume_dir: null|" $C/libero_safety_scgrpo_causal_e7.yaml
  fi
  [ -f causal/train_e7.log ] && mv causal/train_e7.log causal/train_e7_try$n.log
  log "start (try $n) from step ${last:-0}"
  RAY_EXTRA="--min-worker-port=56000 --max-worker-port=57999" TAG=e7 PORT=6671 bash run_causal.sh > causal/train_e7.log 2>&1 &
  tp=$!
  while kill -0 $tp 2>/dev/null; do prune; sleep 60; done
  wait $tp; rc=$?
  prune
  if grep -q "^EXIT=0" causal/train_e7.log; then log "training finished (EXIT=0)"; break; fi
  n=$((n + 1))
  log "training died rc=$rc ($(grep -a '^EXIT' causal/train_e7.log)), latest checkpoint $(latest)"
  [ $n -ge 40 ] && { log "too many restarts, giving up"; exit 1; }
  [ -z "$(latest)" ] && [ $n -ge 3 ] && { log "3 failures before the first checkpoint, giving up"; exit 1; }
  cleanup
  sleep 20
done
cleanup
JOBS=""
for s in 20 40 60 80; do [ -f $CK/global_step_$s/$W ] && JOBS="$JOBS e7_step_$s|$CK/global_step_$s/$W"; done
log "eval jobs:$JOBS"
JOBS="$JOBS" GPUS_LIST="0 1 2 5 6 7" CHAINS="20 21 22 23 24 25" bash run_multi.sh
log "E7 EVAL_LAUNCHED"
