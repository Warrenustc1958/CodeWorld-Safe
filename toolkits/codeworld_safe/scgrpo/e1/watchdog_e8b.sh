#!/bin/bash
# E8-b watchdog (user 10-30): from the next restart on, save every 2 steps; if the E8-b training process
# disappears without EXIT=0, resume automatically from the latest checkpoint (stops only our own e8bres
# leftovers). When step 10 exists, the step-10 eval (GPUs 0/1/2, p20-p22) is started by the resume subshell.
# Log: causal/watchdog_e8b.log
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
CK=$S/causal/e8b/scgrpo_causal_e8b/checkpoints
W=actor/model_state_dict/full_weights.pt
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }
sed -i 's|save_interval: 5|save_interval: 2|' $C/libero_safety_scgrpo_causal_e8bres.yaml
grep -n "save_interval\|resume_dir" $C/libero_safety_scgrpo_causal_e8bres.yaml
running() {  # our E8-b training wrapper alive?
  for q in $(ps -eo pid,args | grep "[b]ash run_causal.sh" | awk '{print $1}'); do
    tr '\0' '\n' < /proc/$q/environ 2>/dev/null | grep -qx "TAG=e8bres" && return 0
  done
  return 1
}
n=0
while true; do
  if grep -q "^EXIT=0" causal/train_e8b.log 2>/dev/null; then log "E8-b training finished (EXIT=0)"; exit 0; fi
  if ! running; then
    n=$((n + 1))
    last=$(ls -d $CK/global_step_* 2>/dev/null | grep -oE "[0-9]+$" | sort -n | while read s; do
             [ -d $CK/global_step_$s/actor/dcp_checkpoint ] && echo $s; done | tail -1)
    log "E8-b not running (restart $n), latest resumable checkpoint: step $last"
    [ -n "$last" ] || { log "no checkpoint, giving up"; exit 2; }
    if [ "$last" -ge 10 ]; then
      log "step 10 exists, launching eval only"
      JOBS="e8b_step_10|$CK/global_step_10/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
      log "E8-b EVAL_LAUNCHED"; exit 0
    fi
    sed -i "s|resume_dir: .*|resume_dir: $CK/global_step_$last|" $C/libero_safety_scgrpo_causal_e8bres.yaml
    sed -i "s|global_step_5/actor/dcp_checkpoint|global_step_$last/actor/dcp_checkpoint|" resume_e8b2.sh
    bash resume_e8b2.sh >> causal/watchdog_e8b.log 2>&1
    sleep 180
  fi
  sleep 60
done
