#!/bin/bash
# Resumable SC-GRPO training loop for config libero_safety_scgrpo_causal_$TAG.yaml (save_interval 2):
# restarts from the latest complete checkpoint after a kill, prunes to every 10th + the 2 latest checkpoints.
#   TAG=e9 PORT=6681 WR=58000-59999 bash launch_train.sh      (prints "TRAIN_DONE <tag>" when finished)
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
TAG=${TAG:?}; PORT=${PORT:?}; WR=${WR:?}
CFG=$C/libero_safety_scgrpo_causal_$TAG.yaml
CK=$S/causal/$TAG/scgrpo_causal_$TAG/checkpoints
W=actor/model_state_dict/full_weights.pt
cd $S
log() { echo "[$(date +%H:%M:%S)] [$TAG] $*"; }
complete() { [ -f $CK/global_step_$1/actor/dcp_checkpoint/.metadata ] && [ -f $CK/global_step_$1/$W ]; }
steps() { ls -d $CK/global_step_* 2>/dev/null | grep -oE "[0-9]+$" | sort -n; }
latest() { for s in $(steps); do complete $s && echo $s; done | tail -1; }
KEEP=${KEEP:-}            # explicit steps to keep (e.g. "10 20 40"); empty = every 10th
KEEP_LATEST=${KEEP_LATEST:-2}
prune() {
  local keep="$(for s in $(steps); do complete $s && echo $s; done | tail -$KEEP_LATEST)"
  for s in $(steps); do
    if [ -n "$KEEP" ]; then echo " $KEEP " | grep -q " $s " && continue
    else [ $((s % 10)) -eq 0 ] && continue; fi
    echo "$keep" | grep -qx "$s" && continue
    complete $s || continue
    d=$CK/global_step_$s
    case "$d" in */checkpoints/global_step_[0-9]*) rm -rf -- "$d" && log "pruned step $s" ;; esac
  done
}
cleanup() {
  local all=""
  for d in /proc/[0-9]*; do
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/rp_c$TAG" && all="$all ${d#/proc/}"
  done
  all=$(echo $all | tr ' ' '\n' | grep -vx "$$" | sort -u)
  [ -n "$all" ] && { kill -TERM $all 2>/dev/null; sleep 8; for q in $all; do kill -0 $q 2>/dev/null && kill -KILL $q 2>/dev/null; done; }
}
n=0
while true; do
  last=$(latest)
  if [ -n "$last" ]; then sed -i "s|resume_dir: .*|resume_dir: $CK/global_step_$last|" $CFG
  else sed -i "s|resume_dir: .*|resume_dir: null|" $CFG; fi
  [ -f causal/train_$TAG.log ] && mv causal/train_$TAG.log causal/train_${TAG}_$(date +%m%d_%H%M%S).log
  log "start (try $n) from step ${last:-0}"
  cleanup
  RAY_EXTRA="--min-worker-port=${WR%-*} --max-worker-port=${WR#*-}" TAG=$TAG PORT=$PORT bash run_causal.sh > causal/train_$TAG.log 2>&1 &
  tp=$!
  while kill -0 $tp 2>/dev/null; do prune; sleep 60; done
  wait $tp; rc=$?
  prune
  if grep -q "^EXIT=0" causal/train_$TAG.log; then log "training finished (EXIT=0)"; break; fi
  n=$((n + 1))
  log "training died rc=$rc, latest checkpoint $(latest)"
  [ $n -ge 1000 ] && { log "too many restarts"; exit 1; }
  sleep 20
done
cleanup
echo "TRAIN_DONE $TAG"
