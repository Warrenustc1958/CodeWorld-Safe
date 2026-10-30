#!/bin/bash
# E8-b resume died again (~15:42, step 6, no EXIT). Stop OUR leftovers of the e8bres head (matched by
# RAY_TMPDIR in environ), then resume from global_step_5 again; eval step 10 on GPUs 0/1/2 (p20-p22).
S=/data_cf/cws_sam/v1/scgrpo
W=actor/model_state_dict/full_weights.pt
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }
ALL=""
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/rp_ce8bres" && ALL="$ALL ${d#/proc/}"
done
for q in $(ps -eo pid,args | grep "[/]tmp/rp_ce8bres/" | awk '{print $1}'); do ALL="$ALL $q"; done
ALL=$(echo $ALL | tr ' ' '\n' | grep -vx "$$" | sort -u)
log "stopping $(echo $ALL | wc -w) leftover e8bres procs"
kill -TERM $ALL 2>/dev/null; sleep 8
for q in $ALL; do kill -0 "$q" 2>/dev/null && kill -KILL "$q" 2>/dev/null; done
sleep 5
CK=$S/causal/e8b/scgrpo_causal_e8b/checkpoints
[ -d $CK/global_step_5/actor/dcp_checkpoint ] || { log "no step-5 checkpoint"; exit 2; }
busy=$(ss -ltn | awk '{print $4}' | grep -oE '[0-9]+$' | awk '$1>=56000 && $1<=57999' | wc -l)
log "ports 56000-57999 in use: $busy"
[ "$busy" -lt 50 ] || { log "worker port range busy"; exit 3; }
for p in 6661 9661; do ss -ltn | grep -q ":$p " && { log "port $p busy"; exit 3; }; done
mv causal/train_e8b.log causal/train_e8b_failed3_$(date +%H%M).log
(
  RAY_EXTRA="--min-worker-port=56000 --max-worker-port=57999" TAG=e8bres PORT=6661 bash run_causal.sh > causal/train_e8b.log 2>&1
  rc=$?; log "E8-b (resume 2) TRAIN rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  JOBS="e8b_step_10|$CK/global_step_10/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
  log "E8-b EVAL_LAUNCHED"
) > causal/e8b_resume2.log 2>&1 < /dev/null &
log "E8-b resume 2 started"
