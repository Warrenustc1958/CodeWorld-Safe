#!/bin/bash
# After the 14:52-14:55 machine-wide failure:
#  1. stop the dead/finished heads: E8-b and E8-a training heads, broken eval heads p29-p31
#  2. resume E8-b from global_step_5 to 10 on GPUs 0-3 (same run dir), then eval step 10 on GPUs 0/1/2 (p20-p22)
#  3. rerun E8-O's unfinished cells on GPUs 5/6/7 with the intact eval heads p23-p25
# Log: causal/resume_e8b.log
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
W=actor/model_state_dict/full_weights.pt
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }
TAG=e8b bash stop_causal_head.sh
TAG=e8a bash stop_causal_head.sh
ALL=""
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qE "^RAY_TMPDIR=/tmp/rp_p(29|30|31)$" && ALL="$ALL ${d#/proc/}"
done
ALL=$(echo $ALL | tr ' ' '\n' | grep -vx "$$" | sort -u)
log "stopping $(echo $ALL | wc -w) procs of broken eval heads p29-p31"
kill -TERM $ALL 2>/dev/null; sleep 8
for q in $ALL; do kill -0 "$q" 2>/dev/null && kill -KILL "$q" 2>/dev/null; done
sleep 5
CK=$S/causal/e8b/scgrpo_causal_e8b/checkpoints
[ -d $CK/global_step_5/actor/dcp_checkpoint ] || { log "no resumable step-5 checkpoint"; exit 2; }
sed -e "s|resume_dir: null|resume_dir: $CK/global_step_5|" \
    -e 's|^# E8 e8b: .*|# E8 e8b resumed from step 5 to 10 (same run dir), GPUs 0-3|' \
    $C/libero_safety_scgrpo_causal_e8b.yaml > $C/libero_safety_scgrpo_causal_e8bres.yaml
diff $C/libero_safety_scgrpo_causal_e8b.yaml $C/libero_safety_scgrpo_causal_e8bres.yaml
grep -q "resume_dir: $CK/global_step_5" $C/libero_safety_scgrpo_causal_e8bres.yaml || { log "config edit failed"; exit 4; }
busy=$(ss -ltn | awk '{print $4}' | grep -oE '[0-9]+$' | awk '$1>=56000 && $1<=57999' | wc -l)
log "ports 56000-57999 in use: $busy"
[ "$busy" -lt 50 ] || { log "worker port range busy"; exit 3; }
for p in 6661 9661; do ss -ltn | grep -q ":$p " && { log "port $p busy"; exit 3; }; done
mv causal/train_e8b.log causal/train_e8b_failed2_$(date +%H%M).log
(
  RAY_EXTRA="--min-worker-port=56000 --max-worker-port=57999" TAG=e8bres PORT=6661 bash run_causal.sh > causal/train_e8b.log 2>&1
  rc=$?; log "E8-b (resume) TRAIN rc=$rc"
  [ $rc -eq 0 ] || exit $rc
  JOBS="e8b_step_10|$CK/global_step_10/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
  log "E8-b EVAL_LAUNCHED"
) &
log "E8-b resume started"
OK=$S/causal/e8o/scgrpo_causal_e8o/checkpoints
JOBS="e8o_step_10|$OK/global_step_10/$W" GPUS_LIST="5 6 7" CHAINS="23 24 25" bash run_multi.sh
log "E8-O rerun launched"
wait
