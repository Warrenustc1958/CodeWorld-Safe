#!/bin/bash
# E4 gate failed -> fallback agreed in the night plan: no E6; continue RM-all from step 20 to step 40
# (does the RM-reward run keep degrading?), then eval step 40. Unattended.
#  1. wait for E1 eval chains p23-p25 (GPUs 5/6/7) to finish
#  2. resume RM-all from global_step_20 on GPUs 4-7, max_steps 40, save only at 40 (disk)
#  3. eval rmall_step_40 with chains p26-p28 on GPUs 5/6/7
# Log: causal/auto_rmall40.log ; markers RMALL40_TRAIN_DONE / RMALL40_EVAL_LAUNCHED
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }
CK=$S/causal/rmall/scgrpo_causal_rmall/checkpoints
[ -d $CK/global_step_20/actor/dcp_checkpoint ] || { log "no resumable step-20 checkpoint"; exit 2; }
sed -e 's|max_steps: 20|max_steps: 40|' -e 's|save_interval: 5|save_interval: 20|' \
    -e "s|resume_dir: null|resume_dir: $CK/global_step_20|" \
    -e 's|^# E1 RM-all: .*|# E1 RM-all continued from step 20 to 40 (same run dir), GPUs 4-7|' \
    $C/libero_safety_scgrpo_causal_rmall.yaml > $C/libero_safety_scgrpo_causal_rmall40.yaml
diff $C/libero_safety_scgrpo_causal_rmall.yaml $C/libero_safety_scgrpo_causal_rmall40.yaml
grep -q "resume_dir: $CK/global_step_20" $C/libero_safety_scgrpo_causal_rmall40.yaml || { log "config edit failed"; exit 4; }
log "waiting for E1 chains p23-p25"
until [ $(cat causal/multi_p2{3,4,5}.log 2>/dev/null | grep -c CHAIN_DONE) -eq 3 ]; do sleep 60; done
sleep 30
for p in 6601 9601; do ss -ltn | grep -q ":$p " && { log "port $p busy"; exit 3; }; done
log "RM-all 20->40 training start"
RAY_EXTRA="--min-worker-port=58000 --max-worker-port=59999" TAG=rmall40 PORT=6601 bash run_causal.sh > causal/train_rmall40.log 2>&1
rc=$?
log "train rc=$rc $(tail -1 causal/train_rmall40.log | cut -c1-80)"
[ $rc -eq 0 ] || exit $rc
echo RMALL40_TRAIN_DONE
W=actor/model_state_dict/full_weights.pt
JOBS="rmall_step_40|$CK/global_step_40/$W" GPUS_LIST="5 6 7" CHAINS="26 27 28" bash run_multi.sh
log "RMALL40_EVAL_LAUNCHED"
