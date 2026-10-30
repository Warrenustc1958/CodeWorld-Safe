#!/bin/bash
# Night automation (E4 -> E6), runs unattended:
#  1. wait for RM v6 training (runs/rm_v6_d2) to finish
#  2. E4 gate on held-out D2 (GPU 4): pass = affordance precision >= .93 and recall >= .75
#  3. if pass: E6 config = RM-all with RM v6 + the gate's best rule (thr x k), save every 10
#  4. wait for E1 eval chains p23-p25 (GPUs 5/6/7) to finish, then train E6 on GPUs 4-7
#  5. eval E6 step 10/20 with chains p26-p28 on GPUs 5/6/7
# Log: /data_cf/cws_sam/v1/scgrpo/causal/auto_e6.log ; markers E4_PASS / E4_FAIL / E6_TRAIN_DONE / E6_EVAL_LAUNCHED
S=/data_cf/cws_sam/v1/scgrpo
R=/mnt/nfs/fjj/Code/CodeWorld-Safe
C=$R/examples/embodiment/config
RUN=/data_cf/cws_rm/runs/rm_v6_d2
PYW=/mnt/nfs/fjj/wan_ac_sft/venv-wan/bin/python
cd $S
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1) RM v6 training
log "waiting for RM v6 training"
while ps -eo args | grep -q "[t]oolkits.codeworld_safe.rm.train_rm.*rm_v6_d2"; do sleep 60; done
[ -f $RUN/best.pt ] || { log "RM v6 training ended without best.pt"; echo E4_FAIL; exit 1; }
log "RM v6 done: $(tail -1 $RUN/log.txt | cut -c1-120)"

# 2) gate
cd $R
PYTHONPATH=$R CUDA_VISIBLE_DEVICES=4 $PYW $S/diag/gate_rm_d2.py v5=/data_cf/cws_rm/runs/rm_v5_tail/best.pt v6=$RUN/best.pt \
  > $S/causal/e4_gate.txt 2>&1
cd $S
grep -v -i warn causal/e4_gate.txt | tail -16
read -r PASS RULE < <($PYW -c "
import json; d=json.load(open('/data_cf/cws_rm/runs/gate_d2.json'))['v6']; b=d['best']; r=d['rules'][b]
print(int(r['P']>=0.93 and r['R']>=0.75), b)")
if [ "$PASS" != 1 ]; then
  log "E4 gate FAILED (best rule $RULE); E6 not started"; echo E4_FAIL; exit 0
fi
THR=${RULE%x*}; K=${RULE#*x}
log "E4_PASS rule thr=$THR k=$K"; echo E4_PASS

# 3) E6 config: identical to RM-all except the reward model and its rule
sed -e 's|causal/rmall"|causal/e6"|' -e 's|"scgrpo_causal_rmall"|"scgrpo_causal_e6"|' \
    -e 's|save_interval: 5|save_interval: 10|' \
    -e "s|from_pretrained: /data_cf/cws_rm/runs/rm_v3_agent_w41k/best.pt|from_pretrained: $RUN/best.pt|" \
    -e "s|success_threshold: 0.8|success_threshold: $THR|" -e "s|success_consecutive: 4|success_consecutive: $K|" \
    -e 's|^# E1 RM-all: .*|# E6: RM-all with RM v6 (D2 hard negatives) and its gated rule, GPUs 4-7|' \
    $C/libero_safety_scgrpo_causal_rmall.yaml > $C/libero_safety_scgrpo_causal_e6.yaml
diff $C/libero_safety_scgrpo_causal_rmall.yaml $C/libero_safety_scgrpo_causal_e6.yaml
grep -q "$RUN/best.pt" $C/libero_safety_scgrpo_causal_e6.yaml || { log "E6 config edit failed"; exit 4; }

# 4) wait for GPUs 5-7 (E1 chains), then train
log "waiting for E1 chains p23-p25"
until [ $(cat causal/multi_p2{3,4,5}.log 2>/dev/null | grep -c CHAIN_DONE) -eq 3 ]; do sleep 60; done
sleep 30
for p in 6591 9591; do ss -ltn | grep -q ":$p " && { log "port $p busy"; exit 3; }; done
log "E6 training start"
RAY_EXTRA="--min-worker-port=58000 --max-worker-port=59999" TAG=e6 PORT=6591 bash run_causal.sh > causal/train_e6.log 2>&1
rc=$?
log "E6 train rc=$rc $(tail -1 causal/train_e6.log | cut -c1-80)"
[ $rc -eq 0 ] || exit $rc
echo E6_TRAIN_DONE

# 5) eval
W=actor/model_state_dict/full_weights.pt
CK=$S/causal/e6/scgrpo_causal_e6/checkpoints
JOBS="e6_step_10|$CK/global_step_10/$W e6_step_20|$CK/global_step_20/$W" GPUS_LIST="5 6 7" CHAINS="26 27 28" bash run_multi.sh
log "E6_EVAL_LAUNCHED"
