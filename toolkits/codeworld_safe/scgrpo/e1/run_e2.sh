#!/bin/bash
cd /data_cf/cws_sam/v1/scgrpo
RAY_EXTRA="--min-worker-port=56000 --max-worker-port=57999" TAG=e2 PORT=6521 bash run_causal.sh > causal/train_e2.log 2>&1
rc=$?
echo "TRAIN rc=$rc $(tail -1 causal/train_e2.log)"
[ $rc -eq 0 ] || exit $rc
CK=/data_cf/cws_sam/v1/scgrpo/causal/e2/scgrpo_causal_e2/checkpoints
JOBS="e2_step_10|$CK/global_step_10/actor/model_state_dict/full_weights.pt e2_step_20|$CK/global_step_20/actor/model_state_dict/full_weights.pt" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
echo E2_EVAL_LAUNCHED
