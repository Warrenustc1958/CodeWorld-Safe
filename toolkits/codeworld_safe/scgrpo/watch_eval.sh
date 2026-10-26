#!/usr/bin/env bash
# Real-eval watcher: base policy first, then the newest SC-GRPO checkpoint not yet evaluated.
# Fixed 16-cell subset (per suite: L0 t0,t1 and L1 t5,t6) x 20 fresh_init/v1 states = 320 episodes.
set -uo pipefail
S=/data_cf/cws_sam/v1/scgrpo
CK=$S/runs/scgrpo_pi05_wan41k_main/checkpoints
export CELLS=""
for s in affordance human_safety obstacle_avoidance obstacle_avoidance_human; do
  CELLS+=" $s:L0:0 $s:L0:1 $s:L1:5 $s:L1:6"
done
NAME=base CKPT= bash $S/eval_real.sh
while true; do
  last=$(ls -d $CK/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
  if [ -n "$last" ] && [ ! -f $S/eval_real/step_$last/.done ] && [ -f $CK/global_step_$last/actor/model_state_dict/full_weights.pt ]; then
    sleep 60  # let the checkpoint finish writing
    NAME=step_$last CKPT=$CK/global_step_$last/actor/model_state_dict/full_weights.pt bash $S/eval_real.sh \
      && touch $S/eval_real/step_$last/.done
    /mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python $S/summ_eval_real.py \
      $(ls $S/eval_real | grep -E "^(base|step_)" | sort -t_ -k2 -n) > $S/eval_real/summary.txt 2>&1
  fi
  grep -q "EXIT=0" $S/train_main.log 2>/dev/null && [ -f $S/eval_real/step_$last/.done ] && break
  sleep 300
done
