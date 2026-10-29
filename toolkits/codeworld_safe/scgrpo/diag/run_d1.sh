#!/usr/bin/env bash
# D1: complete the real evals (60 cells x 20, official protocol); finished cells (EXIT=0) are skipped.
#   GPU 5 eval_real_ev3.sh (run-ev3.lock, Ray 6451): R1 step 10 then R1 step 20 (remaining cells)
#   GPU 6 eval_real.sh     (run.lock,     Ray 6431): R0b8 step 5  (45 non-affordance cells)
#   GPU 7 eval_real_ev2.sh (run-ev2.lock, Ray 6441): R0b8 step 10 (45 non-affordance cells)
# Unfinished cell dirs are moved aside (kept) first so killed/partial cells never mix into results.
set -uo pipefail
S=/data_cf/cws_sam/v1/scgrpo
TS=$(date +%H%M)
for n in r1_step_10 r1_step_20 r0b8_step_5 r0b8_step_10; do
  for x in $S/eval_real/$n/*/; do
    [ -d "$x" ] || continue
    grep -q "EXIT=0" "$x/launcher.log" 2>/dev/null || mv "$x" "$S/eval_real/${n}_failed_${TS}_$(basename $x)"
  done
done
R1=$S/causal/r1/scgrpo_causal_r1/checkpoints
RB=$S/causal/r0b8/scgrpo_causal_r0b8/checkpoints
W=actor/model_state_dict/full_weights.pt
cd $S
( NAME=r1_step_10 CKPT=$R1/global_step_10/$W GPUS=5 bash eval_real_ev3.sh > causal/d1_r1_step10.log 2>&1
  NAME=r1_step_20 CKPT=$R1/global_step_20/$W GPUS=5 bash eval_real_ev3.sh > causal/d1_r1_step20.log 2>&1 ) &
( NAME=r0b8_step_5 CKPT=$RB/global_step_5/$W GPUS=6 bash eval_real.sh > causal/d1_r0b8_step5.log 2>&1 ) &
( NAME=r0b8_step_10 CKPT=$RB/global_step_10/$W GPUS=7 bash eval_real_ev2.sh > causal/d1_r0b8_step10.log 2>&1 ) &
wait
for n in r1_step_10 r1_step_20 r0b8_step_5 r0b8_step_10; do
  echo "$n ok=$(grep -l EXIT=0 $S/eval_real/$n/*/launcher.log 2>/dev/null | wc -l)/60"
done
cd $S/diag
/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python matched_summ.py \
  r1_step_10 r1_step_20 r0b8_step_5 r0b8_step_10 > $S/causal/d1_summary.txt 2>&1
echo D1_DONE
