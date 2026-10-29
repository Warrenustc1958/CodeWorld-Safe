#!/bin/bash
# Evaluate several checkpoints with N independent chains (one GPU each, disjoint cell lists).
#   JOBS="name1|ckpt1 name2|ckpt2" GPUS_LIST="5 6 7" CHAINS="23 24 25" bash run_multi.sh
# Each chain runs its share of every job in order, then one retry pass over its failed cells.
# Unfinished cell dirs are moved aside first. Chains start 90 s apart (EGL/driver contention).
S=/data_cf/cws_sam/v1/scgrpo
cd $S
read -ra G <<< "${GPUS_LIST:?}"; read -ra C <<< "${CHAINS:?}"
N=${#C[@]}
TS=$(date +%H%M)
declare -A SHARE
for job in ${JOBS:?}; do
  NAME=${job%%|*}
  for x in eval_real/$NAME/*/; do
    [ -d "$x" ] || continue
    grep -q "EXIT=0" "$x/launcher.log" 2>/dev/null || mv "$x" "eval_real/${NAME}_failed_${TS}_$(basename $x)"
  done
  i=0
  for s in affordance human_safety obstacle_avoidance obstacle_avoidance_human; do
    for t in $(seq 0 14); do
      lv=L0; [ $t -ge 5 ] && lv=L1; [ $t -ge 10 ] && lv=L2
      grep -q "EXIT=0" "eval_real/$NAME/${s}_${lv}_t${t}/launcher.log" 2>/dev/null && continue
      k=$((i % N)); SHARE[$k,$NAME]+=" $s:$lv:$t"; i=$((i + 1))
    done
  done
  echo "$NAME: $i cells left"
done
for k in $(seq 0 $((N - 1))); do
  LOG=causal/multi_p${C[$k]}.log
  (
    for pass in 1 2; do
      for job in $JOBS; do
        NAME=${job%%|*}; CKPT=${job#*|}
        CELLS="${SHARE[$k,$NAME]}"
        [ -z "$CELLS" ] && continue
        echo "== pass $pass $NAME GPU ${G[$k]}:$CELLS"
        NAME=$NAME CKPT=$CKPT GPUS=${G[$k]} CELLS="$CELLS" bash eval_real_p${C[$k]}.sh
      done
    done
    echo CHAIN_DONE
  ) > $LOG 2>&1 < /dev/null &
  sleep 90
done
