#!/bin/bash
# Split the unfinished cells of one checkpoint over 3 independent eval chains (disjoint cell lists).
#   NAME=r1all_step_10 CKPT=... GPUS_LIST="0 1 2" CHAINS="1 2 3" bash run_split.sh
# Each chain pN = eval_real_pN.sh (own Ray head / worker ports / LIBERO lock), one GPU each.
S=/data_cf/cws_sam/v1/scgrpo
cd $S
NAME=${NAME:?}; CKPT=${CKPT:?}
read -ra G <<< "${GPUS_LIST:?}"; read -ra C <<< "${CHAINS:?}"
TS=$(date +%H%M)
for x in eval_real/$NAME/*/; do  # unfinished / failed cell dirs are kept aside, never mixed in
  [ -d "$x" ] || continue
  grep -q "EXIT=0" "$x/launcher.log" 2>/dev/null || mv "$x" "eval_real/${NAME}_failed_${TS}_$(basename $x)"
done
LEFT=()
for s in affordance human_safety obstacle_avoidance obstacle_avoidance_human; do
  for t in $(seq 0 14); do
    lv=L0; [ $t -ge 5 ] && lv=L1; [ $t -ge 10 ] && lv=L2
    grep -q "EXIT=0" "eval_real/$NAME/${s}_${lv}_t${t}/launcher.log" 2>/dev/null || LEFT+=("$s:$lv:$t")
  done
done
echo "$NAME: ${#LEFT[@]} cells left"
for k in 0 1 2; do
  CELLS=""
  for ((j = k; j < ${#LEFT[@]}; j += 3)); do CELLS+=" ${LEFT[$j]}"; done
  [ -z "$CELLS" ] && continue
  echo "  chain p${C[$k]} GPU ${G[$k]}:$CELLS"
  nohup env NAME=$NAME CKPT=$CKPT GPUS=${G[$k]} CELLS="$CELLS" bash eval_real_p${C[$k]}.sh \
    > causal/split_${NAME}_p${C[$k]}.log 2>&1 < /dev/null &
  sleep ${STAGGER:-0}
done
