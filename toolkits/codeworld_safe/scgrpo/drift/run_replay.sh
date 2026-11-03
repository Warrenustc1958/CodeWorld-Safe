#!/bin/bash
# One p_replay.py shard per card, then the summary.
#   OUT=<dir> CARDS="0 1 ... 15" bash run_replay.sh <dump dir> [<dump dir> ...] -- [p_replay args]
set -uo pipefail
R=/mnt/cpfs/jieduan
C=$R/code/CodeWorld-Safe
OUT=${OUT:?}; CARDS=${CARDS:-"$(seq -s ' ' 0 15)"}
dumps=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do dumps+=("$1"); shift; done; [ "${1:-}" = "--" ] && shift
set +u; source $R/envs/cws_env.sh >/dev/null; set -u  # cws_env.sh expands an unset PYTHONPATH
export PYTHONPATH=$R/code/diffsynth-studio:$C:${PYTHONPATH:-} NUMBA_DISABLE_JIT=1
mkdir -p $OUT
n=$(echo $CARDS | wc -w); i=0; pids=()
for c in $CARDS; do
  CUDA_VISIBLE_DEVICES=$c python $C/toolkits/codeworld_safe/scgrpo/drift/p_replay.py --dump "${dumps[@]}" \
    --out $OUT/rep.jsonl --shard $i --nshard $n "$@" > $OUT/shard_$i.log 2>&1 &
  pids+=($!); i=$((i + 1))
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
python $C/toolkits/codeworld_safe/scgrpo/drift/p_summary.py --rep $OUT/rep.jsonl --out $OUT/summary.json \
  > $OUT/summary.txt 2>&1
echo "EXIT=$fail"
