#!/bin/bash
# Helpers for night_pipeline.sh (sourced).
S=/data_cf/cws_sam/v1/scgrpo
R=/mnt/nfs/fjj/Code/CodeWorld-Safe
C=$R/examples/embodiment/config
RM=/data_cf/cws_rm
N=$S/causal/night
PYW=/mnt/nfs/fjj/wan_ac_sft/venv-wan/bin/python
PYR=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python
W=actor/model_state_dict/full_weights.pt
E7CK=$S/causal/e7/scgrpo_causal_e7/checkpoints
E9CK=$S/causal/e9/scgrpo_causal_e9/checkpoints
mkdir -p $N
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mark() { [ -f $N/$1 ] || { touch $N/$1; log "MARK $1"; }; }
has() { [ -f $N/$1 ]; }
bump() { local c=$(cat $N/$1.count 2>/dev/null || echo 0); echo $((c + 1)) > $N/$1.count; echo $((c + 1)); }
now() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }
alive() { ps -eo args | grep -v grep | grep -qE "$1"; }
cells_ok() { local n=0 x; for x in "$@"; do n=$((n + $(grep -l EXIT=0 $S/eval_real/$x/*/launcher.log 2>/dev/null | wc -l))); done; echo $n; }
kill_tmpdir() {
  local all="" d q
  for d in /proc/[0-9]*; do
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=$1" && all="$all ${d#/proc/}"
  done
  all=$(echo $all | tr ' ' '\n' | grep -vx "$$" | sort -u)
  [ -n "$all" ] || return 0
  kill -TERM $all 2>/dev/null; sleep 6
  for q in $all; do kill -0 $q 2>/dev/null && kill -KILL $q 2>/dev/null; done
}
train_rm() {  # NAME GPU VIEWS EXTRA_SPEC...
  local name=$1 gpu=$2 views=$3; shift 3
  mkdir -p $RM/runs/$name
  ( cd $R && FRAMES=$RM/frames_v5mix nohup setsid bash toolkits/codeworld_safe/rm/launch_rm.sh $gpu $views $name \
      --extra_train "$@" --extra_frac 0.3 --domain_pos_weight --val_splits val newval wannewval \
      --test_splits test newtest wannewtest --mem_gb 12 --workers 8 > $RM/runs/$name/launch.out 2>&1 < /dev/null & )
}
ensure_rm() {  # NAME GPU VIEWS EXTRA_SPEC...   -> NAME_done | NAME_failed
  local name=$1
  { has ${name}_done || has ${name}_failed; } && return 0
  if [ -f $RM/runs/$name/final.json ] && [ -f $RM/runs/$name/best.pt ]; then mark ${name}_done; return 0; fi
  alive "train_rm.*runs/$name( |$)" && return 0
  local t=$(bump $name)
  [ $t -gt 3 ] && { mark ${name}_failed; return 0; }
  [ -n "$(ls -A $RM/runs/$name 2>/dev/null)" ] && mv $RM/runs/$name $RM/runs/${name}_try$(date +%H%M%S)
  log "start RM $name (try $t)"
  train_rm "$@"
}
ensure_eval() {  # KEY NEED CHAINS("a b c") GPUS("x y z") JOB...  (JOB = name|ckpt)
  local key=$1 need=$2 chains=$3 gpus=$4; shift 4
  has ${key}_done && return 0
  local names="" j c
  for j in "$@"; do names="$names ${j%%|*}"; done
  if [ $(cells_ok $names) -ge $need ]; then mark ${key}_done; return 0; fi
  for c in $chains; do alive "eval_real_p$c\.sh" && return 0; done
  alive "run_multi.sh.*$key" && return 0
  local t=$(bump $key)
  if [ $t -gt 4 ]; then log "$key: $(cells_ok $names)/$need after 4 launches, marking partial"; mark ${key}_done; return 0; fi
  for c in $chains; do kill_tmpdir /tmp/rp_p$c; done
  log "launch eval $key (try $t): $(cells_ok $names)/$need done"
  ( cd $S && JOBS="$*" GPUS_LIST="$gpus" CHAINS="$chains" nohup setsid bash run_multi.sh $key \
      > causal/night/eval_$key.launch.log 2>&1 < /dev/null & )
}
