#!/bin/bash
# Keep ONLY the newest complete checkpoint per run (each save ~60 GB:
# full_weights.pt ~15 GB + resumable FSDP DCP ~45 GB).
# An older global_step_* is deleted only after the newest one is complete:
# full_weights.pt non-empty, DCP .metadata present, and no file written in the last 3 min.
#
# Usage: sft_ckpt_janitor.sh [--dry] [--once]
#   --dry   print what would be removed, delete nothing
#   --once  run a single pass and exit
#
# The step is parsed from the directory NAME (global_step_<N>) and compared
# numerically. (The first version sorted full paths with `sort -t_ -k3`, which
# picked a path component instead of N and deleted the newest checkpoint.)
DRY=0; ONCE=0
for a in "$@"; do
  case "$a" in --dry) DRY=1 ;; --once) ONCE=1 ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac
done
ROOT=${JANITOR_ROOT:-/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/sft_oft}

step_of() { local b; b=$(basename "$1"); echo "${b#global_step_}"; }

while true; do
  for ck in "$ROOT"/*/*/checkpoints; do
    [ -d "$ck" ] || continue
    newest=""; newest_n=-1
    for d in "$ck"/global_step_*; do
      [ -d "$d" ] || continue
      n=$(step_of "$d")
      [[ "$n" =~ ^[0-9]+$ ]] || { echo "skip non-numeric $d"; continue; }
      if [ "$n" -gt "$newest_n" ]; then newest_n=$n; newest=$d; fi
    done
    [ -n "$newest" ] || continue
    [ -s "$newest/actor/model_state_dict/full_weights.pt" ] || continue
    [ -f "$newest/actor/dcp_checkpoint/.metadata" ] || continue
    [ -z "$(find "$newest" -type f -mmin -3 | head -1)" ] || continue
    for d in "$ck"/global_step_*; do
      [ -d "$d" ] || continue
      n=$(step_of "$d")
      [[ "$n" =~ ^[0-9]+$ ]] || continue
      [ "$n" -lt "$newest_n" ] || continue
      if [ "$DRY" = 1 ]; then
        echo "$(date '+%F %T') [dry] would remove $d (newest: global_step_$newest_n)"
      else
        rm -rf "$d" && echo "$(date '+%F %T') removed $d (newest: global_step_$newest_n)"
      fi
    done
  done
  echo "$(date '+%F %T') free: $(df -h / | tail -1 | awk '{print $4}')"
  [ "$ONCE" = 1 ] && exit 0
  sleep 300
done
