#!/bin/bash
# Keep ONLY the newest complete checkpoint per run (each save ~60 GB:
# full_weights.pt ~15 GB + resumable FSDP DCP ~45 GB).
# An older global_step_* is deleted only after the newest one is complete:
# full_weights.pt non-empty, DCP .metadata present, and no file written in the last 3 min.
ROOT=/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/sft_oft
while true; do
  for ck in "$ROOT"/*/*/checkpoints; do
    [ -d "$ck" ] || continue
    steps=$(ls -d "$ck"/global_step_* 2>/dev/null | sort -t_ -k3 -n)
    newest=$(echo "$steps" | tail -1)
    [ -n "$newest" ] || continue
    [ -s "$newest/actor/model_state_dict/full_weights.pt" ] || continue
    [ -f "$newest/actor/dcp_checkpoint/.metadata" ] || continue
    [ -z "$(find "$newest" -type f -mmin -3 | head -1)" ] || continue
    for d in $steps; do
      [ "$d" = "$newest" ] && continue
      rm -rf "$d" && echo "$(date '+%F %T') removed $d (newest: $(basename "$newest"))"
    done
  done
  echo "$(date '+%F %T') free: $(df -h / | tail -1 | awk '{print $4}')"
  sleep 300
done
