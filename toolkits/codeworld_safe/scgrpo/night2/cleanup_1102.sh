#!/bin/bash
# 11-02 cleanup: only our finished / duplicated items. Never touches rp_ev*, rp_cm3 (M3 running), other users' data.
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
S=/data_cf/cws_sam/v1/scgrpo
echo "before: $(df -h / | tail -1 | awk '{print $4}') free on /, $(df -h /data_cf | tail -1 | awk '{print $4}') free on /data_cf"
# 1. idle eval-chain Ray heads (no eval_real_p / run_multi process is running)
if ps -eo args | grep -qE "[e]val_real_p|[r]un_multi"; then echo "eval running, skip eval heads"; else
  for c in $(seq 20 31); do kill_tmpdir /tmp/rp_p$c; done; echo "stopped idle eval heads p20-p31"; fi
sleep 5
# 2. /tmp Ray session dirs of finished jobs (skip any dir that still has a live process)
for d in /tmp/rp_c* /tmp/p5_c* /tmp/rp_p*; do
  [ -d "$d" ] || continue
  case "$d" in /tmp/rp_cm3|/tmp/p5_cm3) continue ;; esac
  live=0
  for p in /proc/[0-9]*; do tr '\0' '\n' < $p/environ 2>/dev/null | grep -qx "RAY_TMPDIR=$d" && { live=1; break; }; done 2>/dev/null
  [ $live = 1 ] && { echo "keep (live) $d"; continue; }
  case "$d" in /tmp/rp_c*|/tmp/p5_c*|/tmp/rp_p[0-9]*) rm -rf -- "$d" ;; esac
done
echo "tmp dirs left: $(ls -d /tmp/rp_c* /tmp/p5_c* /tmp/rp_p* 2>/dev/null | tr '\n' ' ')"
# 3. exact duplicates
rm -rf -- $S/causal/m2/dump_dup_rerun && echo "removed m2 duplicate rerun dump"
d=$S/causal/col10a/scgrpo_causal_col10a/checkpoints/global_step_16   # lr-0 final save = copy of e10o step 10
[ -d $d ] && [ ! -L $d ] && rm -rf -- $d && echo "removed $d"
# 4. finished diagnostics, never a model candidate
d=$S/causal/e13d/scgrpo_causal_e13d/checkpoints/global_step_12      # D3 continuation, dump analysed, not evaluated
[ -d $d ] && [ ! -L $d ] && rm -rf -- $d && echo "removed $d"
d=$S/causal/e14/scgrpo_causal_e14/checkpoints/global_step_10        # D2 noise diagnostic, eval done
[ -d $d ] && [ ! -L $d ] && rm -rf -- $d && echo "removed $d"
echo "after: $(df -h / | tail -1 | awk '{print $4}') free on /, $(df -h /data_cf | tail -1 | awk '{print $4}') free on /data_cf"
