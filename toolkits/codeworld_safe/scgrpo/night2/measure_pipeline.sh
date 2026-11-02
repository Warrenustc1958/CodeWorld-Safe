#!/bin/bash
# Pre-registered within-group variance measurement (E:/WM Safety/related_rm_20261101/组内方差测量_预注册.md).
# Run by systemd cws-ms<LANE>.service. Lane 1: M1 (initial-state keyframes, 256 steps). Lane 2: M0 -> M2 -> M3.
# Policy frozen (lr 0); the identical final checkpoint RLinf writes is removed after each job. Lane 2 writes the
# report once both lanes are done. Markers in causal/msnight/.
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
LANE=${1:?}
N=$S/causal/msnight
mkdir -p $N
cd $S
if [ "$LANE" = 1 ]; then JOBS="m1"; PORT=6851; WR=58000-59999
else JOBS="m0 m2 m3"; PORT=6861; WR=56000-57999; fi
launcher_alive() {
  local d
  for d in /proc/[0-9]*; do
    tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q "launch_train.sh" || continue
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "TAG=$1" && return 0
  done 2>/dev/null
  return 1
}
log "measure lane $LANE start pid $$: $JOBS"
for tag in $JOBS; do
  until has ${tag}_done; do
    if grep -qs "TRAIN_DONE $tag" causal/${tag}_driver.log; then
      for d in $S/causal/$tag/scgrpo_causal_$tag/checkpoints/global_step_*; do
        [ -d "$d" ] && [ ! -L "$d" ] || continue
        case "$d" in */causal/m[0-3]/scgrpo_causal_m[0-3]/checkpoints/global_step_[0-9]*) rm -rf -- "$d" && log "$tag: removed $d" ;; esac
      done
      log "$tag: $(cat causal/$tag/dump/*.jsonl 2>/dev/null | wc -l) episodes"
      mark ${tag}_done
    elif ! launcher_alive $tag; then
      log "start/resume $tag"
      ( TAG=$tag PORT=$PORT WR=$WR KEEP="" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
          >> causal/${tag}_driver.log 2>&1 < /dev/null & )
      sleep 30
    fi
    has ${tag}_done || sleep 60
  done
done
mark lane${LANE}_done
if [ "$LANE" = 2 ]; then
  until has lane1_done; do sleep 60; done
  python3 rm2/measure_report.py causal/m0/dump causal/m1/dump causal/m2/dump causal/m3/dump > $N/report.md 2>&1
  mark report_done
  log "report written: $N/report.md"
fi
