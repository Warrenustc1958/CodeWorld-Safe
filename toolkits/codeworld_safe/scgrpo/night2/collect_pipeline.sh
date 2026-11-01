#!/bin/bash
# Phase-1 collection (run by systemd cws-col<LANE>.service): on-policy rollouts with lr 0 (policy frozen), filtered
# and weighted keyframes, per-episode JSON + arrays dump. Jobs of a lane run one after another; each starts from its
# checkpoint (symlinked as the latest step) and runs 6 steps = 3072 episodes. The identical final checkpoint that
# RLinf writes at the end is removed. Lane B waits for E14's eval (GPU memory). Markers in causal/colnight/.
#   bash collect_pipeline.sh A|B
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
LANE=${1:?}
N=$S/causal/colnight
mkdir -p $N
cd $S
if [ "$LANE" = A ]; then JOBS="colbase col10a col10b col10c"; PORT=6811; WR=58000-59999
else JOBS="col12r col13"; PORT=6821; WR=56000-57999; fi  # col11m (E11-SAM s20) dropped 11-02 for time
launcher_alive() {
  local d
  for d in /proc/[0-9]*; do
    tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q "launch_train.sh" || continue
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "TAG=$1" && return 0
  done 2>/dev/null
  return 1
}
log "collect lane $LANE start pid $$: $JOBS"
if [ "$LANE" = B ]; then
  until [ -f causal/e14night/e14_eval_done ]; do sleep 60; done
  log "lane B: E14 eval done, starting"
fi
for tag in $JOBS; do
  until has ${tag}_done; do
    CK=$S/causal/$tag/scgrpo_causal_$tag/checkpoints
    if grep -qs "TRAIN_DONE $tag" causal/${tag}_driver.log; then
      for d in $CK/global_step_*; do  # RLinf's final save: same weights as the start (lr 0); start is a symlink
        [ -d "$d" ] && [ ! -L "$d" ] || continue
        case "$d" in */causal/col*/checkpoints/global_step_[0-9]*) rm -rf -- "$d" && log "$tag: removed $d" ;; esac
      done
      log "$tag: $(cat causal/$tag/dump/*.jsonl 2>/dev/null | wc -l) episodes"
      mark ${tag}_done
    elif ! launcher_alive $tag; then
      log "start/resume $tag"
      ( TAG=$tag PORT=$PORT WR=$WR KEEP="0 10 20 40" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
          >> causal/${tag}_driver.log 2>&1 < /dev/null & )
      sleep 30
    fi
    has ${tag}_done || sleep 60
  done
done
mark lane${LANE}_done
log "collect lane $LANE done"
