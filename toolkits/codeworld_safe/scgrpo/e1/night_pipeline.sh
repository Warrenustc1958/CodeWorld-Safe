#!/bin/bash
# Night pipeline 10-31, run by systemd cws-night.service (Restart=on-failure, KillMode=process): if this script is
# killed it is restarted and continues from markers in causal/night/ + files on disk + live processes.
#   E7 train (driver launch_e7b.sh) -> E7 eval steps 20/40/60/80 (GPUs 0,1,2,5,6,7, chains p20-p25)
#   R3: frames_d2w + RM v7w (dual view, wrong-part episodes 3x)               GPU 3, alongside E7 training
#   R2: collect E7 step 40 (GPU 4) / step 20 (GPU 3) -> frames_r2 -> RM v8     after E7 training
#   gate v7/v7w/v8 on d2gtest -> select (must beat v3) -> E9 = E7 config + selected RM, 8 GPUs, 10 steps
#   -> E9 eval (chains p26-p31) -> causal/summary_morning.txt -> resume anything paused
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
cd $S
log "pipeline start pid $$"
env_alive() { local d; for d in /proc/[0-9]*; do tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "$1" && return 0; done; return 1; }
start() { nohup setsid "$@" < /dev/null > /dev/null 2>&1 & }
clean_pkl() { $PYR - "$1" <<'EOF'
import pathlib, pickle, sys
for f in pathlib.Path(sys.argv[1]).glob("*/batch_*.pkl"):
    try:
        pickle.load(open(f, "rb"))
    except Exception:
        f.unlink(); print("removed partial", f)
EOF
}
ensure_collect() {  # NAME GPU STEP
  local name=$1 gpu=$2 step=$3 out=$S/diag/r2_$1
  { has c_${name}_done || has c_${name}_failed; } && return 0
  grep -qs D2_COLLECT_DONE $out/collect.log && { mark c_${name}_done; return 0; }
  { alive "run_r2.sh $gpu $name" || env_alive "R2_NAME=$name"; } && return 0
  local t=$(bump c_$name)
  [ $t -gt 6 ] && { mark c_${name}_failed; return 0; }
  [ -d $out ] && clean_pkl $out
  log "start collector $name on GPU $gpu (try $t)"
  start bash $S/diag/run_r2.sh $gpu $name $E7CK/global_step_$step $out
}
stop_e7_eval() {
  local p c
  for p in $(ps -eo pid,args | grep -E "[r]un_multi\.sh|[e]val_real_p2[0-5]\.sh" | awk '{print $1}'); do kill $p 2>/dev/null; done
  for c in 20 21 22 23 24 25; do kill_tmpdir /tmp/rp_p$c; done
}
e9phase() { has e9_started && ! has e9eval_done; }
done_or_failed() { has $1_done || has $1_failed; }

while true; do
  T=$(now)
  # ---- E7 training (pause at 07:30 if still not finished, so E9 can run; resumed after the summary)
  if ! has e7train_done; then
    if grep -q "training finished (EXIT=0)" causal/e7_driver.log; then mark e7train_done
    elif ! has e7_paused && [ $T -ge 450 ] && [ $T -lt 600 ]; then
      log "E7 not finished at 07:30: pausing it for E9"
      for p in $(ps -eo pid,args | grep -E "[l]aunch_e7b?\.sh" | awk '{print $1}'); do kill $p; done
      kill_tmpdir /tmp/rp_ce7; mark e7_paused
    elif ! has e7_paused && ! alive "launch_e7b?\.sh"; then
      log "E7 driver not running: restarting from the latest checkpoint"
      ( nohup setsid bash launch_e7b.sh >> causal/e7_driver.log 2>&1 < /dev/null & ); sleep 30
    fi
  fi
  # ---- E7 eval (launch_e7b.sh starts it; this keeps it going), not while E9 runs
  if has e7train_done && ! alive "launch_e7b?\.sh" && ! e9phase && ! has e7eval_paused; then
    J=""; for s in 20 40 60 80; do [ -f $E7CK/global_step_$s/$W ] && J="$J e7_step_$s|$E7CK/global_step_$s/$W"; done
    ensure_eval e7eval $(( $(echo $J | wc -w) * 60 )) "20 21 22 23 24 25" "0 1 2 5 6 7" $J
  fi
  # ---- R3: v7w
  if [ ! -d $RM/frames_d2w ]; then
    log "build frames_d2w"; CUDA_VISIBLE_DEVICES="" $PYW $S/diag/build_frames_mix.py $RM/frames_d2w d2w $S/diag/d2 > $N/build_d2w.log 2>&1
  fi
  [ -d $RM/frames_d2w ] && ! e9phase && \
    ensure_rm rm_v7w_dual 3 2 $RM/frames_v3:wantrain $RM/frames_d2g:d2gtrain $RM/frames_d2w:d2wwrong $RM/frames_d2w:d2wwrong
  # ---- R2 collection (after E7 training; GPU 3 only after v7w)
  if has e7train_done && ! e9phase && ! has r2build_done && ! has r2build_failed; then
    ensure_collect e7_40 4 40
    done_or_failed rm_v7w_dual && ensure_collect e7_20 3 20
  fi
  if ! done_or_failed r2build && done_or_failed c_e7_40 && done_or_failed c_e7_20; then
    D=""; for n in e7_20 e7_40; do has c_${n}_done && D="$D $S/diag/r2_$n"; done
    if [ -z "$D" ]; then mark r2build_failed
    else log "build frames_r2 from$D"
      rm -rf $RM/frames_r2.tmp
      CUDA_VISIBLE_DEVICES="" $PYW $S/diag/build_frames_mix.py $RM/frames_r2 r2 $D > $N/build_r2.log 2>&1
      [ -d $RM/frames_r2 ] && mark r2build_done || { [ $(bump r2build) -ge 3 ] && mark r2build_failed; }
    fi
  fi
  # ---- v8 (needs R2 data); give up at 07:10
  if ! done_or_failed rm_v8_dual; then
    if has r2build_failed || { [ $T -ge 430 ] && [ $T -lt 600 ]; }; then mark rm_v8_dual_failed
    elif has r2build_done && done_or_failed rm_v7w_dual; then
      ensure_rm rm_v8_dual 3 2 $RM/frames_v3:wantrain $RM/frames_d2g:d2gtrain $RM/frames_r2:r2train \
        $RM/frames_d2w:d2wwrong $RM/frames_r2:r2wrong $RM/frames_d2w:d2wwrong $RM/frames_r2:r2wrong
    fi
  fi
  # ---- gate + select
  if ! has select_done && done_or_failed rm_v7w_dual && done_or_failed rm_v8_dual; then
    M="v7=$RM/runs/rm_v7_dual/best.pt"
    has rm_v7w_dual_done && M="$M v7w=$RM/runs/rm_v7w_dual/best.pt"
    has rm_v8_dual_done && M="$M v8=$RM/runs/rm_v8_dual/best.pt"
    log "gate:$M"
    ( cd $R && PYTHONPATH=$R CUDA_VISIBLE_DEVICES=3 timeout 1800 $PYW $S/diag/gate_rm_d2g.py $M > $N/gate.txt 2>&1 )
    $PYW $S/diag/select_rm.py $M > $N/selected.txt 2> $N/select.err
    if [ $(wc -w < $N/selected.txt) -eq 5 ]; then
      read name path thr k views < $N/selected.txt
      sed -e 's|causal/e7"|causal/e9"|' -e 's|"scgrpo_causal_e7"|"scgrpo_causal_e9"|' -e 's|max_steps: 80|max_steps: 10|' \
          -e 's|resume_dir: .*|resume_dir: null|' -e "s|from_pretrained: .*best.pt|from_pretrained: $path|" \
          -e "s|success_threshold: .*|success_threshold: $thr|" -e "s|success_consecutive: .*|success_consecutive: $k|" \
          -e "s|^# E7: .*|# E9: E7/E8-b config with RM $name (views $views, rule ${thr}x$k), 10 steps|" \
          $C/libero_safety_scgrpo_causal_e7.yaml > $C/libero_safety_scgrpo_causal_e9.yaml
      log "selected: $(cat $N/selected.txt)"; mark select_done
    else log "select failed: $(cat $N/select.err)"; fi
  fi
  # ---- E9 training: after selection, once E7 eval is done or at 06:45 (then E7 eval is paused)
  if has select_done && ! has e9train_done && { has e7eval_done || [ $T -ge 405 ]; } && { has e7train_done || has e7_paused; }; then
    if ! has e9_started; then
      has e7eval_done || { log "pausing E7 eval for E9"; stop_e7_eval; mark e7eval_paused; }
      for p in $(ps -eo pid,args | grep -E "[r]un_r2\.sh|[c]ollect_r2\.py" | awk '{print $1}'); do kill $p; done
      sleep 10
      for p in $(ps -eo pid,args | grep -E "[c]ollect_r2\.py" | awk '{print $1}'); do kill -KILL $p; done
      mark e9_started
    fi
    if grep -qs "TRAIN_DONE e9" causal/e9_driver.log; then mark e9train_done
    elif ! alive "launch_train.sh e9"; then
      log "start/resume E9 training"
      ( TAG=e9 PORT=6681 WR=58000-59999 nohup setsid bash launch_train.sh e9 >> causal/e9_driver.log 2>&1 < /dev/null & ); sleep 30
    fi
  fi
  # ---- E9 eval
  has e9train_done && ensure_eval e9eval 60 "26 27 28 29 30 31" "0 1 2 5 6 7" "e9_step_10|$E9CK/global_step_10/$W"
  # ---- summary (when E9 eval is done, or at 09:40 with whatever exists)
  if ! has summary_done && { has e9eval_done || { [ $T -ge 580 ] && [ $T -lt 900 ]; }; }; then
    NM=""; for n in e9_step_10 e8o_step_10 e8b_step_10 e8a_step_10 r1all_step_10 e7_step_20 e7_step_40 e7_step_60 e7_step_80; do
      [ $(cells_ok $n) -gt 0 ] && NM="$NM $n"; done
    { echo "# morning summary $(date '+%m-%d %H:%M')"
      echo "## selected RM: $(cat $N/selected.txt 2>/dev/null)"; cat $N/select.err 2>/dev/null
      echo "## cells done:"; for n in $NM; do echo "  $n $(cells_ok $n)/60"; done
      echo "## markers: $(ls $N | grep -vE "count|log|txt|err" | tr '\n' ' ')"
      echo "## RM gate (d2gtest)"; grep -vi warn $N/gate.txt 2>/dev/null
      echo "## real eval vs base"
      ( cd $S/diag && CUDA_VISIBLE_DEVICES="" $PYR matched_summ.py $NM 2>&1 | grep -vi warn
        CUDA_VISIBLE_DEVICES="" $PYR level_summ.py $NM 2>&1 | grep -vi warn )
    } > causal/summary_morning.txt 2>&1
    mark summary_done
  fi
  # ---- after the summary: resume what was paused for E9
  if has summary_done && ! e9phase; then
    has e7_paused && { rm -f $N/e7_paused; log "resume E7 training"; }
    has e7eval_paused && { rm -f $N/e7eval_paused $N/e7eval.count; log "resume E7 eval"; }
  fi
  if has summary_done && has e7eval_done && ! e9phase; then log "all done"; exit 0; fi
  sleep 60
done
