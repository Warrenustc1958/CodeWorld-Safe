#!/bin/bash
# Phases 2/3/5 (run by systemd cws-rm2.service, Restart=on-failure, KillMode=process; markers in causal/rm2night/):
#   wait for phase-1 collection (lanes A, B) -> build the RM dataset ->
#   R-A (ResNet, GPU 0) | R-B zero-shot eval (GPUs 1,2) | R-C LoRA (GPUs 4-7) -> R-C eval (GPUs 1-3) ->
#   pick the Qwen RM (R-C unless R-B has the higher val frame AP) -> configs E15-A / E15-Q ->
#   train E15-A (10 steps) -> Qwen servers on 8 GPUs + train E15-Q, E15-A eval in parallel -> stop servers ->
#   E15-Q eval. Every stage resumes (marker files, latest checkpoint, finished eval cells).
source /data_cf/cws_sam/v1/scgrpo/night_lib.sh
N=$S/causal/rm2night
mkdir -p $N
cd $S
R2=$S/rm2
DS=$R2/ds_v1
PYO=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python
PYQ=/data_cf/qwen/venv/bin/python
GPUS6="0 1 2 5 6 7"
TRAIN_COLS="colbase col10a col10b col10c col13"
TEST_COLS="col12r"

launcher_alive() {
  local d
  for d in /proc/[0-9]*; do
    tr '\0' ' ' < $d/cmdline 2>/dev/null | grep -q "launch_train.sh" || continue
    tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "TAG=$1" && return 0
  done 2>/dev/null
  return 1
}
train() {  # TAG PORT WR -> returns 0 when TRAIN_DONE
  local tag=$1
  grep -qs "TRAIN_DONE $tag" causal/${tag}_driver.log && return 0
  if ! launcher_alive $tag; then
    log "start/resume $tag"
    ( TAG=$tag PORT=$2 WR=$3 KEEP="10" KEEP_LATEST=1 nohup setsid bash launch_train.sh \
        >> causal/${tag}_driver.log 2>&1 < /dev/null & )
  fi
  return 1
}
server_pids() {  # pids listening on 29700-29707 that are our Qwen RM servers
  local p
  for p in $(ss -ltnp 2>/dev/null | grep -E ":2970[0-7] " | grep -oE "pid=[0-9]+" | cut -d= -f2 | sort -u); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "qwen_rm_server.py" && echo $p
  done
}
start_servers() {  # LORA_OR_EMPTY
  local g lora=$1
  for g in 0 1 2 3 4 5 6 7; do
    curl -s -m 2 http://127.0.0.1:$((29700 + g))/health | grep -q ok && continue
    ( cd $R2 && CUDA_VISIBLE_DEVICES=$g nohup setsid $PYQ qwen_rm_server.py --port $((29700 + g)) \
        ${lora:+--lora $lora} >> $N/qserver_$g.log 2>&1 < /dev/null & )
  done
  for i in $(seq 1 120); do
    local ok=0
    for g in 0 1 2 3 4 5 6 7; do curl -s -m 2 http://127.0.0.1:$((29700 + g))/health | grep -q ok && ok=$((ok + 1)); done
    [ $ok -eq 8 ] && { log "qwen servers up (lora=${lora:-none})"; return 0; }
    sleep 5
  done
  log "qwen servers: only $ok/8 healthy after 10 min"
  return 1
}
stop_servers() {
  local p=$(server_pids)
  [ -n "$p" ] && { kill $p 2>/dev/null; sleep 5; for q in $p; do kill -0 $q 2>/dev/null && kill -9 $q; done; log "qwen servers stopped"; }
}
json() { $PYO -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1]))" "$2"; }
make_cfg() {  # TAG RM_TYPE RM_PATH THRESHOLD
  local tag=$1 F=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config/libero_safety_scgrpo_causal_$1.yaml
  cp /mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config/libero_safety_scgrpo_causal_e10o.yaml $F
  sed -i -e "s#causal/e10o#causal/$tag#; s#scgrpo_causal_e10o#scgrpo_causal_$tag#" \
    -e "s#^  max_steps: 40#  max_steps: 10#; s#^  resume_dir: .*#  resume_dir: null#" \
    -e "s#^    reward_source: real#    reward_source: rm#" \
    -e "s#^      type: CWSRewardModel#      type: $2#; s#^      from_pretrained: .*#      from_pretrained: $3#" \
    -e "s#^    success_threshold: .*#    success_threshold: $4#; s#^    success_consecutive: .*#    success_consecutive: 1#" \
    -e "s#^\# Real goal reward; Wan observations and state reader retained.#\# $tag (phase 5): E10-O with the phase-2 RM $2 ($3) as reward, threshold $4 on the mid and end frame of each chunk.#" \
    -e "/^    success_reward_rule: once/a\\    rm_frames: [3, 7]\\n    cost_source: real\\n    cost_label: grader\\n    dump_dir: $S/causal/$tag/dump" $F
  [ "$2" = QwenRMClient ] && sed -i "/^      from_pretrained: /a\\      port_base: 29700" $F
  grep -nE "reward_source|type: |from_pretrained|port_base|success_threshold|success_consecutive|rm_frames|max_steps" $F
}

log "rm2 pipeline start pid $$"
until [ -f causal/colnight/laneA_done ] && [ -f causal/colnight/laneB_done ]; do sleep 60; done
if ! has ds_done; then
  log "build dataset"
  rm -rf $DS
  $PYO $R2/build_rm_dataset.py $DS --train $TRAIN_COLS --test $TEST_COLS > $N/ds.log 2>&1 && mark ds_done \
    || { log "dataset build failed"; tail -5 $N/ds.log; exit 1; }
fi
if ! has models_done; then
  pids=""
  if ! has ra_done; then
    ( cd $R2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/nfs/fjj/Code/CodeWorld-Safe:. $PYO train_rm_ra.py $DS $R2/ra \
        --epochs 6 --max_fp 0.01 > $N/ra.log 2>&1 && touch $N/ra_done ) & pids="$pids $!"
  fi
  if ! has rb_done; then
    ( cd $R2 && CUDA_VISIBLE_DEVICES=1,2 $PYQ -m torch.distributed.run --nproc_per_node 2 --master_port 29621 \
        rb_eval.py $DS $R2/rb_eval.json --max_fp 0.01 > $N/rb.log 2>&1 && touch $N/rb_done ) & pids="$pids $!"
  fi
  if ! has rc_done; then
    ( cd $R2 && CUDA_VISIBLE_DEVICES=4,5,6,7 $PYQ -m torch.distributed.run --nproc_per_node 4 --master_port 29622 \
        train_rm_rc.py $DS $R2/rc --epochs 1 --bs 8 --max_fp 0.01 > $N/rc.log 2>&1 && touch $N/rc_done ) & pids="$pids $!"
  fi
  wait $pids
  for k in ra rb rc; do has ${k}_done && log "$k done" || log "$k FAILED: $(tail -2 $N/$k.log | tr '\n' ' ')"; done
  has ra_done && has rb_done && has rc_done && mark models_done || exit 1
fi
if ! has rceval_done; then
  ( cd $R2 && CUDA_VISIBLE_DEVICES=1,2,3 $PYQ -m torch.distributed.run --nproc_per_node 3 --master_port 29623 \
      rb_eval.py $DS $R2/rc_eval.json --lora $R2/rc/best --max_fp 0.01 > $N/rceval.log 2>&1 ) && mark rceval_done \
    || { log "R-C eval failed"; exit 1; }
fi
if ! has cfg_done; then
  ap_b=$(json $R2/rb_eval.json "d['val']['frame_ap'] or 0"); ap_c=$(json $R2/rc_eval.json "d['val']['frame_ap'] or 0")
  if $PYO -c "import sys; sys.exit(0 if float('$ap_c') >= float('$ap_b') else 1)"; then
    QLORA=$R2/rc/best; QTHR=$(json $R2/rc_eval.json "d['threshold']"); QNAME=R-C
  else
    QLORA=""; QTHR=$(json $R2/rb_eval.json "d['threshold']"); QNAME=R-B
  fi
  echo "{\"name\": \"$QNAME\", \"lora\": \"$QLORA\", \"threshold\": $QTHR, \"ap_rb\": $ap_b, \"ap_rc\": $ap_c}" > $N/qchoice.json
  log "Qwen RM choice: $(cat $N/qchoice.json)"
  make_cfg e15a CWSRewardModel $R2/ra/best.pt $(json $R2/ra/final.json "d['threshold']") > $N/cfg.log
  make_cfg e15q QwenRMClient qwen3vl_4b${QLORA:+_lora} $QTHR >> $N/cfg.log
  mark cfg_done
fi
QLORA=$(json $N/qchoice.json "d['lora']")
K=checkpoints/global_step_10/$W
while true; do
  if ! has e15a_train_done; then
    train e15a 6831 58000-59999 && mark e15a_train_done
  elif ! has e15q_train_done; then
    [ -n "$(server_pids)" ] && [ $(server_pids | wc -l) -ge 8 ] || start_servers "$QLORA"
    train e15q 6841 56000-57999 && { mark e15q_train_done; stop_servers; }
  fi
  has e15a_train_done && ! has e15a_eval_done && \
    ensure_eval e15a_eval 60 "20 21 22 23 24 25" "$GPUS6" "e15a_step_10|$S/causal/e15a/scgrpo_causal_e15a/$K"
  has e15q_train_done && ! has e15q_eval_done && \
    ensure_eval e15q_eval 60 "26 27 28 29 30 31" "$GPUS6" "e15q_step_10|$S/causal/e15q/scgrpo_causal_e15q/$K"
  if has e15a_eval_done && has e15q_eval_done; then mark rm2_done; log "phase 2/3/5 done"; exit 0; fi
  sleep 60
done
