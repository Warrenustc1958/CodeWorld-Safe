#!/bin/bash
# E2 training failed (worker-port range 52000-53999 contains the dashboard agent default 52365), but
# run_e2.sh went on to start eval chains p20-p22 on missing checkpoints. Stop only run_e2.sh's process
# tree and the E2 Ray head, then relaunch with worker ports 56000-57999 and train-success gating.
S=/data_cf/cws_sam/v1/scgrpo
cd $S
desc() { for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
ALL=""
for q in $(ps -eo pid,args | grep -E "bash run_e2\.sh|eval_real_p2[0-2]\.sh|/tmp/rp_p2[0-2]/" | grep -v grep | awk '{print $1}'); do
  ALL="$ALL $q $(desc $q)"
done
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qE "^RAY_TMPDIR=/tmp/rp_p2[0-2]$" && ALL="$ALL ${d#/proc/}"
done
ALL=$(echo $ALL | tr ' ' '\n' | grep -vx "$$" | sort -u)
echo "stopping $(echo $ALL | wc -w) E2 procs"
kill -TERM $ALL 2>/dev/null; sleep 8
for q in $ALL; do kill -0 "$q" 2>/dev/null && kill -KILL "$q" 2>/dev/null; done
TAG=e2 bash stop_causal_head.sh
TS=$(date +%H%M)   # cells from missing checkpoints: kept aside, never mixed into results
for n in e2_step_10 e2_step_20; do [ -d eval_real/$n ] && mv eval_real/$n eval_real/${n}_failed_${TS}_nockpt; done
echo "E1 chains still alive: $(ps -eo args | grep -cE "[e]val_real_p2[3-5]\.sh")"
W=actor/model_state_dict/full_weights.pt
cat > run_e2.sh <<EOF2
#!/bin/bash
cd $S
RAY_EXTRA="--min-worker-port=56000 --max-worker-port=57999" TAG=e2 PORT=6521 bash run_causal.sh > causal/train_e2.log 2>&1
rc=\$?
echo "TRAIN rc=\$rc \$(tail -1 causal/train_e2.log)"
[ \$rc -eq 0 ] || exit \$rc
CK=$S/causal/e2/scgrpo_causal_e2/checkpoints
JOBS="e2_step_10|\$CK/global_step_10/$W e2_step_20|\$CK/global_step_20/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
echo E2_EVAL_LAUNCHED
EOF2
nohup bash run_e2.sh > causal/e2.log 2>&1 < /dev/null &
sleep 1; echo E2_RELAUNCHED
