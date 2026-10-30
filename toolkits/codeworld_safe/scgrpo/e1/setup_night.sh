#!/bin/bash
# Night plan step 1 (10-30 ~02:00):
#  - stop all eval chains/waiters (p1-p13) and their Ray heads (incl. leftovers matched by RAY_TMPDIR)
#  - new chain scripts p20-p31: own Ray port/client port/worker ports/temp dir/lock, 20-min per-cell timeout
#  - E1 remaining cells (r1all_step_20, rmall_step_10, rmall_step_20) on GPUs 5,6,7 (chains p23-p25)
#  - E2 = RM-all but success termination from the real simulator; GPUs 0-3, 20 steps, save every 10;
#    then eval step 10/20 on GPUs 0,1,2 (chains p20-p22)
set -uo pipefail
S=/data_cf/cws_sam/v1/scgrpo
C=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment/config
cd $S
# 1) stop old eval chains and waiters
desc() { for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
ALL=""
for q in $(ps -eo pid,args | grep -E "eval_real_p[0-9]+\.sh|/tmp/rp_p[0-9]+/|ALL_DONE|train_rmall\.log" | grep -v grep | grep -v setup_night | awk '{print $1}'); do
  ALL="$ALL $q $(desc $q)"
done
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qE "^RAY_TMPDIR=/tmp/rp_p[0-9]+$" && ALL="$ALL ${d#/proc/}"
done
ALL=$(echo $ALL | tr ' ' '\n' | grep -vx "$$" | sort -u)
echo "stopping $(echo $ALL | wc -w) old eval procs"
kill -TERM $ALL 2>/dev/null; sleep 10
for q in $ALL; do kill -0 "$q" 2>/dev/null && kill -KILL "$q" 2>/dev/null; done
sleep 3
echo "left: $(ps -eo args | grep -cE "[e]val_real_p[0-9]+\.sh|[e]val_embodied_agent")"
# 2) chain scripts p20-p31
for i in $(seq 20 31); do
  p=$((6600 + 10 * i)); lo=$((20000 + 1000 * i)); hi=$((lo + 999))
  for q in $p $((p + 3000)); do ss -ltn | grep -q ":$q " && { echo "port $q busy"; exit 3; }; done
  sed -e "s|rp_ev5|rp_p$i|g; s|p5_ev5|p5_p$i|g; s|run-ev5.lock|run-p$i.lock|g; s|RAY_PORT:-6531|RAY_PORT:-$p|" \
      -e "s|--min-worker-port=20000 --max-worker-port=20999|--min-worker-port=$lo --max-worker-port=$hi --ray-client-server-port=$((p + 3000))|" \
      -e "s|  bash evaluations/run_safelibero_pi05.sh|  timeout -k 30 1200 bash evaluations/run_safelibero_pi05.sh|" \
      eval_real_ev5.sh > eval_real_p$i.sh
  grep -q "RAY_PORT:-$p" eval_real_p$i.sh && grep -q "min-worker-port=$lo" eval_real_p$i.sh \
    && grep -q "timeout -k 30 1200" eval_real_p$i.sh || { echo "p$i sed failed"; exit 4; }
done
# 3) E1 remaining
W=actor/model_state_dict/full_weights.pt
R1=$S/causal/r1all/scgrpo_causal_r1all/checkpoints; RM=$S/causal/rmall/scgrpo_causal_rmall/checkpoints
JOBS="rmall_step_10|$RM/global_step_10/$W r1all_step_20|$R1/global_step_20/$W rmall_step_20|$RM/global_step_20/$W" \
  GPUS_LIST="5 6 7" CHAINS="23 24 25" nohup bash run_multi.sh > causal/e1_finish_launch.log 2>&1 < /dev/null &
# 4) E2 config + training head with its own worker-port range
sed -e 's|actor,env,rollout: 4-7|actor,env,rollout: 0-3|' \
    -e 's|causal/rmall"|causal/e2"|' -e 's|"scgrpo_causal_rmall"|"scgrpo_causal_e2"|' \
    -e 's|save_interval: 5|save_interval: 10|' \
    -e 's|termination_source: rm .*|termination_source: real           # E2: episode ends on REAL success/violation; reward is still RM|' \
    -e 's|^# E1 RM-all: .*|# E2: RM-all with real-simulator success termination (only termination differs from RM-all), GPUs 0-3|' \
    $C/libero_safety_scgrpo_causal_rmall.yaml > $C/libero_safety_scgrpo_causal_e2.yaml
diff $C/libero_safety_scgrpo_causal_rmall.yaml $C/libero_safety_scgrpo_causal_e2.yaml
grep -q 'RAY_EXTRA' run_causal.sh || sed -i 's|--disable-usage-stats > \$RAY_TMPDIR/ray_start.log|${RAY_EXTRA:-} --disable-usage-stats > $RAY_TMPDIR/ray_start.log|' run_causal.sh
grep -n "RAY_EXTRA" run_causal.sh
cat > run_e2.sh <<EOF2
#!/bin/bash
cd $S
RAY_EXTRA="--min-worker-port=52000 --max-worker-port=53999" TAG=e2 PORT=6521 bash run_causal.sh > causal/train_e2.log 2>&1
echo "TRAIN \$(tail -1 causal/train_e2.log)"
CK=$S/causal/e2/scgrpo_causal_e2/checkpoints
JOBS="e2_step_10|\$CK/global_step_10/$W e2_step_20|\$CK/global_step_20/$W" GPUS_LIST="0 1 2" CHAINS="20 21 22" bash run_multi.sh
echo E2_EVAL_LAUNCHED
EOF2
nohup bash run_e2.sh > causal/e2.log 2>&1 < /dev/null &
sleep 1; echo NIGHT_STEP1_DONE
