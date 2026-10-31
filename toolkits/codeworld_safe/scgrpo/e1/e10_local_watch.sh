#!/bin/bash
# Local watchdog for E10-O and E11 on gpu006 (run every 5 min by Windows Task Scheduler "CWS_E10_Watch").
# Per experiment: if neither its pipeline, nor a training process, nor an eval process has been seen for 20 min
# (and it is not finished), re-enable its server timer and restart its systemd service (resumes from the latest
# checkpoint). E11 is only watched once E10-O training has finished (E11 waits for it). The task deletes
# itself when both are finished. Log: e10_local_watch.log ; state: e10_last_seen.txt / e11_last_seen.txt
export PATH=/usr/bin:$PATH
D=/c/Users/26692/cws_diag
LOG=$D/e10_local_watch.log
STALL=1200
now=$(date +%s)
ts() { date '+%m-%d %H:%M:%S'; }
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ClearAllForwardings=yes gpu006-root"
out=$($SSH '
  cd /data_cf/cws_sam/v1/scgrpo
  echo E10DONE=$([ -f causal/e10night/e10o_eval_done ] && echo 1 || echo 0)
  echo E10TRAINED=$(grep -qs "TRAIN_DONE e10o" causal/e10o_driver.log && echo 1 || echo 0)
  echo E11DONE=$([ -f causal/e11night/e11sam_eval_done ] && echo 1 || echo 0)
  echo E10ALIVE=$(ps -eo args | grep -cE "[e]10_pipeline\.sh|[c]onfig-name libero_safety_scgrpo_causal_e10o|[e]val_real/e10o_step")
  echo E11ALIVE=$(ps -eo args | grep -cE "[e]11_pipeline\.sh|[c]onfig-name libero_safety_scgrpo_causal_e11s|[e]val_real/e11s")
  echo SVC10=$(systemctl is-active cws-e10.service) SVC11=$(systemctl is-active cws-e11.service)
  echo STEP10=$(grep -aoE "Global Step: *[0-9]+/[0-9]+" causal/train_e10o.log 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+")
  for t in e11sc e11sam; do echo STEP_$t=$(grep -aoE "Global Step: *[0-9]+/[0-9]+" causal/train_$t.log 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+"); done
' 2>&1)
rc=$?
if [ $rc -ne 0 ]; then echo "[$(ts)] ssh failed rc=$rc: $(echo "$out" | tail -1)" >> $LOG; exit 0; fi
v() { echo "$out" | grep -oE "(^| )$1=[^ ]*" | head -1 | cut -d= -f2; }
line=$(echo "$out" | tr '\n' ' ')
if [ "$(v E10DONE)" = 1 ] && [ "$(v E11DONE)" = 1 ]; then
  echo "[$(ts)] E10-O and E11 finished; removing the scheduled task. $line" >> $LOG
  schtasks //Delete //TN CWS_E10_Watch //F >> $LOG 2>&1
  exit 0
fi
check() {  # NAME UNIT ALIVE_COUNT
  local name=$1 unit=$2 alive=$3 seen=$D/$1_last_seen.txt
  [ -f $seen ] || echo $now > $seen
  if [ "${alive:-0}" -gt 0 ]; then echo $now > $seen; return; fi
  local idle=$(( now - $(cat $seen) ))
  echo "[$(ts)] $name: no process for ${idle}s" >> $LOG
  if [ $idle -ge $STALL ]; then
    echo "[$(ts)] $name: >= ${STALL}s idle, restarting $unit" >> $LOG
    $SSH "systemctl enable --now $unit-watch.timer; systemctl restart $unit.service; sleep 3; \
          echo \"$unit: \$(systemctl is-active $unit.service) timer \$(systemctl is-active $unit-watch.timer)\"" >> $LOG 2>&1
    echo $now > $seen
  fi
}
echo "[$(ts)] $line" >> $LOG
[ "$(v E10DONE)" = 1 ] || check e10 cws-e10 "$(v E10ALIVE)"
[ "$(v E10TRAINED)" = 1 ] && [ "$(v E11DONE)" != 1 ] && check e11 cws-e11 "$(v E11ALIVE)"
exit 0
