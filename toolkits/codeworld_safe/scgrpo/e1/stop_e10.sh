#!/bin/bash
# Stop E10-O completely (user 10-31): disable the 5-min watchdog, stop the service, stop our own leftovers
# (launch_train.sh e10o, run_causal.sh and every process with RAY_TMPDIR=/tmp/rp_ce10o). Checkpoints are kept.
systemctl disable --now cws-e10-watch.timer 2>&1 | tail -1
systemctl stop cws-e10.service
desc() { for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
ALL=""
for q in $(ps -eo pid,args | grep -E "[l]aunch_train\.sh e10o|[e]10_pipeline\.sh" | awk '{print $1}'); do ALL="$ALL $q $(desc $q)"; done
for d in /proc/[0-9]*; do
  tr '\0' '\n' < $d/environ 2>/dev/null | grep -qx "RAY_TMPDIR=/tmp/rp_ce10o" && ALL="$ALL ${d#/proc/}"
done
ALL=$(echo $ALL | tr ' ' '\n' | grep -vx "$$" | sort -u)
echo "stopping $(echo $ALL | wc -w) E10-O processes"
[ -n "$ALL" ] && { kill -TERM $ALL 2>/dev/null; sleep 10; for q in $ALL; do kill -0 $q 2>/dev/null && kill -KILL $q 2>/dev/null; done; }
sleep 3
echo "timer: $(systemctl is-active cws-e10-watch.timer)  service: $(systemctl is-active cws-e10.service)"
echo "left: $(ps -eo args | grep -cE '[l]aunch_train\.sh e10o|[r]un_causal\.sh|[/]tmp/rp_ce10o/')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
ls /data_cf/cws_sam/v1/scgrpo/causal/e10o/scgrpo_causal_e10o/checkpoints 2>/dev/null
