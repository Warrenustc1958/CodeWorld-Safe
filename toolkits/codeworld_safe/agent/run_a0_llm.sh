#!/usr/bin/env bash
# Step A0 LLM checks. Run it yourself in an interactive terminal on gpu006:
#   bash /mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/agent/run_a0_llm.sh
# The key is read with `read -s` and lives only in this process's environment.
# NOTE: the default provider is plain HTTP; the key crosses the network unencrypted.
set -u
PROVIDER="${PROVIDER:-default}"
TASKS="${TASKS:-obstacle_avoidance_human_L1_t9 human_safety_L0_t4}"
OUT="${OUT:-/mnt/nfs/fjj/Code/rlinf-safety-runtime/tmp/cws_a0}"
LIBERO=/mnt/nfs/fjj/Code/LIBERO-Safety
cd "$(dirname "$0")"
VAR="CWS_$(echo "$PROVIDER" | tr a-z A-Z)_API_KEY"
read -r -s -p "$VAR: " KEY; echo
export "$VAR=$KEY"; unset KEY
mkdir -p "$OUT"
LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
{
  for role in planner worker; do
    echo "== ping $PROVIDER $role"
    python3 llm_client.py --provider "$PROVIDER" --ping --role "$role" 2>&1 | tail -1
  done
  [ "${PING_ONLY:-0}" = 1 ] && TASKS=""
  for t in $TASKS; do
    echo "== analysis $t"
    python3 failure_analysis.py --provider "$PROVIDER" --task "$t" --out "$OUT"
    echo "== analysis $t (blind)"
    python3 failure_analysis.py --provider "$PROVIDER" --task "$t" --out "$OUT" --blind --libero "$LIBERO"
  done
} 2>&1 | tee "$LOG"
unset "$VAR"
echo "log: $LOG"
