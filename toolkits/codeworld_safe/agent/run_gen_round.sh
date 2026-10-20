#!/usr/bin/env bash
# Step D: generate programs for the hard cells with the LLM. Run it yourself:
#   bash /mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/agent/run_gen_round.sh
# The key is read with `read -s` and only lives in this process's environment.
# NOTE: the default provider is plain HTTP; the key crosses the network unencrypted.
set -u
PROVIDER="${PROVIDER:-default}"
ARCHIVE="${ARCHIVE:-/mnt/nfs/fjj/Code/rlinf-safety-runtime/tmp/cws_archive_d1}"
LIBERO=/mnt/nfs/fjj/Code/LIBERO-Safety
PY=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python
CELLS="${CELLS:-human_safety_L0_t0 human_safety_L0_t1 human_safety_L0_t2 human_safety_L0_t3 human_safety_L0_t4 human_safety_L1_t5 human_safety_L1_t6 human_safety_L1_t7 human_safety_L1_t8 human_safety_L1_t9 obstacle_avoidance_human_L1_t7 obstacle_avoidance_human_L1_t8 obstacle_avoidance_human_L1_t9}"
GAP='{"Q1": 0, "Q2": 15, "Q3": 0, "Q4": 40}'
cd "$(dirname "$0")/../../.."
VAR="CWS_$(echo "$PROVIDER" | tr a-z A-Z)_API_KEY"
read -r -s -p "$VAR: " KEY; echo
export "$VAR=$KEY"; unset KEY
mkdir -p "$ARCHIVE"
LOG="$ARCHIVE/gen_$(date +%Y%m%d_%H%M%S).log"
for c in $CELLS; do
  echo "== $c"
  PYTHONPATH=. "$PY" -m toolkits.codeworld_safe.agent.generate --cell "$c" --gap "$GAP" \
    --archive "$ARCHIVE" --libero "$LIBERO" --provider "$PROVIDER" --n 3 2>&1 | tail -4
done | tee "$LOG"
unset "$VAR"
echo "log: $LOG"
