#!/usr/bin/env bash
# Generate cws_cf_program/v0 programs with the LLM for Step E cells. Run it yourself:
#   bash /mnt/nfs/fjj/Code/CodeWorld-Safe/toolkits/codeworld_safe/agent/run_gen_round.sh
# Cells: by default every cell without a validated program in the Step E archive.
# Gap per cell: read from $STEP_E/gaps.json (written by campaign_step_e after each run);
# cells the campaign has not visited yet get the full quota.
# The key is read with `read -s` and only lives in this process's environment.
# NOTE: the default provider is plain HTTP; the key crosses the network unencrypted.
# Safe to run while campaign_step_e is running: it only appends to programs.jsonl,
# and the campaign reloads the archive before every cell.
set -u
PROVIDER="${PROVIDER:-default}"
STEP_E="${STEP_E:-/data_cf/cws/step_e}"
ARCHIVE="${ARCHIVE:-$STEP_E/archive}"
LIBERO=/mnt/nfs/fjj/Code/LIBERO-Safety
PY=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python
N="${N:-3}"
cd "$(dirname "$0")/../../.."
if [ -z "${CELLS:-}" ]; then
  CELLS=$(python3 - "$ARCHIVE" <<'EOF'
import glob, json, os, sys
have = set()
for line in open(os.path.join(sys.argv[1], "programs.jsonl")):
    p = json.loads(line)
    if p.get("status") == "validated":
        have.add(p["cell"])
cells = sorted(os.path.basename(f)[:-5] for f in glob.glob("toolkits/codeworld_safe/agent/knowledge/tasks/*.json"))
print(" ".join(c for c in cells if c not in have))
EOF
)
fi
echo "cells: $CELLS"
VAR="CWS_$(echo "$PROVIDER" | tr a-z A-Z)_API_KEY"
read -r -s -p "$VAR: " KEY; echo
export "$VAR=$KEY"; unset KEY
LOG="$ARCHIVE/gen_$(date +%Y%m%d_%H%M%S).log"
for c in $CELLS; do
  GAP=$(python3 - "$STEP_E/gaps.json" "$c" <<'EOF'
import json, os, sys
full = {"Q1": 50, "Q2": 20, "Q3": 50, "Q4": 20}
g = json.load(open(sys.argv[1])).get(sys.argv[2]) if os.path.exists(sys.argv[1]) else None
print(json.dumps({q: int((g or full).get(q, 0)) for q in full}))
EOF
)
  echo "== $c gap=$GAP"
  PYTHONPATH=. "$PY" -m toolkits.codeworld_safe.agent.generate --cell "$c" --gap "$GAP" \
    --archive "$ARCHIVE" --libero "$LIBERO" --provider "$PROVIDER" --n "$N" 2>&1 | tail -4
done | tee "$LOG"
unset "$VAR"
echo "log: $LOG"
