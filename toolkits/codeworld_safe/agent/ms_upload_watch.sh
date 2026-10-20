#!/usr/bin/env bash
# Periodically upload finished Step E cells to ModelScope and verify every file.
#   tmux new-session -d -s cwse-ms 'bash toolkits/codeworld_safe/agent/ms_upload_watch.sh'
# A cell is uploaded once campaign_step_e marks it pending_upload and it has no
# upload_verify.json with "ok": true. Failures are retried next pass. Never deletes.
set -u
STEP_E=${STEP_E:-/data_cf/cws/step_e}
INTERVAL=${INTERVAL:-1800}
export MS_TOKEN_FILE=${MS_TOKEN_FILE:-/mnt/nfs/fjj/.modelscope_fjj/sdk_token_cws}
export MODELSCOPE_CREDENTIALS_PATH=/mnt/nfs/fjj/.modelscope_fjj
REPO=/mnt/nfs/fjj/Code/CodeWorld-Safe
LOG=$STEP_E/ms_upload.log
cd "$REPO"
say(){ echo "[ms $(date '+%F %T')] $*" | tee -a "$LOG"; }
say "watcher start, interval ${INTERVAL}s"
while :; do
  CELLS=$(python3 - "$STEP_E" <<'EOF'
import json, os, sys
root = sys.argv[1]
st = json.load(open(os.path.join(root, "state.json")))
out = []
for cell, s in st.items():
    if s.get("status") not in ("pending_upload", "uploaded"):
        continue
    v = os.path.join(root, "runs", cell, "upload_verify.json")
    if os.path.exists(v) and json.load(open(v)).get("ok"):
        continue
    out.append(cell)
print(" ".join(sorted(out)))
EOF
)
  if [ -n "$CELLS" ]; then
    for c in $CELLS; do
      say "upload $c"
      /usr/bin/python3 toolkits/codeworld_safe/agent/upload_cell.py "$STEP_E/runs/$c" "$c" >>"$LOG" 2>&1 \
        && say "verified $c" || say "FAILED $c (retry next pass)"
    done
  fi
  ok=$(ls "$STEP_E"/runs/*/upload_verify.json 2>/dev/null | xargs -r grep -l '"ok": true' | wc -l)
  say "pass done: pending [${CELLS:-none}], verified cells total $ok"
  if grep -q "no active cells left" "$STEP_E/campaign.log" 2>/dev/null && [ -z "$CELLS" ]; then
    say "campaign finished and nothing pending; exit"; exit 0
  fi
  sleep "$INTERVAL"
done
