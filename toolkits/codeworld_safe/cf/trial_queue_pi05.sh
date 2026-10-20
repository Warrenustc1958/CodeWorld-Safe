#!/usr/bin/env bash
# Run pi0.5 trials for archived programs one after another (host lock allows one LIBERO job).
#   ARCHIVE=<dir> bash trial_queue_pi05.sh <program_id> [<program_id> ...]
# Each program is exported from programs.jsonl to $ARCHIVE/programs/<id>.json, trialled with
# trial_pi05.sh, then ingested into the archive. Log: $ARCHIVE/queue_<time>.log
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
REPO=/mnt/nfs/fjj/Code/CodeWorld-Safe
CWS_DATA=${CWS_DATA:-/data_cf/cws}
ARCHIVE=${ARCHIVE:-$CWS_DATA/archive/cws_archive_d1}
PY=$R/.venv-openpi-libero-safety/bin/python
GPU=${GPU:-7}; ENVS=${ENVS:-4}; MAX_BRANCHES=${MAX_BRANCHES:-4}
mkdir -p "$ARCHIVE/programs"
LOG="$ARCHIVE/queue_$(date +%m%d_%H%M).log"
cd "$REPO"
for pid in "$@"; do
  {
    echo "== $(date +%H:%M:%S) $pid"
    P="$ARCHIVE/programs/$pid.json"
    PYTHONPATH=. "$PY" - "$ARCHIVE" "$pid" "$P" <<'EOF' || { echo "export failed"; continue; }
import json, sys
from toolkits.codeworld_safe.agent.archive import Archive
a = Archive(sys.argv[1]); rec = a.program(sys.argv[2])
assert rec.status == "validated", rec.status
open(sys.argv[3], "w").write(json.dumps(rec.program, indent=1))
c = rec.program["cell"]; print(c["suite"], c["level"], c["task_id"])
EOF
    read -r SUITE LEVEL TASK < <(PYTHONPATH=. "$PY" -c "import json;c=json.load(open('$P'))['cell'];print(c['suite'],c['level'],c['task_id'])")
    OUT="$CWS_DATA/rollouts/cws_trial_pi05/${SUITE}_${LEVEL}_t${TASK}_${pid}_$(date +%m%d_%H%M)"
    SUITE=$SUITE LEVEL=$LEVEL TASK=$TASK PROGRAM=$P GPU=$GPU ENVS=$ENVS MAX_BRANCHES=$MAX_BRANCHES OUT=$OUT \
      bash "$REPO/toolkits/codeworld_safe/cf/trial_pi05.sh"
    PYTHONPATH=. OMP_NUM_THREADS=2 "$PY" -m toolkits.codeworld_safe.agent.ingest --audits "$OUT" --archive "$ARCHIVE" 2>&1 \
      | grep -E '"(trials|quadrants|hit_rate|gate4_pass_rate|admitted|duplicates|trigger_hit_rate)"|Q[1-4]"' | tr -d '\n '; echo
  } 2>&1 | tee -a "$LOG"
done
echo "queue done: $LOG" | tee -a "$LOG"
