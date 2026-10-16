#!/usr/bin/env bash
# Upload -> verify -> prune for finished cells of the cf_l0l1_g006 collection.
#   1. upload every finished round dir of CELLS (skips the round the collector
#      is writing and dirs already listed in state/ms_uploaded.txt)
#   2. wait until each cell reports "no pending tasks" (or the collector exits),
#      then upload the remaining rounds of that cell
#   3. verify path+size of every round of the cell against the ModelScope repo
#   4. only if verification is clean: cf_prune.py --cap ${CAP} --apply
# Usage: bash cf_upload_prune.sh obstacle_avoidance_L0 obstacle_avoidance_L1
set -uo pipefail
RT=/mnt/nfs/fjj/Code/rlinf-safety-runtime
R="${COLLECT_ROOT:-${RT}/rollouts/cf_l0l1_g006}"
LOG="${COLLECT_LOG:?set COLLECT_LOG to the running collector log}"
CAP="${CAP:-50}"
export MODELSCOPE_CREDENTIALS_PATH="${MODELSCOPE_CREDENTIALS_PATH:-/mnt/nfs/fjj/.modelscope_fjj}"
DONE_FILE="${R}/state/ms_uploaded.txt"

collector_running() { pgrep -f collect_counterfactual_l0l1.sh >/dev/null; }
cell_done() { grep -aq "^\[cell\] $1: no pending tasks" "${LOG}"; }
# Round dir currently being written (last "[round] output=" while the collector runs).
active_dir() {
  collector_running || return 0
  grep -a '^\[round\] output=' "${LOG}" | tail -1 | sed 's#.*/##'
}
cell_dirs() {
  # Only dirs that contain audits (aborted rounds leave empty dirs).
  for d in "${R}/runs/$1_"*/; do
    [[ -d "${d}" ]] || continue
    find "${d}" -path '*libero_safety_audits*' -name '*.json' -print -quit | grep -q . && basename "${d}"
  done
}
upload_finished() {
  local cell="$1" act todo=()
  act="$(active_dir)"
  for n in $(cell_dirs "${cell}"); do
    [[ "${n}" == "${act}" ]] && continue
    grep -qxF "${n}" "${DONE_FILE}" 2>/dev/null && continue
    todo+=("${R}/runs/${n}")
  done
  (( ${#todo[@]} )) && bash "${RT}/scripts/ms_upload.sh" "${todo[@]}"
  return 0
}

for cell in "$@"; do upload_finished "${cell}"; done

for cell in "$@"; do
  echo "=== ${cell}: waiting for cell completion $(date '+%F %T')"
  while ! cell_done "${cell}"; do
    if ! collector_running; then
      echo "[warn] collector stopped before ${cell} finished; uploading what exists, not pruning" >&2
      upload_finished "${cell}"
      continue 2
    fi
    upload_finished "${cell}"
    sleep 300
  done
  upload_finished "${cell}"
  mapfile -t names < <(cell_dirs "${cell}")
  missing=0
  for n in "${names[@]}"; do grep -qxF "${n}" "${DONE_FILE}" || { echo "[skip-prune] ${n} not uploaded" >&2; missing=1; }; done
  (( missing )) && continue
  echo "=== ${cell}: verifying ${#names[@]} dirs $(date '+%F %T')"
  out="$(cd "${RT}" && python3 tmp/ms_verify.py "${R}" "${names[@]}" 2>/dev/null)"
  echo "${out}"
  if ! grep -qx 'TOTAL_BAD 0' <<< "${out}"; then
    echo "[skip-prune] ${cell}: verification not clean" >&2
    continue
  fi
  suite="${cell%_L*}"; level="${cell##*_}"
  echo "=== ${cell}: pruning cap=${CAP} $(date '+%F %T')"
  python3 "${RT}/tmp/cf_prune.py" "${R}" "${suite}" "${level}" --cap "${CAP}" --apply
done
df -h "${R}" | tail -1
echo "ALL DONE $(date '+%F %T')"
