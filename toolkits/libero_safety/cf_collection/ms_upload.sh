#!/usr/bin/env bash
# Upload finished counterfactual round dirs to the ModelScope dataset repo.
# One `modelscope upload` per round dir; finished dirs are recorded in
# ${DONE_FILE} so re-running skips them. Local data is NOT deleted.
#
# Usage: bash ms_upload.sh <round_dir> [<round_dir> ...]
set -uo pipefail
REPO="${REPO:-Warrenustc1958/LIBERO-Safety-Pi05-rollout}"
COLLECT_ROOT="${COLLECT_ROOT:-/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/cf_l0l1_g006}"
DONE_FILE="${DONE_FILE:-${COLLECT_ROOT}/state/ms_uploaded.txt}"
WORKERS="${WORKERS:-8}"
RETRIES="${RETRIES:-3}"
# root's ~/.modelscope is shared with other users' jobs on this host; keep our
# login in a separate credentials dir so neither side overwrites the other.
export MODELSCOPE_CREDENTIALS_PATH="${MODELSCOPE_CREDENTIALS_PATH:-/mnt/nfs/fjj/.modelscope_fjj}"
if ! python3 -c "from modelscope.hub.api import ModelScopeConfig as C; import sys; sys.exit(0 if C.get_cookies() else 1)" 2>/dev/null; then
  echo "[abort] not logged in under ${MODELSCOPE_CREDENTIALS_PATH}; run:" >&2
  echo "  MODELSCOPE_CREDENTIALS_PATH=${MODELSCOPE_CREDENTIALS_PATH} modelscope login --token <SDK token>" >&2
  exit 3
fi
touch "${DONE_FILE}"

for dir in "$@"; do
  name="$(basename "${dir%/}")"
  if grep -qxF "${name}" "${DONE_FILE}"; then
    echo "[skip] ${name} already uploaded"
    continue
  fi
  size="$(du -sh "${dir}" | cut -f1)"
  nfiles="$(find "${dir}" -type f | wc -l)"
  for attempt in $(seq 1 "${RETRIES}"); do
    echo "[upload] ${name} (${size}, ${nfiles} files) attempt ${attempt} $(date '+%F %T')"
    start=$(date +%s)
    if modelscope upload "${REPO}" "${dir}" "cf_l0l1_g006/runs/${name}" \
        --repo-type dataset --max-workers "${WORKERS}" \
        --commit-message "Add ${name}"; then
      echo "${name}" >> "${DONE_FILE}"
      echo "[done] ${name} in $(( $(date +%s) - start ))s"
      break
    fi
    echo "[fail] ${name} attempt ${attempt}" >&2
    sleep 60
  done
done
echo "ALL DONE $(date '+%F %T')"
