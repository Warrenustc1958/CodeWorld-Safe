#!/usr/bin/env bash
# Wait for the counterfactual collector to finish, then run the LIBERO-Safety
# paper eval (12 cells x 50 fixed trials) with pi0.5's wrist view removed, and
# compare against the dual-view baseline in pi05_libero_safety_paper_5gpu_seed0.
#   WRIST_MODE=mask (default): wrist image zeroed + image_mask False
#   WRIST_MODE=zero          : wrist image zeroed, still attended
set -uo pipefail
RT=/mnt/nfs/fjj/Code/rlinf-safety-runtime
WRIST_MODE="${WRIST_MODE:-mask}"
GPU_RANKS="${GPU_RANKS:-4-5}"          # 50 envs must divide evenly across GPUs
GPU_IDS="${GPU_IDS:-4,5}"
BASE="${RT}/rollouts/pi05_libero_safety_paper_5gpu_seed0"
OUT="${RT}/rollouts/pi05_libero_safety_paper_nowrist_${WRIST_MODE}_seed0"

foreign_gpu_procs() {
  local uuids
  uuids="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F', ' -v ids=",${GPU_IDS}," 'index(ids, ","$1",") {print $2}')"
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
    | grep -F -f <(echo "${uuids}") | wc -l
}

echo "[nowrist] waiting for collector to exit $(date '+%F %T')"
while pgrep -f collect_counterfactual_l0l1.sh >/dev/null; do sleep 300; done
echo "[nowrist] collector exited $(date '+%F %T'); waiting for GPUs ${GPU_IDS}"
while (( $(foreign_gpu_procs) > 0 )); do sleep 120; done

echo "[nowrist] start eval mode=${WRIST_MODE} GPUs=${GPU_RANKS} out=${OUT} $(date '+%F %T')"
RLINF_PI05_WRIST_MODE="${WRIST_MODE}" GPU_RANKS="${GPU_RANKS}" TOTAL_ENVS=50 \
  PAPER_OUTPUT_ROOT="${OUT}" bash "${RT}/eval_pi05_libero_safety.sh"
status=$?
echo "[nowrist] eval exit ${status} $(date '+%F %T')"

python3 "${RT}/tmp/cmp_wrist_eval.py" "${BASE}" "${OUT}" | tee "${OUT}/compare_vs_dualview.txt"
