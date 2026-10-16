#!/usr/bin/env bash
# LIBERO-Safety four-quadrant counterfactual collection for Wan2.2-5B finetuning.
#
# Training split only: L0/L1 of the 4 physical-safety suites (40 tasks); L2 is
# held out for evaluation. Order: human_safety -> obstacle_avoidance ->
# obstacle_avoidance_human -> affordance, L0 before L1 inside each suite.
#
# Per cell, every round runs ONE RLinf job over all still-pending tasks, then
# scripts/cf_coverage.py re-scans the audits, prints per-task coverage
# (Q1..Q4 counts + distinct roots), applies stall detection and decides which
# tasks stay pending. State lives in ${COLLECT_ROOT}/state, so re-running the
# script with the same COLLECT_ROOT resumes and skips finished cells.
#
# Quotas (valid episodes = audit JSON + matching visual NPZ):
#   Q1 >= Q1_MIN, Q3 >= Q3_MIN           each with >= MIN_ROOTS distinct roots
#   Q2+Q4 >= UNSAFE_MIN                   with >= MIN_UNSAFE_ROOTS distinct roots
#   Q2 >= Q2_SOFT                         soft target
# affordance uses AFF_UNSAFE_MIN / AFF_Q2_SOFT (Q2 is near-infeasible there).
# A requirement with no (count, roots) growth for STALL_ROUNDS rounds is
# marked infeasible and no longer drives rounds.
#
# Toggles (default ON):
#   BRANCH_STRATEGY=auto      hazard-directed multi-step branches ("legacy" = old)
#   AUX_SAFETY=1              wrong-part-grasp / hand-contact unsafe predicates
#   Q3_ROOT_SUITES=affordance suites whose safe timeouts (Q3) also open branches
#                             toward Q4 ("" disables)
#
# Usage:
#   bash scripts/collect_counterfactual_l0l1.sh            # full collection
#   SMOKE=1 bash scripts/collect_counterfactual_l0l1.sh    # 1 task, 1 round
set -Eeuo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/nfs/fjj/Code/rlinf-safety-runtime}"
# export.sh sets RLINF_ROOT / LIBERO_SAFETY_PATH / VENV_PATH / MODEL and cds
# into RLINF_ROOT. Keep its OUTPUT_ROOT from leaking into our runs.
# shellcheck disable=SC1091
source "${RUNTIME_ROOT}/export.sh"
unset OUTPUT_ROOT
source "${VENV_PATH}/bin/activate"
cd "${RLINF_ROOT}"
COVERAGE_PY="${RUNTIME_ROOT}/scripts/cf_coverage.py"

# ---- one RLinf/LIBERO job per host (run_safelibero_pi05.sh honours this) ----
RLINF_LIBERO_RUN_LOCK_PATH="${RLINF_LIBERO_RUN_LOCK_PATH:-/tmp/rlinf-libero-run.lock}"
exec 9>"${RLINF_LIBERO_RUN_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another RLinf/LIBERO job owns ${RLINF_LIBERO_RUN_LOCK_PATH}; refusing to start." >&2
  exit 75
fi
export RLINF_LIBERO_RUN_LOCK_HELD=1

# ---- runtime environment ----
export LIBERO_TYPE=safety
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${VENV_PATH}/share/libero-safety}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${LIBERO_SAFETY_PATH}:${RLINF_ROOT}:${PYTHONPATH:-}"

# ImageMagick (wand) from the runtime prefix; needed by the LIBERO import chain.
IMAGEMAGICK_PREFIX="${IMAGEMAGICK_PREFIX:-${RUNTIME_ROOT}/local-packages/imagemagick/prefix}"
IM_LIB="${IMAGEMAGICK_PREFIX}/usr/lib/x86_64-linux-gnu"
export MAGICK_HOME="${IMAGEMAGICK_PREFIX}/usr"
export MAGICK_CONFIGURE_PATH="${IMAGEMAGICK_PREFIX}/etc/ImageMagick-6:${IM_LIB}/ImageMagick-6.9.11/config-Q16:${IMAGEMAGICK_PREFIX}/usr/share/ImageMagick-6"
export MAGICK_CODER_MODULE_PATH="${IM_LIB}/ImageMagick-6.9.11/modules-Q16/coders"
export MAGICK_FILTER_MODULE_PATH="${IM_LIB}/ImageMagick-6.9.11/modules-Q16/filters"
export LD_LIBRARY_PATH="${IM_LIB}:${LD_LIBRARY_PATH:-}"

# gpu003: physical GPUs 0-3 belong to another user's job; use 4-7 only.
# RLinf's scheduler writes each worker's CUDA_VISIBLE_DEVICES from the
# placement ranks, so GPU_RANKS must be *physical* ids and a parent-level
# CUDA_VISIBLE_DEVICES mask is ignored (it would only confuse GPU counting).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
export GPU_RANKS="${GPU_RANKS:-4-7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RLINF_TIMEOUT="${RLINF_TIMEOUT:-720}"
# The per-GPU EGL lock (venv.py:_egl_process_guard) wraps the WHOLE env.step
# (physics + render), so all envs on one GPU step strictly one after another:
# 25 envs/GPU measured 2.1 s/step. EGL_LOCK=0 lets env subprocesses render
# concurrently, but on gpu003 it hung before the first rollout step, so the
# lock stays on by default.
EGL_LOCK="${EGL_LOCK:-1}"
if [[ "${EGL_LOCK}" == "1" ]]; then
  export RLINF_LIBERO_EGL_LOCK_PATH="${RLINF_LIBERO_EGL_LOCK_PATH:-/tmp/rlinf-libero-egl.lock}"
  export RLINF_LIBERO_EGL_LOCK_SCOPE="${RLINF_LIBERO_EGL_LOCK_SCOPE:-per_gpu}"
else
  unset RLINF_LIBERO_EGL_LOCK_PATH RLINF_LIBERO_EGL_LOCK_SCOPE
fi
# THREAD_CAP=1 keeps BLAS/OpenMP pools at one thread per process (useful with
# hundreds of env subprocesses). Off by default: untested with a full run.
if [[ "${THREAD_CAP:-0}" == "1" ]]; then
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fi
unset RAY_ADDRESS
RUN_TOKEN="$(date +%Y%m%d_%H%M%S)"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rq-cf-${RUN_TOKEN}}"
export TMPDIR="${TMPDIR:-/tmp/pq-cf-${RUN_TOKEN}}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

# ---- collection knobs ----
SMOKE="${SMOKE:-0}"
BRANCH_STRATEGY="${BRANCH_STRATEGY:-auto}"
AUX_SAFETY="${AUX_SAFETY:-1}"
Q3_ROOT_SUITES="${Q3_ROOT_SUITES:-affordance}"
Q3_ROOT_HISTORY_STEPS="${Q3_ROOT_HISTORY_STEPS:-512}"
ROOTS_PER_TASK="${ROOTS_PER_TASK:-20}"
MAX_BRANCHES="${MAX_BRANCHES:-8}"
RISK_WINDOW="${RISK_WINDOW:-32}"
EPISODE_STEPS="${EPISODE_STEPS:-512}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
STALL_ROUNDS="${STALL_ROUNDS:-3}"
MAX_FAILED_ROUNDS="${MAX_FAILED_ROUNDS:-2}"
Q1_MIN="${Q1_MIN:-25}"
Q3_MIN="${Q3_MIN:-25}"
UNSAFE_MIN="${UNSAFE_MIN:-50}"
Q2_SOFT="${Q2_SOFT:-10}"
AFF_UNSAFE_MIN="${AFF_UNSAFE_MIN:-25}"
AFF_Q2_SOFT="${AFF_Q2_SOFT:-0}"
MIN_ROOTS="${MIN_ROOTS:-15}"
MIN_UNSAFE_ROOTS="${MIN_UNSAFE_ROOTS:-10}"
read -r -a SUITES <<< "${SUITES:-human_safety obstacle_avoidance obstacle_avoidance_human affordance}"
read -r -a LEVELS <<< "${LEVELS:-0 1}"
TASK_FILTER="${TASK_FILTER:-}"   # e.g. "2" to restrict every cell to task 2
# Shared NFS: stop before a round if free space under COLLECT_ROOT drops below this.
MIN_FREE_GB="${MIN_FREE_GB:-80}"
# LIBERO-Safety L0/L1 tasks ship 50 init states each.
INIT_STATES_PER_TASK="${INIT_STATES_PER_TASK:-50}"
# Kill a round that runs longer than this (guards against a silent hang).
ROUND_TIMEOUT_S="${ROUND_TIMEOUT_S:-14400}"
# Kill a round whose launcher.log has not been written for this long. A crashed
# Ray actor (e.g. pybind11 dealloc abort in a ChannelWorker) leaves the driver
# blocked forever with the GPUs idle; tqdm normally touches the log every step.
LOG_STALL_S="${LOG_STALL_S:-900}"

if [[ "${SMOKE}" == "1" ]]; then
  read -r -a SUITES <<< "${SUITES_SMOKE:-affordance}"
  read -r -a LEVELS <<< "${LEVELS_SMOKE:-0}"
  TASK_FILTER="${TASK_FILTER:-2}"
  ROOTS_PER_TASK="${ROOTS_PER_TASK_SMOKE:-4}"
  MAX_BRANCHES="${MAX_BRANCHES_SMOKE:-2}"
  EPISODE_STEPS="${EPISODE_STEPS_SMOKE:-160}"
  Q3_ROOT_HISTORY_STEPS="${EPISODE_STEPS}"
  MAX_ROUNDS=1
  DEFAULT_COLLECT_ROOT="${RUNTIME_ROOT}/rollouts/cf_l0l1_smoke/${RUN_TOKEN}"
else
  DEFAULT_COLLECT_ROOT="${RUNTIME_ROOT}/rollouts/cf_l0l1"
fi
export COLLECT_ROOT="${COLLECT_ROOT:-${DEFAULT_COLLECT_ROOT}}"
mkdir -p "${COLLECT_ROOT}/runs" "${COLLECT_ROOT}/state"

if [[ "${AUX_SAFETY}" == "1" ]]; then AUX_BOOL=true; else AUX_BOOL=false; fi
MAX_ROLLOUT_STEPS=$(((MAX_BRANCHES + 1) * EPISODE_STEPS))

# Number of GPUs in GPU_RANKS ("0-3" or "0,2,3"); total_num_envs must divide by it.
gpu_count() {
  local spec="$1" n=0 part
  IFS=, read -r -a parts <<< "${spec}"
  for part in "${parts[@]}"; do
    if [[ "${part}" == *-* ]]; then
      n=$((n + ${part#*-} - ${part%-*} + 1))
    else
      n=$((n + 1))
    fi
  done
  echo "${n}"
}
NUM_GPUS="$(gpu_count "${GPU_RANKS}")"
# ENV_PROCS_PER_GPU>1 places several EnvWorkers per GPU (rollout stays one per
# GPU); RLinf routing merges their batches on the rollout side.
ENV_PROCS_PER_GPU="${ENV_PROCS_PER_GPU:-1}"
ENV_WORLD=$((NUM_GPUS * ENV_PROCS_PER_GPU))
placement_overrides=()
if (( ENV_PROCS_PER_GPU > 1 )); then
  placement_overrides+=("++cluster.component_placement.env='${GPU_RANKS}:0-$((ENV_WORLD - 1))'")
fi

# Never share a GPU with someone else's job (gpu003/gpu006 are multi-tenant).
# Called after `ray stop`, so any compute process left on a target GPU is foreign.
# Prints the foreign processes (empty if the GPUs are free).
gpu_foreign_procs() {
  local spec="$1" part ids=()
  IFS=, read -r -a parts <<< "${spec}"
  for part in "${parts[@]}"; do
    if [[ "${part}" == *-* ]]; then ids+=($(seq "${part%-*}" "${part#*-}")); else ids+=("${part}"); fi
  done
  nvidia-smi --id="$(IFS=,; echo "${ids[*]}")" \
    --query-compute-apps=gpu_bus_id,pid,process_name --format=csv,noheader 2>/dev/null || true
}
# Block until the target GPUs are free instead of exiting, so another user's
# job only pauses the collection. GPU_WAIT_MAX_S>0 bounds the wait (exit 5).
GPU_WAIT_POLL_S="${GPU_WAIT_POLL_S:-120}"
GPU_WAIT_MAX_S="${GPU_WAIT_MAX_S:-0}"
wait_gpus_free() {
  local spec="$1" busy waited=0 last_log=-1
  while busy="$(gpu_foreign_procs "${spec}")"; [[ -n "${busy}" ]]; do
    if (( GPU_WAIT_MAX_S > 0 && waited >= GPU_WAIT_MAX_S )); then
      echo "[abort] GPUs ${spec} still busy after ${waited}s:" >&2
      echo "${busy}" >&2
      return 1
    fi
    if (( last_log < 0 || waited - last_log >= 1800 )); then
      echo "[wait] GPUs ${spec} busy with foreign processes, polling every ${GPU_WAIT_POLL_S}s (waited ${waited}s) $(date '+%F %T'):" >&2
      echo "${busy}" >&2
      last_log=${waited}
    fi
    sleep "${GPU_WAIT_POLL_S}"
    waited=$((waited + GPU_WAIT_POLL_S))
  done
  (( waited > 0 )) && echo "[wait] GPUs ${spec} free again after ${waited}s $(date '+%F %T')"
  return 0
}

# Canonical index keeps seeds stable no matter which SUITES subset is run.
suite_index() {
  case "$1" in
    affordance) echo 0 ;; human_safety) echo 1 ;;
    obstacle_avoidance) echo 2 ;; obstacle_avoidance_human) echo 3 ;;
    *) echo "unknown suite $1" >&2; return 2 ;;
  esac
}

task_ids_for_level() {
  python - "$1" "$2" <<'PY' | sed -n 's/^TASK_ID=//p'
import sys
from libero.libero.benchmark import get_benchmark

suite = get_benchmark(sys.argv[1])()
level = int(sys.argv[2])
for task_id in range(suite.get_num_tasks()):
    if int(suite.get_task(task_id).level) == level:
        print(f"TASK_ID={task_id}")
PY
}

in_list() {
  local needle="$1"; shift
  local item
  for item in "$@"; do [[ "${item}" == "${needle}" ]] && return 0; done
  return 1
}

# Prints the coverage table to stderr and "PENDING=.. / ROUND=.." to stdout.
coverage_status() {
  local suite="$1" level="$2" tasks="$3" unsafe_min="$4" q2_soft="$5"
  python "${COVERAGE_PY}" status \
    --collect-root "${COLLECT_ROOT}" --suite "${suite}" --level "${level}" \
    --tasks "${tasks}" --q1-min "${Q1_MIN}" --q3-min "${Q3_MIN}" \
    --unsafe-min "${unsafe_min}" --q2-soft "${q2_soft}" \
    --min-roots "${MIN_ROOTS}" --min-unsafe-roots "${MIN_UNSAFE_ROOTS}" \
    --stall-rounds "${STALL_ROUNDS}" --max-rounds "${MAX_ROUNDS}"
}

read -r -a Q3_ROOT_LIST <<< "${Q3_ROOT_SUITES}"

echo "############################################################"
echo "COLLECT_ROOT=${COLLECT_ROOT}"
echo "suites=${SUITES[*]} levels=${LEVELS[*]} smoke=${SMOKE}"
echo "GPUs: physical ranks=${GPU_RANKS} (${NUM_GPUS}) env_procs/gpu=${ENV_PROCS_PER_GPU} egl_lock=${EGL_LOCK} omp_threads=${OMP_NUM_THREADS:-default}"
echo "roots/task/round=${ROOTS_PER_TASK} max_branches=${MAX_BRANCHES} episode_steps=${EPISODE_STEPS}"
echo "quota: Q1>=${Q1_MIN} Q3>=${Q3_MIN} unsafe>=${UNSAFE_MIN} (affordance ${AFF_UNSAFE_MIN})" \
     "Q2soft=${Q2_SOFT} (affordance ${AFF_Q2_SOFT}) roots>=${MIN_ROOTS}/${MIN_UNSAFE_ROOTS}"
echo "stall_rounds=${STALL_ROUNDS} max_rounds=${MAX_ROUNDS} round_timeout=${ROUND_TIMEOUT_S}s log_stall=${LOG_STALL_S}s min_free=${MIN_FREE_GB}GB"
echo "branch_strategy=${BRANCH_STRATEGY} aux_safety=${AUX_SAFETY} q3_root_suites=${Q3_ROOT_SUITES:-none}"
echo "############################################################"

failed_rounds=0
summary=()

for suite in "${SUITES[@]}"; do
  sidx="$(suite_index "${suite}")"
  if [[ "${suite}" == affordance ]]; then
    unsafe_min="${AFF_UNSAFE_MIN}"; q2_soft="${AFF_Q2_SOFT}"
  else
    unsafe_min="${UNSAFE_MIN}"; q2_soft="${Q2_SOFT}"
  fi
  suite_overrides=()
  if in_list "${suite}" "${Q3_ROOT_LIST[@]}"; then
    # Safe timeouts also branch toward Q4; let branch points cover the whole
    # timeout trajectory instead of the default 256-step tail.
    suite_overrides+=(
      "++env.eval.episode_auditor.counterfactual_branch.root_quadrants=[Q1,Q3]"
      "++env.eval.episode_auditor.counterfactual_branch.strategy_overrides.history_steps=${Q3_ROOT_HISTORY_STEPS}"
    )
  fi

  for level_number in "${LEVELS[@]}"; do
    level="L${level_number}"
    cell="${suite}_${level}"
    mapfile -t task_ids < <(task_ids_for_level "${suite}" "${level_number}")
    if [[ -n "${TASK_FILTER}" ]]; then
      IFS=, read -r -a wanted <<< "${TASK_FILTER}"
      filtered=()
      for t in "${task_ids[@]}"; do in_list "${t}" "${wanted[@]}" && filtered+=("${t}"); done
      task_ids=("${filtered[@]}")
    fi
    if (( ${#task_ids[@]} == 0 )); then
      echo "No tasks for ${cell} (TASK_FILTER=${TASK_FILTER:-none})" >&2
      exit 2
    fi
    task_csv="$(IFS=,; echo "${task_ids[*]}")"

    while true; do
      echo "------------------------------------------------------------"
      echo "[cell] ${cell} tasks=${task_csv} $(date '+%F %T')"
      status_out="$(coverage_status "${suite}" "${level}" "${task_csv}" "${unsafe_min}" "${q2_soft}")"
      pending="$(sed -n 's/^PENDING=//p' <<< "${status_out}")"
      round="$(sed -n 's/^ROUND=//p' <<< "${status_out}")"
      if [[ -z "${pending}" ]]; then
        echo "[cell] ${cell}: no pending tasks (done / infeasible / max rounds)"
        break
      fi
      IFS=, read -r -a batch <<< "${pending}"
      total_envs=$((ROOTS_PER_TASK * ${#batch[@]}))
      # Round up so every EnvWorker gets the same number of envs.
      total_envs=$(((total_envs + ENV_WORLD - 1) / ENV_WORLD * ENV_WORLD))
      # Each task has INIT_STATES_PER_TASK init states, dealt round-robin to
      # EnvWorkers; a worker with more envs than states crashes (IndexError in
      # get_env_fn_params). Cap at the largest multiple of ENV_WORLD that fits.
      max_envs=$((INIT_STATES_PER_TASK * ${#batch[@]} / ENV_WORLD * ENV_WORLD))
      if (( max_envs < ENV_WORLD )); then max_envs=${ENV_WORLD}; fi
      if (( total_envs > max_envs )); then total_envs=${max_envs}; fi
      seed=$((sidx * 100000 + level_number * 10000 + round * 100))
      round_root="${COLLECT_ROOT}/runs/${cell}_tasks${pending//,/-}_round${round}_$(date +%Y%m%d_%H%M%S)"
      mkdir -p "${round_root}"

      echo "[round] ${cell} round=${round} tasks=[${pending}] total_envs=${total_envs}" \
           "($((total_envs / NUM_GPUS))/GPU) seed=${seed}"
      echo "[round] output=${round_root}"
      started_at=$(date +%s)

      free_gb=$(df -BG --output=avail "${COLLECT_ROOT}" | tail -1 | tr -dc '0-9')
      if (( free_gb < MIN_FREE_GB )); then
        echo "[abort] only ${free_gb} GB free under ${COLLECT_ROOT} (< MIN_FREE_GB=${MIN_FREE_GB})" >&2
        exit 6
      fi
      echo "[round] disk free ${free_gb} GB"
      ray stop --force >/dev/null 2>&1 || true
      sleep 3
      wait_gpus_free "${GPU_RANKS}" || exit 5
      started_at=$(date +%s)
      set +e
      MODE=cell \
      SUITE="${suite}" \
      LEVEL="${level}" \
      GPU_RANKS="'${GPU_RANKS}'" \
      TOTAL_ENVS="${total_envs}" \
      EVAL_SEED="${seed}" \
      OUTPUT_ROOT="${round_root}" \
        timeout --kill-after=60 "${ROUND_TIMEOUT_S}" bash evaluations/run_safelibero_pi05.sh \
          "env.eval.task_id_filter=[${pending}]" \
          "rollout.model.num_action_chunks=1" \
          "env.eval.max_episode_steps=${EPISODE_STEPS}" \
          "env.eval.max_steps_per_rollout_epoch=${MAX_ROLLOUT_STEPS}" \
          "env.eval.video_cfg.save_video=false" \
          "env.eval.episode_auditor.enabled=true" \
          "env.eval.episode_auditor.mode=shadow_collect" \
          "env.eval.episode_auditor.include_observations=true" \
          "env.eval.episode_auditor.save_visual_observations=true" \
          "env.eval.episode_auditor.visual_camera_keys=[agentview_image]" \
          "env.eval.episode_auditor.counterfactual_branch.enabled=true" \
          "env.eval.episode_auditor.counterfactual_branch.risk_window_steps=${RISK_WINDOW}" \
          "env.eval.episode_auditor.counterfactual_branch.max_branches_per_root=${MAX_BRANCHES}" \
          "env.eval.episode_auditor.counterfactual_branch.target_quadrants=[Q2,Q4]" \
          "env.eval.episode_auditor.counterfactual_branch.perturbation_range=[0.15,1.0]" \
          "env.eval.episode_auditor.counterfactual_branch.action_dimensions=[0,1,2,3,4,5]" \
          "++env.eval.episode_auditor.counterfactual_branch.strategy=${BRANCH_STRATEGY}" \
          "++env.eval.episode_auditor.aux_safety.enabled=${AUX_BOOL}" \
          "++env.eval.episode_auditor.aux_safety.count_as_unsafe=${AUX_BOOL}" \
          "${suite_overrides[@]}" \
          "${placement_overrides[@]}" \
        > "${round_root}/launcher.log" 2>&1 &
      launcher_pid=$!
      # timeout(1) leads its own process group, so signalling it (or the group)
      # tears down the launcher, the eval driver and its children.
      while kill -0 "${launcher_pid}" 2>/dev/null; do
        sleep 30
        idle=$(( $(date +%s) - $(stat -c %Y "${round_root}/launcher.log") ))
        if (( idle > LOG_STALL_S )); then
          echo "[round] ${cell} round=${round}: launcher.log idle ${idle}s (> ${LOG_STALL_S}s), killing hung round" >&2
          kill -TERM "${launcher_pid}" 2>/dev/null
          sleep 30
          kill -KILL -- "-${launcher_pid}" 2>/dev/null
          ray stop --force >/dev/null 2>&1 || true
          break
        fi
      done
      wait "${launcher_pid}"
      status=$?
      set -e
      elapsed=$(( $(date +%s) - started_at ))

      # RLinf reports 255 during Ray teardown even after a completed run;
      # accept it only if this run's metrics.log reached 100%.
      metrics_file="${round_root}/${cell}/metrics.log"
      completed=0
      if [[ -f "${metrics_file}" ]] \
        && (( $(stat -c %Y "${metrics_file}") >= started_at )) \
        && grep -q 'Progress:.*100.0%' "${metrics_file}"; then
        completed=1
      fi
      preempted=0
      if (( status != 0 && completed == 0 )); then
        # A foreign job that grabbed our GPUs mid-round (OOM / NCCL errors) is
        # not our failure: keep its partial audits, don't consume the round or
        # count towards MAX_FAILED_ROUNDS; the next iteration waits for the GPUs.
        ray stop --force >/dev/null 2>&1 || true
        sleep 5
        foreign="$(gpu_foreign_procs "${GPU_RANKS}")"
        [[ -n "${foreign}" ]] && preempted=1
      fi
      if (( status == 0 || completed == 1 )); then
        echo "[round] ${cell} round=${round} finished in ${elapsed}s (exit ${status})"
        failed_rounds=0
      elif (( preempted == 1 )); then
        echo "[round] ${cell} round=${round} interrupted exit=${status} after ${elapsed}s;" \
             "foreign processes on GPUs ${GPU_RANKS}, retrying after they leave:" >&2
        echo "${foreign}" >&2
        continue
      else
        failed_rounds=$((failed_rounds + 1))
        echo "[round] ${cell} round=${round} FAILED exit=${status} after ${elapsed}s;" \
             "see ${round_root}/launcher.log" >&2
        tail -n 40 "${round_root}/launcher.log" >&2 || true
        if (( failed_rounds >= MAX_FAILED_ROUNDS )); then
          echo "[abort] ${failed_rounds} consecutive failed rounds" >&2
          ray stop --force >/dev/null 2>&1 || true
          exit 4
        fi
      fi
      # Partial audits of a failed round still count; the round is consumed
      # either way so stall detection and MAX_ROUNDS keep bounding the loop.
      python "${COVERAGE_PY}" round-done --collect-root "${COLLECT_ROOT}" \
        --suite "${suite}" --level "${level}" --tasks "${pending}"
    done
    summary+=("${cell}")
  done
done

ray stop --force >/dev/null 2>&1 || true
echo "############################################################"
echo "Final coverage (COLLECT_ROOT=${COLLECT_ROOT})"
for cell in "${summary[@]}"; do
  suite="${cell%_L*}"; level="${cell##*_}"
  if [[ "${suite}" == affordance ]]; then u="${AFF_UNSAFE_MIN}"; q="${AFF_Q2_SOFT}"; else u="${UNSAFE_MIN}"; q="${Q2_SOFT}"; fi
  mapfile -t task_ids < <(task_ids_for_level "${suite}" "${level#L}")
  if [[ -n "${TASK_FILTER}" ]]; then
    IFS=, read -r -a wanted <<< "${TASK_FILTER}"
    filtered=()
    for t in "${task_ids[@]}"; do in_list "${t}" "${wanted[@]}" && filtered+=("${t}"); done
    task_ids=("${filtered[@]}")
  fi
  coverage_status "${suite}" "${level}" "$(IFS=,; echo "${task_ids[*]}")" "${u}" "${q}" >/dev/null
done
echo "############################################################"
