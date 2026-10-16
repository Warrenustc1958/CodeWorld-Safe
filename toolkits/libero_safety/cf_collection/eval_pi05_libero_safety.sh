#!/usr/bin/env bash
# LIBERO-Safety paper protocol for the single released pi0.5 checkpoint:
#   4 physical-safety suites x 3 levels x 5 tasks x 10 fixed trials.
# The paper's three seeds are three independently trained models; this runtime
# contains one released checkpoint, so this script evaluates that checkpoint once.
source /mnt/nfs/fjj/Code/rlinf-safety-runtime/export.sh

GPU_RANKS="${GPU_RANKS:-0-4}"
TOTAL_ENVS="${TOTAL_ENVS:-50}"
EVAL_SEED="${EVAL_SEED:-0}"
OUTPUT_ROOT="${PAPER_OUTPUT_ROOT:-${RUN_ROOT}/rollouts/pi05_libero_safety_paper_5gpu_seed${EVAL_SEED}}"

# ImageMagick is installed under the runtime tree because gpu005 does not have
# the system MagickWand library and this account cannot use sudo. LIBERO-Safety
# imports Wand for its motion-blur perturbation even before an env is created.
IMAGEMAGICK_PREFIX="${RUN_ROOT}/local-packages/imagemagick/prefix"
IMAGEMAGICK_MODULES="${IMAGEMAGICK_PREFIX}/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16"
export MAGICK_HOME="${IMAGEMAGICK_PREFIX}/usr"
export MAGICK_CONFIGURE_PATH="${IMAGEMAGICK_PREFIX}/etc/ImageMagick-6:${IMAGEMAGICK_PREFIX}/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16:${IMAGEMAGICK_PREFIX}/usr/share/ImageMagick-6"
export MAGICK_CODER_MODULE_PATH="${IMAGEMAGICK_MODULES}/coders"
export MAGICK_FILTER_MODULE_PATH="${IMAGEMAGICK_MODULES}/filters"
export LD_LIBRARY_PATH="${IMAGEMAGICK_PREFIX}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Ray's Unix sockets must stay below Linux's path-length limit. Serialize EGL
# context creation per GPU to avoid concurrent driver crashes at 50 envs.
export RAY_TMPDIR="${PI05_RAY_TMPDIR:-/tmp/rp-pi05}"
export TMPDIR="${PI05_TMPDIR:-/tmp/p5-pi05}"
export RLINF_LIBERO_EGL_LOCK_PATH="${RLINF_LIBERO_EGL_LOCK_PATH:-/tmp/rlinf-libero-egl.lock}"
export RLINF_LIBERO_EGL_LOCK_SCOPE="${RLINF_LIBERO_EGL_LOCK_SCOPE:-per_gpu}"

suites=(
  affordance
  human_safety
  obstacle_avoidance
  obstacle_avoidance_human
)
levels=(L0 L1 L2)

mkdir -p "$OUTPUT_ROOT"

for suite in "${suites[@]}"; do
  for level in "${levels[@]}"; do
    cell_dir="${OUTPUT_ROOT}/${suite}_${level}"
    metrics_file="${cell_dir}/metrics.log"

    if [[ -s "$metrics_file" ]] \
      && grep -q 'Progress:.*100.0%' "$metrics_file" \
      && grep -q 'num_trajectories=50' "$metrics_file"; then
      echo "[paper-eval] skip completed cell: ${suite}/${level}"
      continue
    fi

    echo "[paper-eval] start: ${suite}/${level} GPUs=${GPU_RANKS} envs=${TOTAL_ENVS} seed=${EVAL_SEED}"
    started_at=$(date +%s)

    # gpu005 has no Xvfb installation. Use MuJoCo's verified headless EGL
    # backend instead of requiring an X server through xvfb-run.
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MODE=cell \
    SUITE="$suite" \
    LEVEL="$level" \
    GPU_RANKS="'${GPU_RANKS}'" \
    TOTAL_ENVS="$TOTAL_ENVS" \
    EVAL_SEED="$EVAL_SEED" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash evaluations/run_safelibero_pi05.sh \
        env.eval.auto_reset=false \
        env.eval.ignore_terminations=false \
        env.eval.use_fixed_reset_state_ids=true \
        env.eval.use_ordered_reset_state_ids=true \
        env.eval.video_cfg.save_video=false \
        rollout.model.num_action_chunks=5 \
        +rollout.model.openpi.action_horizon=10
    status=$?

    # RLinf currently reports 255 during Ray actor teardown even after a fully
    # completed evaluation. Accept that cleanup-only failure only when this run
    # wrote a complete 50-trajectory metric table; never mask an earlier error.
    metrics_mtime=0
    if [[ -f "$metrics_file" ]]; then
      metrics_mtime=$(stat -c %Y "$metrics_file")
    fi
    if (( status != 0 )); then
      if (( metrics_mtime >= started_at )) \
        && grep -q 'Progress:.*100.0%' "$metrics_file" \
        && grep -q 'num_trajectories=50' "$metrics_file"; then
        echo "[paper-eval] completed ${suite}/${level}; ignoring RLinf teardown status ${status}"
      else
        echo "[paper-eval] FAILED ${suite}/${level} with status ${status}" >&2
        exit "$status"
      fi
    fi
  done
done

echo "[paper-eval] all 12 suite/level cells completed: ${OUTPUT_ROOT}"
#cd /mnt/nfs/fjj/Code/rlinf-safety-runtime && stdbuf -oL -eL bash eval_pi05_libero_safety.sh 2>&1 | tee '/mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/eval_pi05_20261013_145806.log'