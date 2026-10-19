#!/bin/bash
# Evaluate an OpenVLA-OFT policy on LIBERO-Safety (official protocol: any
# constraint violation ends the episode as a failure).
#
# Env vars (defaults in brackets):
#   SUITE [obstacle_avoidance_human]  LEVEL [L2]  GPU [0]  ENVS [50]
#   TASKS  hydra list of global task ids within the level (L0 0-4, L1 5-9, L2 10-14),
#          e.g. '[10]' for one L2 task; empty = all 5 tasks of the level
#   MODEL  HF model dir [RLinf-OpenVLAOFT-LIBERO-130]
#   CKPT   SFT full_weights.pt to load on top of MODEL (optional)
#   UNNORM [libero_130_no_noops_trajall]; with CKPT use libero_safety
#   STATS  extra stats file, e.g. <LIBERO-Safety>/meta/stats.json (needed for libero_safety)
#   OUT    log dir
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
source "$R/tmp/sft_oft_env.sh"
export RAY_TMPDIR=/tmp/re5 TMPDIR=/tmp/te5 RAY_ADDRESS=127.0.0.1:6391
IM=$R/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IM/usr/lib/x86_64-linux-gnu

SUITE=${SUITE:-obstacle_avoidance_human}; LEVEL=${LEVEL:-L2}; GPU=${GPU:-0}; ENVS=${ENVS:-50}
MODEL=${MODEL:-/mnt/nfs/fjj/Models/RLinf-OpenVLAOFT-LIBERO-130}
UNNORM=${UNNORM:-libero_130_no_noops_trajall}
OUT=${OUT:-$R/rollouts/eval_oft/${SUITE}_${LEVEL}_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

extra=("runner.logger.log_path=$OUT")
[ -n "${CKPT:-}" ] && extra+=("runner.ckpt_path=$CKPT")
[ -n "${STATS:-}" ] && extra+=("+rollout.model.dataset_statistics_path=$STATS")
[ -n "${TASKS:-}" ] && extra+=("env.eval.task_id_filter=$TASKS")

cd /mnt/nfs/fjj/Code/RLinf-Safety
VENV_PATH=/mnt/nfs/fjj/Code/.venv-openvlaoft-libero-safety \
LIBERO_SAFETY_PATH=/mnt/nfs/fjj/Code/LIBERO-Safety \
RLINF_RUNTIME_DIR=/tmp/re5rt \
MODE=eval SUITE=$SUITE LEVEL=$LEVEL GPU_RANKS=$GPU TOTAL_ENVS=$ENVS \
MODEL=$MODEL UNNORM_KEY=$UNNORM IS_LORA=False \
bash evaluations/run_safelibero.sh "${extra[@]}" "$@"
echo "EXIT=$?"
