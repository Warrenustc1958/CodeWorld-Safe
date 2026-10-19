#!/bin/bash
# Full-parameter OpenVLA-OFT SFT on LIBERO-Safety demos (gpu005).
# Usage: sft_oft_launch.sh <run_name> <dataset_dir> <gpu_ranks> <max_steps> <save_interval> [extra hydra overrides...]
# All runs connect to one shared Ray head (start_ray_head.sh), so one job exiting
# never tears down the other job's Ray instance.
set -uo pipefail
RUN=$1; DATASET=$2; GPUS=$3; STEPS=$4; SAVE=$5; shift 5

R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
OUT=$R/rollouts/sft_oft/$RUN
mkdir -p "$OUT"
source "$R/tmp/sft_oft_env.sh"
export RAY_ADDRESS=127.0.0.1:6390

cd /mnt/nfs/fjj/Code/RLinf-Safety
VENV_PATH=/mnt/nfs/fjj/Code/.venv-openvlaoft-libero-safety \
DATASET="$DATASET" \
MODEL=/mnt/nfs/fjj/Models/RLinf-OpenVLAOFT-LIBERO-130 \
bash examples/sft/run_safelibero_openvlaoft.sh \
  actor.model.lora_path=null actor.model.is_lora=false \
  "cluster.component_placement.actor='$GPUS'" \
  runner.max_steps="$STEPS" runner.save_interval="$SAVE" \
  actor.optim.total_training_steps="${TOTAL:-$STEPS}" \
  runner.resume_dir="${RESUME:-null}" \
  runner.logger.log_path="$OUT" runner.logger.experiment_name="$RUN" \
  actor.global_batch_size=64 actor.micro_batch_size="${MICRO:-8}" \
  actor.optim.lr=2.0e-5 +actor.optim.lr_scheduler=cosine \
  actor.optim.lr_warmup_steps=500 +actor.optim.min_lr_rate=0.1 \
  data.num_workers=4 \
  "$@"
echo "EXIT=$?"
