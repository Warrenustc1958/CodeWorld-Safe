#!/usr/bin/env bash
# SC-GRPO launcher on gpu006 GPUs 2-7.  STEPS / ROLLOUT_EPOCH / TAG / extra hydra overrides via args.
#   STEPS=2 ROLLOUT_EPOCH=1 TAG=smoke bash run_scgrpo.sh
set -uo pipefail
cd /mnt/nfs/fjj/Code/CodeWorld-Safe
V=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety
source $V/bin/activate
# placement pins GPUs 2-7 (RLinf enumerates all GPUs and ignores CUDA_VISIBLE_DEVICES)
export PYTHONPATH=/mnt/nfs/fjj/Code/related_works/diffsynth-studio:/mnt/nfs/fjj/Code/CodeWorld-Safe:${PYTHONPATH:-}
export RAY_TMPDIR=/tmp/rp_scg TMPDIR=/tmp/p5_scg
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p $RAY_TMPDIR $TMPDIR
TAG=${TAG:-main}
OVR=("runner.logger.experiment_name=scgrpo_pi05_wan41k_${TAG}")
[ -n "${STEPS:-}" ] && OVR+=("runner.max_steps=${STEPS}")
[ -n "${ROLLOUT_EPOCH:-}" ] && OVR+=("env.train.rollout_epoch=${ROLLOUT_EPOCH}")
[ -n "${SAVE_INTER:-}" ] && OVR+=("runner.save_interval=${SAVE_INTER}")
export EMBODIED_PATH=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment
python examples/embodiment/train_embodied_agent.py --config-path $EMBODIED_PATH/config \
  --config-name libero_safety_scgrpo_openpi_pi05 "${OVR[@]}" "$@"
echo "EXIT=$?"
