#!/bin/bash
# Train one CWS reward model run.
# Usage: bash toolkits/codeworld_safe/rm/launch_rm.sh <gpu> <views 1|2> <run_name> [train_rm args]
set -euo pipefail
REPO=/mnt/nfs/fjj/Code/CodeWorld-Safe
PY=/mnt/nfs/fjj/wan_ac_sft/venv-wan/bin/python
FRAMES=${FRAMES:-/data_cf/cws_rm/frames_v1}
INIT=${INIT:-/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/resnet_rm.pth}
GPU=$1; VIEWS=$2; RUN=$3; shift 3
OUT=/data_cf/cws_rm/runs/$RUN
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH=$REPO
CUDA_VISIBLE_DEVICES=$GPU $PY -m toolkits.codeworld_safe.rm.train_rm \
  --frames "$FRAMES" --views "$VIEWS" --out "$OUT" --init "$INIT" "$@" 2>&1 | tee -a "$OUT/stdout.txt"
