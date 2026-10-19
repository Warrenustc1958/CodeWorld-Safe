#!/bin/bash
# Start (once) the shared Ray head used by all OFT SFT runs on gpu005.
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
source "$R/tmp/sft_oft_env.sh"
RAY=/mnt/nfs/fjj/Code/.venv-openvlaoft-libero-safety/bin/ray
if "$RAY" status --address=127.0.0.1:6390 >/dev/null 2>&1; then
  echo "ray head already running"; exit 0
fi
"$RAY" start --head --port=6390 --dashboard-port=8270 --ray-client-server-port=23010 \
  --num-gpus=8 --num-cpus=32 --temp-dir=/tmp/rs5 --disable-usage-stats
