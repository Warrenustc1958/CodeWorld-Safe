#!/bin/bash
# Separate small Ray head for evaluation, so eval never shares (or tears down)
# the training head on port 6390 and never prestarts hundreds of idle workers.
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
source "$R/tmp/sft_oft_env.sh"
export RAY_TMPDIR=/tmp/re5 TMPDIR=/tmp/te5; mkdir -p /tmp/re5 /tmp/te5
RAY=/mnt/nfs/fjj/Code/.venv-openvlaoft-libero-safety/bin/ray
if "$RAY" status --address=127.0.0.1:6391 >/dev/null 2>&1; then
  echo "eval ray head already running"; exit 0
fi
"$RAY" start --head --port=6391 --dashboard-port=8271 --ray-client-server-port=23011 \
  --min-worker-port=24000 --max-worker-port=24999 \
  --num-gpus=8 --num-cpus=16 --temp-dir=/tmp/re5 --disable-usage-stats
