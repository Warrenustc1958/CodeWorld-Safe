#!/usr/bin/env bash
# Causal SC-GRPO run (R1 real-goal reward / R0 RM reward) with its OWN Ray head; never joins other heads.
#   TAG=r1 PORT=6491 bash run_causal.sh [extra hydra overrides]
# Never runs `ray stop`. Stop with TAG=<tag> bash stop_causal_head.sh (process tree of /tmp/rp_c<tag>/).
set -uo pipefail
TAG=${TAG:?}; PORT=${PORT:?}
V=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety
source $V/bin/activate
cd /mnt/nfs/fjj/Code/CodeWorld-Safe
export PYTHONPATH=/mnt/nfs/fjj/Code/related_works/diffsynth-studio:/mnt/nfs/fjj/Code/CodeWorld-Safe:${PYTHONPATH:-}
export RAY_TMPDIR=/tmp/rp_c$TAG TMPDIR=/tmp/p5_c$TAG
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RAY_AUTH_MODE=disabled RAY_ADDRESS=127.0.0.1:$PORT
# real LIBERO inside the env workers (physics only, no rendering); exported before `ray start`
# osmesa (CPU GL, libosmesa6): no EGL/GPU GL contexts in any worker
export LIBERO_TYPE=safety RLINF_LIBERO_SAFETY_AUX=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYOPENGL_OSMESA_LIBRARY=/lib/x86_64-linux-gnu/libOSMesa.so.8
export LIBERO_CONFIG_PATH=$V/share/libero-safety
IM=/mnt/nfs/fjj/Code/rlinf-safety-runtime/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$IM/usr/lib/x86_64-linux-gnu
export MAGICK_CONFIGURE_PATH=$IM/etc/ImageMagick-6:$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16
export MAGICK_CODER_MODULE_PATH=$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
mkdir -p $RAY_TMPDIR $TMPDIR /data_cf/cws_sam/v1/scgrpo/causal/$TAG
if ss -ltn | grep -q ":$PORT "; then
  echo "port $PORT already in use, refusing to join an unknown Ray head"; exit 3
else
  ray start --head --port=$PORT --temp-dir=$RAY_TMPDIR --include-dashboard=false --num-gpus=8 \
    --object-store-memory=$((64 * 1024 ** 3)) --ray-client-server-port=$((PORT + 3000)) \
    ${RAY_EXTRA:-} --disable-usage-stats > $RAY_TMPDIR/ray_start.log 2>&1
fi
ray status --address $RAY_ADDRESS >/dev/null 2>&1 || { echo "Ray head $RAY_ADDRESS not reachable"; cat $RAY_TMPDIR/ray_start.log; exit 3; }
export EMBODIED_PATH=/mnt/nfs/fjj/Code/CodeWorld-Safe/examples/embodiment
python examples/embodiment/train_embodied_agent.py --config-path $EMBODIED_PATH/config \
  --config-name libero_safety_scgrpo_causal_$TAG "$@"
rc=$?
echo "EXIT=$rc"
exit $rc
