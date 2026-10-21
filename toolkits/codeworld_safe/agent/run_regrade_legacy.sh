#!/usr/bin/env bash
# CPU-only replay + regrade of legacy pi0.5 CF audits. Args go to regrade_legacy.py.
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
VENV=$R/.venv-openpi-libero-safety
export LIBERO_TYPE=safety LIBERO_CONFIG_PATH=$VENV/share/libero-safety
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=
export RLINF_LIBERO_SAFETY_AUX=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=$R/local-packages/imagemagick/prefix; IM=$P/usr/lib/x86_64-linux-gnu
export MAGICK_HOME=$P/usr LD_LIBRARY_PATH=$IM:${LD_LIBRARY_PATH:-}
cd /mnt/nfs/fjj/Code/CodeWorld-Safe
export PYTHONPATH=/mnt/nfs/fjj/Code/LIBERO-Safety:.
exec $VENV/bin/python -m toolkits.codeworld_safe.agent.regrade_legacy "$@" < /dev/null
