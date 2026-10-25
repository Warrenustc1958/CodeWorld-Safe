#!/usr/bin/env bash
# Run gen_fresh_init_states.py with the Step E collection environment (LIBERO config, ImageMagick, EGL).
#   GPU=3 bash run_fresh_init.sh --out /data_cf/cws/fresh_init/v1 --n 20
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
V=$R/.venv-openpi-libero-safety
GPU=${GPU:-3}
IM=$R/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$IM/usr/lib/x86_64-linux-gnu
export MAGICK_CONFIGURE_PATH=$IM/etc/ImageMagick-6:$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16
export MAGICK_CODER_MODULE_PATH=$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export LIBERO_CONFIG_PATH=$V/share/libero-safety
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /mnt/nfs/fjj/Code/CodeWorld-Safe
exec $V/bin/python toolkits/codeworld_safe/cf/gen_fresh_init_states.py "$@"
