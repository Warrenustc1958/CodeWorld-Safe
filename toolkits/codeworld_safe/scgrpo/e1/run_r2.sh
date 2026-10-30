#!/usr/bin/env bash
# R2 collector for one policy: GPU NAME CKPT_DIR OUT_DIR  (Wan on the GPU, real LIBERO physics via osmesa)
GPU=$1; NAME=$2; CKPT=$3; OUT=$4
export CUDA_VISIBLE_DEVICES=$GPU R2_NAME=$NAME R2_CKPT=$CKPT R2_OUT=$OUT
export PYTHONPATH=/mnt/nfs/fjj/Code/CodeWorld-Safe:/mnt/nfs/fjj/Code/related_works/diffsynth-studio
export LIBERO_TYPE=safety RLINF_LIBERO_SAFETY_AUX=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export PYOPENGL_OSMESA_LIBRARY=/lib/x86_64-linux-gnu/libOSMesa.so.8
export LIBERO_CONFIG_PATH=/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/share/libero-safety
IM=/mnt/nfs/fjj/Code/rlinf-safety-runtime/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$IM/usr/lib/x86_64-linux-gnu
export MAGICK_CONFIGURE_PATH=$IM/etc/ImageMagick-6:$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16
export MAGICK_CODER_MODULE_PATH=$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
mkdir -p $OUT
cd /data_cf/cws_sam/v1/scgrpo/diag
/mnt/nfs/fjj/Code/rlinf-safety-runtime/.venv-openpi-libero-safety/bin/python collect_r2.py >> $OUT/collect.out 2>&1
echo "EXIT=$?" >> $OUT/collect.out
