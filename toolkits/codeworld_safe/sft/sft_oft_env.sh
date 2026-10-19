# Shared environment for OFT SFT on gpu005. Must be sourced BEFORE `ray start`:
# a running Ray head only passes on variables exported before it started.
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/nfs/fjj/Code/rlinf-safety-runtime/local-packages/ffmpeg7/lib
export RAY_TMPDIR=/tmp/rs5 TMPDIR=/tmp/ts5
mkdir -p /tmp/rs5 /tmp/ts5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_AUTH_MODE=token
# cgroup pids.max for this user is ~21.6k threads; cap per-process thread pools
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 RAYON_NUM_THREADS=8 POLARS_MAX_THREADS=8 TOKENIZERS_PARALLELISM=false
