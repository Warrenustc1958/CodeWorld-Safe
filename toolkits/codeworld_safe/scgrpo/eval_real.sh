#!/usr/bin/env bash
# Real LIBERO-Safety evaluation of a pi0.5 checkpoint (SC-GRPO) on the fresh_init/v1 init states
# (40 cells x 20, unseen by the policy and Wan), grader grades on, no visuals saved.
#   CKPT=<full_weights.pt or empty for the base policy> NAME=<tag> [CELLS="suite:L0:3 ..."] bash eval_real.sh
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
REPO=/mnt/nfs/fjj/Code/CodeWorld-Safe
OUT=/data_cf/cws_sam/v1/scgrpo/eval_real/${NAME:?}
GPUS=${GPUS:-6-7}; ENVS=${ENVS:-20}
INIT_ROOT=${INIT_ROOT:-/data_cf/cws/fresh_init/v1}
CKPT=${CKPT:-}
mkdir -p "$OUT"
IM=$R/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$IM/usr/lib/x86_64-linux-gnu
export MAGICK_CONFIGURE_PATH=$IM/etc/ImageMagick-6:$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16
export MAGICK_CODER_MODULE_PATH=$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false GPU_WAIT_DISABLE=1
export RAY_TMPDIR=/tmp/rp_ev TMPDIR=/tmp/p5_ev
mkdir -p $RAY_TMPDIR $TMPDIR
# Dedicated Ray head: RLinf calls ray.init(address="auto"), which would otherwise join the running
# SC-GRPO training cluster. Never `ray stop` here (it kills every Ray on the host).
RAY_PORT=${RAY_PORT:-6431}
export RAY_AUTH_MODE=disabled
export RAY_ADDRESS=127.0.0.1:$RAY_PORT
if ! $R/.venv-openpi-libero-safety/bin/ray status --address $RAY_ADDRESS >/dev/null 2>&1; then
  $R/.venv-openpi-libero-safety/bin/ray start --head --port=$RAY_PORT --temp-dir=$RAY_TMPDIR     --include-dashboard=false --num-gpus=8 --disable-usage-stats > $RAY_TMPDIR/ray_start.log 2>&1
fi
$R/.venv-openpi-libero-safety/bin/ray status --address $RAY_ADDRESS >/dev/null 2>&1 || { echo "eval Ray head not reachable at $RAY_ADDRESS"; cat $RAY_TMPDIR/ray_start.log; exit 3; }
if [ -z "${CELLS:-}" ]; then
  CELLS=""
  for s in affordance human_safety obstacle_avoidance obstacle_avoidance_human; do
    for t in 0 1 2 3 4; do CELLS+=" $s:L0:$t"; done
    for t in 5 6 7 8 9; do CELLS+=" $s:L1:$t"; done
  done
fi
CK_ARGS=()
[ -n "$CKPT" ] && CK_ARGS=("runner.ckpt_path=$CKPT")
cd "$REPO"
for c in $CELLS; do
  IFS=: read -r suite level task <<< "$c"
  d=$OUT/${suite}_${level}_t${task}
  if grep -q "EXIT=0" "$d/launcher.log" 2>/dev/null; then echo "[skip] $c"; continue; fi
  mkdir -p "$d"
  t0=$(date +%s)
  VENV_PATH=$R/.venv-openpi-libero-safety LIBERO_SAFETY_PATH=/mnt/nfs/fjj/Code/LIBERO-Safety \
  MODEL=$R/checkpoints/pi05_libero_safety_rlinf OUTPUT_ROOT="$d" MODE=cell SUITE=$suite LEVEL=$level \
  GPU_RANKS=$GPUS TOTAL_ENVS=$ENVS EVAL_SEED=0 \
  bash evaluations/run_safelibero_pi05.sh \
    "env.eval.task_id_filter=[$task]" "rollout.model.num_action_chunks=8" \
    "env.eval.max_episode_steps=512" "env.eval.max_steps_per_rollout_epoch=512" \
    "env.eval.video_cfg.save_video=false" "++env.eval.init_states_root=$INIT_ROOT" \
    "env.eval.episode_auditor.enabled=true" "env.eval.episode_auditor.mode=shadow_collect" \
    "env.eval.episode_auditor.include_observations=false" "env.eval.episode_auditor.save_visual_observations=false" \
    "++env.eval.episode_auditor.save_grades=true" \
    "++env.eval.episode_auditor.aux_safety.enabled=true" "++env.eval.episode_auditor.aux_safety.count_as_unsafe=true" \
    "env.eval.episode_auditor.counterfactual_branch.enabled=false" \
    "${CK_ARGS[@]}" > "$d/launcher.log" 2>&1
  echo "EXIT=$? $(( $(date +%s) - t0 ))s" | tee -a "$d/launcher.log"
  echo "[$(date +%H:%M:%S)] $c $(tail -n 1 $d/launcher.log)"
done
echo ALL_DONE
