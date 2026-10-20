#!/usr/bin/env bash
# Trial one cws_cf_program with the pi0.5 policy (gpu006), shadow_collect.
#   SUITE=human_safety LEVEL=L0 TASK=0 PROGRAM=<json> GPU=7 ENVS=4 bash trial_pi05.sh
# pi0.5 replans every step (num_action_chunks=1); branch points stay on 8-step
# boundaries (action_chunk=8) so each one has a sidecar snapshot for the seam check.
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
REPO=/mnt/nfs/fjj/Code/CodeWorld-Safe
SUITE=${SUITE:?}; LEVEL=${LEVEL:?}; TASK=${TASK:?}; PROGRAM=${PROGRAM:?}
GPU=${GPU:-7}; ENVS=${ENVS:-4}; MAX_BRANCHES=${MAX_BRANCHES:-4}; SEED=${SEED:-0}
EPISODE_STEPS=512
MAX_ROLLOUT_STEPS=$(((MAX_BRANCHES + 1) * EPISODE_STEPS))
OUT=${OUT:-$R/rollouts/cws_trial_pi05/${SUITE}_${LEVEL}_t${TASK}_$(basename "$PROGRAM" .json)_$(date +%m%d_%H%M)}
mkdir -p "$OUT"
IM=$R/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$IM/usr/lib/x86_64-linux-gnu
export MAGICK_CONFIGURE_PATH=$IM/etc/ImageMagick-6:$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16
export MAGICK_CODER_MODULE_PATH=$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=/tmp/rp_cws TMPDIR=/tmp/p5_cws
cd "$REPO"
VENV_PATH=$R/.venv-openpi-libero-safety \
LIBERO_SAFETY_PATH=/mnt/nfs/fjj/Code/LIBERO-Safety \
MODEL=$R/checkpoints/pi05_libero_safety_rlinf \
OUTPUT_ROOT="$OUT" MODE=cell SUITE=$SUITE LEVEL=$LEVEL GPU_RANKS=$GPU TOTAL_ENVS=$ENVS EVAL_SEED=$SEED \
bash evaluations/run_safelibero_pi05.sh \
  "env.eval.task_id_filter=[$TASK]" \
  "rollout.model.num_action_chunks=1" \
  "env.eval.max_episode_steps=$EPISODE_STEPS" \
  "env.eval.max_steps_per_rollout_epoch=$MAX_ROLLOUT_STEPS" \
  "env.eval.video_cfg.save_video=false" \
  "env.eval.episode_auditor.enabled=true" \
  "env.eval.episode_auditor.mode=shadow_collect" \
  "env.eval.episode_auditor.include_observations=true" \
  "env.eval.episode_auditor.save_visual_observations=true" \
  "env.eval.episode_auditor.visual_camera_keys=[agentview_image]" \
  "++env.eval.episode_auditor.save_sim_state=true" \
  "++env.eval.episode_auditor.sim_state_chunk=8" \
  "++env.eval.episode_auditor.save_grades=true" \
  "++env.eval.episode_auditor.aux_safety.enabled=true" \
  "++env.eval.episode_auditor.aux_safety.count_as_unsafe=true" \
  "env.eval.episode_auditor.counterfactual_branch.enabled=true" \
  "env.eval.episode_auditor.counterfactual_branch.max_branches_per_root=$MAX_BRANCHES" \
  "env.eval.episode_auditor.counterfactual_branch.target_quadrants=[Q2,Q4]" \
  "++env.eval.episode_auditor.counterfactual_branch.root_quadrants=[Q1,Q3]" \
  "++env.eval.episode_auditor.counterfactual_branch.program=$PROGRAM" \
  "++env.eval.episode_auditor.counterfactual_branch.program_knowledge_dir=$REPO/toolkits/codeworld_safe/agent/knowledge" \
  "++env.eval.episode_auditor.counterfactual_branch.action_chunk=8" \
  > "$OUT/launcher.log" 2>&1
echo "EXIT=$? OUT=$OUT" | tee -a "$OUT/launcher.log"
