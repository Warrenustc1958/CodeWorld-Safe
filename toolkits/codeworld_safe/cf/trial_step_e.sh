#!/usr/bin/env bash
# Step E: one cell, pi0.5, chunk 8, 512 steps, agentview + wrist, shadow_collect.
#   SUITE=human_safety LEVEL=L0 TASK=0 [PROGRAM=<json>] OUT=<dir> GPUS=4-5 ENVS=50 SEED=0 bash trial_step_e.sh
# Without PROGRAM only roots are collected (no counterfactual branches).
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
REPO=/mnt/nfs/fjj/Code/CodeWorld-Safe
SUITE=${SUITE:?}; LEVEL=${LEVEL:?}; TASK=${TASK:?}; OUT=${OUT:?}
PROGRAM=${PROGRAM:-}
GPUS=${GPUS:-4-5}; ENVS=${ENVS:-50}; SEED=${SEED:-0}; MAX_BRANCHES=${MAX_BRANCHES:-3}
CHUNKS=${CHUNKS:-8}; EPISODE_STEPS=512
CAMS=${CAMS:-[agentview_image,robot0_eye_in_hand_image]}
# ROLLOUT_GPUS: put the policy on other GPUs than the env (rendering) workers.
# gpu006 GPU 4 faults in EGL (Xid 31/109), so render on 5 and infer on 4.
ROLLOUT_GPUS=${ROLLOUT_GPUS:-}
PLACE_ARGS=()
[ -n "$ROLLOUT_GPUS" ] && PLACE_ARGS=("++cluster.component_placement.rollout=$ROLLOUT_GPUS")
if [ -n "$PROGRAM" ]; then BR=$MAX_BRANCHES; else BR=0; fi
MAX_ROLLOUT_STEPS=$(((BR + 1) * EPISODE_STEPS))
mkdir -p "$OUT"
IM=$R/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$IM/usr/lib/x86_64-linux-gnu
export MAGICK_CONFIGURE_PATH=$IM/etc/ImageMagick-6:$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/config-Q16
export MAGICK_CODER_MODULE_PATH=$IM/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=/tmp/rp_cwse TMPDIR=/tmp/p5_cwse
BRANCH_ARGS=("env.eval.episode_auditor.counterfactual_branch.enabled=false")
if [ -n "$PROGRAM" ]; then
  BRANCH_ARGS=(
    "env.eval.episode_auditor.counterfactual_branch.enabled=true"
    "env.eval.episode_auditor.counterfactual_branch.max_branches_per_root=$MAX_BRANCHES"
    "env.eval.episode_auditor.counterfactual_branch.target_quadrants=[Q2,Q4]"
    "++env.eval.episode_auditor.counterfactual_branch.root_quadrants=[Q1,Q3]"
    "++env.eval.episode_auditor.counterfactual_branch.program=$PROGRAM"
    "++env.eval.episode_auditor.counterfactual_branch.program_knowledge_dir=$REPO/toolkits/codeworld_safe/agent/knowledge"
    "++env.eval.episode_auditor.counterfactual_branch.action_chunk=8"
  )
fi
cd "$REPO"
VENV_PATH=$R/.venv-openpi-libero-safety \
LIBERO_SAFETY_PATH=/mnt/nfs/fjj/Code/LIBERO-Safety \
MODEL=$R/checkpoints/pi05_libero_safety_rlinf \
OUTPUT_ROOT="$OUT" MODE=cell SUITE=$SUITE LEVEL=$LEVEL GPU_RANKS=$GPUS TOTAL_ENVS=$ENVS EVAL_SEED=$SEED \
bash evaluations/run_safelibero_pi05.sh \
  "env.eval.task_id_filter=[$TASK]" \
  "rollout.model.num_action_chunks=$CHUNKS" \
  "env.eval.max_episode_steps=$EPISODE_STEPS" \
  "env.eval.max_steps_per_rollout_epoch=$MAX_ROLLOUT_STEPS" \
  "env.eval.video_cfg.save_video=false" \
  "env.eval.episode_auditor.enabled=true" \
  "env.eval.episode_auditor.mode=shadow_collect" \
  "env.eval.episode_auditor.include_observations=true" \
  "env.eval.episode_auditor.save_visual_observations=true" \
  "env.eval.episode_auditor.visual_camera_keys=$CAMS" \
  "++env.eval.episode_auditor.save_sim_state=true" \
  "++env.eval.episode_auditor.sim_state_chunk=8" \
  "++env.eval.episode_auditor.save_grades=true" \
  "++env.eval.episode_auditor.aux_safety.enabled=true" \
  "++env.eval.episode_auditor.aux_safety.count_as_unsafe=true" \
  "${BRANCH_ARGS[@]}" \
  "${PLACE_ARGS[@]}" \
  > "$OUT/launcher.log" 2>&1
echo "EXIT=$? OUT=$OUT" | tee -a "$OUT/launcher.log"
