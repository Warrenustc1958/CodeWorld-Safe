#!/usr/bin/env bash
# Step B smoke: one cell, one cws_cf_program, OpenVLA-OFT chunk 8, shadow_collect.
#   GPU=7 ENVS=4 bash toolkits/codeworld_safe/cf/smoke_step_b.sh
# Output: $OUT (audits under libero_safety_audits/, log in $OUT/launcher.log).
set -uo pipefail
R=/mnt/nfs/fjj/Code/rlinf-safety-runtime
REPO=/mnt/nfs/fjj/Code/CodeWorld-Safe
source "$R/tmp/sft_oft_env.sh"
export RAY_TMPDIR=/tmp/re5 TMPDIR=/tmp/te5 RAY_ADDRESS=127.0.0.1:6391
IM=$R/local-packages/imagemagick/prefix
export MAGICK_HOME=$IM/usr LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IM/usr/lib/x86_64-linux-gnu
GPU=${GPU:-7}; ENVS=${ENVS:-4}; MAX_BRANCHES=${MAX_BRANCHES:-2}
MODEL=${MODEL:-/mnt/nfs/fjj/Models/RLinf-OpenVLAOFT-LIBERO-130}
UNNORM=${UNNORM:-libero_130_no_noops_trajall}
PROGRAM=${PROGRAM:-$REPO/toolkits/codeworld_safe/cf/smoke_oa_l1_t5.json}
KNOWLEDGE=$REPO/toolkits/codeworld_safe/agent/knowledge
OUT=${OUT:-${CWS_DATA:-/data_cf/cws}/rollouts/cws_smoke/oa_L1_t5_$(date +%m%d_%H%M)}
EPISODE_STEPS=512
MAX_ROLLOUT_STEPS=$(((MAX_BRANCHES + 1) * EPISODE_STEPS))
mkdir -p "$OUT"
bash "$REPO/toolkits/codeworld_safe/sft/start_eval_ray_head.sh"
cd "$REPO"
VENV_PATH=/mnt/nfs/fjj/Code/.venv-openvlaoft-libero-safety \
LIBERO_SAFETY_PATH=/mnt/nfs/fjj/Code/LIBERO-Safety \
RLINF_RUNTIME_DIR=/tmp/re5rt \
MODE=eval SUITE=obstacle_avoidance LEVEL=L1 GPU_RANKS=$GPU TOTAL_ENVS=$ENVS \
MODEL=$MODEL UNNORM_KEY=$UNNORM IS_LORA=False \
bash evaluations/run_safelibero.sh \
  "runner.logger.log_path=$OUT" \
  "env.eval.task_id_filter=[5]" \
  "env.eval.ignore_terminations=false" \
  "rollout.model.num_action_chunks=8" \
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
  "++env.eval.episode_auditor.aux_safety.enabled=true" \
  "++env.eval.episode_auditor.aux_safety.count_as_unsafe=true" \
  "env.eval.episode_auditor.counterfactual_branch.enabled=true" \
  "env.eval.episode_auditor.counterfactual_branch.max_branches_per_root=$MAX_BRANCHES" \
  "env.eval.episode_auditor.counterfactual_branch.target_quadrants=[Q2,Q4]" \
  "++env.eval.episode_auditor.counterfactual_branch.root_quadrants=[Q1,Q3]" \
  "++env.eval.episode_auditor.counterfactual_branch.program=$PROGRAM" \
  "++env.eval.episode_auditor.counterfactual_branch.program_knowledge_dir=$KNOWLEDGE" \
  "++env.eval.episode_auditor.counterfactual_branch.action_chunk=8" \
  > "$OUT/launcher.log" 2>&1
echo "EXIT=$? OUT=$OUT" | tee -a "$OUT/launcher.log"
