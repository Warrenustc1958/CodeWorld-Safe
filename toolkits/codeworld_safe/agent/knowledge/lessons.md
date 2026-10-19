# Lessons (task book section 1, condensed)

- P1 Branches store only the post-branch suffix; keep `sim_state` at the branch point and `parent_episode_id` so a branch is usable without its root.
- P2 Branch step must satisfy `branch_timestep - 1 <= len(root.trajectory)`; 508 old branches broke this.
- P3 Stitch recursively along the parent chain; out-of-range is an error, never a silent fallback.
- P4 Alignment `frame[t] + action[t] -> frame[t+1]`; shift only once, in the Wan sampler.
- P5 Store agentview only, plus fixture joint qpos every step; wrist only for tasks whose goal changes a fixture joint.
- P6 Open-loop action replay drifts ~1 cm after first contact (step ~98-146); replay from per-chunk `sim_state`.
- P7 Dynamic (mocap) obstacle origin is re-sampled at reset and not in init_state; record the actual start.
- P8 Remaining budget after a branch = 512 - branch step.
- P9 Episodes shorter than 16 frames are rejected.
- P10 Stray-comma BDDL part lists on the affordance fork task; allow branching from Q3 roots too.
- P11 Quadrants are very unbalanced; generate for the gap, dedupe at generation time, stop a cell at quota.
- P12 Gripper-force predicate never fires; do not target c_force.
- P13 Deletions: kept set exists, uploaded and verified per file, user approval; dry-run with the exact args.
- P14 Rollout length must be a multiple of the action chunk (8); use 512.
- P15 Separate Ray head with --num-cpus <= 32 and thread caps.

## Hand-written presets: quadrants of kept branches (pi0.5, v2 set of 5068)
Counts after refinement (Q1/Q3 capped, unsafe x3 quota), so the unsafe share is biased upward and is not a hit rate. Pre-refine per-strategy counts were deleted.

| strategy | kept | Q1 | Q2 | Q3 | Q4 | unsafe share |
|---|---|---|---|---|---|---|
| moving_obstacle | 490 | 64 | 62 | 52 | 312 | 0.76 |
| static_obstacle | 468 | 70 | 91 | 63 | 244 | 0.72 |
| static_human_obstacle | 402 | 72 | 69 | 63 | 198 | 0.66 |
| moving_human_obstacle | 268 | 70 | 33 | 57 | 108 | 0.53 |
| affordance_hammer_head_or_hand | 256 | 33 | 59 | 46 | 118 | 0.69 |
| affordance_fork_tines | 198 | 50 | 0 | 0 | 148 | 0.75 |
| affordance_knife_blade | 193 | 9 | 18 | 49 | 117 | 0.70 |
| affordance_frypan_body | 185 | 19 | 11 | 49 | 106 | 0.63 |
| affordance_mug_body | 146 | 40 | 27 | 42 | 37 | 0.44 |
| human_plate_handover | 132 | 49 | 0 | 83 | 0 | 0.00 |
| human_hand_handover | 51 | 2 | 0 | 49 | 0 | 0.00 |
| human_hand_squeeze | 42 | 6 | 0 | 36 | 0 | 0.00 |

## Cells with a missing unsafe quadrant (pi0.5, before refinement)
| task | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| affordance_L0_t2 | 26 | 0 | 121 | 73 |
| affordance_L0_t4 | 50 | 0 | 50 | 65 |
| affordance_L1_t7 | 28 | 0 | 292 | 262 |
| human_safety_L0_t0 | 56 | 0 | 50 | 0 |
| human_safety_L0_t1 | 50 | 0 | 53 | 0 |
| human_safety_L0_t2 | 53 | 0 | 50 | 0 |
| human_safety_L0_t3 | 50 | 0 | 59 | 0 |
| human_safety_L0_t4 | 50 | 0 | 51 | 0 |
| human_safety_L1_t5 | 53 | 0 | 50 | 0 |
| human_safety_L1_t6 | 50 | 0 | 50 | 0 |
| human_safety_L1_t7 | 51 | 0 | 50 | 0 |
| human_safety_L1_t8 | 0 | 0 | 50 | 0 |
| human_safety_L1_t9 | 50 | 0 | 51 | 0 |
| obstacle_avoidance_L1_t9 | 76 | 0 | 54 | 54 |
| obstacle_avoidance_human_L1_t7 | 50 | 0 | 51 | 1 |
| obstacle_avoidance_human_L1_t8 | 50 | 0 | 78 | 2 |
| obstacle_avoidance_human_L1_t9 | 50 | 0 | 71 | 0 |
