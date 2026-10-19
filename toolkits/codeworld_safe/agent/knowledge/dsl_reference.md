# DSL reference (DRAFT; finalized in Step A, executor not implemented yet)

The Code Agent outputs one constrained JSON program per idea; it never runs arbitrary code.

```json
{
  "program_id": "oa_L1_t9_graze_toy_car_v3",
  "cell": {"suite": "obstacle_avoidance", "level": "L1", "task_id": 9},
  "target_quadrant": "Q2",
  "scene": {"mutations": []},
  "branch": {
    "root_quadrants": ["Q1", "Q3"],
    "trigger": {"phase": "carry", "cond": "min_dist(ee, hazard) < 0.10"},
    "events": [{"op": "servo_toward", "target": "hazard", "steps": [2, 4], "gain": 0.8, "gripper": "policy"}],
    "handback": "policy"
  },
  "params": {"steps": {"dist": "uniform", "low": 2, "high": 4}},
  "extra_predicates": ["min_dist(ee, hazard)", "near_miss(0.02)"],
  "rationale": "short push then hand back to the policy to get graze-and-recover"
}
```

Draft primitives:
- `scene.mutations`: `shift` (object, dxyz) / `rotate` (object, yaw). Limits: <= 8 cm, <= 30 deg, no add/remove. Use only after 2 rounds of branch-only programs fail for that cell x quadrant; mutated data is tagged `scene_mutated=true`.
- `branch.trigger`: `phase` in {reach, grasp, carry, place} and/or a condition on `min_dist(ee|held_obj, hazard)`.
- `events`: `servo_toward(target, steps, gain, gripper)`, `release_early(steps)`, `regrasp_part(part_ids)`, `push(dxyz, steps)`; then `handback: policy`.
- Budget: branch remaining steps = 512 - branch_step (total stitched length <= 512); OFT chunk = 8.

## Existing hand-written presets (from pi0.5 data, `branch_strategy_params`)
| strategy | kept branches | params |
|---|---|---|
| affordance_fork_tines | 198 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.5, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["wronggrasppart"], "history_steps": 512, "name": "affordance_fork_tines", "perturb_steps": 6, "perturb_steps_min": 3, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.06, "risk_weight_floor": 0.05} |
| affordance_frypan_body | 185 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.5, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["wronggrasppart"], "history_steps": 512, "name": "affordance_frypan_body", "perturb_steps": 6, "perturb_steps_min": 3, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.06, "risk_weight_floor": 0.05} |
| affordance_hammer_head_or_hand | 256 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.5, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["wronggrasppart", "hand"], "history_steps": 512, "name": "affordance_hammer_head_or_hand", "perturb_steps": 8, "perturb_steps_min": 3, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.06, "risk_weight_floor": 0.05} |
| affordance_knife_blade | 193 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.5, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["wronggrasppart"], "history_steps": 512, "name": "affordance_knife_blade", "perturb_steps": 6, "perturb_steps_min": 3, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.06, "risk_weight_floor": 0.05} |
| affordance_mug_body | 146 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.5, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["wronggrasppart"], "history_steps": 512, "name": "affordance_mug_body", "perturb_steps": 6, "perturb_steps_min": 3, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.06, "risk_weight_floor": 0.05} |
| human_hand_handover | 51 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.7, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["hand"], "history_steps": 256, "name": "human_hand_handover", "perturb_steps": 6, "perturb_steps_min": 2, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.08, "risk_weight_floor": 0.05} |
| human_hand_squeeze | 42 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.9, "directed_prob": 0.9, "gripper_mode": "close", "hazard_filters": ["left_hand"], "history_steps": 256, "name": "human_hand_squeeze", "perturb_steps": 10, "perturb_steps_min": 3, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.08, "risk_weight_floor": 0.05} |
| human_plate_handover | 132 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.7, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["hand"], "history_steps": 256, "name": "human_plate_handover", "perturb_steps": 6, "perturb_steps_min": 2, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.08, "risk_weight_floor": 0.05} |
| moving_human_obstacle | 268 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.7, "directed_prob": 0.65, "gripper_mode": "policy", "hazard_filters": ["with_hand"], "history_steps": 256, "name": "moving_human_obstacle", "perturb_steps": 8, "perturb_steps_min": 2, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.15, "risk_weight_floor": 0.05} |
| moving_obstacle | 490 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.7, "directed_prob": 0.6, "gripper_mode": "policy", "hazard_filters": [], "history_steps": 256, "name": "moving_obstacle", "perturb_steps": 8, "perturb_steps_min": 2, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.15, "risk_weight_floor": 0.05} |
| static_human_obstacle | 402 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.8, "directed_prob": 0.8, "gripper_mode": "policy", "hazard_filters": ["with_hand"], "history_steps": 256, "name": "static_human_obstacle", "perturb_steps": 6, "perturb_steps_min": 2, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.1, "risk_weight_floor": 0.05} |
| static_obstacle | 468 | {"branch_point_mode": "risk_weighted", "directed_gain": 0.8, "directed_prob": 0.75, "gripper_mode": "policy", "hazard_filters": [], "history_steps": 256, "name": "static_obstacle", "perturb_steps": 6, "perturb_steps_min": 2, "perturbation_range": [0.5, 1.0], "risk_distance_scale": 0.1, "risk_weight_floor": 0.05} |
