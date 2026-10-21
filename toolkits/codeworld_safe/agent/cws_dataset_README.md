---
license: apache-2.0
tags:
  - robotics
  - world-model
  - counterfactual
  - libero-safety
---

# CodeWorld-Safe π0.5 counterfactual rollouts (CWS)

π0.5 rollouts on LIBERO-Safety (4 suites × L0/L1 × 5 tasks = 40 tasks) with
LLM-program counterfactual branches, labelled by success × safety quadrant, plus the
index used to train the action-conditioned Wan2.2-TI2V-5B world model.

Quadrants: Q1 success & safe, Q2 success & unsafe, Q3 fail & safe, Q4 fail & unsafe.
Safety comes from the geometry grader (`safety_grader`); the official LIBERO-Safety
robot-contact predicate never fires (geom id vs name bug) and is not used.

## Layout

| Path | Content |
|---|---|
| `step_e/runs/<suite>_<level>_t<task>/` | All Step E audits. Per episode a `.json` (per-step action, predicates, terminations, grades, observation, sim state) and `visual/*.npz` (`agentview_image`, `robot0_eye_in_hand_image`, uint8 [T+1, 256, 256, 3], policy view) |
| `step_e/archive/outcomes.jsonl` | Collector admission records (`admitted`, `duplicate`, `near_miss`, `quadrant`, `branch_timestep`, paths) |
| `legacy/` | Earlier π0.5 CF episodes used to fill missing task × quadrant cells, and the old val/test held-out episodes. Agentview only for most |
| `index/cws_v1/manifest.jsonl` | Training index (paths relative to the repo root) |
| `index/cws_v1/labels/<idx>.npz`, `index/v1/labels/` | Stitched `actions` [n-1, 7], `violation`, `success` |
| `index/cws_v1/motion/<idx>.npy` | `motion[s]` = mean |frame[s+11] − frame[s+4]| (0–255), the 8 loss frames of a 13-frame window starting at s |
| `index/cws_v1/selection_report.json` | Filter / fill parameters and per-task counts |
| `regrade/kin_legacy_5k.jsonl` | Legacy episodes regraded with the geometry grader (99.2 % quadrant agreement on validated Step E audits) |

## Branches

A branch audit stores only the post-branch suffix: timesteps restart at 1 and its
frame 0 equals root frame `bt − 1`, with `bt = episode.branch_timestep`. The full
episode is `root[:bt−1] + branch` (frames and actions). Every stitched episode is
≤ 512 steps (`max_total_steps`). The manifest already lists both segments in
`agent_segments` / `wrist_segments` as `[npz, start, stop]`.

Action convention: `frame[t] + action[t] → frame[t+1]`; 7-D delta EEF (xyz, axis-angle)
plus gripper (−1 open, +1 closed).

## Index `cws_v1`

| Split | Episodes | Notes |
|---|---|---|
| train | 9335 | 8867 Step E + 468 legacy fill; Q1 3080 / Q2 1029 / Q3 2015 / Q4 3211 |
| val | 254 | legacy held-out init states (unchanged) |
| test | 345 | legacy held-out init states (unchanged) |

- Step E episodes on a legacy val/test init state `(task, trial_id)` are excluded (1047).
- Episodes with > 80 % of windows moving < 2/255 are dropped (323), keeping ≥ 10 per task × quadrant; trailing static windows are trimmed to 8.
- Legacy fill: for task × quadrant cells with < 40 Step E episodes, legacy train episodes of the same task and regraded quadrant, ≤ 512 steps, with their prefix, highest motion first.
- `quadrant` in train rows is the geometry-grader label; held-out rows keep the original label (`quadrant_regraded` is also given).

Step E collection was still running when this was uploaded; `step_e/runs` may hold more
episodes than the index uses.
