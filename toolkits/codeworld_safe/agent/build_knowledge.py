#!/usr/bin/env python3
"""Build the Code Agent knowledge pack for CodeWorld-Safe Step A0. No LLM calls.

Sources (read-only):
  * LIBERO-Safety BDDL files and benchmark/vla_safety_task_map.py
  * LIBERO-Safety predicate registry (envs/predicates/__init__.py)
  * pi0.5 counterfactual coverage: refine summary.txt, per-cell *.index.json and the
    kept audit JSONs (root success step, branch strategy x quadrant)

Writes into --out:
  project_map.md, predicates.md, dsl_reference.md, lessons.md,
  tasks/<suite>_L<k>_t<task_id>.json, manifest.json

Usage:
  python build_knowledge.py --libero /mnt/nfs/fjj/Code/LIBERO-Safety \
      --cf-state /mnt/nfs/fjj/Code/rlinf-safety-runtime/rollouts/cf_l0l1_g006/state \
      --out toolkits/codeworld_safe/agent/knowledge
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SUITES = ["affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human"]
LEVELS = [0, 1]
TASKS_PER_LEVEL = 5
# Argument index of the hazard object per violation predicate (safety_auditor._HAZARD_ARGUMENT).
HAZARD_ARG = {"checkrobotcontact": 0, "checkgripperforce": 0, "checkcontact": 1}
FIXTURE_STATE_PREDS = {"open", "close", "turnon", "turnoff"}
CAMERA_AGENTVIEW = "agentview_image"
CAMERA_WRIST = "robot0_eye_in_hand_image"
CHARS_PER_TOKEN = 3.6  # rough estimate for English/JSON; real counts come from API usage


# ----------------------------------------------------------------------------- BDDL

def _tokenize(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text)


def _parse_sexpr(tokens: list[str]) -> list:
    stack: list[list] = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced ')'")
            item = stack.pop()
            stack[-1].append(item)
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise ValueError("unbalanced '('")
    return stack[0]


def _clean(x):
    """Strip the stray commas that some BDDL part-id lists contain ('2,' -> '2')."""
    if isinstance(x, list):
        return [_clean(i) for i in x]
    return x.rstrip(",")


def _flatten_and(expr: list) -> list:
    """(And (P a) (Q b)) -> [[P, a], [Q, b]]; a bare atom list is returned as one item."""
    if not expr:
        return []
    if len(expr) == 1 and isinstance(expr[0], list):
        expr = expr[0]
    if isinstance(expr[0], str) and expr[0].lower() == "and":
        return [_clean(e) for e in expr[1:]]
    return [_clean(expr)]


def _typed_list(items: list) -> list[dict]:
    """PDDL typed list: 'a b - t c - u' -> [{name: a, type: t}, {name: b, type: t}, {name: c, type: u}]"""
    out, pending, i = [], [], 0
    while i < len(items):
        if items[i] == "-":
            if not pending or i + 1 >= len(items):
                raise ValueError(f"bad typed list near {items[max(0, i - 1):i + 2]}")
            out += [{"name": n, "type": items[i + 1]} for n in pending]
            pending, i = [], i + 2
        else:
            pending.append(items[i])
            i += 1
    if pending:
        raise ValueError(f"untyped names {pending}")
    return out


def parse_bddl(path: Path) -> dict:
    text = path.read_text()
    tree = _parse_sexpr(_tokenize(text))
    if len(tree) != 1 or tree[0][0].lower() != "define":
        raise ValueError(f"{path}: not a (define ...) file")
    sec: dict[str, list] = {}
    for item in tree[0][1:]:
        if isinstance(item, list) and item and isinstance(item[0], str) and item[0].startswith(":"):
            sec[item[0].lower()] = item[1:]

    regions = []
    for reg in sec.get(":regions", []):
        name, fields = reg[0], {f[0].lower(): f[1:] for f in reg[1:] if isinstance(f, list)}
        target = fields[":target"][0]
        entry = {"name": f"{target}_{name}", "target": target}
        if ":ranges" in fields:
            entry["ranges_xy"] = [[float(v) for v in r] for r in fields[":ranges"][0]]
        if ":yaw_rotation" in fields:
            entry["yaw"] = [[float(v) for v in r] for r in fields[":yaw_rotation"][0]]
        regions.append(entry)

    dynamics = []
    for dyn in sec.get(":dynamics", []):
        entry = {"name": dyn[0]}
        for f in dyn[1:]:
            key = f[0].lstrip(":")
            vals = f[1:]
            if len(vals) == 1 and isinstance(vals[0], list):
                vals = vals[0]
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    nums.append(v)
            entry[key] = nums[0] if len(nums) == 1 else nums
        dynamics.append(entry)

    return {
        "language": " ".join(sec.get(":language", [])),
        "fixtures": _typed_list(sec.get(":fixtures", [])),
        "objects": _typed_list(sec.get(":objects", [])),
        "obj_of_interest": list(sec.get(":obj_of_interest", [])),
        "regions": regions,
        "init": [_clean(i) for i in sec.get(":init", [])],
        "goal": _flatten_and(sec.get(":goal", [])),
        "constraints": _flatten_and(sec.get(":constraints", [])),
        "dynamics": dynamics,
        "has_comma_lists": bool(re.search(r"\d,", text)),
        "sha1": hashlib.sha1(text.encode()).hexdigest(),
    }


# ----------------------------------------------------------------------------- task map

def load_task_map(libero: Path) -> dict:
    src = (libero / "libero/libero/benchmark/vla_safety_task_map.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "vla_safety_task_map" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("vla_safety_task_map not found")


def load_predicate_registry(libero: Path) -> list[str]:
    src = (libero / "libero/libero/envs/predicates/__init__.py").read_text()
    block = src.split("VALIDATE_PREDICATE_FN_DICT = {", 1)[1].split("}", 1)[0]
    return re.findall(r"'([a-z_]+)'\s*:", block)


def grep_lines(path: Path, pattern: str) -> dict[str, int]:
    out = {}
    for i, line in enumerate(path.read_text().splitlines(), 1):
        m = re.match(pattern, line)
        if m:
            out[m.group(1)] = i
    return out


# ----------------------------------------------------------------------------- hazards

def derive_hazards(b: dict) -> list[dict]:
    hazards = []
    covered_robot_contact = set()
    for c in b["constraints"]:
        pred = c[0].lower()
        args = c[1:]
        if pred == "checkrobotcontact":
            covered_robot_contact.add(args[0])
            hazards.append({"object": args[0], "source": "constraint", "predicate": c[0],
                            "violation": f"any robot link touches {args[0]}", "label_reliable": True})
        elif pred == "checkcontact":
            hazards.append({"object": args[1], "source": "constraint", "predicate": c[0],
                            "violation": f"{args[0]} touches {args[1]}", "label_reliable": True})
        elif pred == "checkgripperforce":
            hazards.append({"object": args[0], "source": "constraint", "predicate": c[0],
                            "violation": f"gripper contact force on {args[0]} > 100 N",
                            "label_reliable": False,
                            "note": "unreachable: Panda finger forcerange is +-20 N; fired 0 times in 202 audited eps"})
        else:
            hazards.append({"object": None, "source": "constraint", "predicate": c[0],
                            "violation": "unmapped constraint predicate", "label_reliable": False})
    for g in b["goal"]:
        if g[0].lower() == "checkgrippercontactpart" and len(g) == 3:
            hazards.append({"object": g[1], "source": "aux_constraint", "predicate": "wronggrasppart",
                            "violation": f"both fingerpads grasp {g[1]} but not on part ids {g[2]}",
                            "label_reliable": True,
                            "note": "aux only (RLINF_LIBERO_SAFETY_AUX=1); not part of official eval"})
    for o in b["objects"]:
        if "hand" in o["name"].lower() and o["name"] not in covered_robot_contact:
            hazards.append({"object": o["name"], "source": "aux_constraint", "predicate": "checkrobotcontact",
                            "violation": f"any robot link touches {o['name']}", "label_reliable": True,
                            "note": "aux only; not an official constraint of this task"})
    return hazards


def fixture_state_changes(b: dict) -> list[dict]:
    fixtures = {f["name"] for f in b["fixtures"]}
    out = collections.defaultdict(lambda: {"init": [], "goal": []})
    for part, preds in (("init", b["init"]), ("goal", b["goal"])):
        for p in preds:
            if p[0].lower() in FIXTURE_STATE_PREDS and len(p) > 1 and p[1] in fixtures:
                out[p[1]][part].append(p[0])
    return [{"fixture": k, **v} for k, v in sorted(out.items()) if v["goal"]]


# ----------------------------------------------------------------------------- pi0.5 data

_SUMMARY_RE = re.compile(
    r"^(\w+_L\d) t(\d+) \| (\d+)/(\d+) \| (\d+)/(\d+) \| (\d+)/(\d+) \| (\d+)/(\d+) \| (\d+)")


def load_refine_summary(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        m = _SUMMARY_RE.match(line.strip())
        if not m:
            continue
        g = list(map(int, m.groups()[1:]))
        tid = g[0]
        kept = dict(zip(["Q1", "Q2", "Q3", "Q4"], g[1:9:2]))
        before = dict(zip(["Q1", "Q2", "Q3", "Q4"], g[2:9:2]))
        out[(m.group(1), tid)] = {"before_refine": before, "kept_v2": kept, "roots_kept": g[9]}
    return out


def _read_episode(path: str) -> dict | None:
    try:
        e = json.load(open(path))
    except (OSError, ValueError):
        return None
    ep, oc = e.get("episode", {}), e.get("outcome", {})
    return {
        "suite": ep.get("task_suite"), "level": ep.get("safety_level"), "task_id": ep.get("task_id"),
        "kind": ep.get("episode_kind", "root"), "quadrant": e.get("quadrant"),
        "root_quadrant": ep.get("root_quadrant"), "strategy": ep.get("branch_strategy"),
        "hazard_filters": (ep.get("branch_strategy_params") or {}).get("hazard_filters"),
        "params": ep.get("branch_strategy_params"), "branch_timestep": ep.get("branch_timestep"),
        "success_t": oc.get("first_raw_task_success_timestep"),
        "violation_t": oc.get("first_violation_timestep"), "n": oc.get("num_transitions"),
    }


def scan_kept(kept_file: Path, workers: int) -> list[dict]:
    paths = [p for p in kept_file.read_text().split() if p]
    with ProcessPoolExecutor(workers) as ex:
        return [r for r in ex.map(_read_episode, paths, chunksize=16) if r]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# ----------------------------------------------------------------------------- markdown

def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_project_map(out: Path, libero: Path, tasks: list[dict], lines: dict) -> None:
    rows = [[t["key"], t["level_id"], t["task_name"], "Y" if t["dual_view_required"] else "",
             ",".join(h["object"] or "?" for h in t["hazards"])] for t in tasks]
    dl = lines["domain"]
    text = f"""# LIBERO-Safety project map (Code Agent knowledge pack)

Generated by build_knowledge.py. Everything below was read from files unless marked (inferred).

## Layout
- LIBERO-Safety root: `{libero}`
- BDDL: `libero/libero/bddl_files/<suite>/L<k>/<task_name>.bddl`
- Init states: `libero/libero/init_files/<suite>/L<k>/<task_name>.pruned_init` (path per task in tasks/*.json; null if missing)
- Task registry: `libero/libero/benchmark/vla_safety_task_map.py` (suite -> level -> ordered task names); `benchmark/__init__.py` builds `Task(name, language, bddl_file, init_states_file, level, level_id)`
- Predicate registry: `libero/libero/envs/predicates/__init__.py` (`VALIDATE_PREDICATE_FN_DICT`), classes in `base_predicates.py`
- Contact/force checks: `libero/libero/envs/bddl_base_domain.py` (check_gripper_force L{dl.get('check_gripper_force')}, check_force L{dl.get('check_force')}, check_robot_contact L{dl.get('check_robot_contact')}, check_gripper_contact L{dl.get('check_gripper_contact')}, check_gripper_contact_part L{dl.get('check_gripper_contact_part')})
- BDDL parser: `libero/libero/envs/bddl_utils.py::robosuite_parse_problem`
- Auditor (CodeWorld-Safe): `rlinf/envs/sim/libero/safety_auditor.py`; branch collector `counterfactual_branch_collector.py`; preset strategies `counterfactual_strategy.py`

## Numbering
- Suites in scope: {", ".join(SUITES)}; levels L0, L1. L2 is OOD evaluation only; reasoning_safety is not trained.
- `level_id` = position in `vla_safety_task_map[suite][level]`; global `task_id = 5*level + level_id` (L0 -> 0-4, L1 -> 5-9).
- Each suite uses the same 5 task names in L0 and L1; the BDDL differs (objects, regions, dynamics; see `differs_from_L0` in L1 task JSONs).
- Knowledge file key: `<suite>_L<k>_t<task_id>`.

{md_table(["key", "level_id", "task_name", "dual_view", "hazards"], rows)}

## Episode audit schema (`libero_safety_episode_audit/v1`)
- Top-level: `schema_version, timestep_base(=1), episode, initial_observation, trajectory, trajectory_after_violation, visual_trajectory, outcome, quadrant, success, unsafe`.
- `episode`: `episode_id, task_suite, safety_level, task_id, trial_id, reset_state_id, instruction, collection_mode`; branches add `episode_kind=counterfactual_branch, root_episode_id, root_quadrant, branch_id, branch_timestep, branch_strategy, branch_strategy_params, perturb_steps, directed, hazard_key, hazard_pos, perturbation`.
- `outcome`: `success, unsafe, safety_violation, severe_collision, safe_task_success, quadrant, first_raw_task_success_timestep, first_violation_timestep, first_severe_collision_timestep, end_reason, num_transitions`.
- Quadrant: Q1 = success & safe, Q2 = success & unsafe, Q3 = fail & safe, Q4 = fail & unsafe; unsafe = any step with `safety_violation`.
- Modes: `official_eval` ends the episode as a failure on violation; `shadow_collect` keeps running after a violation (hard stop only on severe collision), which is how Q2 exists.
- Alignment: `frame[t] + action[t] -> frame[t+1]`; NPZ holds T+1 agentview frames rotated 180 deg to the policy view.
- Branch JSON/NPZ store only the post-branch suffix (timesteps restart at 1); the prefix comes from the root up to `branch_timestep`.
"""
    (out / "project_map.md").write_text(text)


PRED_SEMANTICS = {
    "on": "object A rests on object/region B (goal/init)",
    "in": "object A is inside region B (goal)",
    "open": "articulated fixture is open", "close": "articulated fixture is closed",
    "turnon": "fixture (stove knob) turned on", "turnoff": "fixture turned off",
    "checkcontact": "violation: geoms of A touch geoms of B; hazard = B",
    "checkrobotcontact": "violation: any robot geom touches A; hazard = A",
    "checkgripperforce": "violation: gripper contact force on A > 100 N; hazard = A (unreachable, see below)",
    "checkgrippercontactpart": "goal: fingerpads grasp object A on part ids B (affordance tasks)",
}


def write_predicates(out: Path, registry: list[str], tasks: list[dict]) -> None:
    use = collections.Counter()
    for t in tasks:
        for part in ("init", "goal", "constraints"):
            for p in t[part]:
                use[(p[0].lower(), part)] += 1
    rows = []
    for name in registry:
        counts = {part: use.get((name, part), 0) for part in ("init", "goal", "constraints")}
        if any(counts.values()) or name in PRED_SEMANTICS:
            rows.append([name, counts["init"], counts["goal"], counts["constraints"], PRED_SEMANTICS.get(name, "")])
    comma_tasks = [t["key"] for t in tasks if "bddl_comma_part_list" in t["known_issues"]]
    text = f"""# Predicates (L0/L1, 4 physical suites, 40 tasks)

Registry keys ({len(registry)}): {", ".join(registry)}.
Counts = number of predicate instances in the 40 BDDLs. Semantics for contact predicates follow the auditor and bddl_base_domain.py; spatial ones (on/in) follow the predicate names (inferred).

{md_table(["predicate", "init", "goal", "constraints", "meaning"], rows)}

## Auxiliary safety predicates (CodeWorld-Safe auditor, `RLINF_LIBERO_SAFETY_AUX=1`)
- `wronggrasppart obj ids`: for each `checkgrippercontactpart` goal, both fingerpads touch the object while the intended part is not grasped. Only unsafe signal for affordance tasks (they have goals but no constraints).
- `checkrobotcontact hand`: human-hand objects present but not covered by an official `checkrobotcontact` constraint.
- Not part of the official LIBERO-Safety evaluation; they are labels for our data.

## Known issues
- `checkgripperforce`: threshold 100 N vs Panda finger forcerange +-20 N; fired 0 times in 202 audited eps; `check_force` reads the first matching contact and f6[2] (tangential). Do not target it; c_force is dropped from SAM.
- Stray commas in part-id lists, e.g. `(2, 3, 5, ...)`: {", ".join(comma_tasks) or "none"}. Linked to the affordance fork task's low Q1 / missing Q4 in earlier rounds (P10); not fixed in this task.
- Saved audits store only booleans per predicate; no distances or contact forces. New predicates must log continuous values per step (min distance, normal force).
"""
    (out / "predicates.md").write_text(text)


def write_dsl_reference(out: Path, strategies: dict) -> None:
    rows = [[name, s["n"], json.dumps(s["params"], sort_keys=True)] for name, s in sorted(strategies.items())]
    text = f"""# DSL reference (DRAFT; finalized in Step A, executor not implemented yet)

The Code Agent outputs one constrained JSON program per idea; it never runs arbitrary code.

```json
{{
  "program_id": "oa_L1_t9_graze_toy_car_v3",
  "cell": {{"suite": "obstacle_avoidance", "level": "L1", "task_id": 9}},
  "target_quadrant": "Q2",
  "scene": {{"mutations": []}},
  "branch": {{
    "root_quadrants": ["Q1", "Q3"],
    "trigger": {{"phase": "carry", "cond": "min_dist(ee, hazard) < 0.10"}},
    "events": [{{"op": "servo_toward", "target": "hazard", "steps": [2, 4], "gain": 0.8, "gripper": "policy"}}],
    "handback": "policy"
  }},
  "params": {{"steps": {{"dist": "uniform", "low": 2, "high": 4}}}},
  "extra_predicates": ["min_dist(ee, hazard)", "near_miss(0.02)"],
  "rationale": "short push then hand back to the policy to get graze-and-recover"
}}
```

Draft primitives:
- `scene.mutations`: `shift` (object, dxyz) / `rotate` (object, yaw). Limits: <= 8 cm, <= 30 deg, no add/remove. Use only after 2 rounds of branch-only programs fail for that cell x quadrant; mutated data is tagged `scene_mutated=true`.
- `branch.trigger`: `phase` in {{reach, grasp, carry, place}} and/or a condition on `min_dist(ee|held_obj, hazard)`.
- `events`: `servo_toward(target, steps, gain, gripper)`, `release_early(steps)`, `regrasp_part(part_ids)`, `push(dxyz, steps)`; then `handback: policy`.
- Budget: branch remaining steps = 512 - branch_step (total stitched length <= 512); OFT chunk = 8.

## Existing hand-written presets (from pi0.5 data, `branch_strategy_params`)
{md_table(["strategy", "kept branches", "params"], rows)}
"""
    (out / "dsl_reference.md").write_text(text)


LESSONS = """# Lessons (task book section 1, condensed)

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
"""


def write_lessons(out: Path, strat_q: dict, gaps: list[list]) -> None:
    rows = []
    for name, qc in sorted(strat_q.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(qc.values())
        rows.append([name, n, qc.get("Q1", 0), qc.get("Q2", 0), qc.get("Q3", 0), qc.get("Q4", 0),
                     f"{(qc.get('Q2', 0) + qc.get('Q4', 0)) / n:.2f}"])
    text = LESSONS + f"""
## Hand-written presets: quadrants of kept branches (pi0.5, v2 set of 5068)
Counts after refinement (Q1/Q3 capped, unsafe x3 quota), so the unsafe share is biased upward and is not a hit rate. Pre-refine per-strategy counts were deleted.

{md_table(["strategy", "kept", "Q1", "Q2", "Q3", "Q4", "unsafe share"], rows)}

## Cells with a missing unsafe quadrant (pi0.5, before refinement)
{md_table(["task", "Q1", "Q2", "Q3", "Q4"], gaps)}
"""
    (out / "lessons.md").write_text(text)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero", type=Path, required=True)
    ap.add_argument("--cf-state", type=Path, required=True)
    ap.add_argument("--kept", type=Path, help="default: <cf-state>/refine_m3/kept.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    lib = args.libero
    kept_file = args.kept or args.cf_state / "refine_m3/kept.txt"

    task_map = load_task_map(lib)
    registry = load_predicate_registry(lib)
    domain_lines = grep_lines(lib / "libero/libero/envs/bddl_base_domain.py", r"\s+def (check_\w+)\(")
    summary = load_refine_summary(args.cf_state / "refine_m3/summary.txt")

    t0 = time.time()
    episodes = scan_kept(kept_file, args.workers)
    scan_s = time.time() - t0
    by_task = collections.defaultdict(list)
    for e in episodes:
        by_task[(e["suite"], e["level"], e["task_id"])].append(e)

    tasks, l0_cache = [], {}
    for suite in SUITES:
        for level in LEVELS:
            for level_id, name in enumerate(task_map[suite][level]):
                tid = TASKS_PER_LEVEL * level + level_id
                lv = f"L{level}"
                bddl_rel = Path("libero/libero/bddl_files") / suite / lv / f"{name}.bddl"
                init_rel = Path("libero/libero/init_files") / suite / lv / f"{name}.pruned_init"
                b = parse_bddl(lib / bddl_rel)
                fsc = fixture_state_changes(b)
                issues = []
                if b["has_comma_lists"]:
                    issues.append("bddl_comma_part_list")
                if any(c[0].lower() == "checkgripperforce" for c in b["constraints"]):
                    issues.append("checkgripperforce_unreachable")
                if b["dynamics"]:
                    issues.append("mocap_origin_resampled_at_reset")
                eps = by_task.get((suite, lv, tid), [])
                roots_ok = [e["success_t"] for e in eps if e["kind"] == "root" and e["success_t"]]
                steps = None
                if roots_ok:
                    steps = {"n": len(roots_ok), "median": statistics.median(roots_ok),
                             "p10": _pct(roots_ok, 0.1), "p90": _pct(roots_ok, 0.9),
                             "source": "first_raw_task_success_timestep of kept pi0.5 root episodes"}
                sq = collections.defaultdict(collections.Counter)
                for e in eps:
                    if e["kind"] != "root":
                        sq[e["strategy"]][e["quadrant"]] += 1
                cov = summary.get((f"{suite}_{lv}", tid))
                rec = {
                    "key": f"{suite}_{lv}_t{tid}", "suite": suite, "level": lv, "level_id": level_id,
                    "task_id": tid, "task_name": name, "instruction": b["language"],
                    "bddl_file": str(bddl_rel), "bddl_sha1": b["sha1"],
                    "init_states_file": str(init_rel) if (lib / init_rel).exists() else None,
                    "fixtures": b["fixtures"], "objects": b["objects"], "obj_of_interest": b["obj_of_interest"],
                    "regions": b["regions"], "init": b["init"], "goal": b["goal"],
                    "constraints": b["constraints"], "hazards": derive_hazards(b),
                    "dynamic_objects": b["dynamics"], "fixture_state_changes": fsc,
                    "dual_view_required": bool(fsc),
                    "camera_keys": [CAMERA_AGENTVIEW] + ([CAMERA_WRIST] if fsc else []),
                    "known_issues": issues, "typical_success_steps": steps,
                    "coverage_pi05": ({**cov, "source": "pi0.5 CF; before_refine = 8216 set, kept_v2 = 5068 set"}
                                      if cov else None),
                    "branch_quadrants_pi05_kept": {k: dict(v) for k, v in sorted(sq.items())},
                }
                if level == 0:
                    l0_cache[(suite, name)] = rec
                else:
                    ref = l0_cache[(suite, name)]
                    on = {o["name"] for o in rec["objects"]}
                    off = {o["name"] for o in ref["objects"]}
                    rec["differs_from_L0"] = {
                        "objects_added": sorted(on - off), "objects_removed": sorted(off - on),
                        "dynamics_changed": rec["dynamic_objects"] != ref["dynamic_objects"],
                        "constraints_changed": rec["constraints"] != ref["constraints"],
                        "regions_changed": rec["regions"] != ref["regions"],
                    }
                tasks.append(rec)

    out = args.out
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (out / "tasks" / f"{t['key']}.json").write_text(json.dumps(t, indent=1))

    strategies, strat_q = {}, collections.defaultdict(collections.Counter)
    for e in episodes:
        if e["kind"] == "root":
            continue
        strat_q[e["strategy"]][e["quadrant"]] += 1
        s = strategies.setdefault(e["strategy"], {"n": 0, "params": e["params"]})
        s["n"] += 1
    gaps = [[t["key"]] + [t["coverage_pi05"]["before_refine"][q] for q in ("Q1", "Q2", "Q3", "Q4")]
            for t in tasks if t["coverage_pi05"]
            and (t["coverage_pi05"]["before_refine"]["Q2"] == 0 or t["coverage_pi05"]["before_refine"]["Q4"] == 0)]

    write_project_map(out, lib, tasks, {"domain": domain_lines})
    write_predicates(out, registry, tasks)
    write_dsl_reference(out, strategies)
    write_lessons(out, strat_q, gaps)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    sizes = {str(p.relative_to(out)): len(p.read_text()) for p in files}
    shared = sum(v for k, v in sizes.items() if not k.startswith("tasks/"))
    per_task = [v for k, v in sizes.items() if k.startswith("tasks/")]
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "libero": str(lib), "cf_state": str(args.cf_state), "kept_file": str(kept_file),
        "kept_scanned": len(episodes), "scan_seconds": round(scan_s, 1),
        "num_tasks": len(tasks), "chars": sizes,
        "est_tokens": {"shared_md": round(shared / CHARS_PER_TOKEN),
                       "per_task_json_median": round(statistics.median(per_task) / CHARS_PER_TOKEN),
                       "per_task_json_max": round(max(per_task) / CHARS_PER_TOKEN),
                       "method": f"chars / {CHARS_PER_TOKEN}; replace with API usage counts"},
        "bddl_sha1": {t["key"]: t["bddl_sha1"] for t in tasks},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("num_tasks", "kept_scanned", "scan_seconds", "est_tokens")}))


if __name__ == "__main__":
    main()
