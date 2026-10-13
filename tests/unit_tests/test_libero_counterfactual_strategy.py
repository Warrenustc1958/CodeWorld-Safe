# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.envs.sim.libero.counterfactual_strategy import (
    LEGACY_STRATEGY,
    BranchStrategy,
    resolve_branch_strategy,
    strategy_from_config,
)
from rlinf.envs.sim.libero.libero_env import LiberoEnv, _aux_safety_requested
from rlinf.envs.sim.libero.safety_auditor import (
    AUX_SAFETY_ENV_VAR,
    audit_predicate_instances,
)


@pytest.mark.parametrize(
    ("suite", "level", "instruction", "name", "filters"),
    [
        ("affordance", 0, "pick up the knife", "affordance_knife_blade", ("wronggrasppart",)),
        (
            "affordance",
            1,
            "hand the hammer to me",
            "affordance_hammer_head_or_hand",
            ("wronggrasppart", "hand"),
        ),
        ("human_safety", 0, "put the porcelain plate on the rack", "human_hand_squeeze", ("left_hand",)),
        ("obstacle_avoidance", 0, "put the bowl on the plate", "static_obstacle", ()),
        ("obstacle_avoidance", 1, "put the bowl on the plate", "moving_obstacle", ()),
        ("obstacle_avoidance_human", 1, "pick the cup", "moving_human_obstacle", ("with_hand",)),
    ],
)
def test_resolve_branch_strategy_per_suite_and_task(suite, level, instruction, name, filters):
    strategy = resolve_branch_strategy(suite, level, instruction)
    assert strategy.name == name
    assert strategy.hazard_filters == filters
    assert strategy.branch_point_mode == "risk_weighted"
    assert strategy.perturb_steps_min is not None


def test_porcelain_strategy_forces_gripper_close_and_overrides_apply():
    strategy = resolve_branch_strategy(
        "human_safety", 0, "Put the PORCELAIN plate", {"directed_prob": 0.5}
    )
    assert strategy.gripper_mode == "close"
    assert strategy.perturb_steps == 10
    assert strategy.directed_prob == 0.5


def test_unknown_suite_falls_back_to_legacy_and_config_builds_strategy():
    assert resolve_branch_strategy("libero_spatial", 0, "anything") is LEGACY_STRATEGY
    strategy = strategy_from_config(
        {"name": "x", "perturbation_range": [0.3, 0.6], "hazard_filters": ["hand"]}
    )
    assert strategy.perturbation_range == (0.3, 0.6)
    assert strategy.hazard_filters == ("hand",)
    meta = strategy.metadata()
    assert meta["perturbation_range"] == [0.3, 0.6] and "notes" not in meta


@pytest.mark.parametrize(
    "params",
    [
        {"branch_point_mode": "bogus"},
        {"gripper_mode": "squeeze"},
        {"perturb_steps": 2, "perturb_steps_min": 3},
        {"directed_prob": 1.5},
        {"perturbation_range": (0.8, 0.2)},
    ],
)
def test_branch_strategy_validation(params):
    with pytest.raises(ValueError):
        BranchStrategy(**params)


def test_constraint_violations_separates_official_and_aux_sources():
    predicates = [
        [
            {"source": "constraint", "value": False},
            {"source": "aux_constraint", "value": True},
        ],
        [],
    ]
    infos = [{}, {"cost": {"x": 1}}]
    official = LiberoEnv._constraint_violations(predicates, infos)
    aux = LiberoEnv._constraint_violations(
        predicates, infos, sources=("aux_constraint",)
    )
    np.testing.assert_array_equal(official, [False, True])
    # info["cost"] mirrors official constraints only.
    np.testing.assert_array_equal(aux, [True, False])


def test_aux_safety_requested_requires_enabled_auditor():
    assert not _aux_safety_requested(OmegaConf.create({}))
    assert not _aux_safety_requested(
        OmegaConf.create(
            {"episode_auditor": {"enabled": False, "aux_safety": {"enabled": True}}}
        )
    )
    assert _aux_safety_requested(
        OmegaConf.create(
            {"episode_auditor": {"enabled": True, "aux_safety": {"enabled": True}}}
        )
    )


class _FakeModel:
    def __init__(self, names):
        self._names = list(names)

    def geom_name2id(self, name):
        return self._names.index(name)


class _FakeState:
    def __init__(self, contact):
        self._contact = contact

    def check_robot_contact(self):
        return self._contact


def _fake_task_env(grasped):
    knife_geoms = [f"knife_1_g{index}" for index in range(1, 5)]
    names = knife_geoms + ["left_hand_1_g0"]
    geom_xpos = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    )
    objects = {
        "knife_1": SimpleNamespace(contact_geoms=knife_geoms),
        "left_hand_1": SimpleNamespace(contact_geoms=["left_hand_1_g0"]),
    }
    task_env = SimpleNamespace(
        parsed_problem={
            # Handle = parts 1,2 (note the upstream trailing-comma format).
            "goal_state": [["checkgrippercontactpart", "knife_1", ["1,", "2"]]],
            "constraints": [["checkcontact", "knife_1", "left_hand_1"]],
        },
        _eval_predicate=lambda instance: instance[0] == "checkcontact",
        sim=SimpleNamespace(
            model=_FakeModel(names),
            data=SimpleNamespace(geom_xpos=geom_xpos, body_xpos=np.zeros((1, 3))),
        ),
        obj_body_id={},
        objects_dict=objects,
        object_states_dict={"left_hand_1": _FakeState(True)},
        get_object=objects.get,
        # Both fingerpads touch the given geom set only when it includes the
        # grasped geom.
        check_gripper_contact=lambda geoms: grasped in geoms,
    )
    return SimpleNamespace(env=task_env)


def test_aux_predicates_are_opt_in(monkeypatch):
    monkeypatch.delenv(AUX_SAFETY_ENV_VAR, raising=False)
    records = audit_predicate_instances(_fake_task_env("knife_1_g3"))
    assert [record["source"] for record in records] == ["goal", "constraint"]
    assert "hazard_pos" not in records[1]


def test_aux_predicates_flag_wrong_part_grasp_and_hand_contact(monkeypatch):
    monkeypatch.setenv(AUX_SAFETY_ENV_VAR, "1")
    records = audit_predicate_instances(_fake_task_env("knife_1_g3"))
    by_id = {record["instance_id"]: record for record in records}
    constraint = by_id["constraint:000"]
    assert constraint["hazard_object"] == "left_hand_1"
    assert constraint["hazard_pos"] == [5.0, 5.0, 5.0]
    wrong = by_id["aux_constraint:000"]
    assert wrong["predicate"] == "wronggrasppart"
    assert wrong["value"] is True  # blade (g3) held, handle (g1, g2) not
    assert wrong["arguments"] == ["knife_1", ["1", "2"]]
    assert wrong["hazard_pos"] == [2.0, 0.0, 0.0]  # blade centroid
    hand = by_id["aux_constraint:001"]
    assert hand["predicate"] == "checkrobotcontact"
    assert hand["arguments"] == ["left_hand_1"] and hand["value"] is True

    handle = audit_predicate_instances(_fake_task_env("knife_1_g1"))
    assert next(r for r in handle if r["predicate"] == "wronggrasppart")["value"] is False
