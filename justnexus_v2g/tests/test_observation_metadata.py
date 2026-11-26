"""Flat provenance must label the exact emitted actor vector."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from justnexus_v2g.gym_env.observations import (
    _household_field_value,
    _system_field_value,
)
from justnexus_v2g.gym_env.recognitional import OfflineRecognitionalRewardWrapper
from justnexus_v2g.rl import make_rl_problem
import justnexus_v2g.rl.problem as problem_module
from rl_studies.config import load_study_inputs


CASES = (
    ("base", None, 52),
    ("distributive", "mean_deviation", 54),
    ("distributive", "max_min", 54),
    ("distributive", "generalized_gini", 54),
    ("procedural", None, 57),
    ("recognitional", None, 52),
    ("intertemporal", None, 62),
    ("distributive_recognitional", None, 56),
    ("epistemic", None, 52),
)


def _synthetic_oracle(day, preferences, *, cache_dir):
    """Supply deterministic day-matched values without solving or writing caches."""
    digest = hashlib.sha256(json.dumps(asdict(day), sort_keys=True).encode()).hexdigest()
    return {
        "actions_kw": np.ones((day.episode_steps - 1, day.n_households)).tolist(),
        "idle_to_oracle_l1_kwh": [24.0] * day.n_households,
        "day_sha256": digest,
        "sha256": digest,
    }


def _assert_labeled_values(observation, metadata, core, day):
    state = core.unwrapped.simulator.state
    native = core.unwrapped
    distribution = native._distributive_observation_context()
    intertemporal = native._intertemporal_observation_context(state)
    appended = {}
    if isinstance(core, OfflineRecognitionalRewardWrapper) and core.appended_observation_fields:
        _, _, worst, excess = core._gap_distribution()
        appended = dict(zip(core.appended_observation_fields, (worst, excess), strict=True))
    for entry in metadata["flat_fields"]:
        if entry["scope"] == "system":
            expected = _system_field_value(
                state, entry["field"], day, distribution, intertemporal,
            )
        elif entry["scope"] == "household":
            index = entry["household_index"]
            expected = _household_field_value(
                state.households[index], entry["field"], day.households[index],
                state, day, household_index=index, intertemporal_context=intertemporal,
            )
        else:
            expected = appended[entry["field"]]
        assert observation[entry["index"]] == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["fixed", "variable"])
@pytest.mark.parametrize("justice,distributive_type,size", CASES)
def test_versioned_flat_map_matches_reset_and_step(
    mode, justice, distributive_type, size, monkeypatch, tmp_path,
):
    monkeypatch.setattr(problem_module, "prepare_oracle_bundle", _synthetic_oracle)
    preference_profile = "declared_needs_v1" if justice == "recognitional" else None
    inputs = load_study_inputs(
        mode=mode, nexus="energy_mobility",
        justice=None if justice == "base" else justice,
        distributive_type=distributive_type,
        preference_profile=preference_profile,
    )
    config = deepcopy(inputs.configuration)
    if mode == "variable":
        config["mode"]["days"]["counts"] = {"train": 1, "validation": 1, "test": 1}
    problem = make_rl_problem(
        mode=mode, nexus="energy_mobility",
        justice=None if justice == "base" else justice,
        distributive_type=distributive_type,
        preference_profile=preference_profile,
        epistemic_priority=0.5 if justice == "epistemic" else None,
        split="test", configuration=config,
        source_params=inputs.source_params, oracle_cache_dir=tmp_path,
    )
    try:
        env = problem.env
        core = env.core if mode == "variable" else env
        day = env.days[0] if mode == "variable" else env.unwrapped.params
        metadata = problem.provenance["observation"]
        assert metadata["schema_version"] == 2
        assert metadata["size"] == size
        assert env.observation_space.shape == (size,)
        assert env.action_space.shape == (5,)
        assert metadata["system_fields"] == list(core.unwrapped.formulation.system_observation_fields)
        assert metadata["household_fields"] == list(core.unwrapped.formulation.household_observation_fields)
        assert [entry["index"] for entry in metadata["flat_fields"]] == list(range(size))
        assert metadata["flat_fields"][0] == {
            "index": 0, "scope": "system", "field": "time_sin",
        }
        labels = config["scenario_labels"]
        for index, label in enumerate(labels):
            entries = [entry for entry in metadata["flat_fields"]
                       if entry["scope"] == "household" and entry["household_index"] == index]
            assert [entry["field"] for entry in entries] == metadata["household_fields"]
            assert all(entry["household_label"] == label for entry in entries)
        if justice == "distributive_recognitional":
            assert metadata["system_fields"][-2:] == [
                "distribution_normalized_worst_mobility_burden",
                "distribution_normalized_excess_mobility_burden",
            ]
            assert metadata["flat_fields"][54:] == [
                {"index": 54, "scope": "appended",
                 "field": "normalized_cumulative_worst_recognitional_gap"},
                {"index": 55, "scope": "appended",
                 "field": "normalized_cumulative_excess_recognitional_gap"},
            ]
        else:
            assert metadata["appended_fields"] == []
        assert json.loads(json.dumps(metadata)) == metadata

        options = {"day_index": 0} if mode == "variable" else None
        reset_observation, _ = env.reset(seed=7, options=options)
        core = env.core if mode == "variable" else env
        assert reset_observation.dtype == np.float32
        _assert_labeled_values(reset_observation, metadata, core, day)
        action = np.asarray([7.0, 4.0, 3.0, 2.0, 5.0], dtype=np.float32)
        step_observation, _, _, _, info = env.step(action)
        assert step_observation.dtype == np.float32
        _assert_labeled_values(step_observation, metadata, core, day)
        requested = [entry for entry in metadata["flat_fields"]
                     if entry["field"] == "requested_ev_power_kw"]
        assert [step_observation[entry["index"]] for entry in requested] == pytest.approx(action)
        assert len({step_observation[entry["index"]] for entry in requested}) == 5
        if justice == "distributive_recognitional":
            gap = info["recognitional_offline_oracle"]["gap_distribution"]
            assert step_observation[54] == pytest.approx(gap["normalized_worst"])
            assert step_observation[55] == pytest.approx(gap["normalized_excess"])
            assert step_observation[54] != step_observation[55]
    finally:
        problem.env.close()


def _assert_missing_labels_are_metadata_only(labeled, unlabeled, day, *, variable):
    labeled_map = labeled.provenance["observation"]
    missing_map = unlabeled.provenance["observation"]
    expected = deepcopy(labeled_map)
    for entry in expected["flat_fields"]:
        if entry["scope"] == "household":
            entry["household_label"] = None
    assert missing_map == expected
    assert missing_map["schema_version"] == 2
    assert [entry["household_index"] for entry in missing_map["flat_fields"]
            if entry["scope"] == "household"] == [
        index for index in range(day.n_households)
        for _ in missing_map["household_fields"]
    ]
    assert json.loads(json.dumps(missing_map)) == missing_map

    options = {"day_index": 0} if variable else None
    labeled_reset, _ = labeled.env.reset(seed=7, options=options)
    unlabeled_reset, _ = unlabeled.env.reset(seed=7, options=options)
    np.testing.assert_array_equal(unlabeled_reset, labeled_reset)
    core = unlabeled.env.core if variable else unlabeled.env
    _assert_labeled_values(unlabeled_reset, missing_map, core, day)
    action = np.asarray([7.0, 4.0, 3.0, 2.0, 5.0], dtype=np.float32)
    labeled_step, labeled_reward, _, _, _ = labeled.env.step(action)
    unlabeled_step, unlabeled_reward, _, _, _ = unlabeled.env.step(action)
    np.testing.assert_array_equal(unlabeled_step, labeled_step)
    assert unlabeled_reward == labeled_reward
    _assert_labeled_values(unlabeled_step, missing_map, core, day)


@pytest.mark.parametrize("justice,distributive_type,size", [
    case for case in CASES
    if case[0] not in ("recognitional", "distributive_recognitional")
])
def test_direct_fixed_configuration_without_scenario_labels(
    justice, distributive_type, size,
):
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility",
        justice=None if justice == "base" else justice,
        distributive_type=distributive_type,
    )
    labeled_config = deepcopy(inputs.configuration)
    unlabeled_config = deepcopy(labeled_config)
    unlabeled_config.pop("scenario_labels")
    kwargs = {
        "mode": "fixed", "nexus": "energy_mobility",
        "justice": None if justice == "base" else justice,
        "distributive_type": distributive_type,
        "epistemic_priority": 0.5 if justice == "epistemic" else None,
        "source_params": inputs.source_params,
    }
    labeled = make_rl_problem(configuration=labeled_config, **kwargs)
    try:
        unlabeled = make_rl_problem(configuration=unlabeled_config, **kwargs)
        try:
            assert unlabeled.env.observation_space.shape == (size,)
            _assert_missing_labels_are_metadata_only(
                labeled, unlabeled, inputs.source_params, variable=False,
            )
        finally:
            unlabeled.env.close()
    finally:
        labeled.env.close()


def test_legacy_variable_generation_without_scenario_labels():
    inputs = load_study_inputs(mode="variable", nexus="energy_mobility")
    labeled_config = deepcopy(inputs.configuration)
    labeled_config.pop("day_variation")
    labeled_config["mode"]["days"] = {
        "counts": {"train": 1, "validation": 1, "test": 1},
        "generator_seed": 32020, "sampling_seed": 32020,
        "load_factor_range": [0.8, 1.2], "pv_factor_range": [0.75, 1.25],
        "departure_options": [[14, 15, 16], [17, 18, 19], [20],
                              [12, 13, 14], [16, 17, 18]],
    }
    unlabeled_config = deepcopy(labeled_config)
    unlabeled_config.pop("scenario_labels")
    kwargs = {
        "mode": "variable", "nexus": "energy_mobility",
        "source_params": inputs.source_params, "split": "test",
    }
    labeled = make_rl_problem(configuration=labeled_config, **kwargs)
    try:
        unlabeled = make_rl_problem(configuration=unlabeled_config, **kwargs)
        try:
            assert unlabeled.provenance["day_generation"]["version"] == "energy_mobility_days_v1"
            assert unlabeled.provenance["day_records"] == labeled.provenance["day_records"]
            assert unlabeled.env.days == labeled.env.days
            _assert_missing_labels_are_metadata_only(
                labeled, unlabeled, unlabeled.env.days[0], variable=True,
            )
        finally:
            unlabeled.env.close()
    finally:
        labeled.env.close()


@pytest.mark.parametrize("justice,preference_profile", [
    ("recognitional", "declared_needs_v1"),
    ("distributive_recognitional", None),
])
def test_oracle_formulations_still_require_scenario_labels(justice, preference_profile):
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice=justice,
        preference_profile=preference_profile,
    )
    config = deepcopy(inputs.configuration)
    config.pop("scenario_labels")
    with pytest.raises(KeyError, match="scenario_labels"):
        make_rl_problem(
            mode="fixed", nexus="energy_mobility", justice=justice,
            preference_profile=preference_profile, configuration=config,
            source_params=inputs.source_params,
        )


def test_named_variable_day_generation_still_requires_scenario_labels():
    inputs = load_study_inputs(mode="variable", nexus="energy_mobility")
    config = deepcopy(inputs.configuration)
    config.pop("scenario_labels")
    config["mode"]["days"]["counts"] = {"train": 1, "validation": 1, "test": 1}
    with pytest.raises(ValueError, match="requires one unique scenario label"):
        make_rl_problem(
            mode="variable", nexus="energy_mobility", configuration=config,
            source_params=inputs.source_params,
        )
