"""Check recognitional preference resolution and day-matched oracle rewards."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.recognitional import prepare_oracle_bundle, resolve_preferences
from rl_studies.config import load_study_inputs


def _inputs(mode="fixed"):
    return load_study_inputs(
        mode=mode, nexus="energy_mobility", justice="recognitional",
        preference_profile="declared_needs_v1",
    )


def test_preferences_require_one_supported_choice_per_scenario_label():
    inputs = _inputs()
    cfg = deepcopy(inputs.configuration["recognitional"])
    labels = inputs.configuration["scenario_labels"]
    assert len(resolve_preferences(cfg, labels)) == 5
    del cfg["preferences"][labels[0]]
    with pytest.raises(ValueError, match="cover each unique"):
        resolve_preferences(cfg, labels)
    cfg["preferences"][labels[0]] = {"priority_order": ["minimum_trip_shortfall", "unknown", "minimum_squared_power"]}
    with pytest.raises(ValueError, match="Invalid priority_order"):
        resolve_preferences(cfg, labels)


def test_custom_priority_order_without_bill_builds_oracles(tmp_path):
    inputs = _inputs()
    cfg = deepcopy(inputs.configuration["recognitional"])
    label = inputs.configuration["scenario_labels"][0]
    cfg["preferences"][label] = {
        "priority_order": ["minimum_trip_shortfall", "minimum_throughput", "minimum_squared_power"],
    }
    preferences = resolve_preferences(cfg, inputs.configuration["scenario_labels"])
    assert preferences[0]["need"] == "custom_order"
    bundle = prepare_oracle_bundle(inputs.source_params, preferences, cache_dir=tmp_path)
    assert bundle["replay"][0]["mismatch_kwh"] < 1e-3


def test_fixed_oracle_replay_and_applied_squared_gap_reward(tmp_path):
    inputs = _inputs()
    preferences = inputs.configuration["recognitional"]["resolved_preferences"]
    bundle = prepare_oracle_bundle(inputs.source_params, preferences, cache_dir=tmp_path)
    assert prepare_oracle_bundle(inputs.source_params, preferences, cache_dir=tmp_path) == bundle
    assert len(bundle["actions_kw"]) == 48
    assert all(entry["mismatch_kwh"] < 1e-3 for entry in bundle["replay"])
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice="recognitional",
        preference_profile="declared_needs_v1", configuration=inputs.configuration,
        source_params=inputs.source_params, oracle_cache_dir=tmp_path,
    )
    try:
        obs, _ = problem.env.reset(seed=0)
        assert obs.shape == (52,)
        action = np.zeros(5, dtype=np.float32)
        _, reward, _, _, info = problem.env.step(action)
        terms = info["reward_terms"]
        oracle = np.asarray(bundle["actions_kw"])[0]
        applied = np.asarray(info["recognitional_offline_oracle"]["applied_action_kw"])
        reference = np.asarray(bundle["idle_to_oracle_l2_kw2h"])
        expected_gap = float(np.sum(np.square(applied - oracle) * 0.5 / reference))
        assert terms["normalized_recognitional_action_gap"] == pytest.approx(expected_gap)
        assert terms["recognitional_reward"] == pytest.approx(-0.2 * expected_gap)
        assert reward == pytest.approx(
            terms["mobility_reward"] + terms["grid_reward"]
            + terms["aggregator_reward"] + terms["recognitional_reward"]
        )
    finally:
        problem.env.close()


def test_variable_days_select_matching_oracle_bundle(tmp_path):
    inputs = _inputs("variable")
    configuration = deepcopy(inputs.configuration)
    configuration["mode"]["days"]["counts"] = {"train": 2, "validation": 1, "test": 1}
    problem = make_rl_problem(
        mode="variable", nexus="energy_mobility", justice="recognitional",
        preference_profile="declared_needs_v1", configuration=configuration,
        source_params=inputs.source_params, oracle_cache_dir=tmp_path,
    )
    try:
        assert len(problem.provenance["oracle_bundles"]) == 2
        assert len(set(problem.provenance["oracle_bundles"].values())) == 2
        for index in range(2):
            obs, info = problem.env.reset(seed=5, options={"day_index": index})
            assert obs.shape == (52,)
            assert info["day_id"] == f"train_{index:03d}"
            _, _, _, _, step_info = problem.env.step(np.zeros(5, dtype=np.float32))
            bundle = problem.env.oracle_bundles[index]
            assert step_info["recognitional_offline_oracle"]["oracle_action_kw"] == pytest.approx(
                bundle["actions_kw"][0]
            )
            assert bundle["day_sha256"] == problem.provenance["day_records"][index]["sha256"]
    finally:
        problem.env.close()


def test_action_gap_weight_is_configurable(tmp_path):
    inputs = _inputs()
    configuration = deepcopy(inputs.configuration)
    configuration["recognitional"]["action_gap_weight"] = 0.4
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice="recognitional",
        preference_profile="declared_needs_v1", configuration=configuration,
        source_params=inputs.source_params, oracle_cache_dir=tmp_path,
    )
    try:
        assert problem.provenance["reward"]["recognitional"]["weight"] == 0.4
        problem.env.reset(seed=0)
        _, _, _, _, info = problem.env.step(np.zeros(5, dtype=np.float32))
        terms = info["reward_terms"]
        assert terms["recognitional_reward"] == pytest.approx(
            -0.4 * terms["normalized_recognitional_action_gap"]
        )
    finally:
        problem.env.close()
