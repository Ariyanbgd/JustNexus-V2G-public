"""Check the dedicated mobility and oracle-gap disparity formulation."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from rl_studies.config import load_study_inputs


def _problem(mode="fixed", *, split="train", configuration=None, cache_dir=None):
    inputs = load_study_inputs(
        mode=mode, nexus="energy_mobility", justice="distributive_recognitional",
    )
    return make_rl_problem(
        mode=mode, nexus="energy_mobility", justice="distributive_recognitional",
        split=split, configuration=configuration or inputs.configuration,
        source_params=inputs.source_params, oracle_cache_dir=cache_dir,
    )


def test_fixed_combined_observation_and_reward_timing(tmp_path):
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice="distributive_recognitional",
    )
    config = deepcopy(inputs.configuration)
    config["combined"]["mobility_disparity_weights"] = {
        "worst_burden_weight": 0.5, "mean_excess_burden_weight": 2.0,
    }
    config["combined"]["action_gap_weight"] = 0.4
    config["combined"]["alignment_disparity_weights"] = {
        "worst_gap_weight": 0.25, "mean_excess_gap_weight": 1.5,
    }
    problem = _problem(configuration=config, cache_dir=tmp_path)
    try:
        obs, _ = problem.env.reset(seed=7)
        assert obs.shape == (56,)
        assert problem.provenance["justice"] == ["distributive", "recognitional"]
        assert problem.provenance["observation"]["size"] == 56
        assert problem.provenance["reward"]["combined"]["action_gap_weight"] == 0.4
        mobility_events = gap_events = 0
        for step in range(48):
            obs, reward, terminated, truncated, info = problem.env.step(
                np.ones(5, dtype=np.float32)
            )
            terms = info["reward_terms"]
            assert obs.shape == (56,)
            assert terms["total_reward"] == pytest.approx(reward)
            assert terms["recognitional_action_reward"] == pytest.approx(
                -0.4 * terms["normalized_recognitional_action_gap"]
            )
            assert reward == pytest.approx(
                terms["mobility_reward"] + terms["grid_reward"]
                + terms["aggregator_reward"] + terms["distributive_reward"]
                + terms["recognitional_action_reward"]
                + terms["recognitional_gap_distribution_reward"]
            )
            if terms["mobility_distribution_finalized"]:
                mobility_events += 1
                assert step == 19  # Simulator departure t=20 is the 20th transition.
                assert info["t"] == 20
                assert terms["worst_mobility_burden_reward"] == pytest.approx(
                    -0.5 * terms["normalized_worst_mobility_burden"]
                )
                assert terms["excess_mobility_burden_reward"] == pytest.approx(
                    -2.0 * terms["normalized_excess_mobility_burden"]
                )
            if terms["recognitional_gap_distribution_finalized"]:
                gap_events += 1
                assert step == 47
                assert terms["worst_recognitional_gap_reward"] == pytest.approx(
                    -0.25 * terms["normalized_worst_recognitional_gap"]
                )
                assert terms["excess_recognitional_gap_reward"] == pytest.approx(
                    -1.5 * terms["normalized_excess_recognitional_gap"]
                )
            if terminated or truncated:
                break
        assert (mobility_events, gap_events) == (1, 1)
    finally:
        problem.env.close()


def test_variable_combined_uses_day_matched_oracles(tmp_path):
    inputs = load_study_inputs(
        mode="variable", nexus="energy_mobility", justice="distributive_recognitional",
    )
    configuration = deepcopy(inputs.configuration)
    configuration["mode"]["days"]["counts"] = {"train": 2, "validation": 1, "test": 1}
    problem = _problem(mode="variable", configuration=configuration, cache_dir=tmp_path)
    try:
        assert len(problem.provenance["oracle_bundles"]) == 2
        for index in range(2):
            obs, info = problem.env.reset(seed=7, options={"day_index": index})
            assert obs.shape == (56,)
            assert info["day_id"] == f"train_{index:03d}"
            _, _, _, _, step_info = problem.env.step(np.zeros(5, dtype=np.float32))
            assert step_info["recognitional_offline_oracle"]["oracle_action_kw"] == pytest.approx(
                problem.env.oracle_bundles[index]["actions_kw"][0]
            )
    finally:
        problem.env.close()


def test_combined_rejects_invalid_weights(tmp_path):
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice="distributive_recognitional",
    )
    configuration = deepcopy(inputs.configuration)
    configuration["combined"]["alignment_disparity_weights"]["worst_gap_weight"] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _problem(configuration=configuration, cache_dir=tmp_path)
