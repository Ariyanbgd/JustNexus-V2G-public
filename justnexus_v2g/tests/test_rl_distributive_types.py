"""Verify the three distributive terms and their original payment timing."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.distributive import distributive_term
from rl_studies.config import load_study_inputs

REQUIREMENTS = [20.0, 14.0, 12.0, 14.0, 14.0]


def _reference_welfare_terms(shortfalls, requirements):
    """Independent five-household reference for the original terminal terms."""
    burdens = [min(shortfall / need, 1.0) for shortfall, need in zip(shortfalls, requirements, strict=True)]
    mean_burden = sum(burdens) / 5
    excess = [max(burden - mean_burden, 0.0) for burden in burdens]
    satisfactions = sorted(1.0 - burden for burden in burdens)
    base = -sum(shortfalls) / sum(requirements)
    return {
        "base": base,
        "mean_deviation": base - max(excess) / 0.8 - (sum(excess) / 5) / 0.24,
        "max_min": -max(burdens),
        "generalized_gini": sum(weight * value for weight, value in zip(
            (0.40, 0.25, 0.17, 0.11, 0.07), satisfactions, strict=True
        )) - 1.0,
    }


@pytest.mark.parametrize(("operator", "expected"), [
    ("mean_deviation", -1.75),
    ("max_min", -1.0),
    ("generalized_gini", -0.65),
])
def test_terms_match_known_two_unserved_households(operator, expected):
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice="distributive",
        distributive_type=operator,
    )
    value, _ = distributive_term(
        [20.0, 14.0, 0.0, 0.0, 0.0], REQUIREMENTS,
        {"operator": operator, **inputs.configuration["distributive"]},
    )
    assert value == pytest.approx(expected)


def test_gini_rank_weights_are_a_real_configurable_choice():
    burdens = [20.0, 14.0, 0.0, 0.0, 0.0]
    original, _ = distributive_term(burdens, REQUIREMENTS, {
        "operator": "generalized_gini",
        "rank_weights_low_to_high": [0.40, 0.25, 0.17, 0.11, 0.07],
    })
    changed, _ = distributive_term(burdens, REQUIREMENTS, {
        "operator": "generalized_gini",
        "rank_weights_low_to_high": [1.0, 0.0, 0.0, 0.0, 0.0],
    })
    assert original == pytest.approx(-0.65)
    assert changed == pytest.approx(-1.0)


@pytest.mark.parametrize(("mode", "operator"), [
    ("fixed", "mean_deviation"),
    ("fixed", "max_min"),
    ("fixed", "generalized_gini"),
    ("variable", "max_min"),
])
def test_original_observation_and_single_last_departure_term(mode, operator):
    inputs = load_study_inputs(
        mode=mode, nexus="energy_mobility", justice="distributive",
        distributive_type=operator,
    )
    problem = make_rl_problem(
        mode=mode, nexus="energy_mobility", justice="distributive",
        distributive_type=operator, configuration=inputs.configuration,
        source_params=inputs.source_params,
    )
    env = problem.env
    try:
        observation, _ = env.reset(seed=7)
        assert observation.shape == (54,)
        assert problem.provenance["observation"]["size"] == 54
        events = []
        for _ in range(48):
            observation, reward, terminated, truncated, info = env.step(
                np.ones(5, dtype=np.float32)
            )
            assert observation.shape == (54,)
            terms = info["reward_terms"]
            base = (
                -terms["departure_energy_shortfall_kwh"] / 74.0
                -terms["import_overload_squared_kw2h"] / 267.9392226234346
                -terms["command_mismatch_kwh"] / 160.0
            )
            event = bool(terms["mobility_distribution_finalized"])
            if event:
                events.append(info["t"])
                welfare = _reference_welfare_terms(info["household_departure_shortfall_kwh"], REQUIREMENTS)
                expected = (
                    welfare["mean_deviation"] - welfare["base"]
                    if operator == "mean_deviation" else welfare[operator]
                )
                assert reward == pytest.approx(base + expected)
                assert terms["distributive_reward"] == pytest.approx(expected)
            else:
                assert reward == pytest.approx(base)
                assert terms["distributive_reward"] == 0.0
            if terminated or truncated:
                break
        assert events == [20]
    finally:
        env.close()


@pytest.mark.parametrize("operator", ["mean_deviation", "max_min", "generalized_gini"])
def test_type_weights_scale_the_terminal_reward(operator):
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice="distributive",
        distributive_type=operator,
    )
    configuration = deepcopy(inputs.configuration)
    settings = configuration["distributive"]
    if operator == "mean_deviation":
        settings.update(worst_burden_weight=0.5, mean_excess_burden_weight=2.0)
    else:
        settings["weight"] = 0.5
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice="distributive",
        distributive_type=operator, configuration=configuration,
        source_params=inputs.source_params,
    )
    try:
        assert all(problem.provenance["reward"]["distributive"][key] == value
                   for key, value in settings.items())
        problem.env.reset(seed=7)
        for _ in range(48):
            _, _, terminated, truncated, info = problem.env.step(
                np.ones(5, dtype=np.float32)
            )
            if info["reward_terms"]["mobility_distribution_finalized"]:
                break
        terms = info["reward_terms"]
        if operator == "mean_deviation":
            details = info["distributive_operator_details"]
            expected = (-0.5 * details["normalized_worst_excess"]
                        -2.0 * details["normalized_mean_excess"])
            assert terms["worst_mobility_burden_reward"] == pytest.approx(
                -0.5 * details["normalized_worst_excess"]
            )
            assert terms["excess_mobility_burden_reward"] == pytest.approx(
                -2.0 * details["normalized_mean_excess"]
            )
        else:
            expected = 0.5 * terms["distributive_operator_value"]
        assert terms["distributive_reward"] == pytest.approx(expected)
    finally:
        problem.env.close()
