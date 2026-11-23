"""Verify the paper-aligned intertemporal study in both day modes."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from rl_studies.config import load_study_inputs


def _problem(mode="fixed", *, weights=None):
    inputs = load_study_inputs(mode=mode, nexus="energy_mobility", justice="intertemporal")
    configuration = deepcopy(inputs.configuration)
    if weights is not None:
        configuration["intertemporal"]["weights"] = weights
    return make_rl_problem(
        mode=mode, nexus="energy_mobility", justice="intertemporal",
        configuration=configuration, source_params=inputs.source_params,
    )


@pytest.mark.parametrize("mode", ["fixed", "variable"])
def test_intertemporal_shape_reward_timing_and_normalization(mode):
    problem = _problem(mode)
    env = problem.env
    try:
        observation, _ = env.reset(seed=7)
        assert observation.shape == (62,)
        assert problem.provenance["observation"]["size"] == 62
        assert problem.provenance["reward"]["normalization"]["excess_daily_battery_throughput_kwh"] == 28.5
        assert problem.provenance["reward"]["normalization"]["terminal_battery_reserve_deficit_kwh"] == 28.5
        terminal_events = 0
        for step in range(48):
            _, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
            terms = info["reward_terms"]
            expected = (
                -terms["departure_energy_shortfall_kwh"] / 74.0
                -terms["import_overload_squared_kw2h"] / 267.9392226234346
                -terms["command_mismatch_kwh"] / 160.0
                -terms["excess_daily_battery_throughput_kwh"] / 28.5
                -terms["terminal_battery_reserve_deficit_kwh"] / 28.5
            )
            assert reward == pytest.approx(expected)
            if terms["terminal_reserve_evaluated"]:
                terminal_events += 1
                assert step == 47
                assert terms["terminal_battery_reserve_deficit_kwh"] == pytest.approx(28.5)
            else:
                assert terms["terminal_battery_reserve_deficit_kwh"] == 0.0
            if terminated or truncated:
                break
        assert terminal_events == 1
    finally:
        env.close()


def test_two_weights_are_configurable_and_validated():
    weights = {"excess_daily_battery_throughput": 2.0, "battery_reserve_deficit": 0.25}
    problem = _problem(weights=weights)
    env = problem.env
    try:
        env.reset(seed=0)
        for _ in range(47):
            env.step(np.zeros(5, dtype=np.float32))
        _, reward, terminated, _, info = env.step(np.zeros(5, dtype=np.float32))
        assert terminated
        terms = info["reward_terms"]
        assert terms["terminal_battery_reserve_reward"] == pytest.approx(-0.25)
        assert reward == pytest.approx(
            terms["mobility_reward"] + terms["grid_reward"]
            + terms["aggregator_reward"] + terms["intertemporal_reward"]
        )
    finally:
        env.close()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _problem(weights={"excess_daily_battery_throughput": -1.0, "battery_reserve_deficit": 1.0})


def test_throughput_weight_scales_incremental_excess():
    problem = _problem(weights={"excess_daily_battery_throughput": 2.0, "battery_reserve_deficit": 0.0})
    env = problem.env
    try:
        env.reset(seed=0)
        env.step(np.zeros(5, dtype=np.float32))
        cap = env._daily_battery_throughput_limits_kwh()[0]
        env.daily_household_battery_throughput_kwh[0] = cap - 0.5
        _, _, _, _, info = env.step(np.asarray([7.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        terms = info["reward_terms"]
        assert terms["excess_daily_battery_throughput_kwh"] == pytest.approx(3.0)
        assert terms["excess_daily_battery_throughput_reward"] == pytest.approx(-2.0 * 3.0 / 28.5)
    finally:
        env.close()
