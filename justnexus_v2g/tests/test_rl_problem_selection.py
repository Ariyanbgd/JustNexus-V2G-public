"""Checks for the public fixed/variable base problem selection."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from rl_studies.config import load_study_inputs


def _problem(mode: str, split: str = "train"):
    inputs = load_study_inputs(mode=mode, nexus="energy_mobility")
    return make_rl_problem(
        mode=mode, nexus="energy_mobility", justice=None, split=split,
        configuration=inputs.configuration, source_params=inputs.source_params,
    )


@pytest.mark.parametrize("mode", ["fixed", "variable"])
def test_day_mode_preserves_base_observation_and_reward(mode):
    problem = _problem(mode)
    env = problem.env
    try:
        observation, _ = env.reset(seed=7)
        assert observation.shape == (52,)
        assert env.action_space.shape == (5,)
        assert problem.provenance["reward"]["weights"] == {
            "unmet_departure_energy": 1.0,
            "grid_constraint": 1.0,
            "total_ev_clipping": 1.0,
        }
        core = env.core if mode == "variable" else env
        assert set(vars(core.reward_weights)) == {
            "unmet_departure_energy", "grid_constraint", "total_ev_clipping",
            "worst_mobility_burden", "excess_mobility_burden",
            "procedural_consistency",
            "unauthorized_discharge_incidence", "excess_daily_battery_throughput",
            "battery_reserve_deficit",
        }
        assert problem.provenance["reward"]["shortfall_timing"] == "each_household_departure"
        steps = 0
        shortfall = 0.0
        while True:
            _, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
            terms = info["reward_terms"]
            assert reward == pytest.approx(
                -terms["departure_energy_shortfall_kwh"] / 74
                - terms["import_overload_squared_kw2h"] / 267.9392226234346
                - terms["command_mismatch_kwh"] / 160
            )
            shortfall += terms["departure_energy_shortfall_kwh"]
            steps += 1
            if terminated or truncated:
                break
        assert steps == 48
        assert shortfall == pytest.approx(74.0)
    finally:
        env.close()


def test_variable_days_use_separate_reproducible_splits():
    train = _problem("variable", "train")
    validation = _problem("variable", "validation")
    repeat = _problem("variable", "train")
    try:
        train_records = train.provenance["day_records"]
        validation_records = validation.provenance["day_records"]
        assert len(train_records) == 64
        assert len(validation_records) == 16
        assert {r["sha256"] for r in train_records}.isdisjoint(
            {r["sha256"] for r in validation_records}
        )
        assert train_records == repeat.provenance["day_records"]
        obs1, info1 = train.env.reset(seed=11)
        obs2, info2 = repeat.env.reset(seed=11)
        assert info1["day_id"] == info2["day_id"]
        np.testing.assert_array_equal(obs1, obs2)
    finally:
        train.env.close()
        validation.env.close()
        repeat.env.close()


def test_base_reward_weights_are_configurable():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility")
    configuration = deepcopy(inputs.configuration)
    weights = {
        "unmet_departure_energy": 0.5,
        "grid_constraint": 2.0,
        "total_ev_clipping": 3.0,
    }
    configuration["base"]["reward_weights"] = weights
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice=None,
        configuration=configuration, source_params=inputs.source_params,
    )
    try:
        assert problem.provenance["reward"]["weights"] == weights
        problem.env.reset(seed=7)
        for _ in range(48):
            _, reward, terminated, truncated, info = problem.env.step(
                np.ones(5, dtype=np.float32)
            )
            terms = info["reward_terms"]
            assert reward == pytest.approx(
                -0.5 * terms["departure_energy_shortfall_kwh"] / 74
                -2.0 * terms["import_overload_squared_kw2h"] / 267.9392226234346
                -3.0 * terms["command_mismatch_kwh"] / 160
            )
            if terminated or truncated:
                break
    finally:
        problem.env.close()
    configuration["base"]["reward_weights"]["grid_constraint"] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_rl_problem(
            mode="fixed", nexus="energy_mobility", justice=None,
            configuration=configuration, source_params=inputs.source_params,
        )
