import numpy as np
import pytest

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.observations import observation_size
from justnexus_v2g.gym_env.rewards import RewardNormalization, RewardWeights


def make_mobility_env():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    config = GymEnvConfig(nexus="energy_mobility")
    env = JustNexusV2GEnv(
        params,
        config=config,
        reward_weights=RewardWeights(
            unmet_departure_energy=1.0,
            grid_constraint=1.0,
            total_ev_clipping=1.0,
        ),
        reward_normalization=RewardNormalization(
            departure_energy_shortfall_kwh=74.0,
            import_overload_squared_kw2h=267.9392226234346,
            command_mismatch_kwh=40.0,
        ),
        grid_operation="soft_import_unlimited_export",
    )
    return params, config, env


def test_energy_mobility_base_has_isolated_observation_contract():
    params, config, env = make_mobility_env()
    observation, _ = env.reset(seed=0)
    assert observation_size(params, config=config) == 52
    assert observation.shape == (52,)
    assert env.formulation.reward_strategy == "energy_mobility"
    env.close()


def test_no_action_mobility_and_grid_reward_terms_match_scenario_baseline():
    params, _, env = make_mobility_env()
    env.reset(seed=0)
    shortfall = squared_overload = mismatch = 0.0
    while not env.simulator.is_done:
        _, _, terminated, truncated, info = env.step(
            np.zeros(params.n_households, dtype=np.float32)
        )
        terms = info["reward_terms"]
        shortfall += terms["departure_energy_shortfall_kwh"]
        squared_overload += terms["import_overload_squared_kw2h"]
        mismatch += terms["command_mismatch_kwh"]
        assert not truncated
        if terminated:
            break
    assert shortfall == pytest.approx(74.0)
    assert squared_overload == pytest.approx(13.055)
    assert mismatch == pytest.approx(0.0)
    env.close()
