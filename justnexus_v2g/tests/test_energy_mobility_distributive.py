import numpy as np
import pytest

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.observations import observation_size
from justnexus_v2g.gym_env.rewards import RewardNormalization, RewardWeights


def make_env():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    config = GymEnvConfig(nexus="energy_mobility", justice=("distributive",))
    env = JustNexusV2GEnv(
        params, config=config,
        reward_weights=RewardWeights(
            unmet_departure_energy=1.0,
            grid_constraint=1.0, total_ev_clipping=1.0,
            worst_mobility_burden=1.0, excess_mobility_burden=1.0,
        ),
        reward_normalization=RewardNormalization(
            departure_energy_shortfall_kwh=74.0,
            import_overload_squared_kw2h=267.9392226234346,
            command_mismatch_kwh=160.0,
        ),
        grid_operation="soft_import_unlimited_export",
    )
    return params, config, env


def test_observation_has_two_normalized_distribution_fields():
    params, config, env = make_env()
    observation, _ = env.reset(seed=0)
    assert observation_size(params, config=config) == 54
    assert observation.shape == (54,)
    assert env.formulation.reward_strategy == "energy_mobility_distributive"
    assert observation[7] == pytest.approx(0.0)
    assert observation[8] == pytest.approx(0.0)
    observation, *_ = env.step(np.asarray([7.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert observation[7] > 0.0
    assert observation[8] > 0.0
    env.close()


def test_final_distribution_reward_occurs_once_and_uses_shortfall_fractions():
    params, _, env = make_env()
    env.reset(seed=0)
    finalized = []
    distributive_reward = 0.0
    final_terms = None
    while not env.simulator.is_done:
        action = np.zeros(params.n_households, dtype=np.float32)
        if env.simulator.state.households[0].ev_connected:
            action[0] = 7.0
        _, _, terminated, _, info = env.step(action)
        terms = info["reward_terms"]
        distributive_reward += terms["distributive_reward"]
        if terms["mobility_distribution_finalized"]:
            finalized.append(info["t"])
            final_terms = terms
        if terminated:
            break
    assert finalized == [20]
    burdens = np.asarray(final_terms["household_mobility_burden_fraction"])
    mean = burdens.mean()
    above = np.maximum(burdens - mean, 0.0)
    expected = -(above.max() / 0.8) - (above.mean() / 0.24)
    assert distributive_reward == pytest.approx(expected)
    assert final_terms["worst_mobility_burden_reward"] == pytest.approx(-above.max() / 0.8)
    assert final_terms["excess_mobility_burden_reward"] == pytest.approx(-above.mean() / 0.24)
    env.close()
