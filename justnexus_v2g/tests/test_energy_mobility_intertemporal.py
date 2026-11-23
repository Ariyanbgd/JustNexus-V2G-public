import numpy as np
import pytest

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.observations import observation_size
from justnexus_v2g.gym_env.rewards import RewardNormalization, RewardWeights


THROUGHPUT_REFERENCE_KWH = 28.5
TERMINAL_RESERVE_REFERENCE_KWH = 28.5
EXPECTED_CAPS = [
    (20.0 + 5.0) / 0.95 + 5.0,
    (14.0 + 6.0) / 0.95 + 6.0,
    (12.0 + 8.0) / 0.95 + 8.0,
    (14.0 + 4.0) / 0.92 + 4.0,
    (14.0 + 5.5) / 0.95 + 5.5,
]


def make_env():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    config = GymEnvConfig(nexus="energy_mobility", justice=("intertemporal",))
    env = JustNexusV2GEnv(
        params,
        config=config,
        reward_weights=RewardWeights(
            unmet_departure_energy=1.0,
            grid_constraint=1.0,
            total_ev_clipping=1.0,
            excess_daily_battery_throughput=1.0,
            battery_reserve_deficit=1.0,
        ),
        reward_normalization=RewardNormalization(
            departure_energy_shortfall_kwh=74.0,
            import_overload_squared_kw2h=267.9392226234346,
            command_mismatch_kwh=160.0,
            excess_daily_battery_throughput_kwh=THROUGHPUT_REFERENCE_KWH,
            terminal_battery_reserve_deficit_kwh=TERMINAL_RESERVE_REFERENCE_KWH,
        ),
        grid_operation="soft_import_unlimited_export",
    )
    return params, config, env


def test_observation_adds_household_throughput_and_reserve_fields():
    params, config, env = make_env()
    observation, info = env.reset(seed=0)
    assert observation_size(params, config=config) == 62
    assert observation.shape == (62,)
    assert observation[[16, 27, 38, 49, 60]].tolist() == pytest.approx([1.0] * 5)
    assert observation[[17, 28, 39, 50, 61]].tolist() == pytest.approx([-0.1] * 5)
    assert info["daily_battery_throughput_limit_kwh_by_household"] == pytest.approx(EXPECTED_CAPS)
    env.close()


def test_incremental_excess_uses_57_kwh_reference():
    _, _, env = make_env()
    env.reset(seed=0)
    env.step(np.zeros(5, dtype=np.float32))
    env.daily_household_battery_throughput_kwh[0] = EXPECTED_CAPS[0] - 0.5
    _, _, _, _, info = env.step(np.asarray([7.0, 0.0, 0.0, 0.0, 0.0]))
    terms = info["reward_terms"]
    assert terms["excess_daily_battery_throughput_kwh"] == pytest.approx(3.0)
    assert terms["normalized_excess_daily_battery_throughput"] == pytest.approx(3.0 / 28.5)
    assert terms["excess_daily_battery_throughput_reward"] == pytest.approx(-3.0 / 28.5)
    env.close()


def test_reserve_deficit_is_normalized_once_at_terminal_state():
    params, _, env = make_env()
    env.reset(seed=0)
    reserve_rewards = []
    final_terms = None
    while not env.simulator.is_done:
        _, _, terminated, _, info = env.step(np.zeros(params.n_households, dtype=np.float32))
        terms = info["reward_terms"]
        reserve_rewards.append(terms["terminal_battery_reserve_reward"])
        if terminated:
            final_terms = terms
    assert reserve_rewards[:-1] == pytest.approx([0.0] * (len(reserve_rewards) - 1))
    assert final_terms["terminal_reserve_evaluated"] == 1.0
    assert final_terms["terminal_battery_reserve_target_kwh_by_household"] == pytest.approx([5.0, 6.0, 8.0, 4.0, 5.5])
    assert final_terms["terminal_battery_reserve_deficit_kwh"] == pytest.approx(28.5)
    assert final_terms["normalized_terminal_battery_reserve_deficit"] == pytest.approx(1.0)
    assert final_terms["terminal_battery_reserve_reward"] == pytest.approx(-1.0)
    env.close()
