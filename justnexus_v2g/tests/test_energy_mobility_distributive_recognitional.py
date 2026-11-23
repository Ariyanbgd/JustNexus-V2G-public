import numpy as np
import pytest

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.recognitional import OfflineRecognitionalRewardWrapper
from justnexus_v2g.gym_env.rewards import RewardNormalization, RewardWeights


def test_gap_distribution_observation_updates_and_reward_occurs_once():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    base = JustNexusV2GEnv(
        params,
        config=GymEnvConfig(nexus="energy_mobility", justice=("distributive",)),
        reward_weights=RewardWeights(
            unmet_departure_energy=1.0,
            grid_constraint=1.0,
            total_ev_clipping=1.0,
            worst_mobility_burden=1.0,
            excess_mobility_burden=1.0,
        ),
        reward_normalization=RewardNormalization(
            departure_energy_shortfall_kwh=74.0,
            import_overload_squared_kw2h=267.9392226234346,
            command_mismatch_kwh=160.0,
        ),
        grid_operation="soft_import_unlimited_export",
    )
    n_steps = params.episode_steps - 1
    oracle_actions = np.ones((n_steps, params.n_households), dtype=float)
    denominators = np.full(params.n_households, n_steps * params.dt_hours)
    env = OfflineRecognitionalRewardWrapper(
        base,
        oracle_actions_kw=oracle_actions,
        idle_to_oracle_gap_kwh=denominators,
        gap_distribution={
            "worst_reference": 0.8,
            "excess_reference": 0.24,
            "worst_weight": 1.0,
            "excess_weight": 1.0,
        },
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (56,)
    assert obs[-2:] == pytest.approx([0.0, 0.0])

    mobility_reward_steps = []
    gap_distribution_reward_steps = []
    for step in range(n_steps):
        action = np.zeros(params.n_households, dtype=np.float32)
        if env.simulator.state.households[0].ev_connected:
            action[0] = 3.0
        obs, _, terminated, truncated, info = env.step(action)
        terms = info["reward_terms"]
        assert obs.shape == (56,)
        assert obs[-2:] == pytest.approx([
            terms["normalized_worst_recognitional_gap"],
            terms["normalized_excess_recognitional_gap"],
        ])
        if terms["distributive_reward"]:
            mobility_reward_steps.append(step + 1)
        if terms["recognitional_gap_distribution_reward"]:
            gap_distribution_reward_steps.append(step + 1)
        if not (terminated or truncated):
            assert terms["recognitional_gap_distribution_reward"] == 0.0
    assert mobility_reward_steps == [20]
    assert gap_distribution_reward_steps == [n_steps]

    gaps = np.asarray(info["recognitional_offline_oracle"]["cumulative_action_gap_kwh"])
    normalized = gaps / denominators
    above_mean = np.maximum(normalized - np.mean(normalized), 0.0)
    expected = -float(np.max(above_mean) / 0.8 + np.mean(above_mean) / 0.24)
    assert terms["recognitional_gap_distribution_reward"] == pytest.approx(expected)
    assert terms["recognitional_gap_distribution_finalized"] == 1.0
    env.close()
