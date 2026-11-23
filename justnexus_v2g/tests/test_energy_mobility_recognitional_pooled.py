import numpy as np
import pytest

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.recognitional import (
    OfflineBatteryTrajectoryGapWrapper,
    OfflineRecognitionalRewardWrapper,
)
from justnexus_v2g.gym_env.rewards import RewardWeights


def test_pooled_action_gap_uses_sum_of_physical_gaps_over_one_reference():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    oracle_power = np.arange(1, params.n_households + 1, dtype=float)
    oracle_actions = np.tile(oracle_power, (params.episode_steps - 1, 1))
    individual_idle_distances = oracle_power * (params.episode_steps - 1) * params.dt_hours
    reference = float(np.sum(individual_idle_distances))

    def wrapped(aggregation: str):
        base = JustNexusV2GEnv(
            params,
            config=GymEnvConfig(nexus="energy_mobility"),
            reward_weights=RewardWeights(),
            grid_operation="soft_import_unlimited_export",
        )
        return OfflineRecognitionalRewardWrapper(
            base,
            oracle_actions_kw=oracle_actions,
            idle_to_oracle_gap_kwh=individual_idle_distances,
            action_gap_aggregation=aggregation,
            action_gap_reference_kwh=reference if aggregation == "pooled_sum" else None,
        )

    pooled = wrapped("pooled_sum")
    mean = wrapped("equal_mean")
    try:
        for env in (pooled, mean):
            obs, _ = env.reset(seed=0)
            assert obs.shape == (52,)
        action = np.zeros(params.n_households, dtype=np.float32)
        action[0] = 1.0
        _, _, _, _, pooled_info = pooled.step(action)
        _, _, _, _, mean_info = mean.step(action)
        applied = np.asarray(pooled_info["recognitional_offline_oracle"]["applied_action_kw"])
        expected_raw = float(np.abs(applied - oracle_power).sum() * params.dt_hours)
        expected_pooled = expected_raw / reference
        pooled_terms = pooled_info["reward_terms"]
        mean_terms = mean_info["reward_terms"]
        assert pooled_terms["recognitional_action_gap_kwh"] == pytest.approx(expected_raw)
        assert pooled_terms["normalized_recognitional_action_gap"] == pytest.approx(expected_pooled)
        assert pooled_terms["recognitional_reward"] == pytest.approx(-expected_pooled)
        assert mean_terms["normalized_recognitional_action_gap"] != pytest.approx(expected_pooled)
    finally:
        pooled.close()
        mean.close()


def test_pooled_squared_action_gap_uses_oracle_energy_as_reference():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    oracle_power = np.arange(1, params.n_households + 1, dtype=float)
    oracle_actions = np.tile(oracle_power, (params.episode_steps - 1, 1))
    individual_idle_distances = oracle_power * (params.episode_steps - 1) * params.dt_hours
    reference = float(np.square(oracle_actions).sum() * params.dt_hours)
    base = JustNexusV2GEnv(
        params,
        config=GymEnvConfig(nexus="energy_mobility"),
        reward_weights=RewardWeights(),
        grid_operation="soft_import_unlimited_export",
    )
    env = OfflineRecognitionalRewardWrapper(
        base,
        oracle_actions_kw=oracle_actions,
        idle_to_oracle_gap_kwh=individual_idle_distances,
        action_gap_aggregation="pooled_squared",
        action_gap_reference_kw2h=reference,
    )
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(params.n_households, dtype=np.float32))
        applied = np.asarray(info["recognitional_offline_oracle"]["applied_action_kw"])
        expected = float(np.square(applied - oracle_power).sum() * params.dt_hours)
        terms = info["reward_terms"]
        assert terms["recognitional_squared_action_gap_kw2h"] == pytest.approx(expected)
        assert terms["normalized_recognitional_action_gap"] == pytest.approx(expected / reference)
        assert terms["recognitional_reward"] == pytest.approx(-expected / reference)
        oracle_info = info["recognitional_offline_oracle"]
        expected_household_scores = 1.0 - np.square(oracle_power) * params.dt_hours / (
            np.square(oracle_actions).sum(axis=0) * params.dt_hours
        )
        assert oracle_info["household_score"] == pytest.approx(expected_household_scores)
        assert oracle_info["pooled_normalized_squared_action_gap"] == pytest.approx(
            expected / reference
        )
        assert "score" not in oracle_info
        total = terms["normalized_recognitional_action_gap"]
        for _ in range(params.episode_steps - 2):
            _, _, _, _, info = env.step(np.zeros(params.n_households, dtype=np.float32))
            total += info["reward_terms"]["normalized_recognitional_action_gap"]
        assert total == pytest.approx(1.0)
        assert info["recognitional_offline_oracle"]["household_score"] == pytest.approx(
            np.zeros(params.n_households)
        )
        assert info["recognitional_offline_oracle"]["pooled_normalized_squared_action_gap"] == pytest.approx(1.0)
    finally:
        env.close()


def test_pooled_battery_gap_matches_no_action_and_exact_oracle_trajectory():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    action = np.full(params.n_households, 0.5, dtype=np.float32)

    def base_env():
        return JustNexusV2GEnv(
            params,
            config=GymEnvConfig(nexus="energy_mobility"),
            reward_weights=RewardWeights(),
            grid_operation="soft_import_unlimited_export",
        )

    trajectories = []
    for control in (np.zeros_like(action), action):
        env = base_env()
        try:
            env.reset(seed=0)
            energy = []
            for _ in range(params.episode_steps - 1):
                env.step(control)
                energy.append([h.ev_energy_kwh for h in env.last_step_result.state.households])
            trajectories.append(np.asarray(energy))
        finally:
            env.close()
    idle, oracle = trajectories
    references = np.square(idle - oracle).sum(axis=0) * params.dt_hours
    assert np.all(references > 0)

    env = OfflineBatteryTrajectoryGapWrapper(
        base_env(),
        oracle_energy_kwh=oracle,
        household_reference_kwh2h=references,
        reference_kwh2h=float(references.sum()),
    )
    try:
        assert env.reset(seed=0)[0].shape == (52,)
        total = 0.0
        for _ in range(params.episode_steps - 1):
            _, _, _, _, info = env.step(np.zeros_like(action))
            total += info["reward_terms"]["normalized_recognitional_battery_gap"]
        assert total == pytest.approx(1.0)
        assert info["recognitional_offline_oracle"]["pooled_normalized_battery_gap"] == pytest.approx(1.0)
        assert info["recognitional_offline_oracle"]["household_score"] == pytest.approx(
            np.zeros(params.n_households)
        )
        env.reset(seed=0)
        for _ in range(params.episode_steps - 1):
            _, _, _, _, info = env.step(action)
            assert info["reward_terms"]["normalized_recognitional_battery_gap"] == pytest.approx(0.0)
        assert info["recognitional_offline_oracle"]["pooled_normalized_battery_gap"] == pytest.approx(0.0)
        assert info["recognitional_offline_oracle"]["household_score"] == pytest.approx(
            np.ones(params.n_households)
        )
    finally:
        env.close()


def test_stacked_action_and_battery_gaps_keep_both_reward_terms():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    shape = (params.episode_steps - 1, params.n_households)
    oracle_actions = np.ones(shape)
    oracle_energy = np.ones(shape)
    dt = params.dt_hours
    action_household_references = np.square(oracle_actions).sum(axis=0) * dt
    battery_household_references = np.square(oracle_energy).sum(axis=0) * dt
    base = JustNexusV2GEnv(
        params,
        config=GymEnvConfig(nexus="energy_mobility"),
        reward_weights=RewardWeights(),
        grid_operation="soft_import_unlimited_export",
    )
    battery = OfflineBatteryTrajectoryGapWrapper(
        base,
        oracle_energy_kwh=oracle_energy,
        household_reference_kwh2h=battery_household_references,
        reference_kwh2h=float(battery_household_references.sum()),
    )
    env = OfflineRecognitionalRewardWrapper(
        battery,
        oracle_actions_kw=oracle_actions,
        idle_to_oracle_gap_kwh=np.abs(oracle_actions).sum(axis=0) * dt,
        action_gap_aggregation="pooled_squared",
        action_gap_reference_kw2h=float(action_household_references.sum()),
    )
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(params.n_households, dtype=np.float32))
        terms = info["reward_terms"]
        assert terms["recognitional_action_reward"] == pytest.approx(
            -terms["normalized_recognitional_action_gap"]
        )
        assert terms["recognitional_battery_reward"] == pytest.approx(
            -terms["normalized_recognitional_battery_gap"]
        )
        assert terms["recognitional_reward"] == pytest.approx(
            terms["recognitional_action_reward"] + terms["recognitional_battery_reward"]
        )
        assert "recognitional_battery_oracle" in info
        assert "recognitional_offline_oracle" in info
    finally:
        env.close()
