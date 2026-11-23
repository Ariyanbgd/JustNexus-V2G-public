import numpy as np
import pytest

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.observations import observation_size
from justnexus_v2g.gym_env.rewards import RewardNormalization, RewardWeights


def make_env():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    config = GymEnvConfig(nexus="energy_mobility", justice=("procedural",))
    env = JustNexusV2GEnv(
        params,
        config=config,
        reward_weights=RewardWeights(
            unmet_departure_energy=1.0,
            grid_constraint=1.0,
            total_ev_clipping=1.0,
            procedural_consistency=1.0,
            unauthorized_discharge_incidence=1.0,
        ),
        reward_normalization=RewardNormalization(
            departure_energy_shortfall_kwh=74.0,
            import_overload_squared_kw2h=267.9392226234346,
            command_mismatch_kwh=160.0,
        ),
        grid_operation="soft_import_unlimited_export",
    )
    return params, config, env


def test_procedural_observation_adds_current_discharge_authority():
    params, config, env = make_env()
    observation, _ = env.reset(seed=0)
    assert observation_size(params, config=config) == 57
    assert observation.shape == (57,)
    assert observation[[16, 26, 36, 46, 56]].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    env.close()


def test_unavailable_discharge_authority_still_allows_charging():
    _, _, env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.asarray([0.0, 4.0, 0.0, 3.7, 7.0]))
    procedural = info["procedural"]
    assert procedural["authority_resolved_request_kw"] == pytest.approx([0.0, 4.0, 0.0, 3.7, 7.0])
    assert procedural["authority_adjustment_kw"] == pytest.approx([0.0] * 5)
    assert info["reward_terms"]["normalized_authority_adjustment"] == 0.0
    assert info["reward_terms"]["procedural_consistency_loss"] == 0.0
    env.close()


def test_only_unauthorized_discharge_is_an_authority_violation():
    _, _, env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.asarray([0.0, 0.0, 0.0, 0.0, -7.0]))
    procedural = info["procedural"]
    assert procedural["authority_resolved_request_kw"] == pytest.approx([0.0] * 5)
    assert procedural["authority_adjustment_kw"] == pytest.approx([0.0, 0.0, 0.0, 0.0, 7.0])
    # At step zero h1, h3 and h4 are discharge-restricted; only h4 violates.
    assert info["reward_terms"]["normalized_authority_adjustment"] == pytest.approx(1.0 / 3.0)
    assert info["reward_terms"]["unauthorized_discharge_incidence_reward"] == pytest.approx(-(1.0 / 3.0) / 48.0)
    assert info["reward_terms"]["command_mismatch_kwh"] == 0.0
    env.close()


def test_consistency_uses_binary_discharge_activity_only():
    _, _, env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.asarray([-7.0, 4.0, 0.0, 3.7, 7.0]))
    procedural = info["procedural"]
    assert procedural["authority_resolved_active"] == [True, False, False, False, False]
    assert procedural["consistency_by_authority_status"] == {
        "available": 1.0,
        "unavailable": 0.0,
    }
    assert info["reward_terms"]["procedural_consistency_loss"] == 0.5
    env.close()
