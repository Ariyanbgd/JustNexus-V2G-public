import numpy as np
import pytest

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.procedural import (
    UNAUTHORIZED_DISCHARGE_EPSILON_KW, apply_procedural_rules,
)
from justnexus_v2g.gym_env.rewards import RewardNormalization, RewardWeights


def env_for():
    params = load_simulation_params("configs/scenario_energy_mobility.yaml")
    weights = RewardWeights(
        unmet_departure_energy=1, grid_constraint=1,
        total_ev_clipping=1, procedural_consistency=1,
        unauthorized_discharge_incidence=1,
    )
    env = JustNexusV2GEnv(
        params, config=GymEnvConfig(nexus="energy_mobility", justice=("procedural",)),
        reward_weights=weights,
        reward_normalization=RewardNormalization(
            departure_energy_shortfall_kwh=74,
            import_overload_squared_kw2h=267.9392226234346,
            command_mismatch_kwh=160,
        ), grid_operation="soft_import_unlimited_export",
    )
    env.reset(seed=0)
    return env


@pytest.mark.parametrize("raw_kw,expected", [
    (-1.0, 1), (-UNAUTHORIZED_DISCHARGE_EPSILON_KW, 0),
    (-UNAUTHORIZED_DISCHARGE_EPSILON_KW / 2, 0), (0.0, 0), (3.0, 0),
])
def test_pre_gate_incident_edge_and_charging(raw_kw, expected):
    env = env_for()
    _, report = apply_procedural_rules([0, 0, 0, 0, raw_kw], env.params, env.simulator.state)
    assert report.unauthorized_discharge_opportunities == 3
    assert report.unauthorized_discharge_incidents == expected
    assert report.unauthorized_discharge_incidence_step == pytest.approx(expected / 3)
    if expected:
        assert report.authority_resolved_request_kw[4] == 0
    env.close()


def test_available_and_limited_window_and_zero_opportunities():
    env = env_for()
    state = env.simulator.state
    _, report = apply_procedural_rules([-2, -2, -2, -2, -2], env.params, state)
    assert report.authority_available == [True, False, True, False, False]
    assert report.unauthorized_discharge_incidents == 3
    # At an authorized limited window, h1 and h3 can discharge.
    state.t = 20
    _, later = apply_procedural_rules([-2] * 5, env.params, state)
    assert later.unauthorized_discharge_opportunities <= 3
    assert later.unauthorized_discharge_incidents == later.unauthorized_discharge_opportunities
    # A report with no restricted households has a zero training loss, not a
    # 100%-compliance evaluation claim.
    from dataclasses import replace
    unrestricted = replace(later, authority_bounds_kw=[replace(b, min_kw=-7) for b in later.authority_bounds_kw])
    assert unrestricted.unauthorized_discharge_opportunities == 0
    assert unrestricted.unauthorized_discharge_incidence_step == 0
    env.close()


def test_reward_adds_only_incidence_and_binary_consistency():
    action = np.array([-7, -1, 0, 0, -7], dtype=np.float32)
    env = env_for()
    try:
        _, reward, _, _, info = env.step(action)
        terms = info["reward_terms"]
        assert terms["unauthorized_discharge_incidents"] == 2
        assert terms["unauthorized_discharge_opportunities"] == 3
        assert terms["unauthorized_discharge_incidence_reward"] == pytest.approx(-2 / 3 / 48)
        assert terms["procedural_consistency_loss"] == 0.5
        assert terms["procedural_consistency_reward"] == pytest.approx(-0.5 / 48)
        assert terms["procedural_reward"] == pytest.approx(-(2 / 3 + 0.5) / 48)
        assert reward == pytest.approx(sum(terms[key] for key in (
            "household_reward", "grid_reward", "aggregator_reward", "procedural_reward"
        )))
    finally:
        env.close()
