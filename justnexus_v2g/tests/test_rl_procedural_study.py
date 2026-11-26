"""The paper's procedural formulation in the clean fixed/variable runner."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.evaluation import evaluate_split
from rl_studies.config import load_study_inputs


class IdlePolicy:
    def predict(self, observation, deterministic=True):
        return np.zeros(5, dtype=np.float32), None


@pytest.mark.parametrize("mode", ["fixed", "variable"])
def test_procedural_formulation_keeps_gate_observation_and_reward(mode):
    inputs = load_study_inputs(
        mode=mode, nexus="energy_mobility", justice="procedural",
    )
    problem = make_rl_problem(
        mode=mode, nexus="energy_mobility", justice="procedural",
        configuration=inputs.configuration, source_params=inputs.source_params,
    )
    try:
        observation, _ = problem.env.reset(seed=0, options={"day_index": 0})
        assert observation.shape == (57,)
        assert problem.provenance["reward"]["procedural"]["weights"] == {
            "procedural_consistency": 1.0,
            "unauthorized_discharge_incidence": 1.0,
        }
        _, reward, _, _, info = problem.env.step(
            np.array([0.0, 0.0, 0.0, 0.0, -7.0], dtype=np.float32)
        )
        terms = info["reward_terms"]
        assert info["procedural"]["authority_resolved_request_kw"] == pytest.approx([0.0] * 5)
        assert terms["unauthorized_discharge_incidents"] == 1.0
        assert terms["unauthorized_discharge_opportunities"] == 3.0
        assert terms["normalized_authority_adjustment"] == pytest.approx(1.0 / 3.0)
        assert terms["unauthorized_discharge_incidence_step"] == pytest.approx(1.0 / 3.0)
        expected = (
            -terms["procedural_consistency_loss"] / 48.0
            -terms["unauthorized_discharge_incidence_step"] / 48.0
        )
        assert terms["procedural_reward"] == pytest.approx(expected)
        assert reward == pytest.approx(
            sum(terms[key] for key in (
                "household_reward", "grid_reward", "aggregator_reward", "procedural_reward"
            ))
        )
    finally:
        problem.env.close()


def test_procedural_test_metrics_include_incidence_and_consistency():
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice="procedural",
    )
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice="procedural",
        split="test", configuration=inputs.configuration,
        source_params=inputs.source_params,
    )
    try:
        result = evaluate_split(IdlePolicy(), problem, seed=7, include_metrics=True)
    finally:
        problem.env.close()
    procedural = result["mean_metrics"]["procedural"]
    assert procedural["prohibited_opportunity_count"] > 0
    assert procedural["prohibited_request_rate"] == 0.0
    assert 0.0 <= procedural["mean_stepwise_discharge_consistency_score"] <= 1.0
    assert procedural["prohibited_execution_rate"] == 0.0


def test_procedural_weights_must_be_finite_and_nonnegative():
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice="procedural",
    )
    configuration = deepcopy(inputs.configuration)
    configuration["procedural"]["weights"]["unauthorized_discharge_incidence"] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_rl_problem(
            mode="fixed", nexus="energy_mobility", justice="procedural",
            configuration=configuration, source_params=inputs.source_params,
        )
