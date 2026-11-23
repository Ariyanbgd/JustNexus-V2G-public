"""Paper metric formulas and explicit day aggregation for the clean studies."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.evaluation import evaluate_split
from justnexus_v2g.rl.metrics import episode_metrics
from rl_studies.config import load_study_inputs


class IdlePolicy:
    def predict(self, observation, deterministic=True):
        return np.zeros(5, dtype=np.float32), None


def test_fixed_idle_policy_reports_paper_normalized_losses():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility")
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", split="test",
        configuration=inputs.configuration, source_params=inputs.source_params,
    )
    try:
        result = evaluate_split(IdlePolicy(), problem, seed=7, include_metrics=True)
    finally:
        problem.env.close()
    metrics = result["episodes"][0]["metrics"]
    assert metrics["operational"]["normalized_mobility_shortfall"] == pytest.approx(1.0)
    assert metrics["distributive"]["household_burden_fractions"] == [1.0] * 5
    assert metrics["distributive"]["worst_above_mean_mobility_burden"] == 0.0
    assert metrics["distributive"]["worst_household_burden"] == 1.0
    assert metrics["intertemporal"]["normalized_terminal_reserve_deficit"] == 1.0
    assert metrics["intertemporal"]["normalized_excess_controlled_throughput"] == 0.0
    assert result["mean_metrics"] == {
        group: {key: value for key, value in fields.items() if isinstance(value, (int, float, list))}
        for group, fields in metrics.items()
    }


def test_oracle_alignment_uses_clipped_household_squared_gaps():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility")
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", split="test",
        configuration=inputs.configuration, source_params=inputs.source_params,
    )
    try:
        oracle = {
            "actions_kw": np.ones((48, 5)).tolist(),
            "idle_to_oracle_l2_kw2h": [24.0] * 5,
            "sha256": "test_reference",
        }
        result = evaluate_split(
            IdlePolicy(), problem, seed=7, include_metrics=True, metric_oracles=[oracle],
        )
    finally:
        problem.env.close()
    alignment = result["mean_metrics"]["recognitional"]
    assert alignment["household_normalized_squared_gaps"] == pytest.approx([1.0] * 5)
    assert alignment["mean_clipped_oracle_alignment"] == pytest.approx(0.0)
    assert alignment["worst_above_mean_alignment_burden"] == pytest.approx(0.0)


def test_intertemporal_policy_can_be_audited_on_distributive_and_recognitional_metrics():
    inputs = load_study_inputs(
        mode="fixed", nexus="energy_mobility", justice="intertemporal",
    )
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice="intertemporal",
        split="test", configuration=inputs.configuration,
        source_params=inputs.source_params,
    )
    try:
        reference = {
            "actions_kw": np.ones((48, 5)).tolist(),
            "idle_to_oracle_l2_kw2h": [24.0] * 5,
            "sha256": "common_reference",
        }
        result = evaluate_split(
            IdlePolicy(), problem, seed=7, include_metrics=True,
            metric_oracles=[reference],
        )
    finally:
        problem.env.close()
    assert set(result["mean_metrics"]) == {
        "operational", "distributive", "recognitional", "intertemporal",
    }
    assert result["mean_metrics"]["distributive"]["worst_household_burden"] == 1.0
    assert result["mean_metrics"]["recognitional"]["mean_clipped_oracle_alignment"] == 0.0


def test_variable_days_report_each_day_then_arithmetic_mean():
    inputs = load_study_inputs(mode="variable", nexus="energy_mobility")
    configuration = deepcopy(inputs.configuration)
    configuration["mode"]["days"]["counts"] = {"train": 1, "validation": 1, "test": 2}
    problem = make_rl_problem(
        mode="variable", nexus="energy_mobility", split="test",
        configuration=configuration, source_params=inputs.source_params,
    )
    try:
        result = evaluate_split(IdlePolicy(), problem, seed=7, include_metrics=True)
    finally:
        problem.env.close()
    assert [row["day_id"] for row in result["episodes"]] == ["test_000", "test_001"]
    for group, fields in result["mean_metrics"].items():
        for key, value in fields.items():
            if isinstance(value, list):
                for i, mean in enumerate(value):
                    assert mean == pytest.approx(sum(row["metrics"][group][key][i]
                                                     for row in result["episodes"]) / 2)
            else:
                assert value == pytest.approx(sum(row["metrics"][group][key]
                                                  for row in result["episodes"]) / 2)


def test_procedural_rate_uses_pregate_incidents_over_opportunities():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility")
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", split="test",
        configuration=inputs.configuration, source_params=inputs.source_params,
    )
    try:
        problem.env.reset(seed=7)
        infos = []
        for step in range(48):
            _, _, _, _, info = problem.env.step(np.zeros(5, dtype=np.float32))
            info["reward_terms"]["unauthorized_discharge_incidents"] = 1 if step == 0 else 0
            info["reward_terms"]["unauthorized_discharge_opportunities"] = 2 if step == 0 else 0
            info["reward_terms"]["procedural_consistency_score"] = 0.75
            infos.append(info)
        metrics = episode_metrics(infos, problem.env.unwrapped.params)
    finally:
        problem.env.close()
    assert metrics["procedural"] == {
        "prohibited_request_count": 1,
        "prohibited_opportunity_count": 2,
        "prohibited_request_rate": 0.5,
        "mean_stepwise_discharge_consistency_score": 0.75,
    }
