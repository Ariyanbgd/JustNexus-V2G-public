"""Check priority-sweep weights, day modes, and validation-front selection."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.epistemic import mark_non_dominated, validate_priority_sweep
from justnexus_v2g.rl.evaluation import evaluate_epistemic_candidate
from rl_studies.config import load_study_inputs


@pytest.mark.parametrize("priority,expected", [
    (0.0, (0.0, 2.0, 1.0)),
    (0.5, (1.0, 1.0, 1.0)),
    (1.0, (2.0, 0.0, 1.0)),
])
def test_epistemic_priority_uses_base_observation_and_reward(priority, expected):
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility", justice="epistemic")
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice="epistemic",
        epistemic_priority=priority, configuration=inputs.configuration,
        source_params=inputs.source_params,
    )
    try:
        assert problem.env.observation_space.shape == (52,)
        assert problem.provenance["reward"]["epistemic"]["priority_weight"] == priority
        weights = problem.provenance["reward"]["weights"]
        assert tuple(weights[key] for key in (
            "unmet_departure_energy", "grid_constraint", "total_ev_clipping",
        )) == expected
        problem.env.reset(seed=7)
        for _ in range(48):
            _, reward, terminated, truncated, info = problem.env.step(
                np.ones(5, dtype=np.float32)
            )
            terms = info["reward_terms"]
            assert reward == pytest.approx(
                -expected[0] * terms["departure_energy_shortfall_kwh"] / 74
                -expected[1] * terms["import_overload_squared_kw2h"] / 267.9392226234346
                -expected[2] * terms["command_mismatch_kwh"] / 160
            )
            if terminated or truncated:
                break
    finally:
        problem.env.close()


def test_variable_epistemic_keeps_shape_and_separate_days():
    inputs = load_study_inputs(mode="variable", nexus="energy_mobility", justice="epistemic")
    configuration = deepcopy(inputs.configuration)
    configuration["mode"]["days"]["counts"] = {"train": 2, "validation": 1, "test": 1}
    problem = make_rl_problem(
        mode="variable", nexus="energy_mobility", justice="epistemic",
        epistemic_priority=0.5, configuration=configuration,
        source_params=inputs.source_params,
    )
    try:
        observation, info = problem.env.reset(seed=11, options={"day_index": 1})
        assert observation.shape == (52,)
        assert info["day_id"] == "train_001"
        assert len(problem.provenance["day_records"]) == 2
    finally:
        problem.env.close()


def test_priority_sweep_inherits_configurable_base_weight_total():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility", justice="epistemic")
    configuration = deepcopy(inputs.configuration)
    configuration["base"]["reward_weights"] = {
        "unmet_departure_energy": 2.0,
        "grid_constraint": 3.0,
        "total_ev_clipping": 0.5,
    }
    configuration["epistemic"]["priority_weights"] = [0.25]
    problem = make_rl_problem(
        mode="fixed", nexus="energy_mobility", justice="epistemic",
        epistemic_priority=0.25, configuration=configuration,
        source_params=inputs.source_params,
    )
    try:
        assert problem.provenance["reward"]["weights"] == {
            "unmet_departure_energy": 1.25,
            "grid_constraint": 3.75,
            "total_ev_clipping": 0.5,
        }
    finally:
        problem.env.close()


def test_priority_config_validation_and_raw_axis_front():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility", justice="epistemic")
    config = deepcopy(inputs.configuration["epistemic"])
    base = inputs.configuration["base"]["reward_weights"]
    priorities, seeds = validate_priority_sweep(config, base)
    assert len(priorities) == 11 and len(seeds) == 3
    config["priority_weights"].append(0.5)
    with pytest.raises(ValueError, match="distinct"):
        validate_priority_sweep(config, base)
    rows = [
        {"mean_mobility_shortfall_kwh": 2.0, "mean_import_overload_squared_kw2h": 1.0},
        {"mean_mobility_shortfall_kwh": 1.0, "mean_import_overload_squared_kw2h": 2.0},
        {"mean_mobility_shortfall_kwh": 3.0, "mean_import_overload_squared_kw2h": 3.0},
    ]
    mark_non_dominated(rows)
    assert [r["validation_non_dominated"] for r in rows] == [True, True, False]


def test_candidate_evaluation_reports_raw_axes_on_explicit_split():
    class IdlePolicy:
        def predict(self, observation, deterministic=True):
            return np.zeros(5, dtype=np.float32), None

    inputs = load_study_inputs(mode="variable", nexus="energy_mobility", justice="epistemic")
    configuration = deepcopy(inputs.configuration)
    configuration["mode"]["days"]["counts"] = {"train": 2, "validation": 2, "test": 1}
    problem = make_rl_problem(
        mode="variable", nexus="energy_mobility", justice="epistemic",
        epistemic_priority=0.5, split="validation", configuration=configuration,
        source_params=inputs.source_params,
    )
    try:
        result = evaluate_epistemic_candidate(IdlePolicy(), problem, seed=20000)
        assert result["split"] == "validation"
        assert [row["day_id"] for row in result["episodes"]] == ["validation_000", "validation_001"]
        assert all(row["steps"] == 48 for row in result["episodes"])
        assert result["mean_mobility_shortfall_kwh"] == pytest.approx(74.0)
        assert result["mean_import_overload_squared_kw2h"] == pytest.approx(
            sum(row["import_overload_squared_kw2h"] for row in result["episodes"]) / 2
        )
    finally:
        problem.env.close()
