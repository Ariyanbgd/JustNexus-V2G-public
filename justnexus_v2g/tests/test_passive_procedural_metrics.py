"""Passive authority reference must be shared without changing execution."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.gym_env.procedural import apply_procedural_rules
from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.evaluation import evaluate_epistemic_candidate, evaluate_split
from justnexus_v2g.rl.metrics import episode_metrics
from rl_studies.config import load_study_inputs


class DischargePolicy:
    def predict(self, observation, deterministic=True):
        return np.array([0, 0, 0, 0, -7], dtype=np.float32), None


@pytest.mark.parametrize("justice", [None, "procedural", "epistemic"])
@pytest.mark.parametrize("mode", ["fixed", "variable"])
def test_passive_metrics_cover_policy_and_day_modes(justice, mode):
    inputs = load_study_inputs(mode=mode, nexus="energy_mobility", justice=justice)
    config = deepcopy(inputs.configuration)
    if mode == "variable":
        config["mode"]["days"]["counts"] = {"train": 1, "validation": 1, "test": 1}
    problem = make_rl_problem(
        mode=mode, nexus="energy_mobility", justice=justice, split="test",
        epistemic_priority=0.5 if justice == "epistemic" else None,
        configuration=config, source_params=inputs.source_params,
    )
    try:
        evaluate = evaluate_epistemic_candidate if justice == "epistemic" else evaluate_split
        report = evaluate(DischargePolicy(), problem, seed=7, include_metrics=True)
    finally:
        problem.env.close()
    metric = report["episodes"][0]["metrics"]["procedural"]
    assert metric["prohibited_request_count"] > 0
    assert metric["prohibited_opportunity_count"] > 0
    assert metric["prohibited_request_rate"] == pytest.approx(
        metric["prohibited_request_count"] / metric["prohibited_opportunity_count"]
    )
    assert metric["authority_application"] == ("enforced" if justice == "procedural" else "shadow")
    if justice == "procedural":
        assert metric["prohibited_execution_count"] == 0
    else:
        # This requested discharge is physically blocked on the selected day.
        assert metric["prohibited_execution_count"] == 0


def test_preaction_window_boundaries_and_zero_opportunities():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility")
    params = inputs.source_params
    for index, start, end in ((1, 12, 32), (3, 30, 44)):
        for t, expected in ((start - 1, True), (start, False), (end - 1, False), (end, True)):
            _, report = apply_procedural_rules([-2.0] * 5, params, decision_step=t)
            assert (report.authority_bounds_kw[index].min_kw >= 0) is expected
    full = deepcopy(params)
    for household in full.households:
        household.procedural.authority_mode = "full"
    _, report = apply_procedural_rules([-2.0] * 5, full, decision_step=0)
    assert report.unauthorized_discharge_opportunities == 0


@pytest.mark.parametrize("justice,expected_execution", [(None, 1), ("procedural", 0)])
def test_connected_no_authority_household_executes_only_without_gate(justice, expected_execution):
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility", justice=justice)
    problem = make_rl_problem(mode="fixed", nexus="energy_mobility", justice=justice,
                              configuration=inputs.configuration, source_params=inputs.source_params)
    try:
        problem.env.reset(seed=7)
        infos = []
        for step in range(48):
            command = [0.0, 0.0, 0.0, 0.0, 7.0 if step == 0 else -7.0 if step == 1 else 0.0]
            _, _, _, _, info = problem.env.step(np.asarray(command, dtype=np.float32))
            infos.append(info)
        metric = episode_metrics(infos, inputs.source_params)["procedural"]
    finally:
        problem.env.close()
    if expected_execution:
        assert infos[1]["households"][4]["applied_ev_power_kw"] < -0.001
    else:
        assert infos[1]["households"][4]["applied_ev_power_kw"] == 0
    assert metric["prohibited_opportunity_count"] == 110
    assert metric["prohibited_request_count"] == 1
    assert metric["prohibited_execution_count"] == expected_execution
    assert metric["prohibited_request_rate"] == pytest.approx(1 / 110)
    assert metric["prohibited_execution_rate"] == pytest.approx(expected_execution / 110)


@pytest.mark.parametrize("invalid", [[0.0], [0.0, 0.0, 0.0, 0.0, float("nan")]])
def test_invalid_raw_vector_rejected(invalid):
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility")
    problem = make_rl_problem(mode="fixed", nexus="energy_mobility",
                              configuration=inputs.configuration, source_params=inputs.source_params)
    try:
        problem.env.reset(seed=7)
        infos = [problem.env.step(np.zeros(5, dtype=np.float32))[-1] for _ in range(48)]
        infos[0]["constraints"]["requested_ev_power_kw"] = invalid
        with pytest.raises(ValueError, match="Invalid pre-gate request"):
            episode_metrics(infos, inputs.source_params)
    finally:
        problem.env.close()


def test_missing_enforcement_trace_rejected():
    inputs = load_study_inputs(mode="fixed", nexus="energy_mobility", justice="procedural")
    problem = make_rl_problem(mode="fixed", nexus="energy_mobility", justice="procedural",
                              configuration=inputs.configuration, source_params=inputs.source_params)
    try:
        problem.env.reset(seed=7)
        infos = [problem.env.step(np.zeros(5, dtype=np.float32))[-1] for _ in range(48)]
        del infos[0]["procedural"]["trace"]
        with pytest.raises(ValueError, match="pre-gate raw trace"):
            episode_metrics(infos, inputs.source_params)
    finally:
        problem.env.close()
