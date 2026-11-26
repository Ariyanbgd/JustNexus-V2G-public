"""Canonical throughput reporting must not alter formulation behavior."""

from copy import deepcopy

import numpy as np
import pytest

from justnexus_v2g.gym_env.throughput import (
    MOBILITY_THROUGHPUT_REFERENCE_ID,
    controlled_throughput_caps_kwh,
)
from justnexus_v2g.rl.day_suite import generate_day_splits
from justnexus_v2g.rl.metrics import episode_metrics, mean_episode_metrics
from justnexus_v2g.rl.problem import _study_env
from rl_studies.config import load_study_inputs
from rl_studies.test import _metric_references


BASE_WEIGHTS = {
    "unmet_departure_energy": 1.0,
    "grid_constraint": 1.0,
    "total_ev_clipping": 1.0,
}
ORACLE = {
    "actions_kw": np.ones((48, 5)).tolist(),
    "idle_to_oracle_l1_kwh": [24.0] * 5,
}
CASES = (
    "base", "mean_deviation", "max_min", "generalized_gini", "recognitional",
    "procedural", "intertemporal", "combined", "epistemic_candidate",
)


def _day(mode):
    inputs = load_study_inputs(mode=mode, nexus="energy_mobility")
    if mode == "fixed":
        return inputs.source_params
    settings = deepcopy(inputs.configuration["mode"]["days"])
    settings["counts"] = {"train": 1, "validation": 1, "test": 1}
    splits, _ = generate_day_splits(
        inputs.source_params, settings,
        variation=inputs.configuration["day_variation"],
        household_labels=inputs.configuration["scenario_labels"],
    )
    return splits["test"][0][0]


def _env(day, case):
    weights = dict(BASE_WEIGHTS)
    options = {}
    if case in {"mean_deviation", "max_min", "generalized_gini"}:
        options["distributive"] = {
            "operator": case,
            **({"worst_burden_weight": 1.0, "mean_excess_burden_weight": 1.0}
               if case == "mean_deviation" else {"weight": 1.0}),
            **({"rank_weights_low_to_high": [0.40, 0.25, 0.17, 0.11, 0.07]}
               if case == "generalized_gini" else {}),
        }
    elif case == "recognitional":
        options.update(recognitional={"action_gap_weight": 0.2}, oracle_bundle=ORACLE)
    elif case == "procedural":
        options["procedural"] = {"weights": {
            "unauthorized_discharge_incidence": 1.0,
            "procedural_consistency": 1.0,
        }}
    elif case == "intertemporal":
        options["intertemporal"] = {"weights": {
            "excess_daily_battery_throughput": 1.0,
            "battery_reserve_deficit": 1.0,
        }}
    elif case == "combined":
        options.update(combined={
            "action_gap_weight": 0.2,
            "mobility_disparity_weights": {
                "worst_burden_weight": 1.0, "mean_excess_burden_weight": 1.0,
            },
            "alignment_disparity_weights": {
                "worst_gap_weight": 1.0, "mean_excess_gap_weight": 1.0,
            },
        }, oracle_bundle=ORACLE)
    elif case == "epistemic_candidate":
        # A candidate uses the base simulator with its scalarized training weights.
        weights.update(unmet_departure_energy=0.6, grid_constraint=1.4)
    return _study_env(day, options.pop("distributive", None), weights, **options)


def _episode(env, power):
    env.reset(seed=7)
    observations, rewards, infos = [], [], []
    while True:
        observation, reward, terminated, truncated, info = env.step(
            np.array([power, 0, 0, 0, 0], dtype=np.float32)
        )
        observations.append(observation)
        rewards.append(reward)
        infos.append(info)
        if terminated or truncated:
            break
    return observations, rewards, infos


@pytest.mark.parametrize("mode", ["fixed", "variable"])
def test_same_executed_trajectory_has_same_evaluation_excess(mode):
    day = _day(mode)
    cap = controlled_throughput_caps_kwh(day, nexus="energy_mobility")
    assert cap[0] == pytest.approx(31.315789473684212)
    reference_applied = None
    reference_metric = None
    for case in CASES:
        env = _env(day, case)
        try:
            observations, rewards, infos = _episode(env, 7.0)
            applied = [[h["applied_ev_power_kw"] for h in info["households"]]
                       for info in infos]
            metric = episode_metrics(infos, day)["intertemporal"]
            assert metric["controlled_throughput_reference"] == MOBILITY_THROUGHPUT_REFERENCE_ID
            assert metric["controlled_throughput_cap_kwh_by_household"] == pytest.approx(cap)
            assert metric["controlled_throughput_kwh_by_household"][0] > cap[0]
            assert metric["excess_controlled_throughput_kwh_by_household"][0] == pytest.approx(
                metric["controlled_throughput_kwh_by_household"][0] - cap[0]
            )
            assert metric["normalized_excess_controlled_throughput"] == pytest.approx(
                metric["excess_controlled_throughput_kwh"] / 28.5
            )
            if reference_applied is None:
                reference_applied, reference_metric = applied, metric
            else:
                assert np.asarray(applied) == pytest.approx(np.asarray(reference_applied))
                assert metric["controlled_throughput_kwh_by_household"] == pytest.approx(
                    reference_metric["controlled_throughput_kwh_by_household"]
                )
                assert metric["excess_controlled_throughput_kwh_by_household"] == pytest.approx(
                    reference_metric["excess_controlled_throughput_kwh_by_household"]
                )
            if case == "base":
                assert observations[0].shape == (52,)
                assert infos[-1]["daily_battery_throughput_limit_kwh_by_household"][0] == 5.0
                assert all("intertemporal_reward" not in info["reward_terms"] for info in infos)
                assert all("excess_daily_battery_throughput_reward" not in info["reward_terms"]
                           for info in infos)
            elif case == "intertemporal":
                assert observations[0].shape == (62,)
                assert infos[-1]["daily_battery_throughput_limit_kwh_by_household"] == pytest.approx(cap)
                assert infos[0]["reward_normalization"]["excess_daily_battery_throughput_kwh"] == 28.5
            elif case == "procedural":
                assert observations[0].shape == (57,)
                assert all(info["procedural"]["authority_resolved_request_kw"][0] == 7.0
                           for info in infos)
            elif case == "epistemic_candidate":
                assert observations[0].shape == (52,)
            assert len(rewards) == day.episode_steps - 1
        finally:
            env.close()


def test_cap_arithmetic_ignores_native_training_excess_and_trip_energy():
    day = _day("fixed")
    env = _env(day, "base")
    try:
        _, _, idle_infos = _episode(env, 0.0)
        cap = controlled_throughput_caps_kwh(day, nexus="energy_mobility")[0]
        for target in (cap - 1.0, cap, 35.0):
            infos = deepcopy(idle_infos)
            power = target / ((day.episode_steps - 1) * day.dt_hours)
            for info in infos:
                info["households"][0]["applied_ev_power_kw"] = power
                info["daily_excess_battery_throughput_kwh_by_household"][0] = 999.0
            metric = episode_metrics(infos, day)["intertemporal"]
            assert metric["controlled_throughput_kwh_by_household"] == pytest.approx(
                [target, 0, 0, 0, 0]
            )
            assert metric["excess_controlled_throughput_kwh_by_household"] == pytest.approx(
                [max(target - cap, 0.0), 0, 0, 0, 0]
            )
            if target == 35.0:
                assert metric["excess_controlled_throughput_kwh"] == pytest.approx(
                    3.684210526315788
                )
        first = episode_metrics(idle_infos, day)
        second = episode_metrics(infos, day)
        mean = mean_episode_metrics([{"metrics": first}, {"metrics": second}])
        assert "controlled_throughput_reference" not in mean["intertemporal"]
        assert mean["intertemporal"]["controlled_throughput_kwh_by_household"] == pytest.approx(
            [17.5, 0, 0, 0, 0]
        )
        assert mean["intertemporal"]["excess_controlled_throughput_kwh"] == pytest.approx(
            3.684210526315788 / 2
        )
        assert _metric_references({})["controlled_throughput_reference"] == (
            MOBILITY_THROUGHPUT_REFERENCE_ID
        )
    finally:
        env.close()


def test_household_excesses_sum_after_individual_caps():
    day = _day("fixed")
    env = _env(day, "base")
    try:
        _, _, infos = _episode(env, 0.0)
        assert sum(h["trip_energy_kwh"] for info in infos
                   for h in info["households"]) > 0.0
        assert episode_metrics(infos, day)["intertemporal"]["controlled_throughput_kwh"] == 0.0
        caps = controlled_throughput_caps_kwh(day, nexus="energy_mobility")
        targets = [35.0, caps[1] + 2.0, caps[2], caps[3] - 1.0, 0.0]
        for info in infos:
            for index, target in enumerate(targets):
                power = target / ((day.episode_steps - 1) * day.dt_hours)
                info["households"][index]["applied_ev_power_kw"] = (
                    -power if index == 1 else power
                )
        metric = episode_metrics(infos, day)["intertemporal"]
        expected = [max(target - cap, 0.0)
                    for target, cap in zip(targets, caps, strict=True)]
        assert metric["controlled_throughput_kwh_by_household"] == pytest.approx(targets)
        assert metric["excess_controlled_throughput_kwh_by_household"] == pytest.approx(expected)
        assert metric["excess_controlled_throughput_kwh"] == pytest.approx(sum(expected))
    finally:
        env.close()


def test_episode_reset_clears_training_accumulator_without_changing_evaluation_cap():
    day = _day("fixed")
    for case in ("base", "intertemporal"):
        env = _env(day, case)
        try:
            _, _, first = _episode(env, 7.0)
            assert episode_metrics(first, day)["intertemporal"]["excess_controlled_throughput_kwh"] > 0
            _, _, second = _episode(env, 0.0)
            assert second[-1]["daily_battery_throughput_kwh_by_household"] == [0.0] * 5
            assert episode_metrics(second, day)["intertemporal"]["excess_controlled_throughput_kwh"] == 0.0
        finally:
            env.close()


@pytest.mark.parametrize("trace_error", ["missing", "duplicate", "out_of_order"])
def test_metrics_reject_incomplete_or_unordered_transition_trace(trace_error):
    day = _day("fixed")
    env = _env(day, "base")
    try:
        _, _, infos = _episode(env, 0.0)
        if trace_error == "missing":
            del infos[1]
        elif trace_error == "duplicate":
            infos[1] = deepcopy(infos[0])
        else:
            infos[1], infos[2] = infos[2], infos[1]
        with pytest.raises(ValueError, match="complete ordered episode trace"):
            episode_metrics(infos, day)
    finally:
        env.close()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_metrics_reject_nonfinite_executed_power(nonfinite):
    day = _day("fixed")
    env = _env(day, "base")
    try:
        _, _, infos = _episode(env, 0.0)
        infos[0]["households"][0]["applied_ev_power_kw"] = nonfinite
        with pytest.raises(ValueError, match="Invalid executed household power trace"):
            episode_metrics(infos, day)
    finally:
        env.close()
