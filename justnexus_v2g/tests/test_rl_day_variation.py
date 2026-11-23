"""Scenario-owned variable-day rules and compatibility with saved run snapshots."""

from copy import deepcopy

import pytest
import yaml

from justnexus_v2g.rl.day_suite import generate_day_splits
from rl_studies.config import DEFAULT_SCENARIO, load_study_inputs


def _scenario_file(tmp_path, variation):
    scenario = yaml.safe_load(DEFAULT_SCENARIO.read_text())
    if variation is None:
        scenario.pop("day_variation")
    else:
        scenario["day_variation"] = variation
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(scenario, sort_keys=False))
    return path


def test_named_scenario_rules_select_which_fields_vary(tmp_path):
    variation = deepcopy(yaml.safe_load(DEFAULT_SCENARIO.read_text())["day_variation"])
    variation["load"] = {"fields": ["base_kw"], "factor_range": [1.1, 1.1]}
    variation["pv"] = {"fields": ["peak_kw"], "factor_range": [0.5, 0.6]}
    variation["departures"]["step_options_by_household"] = {
        "reliable_commuter": [16], "price_flexible_worker": [19],
        "solar_battery_conserver": [20], "low_power_affordability_home": [12],
        "battery_autonomy_home": [18],
    }
    inputs = load_study_inputs(
        mode="variable", nexus="energy_mobility",
        scenario_path=_scenario_file(tmp_path, variation),
    )
    settings = deepcopy(inputs.configuration["mode"]["days"])
    settings["counts"] = {"train": 1, "validation": 1, "test": 1}
    splits, _ = generate_day_splits(
        inputs.source_params, settings,
        variation=inputs.configuration["day_variation"],
        household_labels=inputs.configuration["scenario_labels"],
    )
    day = splits["train"][0][0]
    for source, changed in zip(inputs.source_params.households, day.households, strict=True):
        assert changed.load.base_kw == pytest.approx(round(source.load.base_kw * 1.1, 6))
        assert changed.load.morning_peak_kw == source.load.morning_peak_kw
        assert changed.load.evening_peak_kw == source.load.evening_peak_kw
        assert source.pv.peak_kw * 0.5 <= changed.pv.peak_kw <= source.pv.peak_kw * 0.6
        assert changed.mobility.disconnected_start == changed.mobility.departure_step
        assert changed.mobility.disconnected_end == source.mobility.disconnected_end
    assert [h.mobility.departure_step for h in day.households] == [16, 19, 20, 12, 18]


def test_variable_rules_validate_fields_and_named_departures():
    inputs = load_study_inputs(mode="variable", nexus="energy_mobility")
    settings = deepcopy(inputs.configuration["mode"]["days"])
    settings["counts"] = {"train": 1, "validation": 1, "test": 1}
    variation = deepcopy(inputs.configuration["day_variation"])
    variation["load"]["fields"] = ["battery_capacity_kwh"]
    with pytest.raises(ValueError, match="supported fields"):
        generate_day_splits(inputs.source_params, settings, variation=variation,
                            household_labels=inputs.configuration["scenario_labels"])
    variation = deepcopy(inputs.configuration["day_variation"])
    variation["departures"]["step_options_by_household"].pop("reliable_commuter")
    with pytest.raises(ValueError, match="every scenario label"):
        generate_day_splits(inputs.source_params, settings, variation=variation,
                            household_labels=inputs.configuration["scenario_labels"])


def test_fixed_mode_needs_no_variation_and_older_snapshots_still_generate(tmp_path):
    path = _scenario_file(tmp_path, None)
    fixed = load_study_inputs(mode="fixed", nexus="energy_mobility", scenario_path=path)
    assert "day_variation" not in fixed.configuration
    with pytest.raises(ValueError, match="requires day_variation"):
        load_study_inputs(mode="variable", nexus="energy_mobility", scenario_path=path)
    legacy_settings = {
        "counts": {"train": 1, "validation": 1, "test": 1},
        "generator_seed": 32020, "sampling_seed": 32020,
        "load_factor_range": [0.8, 1.2], "pv_factor_range": [0.75, 1.25],
        "departure_options": [[14, 15, 16], [17, 18, 19], [20],
                              [12, 13, 14], [16, 17, 18]],
    }
    splits, generation = generate_day_splits(fixed.source_params, legacy_settings)
    assert generation["version"] == "energy_mobility_days_v1"
    assert splits["train"][0][1]["sha256"] == (
        "f53e0c86fc36327a25cd5047b899ae78f32369516c2fb40bef8262716a400380"
    )
