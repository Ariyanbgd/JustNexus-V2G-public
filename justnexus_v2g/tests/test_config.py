from justnexus_v2g.core.config import load_simulation_params


def test_load_simulation_params_with_procedural_rules(tmp_path):
    config_path = tmp_path / "scenario.yaml"
    config_path.write_text(
        """
n_households: 2
dt_hours: 0.5
episode_steps: 4
system:
  max_import_kw: 10.0
  max_export_kw: 10.0
households:
  - ev:
      battery_capacity_kwh: 60.0
      max_charge_kw: 7.0
      max_discharge_kw: 7.0
      charge_efficiency: 0.95
      discharge_efficiency: 0.95
      initial_energy_kwh: 10.0
    procedural:
      authority_mode: none
      max_authorized_charge_kw: 3.0
      max_authorized_discharge_kw: 0.0
      fallback_controller: arbitrary
      fallback_power_kw: 2.0
  - ev:
      battery_capacity_kwh: 40.0
      max_charge_kw: 5.0
      max_discharge_kw: 4.0
      charge_efficiency: 0.95
      discharge_efficiency: 0.95
      initial_energy_kwh: 20.0
""",
        encoding="utf-8",
    )

    params = load_simulation_params(config_path)

    first = params.households[0].procedural
    assert first.authority_mode == "none"
    assert first.max_authorized_charge_kw == 3.0
    assert first.max_authorized_discharge_kw == 0.0
    assert first.fallback_controller == "arbitrary"
    assert first.fallback_power_kw == 2.0

    second = params.households[1].procedural
    assert second.authority_mode == "full"
    assert second.max_authorized_charge_kw == 5.0
    assert second.max_authorized_discharge_kw == 4.0


def test_load_simulation_params_with_household_profiles(tmp_path):
    config_path = tmp_path / "scenario_profiles.yaml"
    config_path.write_text(
        """
n_households: 1
dt_hours: 0.5
episode_steps: 4
system:
  max_import_kw: 10.0
  max_export_kw: 10.0
households:
  - ev:
      battery_capacity_kwh: 60.0
      max_charge_kw: 7.0
      max_discharge_kw: 7.0
      charge_efficiency: 0.95
      discharge_efficiency: 0.95
      initial_energy_kwh: 10.0
    mobility:
      departure_step: 3
      disconnected_start: 3
      disconnected_end: 6
      required_departure_energy_kwh: 24.0
      trip_energy_kwh: 12.0
    load:
      base_kw: 1.4
      evening_peak_kw: 0.8
      evening_start: 2
      evening_end: 3
    pv:
      peak_kw: 2.5
      peak_step: 2
      generation_start: 1
      generation_end: 3
      taper_kw_per_step: 0.25
""",
        encoding="utf-8",
    )

    params = load_simulation_params(config_path)
    household = params.households[0]

    assert household.mobility.departure_step == 3
    assert household.mobility.required_departure_energy_kwh == 24.0
    assert household.load.base_kw == 1.4
    assert household.load.evening_peak_kw == 0.8
    assert household.pv.peak_kw == 2.5
    assert household.pv.taper_kw_per_step == 0.25


def test_none_authority_defaults_to_idle_fallback(tmp_path):
    config_path = tmp_path / "scenario_default_fallback.yaml"
    config_path.write_text(
        """
n_households: 1
dt_hours: 0.5
episode_steps: 4
system:
  max_import_kw: 10.0
  max_export_kw: 10.0
households:
  - ev:
      battery_capacity_kwh: 60.0
      max_charge_kw: 7.0
      max_discharge_kw: 7.0
      charge_efficiency: 0.95
      discharge_efficiency: 0.95
      initial_energy_kwh: 10.0
    procedural:
      authority_mode: none
""",
        encoding="utf-8",
    )

    params = load_simulation_params(config_path)

    assert params.households[0].procedural.fallback_controller == "idle"
