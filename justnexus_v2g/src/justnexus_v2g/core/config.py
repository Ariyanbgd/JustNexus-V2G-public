from pathlib import Path

import yaml

from justnexus_v2g.core.params import (
    EVParams,
    HouseholdParams,
    LoadProfileParams,
    MobilityProfileParams,
    ProceduralParams,
    PVProfileParams,
    SimulationParams,
    SystemParams,
)


def _authority_mode_from_config(raw: dict) -> str:
    if "authority_mode" in raw:
        return raw["authority_mode"]

    if raw.get("participation_allowed", True) is False:
        return "none"
    if raw.get("discharge_allowed", True) is False:
        return "charge_only"
    if "max_request_charge_kw" in raw or "max_request_discharge_kw" in raw:
        return "limited"
    return "full"


def _profile_params(raw: dict, cls):
    return cls(**(raw or {}))


def load_simulation_params(config_path: str | Path) -> SimulationParams:
    """Load simulation parameters from a YAML file."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    households = []
    for hh in raw["households"]:
        ev_raw = hh["ev"]
        initial_range = ev_raw.get("initial_energy_range_kwh")
        if initial_range is None:
            initial_min_kwh = ev_raw.get("initial_energy_min_kwh")
            initial_max_kwh = ev_raw.get("initial_energy_max_kwh")
        else:
            if len(initial_range) != 2:
                raise ValueError("initial_energy_range_kwh must contain [min, max].")
            initial_min_kwh = initial_range[0]
            initial_max_kwh = initial_range[1]

        ev = EVParams(
            battery_capacity_kwh=ev_raw["battery_capacity_kwh"],
            max_charge_kw=ev_raw["max_charge_kw"],
            max_discharge_kw=ev_raw["max_discharge_kw"],
            charge_efficiency=ev_raw["charge_efficiency"],
            discharge_efficiency=ev_raw["discharge_efficiency"],
            initial_energy_kwh=ev_raw["initial_energy_kwh"],
            initial_energy_min_kwh=initial_min_kwh,
            initial_energy_max_kwh=initial_max_kwh,
            target_final_energy_kwh=ev_raw.get("target_final_energy_kwh"),
            daily_battery_throughput_limit_kwh=ev_raw.get(
                "daily_battery_throughput_limit_kwh"
            ),
            daily_battery_reserve_fraction=ev_raw.get(
                "daily_battery_reserve_fraction"
            ),
        )
        procedural_raw = hh.get("procedural", {}) or {}
        authority_mode = _authority_mode_from_config(procedural_raw)
        procedural = ProceduralParams(
            authority_mode=authority_mode,
            authority_scope=procedural_raw.get("authority_scope", "all_actions"),
            max_authorized_charge_kw=procedural_raw.get(
                "max_authorized_charge_kw",
                procedural_raw.get("max_request_charge_kw", ev.max_charge_kw),
            ),
            max_authorized_discharge_kw=procedural_raw.get(
                "max_authorized_discharge_kw",
                procedural_raw.get("max_request_discharge_kw", ev.max_discharge_kw),
            ),
            authorized_start_step=procedural_raw.get("authorized_start_step"),
            authorized_end_step=procedural_raw.get("authorized_end_step"),
            fallback_controller=procedural_raw.get(
                "fallback_controller",
                "idle",
            ),
            fallback_power_kw=procedural_raw.get("fallback_power_kw", 0.0),
        )
        households.append(
            HouseholdParams(
                ev=ev,
                procedural=procedural,
                load=_profile_params(hh.get("load", {}), LoadProfileParams),
                pv=_profile_params(hh.get("pv", {}), PVProfileParams),
                mobility=_profile_params(hh.get("mobility", {}), MobilityProfileParams),
            )
        )

    system = SystemParams(
        max_import_kw=raw["system"]["max_import_kw"],
        max_export_kw=raw["system"]["max_export_kw"],
        export_tariff_mode=raw["system"].get("export_tariff_mode", "fixed_low"),
        export_tariff_discount_eur_per_kwh=float(
            raw["system"].get("export_tariff_discount_eur_per_kwh", 0.0)
        ),
        import_tariff_default_eur_per_kwh=raw["system"].get(
            "import_tariff_default_eur_per_kwh"
        ),
        import_tariff_periods=list(raw["system"].get("import_tariff_periods", [])),
    )
    cycle_steps = raw.get("cycle_steps")
    if cycle_steps is not None and cycle_steps <= 0:
        raise ValueError("cycle_steps must be positive when provided.")

    return SimulationParams(
        n_households=raw["n_households"],
        dt_hours=raw["dt_hours"],
        episode_steps=raw["episode_steps"],
        households=households,
        system=system,
        cycle_steps=cycle_steps,
    )
