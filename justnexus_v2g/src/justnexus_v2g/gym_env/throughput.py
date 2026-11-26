"""Scenario reference for controlled EV throughput, independent of justice."""

from __future__ import annotations

from justnexus_v2g.core.params import SimulationParams


MOBILITY_THROUGHPUT_REFERENCE_ID = "energy_mobility_full_day_cap_v1"


def controlled_throughput_caps_kwh(params: SimulationParams, *, nexus: str) -> list[float]:
    """Return mobility charging-plus-reserve allowance caps for each household.

    The cap is an evaluation reference, not a physical constraint. Intertemporal
    training uses the same calculation for its existing observation and reward.
    """
    if nexus != "energy_mobility":
        raise ValueError(f"Unsupported controlled-throughput reference nexus: {nexus}")
    caps = []
    for household in params.households:
        allowance = household.ev.daily_battery_throughput_limit_kwh
        if allowance is None:
            allowance = household.ev.battery_capacity_kwh
        reserve_fraction = household.ev.daily_battery_reserve_fraction
        if reserve_fraction is None:
            reserve_fraction = 0.1
        reserve_target = float(reserve_fraction) * household.ev.battery_capacity_kwh
        required = float(household.mobility.required_departure_energy_kwh or 0.0)
        efficiency = float(household.ev.charge_efficiency)
        essential_charging = (
            (required + reserve_target) / efficiency if efficiency > 1e-12 else 0.0
        )
        caps.append(float(max(essential_charging + float(allowance), 0.0)))
    return caps
