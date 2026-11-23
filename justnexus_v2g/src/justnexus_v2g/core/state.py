from dataclasses import dataclass


@dataclass
class HouseholdState:
    """State of one household at one time step."""
    ev_connected: bool
    ev_energy_kwh: float
    time_to_departure_hours: float
    required_departure_energy_kwh: float
    load_kw: float
    pv_kw: float
    requested_ev_power_kw: float
    applied_ev_power_kw: float
    trip_energy_kwh: float
    battery_throughput_kwh: float
    discharge_burden_kwh: float
    was_deferred: bool
    deferral_count: int
    net_household_power_kw: float
    imported_energy_kwh: float
    exported_energy_kwh: float
    import_cost_eur: float
    export_revenue_eur: float
    net_energy_cost_eur: float
    unmet_demand_kwh: float = 0.0
    pv_curtailment_kwh: float = 0.0
    departure_energy_kwh: float = 0.0
    departure_energy_shortfall_kwh: float = 0.0


@dataclass
class SystemState:
    """Shared system-level state."""
    total_load_kw: float
    total_pv_kw: float
    total_ev_kw: float
    grid_exchange_kw: float
    import_tariff_eur_per_kwh: float
    export_tariff_eur_per_kwh: float
    imported_energy_kwh: float
    exported_energy_kwh: float
    import_cost_eur: float
    export_revenue_eur: float
    net_energy_cost_eur: float
    pre_intervention_grid_exchange_kw: float = 0.0
    unmet_demand_kwh: float = 0.0
    pv_curtailment_kwh: float = 0.0


@dataclass
class SimulatorState:
    """Top-level simulator state."""
    t: int
    households: list[HouseholdState]
    system: SystemState