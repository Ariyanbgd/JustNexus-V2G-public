from dataclasses import dataclass, field


@dataclass
class LoadProfileParams:
    """Household demand profile parameters."""

    base_kw: float | None = None
    evening_peak_kw: float | None = None
    evening_start: int | None = None
    evening_end: int | None = None
    morning_peak_kw: float | None = None
    morning_start: int | None = None
    morning_end: int | None = None


@dataclass
class PVProfileParams:
    """Household PV generation profile parameters."""

    peak_kw: float | None = None
    peak_step: int | None = None
    generation_start: int | None = None
    generation_end: int | None = None
    taper_kw_per_step: float | None = None


@dataclass
class MobilityProfileParams:
    """Household EV connection and mobility-need profile parameters."""

    always_connected: bool = False
    disconnected_start: int | None = None
    disconnected_end: int | None = None
    departure_step: int | None = None
    required_departure_energy_kwh: float | None = None
    trip_energy_kwh: float | None = None


@dataclass
class EVParams:
    """Physical parameters of one EV."""

    battery_capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    initial_energy_kwh: float
    initial_energy_min_kwh: float | None = None
    initial_energy_max_kwh: float | None = None
    target_final_energy_kwh: float | None = None
    daily_battery_throughput_limit_kwh: float | None = None
    daily_battery_reserve_fraction: float | None = None


@dataclass
class ProceduralParams:
    """Procedural authority rules for one household."""

    authority_mode: str = "full"
    authority_scope: str = "all_actions"
    max_authorized_charge_kw: float | None = None
    max_authorized_discharge_kw: float | None = None
    authorized_start_step: int | None = None
    authorized_end_step: int | None = None
    fallback_controller: str = "departure_aware"
    fallback_power_kw: float = 0.0


@dataclass
class HouseholdParams:
    """Parameters of one household."""

    ev: EVParams
    procedural: ProceduralParams = field(default_factory=ProceduralParams)
    load: LoadProfileParams = field(default_factory=LoadProfileParams)
    pv: PVProfileParams = field(default_factory=PVProfileParams)
    mobility: MobilityProfileParams = field(default_factory=MobilityProfileParams)


@dataclass
class SystemParams:
    """Shared system-level parameters."""

    max_import_kw: float
    max_export_kw: float
    export_tariff_mode: str = "fixed_low"
    export_tariff_discount_eur_per_kwh: float = 0.0
    import_tariff_default_eur_per_kwh: float | None = None
    import_tariff_periods: list[dict] = field(default_factory=list)


@dataclass
class SimulationParams:
    """Top-level simulation parameters."""

    n_households: int
    dt_hours: float
    episode_steps: int
    households: list[HouseholdParams]
    system: SystemParams
    cycle_steps: int | None = None
