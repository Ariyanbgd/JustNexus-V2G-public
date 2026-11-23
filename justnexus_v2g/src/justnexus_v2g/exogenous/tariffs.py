from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.exogenous.cycle import profile_step


def import_tariff_profile(
    params: SimulationParams,
    t: int,
    *,
    rng=None,
) -> float:
    """Import tariff in EUR/kWh."""
    local_t = profile_step(params, t)
    if params.system.import_tariff_periods:
        for period in params.system.import_tariff_periods:
            if int(period["start_step"]) <= local_t <= int(period["end_step"]):
                return float(period["eur_per_kwh"])
        if params.system.import_tariff_default_eur_per_kwh is None:
            raise ValueError(
                "A scenario with import_tariff_periods must define "
                "import_tariff_default_eur_per_kwh for uncovered steps."
            )
        return float(params.system.import_tariff_default_eur_per_kwh)
    if 34 <= local_t <= 42:
        return 0.40  # evening peak
    if 16 <= local_t <= 30:
        return 0.25  # daytime
    return 0.15  # off-peak


def export_tariff_profile(
    params: SimulationParams,
    t: int,
    *,
    rng=None,
) -> float:
    """Export tariff in EUR/kWh."""
    if params.system.export_tariff_mode == "dynamic_import":
        return import_tariff_profile(params, t, rng=rng)
    if params.system.export_tariff_mode == "dynamic_import_discount":
        import_tariff = import_tariff_profile(params, t, rng=rng)
        discount = max(params.system.export_tariff_discount_eur_per_kwh, 0.0)
        return max(import_tariff - discount, 0.0)
    local_t = profile_step(params, t)
    if 16 <= local_t <= 30:
        return 0.12  # daytime export value
    return 0.08
