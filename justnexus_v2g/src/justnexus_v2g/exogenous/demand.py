from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.exogenous.cycle import profile_step


def household_demand_profile(
    params: SimulationParams,
    household_index: int,
    t: int,
    *,
    rng=None,
) -> float:
    """Household electricity demand in kW."""
    profile = params.households[household_index].load
    local_t = profile_step(params, t)

    base = profile.base_kw
    if base is None:
        base = 0.8 + 0.1 * household_index

    evening_peak = profile.evening_peak_kw
    if evening_peak is None:
        evening_peak = 0.6

    evening_start = profile.evening_start if profile.evening_start is not None else 34
    evening_end = profile.evening_end if profile.evening_end is not None else 42
    evening_bonus = evening_peak if evening_start <= local_t <= evening_end else 0.0

    morning_bonus = 0.0
    if profile.morning_peak_kw is not None:
        if profile.morning_start is None or profile.morning_end is None:
            raise ValueError(
                "morning_start and morning_end are required with morning_peak_kw"
            )
        if profile.morning_start <= local_t <= profile.morning_end:
            morning_bonus = profile.morning_peak_kw

    return base + morning_bonus + evening_bonus
