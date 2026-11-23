from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.exogenous.cycle import profile_step


def household_pv_profile(
    params: SimulationParams,
    household_index: int,
    t: int,
    *,
    rng=None,
) -> float:
    """Household solar generation in kW."""
    profile = params.households[household_index].pv
    local_t = profile_step(params, t)

    if profile.peak_kw is None:
        if 16 <= local_t <= 30:
            peak = 1.5 - 0.1 * abs(23 - local_t)
            household_scale = 1.0 + 0.05 * household_index
            return max(0.0, peak * household_scale)
        return 0.0

    generation_start = (
        profile.generation_start if profile.generation_start is not None else 16
    )
    generation_end = (
        profile.generation_end if profile.generation_end is not None else 30
    )
    peak_step = profile.peak_step if profile.peak_step is not None else 23
    taper = profile.taper_kw_per_step if profile.taper_kw_per_step is not None else 0.1

    if generation_start <= local_t <= generation_end:
        return max(0.0, profile.peak_kw - taper * abs(peak_step - local_t))

    return 0.0
