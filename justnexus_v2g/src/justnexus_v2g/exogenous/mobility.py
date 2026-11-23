from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.exogenous.cycle import active_cycle_steps, profile_step


def ev_connected_profile(
    params: SimulationParams,
    household_index: int,
    t: int,
    *,
    rng=None,
) -> bool:
    """Return whether the EV is connected at simulation step t."""
    profile = params.households[household_index].mobility
    if profile.always_connected:
        return True
    disconnected_start = profile.disconnected_start
    disconnected_end = profile.disconnected_end

    if disconnected_start is None:
        disconnected_start = 16 + household_index
    if disconnected_end is None:
        disconnected_end = 30 + household_index

    local_t = profile_step(params, t)
    return not _in_window(local_t, disconnected_start, disconnected_end)


def time_to_departure_profile(
    params: SimulationParams,
    household_index: int,
    t: int,
    *,
    rng=None,
) -> float:
    """Hours until the next configured departure."""
    profile = params.households[household_index].mobility
    if profile.always_connected:
        return 0.0
    departure_t = profile.departure_step
    if departure_t is None:
        departure_t = 16 + household_index

    cycle_steps = active_cycle_steps(params)
    if cycle_steps is None:
        if t <= departure_t:
            return (departure_t - t) * params.dt_hours
        return 0.0

    local_t = profile_step(params, t)
    if local_t <= departure_t:
        steps_until_departure = departure_t - local_t
    else:
        steps_until_departure = cycle_steps - local_t + departure_t
    return steps_until_departure * params.dt_hours


def required_departure_energy_profile(
    params: SimulationParams,
    household_index: int,
    t: int,
    *,
    rng=None,
) -> float:
    """Required EV energy before next departure."""
    profile = params.households[household_index].mobility
    if profile.always_connected:
        return 0.0
    requirement = profile.required_departure_energy_kwh
    if requirement is None:
        requirement = 18.0 + 2.0 * household_index

    if time_to_departure_profile(params, household_index, t, rng=rng) > 0.0:
        return requirement

    return 0.0


def trip_energy_profile(
    params: SimulationParams,
    household_index: int,
    t: int,
    *,
    rng=None,
) -> float:
    """Energy consumed by the trip that starts at departure time t."""
    profile = params.households[household_index].mobility
    if profile.always_connected:
        return 0.0
    departure_t = profile.departure_step
    if departure_t is None:
        departure_t = 16 + household_index

    trip_energy = profile.trip_energy_kwh
    if trip_energy is None:
        trip_energy = 8.0 + household_index

    if profile_step(params, t) == departure_t:
        return trip_energy

    return 0.0


def _in_window(t: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end
