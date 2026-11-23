from justnexus_v2g.core.params import SimulationParams


def profile_step(params: SimulationParams, t: int) -> int:
    """Map an episode step to the configured repeating profile step."""
    if params.cycle_steps is None:
        return t
    return t % params.cycle_steps


def active_cycle_steps(params: SimulationParams) -> int | None:
    return params.cycle_steps
