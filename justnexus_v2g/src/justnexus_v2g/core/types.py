from dataclasses import dataclass

from justnexus_v2g.core.action import SimulatorAction
from justnexus_v2g.core.state import SimulatorState


@dataclass
class ConstraintReport:
    """Describes how requested actions were altered by simulator constraints."""
    requested_ev_power_kw: list[float]
    applied_ev_power_kw: list[float]
    clipped_by_connection: list[bool]
    clipped_by_power_limit: list[bool]
    clipped_by_energy_limit: list[bool]
    grid_constrained: bool
    pre_grid_total_ev_kw: float
    post_grid_total_ev_kw: float


@dataclass
class StepResult:
    """Result of one simulator step."""
    state: SimulatorState
    action: SimulatorAction
    done: bool
    constraints: ConstraintReport