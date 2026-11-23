from abc import ABC, abstractmethod

from justnexus_v2g.core.action import SimulatorAction
from justnexus_v2g.core.state import SimulatorState


class BaseController(ABC):
    """Base interface for simulator controllers."""

    @abstractmethod
    def act(self, state: SimulatorState) -> SimulatorAction:
        """Return an action for the current simulator state."""
        raise NotImplementedError
