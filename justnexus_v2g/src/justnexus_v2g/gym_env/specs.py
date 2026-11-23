from dataclasses import dataclass, field
from typing import Literal

AgentRole = Literal["aggregator"]
Nexus = Literal["energy_mobility"]
Justice = Literal[
    "distributive",
    "procedural",
    "recognitional",
    "intertemporal",
]


@dataclass(frozen=True)
class GymEnvConfig:
    """Configuration for the Gym wrapper.

    The high-level declaration `(agent_role, nexus, justice)` defines the RL
    formulation. This release supports energy-mobility problems.
    """

    agent_role: AgentRole = "aggregator"
    nexus: Nexus = "energy_mobility"
    justice: tuple[Justice, ...] = field(default_factory=tuple)

    @property
    def formulation_id(self) -> str:
        if self.justice == ():
            return self.nexus
        return "_".join((self.nexus, *self.justice))
