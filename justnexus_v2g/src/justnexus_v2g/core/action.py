from dataclasses import dataclass


@dataclass
class SimulatorAction:
    """Top-level simulator action.

    ev_power_commands_kw[i] is the requested EV charging/discharging power
    for household i.
    """
    ev_power_commands_kw: list[float]
