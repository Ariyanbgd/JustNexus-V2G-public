from justnexus_v2g.controllers.base import BaseController
from justnexus_v2g.core.action import SimulatorAction
from justnexus_v2g.core.state import SimulatorState


class AlwaysChargeController(BaseController):
    """Simple heuristic: request constant charging for connected EVs."""

    def __init__(self, charge_power_kw: float = 4.0) -> None:
        self.charge_power_kw = charge_power_kw

    def act(self, state: SimulatorState) -> SimulatorAction:
        commands = []
        for household in state.households:
            if household.ev_connected:
                commands.append(self.charge_power_kw)
            else:
                commands.append(0.0)
        return SimulatorAction(ev_power_commands_kw=commands)


class DepartureAwareController(BaseController):
    """Charge only when energy is below the required departure level.

    If departure is close, use urgent charging power.
    Otherwise use normal charging power.
    """

    def __init__(
        self,
        normal_charge_power_kw: float = 4.0,
        urgent_charge_power_kw: float = 7.0,
        urgent_horizon_hours: float = 2.0,
    ) -> None:
        self.normal_charge_power_kw = normal_charge_power_kw
        self.urgent_charge_power_kw = urgent_charge_power_kw
        self.urgent_horizon_hours = urgent_horizon_hours

    def act(self, state: SimulatorState) -> SimulatorAction:
        commands = []

        for household in state.households:
            if not household.ev_connected:
                commands.append(0.0)
                continue

            energy_gap = (
                household.required_departure_energy_kwh - household.ev_energy_kwh
            )

            if energy_gap <= 0.0:
                commands.append(0.0)
                continue

            if (
                household.time_to_departure_hours > 0.0
                and household.time_to_departure_hours <= self.urgent_horizon_hours
            ):
                commands.append(self.urgent_charge_power_kw)
            else:
                commands.append(self.normal_charge_power_kw)

        return SimulatorAction(ev_power_commands_kw=commands)


class PriceAwareController(BaseController):
    """Charge when tariffs are low, avoid charging during peak prices.

    Decides on two thresholds: normal charging when below high_price_threshold,
    urgent charging when below low_price_threshold.
    """

    def __init__(
        self,
        normal_charge_power_kw: float = 4.0,
        urgent_charge_power_kw: float = 7.0,
        high_price_threshold_eur_per_kwh: float = 0.30,
        low_price_threshold_eur_per_kwh: float = 0.20,
    ) -> None:
        self.normal_charge_power_kw = normal_charge_power_kw
        self.urgent_charge_power_kw = urgent_charge_power_kw
        self.high_price_threshold_eur_per_kwh = high_price_threshold_eur_per_kwh
        self.low_price_threshold_eur_per_kwh = low_price_threshold_eur_per_kwh

    def act(self, state: SimulatorState) -> SimulatorAction:
        commands = []
        tariff = state.system.import_tariff_eur_per_kwh

        for household in state.households:
            if not household.ev_connected:
                commands.append(0.0)
                continue

            energy_gap = (
                household.required_departure_energy_kwh - household.ev_energy_kwh
            )

            if energy_gap <= 0.0:
                commands.append(0.0)
                continue

            if tariff <= self.low_price_threshold_eur_per_kwh:
                commands.append(self.urgent_charge_power_kw)
            elif tariff <= self.high_price_threshold_eur_per_kwh:
                commands.append(self.normal_charge_power_kw)
            else:
                commands.append(0.0)

        return SimulatorAction(ev_power_commands_kw=commands)


class EnergyAwareController(BaseController):
    """Energy-only heuristic for cost-aware, grid-limited charging.

    The controller ignores mobility readiness and household-specific justice
    metadata. It charges connected EVs only when import prices are low and
    aggregate grid headroom is available.
    """

    def __init__(
        self,
        charge_power_kw: float = 7.0,
        low_price_threshold_eur_per_kwh: float = 0.20,
        target_grid_usage_kw: float | None = None,
    ) -> None:
        self.charge_power_kw = charge_power_kw
        self.low_price_threshold_eur_per_kwh = low_price_threshold_eur_per_kwh
        self.target_grid_usage_kw = target_grid_usage_kw

    def act(self, state: SimulatorState) -> SimulatorAction:
        connected = [household.ev_connected for household in state.households]
        n_connected = sum(connected)
        if n_connected == 0:
            return SimulatorAction(ev_power_commands_kw=[0.0] * len(state.households))

        if (
            state.system.import_tariff_eur_per_kwh
            > self.low_price_threshold_eur_per_kwh
        ):
            return SimulatorAction(ev_power_commands_kw=[0.0] * len(state.households))

        target_grid_usage = (
            self.target_grid_usage_kw
            if self.target_grid_usage_kw is not None
            else max(state.system.grid_exchange_kw, 0.0)
        )
        current_non_ev_grid_kw = (
            state.system.grid_exchange_kw - state.system.total_ev_kw
        )
        available_ev_power_kw = max(0.0, target_grid_usage - current_non_ev_grid_kw)
        power_per_household = min(
            self.charge_power_kw, available_ev_power_kw / n_connected
        )

        return SimulatorAction(
            ev_power_commands_kw=[
                power_per_household if is_connected else 0.0
                for is_connected in connected
            ]
        )


class GridAwareController(BaseController):
    """Coordinate charging to reduce grid congestion.

    Limits total charging request to stay under a target grid usage.
    Prioritizes households with lower battery levels.
    """

    def __init__(
        self,
        charge_power_kw: float = 7.0,
        target_grid_usage_kw: float = 5.5,
    ) -> None:
        self.charge_power_kw = charge_power_kw
        self.target_grid_usage_kw = target_grid_usage_kw

    def act(self, state: SimulatorState) -> SimulatorAction:
        connected_households = [h for h in state.households if h.ev_connected]

        if not connected_households:
            return SimulatorAction(ev_power_commands_kw=[0.0] * len(state.households))

        total_requested = self.charge_power_kw * len(connected_households)

        if total_requested <= self.target_grid_usage_kw:
            commands = [
                self.charge_power_kw if h.ev_connected else 0.0
                for h in state.households
            ]
        else:
            budget_per_household = self.target_grid_usage_kw / len(connected_households)
            commands = [
                (
                    min(self.charge_power_kw, budget_per_household)
                    if h.ev_connected
                    else 0.0
                )
                for h in state.households
            ]

        return SimulatorAction(ev_power_commands_kw=commands)
