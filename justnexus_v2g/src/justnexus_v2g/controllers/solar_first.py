"""Idle and deterministic solar-first baselines for storage experiments."""

from justnexus_v2g.controllers.base import BaseController
from justnexus_v2g.core.action import SimulatorAction
from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.core.state import SimulatorState
from justnexus_v2g.exogenous.demand import household_demand_profile
from justnexus_v2g.exogenous.pv import household_pv_profile
from justnexus_v2g.exogenous.mobility import ev_connected_profile
from justnexus_v2g.exogenous.tariffs import (
    export_tariff_profile,
    import_tariff_profile,
)


class IdleController(BaseController):
    """No battery intervention: request zero charging and discharging."""

    def act(self, state: SimulatorState) -> SimulatorAction:
        return SimulatorAction(ev_power_commands_kw=[0.0] * len(state.households))


class MaxPriceArbitrageController(BaseController):
    """Use each connected EV at its feasible limit based only on settlement price.

    The controller charges at the maximum feasible rate at low prices and
    discharges at the maximum feasible rate at high prices. It deliberately
    ignores demand, PV, grid limits, and household preferences. Physical
    battery state is used only to avoid infeasible requests.
    """

    def __init__(
        self,
        params: SimulationParams,
        *,
        low_price_threshold_eur_per_kwh: float = 0.20,
        high_price_threshold_eur_per_kwh: float = 0.30,
    ) -> None:
        self.params = params
        self.low_price_threshold_eur_per_kwh = low_price_threshold_eur_per_kwh
        self.high_price_threshold_eur_per_kwh = high_price_threshold_eur_per_kwh

    def act(self, state: SimulatorState) -> SimulatorAction:
        params = self.params
        t = state.t + 1
        tariff = import_tariff_profile(params, t)
        if tariff <= self.low_price_threshold_eur_per_kwh:
            direction = 1.0
        elif tariff >= self.high_price_threshold_eur_per_kwh:
            direction = -1.0
        else:
            return SimulatorAction(ev_power_commands_kw=[0.0] * len(state.households))

        commands = []
        for i, (household, household_params) in enumerate(
            zip(state.households, params.households, strict=True)
        ):
            ev = household_params.ev
            if not ev_connected_profile(params, i, t):
                commands.append(0.0)
            elif direction > 0:
                commands.append(
                    min(
                        ev.max_charge_kw,
                        max(ev.battery_capacity_kwh - household.ev_energy_kwh, 0.0)
                        / (ev.charge_efficiency * params.dt_hours),
                    )
                )
            else:
                commands.append(
                    -min(
                        ev.max_discharge_kw,
                        household.ev_energy_kwh * ev.discharge_efficiency
                        / params.dt_hours,
                    )
                )
        return SimulatorAction(ev_power_commands_kw=commands)


class PriceGridTradeoffController(BaseController):
    """Transparent price arbitrage with a bounded charging grid target.

    The controller uses deterministic next-step load, PV, connection and tariff
    profiles. It first absorbs household PV surplus, then permits profitable
    grid charging up to a configurable aggregate import target. Stored energy is
    discharged when the current export tariff is no lower than every remaining
    tariff in the episode, which also liquidates residual energy at the end.
    """

    def __init__(
        self,
        params: SimulationParams,
        *,
        low_price_threshold_eur_per_kwh: float = 0.20,
        charging_grid_target_kw: float = 12.0,
    ) -> None:
        self.params = params
        self.low_price_threshold_eur_per_kwh = low_price_threshold_eur_per_kwh
        self.charging_grid_target_kw = charging_grid_target_kw

    def _future_export_tariff(self, t: int) -> float:
        remaining = range(t + 1, self.params.episode_steps)
        return max(
            (export_tariff_profile(self.params, step) for step in remaining),
            default=export_tariff_profile(self.params, t),
        )

    def act(self, state: SimulatorState) -> SimulatorAction:
        params = self.params
        t = state.t + 1
        import_tariff = import_tariff_profile(params, t)
        export_tariff = export_tariff_profile(params, t)
        future_export_tariff = self._future_export_tariff(t)

        remaining_steps = range(t + 1, params.episode_steps)
        peak_steps_remaining = sum(
            abs(export_tariff_profile(params, step) - future_export_tariff) < 1e-12
            for step in remaining_steps
        )

        feasible_charge = []
        feasible_discharge = []
        household_net = []
        for i, (household, household_params) in enumerate(
            zip(state.households, params.households, strict=True)
        ):
            ev = household_params.ev
            connected = ev_connected_profile(params, i, t)
            peak_usable_energy_kwh = min(
                ev.battery_capacity_kwh,
                peak_steps_remaining
                * ev.max_discharge_kw
                * params.dt_hours
                / ev.discharge_efficiency,
            )
            feasible_charge.append(
                min(
                    ev.max_charge_kw,
                    max(peak_usable_energy_kwh - household.ev_energy_kwh, 0.0)
                    / (ev.charge_efficiency * params.dt_hours),
                )
                if connected
                else 0.0
            )
            feasible_discharge.append(
                min(
                    ev.max_discharge_kw,
                    household.ev_energy_kwh * ev.discharge_efficiency
                    / params.dt_hours,
                )
                if connected
                else 0.0
            )
            household_net.append(
                household_demand_profile(params, i, t)
                - household_pv_profile(params, i, t)
            )

        # Sell stored energy at the best remaining tariff. Equality makes the
        # rule discharge throughout a flat peak and liquidate after the peak.
        if export_tariff >= future_export_tariff:
            return SimulatorAction(
                ev_power_commands_kw=[-power for power in feasible_discharge]
            )

        profitable = []
        for hp in params.households:
            ev = hp.ev
            future_grid_value = (
                future_export_tariff * ev.charge_efficiency * ev.discharge_efficiency
            )
            profitable.append(future_grid_value > import_tariff)

        commands = [0.0] * params.n_households
        for i in range(params.n_households):
            if not profitable[i]:
                continue
            commands[i] = min(feasible_charge[i], max(-household_net[i], 0.0))

        grid_after_solar_charge = sum(household_net) + sum(commands)
        grid_target = (
            self.charging_grid_target_kw
            if import_tariff <= self.low_price_threshold_eur_per_kwh
            else params.system.max_import_kw
        )
        grid_charge_budget = max(grid_target - grid_after_solar_charge, 0.0)
        remaining = [
            max(feasible_charge[i] - commands[i], 0.0) if profitable[i] else 0.0
            for i in range(params.n_households)
        ]
        total_remaining = sum(remaining)
        if total_remaining > 0.0 and grid_charge_budget > 0.0:
            scale = min(grid_charge_budget / total_remaining, 1.0)
            commands = [
                command + available * scale
                for command, available in zip(commands, remaining, strict=True)
            ]
        return SimulatorAction(ev_power_commands_kw=commands)


class SolarFirstController(BaseController):
    """Store solar surplus and reserve it for import-capacity shortages.

    Uses known deterministic next-step demand/PV/connection profiles and per-EV
    energy, unlike PPO's aggregate observation. No tariff or future-day forecast.
    The simulator applies actions at state.t + 1. Optional authority enforcement
    matches the Gym procedural layer's decision-time authority convention.
    """

    def __init__(self, params: SimulationParams, *, respect_authority: bool = False):
        self.params = params
        self.respect_authority = respect_authority

    def act(self, state: SimulatorState) -> SimulatorAction:
        params = self.params
        t = state.t + 1
        net = sum(household_demand_profile(params, i, t)
                  - household_pv_profile(params, i, t)
                  for i in range(params.n_households))
        discharge = net > params.system.max_import_kw
        budget = net - params.system.max_import_kw if discharge else max(-net, 0.0)
        capacities = []
        for i, (h, hp) in enumerate(zip(state.households, params.households, strict=True)):
            ev = hp.ev
            if not ev_connected_profile(params, i, t):
                capacity = 0.0
            elif discharge:
                capacity = min(ev.max_discharge_kw,
                               h.ev_energy_kwh * ev.discharge_efficiency / params.dt_hours)
            else:
                capacity = min(ev.max_charge_kw,
                               max(ev.battery_capacity_kwh - h.ev_energy_kwh, 0.0)
                               / (ev.charge_efficiency * params.dt_hours))
            capacities.append(max(capacity, 0.0))
        if self.respect_authority:
            # Local import avoids a controller/procedural module import cycle.
            from justnexus_v2g.gym_env.procedural import apply_procedural_rules
            signed = [-v if discharge else v for v in capacities]
            admissible, _ = apply_procedural_rules(signed, params, state)
            capacities = [abs(v) for v in admissible]
        total = sum(capacities)
        scale = min(budget / total, 1.0) if total else 0.0
        sign = -1.0 if discharge else 1.0
        return SimulatorAction(ev_power_commands_kw=[sign * v * scale for v in capacities])
