import numpy as np

from justnexus_v2g.core.action import SimulatorAction
from justnexus_v2g.core.params import HouseholdParams, SimulationParams
from justnexus_v2g.core.state import HouseholdState, SimulatorState, SystemState
from justnexus_v2g.core.types import ConstraintReport, StepResult
from justnexus_v2g.exogenous.demand import household_demand_profile
from justnexus_v2g.exogenous.mobility import (
    ev_connected_profile,
    required_departure_energy_profile,
    time_to_departure_profile,
    trip_energy_profile,
)
from justnexus_v2g.exogenous.pv import household_pv_profile
from justnexus_v2g.exogenous.tariffs import (
    export_tariff_profile,
    import_tariff_profile,
)


class Simulator:
    """Simulator with local and shared-grid constraints."""

    MIN_SERVICE_POWER_KW = 1.0

    def __init__(self, params: SimulationParams, *, enforce_grid_limits: bool = True) -> None:
        if params.n_households != len(params.households):
            raise ValueError(
                "n_households must match the number of household parameter sets."
            )
        self.params = params
        self.enforce_grid_limits = enforce_grid_limits
        self.grid_allocation_strategy = "pro_rata" if enforce_grid_limits else "none"
        self.cumulative_grid_clipping_burden_kwh = [0.0] * params.n_households
        self.state: SimulatorState | None = None
        self.is_done = False
        self.rng = np.random.default_rng()
        self.last_seed: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> SimulatorState:
        """Reset the simulator and install the episode RNG.

        Current exogenous profiles are deterministic, but they receive this
        RNG so stochastic profile implementations can be reproducible under
        the Gymnasium reset seed contract.
        """
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        self.last_seed = seed
        self.cumulative_grid_clipping_burden_kwh = [0.0] * self.params.n_households
        t = 0

        candidates = [
            self._make_household_candidate(
                household_index=i,
                household_params=household_params,
                t=t,
                previous_energy_kwh=self._initial_energy_kwh(household_params),
                previous_connected=True,
                requested_ev_power_kw=0.0,
            )
            for i, household_params in enumerate(self.params.households)
        ]

        households = [
            self._finalize_household_state(
                candidate=candidates[i],
                household_params=self.params.households[i],
                previous_battery_throughput_kwh=0.0,
                previous_discharge_burden_kwh=0.0,
                previous_deferral_count=0,
            )
            for i in range(self.params.n_households)
        ]

        self._balance_household_service(households, t=t)
        system = self._make_system_state(households, t=t)
        self.state = SimulatorState(t=t, households=households, system=system)
        self.is_done = self.state.t >= self.params.episode_steps - 1
        return self.state


    def _initial_energy_kwh(self, household_params: HouseholdParams) -> float:
        ev = household_params.ev
        if ev.initial_energy_min_kwh is None and ev.initial_energy_max_kwh is None:
            return float(ev.initial_energy_kwh)
        if ev.initial_energy_min_kwh is None or ev.initial_energy_max_kwh is None:
            raise ValueError(
                "Both initial_energy_min_kwh and initial_energy_max_kwh are required."
            )
        low = float(ev.initial_energy_min_kwh)
        high = float(ev.initial_energy_max_kwh)
        if low > high:
            raise ValueError("initial energy minimum cannot exceed maximum.")
        low = max(0.0, low)
        high = min(float(ev.battery_capacity_kwh), high)
        return float(self.rng.uniform(low, high))

    def step(self, action: SimulatorAction) -> StepResult:
        if self.state is None:
            raise RuntimeError("Simulator must be reset before step() is called.")

        if self.is_done:
            raise RuntimeError(
                "Episode is already finished. Call reset() before stepping again."
            )

        if len(action.ev_power_commands_kw) != self.params.n_households:
            raise ValueError("Action length must match the number of households.")

        next_t = self.state.t + 1

        candidates: list[dict] = []
        clipped_by_connection: list[bool] = []
        clipped_by_power_limit: list[bool] = []
        clipped_by_energy_limit: list[bool] = []

        for i, household_params in enumerate(self.params.households):
            previous_household = self.state.households[i]
            candidate = self._make_household_candidate(
                household_index=i,
                household_params=household_params,
                t=next_t,
                previous_energy_kwh=previous_household.ev_energy_kwh,
                previous_connected=previous_household.ev_connected,
                requested_ev_power_kw=action.ev_power_commands_kw[i],
            )
            candidates.append(candidate)

            requested = action.ev_power_commands_kw[i]
            local_power_limit_value = self._clip_by_connection_and_power_limit(
                requested_ev_power_kw=requested,
                ev_connected=candidate["ev_connected"],
                max_charge_kw=household_params.ev.max_charge_kw,
                max_discharge_kw=household_params.ev.max_discharge_kw,
            )

            was_disconnected = (not candidate["ev_connected"]) and (requested != 0.0)
            was_power_clipped = candidate["ev_connected"] and (
                requested != local_power_limit_value
            )
            was_energy_clipped = candidate["ev_connected"] and (
                local_power_limit_value != candidate["local_ev_power_kw"]
            )

            clipped_by_connection.append(was_disconnected)
            clipped_by_power_limit.append(was_power_clipped)
            clipped_by_energy_limit.append(was_energy_clipped)

        pre_grid_total_ev_kw = sum(c["local_ev_power_kw"] for c in candidates)
        self._enforce_grid_constraints(candidates)
        self._update_grid_clipping_burden(candidates)
        post_grid_total_ev_kw = sum(c["final_ev_power_kw"] for c in candidates)
        grid_constrained = pre_grid_total_ev_kw != post_grid_total_ev_kw

        households = []
        for i, household_params in enumerate(self.params.households):
            previous_household = self.state.households[i]
            households.append(
                self._finalize_household_state(
                    candidate=candidates[i],
                    household_params=household_params,
                    previous_battery_throughput_kwh=previous_household.battery_throughput_kwh,
                    previous_discharge_burden_kwh=previous_household.discharge_burden_kwh,
                    previous_deferral_count=previous_household.deferral_count,
                )
            )

        self._balance_household_service(households, t=next_t)
        system = self._make_system_state(households, t=next_t)
        self.state = SimulatorState(t=next_t, households=households, system=system)

        done = next_t >= self.params.episode_steps - 1
        self.is_done = done

        constraints = ConstraintReport(
            requested_ev_power_kw=list(action.ev_power_commands_kw),
            applied_ev_power_kw=[h.applied_ev_power_kw for h in households],
            clipped_by_connection=clipped_by_connection,
            clipped_by_power_limit=clipped_by_power_limit,
            clipped_by_energy_limit=clipped_by_energy_limit,
            grid_constrained=grid_constrained,
            pre_grid_total_ev_kw=pre_grid_total_ev_kw,
            post_grid_total_ev_kw=post_grid_total_ev_kw,
        )

        return StepResult(
            state=self.state,
            action=action,
            done=done,
            constraints=constraints,
        )

    def run_episode(self, controller, recorder=None) -> list[StepResult]:
        if self.state is None:
            self.reset()

        results: list[StepResult] = []
        while not self.is_done:
            assert self.state is not None
            action = controller.act(self.state)
            result = self.step(action)
            results.append(result)

            if recorder is not None:
                recorder.record(result)

        return results

    def _make_household_candidate(
        self,
        household_index: int,
        household_params: HouseholdParams,
        t: int,
        previous_energy_kwh: float,
        previous_connected: bool,
        requested_ev_power_kw: float,
    ) -> dict:
        battery_capacity = household_params.ev.battery_capacity_kwh
        max_charge_kw = household_params.ev.max_charge_kw
        max_discharge_kw = household_params.ev.max_discharge_kw
        charge_efficiency = household_params.ev.charge_efficiency
        discharge_efficiency = household_params.ev.discharge_efficiency

        load_kw = household_demand_profile(
            self.params, household_index, t, rng=self.rng
        )
        pv_kw = household_pv_profile(self.params, household_index, t, rng=self.rng)
        ev_connected = ev_connected_profile(
            self.params, household_index, t, rng=self.rng
        )
        time_to_departure_hours = time_to_departure_profile(
            self.params, household_index, t, rng=self.rng
        )
        required_departure_energy_kwh = required_departure_energy_profile(
            self.params, household_index, t, rng=self.rng
        )

        power_after_connection_and_limits = self._clip_by_connection_and_power_limit(
            requested_ev_power_kw=requested_ev_power_kw,
            ev_connected=ev_connected,
            max_charge_kw=max_charge_kw,
            max_discharge_kw=max_discharge_kw,
        )

        local_ev_power_kw = self._clip_by_energy_feasibility(
            requested_ev_power_kw=power_after_connection_and_limits,
            previous_energy_kwh=previous_energy_kwh,
            battery_capacity_kwh=battery_capacity,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
        )

        trip_energy_kwh = 0.0
        if previous_connected and (not ev_connected):
            trip_energy_kwh = trip_energy_profile(
                self.params, household_index, t, rng=self.rng
            )

        return {
            "t": t,
            "household_index": household_index,
            "requested_ev_power_kw": requested_ev_power_kw,
            "local_ev_power_kw": local_ev_power_kw,
            "final_ev_power_kw": local_ev_power_kw,
            "previous_energy_kwh": previous_energy_kwh,
            "ev_connected": ev_connected,
            "time_to_departure_hours": time_to_departure_hours,
            "required_departure_energy_kwh": required_departure_energy_kwh,
            "load_kw": load_kw,
            "pv_kw": pv_kw,
            "trip_energy_kwh": trip_energy_kwh,
        }

    def _finalize_household_state(
        self,
        candidate: dict,
        household_params: HouseholdParams,
        previous_battery_throughput_kwh: float,
        previous_discharge_burden_kwh: float,
        previous_deferral_count: int,
    ) -> HouseholdState:
        battery_capacity = household_params.ev.battery_capacity_kwh
        charge_efficiency = household_params.ev.charge_efficiency
        discharge_efficiency = household_params.ev.discharge_efficiency

        next_energy_kwh = self._update_battery_energy(
            previous_energy_kwh=candidate["previous_energy_kwh"],
            ev_power_kw=candidate["final_ev_power_kw"],
            battery_capacity_kwh=battery_capacity,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
        )

        departure_energy_kwh = 0.0
        departure_energy_shortfall_kwh = 0.0
        if candidate["trip_energy_kwh"] > 0.0:
            departure_energy_kwh = next_energy_kwh
            required = household_params.mobility.required_departure_energy_kwh
            if required is None:
                required = candidate["trip_energy_kwh"]
            departure_energy_shortfall_kwh = max(
                float(required) - departure_energy_kwh, 0.0
            )
            next_energy_kwh = max(0.0, next_energy_kwh - candidate["trip_energy_kwh"])

        battery_throughput_kwh = (
            previous_battery_throughput_kwh
            + abs(candidate["final_ev_power_kw"]) * self.params.dt_hours
        )
        discharge_burden_kwh = (
            previous_discharge_burden_kwh
            + max(-candidate["final_ev_power_kw"], 0.0) * self.params.dt_hours
        )

        was_deferred = (
            candidate["ev_connected"]
            and candidate["required_departure_energy_kwh"] > next_energy_kwh
            and candidate["requested_ev_power_kw"] > 0.0
            and candidate["final_ev_power_kw"] < self.MIN_SERVICE_POWER_KW
        )
        deferral_count = previous_deferral_count + int(was_deferred)

        import_tariff = import_tariff_profile(self.params, candidate["t"])
        export_tariff = export_tariff_profile(self.params, candidate["t"])

        net_household_power_kw = (
            candidate["load_kw"] + candidate["final_ev_power_kw"] - candidate["pv_kw"]
        )
        imported_energy_kwh = max(net_household_power_kw, 0.0) * self.params.dt_hours
        exported_energy_kwh = max(-net_household_power_kw, 0.0) * self.params.dt_hours
        import_cost_eur = imported_energy_kwh * import_tariff
        export_revenue_eur = exported_energy_kwh * export_tariff
        net_energy_cost_eur = import_cost_eur - export_revenue_eur

        return HouseholdState(
            ev_connected=candidate["ev_connected"],
            ev_energy_kwh=next_energy_kwh,
            time_to_departure_hours=candidate["time_to_departure_hours"],
            required_departure_energy_kwh=candidate["required_departure_energy_kwh"],
            load_kw=candidate["load_kw"],
            pv_kw=candidate["pv_kw"],
            requested_ev_power_kw=candidate["requested_ev_power_kw"],
            applied_ev_power_kw=candidate["final_ev_power_kw"],
            trip_energy_kwh=candidate["trip_energy_kwh"],
            battery_throughput_kwh=battery_throughput_kwh,
            discharge_burden_kwh=discharge_burden_kwh,
            was_deferred=was_deferred,
            deferral_count=deferral_count,
            net_household_power_kw=net_household_power_kw,
            imported_energy_kwh=imported_energy_kwh,
            exported_energy_kwh=exported_energy_kwh,
            import_cost_eur=import_cost_eur,
            export_revenue_eur=export_revenue_eur,
            net_energy_cost_eur=net_energy_cost_eur,
            departure_energy_kwh=departure_energy_kwh,
            departure_energy_shortfall_kwh=departure_energy_shortfall_kwh,
        )

    def _enforce_grid_constraints(self, candidates: list[dict]) -> None:
        if not self.enforce_grid_limits:
            return
        total_load_kw = sum(c["load_kw"] for c in candidates)
        total_pv_kw = sum(c["pv_kw"] for c in candidates)
        total_ev_kw = sum(c["local_ev_power_kw"] for c in candidates)

        grid_exchange_kw = total_load_kw + total_ev_kw - total_pv_kw

        max_import_kw = self.params.system.max_import_kw
        max_export_kw = self.params.system.max_export_kw

        if grid_exchange_kw > max_import_kw:
            allowed_total_ev_kw = max_import_kw - total_load_kw + total_pv_kw
            self._scale_positive_ev_power(candidates, allowed_total_ev_kw)
            return

        if grid_exchange_kw < -max_export_kw:
            allowed_total_ev_kw = -max_export_kw - total_load_kw + total_pv_kw
            self._scale_negative_ev_power(candidates, allowed_total_ev_kw)
            return

    def _scale_positive_ev_power(
        self,
        candidates: list[dict],
        allowed_total_ev_kw: float,
    ) -> None:
        self._allocate_pro_rata_positive_power(candidates, allowed_total_ev_kw)

    def _scale_negative_ev_power(
        self,
        candidates: list[dict],
        allowed_total_ev_kw: float,
    ) -> None:
        self._allocate_pro_rata_negative_power(candidates, allowed_total_ev_kw)

    def _allocate_pro_rata_positive_power(
        self,
        candidates: list[dict],
        allowed_total_ev_kw: float,
    ) -> None:
        positive_total = sum(max(c["local_ev_power_kw"], 0.0) for c in candidates)

        if positive_total <= 0.0:
            return

        negative_total = sum(min(c["local_ev_power_kw"], 0.0) for c in candidates)
        target_positive_total = max(0.0, allowed_total_ev_kw - negative_total)
        scale = min(1.0, target_positive_total / positive_total)

        for candidate in candidates:
            if candidate["local_ev_power_kw"] > 0.0:
                candidate["final_ev_power_kw"] = candidate["local_ev_power_kw"] * scale
            else:
                candidate["final_ev_power_kw"] = candidate["local_ev_power_kw"]

    def _allocate_pro_rata_negative_power(
        self,
        candidates: list[dict],
        allowed_total_ev_kw: float,
    ) -> None:
        negative_total = sum(min(c["local_ev_power_kw"], 0.0) for c in candidates)

        if negative_total >= 0.0:
            return

        positive_total = sum(max(c["local_ev_power_kw"], 0.0) for c in candidates)
        target_negative_total = min(0.0, allowed_total_ev_kw - positive_total)
        scale = min(1.0, target_negative_total / negative_total)

        for candidate in candidates:
            if candidate["local_ev_power_kw"] < 0.0:
                candidate["final_ev_power_kw"] = candidate["local_ev_power_kw"] * scale
            else:
                candidate["final_ev_power_kw"] = candidate["local_ev_power_kw"]

    def _update_grid_clipping_burden(self, candidates: list[dict]) -> None:
        for candidate in candidates:
            household_index = candidate["household_index"]
            clipping_kw = abs(
                candidate["local_ev_power_kw"] - candidate["final_ev_power_kw"]
            )
            self.cumulative_grid_clipping_burden_kwh[household_index] += (
                clipping_kw * self.params.dt_hours
            )

    def _clip_by_connection_and_power_limit(
        self,
        requested_ev_power_kw: float,
        ev_connected: bool,
        max_charge_kw: float,
        max_discharge_kw: float,
    ) -> float:
        if not ev_connected:
            return 0.0

        return max(-max_discharge_kw, min(max_charge_kw, requested_ev_power_kw))

    def _clip_by_energy_feasibility(
        self,
        requested_ev_power_kw: float,
        previous_energy_kwh: float,
        battery_capacity_kwh: float,
        charge_efficiency: float,
        discharge_efficiency: float,
    ) -> float:
        dt_hours = self.params.dt_hours

        if requested_ev_power_kw > 0.0:
            remaining_capacity_kwh = battery_capacity_kwh - previous_energy_kwh
            if remaining_capacity_kwh <= 0.0:
                return 0.0
            max_feasible_charge_kw = remaining_capacity_kwh / (
                dt_hours * charge_efficiency
            )
            return min(requested_ev_power_kw, max_feasible_charge_kw)

        if requested_ev_power_kw < 0.0:
            available_energy_kwh = previous_energy_kwh
            if available_energy_kwh <= 0.0:
                return 0.0
            max_feasible_discharge_kw = (
                available_energy_kwh * discharge_efficiency / dt_hours
            )
            return max(requested_ev_power_kw, -max_feasible_discharge_kw)

        return 0.0

    def _update_battery_energy(
        self,
        previous_energy_kwh: float,
        ev_power_kw: float,
        battery_capacity_kwh: float,
        charge_efficiency: float,
        discharge_efficiency: float,
    ) -> float:
        dt_hours = self.params.dt_hours

        if ev_power_kw >= 0.0:
            delta_energy_kwh = ev_power_kw * dt_hours * charge_efficiency
        else:
            delta_energy_kwh = ev_power_kw * dt_hours / discharge_efficiency

        next_energy_kwh = previous_energy_kwh + delta_energy_kwh
        return max(0.0, min(battery_capacity_kwh, next_energy_kwh))

    def _balance_household_service(self, households: list[HouseholdState], t: int) -> None:
        """Shed demand or curtail PV only after feasible battery dispatch."""
        if not self.enforce_grid_limits:
            return
        demand = sum(h.load_kw for h in households)
        pv = sum(h.pv_kw for h in households)
        pre = sum(h.net_household_power_kw for h in households)
        shed = max(pre - self.params.system.max_import_kw, 0.0)
        curtail = max(-self.params.system.max_export_kw - pre, 0.0)
        if shed > demand + 1e-9 or curtail > pv + 1e-9:
            raise RuntimeError("Battery projection left an infeasible service balance.")
        dt = self.params.dt_hours
        for h in households:
            shed_kw = shed * h.load_kw / demand if demand else 0.0
            curtailed_kw = curtail * h.pv_kw / pv if pv else 0.0
            h.unmet_demand_kwh = shed_kw * dt
            h.pv_curtailment_kwh = curtailed_kw * dt
            h.net_household_power_kw += curtailed_kw - shed_kw
            h.imported_energy_kwh = max(h.net_household_power_kw, 0.0) * dt
            h.exported_energy_kwh = max(-h.net_household_power_kw, 0.0) * dt
            h.import_cost_eur = h.imported_energy_kwh * import_tariff_profile(self.params, t)
            h.export_revenue_eur = h.exported_energy_kwh * export_tariff_profile(self.params, t)
            h.net_energy_cost_eur = h.import_cost_eur - h.export_revenue_eur

    def _make_system_state(
        self,
        households: list[HouseholdState],
        t: int,
    ) -> SystemState:
        total_load_kw = sum(h.load_kw for h in households)
        total_pv_kw = sum(h.pv_kw for h in households)
        total_ev_kw = sum(h.applied_ev_power_kw for h in households)
        pre_intervention_kw = total_load_kw + total_ev_kw - total_pv_kw
        grid_exchange_kw = sum(h.net_household_power_kw for h in households)

        import_tariff = import_tariff_profile(self.params, t, rng=self.rng)
        export_tariff = export_tariff_profile(self.params, t, rng=self.rng)

        imported_energy_kwh = max(grid_exchange_kw, 0.0) * self.params.dt_hours
        exported_energy_kwh = max(-grid_exchange_kw, 0.0) * self.params.dt_hours

        import_cost_eur = imported_energy_kwh * import_tariff
        export_revenue_eur = exported_energy_kwh * export_tariff
        net_energy_cost_eur = import_cost_eur - export_revenue_eur

        return SystemState(
            pre_intervention_grid_exchange_kw=pre_intervention_kw,
            unmet_demand_kwh=sum(h.unmet_demand_kwh for h in households),
            pv_curtailment_kwh=sum(h.pv_curtailment_kwh for h in households),
            total_load_kw=total_load_kw,
            total_pv_kw=total_pv_kw,
            total_ev_kw=total_ev_kw,
            grid_exchange_kw=grid_exchange_kw,
            import_tariff_eur_per_kwh=import_tariff,
            export_tariff_eur_per_kwh=export_tariff,
            imported_energy_kwh=imported_energy_kwh,
            exported_energy_kwh=exported_energy_kwh,
            import_cost_eur=import_cost_eur,
            export_revenue_eur=export_revenue_eur,
            net_energy_cost_eur=net_energy_cost_eur,
        )
