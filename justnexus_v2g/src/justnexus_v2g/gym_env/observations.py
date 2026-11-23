from __future__ import annotations

import numpy as np
from gymnasium import spaces

from justnexus_v2g.core.params import HouseholdParams, SimulationParams
from justnexus_v2g.core.state import HouseholdState, SimulatorState
from justnexus_v2g.gym_env.formulations import FormulationSpec, build_formulation
from justnexus_v2g.gym_env.procedural import is_authority_available
from justnexus_v2g.gym_env.specs import GymEnvConfig


def observation_size(
    params: SimulationParams,
    config: GymEnvConfig | None = None,
    formulation: FormulationSpec | None = None,
) -> int:
    formulation = formulation or build_formulation(config or GymEnvConfig())
    return len(formulation.system_observation_fields) + (
        params.n_households * len(formulation.household_observation_fields)
    )


def make_observation_space(
    params: SimulationParams,
    config: GymEnvConfig | None = None,
    formulation: FormulationSpec | None = None,
) -> spaces.Box:
    size = observation_size(params, config, formulation)
    return spaces.Box(
        low=np.full(size, -np.inf, dtype=np.float32),
        high=np.full(size, np.inf, dtype=np.float32),
        dtype=np.float32,
    )


def state_to_observation(
    state: SimulatorState,
    config: GymEnvConfig | None = None,
    formulation: FormulationSpec | None = None,
    params: SimulationParams | None = None,
    distribution_context: dict[str, float] | None = None,
    intertemporal_context: dict[str, object] | None = None,
) -> np.ndarray:
    formulation = formulation or build_formulation(config or GymEnvConfig())
    values: list[float] = []

    for field in formulation.system_observation_fields:
        values.append(
            _system_field_value(
                state,
                field,
                params,
                distribution_context,
                intertemporal_context,
            )
        )

    for i, household in enumerate(state.households):
        household_params = params.households[i] if params is not None else None
        for field in formulation.household_observation_fields:
            values.append(
                _household_field_value(
                    household, field, household_params, state, params,
                    household_index=i,
                    intertemporal_context=intertemporal_context,
                )
            )

    return np.asarray(values, dtype=np.float32)


def _system_field_value(
    state: SimulatorState,
    field: str,
    params: SimulationParams | None = None,
    distribution_context: dict[str, float] | None = None,
    intertemporal_context: dict[str, object] | None = None,
) -> float:
    if field == "time_sin":
        return float(np.sin(_time_angle(state, params)))
    if field == "time_cos":
        return float(np.cos(_time_angle(state, params)))
    if field.startswith("distribution_"):
        return float((distribution_context or {}).get(field, 0.0))
    if field == "requested_total_ev_power_kw":
        return float(sum(h.requested_ev_power_kw for h in state.households))
    if field == "applied_total_ev_power_kw":
        return float(sum(h.applied_ev_power_kw for h in state.households))
    return float(getattr(state.system, field))


def _time_angle(state: SimulatorState, params: SimulationParams | None) -> float:
    if params is None:
        return 0.0
    period_steps = params.cycle_steps or params.episode_steps
    if period_steps <= 0:
        return 0.0
    cycle_t = state.t % period_steps
    return float(2.0 * np.pi * cycle_t / period_steps)


def _household_field_value(
    household: HouseholdState,
    field: str,
    household_params: HouseholdParams | None = None,
    state: SimulatorState | None = None,
    params: SimulationParams | None = None,
    household_index: int | None = None,
    intertemporal_context: dict[str, object] | None = None,
) -> float:
    ev_fields = {
        "battery_capacity_kwh",
        "max_charge_kw",
        "max_discharge_kw",
    }
    procedural_fields = {
        "authority_available",
    }

    if field in {
        "remaining_daily_throughput_budget_fraction",
        "battery_reserve_margin_fraction",
    }:
        if household_index is None:
            raise ValueError(f"Intertemporal field {field!r} requires household index.")
        values = (intertemporal_context or {}).get(field, [])
        if not isinstance(values, list) or household_index >= len(values):
            return 0.0
        return float(values[household_index])

    if field in ev_fields:
        if household_params is None:
            raise ValueError(f"Household parameter field {field!r} requires params.")
        return float(getattr(household_params.ev, field))

    if field in procedural_fields:
        if household_params is None:
            raise ValueError(f"Procedural field {field!r} requires params.")
        if field == "authority_available":
            return float(
                is_authority_available(
                    household_params.procedural.authority_mode,
                    household_params,
                    state.t if state is not None else None,
                    (params.cycle_steps or params.episode_steps) if params is not None else None,
                )
            )

    if field == "time_to_departure_hours" and not household.ev_connected:
        return 0.0

    if field == "ev_soc_fraction":
        if household_params is None:
            raise ValueError(f"Household parameter field {field!r} requires params.")
        capacity = household_params.ev.battery_capacity_kwh
        if capacity <= 0:
            return 0.0
        return float(np.clip(household.ev_energy_kwh / capacity, 0.0, 1.0))

    value = getattr(household, field)
    return float(value)
