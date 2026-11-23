from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from gymnasium import Env, spaces

from justnexus_v2g.core.action import SimulatorAction
from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.core.types import StepResult
from justnexus_v2g.engine.simulator import Simulator
from justnexus_v2g.gym_env.formulations import FormulationSpec, build_formulation
from justnexus_v2g.gym_env.observations import (
    make_observation_space,
    state_to_observation,
)
from justnexus_v2g.gym_env.procedural import ProceduralReport, apply_procedural_rules
from justnexus_v2g.gym_env.rewards import (
    RewardWeights,
    RewardNormalization,
    calculate_reward,
)
from justnexus_v2g.gym_env.specs import GymEnvConfig


class JustNexusV2GEnv(Env[np.ndarray, np.ndarray]):
    """Gymnasium wrapper around the existing JustNexus V2G simulator.

    The first supported decision-maker is a centralized aggregator whose
    action is one requested EV power command per household.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: SimulationParams,
        config: GymEnvConfig | None = None,
        reward_weights: RewardWeights | None = None,
        reward_normalization: RewardNormalization | None = None,
        grid_operation: str = "hard_limits",
    ) -> None:
        self.params = params
        self.config = config or GymEnvConfig()
        self.formulation: FormulationSpec = build_formulation(self.config)
        if grid_operation not in {"hard_limits", "soft_import_unlimited_export"}:
            raise ValueError(f"Unknown grid_operation: {grid_operation}")
        unsupported_soft_grid_modifiers = set(
            self.formulation.applied_justice_modifiers
        ) - {"distributive", "procedural", "intertemporal"}
        if grid_operation != "hard_limits" and unsupported_soft_grid_modifiers:
            raise ValueError(
                "Soft import is supported only for base, distributive, procedural, and intertemporal energy."
            )
        self.grid_operation = grid_operation
        self.reward_weights = reward_weights or RewardWeights()
        self.reward_normalization = reward_normalization or RewardNormalization()
        unsupported_normalized_modifiers = set(
            self.formulation.applied_justice_modifiers
        ) - {"distributive", "procedural", "intertemporal"}
        if reward_normalization is not None and unsupported_normalized_modifiers:
            raise ValueError(
                "Reward normalization is supported only for base, distributive, procedural, and intertemporal energy."
            )

        if self.config.agent_role != "aggregator":
            raise ValueError("Only agent_role='aggregator' is currently supported.")

        self.simulator = Simulator(params, enforce_grid_limits=grid_operation == "hard_limits")
        self.last_step_result: StepResult | None = None
        self.household_departure_shortfall_kwh = [0.0] * params.n_households
        self.household_departed = [False] * params.n_households
        self.daily_household_battery_throughput_kwh = [0.0] * params.n_households
        self.daily_household_excess_battery_throughput_kwh = [0.0] * params.n_households
        self.observation_space = make_observation_space(
            params,
            formulation=self.formulation,
        )
        self.action_space = self._make_action_space(params)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        state = self.simulator.reset(seed=seed, rng=self.np_random)
        self.last_step_result = None
        self.household_departure_shortfall_kwh = [0.0] * self.params.n_households
        self.household_departed = [False] * self.params.n_households
        self.daily_household_battery_throughput_kwh = [0.0] * self.params.n_households
        self.daily_household_excess_battery_throughput_kwh = [
            0.0
        ] * self.params.n_households
        return (
            state_to_observation(
                state,
                formulation=self.formulation,
                params=self.params,
                distribution_context=self._distributive_observation_context(),
                intertemporal_context=self._intertemporal_observation_context(state),
            ),
            self._make_info(),
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_simulator_action = self._action_to_simulator_action(action)
        procedural_report = self._apply_procedural_layer(
            raw_simulator_action.ev_power_commands_kw
        )
        simulator_action = SimulatorAction(
            ev_power_commands_kw=procedural_report.authority_resolved_request_kw
        )
        result = self.simulator.step(simulator_action)
        self.last_step_result = result
        newly_departed = False
        for i, household in enumerate(result.state.households):
            if household.trip_energy_kwh > 0.0:
                self.household_departure_shortfall_kwh[i] = float(
                    household.departure_energy_shortfall_kwh
                )
                self.household_departed[i] = True
                newly_departed = True
        mobility_distribution_finalized = newly_departed and all(
            self.household_departed
        )
        daily_battery_report = self._update_daily_battery_throughput(result)
        reward, reward_terms = calculate_reward(
            result=result,
            params=self.params,
            weights=self.reward_weights,
            normalization=self.reward_normalization,
            config=self.config,
            formulation=self.formulation,
            procedural_report=procedural_report,
            daily_battery_report=daily_battery_report,
            household_departure_shortfall_kwh=(
                self.household_departure_shortfall_kwh
            ),
            mobility_distribution_finalized=mobility_distribution_finalized,
        )
        observation = state_to_observation(
            result.state,
            formulation=self.formulation,
            params=self.params,
            distribution_context=self._distributive_observation_context(),
            intertemporal_context=self._intertemporal_observation_context(result.state),
        )
        info = self._make_info(
            constraints=asdict(result.constraints),
            reward_terms=reward_terms,
            procedural_report=procedural_report,
        )
        terminated = result.done
        truncated = False
        return observation, reward, terminated, truncated, info

    def _make_action_space(self, params: SimulationParams) -> spaces.Box:
        lows = [-household.ev.max_discharge_kw for household in params.households]
        highs = [household.ev.max_charge_kw for household in params.households]
        return spaces.Box(
            low=np.asarray(lows, dtype=np.float32),
            high=np.asarray(highs, dtype=np.float32),
            dtype=np.float32,
        )

    def _update_daily_battery_throughput(
        self, result: StepResult
    ) -> dict[str, list[float] | float]:
        if self._is_first_step_of_daily_cycle(result.state.t):
            self.daily_household_battery_throughput_kwh = [
                0.0
            ] * self.params.n_households
            self.daily_household_excess_battery_throughput_kwh = [
                0.0
            ] * self.params.n_households

        limits = self._daily_battery_throughput_limits_kwh()
        step_throughput = [
            abs(power_kw) * self.params.dt_hours
            for power_kw in result.constraints.applied_ev_power_kw
        ]
        previous_excess = list(self.daily_household_excess_battery_throughput_kwh)
        daily_throughput = [
            previous + step
            for previous, step in zip(
                self.daily_household_battery_throughput_kwh,
                step_throughput,
                strict=True,
            )
        ]
        daily_excess = [
            max(value - limit, 0.0)
            for value, limit in zip(daily_throughput, limits, strict=True)
        ]
        excess_increment = [
            max(current - previous, 0.0)
            for current, previous in zip(daily_excess, previous_excess, strict=True)
        ]

        self.daily_household_battery_throughput_kwh = daily_throughput
        self.daily_household_excess_battery_throughput_kwh = daily_excess

        return {
            "step_battery_throughput_kwh_by_household": step_throughput,
            "daily_battery_throughput_kwh_by_household": daily_throughput,
            "daily_battery_throughput_limit_kwh_by_household": limits,
            "daily_excess_battery_throughput_kwh_by_household": daily_excess,
            "excess_daily_battery_throughput_kwh_by_household": excess_increment,
            "excess_daily_battery_throughput_kwh": float(sum(excess_increment)),
        }

    def _is_first_step_of_daily_cycle(self, t: int) -> bool:
        cycle_steps = self.params.cycle_steps
        return (
            cycle_steps is not None and cycle_steps > 0 and (t - 1) % cycle_steps == 0
        )

    def _daily_battery_throughput_limits_kwh(self) -> list[float]:
        limits = []
        mobility_intertemporal = (
            self.config.nexus == "energy_mobility"
            and "intertemporal" in self.formulation.applied_justice_modifiers
        )
        for household in self.params.households:
            allowance = household.ev.daily_battery_throughput_limit_kwh
            if allowance is None:
                allowance = household.ev.battery_capacity_kwh
            if mobility_intertemporal:
                reserve_fraction = household.ev.daily_battery_reserve_fraction
                if reserve_fraction is None:
                    reserve_fraction = 0.1
                reserve_target = (
                    float(reserve_fraction) * household.ev.battery_capacity_kwh
                )
                required = float(
                    household.mobility.required_departure_energy_kwh or 0.0
                )
                efficiency = float(household.ev.charge_efficiency)
                essential_charging = (
                    (required + reserve_target) / efficiency
                    if efficiency > 1e-12
                    else 0.0
                )
                limit = essential_charging + float(allowance)
            else:
                limit = float(allowance)
            limits.append(float(max(limit, 0.0)))
        return limits

    def _distributive_observation_context(self) -> dict[str, float]:
        state = self.simulator.state
        burdens = []
        for i, household_params in enumerate(self.params.households):
            requirement = float(
                household_params.mobility.required_departure_energy_kwh or 0.0
            )
            if requirement <= 0.0:
                burdens.append(0.0)
            elif self.household_departed[i]:
                burdens.append(
                    self.household_departure_shortfall_kwh[i] / requirement
                )
            else:
                energy = (
                    state.households[i].ev_energy_kwh if state is not None else 0.0
                )
                burdens.append(max(requirement - energy, 0.0) / requirement)
        mean_burden = sum(burdens) / len(burdens) if burdens else 0.0
        excesses = [max(value - mean_burden, 0.0) for value in burdens]
        worst = max(excesses, default=0.0)
        excess = sum(excesses) / len(excesses) if excesses else 0.0
        return {
            "distribution_normalized_worst_mobility_burden": worst / 0.8,
            "distribution_normalized_excess_mobility_burden": excess / 0.24,
        }

    def _intertemporal_observation_context(self, state) -> dict[str, object]:
        limits = self._daily_battery_throughput_limits_kwh()
        remaining_budget = [
            (limit - throughput) / limit if limit > 1e-12 else 0.0
            for throughput, limit in zip(
                self.daily_household_battery_throughput_kwh, limits, strict=True
            )
        ]
        reserve_targets = self._daily_battery_reserve_targets_kwh()
        reserve_margin = [
            (household.ev_energy_kwh - target)
            / household_params.ev.battery_capacity_kwh
            if household_params.ev.battery_capacity_kwh > 1e-12
            else 0.0
            for household, household_params, target in zip(
                state.households, self.params.households, reserve_targets, strict=True
            )
        ]
        return {
            "remaining_daily_throughput_budget_fraction": remaining_budget,
            "battery_reserve_margin_fraction": reserve_margin,
        }

    def _daily_battery_reserve_targets_kwh(self) -> list[float]:
        targets = []
        for household in self.params.households:
            fraction = household.ev.daily_battery_reserve_fraction
            if fraction is None:
                fraction = 0.1
            fraction = min(max(float(fraction), 0.0), 1.0)
            targets.append(fraction * household.ev.battery_capacity_kwh)
        return targets

    def _apply_procedural_layer(
        self,
        raw_ev_power_kw: list[float],
    ) -> ProceduralReport:
        if "procedural" in self.formulation.applied_justice_modifiers:
            _, report = apply_procedural_rules(
                raw_ev_power_kw, self.params, self.simulator.state
            )
            return report

        return ProceduralReport(
            raw_aggregator_request_kw=list(raw_ev_power_kw),
            authority_resolved_request_kw=list(raw_ev_power_kw),
            authority_adjustment_kw=[0.0] * self.params.n_households,
            request_source=["aggregator"] * self.params.n_households,
            authority_modes=["full"] * self.params.n_households,
            authority_bounds_kw=[
                {"min_kw": float(low), "max_kw": float(high)}
                for low, high in zip(
                    self.action_space.low.tolist(),
                    self.action_space.high.tolist(),
                    strict=True,
                )
            ],
            fallback_request_kw=[0.0] * self.params.n_households,
            ignored_aggregator_request_kw=[0.0] * self.params.n_households,
            authority_available=[True] * self.params.n_households,
            activity_threshold_charge_kw=[
                0.05 * household.ev.max_charge_kw for household in self.params.households
            ],
            activity_threshold_discharge_kw=[
                0.05 * household.ev.max_discharge_kw for household in self.params.households
            ],
            authority_resolved_active=[
                (float(value) > 0.05 * household.ev.max_charge_kw
                 or float(value) < -0.05 * household.ev.max_discharge_kw)
                for value, household in zip(
                    raw_ev_power_kw,
                    self.params.households, strict=True,
                )
            ],
            normalized_authority_adjustment=[0.0] * self.params.n_households,
            authority_restriction_active=[False] * self.params.n_households,
            consistency_by_authority_status={"available": 0.0},
        )

    def _action_to_simulator_action(self, action: np.ndarray) -> SimulatorAction:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape {action_array.shape} does not match "
                f"{self.action_space.shape}."
            )

        clipped_action = np.clip(action_array, self.action_space.low, self.action_space.high)
        return SimulatorAction(ev_power_commands_kw=clipped_action.tolist())

    def _make_info(
        self,
        constraints: dict[str, Any] | None = None,
        reward_terms: dict[str, float] | None = None,
        procedural_report: ProceduralReport | None = None,
    ) -> dict[str, Any]:
        state = self.simulator.state
        info: dict[str, Any] = {
            "agent_role": self.config.agent_role,
            "nexus": self.config.nexus,
            "justice": list(self.config.justice),
            "formulation": self.formulation.id,
            "applied_justice_modifiers": list(
                self.formulation.applied_justice_modifiers
            ),
            "grid_allocation_strategy": self.simulator.grid_allocation_strategy,
            "reward_weights": asdict(self.reward_weights),
            "dt_hours": self.params.dt_hours,
            "episode_steps": self.params.episode_steps,
            "cycle_steps": self.params.cycle_steps,
        }
        if state is not None:
            info["t"] = state.t
            info["system"] = asdict(state.system)
            info["households"] = [asdict(h) for h in state.households]
            info["cumulative_grid_clipping_burden_kwh"] = list(
                self.simulator.cumulative_grid_clipping_burden_kwh
            )
            info["household_departure_shortfall_kwh"] = list(
                self.household_departure_shortfall_kwh
            )
            info["household_departed"] = list(self.household_departed)
            if any(
                field.startswith("distribution_")
                for field in self.formulation.system_observation_fields
            ):
                info["distributive_observation_context"] = (
                    self._distributive_observation_context()
                )
            if any(
                field in {
                    "remaining_daily_throughput_budget_fraction",
                    "battery_reserve_margin_fraction",
                }
                for field in self.formulation.household_observation_fields
            ):
                info["intertemporal_observation_context"] = (
                    self._intertemporal_observation_context(state)
                )
            info["daily_battery_throughput_kwh_by_household"] = list(
                self.daily_household_battery_throughput_kwh
            )
            info["daily_excess_battery_throughput_kwh_by_household"] = list(
                self.daily_household_excess_battery_throughput_kwh
            )
            info["daily_battery_throughput_limit_kwh_by_household"] = (
                self._daily_battery_throughput_limits_kwh()
            )
            info["daily_battery_reserve_target_kwh_by_household"] = (
                self._daily_battery_reserve_targets_kwh()
            )
        if constraints is not None:
            info["constraints"] = constraints
        if reward_terms is not None:
            info["reward_terms"] = reward_terms
            info["reward_normalization"] = asdict(self.reward_normalization)
            if "household_reward" in reward_terms:
                info["grid_operation"] = self.grid_operation
                if "distributive_reward" in reward_terms:
                    info["reward_formulation"] = (
                        "energy_mobility_distributive_v1"
                        if self.config.nexus == "energy_mobility"
                        else "soft_import_distributive_savings_v1"
                    )
                elif "intertemporal_reward" in reward_terms:
                    info["reward_formulation"] = (
                        "energy_mobility_intertemporal_v1"
                        if self.config.nexus == "energy_mobility"
                        else "soft_import_intertemporal_v1"
                    )
                else:
                    info["reward_formulation"] = (
                        "soft_import_three_terms_v1"
                        if self.grid_operation != "hard_limits"
                        else "aggregate_stakeholders_v1"
                    )
                info["stakeholder_rewards"] = {
                    "households": reward_terms["household_reward"],
                    "grid": reward_terms["grid_reward"],
                    "aggregator": reward_terms["aggregator_reward"],
                }
                if "distributive_reward" in reward_terms:
                    info["stakeholder_rewards"]["distribution"] = (
                        reward_terms["distributive_reward"]
                    )
        if (
            procedural_report is not None
            and "procedural" in self.formulation.applied_justice_modifiers
        ):
            procedural_info = asdict(procedural_report) | {
                "total_authority_adjustment_kw": (
                    procedural_report.total_authority_adjustment_kw
                ),
                "num_households_with_authority_adjustment": (
                    procedural_report.num_households_with_authority_adjustment
                ),
            }
            if constraints is not None:
                grid_applied = list(constraints["applied_ev_power_kw"])
                authority_resolved = procedural_report.authority_resolved_request_kw
                procedural_info["trace"] = {
                    "raw_aggregator_request_kw": (
                        procedural_report.raw_aggregator_request_kw
                    ),
                    "authority_resolved_request_kw": authority_resolved,
                    "grid_applied_request_kw": grid_applied,
                    "authority_adjustment_kw": (
                        procedural_report.authority_adjustment_kw
                    ),
                    "grid_adjustment_kw": [
                        abs(requested - applied)
                        for requested, applied in zip(
                            authority_resolved, grid_applied, strict=True
                        )
                    ],
                    "request_source": procedural_report.request_source,
                    "authority_modes": procedural_report.authority_modes,
                    "authority_bounds_kw": procedural_info["authority_bounds_kw"],
                    "fallback_request_kw": procedural_report.fallback_request_kw,
                    "ignored_aggregator_request_kw": (
                        procedural_report.ignored_aggregator_request_kw
                    ),
                    "grid_allocation_strategy": self.simulator.grid_allocation_strategy,
                }
            info["procedural"] = procedural_info
        return info
