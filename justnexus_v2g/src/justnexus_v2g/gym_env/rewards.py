from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.core.types import StepResult
from justnexus_v2g.gym_env.formulations import FormulationSpec, build_formulation
from justnexus_v2g.gym_env.procedural import ProceduralReport, UNAUTHORIZED_DISCHARGE_EPSILON_KW
from justnexus_v2g.gym_env.specs import GymEnvConfig


@dataclass(frozen=True)
class RewardNormalization:
    """Fixed positive divisors for raw base reward terms; defaults preserve units."""

    command_mismatch_kwh: float = 1.0
    import_overload_squared_kw2h: float = 1.0
    excess_daily_battery_throughput_kwh: float = 1.0
    terminal_battery_reserve_deficit_kwh: float = 1.0
    departure_energy_shortfall_kwh: float = 1.0

    def __post_init__(self):
        for name, divisor in asdict(self).items():
            if not math.isfinite(divisor) or divisor <= 0:
                raise ValueError(f"Normalization divisor {name} must be finite and positive.")


@dataclass(frozen=True)
class RewardWeights:
    """Transparent scalarization weights for Gym rewards."""

    unmet_departure_energy: float = 0.0
    grid_constraint: float = 0.1
    total_ev_clipping: float = 0.1
    worst_mobility_burden: float = 0.0
    excess_mobility_burden: float = 0.0
    procedural_consistency: float = 0.0
    unauthorized_discharge_incidence: float = 0.0
    excess_daily_battery_throughput: float = 0.0
    battery_reserve_deficit: float = 0.0


def calculate_reward(
    result: StepResult,
    params: SimulationParams,
    weights: RewardWeights,
    config: GymEnvConfig | None = None,
    formulation: FormulationSpec | None = None,
    procedural_report: ProceduralReport | None = None,
    daily_battery_report: dict[str, object] | None = None,
    household_departure_shortfall_kwh: list[float] | None = None,
    mobility_distribution_finalized: bool = False,
    normalization: RewardNormalization | None = None,
) -> tuple[float, dict[str, float]]:
    formulation = formulation or build_formulation(config or GymEnvConfig())
    if formulation.reward_strategy == "energy_mobility":
        reward, terms = _calculate_energy_mobility_reward(
            result, params, weights, normalization
        )
    elif formulation.reward_strategy == "energy_mobility_distributive":
        reward, terms = _calculate_energy_mobility_distributive_reward(
            result, params, weights, normalization,
            household_departure_shortfall_kwh,
            mobility_distribution_finalized,
        )
    else:
        raise ValueError(f"Unsupported reward strategy: {formulation.reward_strategy}")

    if "intertemporal" in formulation.applied_justice_modifiers:
        intertemporal_reward, intertemporal_terms = _calculate_energy_mobility_intertemporal_reward(
            result, params, weights, normalization, daily_battery_report
        )
        terms.update(intertemporal_terms)
        reward += intertemporal_reward

    if (
        "procedural" in formulation.applied_justice_modifiers
        and procedural_report is not None
    ):
        total_adjustment = procedural_report.total_authority_adjustment_kw
        normalized_adjustment = procedural_report.mean_normalized_authority_adjustment
        consistency_gap = procedural_report.procedural_consistency_loss
        terms["authority_adjustment_kw"] = total_adjustment
        terms["normalized_authority_adjustment"] = normalized_adjustment
        terms["procedural_consistency_loss"] = consistency_gap
        terms["authority_compliance_score"] = 1.0 - normalized_adjustment
        terms["procedural_consistency_score"] = 1.0 - consistency_gap
        reward_horizon = max(params.episode_steps - 1, 1)
        consistency_reward = (
            -weights.procedural_consistency * consistency_gap / reward_horizon
        )
        # Mean incident rate per restricted household this step; dividing by the
        # 48-transition horizon matches the paper's procedural reward.
        # This episode sum differs from pooled household-time evaluation incidence.
        incidence = procedural_report.unauthorized_discharge_incidence_step
        incidence_reward = (
            -weights.unauthorized_discharge_incidence * incidence / reward_horizon
        )
        terms["unauthorized_discharge_incidents"] = float(procedural_report.unauthorized_discharge_incidents)
        terms["unauthorized_discharge_opportunities"] = float(procedural_report.unauthorized_discharge_opportunities)
        terms["unauthorized_discharge_epsilon_kw"] = UNAUTHORIZED_DISCHARGE_EPSILON_KW
        terms["unauthorized_discharge_incidence_step"] = incidence
        terms["unauthorized_discharge_incidence_reward"] = incidence_reward
        terms["procedural_reward_horizon_steps"] = float(reward_horizon)
        terms["procedural_consistency_reward"] = consistency_reward
        terms["procedural_reward"] = consistency_reward + incidence_reward
        reward += consistency_reward + incidence_reward

    return float(reward), terms


def _calculate_energy_mobility_intertemporal_reward(
    result, params, weights, normalization, daily_battery_report
):
    """Soft battery-health excess plus one terminal emergency-reserve deficit."""
    report = daily_battery_report or {}
    scales = normalization or RewardNormalization()
    excess_increment = float(
        report.get("excess_daily_battery_throughput_kwh", 0.0)
    )
    normalized_excess = (
        excess_increment / scales.excess_daily_battery_throughput_kwh
    )
    terminal = bool(result.done)
    reserve_targets = [
        _daily_battery_reserve_target_kwh(household)
        for household in params.households
    ]
    reserve_deficits = (
        _battery_reserve_deficit_by_household(result, params)
        if terminal
        else [0.0] * params.n_households
    )
    reserve_deficit = float(sum(reserve_deficits))
    normalized_reserve = (
        reserve_deficit / scales.terminal_battery_reserve_deficit_kwh
    )
    throughput_reward = (
        -weights.excess_daily_battery_throughput * normalized_excess
    )
    reserve_reward = -weights.battery_reserve_deficit * normalized_reserve
    terms = {
        "excess_daily_battery_throughput_kwh": excess_increment,
        "normalized_excess_daily_battery_throughput": normalized_excess,
        "daily_battery_throughput_kwh_by_household": list(
            report.get("daily_battery_throughput_kwh_by_household", [])
        ),
        "daily_battery_throughput_limit_kwh_by_household": list(
            report.get("daily_battery_throughput_limit_kwh_by_household", [])
        ),
        "daily_excess_battery_throughput_kwh_by_household": list(
            report.get("daily_excess_battery_throughput_kwh_by_household", [])
        ),
        "per_household_excess_daily_battery_throughput_kwh": list(
            report.get("excess_daily_battery_throughput_kwh_by_household", [])
        ),
        "terminal_reserve_evaluated": float(terminal),
        "terminal_battery_reserve_target_kwh_by_household": reserve_targets,
        "terminal_battery_reserve_deficit_kwh_by_household": reserve_deficits,
        "terminal_battery_reserve_deficit_kwh": reserve_deficit,
        "normalized_terminal_battery_reserve_deficit": normalized_reserve,
        "excess_daily_battery_throughput_reward": throughput_reward,
        "terminal_battery_reserve_reward": reserve_reward,
        "battery_reserve_deficit_reward": reserve_reward,
        "intertemporal_reward": throughput_reward + reserve_reward,
    }
    return float(throughput_reward + reserve_reward), terms


def _calculate_energy_mobility_reward(result, params, weights, normalization=None):
    """Household mobility, shared-grid safety, and execution feasibility."""
    scales = normalization or RewardNormalization()
    shortfall_by_household = [
        float(h.departure_energy_shortfall_kwh) for h in result.state.households
    ]
    shortfall = sum(shortfall_by_household)
    excess_kw = max(
        result.state.system.grid_exchange_kw - params.system.max_import_kw, 0.0
    )
    overload = excess_kw ** 2 * params.dt_hours
    mismatch = _total_ev_clipping_kw(result) * params.dt_hours
    normalized_shortfall = shortfall / scales.departure_energy_shortfall_kwh
    normalized_overload = overload / scales.import_overload_squared_kw2h
    normalized_mismatch = mismatch / scales.command_mismatch_kwh
    household_reward = -weights.unmet_departure_energy * normalized_shortfall
    grid_reward = -weights.grid_constraint * normalized_overload
    aggregator_reward = -weights.total_ev_clipping * normalized_mismatch
    terms = {
        "departure_energy_shortfall_kwh": shortfall,
        "per_household_departure_energy_shortfall_kwh": shortfall_by_household,
        "import_overload_kw": excess_kw,
        "import_overload_squared_kw2h": overload,
        "command_mismatch_kwh": mismatch,
        "normalized_departure_energy_shortfall_kwh": normalized_shortfall,
        "normalized_import_overload_squared_kw2h": normalized_overload,
        "normalized_command_mismatch_kwh": normalized_mismatch,
        "mobility_reward": household_reward,
        "household_reward": household_reward,
        "grid_reward": grid_reward,
        "aggregator_reward": aggregator_reward,
    }
    return float(household_reward + grid_reward + aggregator_reward), terms


def _calculate_energy_mobility_distributive_reward(
    result, params, weights, normalization=None,
    household_departure_shortfall_kwh=None,
    mobility_distribution_finalized=False,
):
    """Base mobility reward plus one final inequality penalty."""
    base_reward, terms = _calculate_energy_mobility_reward(
        result, params, weights, normalization
    )
    shortfalls = list(
        household_departure_shortfall_kwh
        if household_departure_shortfall_kwh is not None
        else [h.departure_energy_shortfall_kwh for h in result.state.households]
    )
    burdens = []
    for shortfall, household in zip(shortfalls, params.households, strict=True):
        requirement = float(household.mobility.required_departure_energy_kwh or 0.0)
        burdens.append(float(shortfall) / requirement if requirement > 0.0 else 0.0)
    mean_burden = sum(burdens) / len(burdens) if burdens else 0.0
    excesses = [max(value - mean_burden, 0.0) for value in burdens]
    worst = max(excesses, default=0.0)
    excess = sum(excesses) / len(excesses) if excesses else 0.0
    normalized_worst = worst / 0.8
    normalized_excess = excess / 0.24
    active = 1.0 if mobility_distribution_finalized else 0.0
    worst_reward = -weights.worst_mobility_burden * normalized_worst * active
    excess_reward = -weights.excess_mobility_burden * normalized_excess * active
    distributive_reward = worst_reward + excess_reward
    terms.update(
        household_mobility_burden_fraction=(
            burdens if mobility_distribution_finalized else [0.0] * len(burdens)
        ),
        mean_mobility_burden_fraction=mean_burden * active,
        worst_mobility_burden=worst * active,
        excess_mobility_burden=excess * active,
        normalized_worst_mobility_burden=normalized_worst * active,
        normalized_excess_mobility_burden=normalized_excess * active,
        mobility_distribution_finalized=active,
        worst_mobility_burden_reward=worst_reward,
        excess_mobility_burden_reward=excess_reward,
        distributive_reward=distributive_reward,
    )
    return float(base_reward + distributive_reward), terms


def _battery_reserve_deficit_by_household(
    result: StepResult,
    params: SimulationParams,
) -> list[float]:
    return [
        float(
            max(
                _daily_battery_reserve_target_kwh(household_params)
                - household.ev_energy_kwh,
                0.0,
            )
        )
        for household, household_params in zip(
            result.state.households,
            params.households,
            strict=True,
        )
    ]


def _daily_battery_reserve_target_kwh(household_params) -> float:
    ev = household_params.ev
    fraction = ev.daily_battery_reserve_fraction
    if fraction is None:
        fraction = 0.1
    fraction = min(max(float(fraction), 0.0), 1.0)
    return float(fraction * ev.battery_capacity_kwh)


def _total_ev_clipping_kw(result: StepResult) -> float:
    return sum(_household_clipping_kw(result))


def _household_clipping_kw(result: StepResult) -> list[float]:
    return [
        abs(requested - applied)
        for requested, applied in zip(
            result.constraints.requested_ev_power_kw,
            result.constraints.applied_ev_power_kw,
            strict=True,
        )
    ]
