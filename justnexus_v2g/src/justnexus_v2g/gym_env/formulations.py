from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from justnexus_v2g.gym_env.specs import GymEnvConfig, Justice, Nexus

RewardStrategy = Literal["energy_mobility", "energy_mobility_distributive"]


def _append_unique(
    existing: tuple[str, ...],
    additions: tuple[str, ...],
) -> tuple[str, ...]:
    return (*existing, *(field for field in additions if field not in existing))


ENERGY_MOBILITY_SYSTEM_OBSERVATION_FIELDS = (
    "time_sin",
    "time_cos",
    "total_load_kw",
    "total_pv_kw",
    "requested_total_ev_power_kw",
    "applied_total_ev_power_kw",
    "grid_exchange_kw",
)

ENERGY_MOBILITY_HOUSEHOLD_OBSERVATION_FIELDS = (
    "requested_ev_power_kw",
    "applied_ev_power_kw",
    "ev_connected",
    "ev_soc_fraction",
    "time_to_departure_hours",
    "required_departure_energy_kwh",
    "battery_capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
)

ENERGY_MOBILITY_DISTRIBUTIVE_SYSTEM_OBSERVATION_FIELDS = (
    "distribution_normalized_worst_mobility_burden",
    "distribution_normalized_excess_mobility_burden",
)

ENERGY_INTERTEMPORAL_SYSTEM_OBSERVATION_FIELDS: tuple[str, ...] = ()

ENERGY_INTERTEMPORAL_HOUSEHOLD_OBSERVATION_FIELDS = (
    "remaining_daily_throughput_budget_fraction",
    "battery_reserve_margin_fraction",
)

ENERGY_PROCEDURAL_SYSTEM_OBSERVATION_FIELDS: tuple[str, ...] = ()

ENERGY_PROCEDURAL_HOUSEHOLD_OBSERVATION_FIELDS = (
    "authority_available",
)


@dataclass(frozen=True)
class FormulationSpec:
    id: str
    nexus: Nexus
    justice: tuple[Justice, ...]
    applied_justice_modifiers: tuple[Justice, ...]
    system_observation_fields: tuple[str, ...]
    household_observation_fields: tuple[str, ...]
    reward_strategy: RewardStrategy


@dataclass(frozen=True)
class JusticeModifier:
    justice: Justice
    system_observation_fields: tuple[str, ...] | None = None
    household_observation_fields: tuple[str, ...] | None = None
    reward_strategy: RewardStrategy | None = None

    def apply(self, formulation: FormulationSpec) -> FormulationSpec:
        return replace(
            formulation,
            applied_justice_modifiers=(
                *formulation.applied_justice_modifiers,
                self.justice,
            ),
            system_observation_fields=(
                _append_unique(
                    formulation.system_observation_fields,
                    self.system_observation_fields,
                )
                if self.system_observation_fields is not None
                else formulation.system_observation_fields
            ),
            household_observation_fields=(
                _append_unique(
                    formulation.household_observation_fields,
                    self.household_observation_fields,
                )
                if self.household_observation_fields is not None
                else formulation.household_observation_fields
            ),
            reward_strategy=(
                self.reward_strategy
                if self.reward_strategy is not None
                else formulation.reward_strategy
            ),
        )


def build_formulation(config: GymEnvConfig) -> FormulationSpec:
    formulation = _base_formulation(config.nexus, config.justice)
    for justice in config.justice:
        modifier = JUSTICE_MODIFIERS.get((config.nexus, justice))
        if modifier is None:
            raise ValueError(f"Unsupported native formulation: {config.nexus}/{justice}")
        formulation = modifier.apply(formulation)
    return formulation


def _base_formulation(
    nexus: Nexus,
    justice: tuple[Justice, ...],
) -> FormulationSpec:
    if nexus != "energy_mobility":
        raise ValueError(f"Unsupported nexus: {nexus}")
    return FormulationSpec(
        id=_formulation_id(nexus, justice),
        nexus=nexus,
        justice=justice,
        applied_justice_modifiers=(),
        system_observation_fields=ENERGY_MOBILITY_SYSTEM_OBSERVATION_FIELDS,
        household_observation_fields=ENERGY_MOBILITY_HOUSEHOLD_OBSERVATION_FIELDS,
        reward_strategy="energy_mobility",
    )


def _formulation_id(nexus: Nexus, justice: tuple[Justice, ...]) -> str:
    if justice == ():
        return nexus
    return "_".join((nexus, *justice))


ENERGY_MOBILITY_DISTRIBUTIVE_MODIFIER = JusticeModifier(
    justice="distributive",
    system_observation_fields=ENERGY_MOBILITY_DISTRIBUTIVE_SYSTEM_OBSERVATION_FIELDS,
    household_observation_fields=(),
    reward_strategy="energy_mobility_distributive",
)

INTERTEMPORAL_MODIFIER = JusticeModifier(
    justice="intertemporal",
    system_observation_fields=ENERGY_INTERTEMPORAL_SYSTEM_OBSERVATION_FIELDS,
    household_observation_fields=ENERGY_INTERTEMPORAL_HOUSEHOLD_OBSERVATION_FIELDS,
)

PROCEDURAL_MODIFIER = JusticeModifier(
    justice="procedural",
    system_observation_fields=ENERGY_PROCEDURAL_SYSTEM_OBSERVATION_FIELDS,
    household_observation_fields=ENERGY_PROCEDURAL_HOUSEHOLD_OBSERVATION_FIELDS,
)

JUSTICE_MODIFIERS: dict[tuple[Nexus, Justice], JusticeModifier] = {
    ("energy_mobility", "distributive"): ENERGY_MOBILITY_DISTRIBUTIVE_MODIFIER,
    ("energy_mobility", "procedural"): PROCEDURAL_MODIFIER,
    ("energy_mobility", "intertemporal"): INTERTEMPORAL_MODIFIER,
}
