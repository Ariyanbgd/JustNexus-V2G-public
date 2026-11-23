from __future__ import annotations

from dataclasses import dataclass

from justnexus_v2g.controllers import (
    AlwaysChargeController,
    DepartureAwareController,
    GridAwareController,
    PriceAwareController,
)
from justnexus_v2g.core.params import HouseholdParams, SimulationParams
from justnexus_v2g.core.state import HouseholdState, SimulatorState


# Fixed raw-request resolution threshold, shared by training and evaluation.
UNAUTHORIZED_DISCHARGE_EPSILON_KW = 1e-3


@dataclass(frozen=True)
class AuthorityBounds:
    min_kw: float
    max_kw: float


@dataclass(frozen=True)
class ProceduralReport:
    """Reports how authority rules resolved aggregator requests."""

    raw_aggregator_request_kw: list[float]
    authority_resolved_request_kw: list[float]
    authority_adjustment_kw: list[float]
    request_source: list[str]
    authority_modes: list[str]
    authority_bounds_kw: list[AuthorityBounds]
    fallback_request_kw: list[float]
    ignored_aggregator_request_kw: list[float]
    authority_available: list[bool]
    activity_threshold_charge_kw: list[float]
    activity_threshold_discharge_kw: list[float]
    authority_resolved_active: list[bool]
    normalized_authority_adjustment: list[float]
    authority_restriction_active: list[bool]
    consistency_by_authority_status: dict[str, float]

    @property
    def total_authority_adjustment_kw(self) -> float:
        return float(sum(self.authority_adjustment_kw))

    @property
    def num_households_with_authority_adjustment(self) -> int:
        return sum(v > 0.0 for v in self.authority_adjustment_kw)

    @property
    def mean_normalized_authority_adjustment(self) -> float:
        restricted_adjustments = [
            value
            for value, restricted in zip(
                self.normalized_authority_adjustment,
                self.authority_restriction_active,
                strict=True,
            )
            if restricted
        ]
        if not restricted_adjustments:
            return 0.0
        return float(sum(restricted_adjustments) / len(restricted_adjustments))

    @property
    def procedural_consistency_loss(self) -> float:
        values = list(self.consistency_by_authority_status.values())
        return float(sum(values) / len(values)) if values else 0.0

    @property
    def unauthorized_discharge_opportunities(self) -> int:
        return sum(bound.min_kw >= 0.0 for bound in self.authority_bounds_kw)

    @property
    def unauthorized_discharge_incidents(self) -> int:
        return sum(
            bound.min_kw >= 0.0 and raw < -UNAUTHORIZED_DISCHARGE_EPSILON_KW
            for raw, bound in zip(self.raw_aggregator_request_kw, self.authority_bounds_kw, strict=True)
        )

    @property
    def unauthorized_discharge_incidence_step(self) -> float:
        n = self.unauthorized_discharge_opportunities
        return self.unauthorized_discharge_incidents / n if n else 0.0

    @property
    def raw_ev_power_kw(self) -> list[float]:
        return self.raw_aggregator_request_kw

    @property
    def admissible_ev_power_kw(self) -> list[float]:
        return self.authority_resolved_request_kw

    @property
    def per_household_violation_kw(self) -> list[float]:
        return self.authority_adjustment_kw

    @property
    def total_violation_kw(self) -> float:
        return self.total_authority_adjustment_kw

    @property
    def num_households_with_violation(self) -> int:
        return self.num_households_with_authority_adjustment


def apply_procedural_rules(
    raw_ev_power_kw: list[float],
    params: SimulationParams,
    state: SimulatorState | None = None,
) -> tuple[list[float], ProceduralReport]:
    """Resolve aggregator requests against household authority boundaries."""

    resolved: list[float] = []
    adjustments: list[float] = []
    request_source: list[str] = []
    authority_modes: list[str] = []
    bounds: list[AuthorityBounds] = []
    fallback_requests: list[float] = []
    ignored_requests: list[float] = []
    authority_availability: list[bool] = []
    charge_thresholds: list[float] = []
    discharge_thresholds: list[float] = []
    active: list[bool] = []
    normalized_adjustments: list[float] = []
    restrictions_active: list[bool] = []

    household_states = (
        state.households if state is not None else [None] * params.n_households
    )

    for raw, household_params, household_state in zip(
        raw_ev_power_kw, params.households, household_states, strict=True
    ):
        rules = household_params.procedural
        if rules.authority_scope not in {"all_actions", "discharge_only"}:
            raise ValueError(
                f"Unknown procedural authority_scope {rules.authority_scope!r}."
            )
        discharge_only = rules.authority_scope == "discharge_only"
        authority_available = is_authority_available(
            rules.authority_mode,
            household_params,
            state.t if state is not None else None,
            params.cycle_steps or params.episode_steps,
        )
        authority_mode = "available" if authority_available else "unavailable"
        authority_availability.append(authority_available)
        authority_modes.append(authority_mode)
        authority_bounds = _authority_bounds(household_params, authority_available)
        if discharge_only and not authority_available:
            authority_bounds = AuthorityBounds(
                min_kw=0.0,
                max_kw=float(household_params.ev.max_charge_kw),
            )
        bounds.append(authority_bounds)
        restrictions_active.append(
            authority_bounds.min_kw > -float(household_params.ev.max_discharge_kw)
            or (
                not discharge_only
                and authority_bounds.max_kw < float(household_params.ev.max_charge_kw)
            )
        )

        fallback = 0.0
        if authority_available:
            value = _clip(raw, authority_bounds)
            source = "aggregator"
            ignored = 0.0
            authority_adjustment = abs(float(raw) - float(value))
        elif discharge_only and float(raw) >= 0.0:
            # Discharge consent never blocks mobility charging. Physical battery
            # feasibility is still applied downstream.
            value = min(float(raw), float(household_params.ev.max_charge_kw))
            source = "aggregator_charge"
            ignored = 0.0
            authority_adjustment = 0.0
        else:
            value = 0.0
            source = "unavailable"
            ignored = float(raw)
            authority_adjustment = abs(float(raw))

        resolved.append(float(value))
        adjustments.append(float(authority_adjustment))
        request_source.append(source)
        fallback_requests.append(float(fallback))
        ignored_requests.append(float(ignored))
        charge_threshold = 0.05 * float(household_params.ev.max_charge_kw)
        discharge_threshold = 0.05 * float(household_params.ev.max_discharge_kw)
        charge_thresholds.append(charge_threshold)
        discharge_thresholds.append(discharge_threshold)
        active.append(
            float(value) < -discharge_threshold
            if discharge_only
            else (
                float(value) > charge_threshold
                or float(value) < -discharge_threshold
            )
        )
        adjustment_scale = (
            float(household_params.ev.max_charge_kw)
            if float(raw) >= 0.0
            else float(household_params.ev.max_discharge_kw)
        )
        normalized_adjustments.append(
            min(float(authority_adjustment) / adjustment_scale, 1.0)
            if adjustment_scale > 0.0
            else 0.0
        )

    consistency_by_status: dict[str, float] = {}
    for status in ("available", "unavailable"):
        status_activity = [
            value
            for value, mode in zip(active, authority_modes, strict=True)
            if mode == status
        ]
        if status_activity:
            consistency_by_status[status] = float(
                int(any(status_activity)) - int(all(status_activity))
            )

    report = ProceduralReport(
        raw_aggregator_request_kw=list(raw_ev_power_kw),
        authority_resolved_request_kw=resolved,
        authority_adjustment_kw=adjustments,
        request_source=request_source,
        authority_modes=authority_modes,
        authority_bounds_kw=bounds,
        fallback_request_kw=fallback_requests,
        ignored_aggregator_request_kw=ignored_requests,
        authority_available=authority_availability,
        activity_threshold_charge_kw=charge_thresholds,
        activity_threshold_discharge_kw=discharge_thresholds,
        authority_resolved_active=active,
        normalized_authority_adjustment=normalized_adjustments,
        authority_restriction_active=restrictions_active,
        consistency_by_authority_status=consistency_by_status,
    )
    return resolved, report


def effective_authority_mode(authority_mode: str) -> str:
    return authority_mode


def is_authority_available(
    authority_mode: str,
    household: HouseholdParams,
    t: int | None,
    cycle_steps: int | None,
) -> bool:
    if authority_mode == "full":
        return True
    if authority_mode == "none":
        return False
    if authority_mode == "charge_only":
        return True
    if authority_mode == "limited":
        return _within_authorized_window(household, t, cycle_steps)
    raise ValueError(f"Unknown authority_mode {authority_mode!r}.")


def _within_authorized_window(
    household: HouseholdParams,
    t: int | None,
    cycle_steps: int | None,
) -> bool:
    start = household.procedural.authorized_start_step
    end = household.procedural.authorized_end_step
    if start is None or end is None or t is None:
        return True
    period = int(cycle_steps or 0)
    if period <= 0:
        return True
    cycle_t = int(t) % period
    start = int(start) % period
    end = int(end) % period
    if start == end:
        return True
    if start < end:
        return start <= cycle_t < end
    return cycle_t >= start or cycle_t < end


def _authority_bounds(household: HouseholdParams, authority_available: bool | None = None) -> AuthorityBounds:
    rules = household.procedural
    max_charge = (
        rules.max_authorized_charge_kw
        if rules.max_authorized_charge_kw is not None
        else household.ev.max_charge_kw
    )
    max_discharge = (
        rules.max_authorized_discharge_kw
        if rules.max_authorized_discharge_kw is not None
        else household.ev.max_discharge_kw
    )

    if authority_available is False:
        return AuthorityBounds(min_kw=0.0, max_kw=0.0)
    if rules.authority_mode == "charge_only":
        return AuthorityBounds(min_kw=0.0, max_kw=float(max_charge))
    if rules.authority_mode in {"limited", "full", "none"}:
        return AuthorityBounds(min_kw=-float(max_discharge), max_kw=float(max_charge))
    raise ValueError(f"Unknown authority_mode {rules.authority_mode!r}.")


def _fallback_request(
    household: HouseholdParams,
    household_state: HouseholdState | None,
    simulator_state: SimulatorState | None,
) -> float:
    controller = household.procedural.fallback_controller
    if controller == "idle":
        return 0.0
    if controller in {"arbitrary", "local_charge"}:
        if household_state is not None and not household_state.ev_connected:
            return 0.0
        return household.procedural.fallback_power_kw

    if household_state is None or simulator_state is None:
        return 0.0

    local_state = SimulatorState(
        t=simulator_state.t,
        households=[household_state],
        system=simulator_state.system,
    )
    local_controller = _make_local_controller(controller, household)
    action = local_controller.act(local_state)
    return float(action.ev_power_commands_kw[0])


def _make_local_controller(controller: str, household: HouseholdParams):
    fallback_power = household.procedural.fallback_power_kw
    if controller == "always_charge":
        return AlwaysChargeController(charge_power_kw=fallback_power or 4.0)
    if controller == "departure_aware":
        return DepartureAwareController()
    if controller == "price_aware":
        return PriceAwareController()
    if controller == "grid_aware":
        return GridAwareController()
    raise ValueError(f"Unknown fallback_controller {controller!r}.")


def _clip(value: float, bounds: AuthorityBounds) -> float:
    return min(max(float(value), bounds.min_kw), bounds.max_kw)
