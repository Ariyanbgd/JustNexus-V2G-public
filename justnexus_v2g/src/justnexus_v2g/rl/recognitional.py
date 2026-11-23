"""Declared-needs, household-alone oracle construction for mobility studies."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.exogenous.demand import household_demand_profile
from justnexus_v2g.exogenous.mobility import ev_connected_profile, trip_energy_profile
from justnexus_v2g.exogenous.pv import household_pv_profile
from justnexus_v2g.exogenous.tariffs import export_tariff_profile, import_tariff_profile
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv

ORACLE_VERSION = "declared_needs_clarabel_v1"
PREFERENCE_ORDERS = {
    "departure_reserve": (
        "minimum_trip_shortfall", "minimum_reserve_shortfall", "minimum_throughput",
        "minimum_bill", "minimum_squared_power",
    ),
    "affordable_scheduling": (
        "minimum_trip_shortfall", "minimum_bill", "minimum_throughput",
        "minimum_squared_power",
    ),
    "battery_preservation": (
        "minimum_trip_shortfall", "minimum_throughput", "minimum_bill",
        "minimum_squared_power",
    ),
    "minimum_mobility_cost": (
        "minimum_trip_shortfall", "minimum_incremental_bill", "minimum_throughput",
        "minimum_squared_power",
    ),
    "no_discharge": (
        "minimum_trip_shortfall", "minimum_discharge", "minimum_bill",
        "minimum_throughput", "minimum_squared_power",
    ),
}
STAGE_TOLERANCE = {
    "minimum_trip_shortfall": 1e-6,
    "minimum_reserve_shortfall": 1e-6,
    "minimum_bill": 1e-5,
    "minimum_incremental_bill": 1e-5,
    "minimum_discharge": 1e-5,
    "minimum_throughput": 1e-5,
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def resolve_preferences(config: dict, labels: list[str]) -> list[dict]:
    """Validate each household's ordered lexicographic objectives."""
    selected = config["preferences"]
    if not isinstance(selected, dict) or len(labels) != len(set(labels)) or set(selected) != set(labels):
        raise ValueError("Recognitional preferences must cover each unique scenario label exactly once")
    resolved = []
    for label in labels:
        choice = selected[label]
        if not isinstance(choice, dict) or not isinstance(choice.get("priority_order"), list):
            raise ValueError(f"Preference {label} needs an ordered priority_order list")
        order = choice["priority_order"]
        if (len(order) < 2 or order[0] != "minimum_trip_shortfall"
                or order[-1] != "minimum_squared_power"
                or len(order) != len(set(order))
                or any(item not in STAGE_TOLERANCE and item != "minimum_squared_power" for item in order)):
            raise ValueError(
                f"Invalid priority_order for {label}: start with minimum_trip_shortfall, "
                "end with minimum_squared_power, and use each supported objective at most once"
            )
        uses_reserve = "minimum_reserve_shortfall" in order
        expected = {"priority_order", "reserve_kwh"} if uses_reserve else {"priority_order"}
        if set(choice) != expected:
            raise ValueError(f"Preference {label} requires exactly {sorted(expected)}")
        reserve = float(choice.get("reserve_kwh", 0.0))
        if not math.isfinite(reserve) or (uses_reserve and reserve <= 0):
            raise ValueError(f"Invalid departure reserve for {label}")
        need = next((name for name, historical in PREFERENCE_ORDERS.items()
                     if tuple(order) == historical), "custom_order")
        resolved.append({
            "scenario_label": label, "need": need, "departure_reserve_kwh": reserve,
            "priority_order": list(order),
        })
    return resolved


def _profiles(params: SimulationParams, index: int) -> dict[str, np.ndarray]:
    steps = range(1, params.episode_steps)
    return {
        "connected": np.asarray([ev_connected_profile(params, index, t) for t in steps], dtype=float),
        "trips": np.asarray([trip_energy_profile(params, index, t) for t in steps], dtype=float),
        "net_load": np.asarray([
            household_demand_profile(params, index, t) - household_pv_profile(params, index, t)
            for t in steps
        ], dtype=float),
        "import_tariff": np.asarray([import_tariff_profile(params, t) for t in steps], dtype=float),
        "export_tariff": np.asarray([export_tariff_profile(params, t) for t in steps], dtype=float),
    }


def _settlement(net_load, actions, imports, exports, dt) -> float:
    net = net_load + actions
    return float(dt * np.sum(imports * np.maximum(net, 0.0) - exports * np.maximum(-net, 0.0)))


def _solve_household(params: SimulationParams, index: int, preference: dict) -> dict:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("Building recognitional oracles requires cvxpy and CLARABEL") from exc

    household = params.households[index]
    ev = household.ev
    dt = float(params.dt_hours)
    data = _profiles(params, index)
    count = len(data["connected"])
    charge = cp.Variable(count, nonneg=True)
    discharge = cp.Variable(count, nonneg=True)
    energy = cp.Variable(count, nonneg=True)
    shortfall = cp.Variable(count, nonneg=True)
    constraints = [
        charge <= float(ev.max_charge_kw) * data["connected"],
        discharge <= float(ev.max_discharge_kw) * data["connected"],
        energy <= float(ev.battery_capacity_kwh),
    ]
    requirement = float(household.mobility.required_departure_energy_kwh or 0.0)
    initial = float(ev.initial_energy_kwh)
    departure_before = None
    for step, trip in enumerate(data["trips"]):
        before = (
            (initial if step == 0 else energy[step - 1])
            + dt * float(ev.charge_efficiency) * charge[step]
            - dt / float(ev.discharge_efficiency) * discharge[step]
        )
        if trip > 0:
            if departure_before is not None:
                raise ValueError("Recognitional oracle expects one trip per household per day")
            departure_before = before
            constraints.extend([
                shortfall[step] >= requirement - before,
                energy[step] == before - trip + shortfall[step],
            ])
        else:
            constraints.extend([shortfall[step] == 0, energy[step] == before])
    if departure_before is None:
        raise ValueError(f"Household {index} has no departure")
    net = data["net_load"] + charge - discharge
    spread = data["import_tariff"] - data["export_tariff"]
    if np.any(spread < -1e-10):
        raise ValueError("Oracle bill objective requires import tariff >= export tariff")
    bill = dt * cp.sum(cp.multiply(data["export_tariff"], net) + cp.multiply(spread, cp.pos(net)))
    no_action_bill = _settlement(data["net_load"], np.zeros(count), data["import_tariff"], data["export_tariff"], dt)
    reserve = float(preference["departure_reserve_kwh"])
    objectives = {
        "minimum_trip_shortfall": cp.sum(shortfall),
        "minimum_reserve_shortfall": cp.pos(requirement + reserve - departure_before),
        "minimum_bill": bill,
        "minimum_incremental_bill": bill - no_action_bill,
        "minimum_discharge": dt * cp.sum(discharge),
        "minimum_throughput": dt * cp.sum(charge + discharge),
        "minimum_squared_power": dt * cp.sum_squares(charge - discharge),
    }
    limits = list(constraints)
    optima = {}
    for stage in preference["priority_order"]:
        problem = cp.Problem(cp.Minimize(objectives[stage]), limits)
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, max_iter=2000)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Oracle household {index}, {stage}: {problem.status}")
        optimum = float(problem.value)
        optima[stage] = optimum
        if stage != "minimum_squared_power":
            limits = limits + [objectives[stage] <= optimum + STAGE_TOLERANCE[stage]]
    return {
        "actions": np.asarray(charge.value - discharge.value, dtype=np.float32),
        "optima": optima, "profiles": data, "no_action_bill": no_action_bill,
    }


def _verify_replay(params: SimulationParams, index: int, preference: dict, solved: dict) -> dict:
    one = SimulationParams(
        n_households=1, dt_hours=params.dt_hours, episode_steps=params.episode_steps,
        households=[deepcopy(params.households[index])], system=deepcopy(params.system),
        cycle_steps=params.cycle_steps,
    )
    env = JustNexusV2GEnv(
        one, config=GymEnvConfig(agent_role="aggregator", nexus="energy_mobility"),
        grid_operation="soft_import_unlimited_export",
    )
    shortfall = mismatch = discharge = 0.0
    applied = []
    reserve_at_departure = None
    try:
        env.reset(seed=0)
        for action in solved["actions"]:
            _, _, terminated, truncated, info = env.step(np.asarray([action], dtype=np.float32))
            result = env.last_step_result
            actual = float(result.constraints.applied_ev_power_kw[0])
            applied.append(actual)
            shortfall += float(info["reward_terms"]["departure_energy_shortfall_kwh"])
            mismatch += float(info["reward_terms"]["command_mismatch_kwh"])
            discharge += max(-actual, 0.0) * params.dt_hours
            if result.state.households[0].departure_energy_shortfall_kwh >= 0 and (
                result.state.t == params.households[index].mobility.departure_step
            ):
                reserve_at_departure = float(result.state.households[0].ev_energy_kwh)
            if terminated or truncated:
                break
    finally:
        env.close()
    if len(applied) != params.episode_steps - 1:
        raise RuntimeError(f"Oracle household {index} replay did not cover the full day")
    if shortfall > solved["optima"]["minimum_trip_shortfall"] + 1e-3 or mismatch > 1e-3:
        raise RuntimeError(f"Oracle household {index} failed simulator shortfall/mismatch replay")
    if "minimum_reserve_shortfall" in preference["priority_order"] and (
        reserve_at_departure is None or reserve_at_departure + solved["optima"]["minimum_reserve_shortfall"]
        + 1e-3 < preference["departure_reserve_kwh"]
    ):
        raise RuntimeError(f"Oracle household {index} failed departure reserve replay")
    if ("minimum_discharge" in preference["priority_order"]
            and discharge > solved["optima"]["minimum_discharge"] + 1e-3):
        raise RuntimeError(f"Oracle household {index} failed discharge replay")
    data = solved["profiles"]
    bill = _settlement(data["net_load"], np.asarray(applied), data["import_tariff"], data["export_tariff"], params.dt_hours)
    bill_key = next((key for key in ("minimum_incremental_bill", "minimum_bill")
                     if key in solved["optima"]), None)
    if bill_key is not None:
        bill_optimum = solved["optima"][bill_key] + (
            solved["no_action_bill"] if bill_key == "minimum_incremental_bill" else 0.0
        )
        if bill > bill_optimum + 1e-3:
            raise RuntimeError(f"Oracle household {index} failed bill replay")
    return {"shortfall_kwh": shortfall, "mismatch_kwh": mismatch, "bill_eur": bill}


def prepare_oracle_bundle(params: SimulationParams, preferences: list[dict], *, cache_dir: Path | None = None) -> dict:
    """Solve and cache the five historical-style oracles for one deterministic day."""
    if len(preferences) != params.n_households:
        raise ValueError("One recognitional preference is required per household")
    for household in params.households:
        ev = household.ev
        if (ev.initial_energy_min_kwh is not None or ev.initial_energy_max_kwh is not None) and (
            ev.initial_energy_min_kwh != ev.initial_energy_max_kwh
        ):
            raise ValueError("Full-day offline oracles require deterministic initial EV energy")
    # Match the day-suite fingerprint exactly, including JSON spacing.
    day_sha = hashlib.sha256(json.dumps(asdict(params), sort_keys=True).encode()).hexdigest()
    preference_sha = _digest(preferences)
    try:
        solver_version = version("clarabel")
    except Exception:
        solver_version = "unknown"
    key = _digest({"version": ORACLE_VERSION, "solver_version": solver_version,
                   "day": day_sha, "preferences": preference_sha})
    path = Path(cache_dir) / f"{key}.json" if cache_dir is not None else None
    if path is not None and path.exists():
        bundle = json.loads(path.read_text())
        if bundle.get("key") != key or bundle.get("sha256") != _digest({k: v for k, v in bundle.items() if k != "sha256"}):
            raise ValueError(f"Oracle cache integrity check failed: {path}")
        return bundle
    actions = []
    replay = []
    for index, preference in enumerate(preferences):
        solved = _solve_household(params, index, preference)
        actions.append(solved["actions"])
        replay.append(_verify_replay(params, index, preference, solved))
    matrix = np.asarray(actions, dtype=float).T
    dt = float(params.dt_hours)
    l1 = (np.abs(matrix).sum(axis=0) * dt).tolist()
    l2 = (np.square(matrix).sum(axis=0) * dt).tolist()
    if any(not math.isfinite(x) or x <= 0 for x in l1 + l2):
        raise ValueError("Every household oracle needs a positive action-gap reference")
    bundle = {
        "key": key, "version": ORACLE_VERSION, "solver": "CLARABEL",
        "solver_version": solver_version, "day_sha256": day_sha,
        "preferences_sha256": preference_sha, "preferences": preferences,
        "actions_kw": matrix.tolist(), "idle_to_oracle_l1_kwh": l1,
        "idle_to_oracle_l2_kw2h": l2, "replay": replay,
    }
    bundle["sha256"] = _digest(bundle)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        pending.replace(path)
    return bundle
