"""Paper-aligned episode metrics, separate from training reward weights."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from justnexus_v2g.gym_env.throughput import (
    MOBILITY_THROUGHPUT_REFERENCE_ID,
    controlled_throughput_caps_kwh,
)
from justnexus_v2g.gym_env.procedural import (
    UNAUTHORIZED_DISCHARGE_EPSILON_KW, apply_procedural_rules,
)
from justnexus_v2g.rl.recognitional import prepare_oracle_bundle

MOBILITY_REFERENCE_KWH = 74.0
GRID_REFERENCE_KW2H = 267.9392226234346
MISMATCH_REFERENCE_KWH = 160.0
RESERVE_REFERENCE_KWH = 28.5


def prepare_metric_oracles(problem, resolved_preferences: list[dict], *, cache_dir: Path):
    """Use one declared reference profile for every policy in a test split."""
    days = (
        problem.env.days if problem.provenance["mode"] == "variable"
        else (problem.env.unwrapped.params,)
    )
    return [prepare_oracle_bundle(day, resolved_preferences, cache_dir=cache_dir)
            for day in days]


def episode_metrics(infos: list[dict], params, *, oracle_bundle: dict | None = None) -> dict:
    """Calculate one episode's reported metrics from executed transitions."""
    if not infos:
        raise ValueError("Paper metrics require a completed episode")
    last = infos[-1]
    if last["t"] != params.episode_steps - 1:
        raise ValueError("Paper metrics require the final episode transition")
    if len(infos) != params.episode_steps - 1 or any(
        info["t"] != step for step, info in enumerate(infos, start=1)
    ):
        raise ValueError("Paper metrics require a complete ordered episode trace")
    n = params.n_households
    dt = float(params.dt_hours)
    shortfalls = [float(x) for x in last["household_departure_shortfall_kwh"]]
    requirements = [float(h.mobility.required_departure_energy_kwh)
                    for h in params.households]
    if len(shortfalls) != n or len(requirements) != n or any(q <= 0 for q in requirements):
        raise ValueError("Invalid household shortfall/requirement vector")
    mobility = math.fsum(shortfalls)
    grid = math.fsum(float(info["reward_terms"]["import_overload_squared_kw2h"])
                     for info in infos)
    mismatch = math.fsum(float(info["reward_terms"]["command_mismatch_kwh"])
                         for info in infos)
    step_shortfall = math.fsum(float(info["reward_terms"]["departure_energy_shortfall_kwh"])
                              for info in infos)
    if not math.isclose(mobility, step_shortfall, abs_tol=1e-7):
        raise ValueError("Final household shortfalls disagree with transition terms")
    burdens = [min(max(s / q, 0.0), 1.0)
               for s, q in zip(shortfalls, requirements, strict=True)]
    mean_burden = math.fsum(burdens) / n
    excesses = [max(b - mean_burden, 0.0) for b in burdens]

    targets = [float(x) for x in last["daily_battery_reserve_target_kwh_by_household"]]
    final_energy = [float(h["ev_energy_kwh"]) for h in last["households"]]
    if len(targets) != n or not math.isclose(math.fsum(targets), RESERVE_REFERENCE_KWH,
                                            abs_tol=1e-8):
        raise ValueError("Reserve targets differ from the paper's 28.5 kWh reference")
    reserve_deficits = [max(target - energy, 0.0)
                        for target, energy in zip(targets, final_energy, strict=True)]
    applied = [[float(h["applied_ev_power_kw"]) for h in info["households"]]
               for info in infos]
    if any(len(row) != n or any(not math.isfinite(power) for power in row)
           for row in applied):
        raise ValueError("Invalid executed household power trace")
    throughput = [math.fsum(abs(row[i]) * dt for row in applied)
                  for i in range(n)]
    caps = controlled_throughput_caps_kwh(params, nexus="energy_mobility")
    throughput_excess = [max(value - cap, 0.0)
                         for value, cap in zip(throughput, caps, strict=True)]

    result = {
        "operational": {
            "mobility_shortfall_kwh": mobility,
            "normalized_mobility_shortfall": mobility / MOBILITY_REFERENCE_KWH,
            "import_overload_squared_kw2h": grid,
            "normalized_grid_overload": grid / GRID_REFERENCE_KW2H,
            "command_mismatch_kwh": mismatch,
            "normalized_command_mismatch": mismatch / MISMATCH_REFERENCE_KWH,
        },
        "distributive": {
            "household_shortfall_kwh": shortfalls,
            "household_burden_fractions": burdens,
            "worst_above_mean_mobility_burden": max(excesses),
            "mean_above_mean_mobility_burden": math.fsum(excesses) / n,
            "worst_household_burden": max(burdens),
        },
        "intertemporal": {
            "controlled_throughput_reference": MOBILITY_THROUGHPUT_REFERENCE_ID,
            "controlled_throughput_cap_kwh_by_household": caps,
            "controlled_throughput_kwh_by_household": throughput,
            "controlled_throughput_kwh": math.fsum(throughput),
            "terminal_reserve_deficit_kwh_by_household": reserve_deficits,
            "terminal_reserve_deficit_kwh": math.fsum(reserve_deficits),
            "normalized_terminal_reserve_deficit": (
                math.fsum(reserve_deficits) / RESERVE_REFERENCE_KWH
            ),
            "excess_controlled_throughput_kwh_by_household": throughput_excess,
            "excess_controlled_throughput_kwh": math.fsum(throughput_excess),
            "normalized_excess_controlled_throughput": (
                math.fsum(throughput_excess) / RESERVE_REFERENCE_KWH
            ),
        },
    }

    if oracle_bundle is not None:
        oracle = np.asarray(oracle_bundle["actions_kw"], dtype=float)
        reference = np.asarray(oracle_bundle["idle_to_oracle_l2_kw2h"], dtype=float)
        applied = np.asarray([
            [float(h["applied_ev_power_kw"]) for h in info["households"]]
            for info in infos
        ], dtype=float)
        if oracle.shape != applied.shape or reference.shape != (n,) or np.any(reference <= 0):
            raise ValueError("Metric oracle schedule/denominators do not match episode")
        normalized_gaps = np.square(applied - oracle).sum(axis=0) * dt / reference
        alignments = np.clip(1.0 - normalized_gaps, 0.0, 1.0)
        alignment_burdens = 1.0 - alignments
        result["recognitional"] = {
            "household_normalized_squared_gaps": normalized_gaps.tolist(),
            "household_clipped_alignments": alignments.tolist(),
            "mean_clipped_oracle_alignment": float(alignments.mean()),
            "worst_above_mean_alignment_burden": float(
                np.maximum(alignment_burdens - alignment_burdens.mean(), 0.0).max()
            ),
            "reference_oracle_sha256": oracle_bundle["sha256"],
        }

    raw_count = executed_count = opportunity_count = 0
    shadow_consistency = []
    for info in infos:
        t = int(info["t"]) - 1  # reports describe the action selected before transition t
        enforced = "procedural" in info.get("applied_justice_modifiers", [])
        if enforced != ("procedural" in info):
            raise ValueError("Procedural enforcement trace presence disagrees with formulation")
        if enforced:
            trace = info["procedural"].get("trace")
            if not isinstance(trace, dict) or "raw_aggregator_request_kw" not in trace:
                raise ValueError("Procedural enforcement requires the pre-gate raw trace")
            raw = trace["raw_aggregator_request_kw"]
        else:
            raw = info["constraints"]["requested_ev_power_kw"]
        if len(raw) != n or any(not math.isfinite(float(power)) for power in raw):
            raise ValueError("Invalid pre-gate request vector")
        raw = [float(power) for power in raw]
        actual = [float(h["applied_ev_power_kw"]) for h in info["households"]]
        _, report = apply_procedural_rules(
            raw, params, decision_step=t,
        )
        prohibited = [bound.min_kw >= 0.0 for bound in report.authority_bounds_kw]
        opportunity_count += sum(prohibited)
        raw_count += sum(flag and power < -UNAUTHORIZED_DISCHARGE_EPSILON_KW
                         for flag, power in zip(prohibited, raw, strict=True))
        executed_count += sum(flag and power < -UNAUTHORIZED_DISCHARGE_EPSILON_KW
                              for flag, power in zip(prohibited, actual, strict=True))
        shadow_consistency.append(1.0 - report.procedural_consistency_loss)
    result["procedural"] = {
        "authority_reference": "scenario_declared_household_procedural_rules_v1",
        "opportunity_definition": "authority_bounds_min_kw_nonnegative_at_pre_action_time",
        "discharge_threshold_kw": -UNAUTHORIZED_DISCHARGE_EPSILON_KW,
        "prohibited_request_count": raw_count,
        "prohibited_execution_count": executed_count,
        "prohibited_opportunity_count": opportunity_count,
        "prohibited_request_rate": raw_count / opportunity_count if opportunity_count else None,
        "prohibited_execution_rate": executed_count / opportunity_count if opportunity_count else None,
        "shadow_authority_resolved_consistency_score": math.fsum(shadow_consistency) / len(infos),
        "mean_stepwise_discharge_consistency_score": math.fsum(shadow_consistency) / len(infos),
        "authority_application": "enforced" if all("procedural" in info for info in infos) else "shadow",
    }

    for group in result.values():
        for value in group.values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Nonfinite paper metric")
    return result


def mean_episode_metrics(episodes: list[dict]) -> dict:
    """Average per-day numeric metrics; retain household vectors by position."""
    if not episodes:
        raise ValueError("Cannot average an empty episode set")
    groups = [row["metrics"] for row in episodes]
    result = {}
    for group_name in groups[0]:
        if not all(group_name in group for group in groups):
            continue
        result[group_name] = {}
        for key, first in groups[0][group_name].items():
            values = [group[group_name][key] for group in groups]
            if isinstance(first, (int, float)) and all(isinstance(v, (int, float)) for v in values):
                result[group_name][key] = math.fsum(values) / len(values)
            elif isinstance(first, list) and all(isinstance(v, list) and len(v) == len(first)
                                                  for v in values):
                result[group_name][key] = [math.fsum(row[i] for row in values) / len(values)
                                           for i in range(len(first))]
    return result
