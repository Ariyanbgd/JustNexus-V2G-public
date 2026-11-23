"""Declared mobility/grid priority sweep for the epistemic study."""

from __future__ import annotations

import math


def validate_priority_sweep(config: dict, base_weights: dict) -> tuple[list[float], list[int]]:
    """Validate the explicit priority/seed choices without imposing a grid size."""
    if not isinstance(config, dict) or set(config) != {"priority_weights", "training_seeds", "evaluation_seed"}:
        raise ValueError("Epistemic config needs priority_weights, training_seeds, and evaluation_seed")
    priorities = config["priority_weights"]
    seeds = config["training_seeds"]
    if not isinstance(priorities, list) or not priorities:
        raise ValueError("Epistemic priority_weights must be a nonempty list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Epistemic training_seeds must be a nonempty list")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or not 0 <= x <= 1 for x in priorities):
        raise ValueError("Every epistemic priority weight must be finite and in [0, 1]")
    if len(set(map(float, priorities))) != len(priorities):
        raise ValueError("Epistemic priority weights must be distinct")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError("Epistemic training seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Epistemic training seeds must be distinct")
    evaluation_seed = config["evaluation_seed"]
    if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int) or evaluation_seed < 0:
        raise ValueError("Epistemic evaluation_seed must be a nonnegative integer")
    if base_weights["unmet_departure_energy"] + base_weights["grid_constraint"] <= 0:
        raise ValueError("Epistemic mobility and grid base weights must have positive total")
    return [float(x) for x in priorities], list(seeds)


def scalarized_reward_weights(base_weights: dict, priority: float) -> dict:
    """Return base-term weights for one declared mobility/grid priority."""
    total = base_weights["unmet_departure_energy"] + base_weights["grid_constraint"]
    return {
        "unmet_departure_energy": total * priority,
        "grid_constraint": total * (1.0 - priority),
        "total_ev_clipping": base_weights["total_ev_clipping"],
    }


def mark_non_dominated(rows: list[dict], *, flag_key: str = "validation_non_dominated") -> None:
    """Mark candidates using mean raw mobility shortfall and squared overload."""
    for row in rows:
        shortfall = row["mean_mobility_shortfall_kwh"]
        overload = row["mean_import_overload_squared_kw2h"]
        row[flag_key] = not any(
            other is not row
            and other["mean_mobility_shortfall_kwh"] <= shortfall + 1e-9
            and other["mean_import_overload_squared_kw2h"] <= overload + 1e-9
            and (other["mean_mobility_shortfall_kwh"] < shortfall - 1e-9
                 or other["mean_import_overload_squared_kw2h"] < overload - 1e-9)
            for other in rows
        )
