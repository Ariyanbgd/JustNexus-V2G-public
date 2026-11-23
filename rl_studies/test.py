"""Evaluate selected PPO models and saved epistemic candidates on test days."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
ORACLE_CACHE = REPOSITORY / "rl_studies" / "oracle_cache"
sys.path.insert(0, str(REPOSITORY / "justnexus_v2g" / "src"))

from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.epistemic import mark_non_dominated
from justnexus_v2g.rl.evaluation import evaluate_epistemic_candidate, evaluate_split
from justnexus_v2g.rl.metrics import (
    GRID_REFERENCE_KW2H, MISMATCH_REFERENCE_KWH, MOBILITY_REFERENCE_KWH,
    RESERVE_REFERENCE_KWH, prepare_metric_oracles,
)
from justnexus_v2g.core.config import load_simulation_params


def _metric_references(configuration: dict) -> dict:
    return {
        "mobility_shortfall_kwh": MOBILITY_REFERENCE_KWH,
        "import_overload_squared_kw2h": GRID_REFERENCE_KW2H,
        "command_mismatch_kwh": MISMATCH_REFERENCE_KWH,
        "terminal_reserve_and_throughput_kwh": RESERVE_REFERENCE_KWH,
        "oracle_preference_profile": configuration.get("metric_reference", {}).get("profile"),
    }


def _test_epistemic_portfolio(output: Path, record: dict, configuration: dict,
                              scenario_path: Path) -> None:
    """Test every saved candidate; preserve validation selection separately."""
    from stable_baselines3 import PPO

    source = record["problem"]
    selection = record["model_selection"]
    portfolio_bytes = (output / selection["validation_portfolio"]).read_bytes()
    if hashlib.sha256(portfolio_bytes).hexdigest() != selection["validation_portfolio_sha256"]:
        raise ValueError("Validation portfolio changed since training")
    portfolio = json.loads(portfolio_bytes)
    candidates = portfolio["candidates"]
    selected = portfolio["validation_front"]
    if len(candidates) != selection["candidate_count"] or len(selected) != selection["selected_count"]:
        raise ValueError("Validation portfolio counts differ from run record")
    # Recompute the front from the saved validation axes; test values cannot
    # introduce or remove a candidate.
    mark_non_dominated(candidates)
    expected_front = [row["checkpoint"] for row in candidates if row["validation_non_dominated"]]
    if selected != expected_front:
        raise ValueError("Validation portfolio front does not match candidate axes")
    results = []
    scenario = load_simulation_params(scenario_path)
    metric_oracles = None
    for candidate in candidates:
        checkpoint = output / candidate["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != candidate["checkpoint_sha256"]:
            raise ValueError(f"Candidate checkpoint changed: {candidate['checkpoint']}")
        problem = make_rl_problem(
            mode=source["mode"], nexus=source["nexus"], justice="epistemic",
            split="test", epistemic_priority=candidate["priority_weight"],
            configuration=configuration, source_params=scenario,
        )
        try:
            for key in ("base_config_sha256", "mode_config_sha256",
                        "justice_config_sha256", "scenario_sha256",
                        "metric_reference_config_sha256"):
                if key not in source:
                    continue
                if problem.provenance[key] != source[key]:
                    raise ValueError(f"{key} changed since training")
            if metric_oracles is None and "metric_reference" in configuration:
                metric_oracles = prepare_metric_oracles(
                    problem, configuration["metric_reference"]["resolved_preferences"],
                    cache_dir=ORACLE_CACHE,
                )
            model = PPO.load(checkpoint, device="cpu")
            if (model.observation_space != problem.env.observation_space
                    or model.action_space != problem.env.action_space):
                raise ValueError("Candidate checkpoint has mismatched spaces")
            test_result = evaluate_epistemic_candidate(
                model, problem, seed=int(configuration["epistemic"]["evaluation_seed"]) + 10_000,
                include_metrics=True, metric_oracles=metric_oracles,
            )
            results.append({
                "checkpoint": candidate["checkpoint"],
                "validation_selected": candidate["checkpoint"] in selected,
                "priority_weight": candidate["priority_weight"],
                "training_seed": candidate["training_seed"],
                "window": candidate["window"],
                "validation_mobility_shortfall_kwh": candidate["mean_mobility_shortfall_kwh"],
                "validation_import_overload_squared_kw2h": candidate["mean_import_overload_squared_kw2h"],
                **test_result,
            })
        finally:
            problem.env.close()
    test_axes = [{
        "checkpoint": row["checkpoint"],
        "mean_mobility_shortfall_kwh": row["mean_mobility_shortfall_kwh"],
        "mean_import_overload_squared_kw2h": row["mean_import_overload_squared_kw2h"],
    } for row in results]
    mark_non_dominated(test_axes, flag_key="test_non_dominated")
    test_front = [row["checkpoint"] for row in test_axes if row["test_non_dominated"]]
    for result_row, axes_row in zip(results, test_axes, strict=True):
        result_row["test_non_dominated"] = axes_row["test_non_dominated"]
    result = {
        "study_kind": "epistemic_priority_sweep",
        "trained_justice": source["justice"],
        "selected_by": "validation_pareto_non_dominance",
        "selection_axes": portfolio["pareto_axes"],
        "validation_candidate_count": len(candidates),
        "validation_selected_count": len(selected),
        "validation_front": selected,
        "test_candidate_count": len(results),
        "test_front": test_front,
        "test_front_count": len(test_front),
        "test_front_role": "descriptive_only; does_not_change_validation_selection",
        "metric_references": _metric_references(configuration),
        "independent_day_split": source["mode"] == "variable",
        "fixed_day_is_in_sample": source["mode"] == "fixed",
        "smoke": bool(record["smoke"]),
        "test_results": results,
    }
    with (output / "test.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output / "test.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a completed run using its saved mode, nexus, justice, and config."
    )
    parser.add_argument("run", type=Path,
                        help="Completed rl_studies training run directory; all study choices come from run.json")
    parser.add_argument("--inspect", action="store_true",
                        help="Show saved choices and planned test size without evaluating models")
    args = parser.parse_args()
    output = args.run.resolve()
    record = json.loads((output / "run.json").read_text())
    if record["status"] != "complete":
        raise ValueError("Test requires a completed training and validation run")
    source = record["problem"]
    mode = source["mode"]
    snapshots = record.get("input_snapshots")
    if snapshots is None:
        raise ValueError("This run predates input snapshots and cannot be tested independently of its original configs")
    config_path = output / snapshots["configuration"]
    scenario_path = output / snapshots["scenario"]
    config_bytes = config_path.read_bytes()
    scenario_bytes = scenario_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != snapshots["configuration_sha256"]:
        raise ValueError("Resolved study config snapshot changed since training")
    if hashlib.sha256(scenario_bytes).hexdigest() != snapshots["scenario_sha256"]:
        raise ValueError("Scenario snapshot changed since training")
    if args.inspect:
        configuration = json.loads(config_bytes)
        selection = record["model_selection"]
        print(json.dumps({
            "run": str(output),
            "mode": source["mode"],
            "nexus": source["nexus"],
            "justice": source["justice"] or ["base"],
            "distributive_type": source.get("distributive_type"),
            "preference_profile": source.get("preference_profile"),
            "test_day_count": (
                configuration["mode"]["days"]["counts"]["test"]
                if mode == "variable" else 1
            ),
            "test_candidate_count": selection.get("candidate_count", 1),
            "validation_selected_count": selection.get("selected_count", 1),
            "test_report_exists": (output / "test.json").exists(),
        }, indent=2, sort_keys=True))
        return
    if (output / "test.json").exists():
        raise FileExistsError("test.json already exists; the reserved test is evaluated once per run")
    if record.get("study_kind") == "epistemic_priority_sweep":
        _test_epistemic_portfolio(output, record, json.loads(config_bytes), scenario_path)
        return
    configuration = json.loads(config_bytes)
    problem = make_rl_problem(
        mode=mode, nexus=source["nexus"], justice=source["justice"], split="test",
        distributive_type=source.get("distributive_type"),
        preference_profile=source.get("preference_profile"),
        oracle_cache_dir=ORACLE_CACHE,
        configuration=configuration,
        source_params=load_simulation_params(scenario_path),
    )
    try:
        keys = ["base_config_sha256", "mode_config_sha256", "scenario_sha256"]
        if "justice_config_sha256" in source:
            keys.append("justice_config_sha256")
        if "metric_reference_config_sha256" in source:
            keys.append("metric_reference_config_sha256")
        for key in keys:
            if problem.provenance[key] != source[key]:
                raise ValueError(f"{key} changed since training; refusing a mismatched test")
        if mode == "fixed" and "oracle_bundles" in source:
            if problem.provenance["oracle_bundles"] != source["oracle_bundles"]:
                raise ValueError("Fixed-day oracle bundle changed since training")
        from stable_baselines3 import PPO

        model = PPO.load(output / record["model_selection"]["selected_model"], device="cpu")
        if model.observation_space != problem.env.observation_space:
            raise ValueError("Selected model observation space differs from test problem")
        if model.action_space != problem.env.action_space:
            raise ValueError("Selected model action space differs from test problem")
        metric_oracles = (
            prepare_metric_oracles(
                problem, configuration["metric_reference"]["resolved_preferences"],
                cache_dir=ORACLE_CACHE,
            ) if "metric_reference" in configuration else None
        )
        result = evaluate_split(
            model, problem, seed=int(record["training"]["seed"]) + 20_000,
            include_metrics=True, metric_oracles=metric_oracles,
        )
    finally:
        problem.env.close()

    result.update({
        "model": record["model_selection"]["selected_model"],
        "trained_justice": source["justice"] or ["base"],
        "selected_by": "mean_validation_episode_return",
        "independent_day_split": mode == "variable",
        "smoke": bool(record["smoke"]),
        "metric_references": _metric_references(configuration),
    })
    if "oracle_bundles" in source:
        result["oracle_bundles"] = problem.provenance["oracle_bundles"]
    with (output / "test.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output / "test.json")


if __name__ == "__main__":
    main()
