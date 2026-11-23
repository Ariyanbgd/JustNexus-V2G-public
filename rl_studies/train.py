"""Readable PPO loop; the library builds the selected RL problem."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "justnexus_v2g" / "src"))

from justnexus_v2g.rl import make_rl_problem
from justnexus_v2g.rl.distributive import TYPES as DISTRIBUTIVE_TYPES
from justnexus_v2g.rl.epistemic import mark_non_dominated, validate_priority_sweep
from justnexus_v2g.rl.evaluation import (ValidationCheckpointCallback,
                                           evaluate_epistemic_candidate)
from rl_studies.config import (
    DEFAULT_CONFIG_DIR, DEFAULT_SCENARIO, JUSTICE_CHOICES, MODE_CHOICES,
    NEXUS_CHOICES, available_study_options, load_study_inputs,
)

# Select the study here, or pass the equivalent command-line options.
# MODE: fixed | variable
# NEXUS: energy_mobility (the implemented scenario)
# JUSTICE: base | distributive | procedural | recognitional | intertemporal | epistemic |
#          distributive_recognitional
# DISTRIBUTIVE_TYPE when JUSTICE=distributive: mean_deviation | max_min |
#                                                generalized_gini
# PREFERENCE_PROFILE when JUSTICE=recognitional: named profile in its config.yaml
# Use --list-options to see the current YAML-defined profiles and priorities.
MODE = "fixed"
NEXUS = "energy_mobility"
JUSTICE = "base"
DISTRIBUTIVE_TYPE = "mean_deviation"
PREFERENCE_PROFILE = "declared_needs_v1"
ORACLE_CACHE = REPOSITORY / "rl_studies" / "oracle_cache"


def _train_epistemic_sweep(args, inputs) -> None:
    """Train scalar priority policies and select a Pareto portfolio on validation."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    configuration = inputs.configuration
    priorities, seeds = validate_priority_sweep(
        configuration["epistemic"], configuration["base"]["reward_weights"]
    )
    if args.inspect:
        problem = make_rl_problem(
            mode=args.mode, nexus=args.nexus, justice="epistemic",
            epistemic_priority=priorities[0], configuration=configuration,
            source_params=inputs.source_params,
        )
        try:
            windows_per_run = (
                int(problem.training["total_timesteps"])
                + int(problem.evaluation["checkpoint_window_steps"]) - 1
            ) // int(problem.evaluation["checkpoint_window_steps"])
            print(json.dumps({
                "problem": problem.provenance,
                "priority_weights": priorities,
                "training_seeds": seeds,
                "training_runs": len(priorities) * len(seeds),
                "candidate_windows_per_run": windows_per_run,
                "candidate_policies": len(priorities) * len(seeds) * windows_per_run,
                "observation_shape": list(problem.env.observation_space.shape),
                "training": problem.training,
                "evaluation": problem.evaluation,
            }, indent=2, sort_keys=True))
        finally:
            problem.env.close()
        return

    # A smoke run checks the portfolio path with two priorities and one seed.
    # It is labeled and snapshotted separately from a full configured sweep.
    active_priorities = priorities[:2] if args.smoke else priorities
    active_seeds = seeds[:1] if args.smoke else seeds
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or REPOSITORY / "rl_studies" / "runs" / args.nexus
              / "epistemic" / args.mode / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_bytes = (json.dumps(configuration, indent=2, sort_keys=True) + "\n").encode()
    (output / "resolved_config.json").write_bytes(config_bytes)
    (output / "scenario.yaml").write_bytes(inputs.scenario_bytes)
    subruns = []
    first_provenance = None
    for priority_index, priority in enumerate(active_priorities):
        for seed in active_seeds:
            run_dir = output / "policies" / f"priority_{priority_index:03d}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=False)
            problem = make_rl_problem(
                mode=args.mode, nexus=args.nexus, justice="epistemic",
                epistemic_priority=priority, configuration=configuration,
                source_params=inputs.source_params,
            )
            validation_problem = make_rl_problem(
                mode=args.mode, nexus=args.nexus, justice="epistemic", split="validation",
                epistemic_priority=priority, configuration=configuration,
                source_params=inputs.source_params,
            )
            if first_provenance is None:
                first_provenance = problem.provenance
            training = dict(problem.training)
            training["seed"] = seed
            evaluation = dict(problem.evaluation)
            if args.smoke:
                training["total_timesteps"] = 64
                training["device"] = "cpu"
                training["ppo"] = {**training["ppo"], "n_steps": 32, "batch_size": 32,
                                   "n_epochs": 1}
                evaluation["interval_steps"] = 32
            monitored = Monitor(problem.env, filename=str(run_dir / "monitor.csv"))
            try:
                model = PPO("MlpPolicy", monitored, seed=seed,
                            device=training["device"], verbose=0, **training["ppo"])
                callback = ValidationCheckpointCallback(
                    problem=validation_problem, output=run_dir, seed=seed,
                    interval_steps=int(evaluation["interval_steps"]),
                    window_steps=int(evaluation["checkpoint_window_steps"]),
                    planned_total_timesteps=int(training["total_timesteps"]),
                )
                model.learn(total_timesteps=int(training["total_timesteps"]), callback=callback)
                model.save(run_dir / "final_model.zip")
                shutil.copy2(run_dir / "best_validation_model.zip", run_dir / "selected_model.zip")
                windows = json.loads((run_dir / "validation.json").read_text())["windows"]
                subruns.append({
                    "priority_weight": priority, "training_seed": seed,
                    "directory": str(run_dir.relative_to(output)),
                    "actual_timesteps": int(model.num_timesteps),
                    "window_count": len(windows),
                    "training": training,
                    "evaluation": evaluation,
                    "reward_weights": problem.provenance["reward"]["weights"],
                    "sampled_training_day_ids": list(getattr(problem.env, "sampled_day_ids", [])),
                })
            finally:
                monitored.close()
                validation_problem.env.close()

    candidates = []
    common_seed = int(configuration["epistemic"]["evaluation_seed"])
    for subrun in subruns:
        validation_problem = make_rl_problem(
            mode=args.mode, nexus=args.nexus, justice="epistemic", split="validation",
            epistemic_priority=subrun["priority_weight"], configuration=configuration,
            source_params=inputs.source_params,
        )
        try:
            run_dir = output / subrun["directory"]
            windows = json.loads((run_dir / "validation.json").read_text())["windows"]
            for index, window in enumerate(windows, start=1):
                checkpoint = run_dir / window["checkpoint_path"]
                model = PPO.load(checkpoint, device="cpu")
                result = evaluate_epistemic_candidate(model, validation_problem,
                                                       seed=common_seed)
                candidates.append({
                    "priority_weight": subrun["priority_weight"],
                    "training_seed": subrun["training_seed"],
                    "window": index,
                    "selected_training_step": window["selected_step"],
                    "checkpoint": str(checkpoint.relative_to(output)),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    **result,
                })
        finally:
            validation_problem.env.close()
    mark_non_dominated(candidates)
    front = [row["checkpoint"] for row in candidates if row["validation_non_dominated"]]
    portfolio_bytes = (json.dumps({
        "selection_split": "validation", "pareto_axes": [
            "mean_mobility_shortfall_kwh", "mean_import_overload_squared_kw2h",
        ],
        "candidates": candidates, "validation_front": front,
    }, indent=2, sort_keys=True) + "\n").encode()
    (output / "validation_portfolio.json").write_bytes(portfolio_bytes)
    (output / "run.json").write_text(json.dumps({
        "status": "complete", "study_kind": "epistemic_priority_sweep",
        "problem": first_provenance, "smoke": bool(args.smoke),
        "subruns": subruns,
        "input_snapshots": {
            "configuration": "resolved_config.json",
            "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "scenario": "scenario.yaml",
            "scenario_sha256": hashlib.sha256(inputs.scenario_bytes).hexdigest(),
        },
        "model_selection": {
            "validation_portfolio": "validation_portfolio.json",
            "validation_portfolio_sha256": hashlib.sha256(portfolio_bytes).hexdigest(),
            "candidate_count": len(candidates),
            "selected_count": len(front),
            "test_rule": "evaluate_all_candidates; report_descriptive_test_front",
        },
    }, indent=2, sort_keys=True) + "\n")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODE_CHOICES, default=MODE,
                        help="Day regime; fixed reuses one day, variable uses separate day splits")
    parser.add_argument("--nexus", choices=NEXUS_CHOICES, default=NEXUS,
                        help="Nexus scenario (currently energy_mobility)")
    parser.add_argument("--justice", choices=JUSTICE_CHOICES, default=JUSTICE,
                        help="Justice formulation; base uses no justice modifier")
    parser.add_argument("--distributive-type", choices=DISTRIBUTIVE_TYPES,
                        help="Required justice variant when --justice distributive")
    parser.add_argument("--preference-profile",
                        help="Named profile inside the recognitional config.yaml")
    parser.add_argument("--list-options", action="store_true",
                        help="Print supported study choices and YAML-defined variants without training")
    parser.add_argument("--inspect", action="store_true", help="Resolve the selected problem without training")
    parser.add_argument("--smoke", action="store_true", help="Run only 64 CPU training steps")
    parser.add_argument("--output", type=Path, help="New run directory; existing paths are refused")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR,
                        help="Directory containing the study YAML configs")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO,
                        help="Simulator scenario YAML supplied to this study")
    args = parser.parse_args()
    if args.list_options:
        print(json.dumps(available_study_options(config_dir=args.config_dir), indent=2))
        return
    justice = None if args.justice == "base" else args.justice
    if justice is None and args.distributive_type is not None:
        parser.error("--distributive-type requires --justice distributive")
    if justice != "recognitional" and args.preference_profile is not None:
        parser.error("--preference-profile requires --justice recognitional")
    distributive_type = (
        args.distributive_type or DISTRIBUTIVE_TYPE
        if justice == "distributive" else None
    )
    preference_profile = (
        args.preference_profile or PREFERENCE_PROFILE
        if justice == "recognitional" else None
    )

    inputs = load_study_inputs(
        mode=args.mode, nexus=args.nexus, justice=justice,
        distributive_type=distributive_type,
        preference_profile=preference_profile,
        config_dir=args.config_dir, scenario_path=args.scenario,
    )
    if justice == "epistemic":
        _train_epistemic_sweep(args, inputs)
        return
    problem = make_rl_problem(
        mode=args.mode, nexus=args.nexus, justice=justice,
        distributive_type=distributive_type,
        preference_profile=preference_profile,
        oracle_cache_dir=ORACLE_CACHE,
        configuration=inputs.configuration, source_params=inputs.source_params,
    )
    env = problem.env
    training = dict(problem.training)
    seed = int(training["seed"])
    try:
        if args.inspect:
            observation, info = env.reset(seed=seed)
            preview = dict(problem.provenance)
            if "day_records" in preview:
                records = preview["day_records"]
                preview["day_records"] = {"count": len(records), "first": records[0]}
            print(json.dumps({
                "problem": preview,
                "training": training,
                "evaluation": problem.evaluation,
                "observation_shape": list(observation.shape),
                "action_low_kw": env.action_space.low.tolist(),
                "action_high_kw": env.action_space.high.tolist(),
                "first_day": info.get("day_id", "fixed_reference"),
            }, indent=2, sort_keys=True))
            return

        if args.smoke:
            training["total_timesteps"] = 64
            training["device"] = "cpu"
            training["ppo"] = {**training["ppo"], "n_steps": 32, "batch_size": 32, "n_epochs": 1}
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        variant_path = Path(justice or "base")
        if distributive_type is not None:
            variant_path /= distributive_type
        if preference_profile is not None:
            variant_path /= preference_profile
        output = (args.output or REPOSITORY / "rl_studies" / "runs" / args.nexus
                  / variant_path / args.mode / f"seed_{seed}" / stamp).resolve()
        output.mkdir(parents=True, exist_ok=False)
        config_bytes = (
            json.dumps(inputs.configuration, indent=2, sort_keys=True) + "\n"
        ).encode()
        (output / "resolved_config.json").write_bytes(config_bytes)
        (output / "scenario.yaml").write_bytes(inputs.scenario_bytes)

        monitored_env = Monitor(env, filename=str(output / "monitor.csv"))
        validation_problem = make_rl_problem(
            mode=args.mode, nexus=args.nexus, justice=justice,
            distributive_type=distributive_type, split="validation",
            preference_profile=preference_profile, oracle_cache_dir=ORACLE_CACHE,
            configuration=inputs.configuration, source_params=inputs.source_params,
        )
        model = PPO(
            "MlpPolicy", monitored_env, seed=seed,
            device=training["device"], verbose=0, **training["ppo"],
        )
        evaluation = dict(problem.evaluation)
        if args.smoke:
            evaluation["interval_steps"] = 32
        callback = ValidationCheckpointCallback(
            problem=validation_problem,
            output=output,
            seed=seed,
            interval_steps=int(evaluation["interval_steps"]),
            window_steps=int(evaluation["checkpoint_window_steps"]),
        )
        try:
            model.learn(total_timesteps=int(training["total_timesteps"]), callback=callback)
        finally:
            validation_problem.env.close()
        model.save(output / "final_model.zip")
        shutil.copy2(output / "best_validation_model.zip", output / "selected_model.zip")

        record = {
            "status": "complete",
            "problem": problem.provenance,
            "training": training,
            "evaluation_config": evaluation,
            "smoke": bool(args.smoke),
            "actual_timesteps": int(model.num_timesteps),
            "n_training_envs": int(model.n_envs),
            "sampled_training_day_ids": list(getattr(env, "sampled_day_ids", [])),
            "input_snapshots": {
                "configuration": "resolved_config.json",
                "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "scenario": "scenario.yaml",
                "scenario_sha256": hashlib.sha256(inputs.scenario_bytes).hexdigest(),
            },
            "model_selection": {
                "metric": "mean_validation_episode_return",
                "best_step": callback.best_step,
                "best_mean_episode_return": callback.best_mean_return,
                "selected_model": "selected_model.zip",
                "source_model": "best_validation_model.zip",
                "final_model": "final_model.zip",
                "validation_manifest": "validation.json",
                "fixed_day_is_in_sample": args.mode == "fixed",
            },
        }
        (output / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        monitored_env.close()
        print(output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
