"""Deterministic, explicit-day evaluation and validation checkpoint selection."""

from __future__ import annotations

import json
import math
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from justnexus_v2g.rl.problem import RLProblem
from justnexus_v2g.rl.metrics import episode_metrics, mean_episode_metrics


def _write_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def evaluate_split(model, problem: RLProblem, *, seed: int,
                   include_metrics: bool = False,
                   metric_oracles: list[dict] | None = None) -> dict:
    """Visit each day in the selected split exactly once, without sampling days."""
    mode = problem.provenance["mode"]
    day_ids = (
        [record["day_id"] for record in problem.provenance["day_records"]]
        if mode == "variable" else ["fixed_reference"]
    )
    episodes = []
    for index, expected_day_id in enumerate(day_ids):
        options = {"day_index": index} if mode == "variable" else None
        obs, info = problem.env.reset(seed=seed + index, options=options)
        day_id = info.get("day_id", "fixed_reference")
        if day_id != expected_day_id:
            raise RuntimeError(f"Requested {expected_day_id}, received {day_id}")
        episode_return = 0.0
        steps = 0
        infos = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = problem.env.step(action)
            episode_return += float(reward)
            steps += 1
            if include_metrics:
                infos.append(step_info)
            if terminated or truncated:
                break
        if not math.isfinite(episode_return):
            raise ValueError(f"Nonfinite evaluation return on {day_id}")
        episode = {"day_id": day_id, "steps": steps, "episode_return": episode_return}
        if include_metrics:
            params = problem.env.days[index] if mode == "variable" else problem.env.unwrapped.params
            oracle = metric_oracles[index] if metric_oracles is not None else None
            episode["metrics"] = episode_metrics(infos, params, oracle_bundle=oracle)
        episodes.append(episode)
    result = {
        "split": problem.provenance["split"],
        "deterministic": True,
        "episodes": episodes,
        "mean_episode_return": sum(item["episode_return"] for item in episodes) / len(episodes),
    }
    if include_metrics:
        result["mean_metrics"] = mean_episode_metrics(episodes)
    return result


def evaluate_epistemic_candidate(model, problem: RLProblem, *, seed: int,
                                 include_metrics: bool = False,
                                 metric_oracles: list[dict] | None = None) -> dict:
    """Evaluate the two declared Pareto axes on every selected split day."""
    if problem.provenance["justice"] != ["epistemic"]:
        raise ValueError("Candidate evaluation requires an epistemic problem")
    mode = problem.provenance["mode"]
    day_ids = (
        [record["day_id"] for record in problem.provenance["day_records"]]
        if mode == "variable" else ["fixed_reference"]
    )
    episodes = []
    for index, expected_id in enumerate(day_ids):
        options = {"day_index": index} if mode == "variable" else None
        obs, info = problem.env.reset(seed=seed + index, options=options)
        if info.get("day_id", "fixed_reference") != expected_id:
            raise RuntimeError(f"Requested {expected_id}, received a different day")
        result = {"day_id": expected_id, "steps": 0, "episode_return": 0.0,
                  "mobility_shortfall_kwh": 0.0, "import_overload_squared_kw2h": 0.0,
                  "command_mismatch_kwh": 0.0}
        infos = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = problem.env.step(action)
            terms = step_info["reward_terms"]
            result["episode_return"] += float(reward)
            result["mobility_shortfall_kwh"] += float(terms["departure_energy_shortfall_kwh"])
            result["import_overload_squared_kw2h"] += float(terms["import_overload_squared_kw2h"])
            result["command_mismatch_kwh"] += float(terms["command_mismatch_kwh"])
            result["steps"] += 1
            if include_metrics:
                infos.append(step_info)
            if terminated or truncated:
                break
        if not all(math.isfinite(value) for key, value in result.items() if key not in {"day_id", "steps"}):
            raise ValueError(f"Nonfinite epistemic evaluation on {expected_id}")
        if include_metrics:
            params = problem.env.days[index] if mode == "variable" else problem.env.unwrapped.params
            oracle = metric_oracles[index] if metric_oracles is not None else None
            result["metrics"] = episode_metrics(infos, params, oracle_bundle=oracle)
        episodes.append(result)
    summary = {
        "split": problem.provenance["split"], "deterministic": True,
        "episodes": episodes,
        "mean_episode_return": sum(row["episode_return"] for row in episodes) / len(episodes),
        "mean_mobility_shortfall_kwh": sum(row["mobility_shortfall_kwh"] for row in episodes) / len(episodes),
        "mean_import_overload_squared_kw2h": sum(row["import_overload_squared_kw2h"] for row in episodes) / len(episodes),
        "mean_command_mismatch_kwh": sum(row["command_mismatch_kwh"] for row in episodes) / len(episodes),
    }
    if include_metrics:
        summary["mean_metrics"] = mean_episode_metrics(episodes)
    return summary


class ValidationCheckpointCallback(BaseCallback):
    """Select global best validation return and one best checkpoint per step window."""

    def __init__(self, *, problem: RLProblem, output: Path, seed: int,
                 interval_steps: int, window_steps: int,
                 planned_total_timesteps: int | None = None):
        super().__init__()
        if problem.provenance["split"] != "validation":
            raise ValueError("Checkpoint selection requires a validation problem")
        if interval_steps <= 0 or window_steps <= 0:
            raise ValueError("Evaluation interval and window length must be positive")
        self.problem = problem
        self.output = output
        self.seed = seed
        self.interval_steps = interval_steps
        self.window_steps = window_steps
        self.planned_total_timesteps = planned_total_timesteps
        self.next_evaluation_step = interval_steps
        self.evaluations: list[dict] = []
        self.windows: dict[int, dict] = {}
        self.best_mean_return = -math.inf
        self.best_step: int | None = None

    def _on_training_start(self) -> None:
        (self.output / "window_checkpoints").mkdir(parents=True, exist_ok=True)

    def _evaluate(self) -> None:
        result = evaluate_split(self.model, self.problem, seed=self.seed + 10_000)
        step = int(self.num_timesteps)
        score = float(result["mean_episode_return"])
        result["step"] = step
        self.evaluations.append(result)
        if score > self.best_mean_return:
            self.best_mean_return = score
            self.best_step = step
            self.model.save(self.output / "best_validation_model.zip")

        effective_step = min(step, self.planned_total_timesteps) if self.planned_total_timesteps else step
        window_index = (effective_step - 1) // self.window_steps
        window_start = window_index * self.window_steps
        window_end = window_start + self.window_steps
        previous = self.windows.get(window_index)
        if previous is None or score >= previous["mean_episode_return"] - 1e-6:
            checkpoint = self.output / "window_checkpoints" / (
                f"window_{window_start:07d}_{window_end:07d}.zip"
            )
            self.model.save(checkpoint)
            self.windows[window_index] = {
                "window_start_step": window_start,
                "window_end_step": window_end,
                "selected_step": step,
                "mean_episode_return": score,
                "checkpoint_path": str(checkpoint.relative_to(self.output)),
            }
        self._save_manifest()

    def _save_manifest(self) -> None:
        _write_json(self.output / "validation.json", {
            "selection_metric": "mean_episode_return",
            "tie_break": "first_strict_global_best; latest_within_1e-6_per_window",
            "interval_steps": self.interval_steps,
            "checkpoint_window_steps": self.window_steps,
            "best_step": self.best_step,
            "best_mean_episode_return": self.best_mean_return,
            "evaluations": self.evaluations,
            "windows": [self.windows[key] for key in sorted(self.windows)],
        })

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_evaluation_step:
            self._evaluate()
            while self.next_evaluation_step <= self.num_timesteps:
                self.next_evaluation_step += self.interval_steps
        return True

    def _on_training_end(self) -> None:
        if not self.evaluations or self.evaluations[-1]["step"] != self.num_timesteps:
            self._evaluate()
