"""Build the selected energy–mobility RL problem from supplied inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.gym_env import GymEnvConfig, JustNexusV2GEnv
from justnexus_v2g.gym_env.formulations import (
    ENERGY_MOBILITY_DISTRIBUTIVE_SYSTEM_OBSERVATION_FIELDS,
    ENERGY_MOBILITY_HOUSEHOLD_OBSERVATION_FIELDS,
    ENERGY_MOBILITY_SYSTEM_OBSERVATION_FIELDS,
    ENERGY_INTERTEMPORAL_HOUSEHOLD_OBSERVATION_FIELDS,
    ENERGY_PROCEDURAL_HOUSEHOLD_OBSERVATION_FIELDS,
)
from justnexus_v2g.gym_env.rewards import RewardNormalization, RewardWeights
from justnexus_v2g.gym_env.procedural import UNAUTHORIZED_DISCHARGE_EPSILON_KW
from justnexus_v2g.rl.day_suite import generate_day_splits
from justnexus_v2g.rl.distributive import DistributiveTerminalRewardEnv, TYPES, distributive_term
from justnexus_v2g.rl.epistemic import scalarized_reward_weights, validate_priority_sweep
from justnexus_v2g.rl.recognitional import prepare_oracle_bundle, resolve_preferences
from justnexus_v2g.gym_env.recognitional import OfflineRecognitionalRewardWrapper

BASE_REWARD_WEIGHT_NAMES = frozenset({
    "unmet_departure_energy", "grid_constraint", "total_ev_clipping",
})
BASE_REWARD_NORMALIZATION = {
    "departure_energy_shortfall_kwh": 74.0,
    "import_overload_squared_kw2h": 267.9392226234346,
    "command_mismatch_kwh": 160.0,
}
BASE_GRID_OPERATION = "soft_import_unlimited_export"
INTERTEMPORAL_NORMALIZATION = {
    "excess_daily_battery_throughput_kwh": 28.5,
    "terminal_battery_reserve_deficit_kwh": 28.5,
}
INTERTEMPORAL_WEIGHT_NAMES = frozenset({
    "excess_daily_battery_throughput", "battery_reserve_deficit",
})
PROCEDURAL_WEIGHT_NAMES = frozenset({
    "procedural_consistency", "unauthorized_discharge_incidence",
})

@dataclass
class RLProblem:
    env: gym.Env
    training: dict
    evaluation: dict
    provenance: dict


def _validate_day(params) -> None:
    requirements = [h.mobility.required_departure_energy_kwh for h in params.households]
    if (params.n_households != 5 or params.dt_hours != 0.5
            or params.system.max_import_kw != 9.0
            or requirements != [20.0, 14.0, 12.0, 14.0, 14.0]):
        raise ValueError("Day does not match the paper-aligned energy–mobility base profile")


def _study_env(params, distributive: dict | None, base_weights: dict,
               recognitional: dict | None = None, oracle_bundle: dict | None = None,
               intertemporal: dict | None = None, combined: dict | None = None,
               procedural: dict | None = None) -> gym.Env:
    _validate_day(params)
    # RewardWeights has nonzero defaults for other formulations. This study
    # activates only the three configurable base terms.
    weights = {name: 0.0 for name in RewardWeights.__dataclass_fields__}
    weights.update(base_weights)
    if intertemporal is not None:
        weights.update(intertemporal["weights"])
    if procedural is not None:
        weights.update(procedural["weights"])
    if distributive is not None:
        # The study wrapper owns the selected terminal term. The native
        # distributive modifier supplies the historical 54-value observation.
        weights.update(worst_mobility_burden=0.0, excess_mobility_burden=0.0)
    if combined is not None:
        weights.update(
            worst_mobility_burden=combined["mobility_disparity_weights"]["worst_burden_weight"],
            excess_mobility_burden=combined["mobility_disparity_weights"]["mean_excess_burden_weight"],
        )
    if distributive is not None or combined is not None:
        native_justice = ("distributive",)
    elif intertemporal is not None:
        native_justice = ("intertemporal",)
    elif procedural is not None:
        native_justice = ("procedural",)
    else:
        native_justice = ()
    core = JustNexusV2GEnv(
        params=params,
        config=GymEnvConfig(
            agent_role="aggregator", nexus="energy_mobility",
            justice=native_justice,
        ),
        reward_weights=RewardWeights(**weights),
        reward_normalization=RewardNormalization(**{
            **BASE_REWARD_NORMALIZATION,
            **(INTERTEMPORAL_NORMALIZATION if intertemporal is not None else {}),
        }),
        grid_operation=BASE_GRID_OPERATION,
    )
    if distributive is not None:
        return DistributiveTerminalRewardEnv(core, distributive)
    if recognitional is not None or combined is not None:
        if oracle_bundle is None:
            raise ValueError("Recognitional environment needs a day-matched oracle bundle")
        recognitional_config = combined if combined is not None else recognitional
        gap_distribution = None
        if combined is not None:
            gap_distribution = {
                "worst_reference": 0.8,
                "excess_reference": 0.24,
                "household_gap_cap": 1.0,
                "worst_weight": combined["alignment_disparity_weights"]["worst_gap_weight"],
                "excess_weight": combined["alignment_disparity_weights"]["mean_excess_gap_weight"],
            }
        return OfflineRecognitionalRewardWrapper(
            core,
            oracle_actions_kw=oracle_bundle["actions_kw"],
            idle_to_oracle_gap_kwh=oracle_bundle["idle_to_oracle_l1_kwh"],
            weight=recognitional_config["action_gap_weight"],
            action_gap_aggregation="household_normalized_squared_sum",
            gap_distribution=gap_distribution,
            formulation_name=(
                "energy_mobility_distributive_recognitional_household_gap_mean_v1"
                if combined is not None else "energy_mobility_recognitional"
            ),
            applied_justice_modifiers=(
                ["distributive", "recognitional"] if combined is not None else ["recognitional"]
            ),
        )
    return core


class VariableDayEnv(gym.Env):
    """Sample a train/validation/test day while preserving the base interface."""

    metadata = {"render_modes": []}

    def __init__(self, days, day_ids: list[str], distributive: dict | None,
                 base_weights: dict,
                 *, sampling_seed: int,
                 recognitional: dict | None = None,
                 oracle_bundles: list[dict] | None = None,
                 intertemporal: dict | None = None,
                 combined: dict | None = None,
                 procedural: dict | None = None):
        super().__init__()
        if not days or len(days) != len(day_ids):
            raise ValueError("days and day_ids must be nonempty and aligned")
        self.days = tuple(days)
        self.day_ids = tuple(day_ids)
        self.distributive = distributive
        self.base_weights = base_weights
        self.recognitional = recognitional
        self.intertemporal = intertemporal
        self.combined = combined
        self.procedural = procedural
        self.oracle_bundles = tuple(oracle_bundles) if oracle_bundles is not None else None
        if (recognitional is not None or combined is not None) and (
            self.oracle_bundles is None or len(self.oracle_bundles) != len(days)
        ):
            raise ValueError("One oracle bundle is required for each variable day")
        self.sampling_seed = int(sampling_seed)
        self._day_rng = np.random.default_rng(self.sampling_seed)
        self.core = _study_env(self.days[0], distributive, base_weights, recognitional,
                               self.oracle_bundles[0] if self.oracle_bundles else None,
                               intertemporal, combined, procedural)
        self.observation_space = self.core.observation_space
        self.action_space = self.core.action_space
        first = self.days[0]
        for index, day in enumerate(self.days[1:], start=1):
            _validate_day(day)
            if day.episode_steps != first.episode_steps or day.cycle_steps != first.cycle_steps:
                raise ValueError("Days must share episode and cycle lengths")
            other = _study_env(day, distributive, base_weights, recognitional,
                               self.oracle_bundles[index] if self.oracle_bundles else None,
                               intertemporal, combined, procedural)
            try:
                if (other.observation_space.shape != self.observation_space.shape
                        or not np.array_equal(other.action_space.low, self.action_space.low)
                        or not np.array_equal(other.action_space.high, self.action_space.high)):
                    raise ValueError("Days must share observation and action spaces")
            finally:
                other.close()
        self.current_day_index = 0
        self.sampled_day_ids: list[str] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._day_rng = np.random.default_rng(
                np.random.SeedSequence([self.sampling_seed, int(seed)])
            )
        options = options or {}
        index = int(options["day_index"]) if "day_index" in options else int(
            self._day_rng.integers(len(self.days))
        )
        if not 0 <= index < len(self.days):
            raise IndexError("day_index outside split")
        episode_seed = int(self._day_rng.integers(0, 2**32 - 1))
        self.core.close()
        self.core = _study_env(
            self.days[index], self.distributive, self.base_weights, self.recognitional,
            self.oracle_bundles[index] if self.oracle_bundles else None,
            self.intertemporal, self.combined, self.procedural,
        )
        self.current_day_index = index
        day_id = self.day_ids[index]
        self.sampled_day_ids.append(day_id)
        observation, info = self.core.reset(seed=episode_seed)
        return observation, {**info, "day_id": day_id, "day_index": index}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.core.step(action)
        return observation, reward, terminated, truncated, {
            **info, "day_id": self.day_ids[self.current_day_index],
            "day_index": self.current_day_index,
        }

    def close(self):
        self.core.close()


def make_rl_problem(*, mode: str, nexus: str, configuration: dict,
                    source_params: SimulationParams, justice=None,
                    distributive_type: str | None = None,
                    preference_profile: str | None = None,
                    epistemic_priority: float | None = None,
                    oracle_cache_dir: str | Path | None = None,
                    split: str = "train") -> RLProblem:
    """Build the requested problem from supplied configuration and simulator parameters."""
    if nexus != "energy_mobility":
        raise ValueError("Current registry supports energy_mobility only")
    if justice in (None, (), []):
        justice_name = None
    elif justice in ("distributive", ("distributive",), ["distributive"]):
        justice_name = "distributive"
    elif justice in ("recognitional", ("recognitional",), ["recognitional"]):
        justice_name = "recognitional"
    elif justice in ("intertemporal", ("intertemporal",), ["intertemporal"]):
        justice_name = "intertemporal"
    elif justice in ("procedural", ("procedural",), ["procedural"]):
        justice_name = "procedural"
    elif justice in ("epistemic", ("epistemic",), ["epistemic"]):
        justice_name = "epistemic"
    elif justice in ("distributive_recognitional", ("distributive", "recognitional"), ["distributive", "recognitional"]):
        justice_name = "distributive_recognitional"
    else:
        raise ValueError("Current registry supports base, distributive, procedural, recognitional, intertemporal, epistemic, or distributive_recognitional justice")
    if justice_name is None and distributive_type is not None:
        raise ValueError("distributive_type requires distributive justice")
    if justice_name == "distributive" and distributive_type not in TYPES:
        raise ValueError(f"distributive_type must be one of {TYPES}")
    if justice_name == "recognitional" and not preference_profile:
        raise ValueError("preference_profile is required for recognitional justice")
    if justice_name != "recognitional" and preference_profile is not None:
        raise ValueError("preference_profile requires recognitional justice")
    if justice_name != "epistemic" and epistemic_priority is not None:
        raise ValueError("epistemic_priority requires epistemic justice")
    if mode not in ("fixed", "variable"):
        raise ValueError("mode must be fixed or variable")
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    profile = configuration["base"]
    mode_config = configuration["mode"]
    sources = configuration["sources"]
    if set(profile) != {"training", "reward_weights"} or mode_config["mode"] != mode:
        raise ValueError("Resolved config does not match requested problem")
    base_weights = profile["reward_weights"]
    if not isinstance(base_weights, dict) or set(base_weights) != BASE_REWARD_WEIGHT_NAMES:
        raise ValueError("Base reward needs exactly the three active term weights")
    for name, value in base_weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Base reward weight {name} must be finite and nonnegative")
    distributive_settings = configuration.get("distributive")
    distributive = None
    recognitional = configuration.get("recognitional")
    intertemporal = configuration.get("intertemporal")
    procedural = configuration.get("procedural")
    epistemic = configuration.get("epistemic")
    combined = configuration.get("combined")
    if justice_name == "distributive":
        if distributive_settings is None:
            raise ValueError("Distributive settings missing")
        if distributive_type == "generalized_gini":
            if set(distributive_settings) != {"weight", "rank_weights_low_to_high"}:
                raise ValueError("Gini settings require weight and rank_weights_low_to_high")
            distributive = {"operator": distributive_type, **distributive_settings}
        elif distributive_type == "mean_deviation":
            if set(distributive_settings) != {"worst_burden_weight", "mean_excess_burden_weight"}:
                raise ValueError("Mean deviation needs worst_burden_weight and mean_excess_burden_weight")
            distributive = {"operator": distributive_type, **distributive_settings}
        elif set(distributive_settings) != {"weight"}:
            raise ValueError(f"{distributive_type} settings require only weight")
        else:
            distributive = {"operator": distributive_type, **distributive_settings}
        distributive_term(
            [0.0] * source_params.n_households,
            [float(h.mobility.required_departure_energy_kwh) for h in source_params.households],
            distributive,
        )
    elif distributive_settings is not None:
        raise ValueError("Base problem cannot receive a distributive config")
    if justice_name == "recognitional":
        if recognitional is None or set(recognitional) != {"preferences", "resolved_preferences", "action_gap_weight"}:
            raise ValueError("Recognitional config needs preferences and action_gap_weight")
        gap_weight = recognitional["action_gap_weight"]
        if isinstance(gap_weight, bool) or not isinstance(gap_weight, (int, float)) or not math.isfinite(gap_weight) or gap_weight < 0:
            raise ValueError("Recognitional action_gap_weight must be finite and nonnegative")
        labels = configuration["scenario_labels"]
        if len(labels) != source_params.n_households or (
            recognitional["resolved_preferences"] != resolve_preferences(recognitional, labels)
        ):
            raise ValueError("Recognitional preferences do not match scenario labels")
    elif recognitional is not None:
        raise ValueError("Only recognitional problem may receive a recognitional config")
    if justice_name == "intertemporal":
        if (not isinstance(intertemporal, dict) or set(intertemporal) != {"weights"}
                or not isinstance(intertemporal["weights"], dict)
                or set(intertemporal["weights"]) != INTERTEMPORAL_WEIGHT_NAMES
        ):
            raise ValueError("Intertemporal config needs exactly the two term weights")
        for name, value in intertemporal["weights"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Intertemporal weight {name} must be finite and nonnegative")
    elif intertemporal is not None:
        raise ValueError("Only intertemporal problem may receive intertemporal weights")
    if justice_name == "procedural":
        if (not isinstance(procedural, dict) or set(procedural) != {"weights"}
                or not isinstance(procedural["weights"], dict)
                or set(procedural["weights"]) != PROCEDURAL_WEIGHT_NAMES):
            raise ValueError("Procedural config needs exactly the two paper reward weights")
        for name, value in procedural["weights"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Procedural weight {name} must be finite and nonnegative")
    elif procedural is not None:
        raise ValueError("Only procedural problem may receive procedural weights")
    if justice_name == "epistemic":
        priorities, _ = validate_priority_sweep(epistemic, base_weights)
        if epistemic_priority is None or isinstance(epistemic_priority, bool) or not isinstance(epistemic_priority, (int, float)) or not math.isfinite(epistemic_priority) or float(epistemic_priority) not in priorities:
            raise ValueError("epistemic_priority must be one of the configured priority_weights")
        selected_weights = scalarized_reward_weights(base_weights, float(epistemic_priority))
    else:
        if epistemic is not None:
            raise ValueError("Only epistemic problem may receive an epistemic config")
        selected_weights = base_weights
    if justice_name == "distributive_recognitional":
        if not isinstance(combined, dict) or set(combined) != {
            "preferences", "resolved_preferences", "action_gap_weight",
            "mobility_disparity_weights", "alignment_disparity_weights",
        }:
            raise ValueError("Combined config needs preferences, action gap, and both disparity weight pairs")
        weight_groups = (
            (combined["mobility_disparity_weights"],
             {"worst_burden_weight", "mean_excess_burden_weight"}),
            (combined["alignment_disparity_weights"],
             {"worst_gap_weight", "mean_excess_gap_weight"}),
        )
        for group, expected in weight_groups:
            if not isinstance(group, dict) or set(group) != expected:
                raise ValueError(f"Combined disparity weights need exactly {sorted(expected)}")
            for name, value in group.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Combined weight {name} must be finite and nonnegative")
        gap_weight = combined["action_gap_weight"]
        if isinstance(gap_weight, bool) or not isinstance(gap_weight, (int, float)) or not math.isfinite(gap_weight) or gap_weight < 0:
            raise ValueError("Combined action_gap_weight must be finite and nonnegative")
        labels = configuration["scenario_labels"]
        if len(labels) != source_params.n_households or (
            combined["resolved_preferences"] != resolve_preferences(combined, labels)
        ):
            raise ValueError("Combined preferences do not match scenario labels")
    elif combined is not None:
        raise ValueError("Only combined problem may receive a combined config")
    oracle_config = combined if combined is not None else recognitional
    source = source_params
    _validate_day(source)
    if combined is not None:
        expected_observation_size = 56
        reward_profile = "energy_mobility_distributive_recognitional_household_gap_mean_v1"
    elif distributive is not None:
        expected_observation_size = 54
        reward_profile = "energy_mobility_three_terms_v1"
    elif intertemporal is not None:
        expected_observation_size = 62
        reward_profile = "energy_mobility_intertemporal_v1"
    elif procedural is not None:
        expected_observation_size = 57
        reward_profile = "energy_mobility_procedural_v1"
    elif epistemic is not None:
        expected_observation_size = 52
        reward_profile = "energy_mobility_epistemic_priority_sweep_v1"
    else:
        expected_observation_size = 52
        reward_profile = "energy_mobility_three_terms_v1"
    provenance = {
        "mode": mode,
        "nexus": nexus,
        "justice": (["distributive", "recognitional"] if combined is not None else
                    ([justice_name] if justice_name is not None else [])),
        "distributive_type": distributive_type,
        "preference_profile": preference_profile,
        "epistemic_priority": epistemic_priority,
        "split": split,
        "base_config": sources["base_config"],
        "base_config_sha256": sources["base_config_sha256"],
        "mode_config": sources["mode_config"],
        "mode_config_sha256": sources["mode_config_sha256"],
        "scenario": sources["scenario"],
        "scenario_sha256": sources["scenario_sha256"],
        **({
            "metric_reference_config": sources["metric_reference_config"],
            "metric_reference_config_sha256": sources["metric_reference_config_sha256"],
            "metric_reference_profile": configuration["metric_reference"]["profile"],
        } if "metric_reference" in configuration else {}),
        "observation": {
            "system_fields": list(ENERGY_MOBILITY_SYSTEM_OBSERVATION_FIELDS) + (
                list(ENERGY_MOBILITY_DISTRIBUTIVE_SYSTEM_OBSERVATION_FIELDS)
                if distributive is not None or combined is not None else []
            ) + (
                ["normalized_cumulative_worst_recognitional_gap",
                 "normalized_cumulative_excess_recognitional_gap"]
                if combined is not None else []
            ),
            "household_fields": list(ENERGY_MOBILITY_HOUSEHOLD_OBSERVATION_FIELDS) + (
                list(ENERGY_INTERTEMPORAL_HOUSEHOLD_OBSERVATION_FIELDS)
                if intertemporal is not None else []
            ) + (
                list(ENERGY_PROCEDURAL_HOUSEHOLD_OBSERVATION_FIELDS)
                if procedural is not None else []
            ),
            "size": expected_observation_size,
        },
        "reward": {
            "profile": reward_profile,
            "weights": {**selected_weights, **(procedural["weights"] if procedural is not None else {}), **({
                "worst_mobility_burden": combined["mobility_disparity_weights"]["worst_burden_weight"],
                "excess_mobility_burden": combined["mobility_disparity_weights"]["mean_excess_burden_weight"],
            } if combined is not None else {})},
            "normalization": {
                **BASE_REWARD_NORMALIZATION,
                **(INTERTEMPORAL_NORMALIZATION if intertemporal is not None else {}),
            },
            "shortfall_timing": "each_household_departure",
            "grid_operation": BASE_GRID_OPERATION,
        },
    }
    if distributive is not None:
        if "justice_config" in sources:
            provenance["justice_config"] = sources["justice_config"]
            provenance["justice_config_sha256"] = sources["justice_config_sha256"]
        provenance["reward"]["distributive"] = dict(distributive)
    if recognitional is not None:
        provenance["justice_config"] = sources["justice_config"]
        provenance["justice_config_sha256"] = sources["justice_config_sha256"]
        provenance["reward"]["recognitional"] = {
            "profile": "household_normalized_squared_gap_mean_v1",
            "normalization": "per_household_idle_to_oracle_l2_kw2h",
            "aggregation": "household_normalized_squared_sum",
            "weight": recognitional["action_gap_weight"],
            "training_clip": False,
        }
        provenance["preferences"] = recognitional["resolved_preferences"]
    if intertemporal is not None:
        provenance["justice_config"] = sources["justice_config"]
        provenance["justice_config_sha256"] = sources["justice_config_sha256"]
        provenance["reward"]["intertemporal"] = {
            "weights": dict(intertemporal["weights"]),
            "normalization": dict(INTERTEMPORAL_NORMALIZATION),
            "throughput_timing": "incremental_excess_each_step",
            "reserve_timing": "terminal_once",
        }
    if procedural is not None:
        provenance["justice_config"] = sources["justice_config"]
        provenance["justice_config_sha256"] = sources["justice_config_sha256"]
        provenance["reward"]["procedural"] = {
            "weights": dict(procedural["weights"]),
            "gate": "hard_discharge_authorization_before_physical_limits",
            "reward_horizon_steps": source.episode_steps - 1,
            "prohibited_request_threshold_kw": -UNAUTHORIZED_DISCHARGE_EPSILON_KW,
        }
    if epistemic is not None:
        provenance["justice_config"] = sources["justice_config"]
        provenance["justice_config_sha256"] = sources["justice_config_sha256"]
        provenance["reward"]["epistemic"] = {
            "priority_weight": float(epistemic_priority),
            "mobility_grid_weight_total": base_weights["unmet_departure_energy"] + base_weights["grid_constraint"],
            "candidate_priorities": list(priorities),
            "training_seeds": list(epistemic["training_seeds"]),
            "pareto_axes": ["departure_energy_shortfall_kwh", "import_overload_squared_kw2h"],
        }
    if combined is not None:
        provenance["justice_config"] = sources["justice_config"]
        provenance["justice_config_sha256"] = sources["justice_config_sha256"]
        provenance["preferences"] = combined["resolved_preferences"]
        provenance["reward"]["combined"] = {
            "mobility_disparity_weights": dict(combined["mobility_disparity_weights"]),
            "action_gap_weight": combined["action_gap_weight"],
            "alignment_disparity_weights": dict(combined["alignment_disparity_weights"]),
            "alignment_gap_cap": 1.0,
            "worst_reference": 0.8,
            "mean_excess_reference": 0.24,
            "mobility_disparity_timing": "last_departure_once",
            "alignment_disparity_timing": "episode_end_once",
        }
    cache = Path(oracle_cache_dir) if oracle_cache_dir is not None else None
    if mode == "fixed":
        bundle = (prepare_oracle_bundle(source, oracle_config["resolved_preferences"], cache_dir=cache)
                  if oracle_config is not None else None)
        env = _study_env(source, distributive, selected_weights, recognitional, bundle, intertemporal, combined, procedural)
        provenance["day_ids"] = ["fixed_reference"]
        if bundle is not None:
            provenance["oracle_bundles"] = {"fixed_reference": bundle["sha256"]}
    else:
        splits, generation = generate_day_splits(
            source, mode_config["days"],
            variation=configuration.get("day_variation"),
            household_labels=configuration.get("scenario_labels"),
        )
        selected = splits[split]
        for day, _ in selected:
            _validate_day(day)
        bundles = (
            [prepare_oracle_bundle(day, oracle_config["resolved_preferences"], cache_dir=cache)
             for day, _ in selected]
            if oracle_config is not None else None
        )
        if bundles is not None:
            for (_, record), bundle in zip(selected, bundles, strict=True):
                if bundle["day_sha256"] != record["sha256"]:
                    raise RuntimeError(f"Oracle bundle does not match day {record['day_id']}")
        env = VariableDayEnv(
            [day for day, _ in selected],
            [record["day_id"] for _, record in selected],
            distributive,
            selected_weights,
            sampling_seed=int(mode_config["days"]["sampling_seed"]),
            recognitional=recognitional,
            oracle_bundles=bundles,
            intertemporal=intertemporal,
            combined=combined,
            procedural=procedural,
        )
        provenance["day_generation"] = generation
        provenance["day_records"] = [record for _, record in selected]
        if bundles is not None:
            provenance["oracle_bundles"] = {
                record["day_id"]: bundle["sha256"]
                for (_, record), bundle in zip(selected, bundles, strict=True)
            }
    expected_shape = (expected_observation_size,)
    if env.observation_space.shape != expected_shape:
        env.close()
        raise RuntimeError(f"Unexpected energy–mobility observation shape: {env.observation_space.shape}")
    evaluation = dict(mode_config["evaluation"])
    if (int(evaluation["interval_steps"]) <= 0
            or int(evaluation["checkpoint_window_steps"]) <= 0):
        env.close()
        raise ValueError("Evaluation interval and checkpoint window must be positive")
    return RLProblem(
        env=env,
        training=dict(profile["training"]),
        evaluation=evaluation,
        provenance=provenance,
    )
