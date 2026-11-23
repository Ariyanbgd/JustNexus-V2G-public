"""Day-matched offline recognitional reward for energy–mobility studies."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from gymnasium import Wrapper, spaces

from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.gym_env.env import JustNexusV2GEnv


class OfflineRecognitionalRewardWrapper(
    Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Adds the normalized applied-action gap to exact offline household schedules."""

    def __init__(
        self,
        env: JustNexusV2GEnv,
        *,
        oracle_actions_kw: Sequence[Sequence[float]],
        idle_to_oracle_gap_kwh: Sequence[float],
        weight: float = 1.0,
        action_gap_aggregation: str = "equal_mean",
        action_gap_reference_kwh: float | None = None,
        action_gap_reference_kw2h: float | None = None,
        gap_distribution: dict[str, float] | None = None,
        include_gap_distribution_observation: bool = True,
        formulation_name: str | None = None,
        applied_justice_modifiers: Sequence[str] | None = None,
    ) -> None:
        super().__init__(env)
        actions = np.asarray(oracle_actions_kw, dtype=float)
        denominators = np.asarray(idle_to_oracle_gap_kwh, dtype=float)
        expected = (env.params.episode_steps - 1, env.params.n_households)
        if actions.shape != expected:
            raise ValueError(
                f"Offline oracle action shape {actions.shape} does not match {expected}."
            )
        if denominators.shape != (env.params.n_households,):
            raise ValueError("One recognitional normalization denominator is required per household.")
        if np.any(~np.isfinite(denominators)) or np.any(denominators <= 0):
            raise ValueError("Recognitional normalization denominators must be finite and positive.")
        self.oracle_actions_kw = actions
        self.idle_to_oracle_gap_kwh = denominators
        self.weight = float(weight)
        if action_gap_aggregation not in {
            "equal_mean", "pooled_sum", "pooled_squared", "household_normalized_squared_sum"
        }:
            raise ValueError(f"Unsupported action-gap aggregation: {action_gap_aggregation}")
        if action_gap_aggregation == "pooled_sum":
            if action_gap_reference_kwh is None or not np.isfinite(action_gap_reference_kwh) or action_gap_reference_kwh <= 0:
                raise ValueError("Pooled action gap requires a positive, finite reference in kWh.")
        if action_gap_aggregation == "pooled_squared":
            if action_gap_reference_kw2h is None or not np.isfinite(action_gap_reference_kw2h) or action_gap_reference_kw2h <= 0:
                raise ValueError("Pooled squared action gap requires a positive, finite reference in kW²h.")
        self.action_gap_aggregation = action_gap_aggregation
        self.action_gap_reference_kwh = (
            float(action_gap_reference_kwh) if action_gap_reference_kwh is not None else None
        )
        self.action_gap_reference_kw2h = (
            float(action_gap_reference_kw2h) if action_gap_reference_kw2h is not None else None
        )
        self.idle_to_oracle_squared_gap_kw2h = (
            np.square(actions).sum(axis=0) * float(env.params.dt_hours)
        )
        if action_gap_aggregation in {
            "pooled_squared", "household_normalized_squared_sum"
        } and np.any(
            self.idle_to_oracle_squared_gap_kw2h <= 0
        ):
            raise ValueError("Squared household alignment requires nonzero oracle action for every household.")
        self.gap_distribution = gap_distribution is not None
        self.include_gap_distribution_observation = (
            self.gap_distribution and bool(include_gap_distribution_observation)
        )
        if self.gap_distribution:
            config = gap_distribution or {}
            self.gap_worst_reference = float(config.get("worst_reference", 0.8))
            self.gap_excess_reference = float(config.get("excess_reference", 0.24))
            self.gap_worst_weight = float(config.get("worst_weight", 1.0))
            self.gap_excess_weight = float(config.get("excess_weight", 1.0))
            gap_cap = config.get("household_gap_cap")
            self.gap_distribution_household_cap = (
                float(gap_cap) if gap_cap is not None else None
            )
            if self.gap_worst_reference <= 0 or self.gap_excess_reference <= 0:
                raise ValueError("Gap-distribution references must be positive.")
            if not np.isfinite([
                self.gap_worst_reference, self.gap_excess_reference,
                self.gap_worst_weight, self.gap_excess_weight,
            ]).all():
                raise ValueError("Gap-distribution references and weights must be finite.")
            if (self.gap_distribution_household_cap is not None
                    and (not np.isfinite(self.gap_distribution_household_cap)
                         or self.gap_distribution_household_cap <= 0)):
                raise ValueError("Gap-distribution household cap must be finite and positive.")
            if self.include_gap_distribution_observation:
                self.observation_space = spaces.Box(
                    low=np.concatenate((
                        env.observation_space.low,
                        np.zeros(2, dtype=np.float32),
                    )).astype(np.float32),
                    high=np.concatenate((
                        env.observation_space.high,
                        np.full(2, np.inf, dtype=np.float32),
                    )).astype(np.float32),
                    dtype=np.float32,
                )
        self.formulation_name = formulation_name
        self.applied_justice_modifiers = (
            list(applied_justice_modifiers)
            if applied_justice_modifiers is not None
            else None
        )
        self._step_index = 0
        self._cumulative_gap_kwh = np.zeros(env.params.n_households, dtype=float)
        self._cumulative_squared_gap_kw2h = np.zeros(env.params.n_households, dtype=float)

    @property
    def params(self) -> SimulationParams:
        return self.env.params

    @property
    def simulator(self):
        return self.env.simulator

    @property
    def last_step_result(self):
        return self.env.last_step_result

    @property
    def config(self):
        return self.env.config

    def reset(self, **kwargs):
        self._step_index = 0
        self._cumulative_gap_kwh = np.zeros(self.env.params.n_households, dtype=float)
        self._cumulative_squared_gap_kw2h = np.zeros(self.env.params.n_households, dtype=float)
        obs, info = self.env.reset(**kwargs)
        return self._with_gap_distribution_observation(obs), info

    def _normalized_household_gaps(self) -> np.ndarray:
        if self.action_gap_aggregation in {
            "pooled_squared", "household_normalized_squared_sum"
        }:
            return self._cumulative_squared_gap_kw2h / self.idle_to_oracle_squared_gap_kw2h
        return self._cumulative_gap_kwh / self.idle_to_oracle_gap_kwh

    def _gap_distribution(self) -> tuple[float, float, float, float]:
        gaps = self._normalized_household_gaps()
        if self.gap_distribution_household_cap is not None:
            gaps = np.minimum(gaps, self.gap_distribution_household_cap)
        excesses = np.maximum(gaps - float(np.mean(gaps)), 0.0)
        worst = float(np.max(excesses))
        excess = float(np.mean(excesses))
        return (
            worst, excess,
            worst / self.gap_worst_reference,
            excess / self.gap_excess_reference,
        )

    def _with_gap_distribution_observation(self, obs: np.ndarray) -> np.ndarray:
        if not self.include_gap_distribution_observation:
            return obs
        _, _, normalized_worst, normalized_excess = self._gap_distribution()
        return np.concatenate((
            obs, np.asarray([normalized_worst, normalized_excess], dtype=np.float32)
        ))

    def step(self, action: np.ndarray):
        if self._step_index >= len(self.oracle_actions_kw):
            raise RuntimeError("Offline oracle schedule is shorter than the environment episode.")
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        result = self.env.last_step_result
        if result is None:
            raise RuntimeError("Wrapped environment did not expose its step result.")
        oracle = self.oracle_actions_kw[self._step_index]
        applied = np.asarray(result.constraints.applied_ev_power_kw, dtype=float)
        gaps_kwh = np.abs(applied - oracle) * float(self.env.params.dt_hours)
        squared_gaps_kw2h = np.square(applied - oracle) * float(self.env.params.dt_hours)
        normalized = (
            squared_gaps_kw2h / self.idle_to_oracle_squared_gap_kw2h
            if self.action_gap_aggregation in {
                "pooled_squared", "household_normalized_squared_sum"
            }
            else gaps_kwh / self.idle_to_oracle_gap_kwh
        )
        if self.action_gap_aggregation == "pooled_sum":
            normalized_gap = float(np.sum(gaps_kwh) / self.action_gap_reference_kwh)
        elif self.action_gap_aggregation == "pooled_squared":
            normalized_gap = float(np.sum(squared_gaps_kw2h) / self.action_gap_reference_kw2h)
        elif self.action_gap_aggregation == "household_normalized_squared_sum":
            normalized_gap = float(np.sum(normalized))
        else:
            normalized_gap = float(np.mean(normalized))
        recognitional_reward = -self.weight * normalized_gap
        reward = float(base_reward) + recognitional_reward
        self._cumulative_gap_kwh += gaps_kwh
        self._cumulative_squared_gap_kw2h += squared_gaps_kw2h
        if self.gap_distribution:
            worst, excess, normalized_worst, normalized_excess = self._gap_distribution()
            finalized = bool(terminated or truncated)
            worst_reward = -self.gap_worst_weight * normalized_worst if finalized else 0.0
            excess_reward = -self.gap_excess_weight * normalized_excess if finalized else 0.0
            reward += worst_reward + excess_reward
            obs = self._with_gap_distribution_observation(obs)
        scores = np.clip(
            1.0 - self._normalized_household_gaps(),
            0.0,
            1.0,
        )
        info = dict(info)
        if self.formulation_name is not None:
            info["formulation"] = self.formulation_name
        if self.applied_justice_modifiers is not None:
            info["applied_justice_modifiers"] = list(
                self.applied_justice_modifiers
            )
        terms = dict(info.get("reward_terms", {}))
        terms.update(
            recognitional_action_gap_kwh=float(np.sum(gaps_kwh)),
            recognitional_squared_action_gap_kw2h=float(np.sum(squared_gaps_kw2h)),
            normalized_recognitional_action_gap=normalized_gap,
            recognitional_action_reward=recognitional_reward,
            recognitional_reward=float(terms.get("recognitional_reward", 0.0)) + recognitional_reward,
        )
        if self.gap_distribution:
            terms.update(
                worst_recognitional_gap=worst,
                excess_recognitional_gap=excess,
                normalized_worst_recognitional_gap=normalized_worst,
                normalized_excess_recognitional_gap=normalized_excess,
                worst_recognitional_gap_reward=worst_reward,
                excess_recognitional_gap_reward=excess_reward,
                recognitional_gap_distribution_reward=worst_reward + excess_reward,
                recognitional_gap_distribution_finalized=float(finalized),
            )
        terms["total_reward"] = reward
        info["reward_terms"] = terms
        oracle_info = {
            "step_index": self._step_index,
            "oracle_action_kw": oracle.tolist(),
            "applied_action_kw": applied.tolist(),
            "action_gap_kwh": gaps_kwh.tolist(),
            "squared_action_gap_kw2h": squared_gaps_kw2h.tolist(),
            "normalized_action_gap": normalized.tolist(),
            "cumulative_action_gap_kwh": self._cumulative_gap_kwh.tolist(),
            "cumulative_squared_action_gap_kw2h": self._cumulative_squared_gap_kw2h.tolist(),
            "household_squared_gap_reference_kw2h": self.idle_to_oracle_squared_gap_kw2h.tolist(),
            "household_score": scores.tolist(),
            "weight": self.weight,
            "action_gap_aggregation": self.action_gap_aggregation,
            "action_gap_reference_kwh": self.action_gap_reference_kwh,
            "action_gap_reference_kw2h": self.action_gap_reference_kw2h,
        }
        if self.action_gap_aggregation == "pooled_squared":
            oracle_info["pooled_normalized_squared_action_gap"] = float(
                np.sum(self._cumulative_squared_gap_kw2h) / self.action_gap_reference_kw2h
            )
        elif self.action_gap_aggregation == "household_normalized_squared_sum":
            oracle_info["summed_household_normalized_squared_action_gap"] = float(
                np.sum(self._cumulative_squared_gap_kw2h / self.idle_to_oracle_squared_gap_kw2h)
            )
        else:
            oracle_info["score"] = float(np.mean(scores))
        if "recognitional_offline_oracle" in info:
            info["recognitional_battery_oracle"] = info["recognitional_offline_oracle"]
        info["recognitional_offline_oracle"] = oracle_info
        if self.gap_distribution:
            info["recognitional_offline_oracle"]["gap_distribution"] = {
                "worst": worst,
                "excess": excess,
                "normalized_worst": normalized_worst,
                "normalized_excess": normalized_excess,
                "finalized": finalized,
            }
        self._step_index += 1
        return obs, reward, terminated, truncated, info


class OfflineBatteryTrajectoryGapWrapper(
    Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Penalize pooled squared battery-energy distance from household oracle paths."""

    def __init__(
        self,
        env: JustNexusV2GEnv,
        *,
        oracle_energy_kwh: Sequence[Sequence[float]],
        household_reference_kwh2h: Sequence[float],
        reference_kwh2h: float,
        weight: float = 1.0,
        formulation_name: str | None = None,
        applied_justice_modifiers: Sequence[str] | None = None,
    ) -> None:
        super().__init__(env)
        energy = np.asarray(oracle_energy_kwh, dtype=float)
        references = np.asarray(household_reference_kwh2h, dtype=float)
        expected = (env.params.episode_steps - 1, env.params.n_households)
        if energy.shape != expected or not np.isfinite(energy).all():
            raise ValueError(f"Oracle battery-energy shape must be {expected} and finite.")
        if references.shape != (env.params.n_households,) or not np.isfinite(references).all() or np.any(references <= 0):
            raise ValueError("One positive, finite battery-gap reference is required per household.")
        if not np.isfinite(reference_kwh2h) or reference_kwh2h <= 0 or not np.isclose(
            reference_kwh2h, references.sum(), rtol=1e-9
        ):
            raise ValueError("Pooled battery-gap reference must equal the household reference sum.")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Battery-gap weight must be finite and nonnegative.")
        self.oracle_energy_kwh = energy
        self.household_reference_kwh2h = references
        self.reference_kwh2h = float(reference_kwh2h)
        self.weight = float(weight)
        self.formulation_name = formulation_name
        self.applied_justice_modifiers = (
            list(applied_justice_modifiers) if applied_justice_modifiers is not None else None
        )
        self._step_index = 0
        self._cumulative_gap_kwh2h = np.zeros(env.params.n_households, dtype=float)

    @property
    def params(self) -> SimulationParams:
        return self.env.params

    @property
    def simulator(self):
        return self.env.simulator

    @property
    def last_step_result(self):
        return self.env.last_step_result

    @property
    def config(self):
        return self.env.config

    def reset(self, **kwargs):
        self._step_index = 0
        self._cumulative_gap_kwh2h.fill(0.0)
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        if self._step_index >= len(self.oracle_energy_kwh):
            raise RuntimeError("Offline oracle battery trajectory is shorter than the episode.")
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        result = self.env.last_step_result
        if result is None:
            raise RuntimeError("Wrapped environment did not expose its step result.")
        actual = np.asarray([h.ev_energy_kwh for h in result.state.households], dtype=float)
        oracle = self.oracle_energy_kwh[self._step_index]
        gaps = np.square(actual - oracle) * float(self.env.params.dt_hours)
        normalized_gap = float(gaps.sum() / self.reference_kwh2h)
        recognitional_reward = -self.weight * normalized_gap
        self._cumulative_gap_kwh2h += gaps
        info = dict(info)
        if self.formulation_name is not None:
            info["formulation"] = self.formulation_name
        if self.applied_justice_modifiers is not None:
            info["applied_justice_modifiers"] = list(self.applied_justice_modifiers)
        terms = dict(info.get("reward_terms", {}))
        terms.update(
            recognitional_battery_gap_kwh2h=float(gaps.sum()),
            normalized_recognitional_battery_gap=normalized_gap,
            recognitional_battery_reward=recognitional_reward,
            recognitional_reward=recognitional_reward,
        )
        info["reward_terms"] = terms
        info["recognitional_offline_oracle"] = {
            "step_index": self._step_index,
            "oracle_energy_kwh": oracle.tolist(),
            "actual_energy_kwh": actual.tolist(),
            "battery_gap_kwh2h": gaps.tolist(),
            "cumulative_battery_gap_kwh2h": self._cumulative_gap_kwh2h.tolist(),
            "household_reference_kwh2h": self.household_reference_kwh2h.tolist(),
            "household_score": np.clip(
                1.0 - self._cumulative_gap_kwh2h / self.household_reference_kwh2h,
                0.0, 1.0,
            ).tolist(),
            "pooled_normalized_battery_gap": float(
                self._cumulative_gap_kwh2h.sum() / self.reference_kwh2h
            ),
            "weight": self.weight,
            "battery_gap_reference_kwh2h": self.reference_kwh2h,
        }
        self._step_index += 1
        return observation, float(base_reward) + recognitional_reward, terminated, truncated, info
