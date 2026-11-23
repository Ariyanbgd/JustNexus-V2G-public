"""Original-timing distributive rewards for the energy–mobility study."""

from __future__ import annotations

import math

import gymnasium as gym

TYPES = ("mean_deviation", "max_min", "generalized_gini")
WORST_BURDEN_DIVISOR = 0.8
EXCESS_BURDEN_DIVISOR = 0.24


def distributive_term(shortfalls, requirements, reward_config: dict) -> tuple[float, dict]:
    """Return the selected dimensionless term from all five departure outcomes."""
    if len(shortfalls) != 5 or len(requirements) != 5:
        raise ValueError("Distributive study requires five household outcomes")
    if any(not math.isfinite(float(q)) or float(q) <= 0 for q in requirements):
        raise ValueError("Departure requirements must be positive and finite")
    if any(not math.isfinite(float(x)) or float(x) < 0 for x in shortfalls):
        raise ValueError("Departure shortfalls must be nonnegative and finite")
    burdens = [min(float(x) / float(q), 1.0) for x, q in zip(shortfalls, requirements, strict=True)]
    satisfactions = [1.0 - b for b in burdens]
    kind = reward_config["operator"]
    if kind == "mean_deviation":
        mean_burden = sum(burdens) / 5
        excess = [max(b - mean_burden, 0.0) for b in burdens]
        normalized_worst = max(excess) / WORST_BURDEN_DIVISOR
        normalized_excess = (sum(excess) / 5) / EXCESS_BURDEN_DIVISOR
        term = -normalized_worst - normalized_excess
        details = {
            "normalized_worst_excess": normalized_worst,
            "normalized_mean_excess": normalized_excess,
        }
    elif kind == "max_min":
        term = -max(burdens)
        details = {"worst_burden_fraction": max(burdens)}
    elif kind == "generalized_gini":
        weights = [float(value) for value in reward_config["rank_weights_low_to_high"]]
        if (len(weights) != 5 or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-9)):
            raise ValueError("Five nonnegative Gini rank weights must sum to one")
        ranked_welfare = sum(w * s for w, s in zip(weights, sorted(satisfactions), strict=True))
        term = ranked_welfare - 1.0
        details = {"ranked_satisfaction_welfare": ranked_welfare}
    else:
        raise ValueError(f"Unknown distributive type: {kind}")
    return float(term), {"burden_fractions": burdens, "satisfaction_fractions": satisfactions, **details}


class DistributiveTerminalRewardEnv(gym.Wrapper):
    """Keep native 54 observations and add one chosen term at the last departure."""

    def __init__(self, env: gym.Env, reward_config: dict):
        super().__init__(env)
        self.reward_config = dict(reward_config)
        self.operator = self.reward_config["operator"]
        if self.operator not in TYPES:
            raise ValueError(f"Unknown distributive type: {self.operator}")
        weight_names = (
            ("worst_burden_weight", "mean_excess_burden_weight")
            if self.operator == "mean_deviation" else ("weight",)
        )
        for name in weight_names:
            value = self.reward_config[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Distributive {name} must be finite and nonnegative")
        self.coefficient = float(self.reward_config["weight"]) if self.operator != "mean_deviation" else None
        self._event_count = 0

    def reset(self, *, seed=None, options=None):
        self._event_count = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, base_reward, terminated, truncated, original_info = self.env.step(action)
        info = dict(original_info)
        terms = dict(info["reward_terms"])
        active = bool(terms["mobility_distribution_finalized"])
        operator_value = 0.0
        details = None
        if active:
            if self._event_count:
                raise RuntimeError("Distributive reward may be paid only once per episode")
            requirements = [
                float(h.mobility.required_departure_energy_kwh)
                for h in self.env.params.households
            ]
            operator_value, details = distributive_term(
                info["household_departure_shortfall_kwh"], requirements, self.reward_config
            )
            self._event_count = 1
        if (terminated or truncated) and self._event_count != 1:
            raise RuntimeError("Complete distributive episode needs one last-departure reward")
        if self.operator == "mean_deviation" and details is not None:
            worst_reward = -self.reward_config["worst_burden_weight"] * details["normalized_worst_excess"]
            excess_reward = -self.reward_config["mean_excess_burden_weight"] * details["normalized_mean_excess"]
            distributive_reward = worst_reward + excess_reward
        else:
            distributive_reward = (self.coefficient or 0.0) * operator_value
        reward = float(base_reward + distributive_reward)
        terms["distributive_reward"] = distributive_reward
        terms["distributive_operator_value"] = operator_value
        terms["total_reward"] = reward
        if self.operator == "mean_deviation" and details is not None:
            terms["worst_mobility_burden_reward"] = worst_reward
            terms["excess_mobility_burden_reward"] = excess_reward
        info["reward_terms"] = terms
        info["reward_formulation"] = f"energy_mobility_distributive_{self.operator}_additive_v1"
        info["distributive_type"] = self.operator
        info["distributive_operator_parameters"] = dict(self.reward_config)
        info["core_reward_weights"] = info["reward_weights"]
        active_weights = dict(info["reward_weights"])
        if self.operator == "mean_deviation":
            active_weights["worst_mobility_burden"] = self.reward_config["worst_burden_weight"]
            active_weights["excess_mobility_burden"] = self.reward_config["mean_excess_burden_weight"]
        else:
            active_weights[self.operator] = self.coefficient
        info["reward_weights"] = active_weights
        if details is not None:
            info["distributive_operator_details"] = details
        info["stakeholder_rewards"] = {**info["stakeholder_rewards"], "distribution": distributive_reward}
        return observation, reward, terminated, truncated, info
