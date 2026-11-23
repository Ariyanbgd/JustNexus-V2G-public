"""Deterministic in-memory day variation for the energy–mobility scenario."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict

import numpy as np

VERSION = "energy_mobility_days_v2"
LEGACY_VERSION = "energy_mobility_days_v1"
LOAD_FIELDS = frozenset({"base_kw", "morning_peak_kw", "evening_peak_kw"})
PV_FIELDS = frozenset({"peak_kw", "taper_kw_per_step"})


def _factor_range(raw, name: str) -> tuple[float, float]:
    if (not isinstance(raw, (list, tuple)) or len(raw) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   for value in raw)):
        raise ValueError(f"{name} must be a pair of positive numbers")
    low, high = (float(value) for value in raw)
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError(f"{name} must be a positive ordered finite range")
    return low, high


def _selected_fields(raw, allowed: frozenset[str], name: str) -> tuple[str, ...]:
    if (not isinstance(raw, list) or not raw or any(
            not isinstance(field, str) or field not in allowed for field in raw
    ) or len(raw) != len(set(raw))):
        raise ValueError(f"{name} must list distinct supported fields: {sorted(allowed)}")
    return tuple(raw)


def _has_distinct_assignment(options: tuple[tuple[int, ...], ...]) -> bool:
    ordered = sorted(options, key=len)

    def assign(index: int, used: set[int]) -> bool:
        if index == len(ordered):
            return True
        return any(
            step not in used and assign(index + 1, used | {step})
            for step in ordered[index]
        )

    return assign(0, set())


def _scenario_variation(source, variation: dict, labels: list[str]):
    if not isinstance(variation, dict) or set(variation) != {"load", "pv", "departures"}:
        raise ValueError("day_variation needs load, pv, and departures sections")
    if (not isinstance(labels, list) or len(labels) != source.n_households
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("day_variation requires one unique scenario label per household")
    load = variation["load"]
    pv = variation["pv"]
    departures = variation["departures"]
    if (not isinstance(load, dict) or set(load) != {"fields", "factor_range"}
            or not isinstance(pv, dict) or set(pv) != {"fields", "factor_range"}):
        raise ValueError("day_variation load and pv each need fields and factor_range")
    load_fields = _selected_fields(load["fields"], LOAD_FIELDS, "day_variation.load.fields")
    pv_fields = _selected_fields(pv["fields"], PV_FIELDS, "day_variation.pv.fields")
    load_range = _factor_range(load["factor_range"], "day_variation.load.factor_range")
    pv_range = _factor_range(pv["factor_range"], "day_variation.pv.factor_range")
    if (not isinstance(departures, dict)
            or set(departures) != {"all_distinct", "step_options_by_household"}
            or not isinstance(departures["all_distinct"], bool)
            or not isinstance(departures["step_options_by_household"], dict)
            or set(departures["step_options_by_household"]) != set(labels)):
        raise ValueError("day_variation.departures needs all_distinct and options for every scenario label")
    options = []
    for label, household in zip(labels, source.households, strict=True):
        steps = departures["step_options_by_household"][label]
        if (not isinstance(steps, list) or not steps
                or any(isinstance(step, bool) or not isinstance(step, int)
                       or not 0 < step < source.episode_steps - 1
                       or (household.mobility.disconnected_end is not None
                           and step >= household.mobility.disconnected_end)
                       for step in steps)
                or len(steps) != len(set(steps))):
            raise ValueError(f"Invalid departure steps for {label}")
        options.append(tuple(steps))
    ordered_options = tuple(options)
    all_distinct = departures["all_distinct"]
    if all_distinct and not _has_distinct_assignment(ordered_options):
        raise ValueError("No distinct departure assignment is possible")
    return load_fields, pv_fields, load_range, pv_range, ordered_options, all_distinct


def generate_day_splits(
    source, settings: dict, *, variation: dict | None = None,
    household_labels: list[str] | None = None,
):
    """Return generated days and provenance without writing scenario files.

    The bundled scenario recipe reproduces the historical Phase 3B day values.
    Other scenario recipes can select supported fields, ranges, and departures.
    Days are generated in memory, not read from frozen YAML artifacts.
    """
    counts = settings.get("counts", {})
    if set(counts) != {"train", "validation", "test"} or any(
        not isinstance(value, int) or value < 1 for value in counts.values()
    ):
        raise ValueError("days.counts needs positive train, validation, and test sizes")
    seed = int(settings["generator_seed"])
    if seed < 0:
        raise ValueError("generator_seed must be nonnegative")
    if variation is None:
        # Saved runs created before variation moved into the scenario still
        # carry these resolved settings in their immutable config snapshot.
        load_fields = ("base_kw", "morning_peak_kw", "evening_peak_kw")
        pv_fields = ("peak_kw",)
        load_range = _factor_range(settings["load_factor_range"], "load_factor_range")
        pv_range = _factor_range(settings["pv_factor_range"], "pv_factor_range")
        departure_options = tuple(tuple(int(x) for x in group)
                                  for group in settings["departure_options"])
        if len(departure_options) != source.n_households or any(not group for group in departure_options):
            raise ValueError("Departure options must cover every household")
        all_distinct = True
        version = LEGACY_VERSION
    else:
        (load_fields, pv_fields, load_range, pv_range, departure_options,
         all_distinct) = _scenario_variation(source, variation, household_labels)
        version = VERSION
    rng = np.random.default_rng(seed)
    splits = {}
    seen = set()
    for split in ("train", "validation", "test"):
        generated = []
        for number in range(counts[split]):
            day = deepcopy(source)
            for household in day.households:
                load_factor = float(rng.uniform(*load_range))
                pv_factor = float(rng.uniform(*pv_range))
                for field in load_fields:
                    value = getattr(household.load, field)
                    if value is not None:
                        setattr(household.load, field, round(float(value) * load_factor, 6))
                for field in pv_fields:
                    value = getattr(household.pv, field)
                    if value is not None:
                        setattr(household.pv, field, round(float(value) * pv_factor, 6))
            for _ in range(10_000):
                departures = [int(rng.choice(group)) for group in departure_options]
                if not all_distinct or len(set(departures)) == source.n_households:
                    break
            else:
                raise RuntimeError("Could not draw distinct departure times")
            for household, departure in zip(day.households, departures, strict=True):
                household.mobility.departure_step = departure
                household.mobility.disconnected_start = departure
            digest = hashlib.sha256(json.dumps(asdict(day), sort_keys=True).encode()).hexdigest()
            if digest in seen:
                raise RuntimeError("Generated duplicate day across splits")
            seen.add(digest)
            generated.append((day, {
                "day_id": f"{split}_{number:03d}",
                "split": split,
                "sha256": digest,
                "departure_steps": departures,
            }))
        splits[split] = generated
    generation = {"version": version, "generator_seed": seed, **settings}
    if variation is not None:
        generation["variation"] = variation
    return splits, generation
