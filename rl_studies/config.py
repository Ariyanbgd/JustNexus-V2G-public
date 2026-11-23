"""Load study-owned YAML and hand resolved inputs to the library builder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from justnexus_v2g.core.config import load_simulation_params
from justnexus_v2g.core.params import SimulationParams
from justnexus_v2g.rl.distributive import TYPES as DISTRIBUTIVE_TYPES
from justnexus_v2g.rl.recognitional import resolve_preferences

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_SCENARIO = REPOSITORY / "justnexus_v2g" / "configs" / "scenario_energy_mobility.yaml"

MODE_CHOICES = ("fixed", "variable")
NEXUS_CHOICES = ("energy_mobility",)
JUSTICE_CHOICES = (
    "base", "distributive", "procedural", "recognitional", "intertemporal",
    "epistemic", "distributive_recognitional",
)


@dataclass(frozen=True)
class StudyInputs:
    configuration: dict
    source_params: SimulationParams
    scenario_bytes: bytes


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY))
    except ValueError:
        return str(path)


def available_study_options(*, config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict:
    """List implemented selectors and YAML-defined variants without building an environment."""
    root = Path(config_dir).resolve() / "energy_mobility"
    distributive = yaml.safe_load((root / "distributive" / "config.yaml").read_text())
    recognitional = yaml.safe_load((root / "recognitional" / "config.yaml").read_text())
    epistemic = yaml.safe_load((root / "epistemic" / "config.yaml").read_text())
    return {
        "modes": list(MODE_CHOICES),
        "nexuses": list(NEXUS_CHOICES),
        "justices": list(JUSTICE_CHOICES),
        "distributive_types": [name for name in DISTRIBUTIVE_TYPES
                               if name in distributive["types"]],
        "recognitional_preference_profiles": list(recognitional["profiles"]),
        "epistemic_priority_weights": epistemic["priority_weights"],
        "epistemic_training_seeds": epistemic["training_seeds"],
        "configuration_directory": _display_path(Path(config_dir).resolve()),
    }


def load_study_inputs(*, mode: str, nexus: str, justice=None,
                      distributive_type: str | None = None,
                      preference_profile: str | None = None,
                      config_dir: str | Path = DEFAULT_CONFIG_DIR,
                      scenario_path: str | Path = DEFAULT_SCENARIO) -> StudyInputs:
    """Resolve files here, so the library never needs to locate study configs."""
    if nexus != "energy_mobility" or justice not in (None, (), [], "distributive", "procedural", "recognitional", "intertemporal", "epistemic", "distributive_recognitional"):
        raise ValueError("Current study config supports energy_mobility base, distributive, procedural, recognitional, intertemporal, epistemic, or distributive_recognitional")
    if mode not in ("fixed", "variable"):
        raise ValueError("mode must be fixed or variable")
    if justice == "distributive" and distributive_type not in DISTRIBUTIVE_TYPES:
        raise ValueError(f"distributive_type must be one of {DISTRIBUTIVE_TYPES}")
    if justice != "distributive" and distributive_type is not None:
        raise ValueError("distributive_type requires distributive justice")
    if justice == "recognitional" and not preference_profile:
        raise ValueError("preference_profile is required for recognitional justice")
    if justice != "recognitional" and preference_profile is not None:
        raise ValueError("preference_profile requires recognitional justice")
    root = Path(config_dir).resolve()
    base_path = root / nexus / "base" / "config.yaml"
    scenario_file = Path(scenario_path).resolve()
    base_bytes = base_path.read_bytes()
    base_data = yaml.safe_load(base_bytes)
    if not isinstance(base_data, dict) or set(base_data) != {"reward_weights", "training", "modes", "paper_metric_reference_profile"}:
        raise ValueError("Base config needs reward_weights, training, modes, and paper_metric_reference_profile")
    if not isinstance(base_data["modes"], dict) or set(base_data["modes"]) != {"fixed", "variable"}:
        raise ValueError("Base config needs fixed and variable mode sections")
    scenario_bytes = scenario_file.read_bytes()
    scenario_data = yaml.safe_load(scenario_bytes)
    scenario_labels = [item["label"] for item in scenario_data["households"]]
    reference_path = root / nexus / "recognitional" / "config.yaml"
    reference_bytes = reference_path.read_bytes()
    reference_profiles = yaml.safe_load(reference_bytes)["profiles"]
    reference_name = base_data["paper_metric_reference_profile"]
    if reference_name not in reference_profiles:
        raise ValueError(f"Paper metric reference profile {reference_name} missing from recognitional config")
    reference_preferences = resolve_preferences(reference_profiles[reference_name], scenario_labels)
    configuration = {
        "base": {key: base_data[key] for key in ("reward_weights", "training")},
        "mode": {"mode": mode, **base_data["modes"][mode]},
        "metric_reference": {
            "profile": reference_name,
            "resolved_preferences": reference_preferences,
        },
        "scenario_labels": scenario_labels,
        "sources": {
            "base_config": _display_path(base_path),
            "base_config_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "mode_config": _display_path(base_path),
            "mode_config_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "scenario": _display_path(scenario_file),
            "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
            "metric_reference_config": _display_path(reference_path),
            "metric_reference_config_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        },
    }
    if mode == "variable":
        if "day_variation" not in scenario_data:
            raise ValueError("Variable mode requires day_variation in the scenario YAML")
        configuration["day_variation"] = scenario_data["day_variation"]
    if justice == "distributive":
        variant_path = root / nexus / "distributive" / "config.yaml"
        variant_bytes = variant_path.read_bytes()
        variant = yaml.safe_load(variant_bytes)
        if not isinstance(variant, dict) or set(variant) != {"types"} or not isinstance(variant["types"], dict):
            raise ValueError("Distributive config needs a types mapping")
        if distributive_type not in variant["types"]:
            raise ValueError(f"Distributive type {distributive_type} missing from config")
        configuration["distributive"] = variant["types"][distributive_type]
        configuration["sources"].update({
            "justice_config": _display_path(variant_path),
            "justice_config_sha256": hashlib.sha256(variant_bytes).hexdigest(),
        })
    if justice == "procedural":
        variant_path = root / nexus / "procedural" / "config.yaml"
        variant_bytes = variant_path.read_bytes()
        variant = yaml.safe_load(variant_bytes)
        configuration["procedural"] = variant
        configuration["sources"].update({
            "justice_config": _display_path(variant_path),
            "justice_config_sha256": hashlib.sha256(variant_bytes).hexdigest(),
        })
    if justice == "recognitional":
        variant_path = root / nexus / "recognitional" / "config.yaml"
        variant_bytes = variant_path.read_bytes()
        variant = yaml.safe_load(variant_bytes)
        if not isinstance(variant, dict) or set(variant) != {"profiles"} or not isinstance(variant["profiles"], dict):
            raise ValueError("Recognitional config needs a profiles mapping")
        if preference_profile not in variant["profiles"]:
            raise ValueError(f"Preference profile {preference_profile} missing from config")
        profile = variant["profiles"][preference_profile]
        profile["resolved_preferences"] = resolve_preferences(profile, scenario_labels)
        configuration["recognitional"] = profile
        configuration["sources"].update({
            "justice_config": _display_path(variant_path),
            "justice_config_sha256": hashlib.sha256(variant_bytes).hexdigest(),
        })
    if justice == "intertemporal":
        variant_path = root / nexus / "intertemporal" / "config.yaml"
        variant_bytes = variant_path.read_bytes()
        configuration["intertemporal"] = yaml.safe_load(variant_bytes)
        configuration["sources"].update({
            "justice_config": _display_path(variant_path),
            "justice_config_sha256": hashlib.sha256(variant_bytes).hexdigest(),
        })
    if justice == "epistemic":
        variant_path = root / nexus / "epistemic" / "config.yaml"
        variant_bytes = variant_path.read_bytes()
        configuration["epistemic"] = yaml.safe_load(variant_bytes)
        configuration["sources"].update({
            "justice_config": _display_path(variant_path),
            "justice_config_sha256": hashlib.sha256(variant_bytes).hexdigest(),
        })
    if justice == "distributive_recognitional":
        variant_path = root / nexus / "distributive_recognitional" / "config.yaml"
        variant_bytes = variant_path.read_bytes()
        variant = yaml.safe_load(variant_bytes)
        variant["resolved_preferences"] = resolve_preferences(variant, scenario_labels)
        configuration["combined"] = variant
        configuration["sources"].update({
            "justice_config": _display_path(variant_path),
            "justice_config_sha256": hashlib.sha256(variant_bytes).hexdigest(),
        })
    return StudyInputs(
        configuration=configuration,
        source_params=load_simulation_params(scenario_file),
        scenario_bytes=scenario_bytes,
    )
