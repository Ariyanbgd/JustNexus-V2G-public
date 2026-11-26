"""Bounded, path-free provenance for new study artifacts.

Legacy records without this schema remain explicitly unverified. No historical
artifact is rewritten or assigned a guessed source commit.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

SCHEMA = "study-provenance-v1"
DEPENDENCIES = ("justnexus-v2g", "numpy", "gymnasium", "stable-baselines3",
                "cvxpy", "clarabel", "PyYAML", "torch", "pytest")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _imported_library_dir() -> Path:
    return Path(importlib.import_module("justnexus_v2g").__file__).resolve().parent


def source_identity(root: Path) -> dict:
    """Hash a bounded executable tree, including untracked Python source.

    The digest uses relative names and contents; absolute paths and Git status
    filenames never enter the record. A source tree larger than the bound fails
    explicitly instead of silently producing an incomplete fingerprint.
    """
    root = root.resolve()
    library = _imported_library_dir()
    checkout_library = root / "justnexus_v2g" / "src" / "justnexus_v2g"
    if not checkout_library.exists():
        checkout_library = root / "justnexus_v2g"
    library_origin = "checkout" if library == checkout_library.resolve() else "external"
    files = sorted(path for directory in (library, root / "rl_studies")
                   for path in directory.rglob("*.py")
                   if path.is_file() and not any(part in {"runs", "oracle_cache", "__pycache__"}
                                                 for part in path.relative_to(directory).parts))
    if not files:
        raise ValueError("No executable Python source available for fingerprint")
    if len(files) > 2000:
        raise ValueError("Executable source fingerprint exceeds 2000 files")
    digest = hashlib.sha256()
    for path in files:
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Executable source fingerprint contains a file over 10 MiB")
        prefix = "library" if path.is_relative_to(library) else "rl_studies"
        directory = library if prefix == "library" else root / "rl_studies"
        relative = (prefix + "/" + path.relative_to(directory).as_posix()).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    commit = _git(root, "rev-parse", "HEAD")
    git_root = _git(root, "rev-parse", "--show-toplevel")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    available = git_root is not None and Path(git_root).resolve() == root and status is not None
    return {
        "commit": commit if available and commit and len(commit) == 40 else None,
        "working_tree": ("dirty" if status else "clean") if available else "unavailable",
        "fingerprint_sha256": digest.hexdigest(),
        "fingerprint_file_count": len(files),
        "fingerprint_scope": "Imported justnexus_v2g and rl_studies Python files; max 2000 files, 10 MiB each",
        "library_origin": library_origin,
    }


def dependency_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def artifact_hashes(output: Path, relative_paths: list[str]) -> dict[str, str]:
    hashes = {}
    for name in sorted(set(relative_paths)):
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe study artifact")
        hashes[name] = sha256_file(path)
    return hashes


def make_provenance(root: Path, output: Path, artifacts: list[str], problem: dict,
                    source_at_start: dict, dependencies_at_start: dict,
                    validation_oracle_bundles: dict | None = None) -> dict:
    from justnexus_v2g.rl.metrics import (
        GRID_REFERENCE_KW2H, MISMATCH_REFERENCE_KWH, MOBILITY_REFERENCE_KWH,
        RESERVE_REFERENCE_KWH,
    )
    from justnexus_v2g.gym_env.throughput import MOBILITY_THROUGHPUT_REFERENCE_ID

    source_at_end = source_identity(root)
    dependencies_at_end = dependency_versions()
    stable = source_at_start == source_at_end and dependencies_at_start == dependencies_at_end
    return {
        "schema": SCHEMA,
        "source": source_at_start,
        "dependencies": dependencies_at_start,
        "completion_source": source_at_end,
        "completion_dependencies": dependencies_at_end,
        "source_stability": "stable" if stable else "changed_during_run",
        "artifacts_sha256": artifact_hashes(output, artifacts),
        "evaluation_schema": "rl_studies/evaluation-v1",
        "metric_references": {
            "mobility_shortfall_kwh": MOBILITY_REFERENCE_KWH,
            "import_overload_squared_kw2h": GRID_REFERENCE_KW2H,
            "command_mismatch_kwh": MISMATCH_REFERENCE_KWH,
            "terminal_reserve_and_throughput_kwh": RESERVE_REFERENCE_KWH,
            "controlled_throughput_reference": MOBILITY_THROUGHPUT_REFERENCE_ID,
            "metric_reference_config_sha256": problem.get("metric_reference_config_sha256"),
        },
        "oracle_bundles": {
            "train": problem.get("oracle_bundles", "not_applicable"),
            "validation": validation_oracle_bundles or "not_applicable",
        },
    }


def verify_provenance(record: dict, root: Path, output: Path) -> dict:
    if "experimental_provenance" not in record:
        return {"status": "unknown_legacy", "reason": "No experimental_provenance in run.json"}
    saved = record["experimental_provenance"]
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
        raise ValueError("Unsupported experimental provenance schema")
    manifest = saved.get("artifacts_sha256")
    source = saved.get("source")
    if (not isinstance(manifest, dict) or not manifest
            or not isinstance(source, dict)
            or not isinstance(source.get("fingerprint_sha256"), str)
            or not SHA256.fullmatch(source["fingerprint_sha256"])
            or saved.get("source_stability") != "stable"
            or saved.get("completion_source") != source
            or not isinstance(saved.get("dependencies"), dict)
            or saved.get("completion_dependencies") != saved["dependencies"]
            or not isinstance(saved.get("metric_references"), dict)
            or not saved.get("evaluation_schema")
            or "oracle_bundles" not in saved):
        raise ValueError("Incomplete experimental provenance")
    snapshots = record.get("input_snapshots")
    selection = record.get("model_selection")
    if not isinstance(snapshots, dict) or not isinstance(selection, dict):
        raise ValueError("Missing recorded inputs or model selection")
    required = set()
    for label in ("configuration", "scenario"):
        name = snapshots.get(label)
        digest = snapshots.get(label + "_sha256")
        if not isinstance(name, str) or not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("Invalid input snapshot reference")
        required.add(name)
        if manifest.get(name) != digest:
            raise ValueError("Missing or mismatched input snapshot hash")
    if record.get("study_kind") == "epistemic_priority_sweep":
        portfolio_name = selection.get("validation_portfolio")
        portfolio_hash = selection.get("validation_portfolio_sha256")
        if (portfolio_name != "validation_portfolio.json" or not isinstance(portfolio_hash, str)
                or not SHA256.fullmatch(portfolio_hash)
                or manifest.get(portfolio_name) != portfolio_hash):
            raise ValueError("Missing or mismatched portfolio hash")
        required.add(portfolio_name)
        # Verify the portfolio before using its candidate references.
        if artifact_hashes(output, [portfolio_name])[portfolio_name] != portfolio_hash:
            raise ValueError("Recorded portfolio hash changed")
        portfolio = json.loads((output / portfolio_name).read_text())
        candidates = portfolio.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Missing portfolio candidates")
        for candidate in candidates:
            name = candidate.get("checkpoint") if isinstance(candidate, dict) else None
            digest = candidate.get("checkpoint_sha256") if isinstance(candidate, dict) else None
            if (not isinstance(name, str) or not isinstance(digest, str)
                    or not SHA256.fullmatch(digest) or manifest.get(name) != digest):
                raise ValueError("Missing or mismatched candidate checkpoint hash")
            required.add(name)
        subruns = record.get("subruns")
        if not isinstance(subruns, list) or not subruns:
            raise ValueError("Missing epistemic subruns")
        for subrun in subruns:
            directory = subrun.get("directory") if isinstance(subrun, dict) else None
            if not isinstance(directory, str) or not directory:
                raise ValueError("Invalid epistemic subrun directory")
            required.update(f"{directory}/{name}" for name in
                            ("selected_model.zip", "final_model.zip", "validation.json"))
    else:
        if selection.get("selected_model") != "selected_model.zip":
            raise ValueError("Selected checkpoint reference changed")
        for key in ("selected_model", "source_model", "final_model", "validation_manifest"):
            name = selection.get(key)
            if not isinstance(name, str) or not name:
                raise ValueError("Missing selected checkpoint reference")
            required.add(name)
    if not required.issubset(manifest):
        raise ValueError("Required artifact hash missing from provenance")
    for name, expected in manifest.items():
        if not isinstance(name, str) or not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError("Invalid artifact hash entry")
        actual = artifact_hashes(output, [name])[name]
        if actual != expected:
            raise ValueError("Recorded artifact hash changed: " + name)
    current = source_identity(root)
    if current["fingerprint_sha256"] != source["fingerprint_sha256"]:
        raise ValueError("Executable source fingerprint changed since training")
    return {"status": "hashes_verified", "schema": SCHEMA,
            "training": saved, "evaluation_source": current,
            "evaluation_dependencies": dependency_versions()}
