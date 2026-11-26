"""Bounded provenance and replay-integrity tests; no model training."""

import copy
import json

import pytest

from rl_studies import provenance


def source_root(tmp_path, monkeypatch):
    root = tmp_path / "source"
    library = root / "justnexus_v2g/src/justnexus_v2g"
    library.mkdir(parents=True)
    (library / "__init__.py").write_text("# library\n")
    study = root / "rl_studies"
    study.mkdir()
    (study / "train.py").write_text("# study\n")
    monkeypatch.setattr(provenance, "_imported_library_dir", lambda: library.resolve())
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    return root


@pytest.mark.parametrize("status,expected", [("", "clean"), ("?? new.py", "dirty")])
def test_git_source_and_ignored_file_change(tmp_path, monkeypatch, status, expected):
    root = source_root(tmp_path, monkeypatch)

    def fake_git(root, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40
        if args[0] == "rev-parse":
            return str(root)
        return status

    monkeypatch.setattr(provenance, "_git", fake_git)
    before = provenance.source_identity(root)
    assert before["working_tree"] == expected
    assert before["commit"] == "a" * 40
    assert before["library_origin"] == "checkout"
    (root / "rl_studies" / "ignored.py").write_text("value = 1\n")
    assert provenance.source_identity(root)["fingerprint_sha256"] != before["fingerprint_sha256"]


def test_no_git_hashes_actual_imported_library(tmp_path, monkeypatch):
    root = source_root(tmp_path, monkeypatch)
    before = provenance.source_identity(root)
    assert before["working_tree"] == "unavailable"
    assert before["commit"] is None
    (root / "justnexus_v2g/src/justnexus_v2g/__init__.py").write_text("# changed\n")
    assert provenance.source_identity(root)["fingerprint_sha256"] != before["fingerprint_sha256"]


def test_external_import_is_hashed(tmp_path, monkeypatch):
    root = source_root(tmp_path, monkeypatch)
    external = tmp_path / "installed" / "justnexus_v2g"
    external.mkdir(parents=True)
    (external / "__init__.py").write_text("# actual imported code\n")
    monkeypatch.setattr(provenance, "_imported_library_dir", lambda: external.resolve())
    before = provenance.source_identity(root)
    assert before["library_origin"] == "external"
    (external / "__init__.py").write_text("# changed imported code\n")
    assert provenance.source_identity(root)["fingerprint_sha256"] != before["fingerprint_sha256"]


def record_fixture(tmp_path, monkeypatch, kind):
    root = source_root(tmp_path, monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    names = ["resolved_config.json", "scenario.yaml"]
    snapshots = {"configuration": names[0], "scenario": names[1]}
    if kind == "epistemic_priority_sweep":
        candidate = "policies/p0/window.zip"
        names += [candidate, "policies/p0/selected_model.zip",
                  "policies/p0/final_model.zip", "policies/p0/validation.json"]
    else:
        names += ["selected_model.zip", "best_validation_model.zip",
                  "final_model.zip", "validation.json"]
    for name in names:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for label in ("configuration", "scenario"):
        snapshots[label + "_sha256"] = provenance.sha256_file(output / snapshots[label])
    if kind == "epistemic_priority_sweep":
        portfolio = {"candidates": [{"checkpoint": candidate,
                                     "checkpoint_sha256": provenance.sha256_file(output / candidate)}]}
        (output / "validation_portfolio.json").write_text(json.dumps(portfolio))
        names.append("validation_portfolio.json")
        selection = {"validation_portfolio": "validation_portfolio.json",
                     "validation_portfolio_sha256": provenance.sha256_file(
                         output / "validation_portfolio.json")}
    else:
        selection = {"selected_model": "selected_model.zip",
                     "source_model": "best_validation_model.zip",
                     "final_model": "final_model.zip", "validation_manifest": "validation.json"}
    start = provenance.source_identity(root)
    saved = provenance.make_provenance(root, output, names,
                                       {"oracle_bundles": {"day": "b" * 64}},
                                       start, provenance.dependency_versions(),
                                       {"validation_day": "c" * 64})
    record = {"study_kind": kind, "input_snapshots": snapshots,
              "model_selection": selection, "experimental_provenance": saved}
    if kind == "epistemic_priority_sweep":
        record["subruns"] = [{"directory": "policies/p0"}]
    return root, output, record


@pytest.mark.parametrize("kind", ["single_policy", "epistemic_priority_sweep"])
def test_new_records_verify_tamper_and_missing_file(tmp_path, monkeypatch, kind):
    root, output, record = record_fixture(tmp_path, monkeypatch, kind)
    saved = record["experimental_provenance"]
    assert saved["source_stability"] == "stable"
    assert saved["oracle_bundles"] == {
        "train": {"day": "b" * 64}, "validation": {"validation_day": "c" * 64},
    }
    assert provenance.verify_provenance(record, root, output)["status"] == "hashes_verified"
    target = output / "scenario.yaml"
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash changed|snapshot"):
        provenance.verify_provenance(record, root, output)
    target.unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        provenance.verify_provenance(record, root, output)


@pytest.mark.parametrize("kind,missing", [
    ("single_policy", "selected_model.zip"),
    ("single_policy", "scenario.yaml"),
    ("epistemic_priority_sweep", "validation_portfolio.json"),
    ("epistemic_priority_sweep", "policies/p0/window.zip"),
])
def test_required_hash_cannot_be_removed(tmp_path, monkeypatch, kind, missing):
    root, output, record = record_fixture(tmp_path, monkeypatch, kind)
    del record["experimental_provenance"]["artifacts_sha256"][missing]
    with pytest.raises(ValueError, match="hash"):
        provenance.verify_provenance(record, root, output)


def test_selected_model_substitution_rejected_even_if_hashed(tmp_path, monkeypatch):
    root, output, record = record_fixture(tmp_path, monkeypatch, "single_policy")
    (output / "other.zip").write_bytes(b"other")
    record["experimental_provenance"]["artifacts_sha256"]["other.zip"] = provenance.sha256_file(
        output / "other.zip")
    record["model_selection"]["selected_model"] = "other.zip"
    with pytest.raises(ValueError, match="Selected checkpoint reference changed"):
        provenance.verify_provenance(record, root, output)


def test_malformed_schema_never_falls_back_to_legacy(tmp_path, monkeypatch):
    root, output, record = record_fixture(tmp_path, monkeypatch, "single_policy")
    for malformed in (None, {}, {"schema": "unknown"}):
        variant = copy.deepcopy(record)
        variant["experimental_provenance"] = malformed
        with pytest.raises(ValueError):
            provenance.verify_provenance(variant, root, output)
    record["experimental_provenance"]["artifacts_sha256"] = {}
    with pytest.raises(ValueError, match="Incomplete"):
        provenance.verify_provenance(record, root, output)


def test_source_change_during_run_marks_uncertain(tmp_path, monkeypatch):
    root, output, record = record_fixture(tmp_path, monkeypatch, "single_policy")
    start = provenance.source_identity(root)
    (root / "rl_studies" / "train.py").write_text("# changed during run\n")
    changed = provenance.make_provenance(
        root, output, list(record["experimental_provenance"]["artifacts_sha256"]),
        {}, start, provenance.dependency_versions())
    assert changed["source"] == start
    assert changed["source_stability"] == "changed_during_run"
    record["experimental_provenance"] = changed
    with pytest.raises(ValueError, match="Incomplete"):
        provenance.verify_provenance(record, root, output)


def test_executable_mismatch_blocks_replay(tmp_path, monkeypatch):
    root, output, record = record_fixture(tmp_path, monkeypatch, "single_policy")
    (root / "rl_studies" / "train.py").write_text("# changed before test\n")
    with pytest.raises(ValueError, match="fingerprint changed"):
        provenance.verify_provenance(record, root, output)


def test_legacy_and_path_escape(tmp_path):
    assert provenance.verify_provenance({}, tmp_path, tmp_path)["status"] == "unknown_legacy"
    with pytest.raises(ValueError, match="Missing or unsafe"):
        provenance.artifact_hashes(tmp_path, ["../secret"])
