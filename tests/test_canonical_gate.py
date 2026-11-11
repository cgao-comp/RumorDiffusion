from __future__ import annotations

import json

import pytest

from Libs.scripts.validate_canonical_results import validate_canonical_root, validate_run_directory
from Libs.scripts.generate_paper_assets import validate_canonical_table_registry
from Libs.utils.artifact_schema import file_sha256


def _prediction():
    return {
        "run_id": "run-1", "dataset": "twitter15", "split": "test", "source_id": "s1",
        "horizon_unit": "minutes", "horizon_value": 10, "model": "rumordiffusion",
        "model_variant": "canonical", "y_true": 0, "logits": [1.0, 0.0],
        "probs_uncalibrated": [0.7, 0.3], "probs_calibrated": [0.7, 0.3], "y_pred": 0,
        "confidence": 0.7, "uncertainty_score": 0.0, "accepted": True, "decision": "accept",
        "split_seed": 42, "ctdm_seed": 42, "arbiter_seed": 42, "inference_seed": 42,
        "calibration_tag": "val_temperature", "sampling_S": 1,
        "checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64, "split_sha256": "c" * 64,
    }


def _write_run(run_dir):
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.pt"
    predictions = run_dir / "predictions.jsonl"
    metrics = run_dir / "metrics.json"
    checkpoint.write_bytes(b"checkpoint")
    predictions.write_text(json.dumps(_prediction()) + "\n", encoding="utf-8")
    metrics.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": "run-1",
                "metric_definition": {"class_space": [0, 1], "ece_bins": 15, "ci_method": "student_t_seed_level"},
                "metrics": {"accuracy": 1.0},
                "source_predictions": {"path": "predictions.jsonl", "sha256": file_sha256(predictions), "schema_version": "1"},
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1", "run_id": "run-1", "status": "validated", "dataset": "twitter15",
        "split": "test", "horizon": {"unit": "minutes", "value": 10},
        "model": {"name": "rumordiffusion", "variant": "canonical"},
        "resolved_config": {"path": "resolved.yaml", "sha256": "d" * 64, "payload": {"x": 1}},
        "code": {"git_commit": "e" * 40, "dirty": False, "dirty_diff_sha256": "f" * 64},
        "checkpoint": {"path": "checkpoint.pt", "sha256": file_sha256(checkpoint), "metadata": {"format_version": 1}},
        "predictions": {"path": "predictions.jsonl", "sha256": file_sha256(predictions), "schema_version": "1"},
        "metrics": {"path": "metrics.json", "sha256": file_sha256(metrics), "command": "metrics"},
        "seeds": {"split": 42, "ctdm": 42, "arbiter": 42, "inference": 42, "bootstrap": 42, "missingness": 42},
        "calibration_tag": "val_temperature", "sampling_S": 1,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_canonical_gate_checks_artifact_hashes(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    validate_run_directory(run_dir)
    (run_dir / "metrics.json").write_text(json.dumps({"schema_version": "1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_run_directory(run_dir)


def test_canonical_gate_rejects_stale_metric_source_hash(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["predictions"]["sha256"] = "d" * 64
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_run_directory(run_dir)


def test_canonical_gate_rejects_dirty_code_provenance(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["code"]["dirty"] = True
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        validate_run_directory(run_dir)


def test_canonical_gate_rejects_duplicate_run_keys(tmp_path):
    root = tmp_path / "canonical"
    _write_run(root / "runs" / "run-a")
    _write_run(root / "runs" / "run-b")
    with pytest.raises(ValueError, match="Duplicate canonical key"):
        validate_canonical_root(root)


def test_canonical_gate_rejects_prediction_manifest_key_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    prediction_path = run_dir / "predictions.jsonl"
    row = json.loads(prediction_path.read_text(encoding="utf-8"))
    row["horizon_value"] = 20
    prediction_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["source_predictions"]["sha256"] = file_sha256(prediction_path)
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["predictions"]["sha256"] = file_sha256(prediction_path)
    manifest["metrics"]["sha256"] = file_sha256(metrics_path)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match manifest"):
        validate_run_directory(run_dir)


def test_table_registry_must_point_to_validated_metric_artifact(tmp_path):
    root = tmp_path / "canonical"
    run_dir = root / "runs" / "run-a"
    _write_run(run_dir)
    metrics_path = run_dir / "metrics.json"
    registry = root / "aggregates" / "table_cell_registry.json"
    registry.parent.mkdir()
    registry.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "cells": {
                    "twitter15.rumordiffusion.f1": {
                        "artifact_path": str(metrics_path),
                        "artifact_sha256": file_sha256(metrics_path),
                        "metric_pointer": "/metrics/accuracy",
                        "value": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = validate_canonical_table_registry(root, registry)
    assert report["verified_cells"]["twitter15.rumordiffusion.f1"]["value"] == 1.0
