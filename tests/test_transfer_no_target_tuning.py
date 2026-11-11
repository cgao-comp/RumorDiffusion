from __future__ import annotations

import json

import pytest

from Libs.exp.aggregate_results import aggregate_prediction_rows
from Libs.exp.evaluate_transfer import evaluate
from Libs.utils.artifact_schema import file_sha256, write_predictions_parquet


def _prediction_row(*, run_id: str, dataset: str, source_id: str):
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": "test",
        "source_id": source_id,
        "horizon_unit": "minutes",
        "horizon_value": 10,
        "model": "rumordiffusion",
        "model_variant": "canonical",
        "split_seed": 42,
        "ctdm_seed": 42,
        "arbiter_seed": 42,
        "inference_seed": 42,
        "calibration_tag": "val_temperature",
        "sampling_S": 1,
        "y_true": 0,
        "logits": [2.0, 0.0],
        "probs_uncalibrated": [0.88, 0.12],
        "probs_calibrated": [0.88, 0.12],
        "y_pred": 0,
        "confidence": 0.88,
        "uncertainty_score": 0.12,
        "accepted": True,
        "decision": "accept",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "split_sha256": "c" * 64,
    }


def _write_minimal_run(run_dir, *, run_id: str, dataset: str, target_calibration=None):
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "checkpoint.pt"
    predictions = run_dir / "predictions.parquet"
    metrics = run_dir / "metrics.json"
    checkpoint.write_bytes(b"checkpoint")
    rows = [_prediction_row(run_id=run_id, dataset=dataset, source_id=f"{dataset}-1")]
    write_predictions_parquet(rows, predictions)
    metric_payload = aggregate_prediction_rows(rows, num_classes=2)
    metric_payload["source_predictions"] = {
        "path": "predictions.parquet",
        "sha256": file_sha256(predictions),
        "schema_version": "1",
    }
    metrics.write_text(json.dumps(metric_payload), encoding="utf-8")
    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "status": "validated",
        "dataset": dataset,
        "split": "test",
        "horizon": {"unit": "minutes", "value": 10},
        "model": {"name": "rumordiffusion", "variant": "canonical"},
        "resolved_config": {"path": "resolved.yaml", "sha256": "d" * 64, "payload": {"dataset": dataset}},
        "code": {"git_commit": "e" * 40, "dirty": False, "dirty_diff_sha256": "f" * 64},
        "checkpoint": {"path": "checkpoint.pt", "sha256": file_sha256(checkpoint), "metadata": {"format_version": 1}},
        "predictions": {"path": "predictions.parquet", "sha256": file_sha256(predictions), "schema_version": "1"},
        "metrics": {"path": "metrics.json", "sha256": file_sha256(metrics), "command": "metrics"},
        "seeds": {"split": 42, "ctdm": 42, "arbiter": 42, "inference": 42, "bootstrap": 42, "missingness": 42},
        "calibration_tag": "val_temperature",
        "sampling_S": 1,
    }
    if target_calibration is not None:
        calibration = run_dir / "calibration.json"
        calibration.write_text(json.dumps(target_calibration), encoding="utf-8")
        manifest["calibration"] = {"path": "calibration.json", "sha256": file_sha256(calibration), "schema_version": "1"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_transfer_runner_rejects_target_label_calibration(tmp_path):
    source_run = tmp_path / "source"
    target_run = tmp_path / "target"
    _write_minimal_run(source_run, run_id="source-run", dataset="twitter15")
    _write_minimal_run(
        target_run,
        run_id="target-run",
        dataset="twitter16",
        target_calibration={"fit_split": "test"},
    )
    (source_run / "calibration.json").write_text(
        json.dumps({"schema_version": "1", "fit_split": "val", "threshold_fit_split": "val"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target-label calibration"):
        evaluate(source_run, target_run, num_classes=2)


def test_transfer_runner_writes_source_only_artifact(tmp_path):
    source_run = tmp_path / "source"
    target_run = tmp_path / "target"
    _write_minimal_run(source_run, run_id="source-run", dataset="twitter15")
    _write_minimal_run(
        target_run,
        run_id="target-run",
        dataset="twitter16",
        target_calibration={"fit_split": "source_val"},
    )
    (source_run / "calibration.json").write_text(
        json.dumps({"schema_version": "1", "fit_split": "val", "threshold_fit_split": "val"}),
        encoding="utf-8",
    )

    payload = evaluate(source_run, target_run, num_classes=2)
    assert payload["transfer_mode"] == "source_only"
    assert payload["target_test_report_only"] is True
    assert json.loads((target_run / "transfer.json").read_text(encoding="utf-8"))["source_predictions"]["sha256"]
