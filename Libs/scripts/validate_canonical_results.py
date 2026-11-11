"""Fail-closed validation for canonical run and prediction artifacts.

This command validates manifests, Parquet/JSONL predictions, and metrics whose
source prediction hash is still current. It never selects a latest result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from Libs.utils.artifact_schema import (
    canonical_key,
    file_sha256,
    read_artifact_rows,
    validate_metrics_artifact,
    validate_required_fields,
    validate_run_manifest,
)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Prediction row {line_number} is not an object: {path}")
        rows.append(payload)
    return rows


def _validate_declared_schema(payload: Mapping[str, Any], schema_name: str) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / schema_name
    if not schema_path.is_file():
        raise FileNotFoundError(f"Missing canonical schema: {schema_path}")
    try:
        import jsonschema

        jsonschema.validate(payload, _load_json(schema_path))
    except ImportError as exc:  # pragma: no cover - dependency is part of CI
        raise RuntimeError("jsonschema is required for canonical validation") from exc


def _finite_number(value: Any, context: str) -> float:
    import math

    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite")
    return float(value)


def _validate_prediction_source_record(payload: Mapping[str, Any], run_dir: Path, prediction_path: Path, context: str) -> None:
    source = payload.get("source_predictions")
    if source is None:
        return
    if not isinstance(source, Mapping):
        raise ValueError(f"{context} source_predictions must be an object")
    validate_required_fields(source, ("path", "sha256", "schema_version"), f"{context} source_predictions")
    declared_path = Path(str(source["path"]))
    if not declared_path.is_absolute():
        declared_path = run_dir / declared_path
    if declared_path.resolve() != prediction_path.resolve():
        raise ValueError(f"{context} source_predictions path does not match manifest predictions")
    if file_sha256(declared_path) != str(source["sha256"]):
        raise ValueError(f"{context} source_predictions hash mismatch")


def _validate_calibration_artifact(payload: Mapping[str, Any]) -> None:
    validate_required_fields(payload, ("schema_version", "fit_split", "threshold_fit_split", "method"), "calibration artifact")
    fit_split = str(payload["fit_split"])
    threshold_split = str(payload["threshold_fit_split"])
    if fit_split not in {"val", "source_val"} or threshold_split not in {"val", "source_val"}:
        raise ValueError("calibration and threshold fitting must use validation/source-validation only")
    if "test" in {fit_split, threshold_split}:
        raise ValueError("calibration artifact declares test-label tuning")
    if payload.get("method") == "temperature":
        temperature = _finite_number(payload.get("temperature"), "calibration temperature")
        if temperature <= 0.0:
            raise ValueError("calibration temperature must be positive")
    if "reject_threshold" in payload:
        threshold = _finite_number(payload["reject_threshold"], "reject threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("reject threshold must lie in [0,1]")


def _validate_selective_artifact(payload: Mapping[str, Any], run_dir: Path, prediction_path: Path) -> None:
    validate_required_fields(payload, ("schema_version", "threshold_fit_split", "coverage"), "selective metrics artifact")
    if str(payload["threshold_fit_split"]) not in {"val", "source_val"}:
        raise ValueError("selective thresholds must be validation/source-validation selected")
    for field in ("coverage", "selective_risk", "resolved_accuracy", "resolved_macro_f1", "aurc", "eaurc"):
        if field in payload:
            value = _finite_number(payload[field], f"selective metrics {field}")
            if field == "coverage" and not 0.0 <= value <= 1.0:
                raise ValueError("selective coverage must lie in [0,1]")
    _validate_prediction_source_record(payload, run_dir, prediction_path, "selective metrics artifact")


def _validate_latency_artifact(payload: Mapping[str, Any], run_dir: Path, prediction_path: Path) -> None:
    validate_required_fields(payload, ("schema_version", "measured", "compute_latency_ms", "detection_latency"), "latency artifact")
    measured = bool(payload["measured"])
    compute = payload["compute_latency_ms"]
    if not isinstance(compute, Mapping):
        raise ValueError("latency compute_latency_ms must be an object")
    n = int(compute.get("n", 0))
    if measured:
        if n <= 0:
            raise ValueError("measured latency artifact must contain at least one profiled sample")
        for field in ("mean_ms", "p50_ms", "p95_ms"):
            value = _finite_number(compute.get(field), f"latency {field}")
            if value < 0.0:
                raise ValueError(f"latency {field} must be non-negative")
    else:
        forbidden_stats = {"mean", "p50", "p95", "mean_ms", "p50_ms", "p95_ms"}.intersection(compute)
        if forbidden_stats:
            raise ValueError(
                "unmeasured latency artifact must not contain synthetic latency statistics: "
                + ", ".join(sorted(forbidden_stats))
            )
        if n != 0:
            raise ValueError("unmeasured latency artifact must declare n=0")
    detection = payload["detection_latency"]
    if not isinstance(detection, Mapping) or detection.get("unit") not in {"minutes", "hop", "count"}:
        raise ValueError("latency detection_latency must declare a valid unit")
    if _finite_number(detection.get("value"), "detection latency value") <= 0.0:
        raise ValueError("detection latency value must be positive")
    _validate_prediction_source_record(payload, run_dir, prediction_path, "latency artifact")


def validate_run_directory(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing canonical manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    _validate_declared_schema(manifest, "run_manifest.schema.json")
    validate_run_manifest(manifest)
    code = manifest.get("code")
    if not isinstance(code, Mapping) or bool(code.get("dirty", True)):
        raise ValueError("canonical validation rejects dirty or incomplete git provenance")
    artifact_paths = {}
    required_artifacts = ["checkpoint", "predictions", "metrics"]
    if str(manifest.get("schema_version")) == "2":
        required_artifacts.extend(["environment", "raw_manifest", "processed_manifest", "split_manifest", "calibration", "selective_metrics", "latency"])
    for artifact_name in required_artifacts:
        record = manifest.get(artifact_name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Manifest {artifact_name} record is missing")
        validate_required_fields(record, ("path", "sha256"), f"manifest {artifact_name}")
        artifact_path = Path(str(record["path"]))
        if not artifact_path.is_absolute():
            artifact_path = run_dir / artifact_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest {artifact_name} artifact is missing: {artifact_path}")
        observed_hash = file_sha256(artifact_path)
        if observed_hash != str(record["sha256"]):
            raise ValueError(f"Manifest {artifact_name} hash mismatch: {artifact_path}")
        artifact_paths[artifact_name] = artifact_path
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise ValueError("canonical checkpoint record is missing")
    validate_required_fields(checkpoint_record, ("metadata",), "checkpoint record")
    if not isinstance(checkpoint_record["metadata"], Mapping) or not checkpoint_record["metadata"]:
        raise ValueError("canonical checkpoint metadata is missing")
    if str(manifest.get("schema_version")) == "2":
        execution = manifest.get("execution")
        if not isinstance(execution, Mapping) or int(execution.get("exit_code", 1)) != 0:
            raise ValueError("canonical execution record is incomplete or unsuccessful")
        command = execution.get("command")
        if not isinstance(command, Mapping):
            raise ValueError("canonical execution command record is missing")
        command_path = Path(str(command["path"]))
        if not command_path.is_absolute():
            command_path = run_dir / command_path
        if not command_path.is_file() or file_sha256(command_path) != str(command["sha256"]):
            raise ValueError("canonical command artifact is missing or hash-mismatched")
        logs = execution.get("logs")
        if not isinstance(logs, list) or not logs:
            raise ValueError("canonical execution logs are missing")
        for log in logs:
            if not isinstance(log, Mapping):
                raise ValueError("canonical log record is invalid")
            log_path = Path(str(log["path"]))
            if not log_path.is_absolute():
                log_path = run_dir / log_path
            if not log_path.is_file() or file_sha256(log_path) != str(log["sha256"]):
                raise ValueError(f"canonical log artifact is missing or hash-mismatched: {log_path}")

    prediction_path = artifact_paths["predictions"]
    metrics_payload = json.loads(artifact_paths["metrics"].read_text(encoding="utf-8"))
    if not isinstance(metrics_payload, Mapping):
        raise ValueError("metrics artifact must contain a JSON object")
    _validate_declared_schema(metrics_payload, "metrics.schema.json")
    validate_metrics_artifact(metrics_payload)
    source_predictions = metrics_payload["source_predictions"]
    expected_prediction_hash = str(source_predictions["sha256"])
    if expected_prediction_hash != str(manifest["predictions"]["sha256"]):
        raise ValueError("metrics artifact is stale: prediction hash differs from manifest")
    declared_source_path = Path(str(source_predictions["path"]))
    if not declared_source_path.is_absolute():
        declared_source_path = run_dir / declared_source_path
    if declared_source_path.resolve() != prediction_path.resolve():
        raise ValueError("metrics artifact source prediction path does not match manifest")
    if str(manifest.get("schema_version")) == "2":
        _validate_calibration_artifact(_load_json(artifact_paths["calibration"]))
        _validate_selective_artifact(_load_json(artifact_paths["selective_metrics"]), run_dir, prediction_path)
        _validate_latency_artifact(_load_json(artifact_paths["latency"]), run_dir, prediction_path)
    rows = read_artifact_rows(prediction_path)
    for row in rows:
        _validate_declared_schema(row, "predictions.schema.json")
    # A prediction artifact contains one row per source. All rows must describe
    # a single run-level canonical key and each source appears exactly once.
    keys = [canonical_key(row) for row in rows]
    if len(set(keys)) != 1:
        raise ValueError("Prediction rows contain inconsistent canonical keys")
    manifest_key = canonical_key(
        {
            "dataset": manifest["dataset"],
            "split": manifest["split"],
            "horizon_unit": manifest["horizon"]["unit"],
            "horizon_value": manifest["horizon"]["value"],
            "model": manifest["model"]["name"],
            "model_variant": manifest["model"]["variant"],
            "split_seed": manifest["seeds"]["split"],
            "ctdm_seed": manifest["seeds"]["ctdm"],
            "arbiter_seed": manifest["seeds"]["arbiter"],
            "inference_seed": manifest["seeds"]["inference"],
            "calibration_tag": manifest["calibration_tag"],
            "sampling_S": manifest["sampling_S"],
        }
    )
    if keys[0] != manifest_key:
        raise ValueError("Prediction canonical key does not match manifest canonical key")
    source_ids = [str(row["source_id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Prediction artifact contains duplicate source IDs")
    if str(metrics_payload["run_id"]) != str(manifest["run_id"]):
        raise ValueError("metrics artifact run_id does not match manifest")
    if any(str(row["run_id"]) != str(manifest["run_id"]) for row in rows):
        raise ValueError("prediction row run_id does not match manifest")
    return manifest


def validate_canonical_root(root: Path) -> None:
    """Validates every run and rejects duplicate run-level canonical keys."""
    runs_root = root / "runs" if (root / "runs").is_dir() else root
    manifests = sorted(path.parent for path in runs_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No run manifests found under {runs_root}")
    rows = []
    paths = []
    for run_dir in manifests:
        manifest = validate_run_directory(run_dir)
        row = {
            "dataset": manifest["dataset"],
            "split": manifest["split"],
            "horizon_unit": manifest["horizon"]["unit"],
            "horizon_value": manifest["horizon"]["value"],
            "model": manifest["model"]["name"],
            "model_variant": manifest["model"]["variant"],
            "split_seed": manifest["seeds"]["split"],
            "ctdm_seed": manifest["seeds"]["ctdm"],
            "arbiter_seed": manifest["seeds"]["arbiter"],
            "inference_seed": manifest["seeds"]["inference"],
            "calibration_tag": manifest["calibration_tag"],
            "sampling_S": manifest["sampling_S"],
        }
        rows.append(row)
        paths.append(str(run_dir))
    seen = {}
    for index, row in enumerate(rows):
        key = canonical_key(row)
        if key in seen:
            raise ValueError(f"Duplicate canonical key at {paths[seen[key]]} and {paths[index]}: {key!r}")
        seen[key] = index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if (args.run_dir / "manifest.json").is_file():
        validate_run_directory(args.run_dir)
    else:
        validate_canonical_root(args.run_dir)
    print(f"canonical validation passed: {args.run_dir}")


if __name__ == "__main__":
    main()
