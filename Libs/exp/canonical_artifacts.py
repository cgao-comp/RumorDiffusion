"""Canonical run artifact writer used by training and evaluation runners.

The writer does not invent checkpoints, predictions, or metrics. It accepts
already-produced model outputs, derives metrics from them, and leaves a run in
``Output/exploratory`` unless all caller-supplied provenance is complete.
"""

from __future__ import annotations

import shutil
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from Libs.exp.aggregate_results import aggregate_prediction_rows
from Libs.utils.artifact_schema import (
    canonical_json_hash,
    canonical_key,
    file_sha256,
    omit_optional_none_fields,
    write_predictions_parquet,
)
from Libs.utils.provenance import artifact_record, deterministic_run_id, environment_snapshot, git_state, write_json
from Libs.exp.trainer import unpack_checkpoint_payload


def prediction_rows_from_logits(
    *, run_id: str, dataset: str, split: str, source_ids: Sequence[str], logits: torch.Tensor,
    targets: torch.Tensor, horizon_unit: str, horizon_value: float, model: str, model_variant: str,
    seeds: Mapping[str, int], calibration_tag: str, sampling_S: int, checkpoint_sha256: str,
    config_sha256: str, split_sha256: str, temperature: float = 1.0,
    compute_latency_ms: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("canonical report prediction export requires test split")
    if len(source_ids) != int(logits.size(0)) or logits.size(0) != targets.numel():
        raise ValueError("source IDs, logits and targets must align")
    if logits.ndim != 2 or logits.size(1) < 2 or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite [N,C]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    uncalibrated = torch.softmax(logits.detach().cpu(), dim=1)
    calibrated = torch.softmax(logits.detach().cpu() / float(temperature), dim=1)
    y_pred = calibrated.argmax(dim=1)
    latency = list(compute_latency_ms or [])
    if latency and len(latency) != len(source_ids):
        raise ValueError("latency rows must align with source IDs")
    rows = []
    for index, source_id in enumerate(source_ids):
        probability = calibrated[index]
        row = {
            "run_id": run_id, "dataset": dataset, "split": split, "source_id": str(source_id),
            "horizon_unit": horizon_unit, "horizon_value": float(horizon_value), "model": model,
            "model_variant": model_variant, "split_seed": int(seeds["split"]), "ctdm_seed": int(seeds["ctdm"]),
            "arbiter_seed": int(seeds["arbiter"]), "inference_seed": int(seeds["inference"]),
            "calibration_tag": calibration_tag, "sampling_S": int(sampling_S), "y_true": int(targets[index].item()),
            "logits": [float(value) for value in logits[index].detach().cpu().tolist()],
            "probs_uncalibrated": [float(value) for value in uncalibrated[index].tolist()],
            "probs_calibrated": [float(value) for value in probability.tolist()], "y_pred": int(y_pred[index].item()),
            "confidence": float(probability[y_pred[index]].item()), "uncertainty_score": float(1.0 - probability[y_pred[index]].item()),
            "accepted": True, "decision": "accept",
            "checkpoint_sha256": checkpoint_sha256, "config_sha256": config_sha256, "split_sha256": split_sha256,
        }
        if latency:
            row["compute_latency_ms"] = float(latency[index])
        rows.append(row)
    return rows


def canonical_run_identity(
    *, run_inputs: Mapping[str, Any], exp_config: Mapping[str, Any], model_config: Mapping[str, Any],
    raw_manifest_path: str | Path, processed_manifest_path: str | Path, split_manifest_path: str | Path,
) -> dict[str, Any]:
    """Return the stable, non-wall-clock identity payload for a canonical run."""
    required = {"dataset", "split", "horizon", "model", "seeds", "calibration_tag", "sampling_S"}
    missing = required.difference(run_inputs)
    if missing:
        raise ValueError(f"canonical run inputs missing: {', '.join(sorted(missing))}")
    resolved_exp_config = omit_optional_none_fields(exp_config)
    resolved_model_config = omit_optional_none_fields(model_config)
    return {
        "dataset": run_inputs["dataset"],
        "split": run_inputs["split"],
        "horizon": run_inputs["horizon"],
        "model": run_inputs["model"],
        "seeds": dict(run_inputs["seeds"]),
        "calibration_tag": run_inputs["calibration_tag"],
        "sampling_S": int(run_inputs["sampling_S"]),
        "resolved_config_sha256": canonical_json_hash(
            {"experiment": resolved_exp_config, "model": resolved_model_config}
        ),
        "raw_manifest_sha256": file_sha256(raw_manifest_path),
        "processed_manifest_sha256": file_sha256(processed_manifest_path),
        "split_manifest_sha256": file_sha256(split_manifest_path),
    }


def canonical_run_id(
    *, run_inputs: Mapping[str, Any], exp_config: Mapping[str, Any], model_config: Mapping[str, Any],
    raw_manifest_path: str | Path, processed_manifest_path: str | Path, split_manifest_path: str | Path,
) -> str:
    return deterministic_run_id(
        canonical_run_identity(
            run_inputs=run_inputs,
            exp_config=exp_config,
            model_config=model_config,
            raw_manifest_path=raw_manifest_path,
            processed_manifest_path=processed_manifest_path,
            split_manifest_path=split_manifest_path,
        )
    )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=True, allow_unicode=True), encoding="utf-8")


def _relative_record(base: Path, path: Path, schema_version: str, **extra: Any) -> dict[str, Any]:
    record = {"path": str(path.relative_to(base)), "sha256": file_sha256(path), "schema_version": schema_version}
    record.update(extra)
    return record


def _with_source_predictions(payload: Mapping[str, Any], prediction_record: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["source_predictions"] = {
        "path": "predictions.parquet",
        "sha256": str(prediction_record["sha256"]),
        "schema_version": str(prediction_record.get("schema_version", "1")),
    }
    return record


def _manifest_canonical_key(manifest: Mapping[str, Any]) -> tuple[Any, ...]:
    return canonical_key(
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


def _reject_duplicate_canonical_key(canonical_root: Path, staged_manifest: Mapping[str, Any]) -> None:
    runs_root = canonical_root / "runs"
    if not runs_root.is_dir():
        return
    staged_key = _manifest_canonical_key(staged_manifest)
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_canonical_key(manifest) == staged_key:
            raise FileExistsError(f"duplicate canonical key conflicts with existing run: {manifest_path.parent}")


def write_canonical_run(
    output_root: str | Path, *, protocol_id: str, run_inputs: Mapping[str, Any], checkpoint_path: str | Path,
    exp_config: Mapping[str, Any], model_config: Mapping[str, Any], split_manifest_path: str | Path,
    raw_manifest_path: str | Path, processed_manifest_path: str | Path, prediction_rows: Sequence[Mapping[str, Any]],
    num_classes: int, calibration: Mapping[str, Any], selective_metrics: Mapping[str, Any], latency: Mapping[str, Any],
    repo_root: str | Path = ".",
) -> Path:
    required = {"dataset", "split", "horizon", "model", "seeds", "calibration_tag", "sampling_S"}
    missing = required.difference(run_inputs)
    if missing:
        raise ValueError(f"canonical run inputs missing: {', '.join(sorted(missing))}")
    execution = run_inputs.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("canonical run inputs require an execution record")
    for field in ("start_time", "end_time", "exit_code", "command", "log_path"):
        if field not in execution:
            raise ValueError(f"execution record missing {field}")
    if int(execution["exit_code"]) != 0:
        raise ValueError("only successful process exits can enter Output/canonical")
    log_path = Path(str(execution["log_path"]))
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    if not str(execution["command"]).strip():
        raise ValueError("execution command cannot be empty")
    for required_path in (checkpoint_path, split_manifest_path, raw_manifest_path, processed_manifest_path):
        if not Path(required_path).is_file():
            raise FileNotFoundError(required_path)
    resolved_exp_config = omit_optional_none_fields(exp_config)
    resolved_model_config = omit_optional_none_fields(model_config)
    run_id = canonical_run_id(
        run_inputs=run_inputs,
        exp_config=resolved_exp_config,
        model_config=resolved_model_config,
        raw_manifest_path=raw_manifest_path,
        processed_manifest_path=processed_manifest_path,
        split_manifest_path=split_manifest_path,
    )
    if any(str(row.get("run_id")) != run_id for row in prediction_rows):
        raise ValueError("prediction rows must carry the deterministic canonical run_id")
    code_state = git_state(repo_root)
    if bool(code_state["dirty"]):
        raise ValueError("cannot publish a canonical run from a dirty code checkout")
    output_root = Path(output_root)
    canonical_root = output_root / "canonical" / protocol_id
    run_dir = canonical_root / "runs" / run_id
    staging_dir = output_root / "exploratory" / protocol_id / "staging" / run_id
    failed_dir = output_root / "failed" / protocol_id / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"canonical run directory already exists: {run_dir}")
    if staging_dir.exists():
        raise FileExistsError(f"canonical staging directory already exists: {staging_dir}")

    try:
        staging_dir.mkdir(parents=True)
        _write_yaml(staging_dir / "resolved_exp.yaml", resolved_exp_config)
        _write_yaml(staging_dir / "resolved_model.yaml", resolved_model_config)
        env_payload = environment_snapshot(repo_root)
        if isinstance(run_inputs.get("environment"), Mapping):
            env_payload.update(dict(run_inputs["environment"]))
        write_json(staging_dir / "environment.json", env_payload)
        (staging_dir / "command.txt").write_text(str(execution["command"]).strip() + "\n", encoding="utf-8")

        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        _, checkpoint_metadata = unpack_checkpoint_payload(checkpoint_payload)
        if not isinstance(checkpoint_metadata, Mapping):
            raise ValueError("canonical checkpoint must contain metadata")

        checkpoint_dest = staging_dir / "checkpoints" / Path(checkpoint_path).name
        checkpoint_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, checkpoint_dest)
        log_dest = staging_dir / "logs" / log_path.name
        log_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, log_dest)

        prediction_record = write_predictions_parquet(prediction_rows, staging_dir / "predictions.parquet")
        metrics = aggregate_prediction_rows(prediction_rows, num_classes=num_classes)
        metrics["source_predictions"] = {"path": "predictions.parquet", "sha256": prediction_record["sha256"], "schema_version": "1"}
        write_json(staging_dir / "metrics.json", metrics)
        write_json(staging_dir / "calibration.json", dict(calibration))
        write_json(staging_dir / "selective_metrics.json", _with_source_predictions(selective_metrics, prediction_record))
        write_json(staging_dir / "latency.json", _with_source_predictions(latency, prediction_record))
        identity = canonical_run_identity(
            run_inputs=run_inputs,
            exp_config=exp_config,
            model_config=model_config,
            raw_manifest_path=raw_manifest_path,
            processed_manifest_path=processed_manifest_path,
            split_manifest_path=split_manifest_path,
        )
        manifest = {
            "schema_version": "2", "run_id": run_id, "status": "validated", "dataset": run_inputs["dataset"], "split": run_inputs["split"],
            "horizon": run_inputs["horizon"], "model": run_inputs["model"],
            "canonical_identity": {"payload": identity, "sha256": canonical_json_hash(identity)},
            "resolved_config": {"path": "resolved_exp.yaml", "sha256": file_sha256(staging_dir / "resolved_exp.yaml"), "payload": resolved_exp_config},
            "resolved_model_config": {"path": "resolved_model.yaml", "sha256": file_sha256(staging_dir / "resolved_model.yaml"), "payload": resolved_model_config},
            "code": code_state, "environment": _relative_record(staging_dir, staging_dir / "environment.json", "1"),
            "raw_manifest": artifact_record(Path(raw_manifest_path).resolve(), "1"),
            "processed_manifest": artifact_record(Path(processed_manifest_path).resolve(), "1"),
            "split_manifest": artifact_record(Path(split_manifest_path).resolve(), "1"),
            "checkpoint": _relative_record(staging_dir, checkpoint_dest, "1", metadata=dict(checkpoint_metadata)),
            "predictions": {**prediction_record, "path": "predictions.parquet"},
            "metrics": _relative_record(staging_dir, staging_dir / "metrics.json", "1", command="Libs.exp.aggregate_results"),
            "calibration": _relative_record(staging_dir, staging_dir / "calibration.json", "1"),
            "selective_metrics": _relative_record(staging_dir, staging_dir / "selective_metrics.json", "1"),
            "latency": _relative_record(staging_dir, staging_dir / "latency.json", "1"),
            "execution": {"start_time": execution["start_time"], "end_time": execution["end_time"], "exit_code": int(execution["exit_code"]), "command": _relative_record(staging_dir, staging_dir / "command.txt", "1"), "logs": [_relative_record(staging_dir, log_dest, "1")]},
            "seeds": dict(run_inputs["seeds"]), "calibration_tag": run_inputs["calibration_tag"], "sampling_S": run_inputs["sampling_S"],
        }
        write_json(staging_dir / "manifest.json", manifest)

        from Libs.scripts.validate_canonical_results import validate_run_directory

        validate_run_directory(staging_dir)
        _reject_duplicate_canonical_key(canonical_root, manifest)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(run_dir)
        return run_dir
    except Exception:
        if staging_dir.exists():
            failed_dir.parent.mkdir(parents=True, exist_ok=True)
            if failed_dir.exists():
                alt_failed_dir = failed_dir.with_name(f"{failed_dir.name}-{run_id}")
                staging_dir.rename(alt_failed_dir)
                raise
            staging_dir.rename(failed_dir)
        raise
