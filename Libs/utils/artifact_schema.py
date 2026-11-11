"""Small, dependency-light validators for canonical evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PLACEHOLDER_VALUES = {"", "tbd", "pending", "placeholder", "nan", "none", "null"}
OPTIONAL_EMPTY_CONFIG_KEYS = frozenset({"cuda_devices", "gres", "nodelist", "source_ids"})


def omit_optional_none_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy a config payload while omitting optional unset mapping fields.

    YAML represents an absent optional value as ``null`` when a key is written
    without a value. Some inherited Slurm/explanation settings use an empty
    string for the same purpose. Canonical artifacts reject placeholders, so
    callers use this only when serializing resolved configuration/provenance
    records. Empty values are omitted only for the explicit optional-key
    allowlist; all other placeholders remain and fail validation. A null list
    element is rejected rather than removed because removing it could change an
    ordered configuration's meaning.
    """

    def is_placeholder(value: Any) -> bool:
        return isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES

    def clean(value: Any, context: str, *, omit_placeholders: bool = False) -> Any:
        if isinstance(value, Mapping):
            cleaned: dict[str, Any] = {}
            for key, child in value.items():
                key_text = str(key)
                optional_key = key_text in OPTIONAL_EMPTY_CONFIG_KEYS
                if child is None or (optional_key and is_placeholder(child)):
                    continue
                cleaned_child = clean(
                    child,
                    f"{context}.{key_text}",
                    omit_placeholders=optional_key,
                )
                if optional_key and isinstance(cleaned_child, list) and not cleaned_child:
                    continue
                cleaned[key_text] = cleaned_child
            return cleaned
        if isinstance(value, (list, tuple)):
            cleaned = []
            for index, child in enumerate(value):
                if child is None:
                    raise ValueError(
                        f"{context}[{index}] is null; ordered canonical config values cannot be omitted"
                    )
                if omit_placeholders and is_placeholder(child):
                    continue
                cleaned.append(
                    clean(child, f"{context}[{index}]", omit_placeholders=omit_placeholders)
                )
            return cleaned
        return value

    return clean(payload, "config")


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    fields = (
        "dataset", "split", "horizon_unit", "horizon_value", "model", "model_variant",
        "split_seed", "ctdm_seed", "arbiter_seed", "inference_seed", "calibration_tag", "sampling_S",
    )
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"Canonical key missing fields: {', '.join(missing)}")
    return tuple(row[field] for field in fields)


def ensure_unique_canonical_keys(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: dict[tuple[Any, ...], int] = {}
    for index, row in enumerate(rows):
        key = canonical_key(row)
        if key in seen:
            raise ValueError(f"Duplicate canonical key at rows {seen[key]} and {index}: {key!r}")
        seen[key] = index


def reject_placeholders(value: Any, context: str = "artifact") -> None:
    if value is None:
        raise ValueError(f"{context} contains a null/placeholder value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite value")
    if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_VALUES:
        raise ValueError(f"{context} contains a placeholder value")
    if isinstance(value, Mapping):
        for key, child in value.items():
            reject_placeholders(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_placeholders(child, f"{context}[{index}]")


def validate_required_fields(payload: Mapping[str, Any], required: Sequence[str], context: str) -> None:
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    validate_required_fields(
        manifest,
        ("schema_version", "run_id", "status", "dataset", "split", "horizon", "model", "resolved_config", "code", "checkpoint", "predictions", "metrics", "seeds", "calibration_tag", "sampling_S"),
        "run manifest",
    )
    if manifest.get("status") != "validated":
        raise ValueError("canonical run manifest status must be validated")
    horizon = manifest["horizon"]
    if not isinstance(horizon, Mapping) or horizon.get("unit") not in {"minutes", "hop", "count"}:
        raise ValueError("run manifest horizon must declare unit minutes, hop, or count")
    if float(horizon.get("value", 0)) <= 0:
        raise ValueError("run manifest horizon value must be positive")
    if int(manifest.get("sampling_S", 0)) < 1:
        raise ValueError("run manifest sampling_S must be positive")
    if str(manifest.get("schema_version")) == "2":
        validate_required_fields(
            manifest,
            ("resolved_model_config", "environment", "raw_manifest", "processed_manifest", "split_manifest", "calibration", "selective_metrics", "latency", "execution"),
            "canonical v2 run manifest",
        )
    reject_placeholders(manifest, "run manifest")


def validate_metrics_artifact(metrics: Mapping[str, Any]) -> None:
    validate_required_fields(metrics, ("schema_version", "run_id", "metric_definition", "metrics", "source_predictions"), "metrics artifact")
    definition = metrics["metric_definition"]
    if not isinstance(definition, Mapping):
        raise ValueError("metric_definition must be an object")
    validate_required_fields(definition, ("class_space", "ece_bins", "ci_method"), "metric definition")
    class_space = definition["class_space"]
    if not isinstance(class_space, Sequence) or len(class_space) < 2:
        raise ValueError("metric class_space must declare at least two classes")
    if int(definition["ece_bins"]) < 1:
        raise ValueError("metric ece_bins must be positive")
    source = metrics["source_predictions"]
    if not isinstance(source, Mapping):
        raise ValueError("metrics source_predictions must be an object")
    validate_required_fields(source, ("path", "sha256", "schema_version"), "metrics source_predictions")
    reject_placeholders(metrics, "metrics artifact")


def validate_prediction_row(row: Mapping[str, Any]) -> None:
    validate_required_fields(
        row,
        ("run_id", "dataset", "split", "source_id", "horizon_unit", "horizon_value", "model", "model_variant", "y_true", "logits", "probs_uncalibrated", "probs_calibrated", "y_pred", "confidence", "uncertainty_score", "accepted", "decision", "split_seed", "ctdm_seed", "arbiter_seed", "inference_seed", "calibration_tag", "sampling_S", "checkpoint_sha256", "config_sha256", "split_sha256"),
        "prediction row",
    )
    if row["horizon_unit"] not in {"minutes", "hop", "count"}:
        raise ValueError("prediction horizon_unit is invalid")
    if float(row["horizon_value"]) <= 0:
        raise ValueError("prediction horizon_value must be positive")
    logits = row["logits"]
    probs = row["probs_uncalibrated"]
    calibrated = row["probs_calibrated"]
    if not isinstance(logits, Sequence) or len(logits) < 2 or len(probs) != len(logits) or len(calibrated) != len(logits):
        raise ValueError("prediction logits/probability vectors must have equal length >= 2")
    reject_placeholders(row, "prediction row")


def validate_prediction_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("prediction artifact is empty")
    for row in materialized:
        validate_prediction_row(row)


def write_predictions_parquet(rows: Iterable[Mapping[str, Any]], path: str | Path, *, schema_version: str = "1") -> dict[str, Any]:
    """Write validated prediction rows as a deterministic Parquet artifact."""
    materialized = [dict(row) for row in rows]
    validate_prediction_rows(materialized)
    if not materialized:
        raise ValueError("prediction artifact is empty")
    try:
        import pandas as pd

        frame = pd.DataFrame(materialized)
        # Nested vectors are intentionally stored as JSON strings for broad
        # Arrow compatibility; the schema records their logical field names.
        for field in ("logits", "probs_uncalibrated", "probs_calibrated"):
            frame[field] = frame[field].map(lambda value: json.dumps(list(value), separators=(",", ":")))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    except ImportError as exc:  # pragma: no cover - dependency is checked by CI
        raise RuntimeError("Parquet output requires pandas and pyarrow") from exc
    return {"path": str(path), "sha256": file_sha256(path), "schema_version": schema_version, "rows": len(materialized)}


def read_predictions_parquet(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parquet input requires pandas and pyarrow") from exc
    frame = pd.read_parquet(path)
    rows = frame.to_dict(orient="records")
    for row in rows:
        for field in ("logits", "probs_uncalibrated", "probs_calibrated"):
            value = row.get(field)
            if isinstance(value, str):
                row[field] = json.loads(value)
    validate_prediction_rows(rows)
    return rows


def write_trajectories_parquet(rows: Iterable[Mapping[str, Any]], path: str | Path) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("trajectory artifact is empty")
    required = {"trajectory_id", "trajectory_seed", "source_id", "logits", "probabilities"}
    missing = required.difference(materialized[0])
    if missing:
        raise ValueError(f"trajectory rows missing required fields: {', '.join(sorted(missing))}")
    try:
        import pandas as pd

        frame = pd.DataFrame(materialized)
        for field in ("logits", "probabilities"):
            frame[field] = frame[field].map(lambda value: json.dumps(value, separators=(",", ":")))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parquet output requires pandas and pyarrow") from exc
    return {"path": str(path), "sha256": file_sha256(path), "schema_version": "1", "rows": len(materialized)}


def read_artifact_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.suffix == ".parquet":
        return read_predictions_parquet(target)
    if target.suffix == ".jsonl":
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_prediction_rows(rows)
        return rows
    raise ValueError(f"Unsupported prediction artifact format: {target.suffix}")


def assert_artifact_fresh(source_path: str | Path, source_sha256: str, *, context: str = "artifact") -> None:
    observed = file_sha256(source_path)
    if observed != source_sha256:
        raise ValueError(f"{context} is stale: expected {source_sha256}, observed {observed}")
