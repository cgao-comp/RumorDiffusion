"""Shared I/O contract for focused canonical experiment runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from Libs.scripts.validate_canonical_results import validate_run_directory
from Libs.utils.artifact_schema import assert_artifact_fresh, file_sha256, read_artifact_rows
from Libs.utils.provenance import write_json


def load_validated_predictions(run_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = dict(validate_run_directory(root))
    if manifest.get("status") != "validated":
        raise ValueError("focused canonical runners only accept validated runs")
    record = manifest.get("predictions")
    if not isinstance(record, Mapping):
        raise ValueError("manifest does not declare predictions")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = root / path
    assert_artifact_fresh(path, str(record["sha256"]), context="prediction artifact")
    return manifest, read_artifact_rows(path), path


def write_derived_artifact(path: str | Path, payload: Mapping[str, Any], prediction_path: str | Path) -> dict[str, Any]:
    record = dict(payload)
    record["source_predictions"] = {"path": str(prediction_path), "sha256": file_sha256(prediction_path), "schema_version": "1"}
    write_json(path, record)
    return {"path": str(path), "sha256": file_sha256(path), "schema_version": "1"}
