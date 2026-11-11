"""Strict source-only transfer evidence checker and metric exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Libs.exp.aggregate_results import aggregate_prediction_rows
from Libs.exp.canonical_io import load_validated_predictions, write_derived_artifact


def _calibration_payload(run_root: Path, record_or_payload: Any) -> dict[str, Any] | None:
    if record_or_payload is None:
        return None
    if not isinstance(record_or_payload, dict):
        raise ValueError("calibration record must be an object")
    if "path" not in record_or_payload:
        return dict(record_or_payload)
    path = Path(str(record_or_payload["path"]))
    if not path.is_absolute():
        path = run_root / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration artifact must be an object: {path}")
    return payload


def evaluate(source_run: str | Path, target_run: str | Path, *, num_classes: int, output: str | Path | None = None) -> dict[str, Any]:
    source_root, target_root = Path(source_run), Path(target_run)
    source_manifest, source_rows, _ = load_validated_predictions(source_root)
    target_manifest, target_rows, target_prediction_path = load_validated_predictions(target_root)
    if source_manifest["dataset"] == target_manifest["dataset"]:
        raise ValueError("transfer source and target datasets must differ")
    calibration = json.loads((source_root / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("fit_split") != "val" or calibration.get("threshold_fit_split", "val") != "val":
        raise ValueError("transfer calibration/threshold must be source validation only")
    target_calibration = _calibration_payload(target_root, target_manifest.get("calibration"))
    if target_calibration and str(target_calibration.get("fit_split", "val")) != "source_val":
        raise ValueError("target manifest declares target-label calibration")
    target_metrics = aggregate_prediction_rows(target_rows, num_classes=num_classes)
    source_metrics = aggregate_prediction_rows(source_rows, num_classes=num_classes)
    source_f1 = float(source_metrics["metrics"]["macro_f1"])
    target_f1 = float(target_metrics["metrics"]["macro_f1"])
    payload = {
        "schema_version": "1", "transfer_mode": "source_only", "source_dataset": source_manifest["dataset"], "target_dataset": target_manifest["dataset"],
        "source_run_id": source_manifest["run_id"], "target_run_id": target_manifest["run_id"], "source_macro_f1": source_f1,
        "target_macro_f1": target_f1, "seedwise_retention": target_f1 / source_f1 if source_f1 else None,
        "target_metrics": target_metrics["metrics"], "target_ids_sha256": target_metrics["source_ids_sha256"],
        "calibration_source": {"run_id": source_manifest["run_id"], "fit_split": "val"}, "target_test_report_only": True,
    }
    target = Path(output) if output else target_root / "transfer.json"
    write_derived_artifact(target, payload, target_prediction_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("target_run", type=Path)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.source_run, args.target_run, num_classes=args.num_classes, output=args.output)


if __name__ == "__main__":
    main()
