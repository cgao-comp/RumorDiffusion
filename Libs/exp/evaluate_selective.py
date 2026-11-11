"""Compute validation-thresholded selective metrics from canonical predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from Libs.exp.canonical_io import load_validated_predictions, write_derived_artifact
from Libs.utils.selective_metrics import selective_metrics


def _threshold_from_calibration(run_dir: Path) -> float:
    artifact = run_dir / "calibration.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("fit_split") != "val" or payload.get("threshold_fit_split", "val") != "val":
        raise ValueError("selective threshold must be selected on validation only")
    if "reject_threshold" not in payload:
        raise ValueError("calibration artifact lacks validation-selected reject_threshold")
    return float(payload["reject_threshold"])


def _artifact_curve(curve: Any) -> list[dict[str, float | str]]:
    coverage_values = curve["coverage"] if isinstance(curve, dict) else [row["coverage"] for row in curve]
    risk_values = curve["risk"] if isinstance(curve, dict) else [row["risk"] for row in curve]
    threshold_values = curve["threshold"] if isinstance(curve, dict) else [row.get("threshold") for row in curve]
    rows: list[dict[str, float | str]] = []
    for coverage, risk, threshold in zip(coverage_values, risk_values, threshold_values):
        row: dict[str, float | str] = {"coverage": float(coverage), "risk": float(risk)}
        threshold_value = float(threshold)
        if math.isfinite(threshold_value):
            row["threshold"] = threshold_value
        else:
            row["threshold_policy"] = "above_max_confidence"
        rows.append(row)
    return rows


def evaluate(run_dir: str | Path, *, num_classes: int, threshold: float | None = None, output: str | Path | None = None) -> dict:
    root = Path(run_dir)
    manifest, rows, prediction_path = load_validated_predictions(root)
    chosen_threshold = _threshold_from_calibration(root) if threshold is None else float(threshold)
    if threshold is not None:
        raise ValueError("canonical selective evaluation requires a threshold stored in validation calibration.json")
    payload = selective_metrics(
        [row["y_true"] for row in rows],
        [row["probs_calibrated"] for row in rows],
        threshold=chosen_threshold,
        num_classes=num_classes,
        y_pred=[row["y_pred"] for row in rows],
        detection_delay=[float(row["horizon_value"]) for row in rows],
    )
    payload["risk_coverage_curve"] = _artifact_curve(payload["risk_coverage_curve"])
    payload.update({"schema_version": "1", "run_id": manifest["run_id"], "threshold_fit_split": "val", "metric_code": "Libs.utils.selective_metrics"})
    target = Path(output) if output else root / "derived" / "selective_metrics.json"
    write_derived_artifact(target, payload, prediction_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.run_dir, num_classes=args.num_classes, output=args.output)


if __name__ == "__main__":
    main()
