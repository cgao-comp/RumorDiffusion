"""Prediction-first aggregation for canonical runs.

Aggregates are derived from validated per-example prediction artifacts. Legacy
CSV files are intentionally not accepted by this module.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from Libs.utils.artifact_schema import (
    assert_artifact_fresh,
    canonical_json_hash,
    file_sha256,
    read_artifact_rows,
    validate_prediction_rows,
)
from Libs.utils.metrics import (
    holm_bonferroni,
    metrics_from_probabilities,
    paired_bootstrap,
    seed_level_summary,
)


def aggregate_prediction_rows(rows: Sequence[Mapping[str, Any]], *, num_classes: int, ece_bins: int = 15) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty prediction artifact")
    validate_prediction_rows(rows)
    source_ids = [str(row["source_id"]) for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("prediction artifact contains duplicate source IDs")
    keys = {(row["dataset"], row["split"], row["horizon_unit"], row["horizon_value"], row["model"], row["model_variant"], row["split_seed"], row["ctdm_seed"], row["arbiter_seed"], row["inference_seed"], row["calibration_tag"], row["sampling_S"]) for row in rows}
    if len(keys) != 1:
        raise ValueError("prediction rows contain inconsistent canonical keys")
    y_true = torch.tensor([int(row["y_true"]) for row in rows], dtype=torch.long)
    probs_uncal = torch.tensor([row["probs_uncalibrated"] for row in rows], dtype=torch.float64)
    probs_cal = torch.tensor([row["probs_calibrated"] for row in rows], dtype=torch.float64)
    if probs_uncal.size(1) != num_classes or probs_cal.size(1) != num_classes:
        raise ValueError("prediction class dimension does not match declared class space")
    uncal = metrics_from_probabilities(probs_uncal, y_true, num_classes, ece_bins)
    cal = metrics_from_probabilities(probs_cal, y_true, num_classes, ece_bins)
    latency = [float(row["compute_latency_ms"]) for row in rows if row.get("compute_latency_ms") is not None]
    result = {
        "schema_version": "1",
        "run_id": str(rows[0]["run_id"]),
        "dataset": str(rows[0]["dataset"]),
        "split": str(rows[0]["split"]),
        "horizon": {"unit": rows[0]["horizon_unit"], "value": rows[0]["horizon_value"]},
        "model": {"name": rows[0]["model"], "variant": rows[0]["model_variant"]},
        "sample_count": len(rows),
        "source_ids_sha256": canonical_json_hash(sorted(source_ids)),
        "metric_definition": {
            "class_space": list(range(num_classes)),
            "ece_bins": int(ece_bins),
            "ci_method": "student_t_seed_level",
            "test_report_only": True,
        },
        "metrics": {
            "accuracy": cal["acc"],
            "macro_f1": cal["macro_f1"],
            "per_class": cal["per_class"],
            "nll_uncalibrated": uncal["nll"],
            "nll_calibrated": cal["nll"],
            "ece_uncalibrated": compute_ece_value(probs_uncal, y_true, ece_bins),
            "ece_calibrated": compute_ece_value(probs_cal, y_true, ece_bins),
            "brier_uncalibrated": float(uncal["brier"]),
            "brier_calibrated": float(cal["brier"]),
        },
        "source_predictions": {"schema_version": "1", "rows": len(rows)},
    }
    if latency:
        ordered = sorted(latency)
        result["latency"] = {
            "compute_latency_ms": {"mean_ms": sum(latency) / len(latency), "p50_ms": _quantile(ordered, 0.50), "p95_ms": _quantile(ordered, 0.95), "n": len(latency)},
            "detection_latency": {"measured": False, "unit": rows[0]["horizon_unit"], "value": rows[0]["horizon_value"]},
        }
    return result


def compute_ece_value(probs: torch.Tensor, targets: torch.Tensor, bins: int) -> float:
    from Libs.utils.metrics import compute_ece

    return compute_ece(probs, targets, n_bins=bins)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty sequence")
    position = (len(values) - 1) * q
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    weight = position - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def aggregate_run(run_dir: str | Path, *, num_classes: int, ece_bins: int = 15) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    prediction_record = manifest.get("predictions")
    if not isinstance(prediction_record, Mapping):
        raise ValueError("manifest has no predictions record")
    prediction_path = Path(str(prediction_record["path"]))
    if not prediction_path.is_absolute():
        prediction_path = root / prediction_path
    assert_artifact_fresh(prediction_path, str(prediction_record["sha256"]), context="prediction artifact")
    rows = read_artifact_rows(prediction_path)
    metrics = aggregate_prediction_rows(rows, num_classes=num_classes, ece_bins=ece_bins)
    metrics["source_predictions"] = {"path": str(prediction_path), "sha256": file_sha256(prediction_path), "schema_version": str(prediction_record.get("schema_version", "1"))}
    return metrics


def aggregate_canonical_protocol(
    canonical_root: str | Path,
    *,
    num_classes: int,
    expected_seeds: Sequence[int] = (42, 3407, 126, 218, 91),
    ece_bins: int = 15,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate validated canonical runs with complete seed coverage.

    Runs are grouped by dataset/split/horizon/model/calibration/sampling. The
    inference seed is the formal seed identity for ordinary main-result groups;
    repeated or missing seeds fail closed instead of being pooled or selected.
    """
    from Libs.scripts.validate_canonical_results import validate_canonical_root

    root = Path(canonical_root)
    validate_canonical_root(root)
    runs_root = root / "runs" if (root / "runs").is_dir() else root
    expected = sorted({int(seed) for seed in expected_seeds})
    if not expected:
        raise ValueError("expected_seeds must be non-empty")
    groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any], Path]]] = defaultdict(list)
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = Path(str(manifest["metrics"]["path"]))
        if not metrics_path.is_absolute():
            metrics_path = run_dir / metrics_path
        metrics = aggregate_run(run_dir, num_classes=num_classes, ece_bins=ece_bins)
        key = (
            manifest["dataset"],
            manifest["split"],
            manifest["horizon"]["unit"],
            float(manifest["horizon"]["value"]),
            manifest["model"]["name"],
            manifest["model"]["variant"],
            manifest["calibration_tag"],
            int(manifest["sampling_S"]),
        )
        groups[key].append((manifest, metrics, metrics_path))

    aggregates: list[dict[str, Any]] = []
    for key, entries in sorted(groups.items(), key=lambda item: repr(item[0])):
        seed_to_entry: dict[int, tuple[dict[str, Any], dict[str, Any], Path]] = {}
        for manifest, metrics, metrics_path in entries:
            seed = int(manifest["seeds"]["inference"])
            if seed in seed_to_entry:
                raise ValueError(f"duplicate formal inference seed {seed} in canonical group {key!r}")
            seed_to_entry[seed] = (manifest, metrics, metrics_path)
        observed = sorted(seed_to_entry)
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            unexpected = sorted(set(observed) - set(expected))
            raise ValueError(
                f"canonical seed coverage incomplete for {key!r}: missing={missing}, unexpected={unexpected}"
            )
        metric_names = sorted(
            {
                name
                for _, metrics, _ in seed_to_entry.values()
                for name, value in metrics["metrics"].items()
                if isinstance(value, (int, float))
            }
        )
        seed_metrics = {
            name: seed_level_summary([float(seed_to_entry[seed][1]["metrics"][name]) for seed in expected])
            for name in metric_names
        }
        aggregates.append(
            {
                "identity": {
                    "dataset": key[0],
                    "split": key[1],
                    "horizon": {"unit": key[2], "value": key[3]},
                    "model": {"name": key[4], "variant": key[5]},
                    "calibration_tag": key[6],
                    "sampling_S": key[7],
                },
                "seed_coverage": {
                    "expected": expected,
                    "observed": observed,
                    "complete": True,
                    "seed_field": "inference_seed",
                },
                "metrics": seed_metrics,
                "source_runs": [
                    {
                        "run_id": seed_to_entry[seed][0]["run_id"],
                        "seed": seed,
                        "metrics_path": str(seed_to_entry[seed][2]),
                        "metrics_sha256": file_sha256(seed_to_entry[seed][2]),
                    }
                    for seed in expected
                ],
            }
        )
    payload = {
        "schema_version": "1",
        "canonical_root": str(root),
        "expected_seeds": expected,
        "metric_definition": {
            "ece_bins": int(ece_bins),
            "ci_method": "student_t_seed_level",
            "confidence": 0.95,
            "test_report_only": True,
        },
        "groups": aggregates,
    }
    target = Path(output) if output else root / "aggregates" / "main_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def aggregate_seed_metrics(seed_metrics: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(item["metrics"][metric]) for item in seed_metrics]
    return seed_level_summary(values)


def paired_significance(reference_rows: Sequence[Mapping[str, Any]], comparison_rows: Sequence[Mapping[str, Any]], *, n_bootstrap: int = 2000, seed: int = 42) -> dict[str, Any]:
    def score_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {str(row["source_id"]): float(int(row["y_pred"]) == int(row["y_true"])) for row in rows}

    return paired_bootstrap(score_map(reference_rows), score_map(comparison_rows), n_bootstrap=n_bootstrap, seed=seed)


def holm_correct_significance(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    raw = {name: float(result["p_value"]) for name, result in comparisons.items()}
    adjusted = holm_bonferroni(raw)
    return {name: {**dict(result), "p_value_holm": adjusted[name], "p_value_unadjusted": raw[name], "correction": "holm"} for name, result in comparisons.items()}


def build_table_cell_registry(metric_artifacts: Iterable[Mapping[str, Any]], output_path: str | Path) -> dict[str, Any]:
    cells = {}
    for artifact in metric_artifacts:
        if "artifact_path" not in artifact or "metric_pointer" not in artifact or "value" not in artifact:
            raise ValueError("table cell requires artifact_path, metric_pointer and value")
        value = artifact["value"]
        if value is None or (isinstance(value, float) and not torch.isfinite(torch.tensor(value))):
            raise ValueError("table cell contains a non-finite value")
        key = str(artifact.get("cell_key", f"cell_{len(cells)}"))
        if key in cells:
            raise ValueError(f"duplicate table cell key: {key}")
        path = Path(str(artifact["artifact_path"]))
        cells[key] = {"artifact_path": str(path), "artifact_sha256": file_sha256(path), "metric_pointer": str(artifact["metric_pointer"]), "value": value}
    payload = {"schema_version": "1", "cells": cells}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def write_discrepancy_report(
    authentic_metrics: Mapping[str, Any], *, legacy_metrics: Mapping[str, Any] | None = None,
    json_path: str | Path, markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare authentic aggregates to legacy records without importing either into canonical data."""
    authentic = authentic_metrics.get("metrics", authentic_metrics)
    legacy = legacy_metrics.get("metrics", legacy_metrics) if legacy_metrics is not None else None
    comparisons = {}
    if legacy is not None:
        for key, value in authentic.items():
            if isinstance(value, (int, float)) and isinstance(legacy.get(key), (int, float)):
                comparisons[key] = {"authentic": float(value), "legacy": float(legacy[key]), "delta": float(value) - float(legacy[key])}
    payload = {"schema_version": "1", "status": "compared" if legacy is not None else "legacy_not_supplied", "authentic_run_id": authentic_metrics.get("run_id"), "comparisons": comparisons, "canonical_unchanged": True}
    target = Path(json_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        md = ["# Canonical discrepancy report", "", f"Status: `{payload['status']}`", "", "Authentic metrics remain the canonical source; legacy values are comparison-only.", ""]
        if comparisons:
            md.extend(["| Metric | Authentic | Legacy | Delta |", "|---|---:|---:|---:|"])
            md.extend(f"| {key} | {item['authentic']:.6g} | {item['legacy']:.6g} | {item['delta']:.6g} |" for key, item in comparisons.items())
        else:
            md.append("No legacy aggregate was supplied; no discrepancy value was inferred.")
        Path(markdown_path).write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()
    if args.canonical_root is not None:
        payload = aggregate_canonical_protocol(
            args.canonical_root,
            num_classes=args.num_classes,
            ece_bins=args.ece_bins,
            output=args.output,
        )
        output = args.output or args.canonical_root / "aggregates" / "main_results.json"
    else:
        if args.run_dir is None:
            parser.error("run_dir or --canonical-root is required")
        payload = aggregate_run(args.run_dir, num_classes=args.num_classes, ece_bins=args.ece_bins)
        output = args.output or args.run_dir / "metrics.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote prediction-derived metrics: {output}")


if __name__ == "__main__":
    main()
