"""Summarise recorded compute and detection latency from canonical predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from Libs.exp.aggregate_results import _quantile
from Libs.exp.canonical_io import load_validated_predictions, write_derived_artifact


def evaluate(run_dir: str | Path, *, output: str | Path | None = None) -> dict:
    root = Path(run_dir)
    manifest, rows, prediction_path = load_validated_predictions(root)
    values = [float(row["compute_latency_ms"]) for row in rows if row.get("compute_latency_ms") is not None]
    if not values:
        raise ValueError("predictions do not contain compute_latency_ms; profile before canonical publication")
    ordered = sorted(values)
    payload = {
        "schema_version": "1", "run_id": manifest["run_id"],
        "measured": True,
        "measurement": {"cuda_synchronized": bool(manifest.get("latency_profile", {}).get("cuda_synchronized", False)), "warmup_batches": manifest.get("latency_profile", {}).get("warmup_batches"), "fixed_batch_size": manifest.get("batch_size"), "fixed_sample_order": bool(manifest.get("latency_profile", {}).get("fixed_sample_order", False))},
        "compute_latency_ms": {"mean_ms": sum(values) / len(values), "p50_ms": _quantile(ordered, 0.5), "p95_ms": _quantile(ordered, 0.95), "n": len(values)},
        "detection_latency": {"measured": False, "unit": rows[0]["horizon_unit"], "value": rows[0]["horizon_value"]},
    }
    target = Path(output) if output else root / "derived" / "latency.json"
    write_derived_artifact(target, payload, prediction_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.run_dir, output=args.output)


if __name__ == "__main__":
    main()
