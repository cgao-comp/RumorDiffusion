"""Export predeclared case-study predictions without ranking on test outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from Libs.exp.canonical_io import load_validated_predictions
from Libs.utils.artifact_schema import file_sha256


def export(run_dir: str | Path, source_ids: Sequence[str], output: str | Path) -> None:
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("case-study source IDs must be non-empty and unique")
    manifest, rows, prediction_path = load_validated_predictions(run_dir)
    indexed = {str(row["source_id"]): row for row in rows}
    missing = [source_id for source_id in source_ids if source_id not in indexed]
    if missing:
        raise ValueError(f"predeclared case IDs are absent from predictions: {missing}")
    payload = {"schema_version": "1", "run_id": manifest["run_id"], "selection": "predeclared_source_ids", "source_predictions": {"path": str(prediction_path), "sha256": file_sha256(prediction_path)}, "cases": [indexed[source_id] for source_id in source_ids]}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-ids", required=True, help="Comma-separated, predeclared source IDs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run_dir, [item.strip() for item in args.source_ids.split(",") if item.strip()], args.output)


if __name__ == "__main__":
    main()
