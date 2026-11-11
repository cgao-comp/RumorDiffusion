"""Audit factorial CTDM/arbiter/inference seed runs without retraining them."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from Libs.scripts.validate_canonical_results import validate_canonical_root
from Libs.utils.metrics import seed_level_summary


STAGES = ("ctdm", "arbiter", "inference")


def evaluate(root: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    canonical_root = Path(root)
    validate_canonical_root(canonical_root)
    runs_root = canonical_root / "runs" if (canonical_root / "runs").is_dir() else canonical_root
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment = manifest.get("factorial_seed_experiment")
        if not isinstance(experiment, dict):
            continue
        varied = str(experiment.get("varied_stage", ""))
        if varied not in {*STAGES, "all"}:
            raise ValueError(f"invalid factorial varied_stage in {manifest_path}: {varied}")
        metrics = json.loads((manifest_path.parent / str(manifest["metrics"]["path"])).read_text(encoding="utf-8"))
        groups[varied].append({"run_id": manifest["run_id"], "seeds": manifest["seeds"], "metrics": metrics["metrics"]})
    if not groups:
        raise ValueError("no factorial seed manifests found")
    result: dict[str, Any] = {"schema_version": "1", "stages": {}}
    for stage, entries in groups.items():
        varying = STAGES if stage == "all" else (stage,)
        fixed = [name for name in STAGES if name not in varying]
        if stage != "all" and len({entry["seeds"][stage] for entry in entries}) < 2:
            raise ValueError(f"factorial group {stage} does not vary {stage} seed")
        if stage != "all" and any(len({entry["seeds"][name] for entry in entries}) != 1 for name in fixed):
            raise ValueError(f"factorial group {stage} changes a stage declared fixed")
        metric_names = sorted(set.intersection(*(set(entry["metrics"]) for entry in entries)))
        result["stages"][stage] = {"runs": [entry["run_id"] for entry in entries], "summaries": {name: seed_level_summary([float(entry["metrics"][name]) for entry in entries]) for name in metric_names if isinstance(entries[0]["metrics"][name], (float, int))}}
    target = Path(output) if output else canonical_root / "aggregates" / "factorial_seeds.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluate(args.canonical_root, output=args.output)


if __name__ == "__main__":
    main()
