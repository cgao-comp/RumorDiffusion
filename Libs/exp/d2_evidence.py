"""Qualify the additional D2 cohort without changing the frozen main cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from Libs.exp.aggregate_results import aggregate_prediction_rows
from Libs.exp.d2_run import MODEL_NAME, PROTOCOL, _validate_inputs
from Libs.exp.d2_training import DATASETS, SEEDS, SEARCH_SPACE, observation_windows
from Libs.scripts.validate_canonical_results import validate_run_directory
from Libs.utils.artifact_schema import canonical_json_hash, file_sha256, read_artifact_rows
from Libs.utils.provenance import write_json


def qualify_case(directory: Path, dataset: str, seed: int) -> list[dict]:
    index = json.loads((directory / "published.json").read_text())
    if index["dataset"] != dataset or index["seed"] != seed:
        raise ValueError("D2 publication index has the wrong case identity")
    checkpoint = directory / "checkpoint.pt"
    checksum = file_sha256(checkpoint)
    if index["checkpoint_sha256"] != checksum:
        raise ValueError("D2 checkpoint changed after publication")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    if meta["dataset"] != dataset or meta["seed"] != seed or meta["model_name"] != MODEL_NAME:
        raise ValueError("D2 checkpoint has the wrong case identity")
    if meta["code"]["dirty"] or meta["training"]["diagnostic"]:
        raise ValueError("Dirty-source or diagnostic training is not scientific evidence")
    if not meta["independent_implementation"] or meta["published_equivalence_claimed"]:
        raise ValueError("Independent D2 implementation must be identified accurately")
    _validate_inputs(meta["inputs"])
    if canonical_json_hash(meta["configuration"]) != meta["config_hash"]:
        raise ValueError("Checkpoint resolved configuration does not match its digest")
    config = meta["configuration"]["model"]
    for stage, budget in (("forecaster", "forecast_epochs"), ("search", "search_epochs"), ("classifier", "final_epochs")):
        record = meta["training"][stage]
        epochs = record["epochs_run"]
        if not config["training"]["min_epochs"] <= epochs <= config["training"][budget]:
            raise ValueError(f"{stage} did not satisfy its declared training budget")
        if len(record["history"]) != epochs or not 1 <= record["best_epoch"] <= epochs:
            raise ValueError(f"{stage} training history is incomplete")
        if record["metadata"]["training_split"] != "train" or record["metadata"]["selection_split"] != "val":
            raise ValueError("Training and selection split roles changed")
    if not meta["training"]["forecaster_frozen"] or not meta["training"]["classifier_retrained_from_scratch"]:
        raise ValueError("D2 stage separation is not established")
    architecture = meta["training"]["selected_architecture"]
    if set(architecture) != set(SEARCH_SPACE) or any(architecture[k] not in SEARCH_SPACE[k] for k in SEARCH_SPACE):
        raise ValueError("Selected architecture is outside the declared search space")
    split = json.loads(Path(meta["inputs"]["split_manifest.json"]["path"]).read_text())
    if (meta["training"]["training_sources"] != split["splits"]["train"]["sample_count"] or
            meta["training"]["validation_sources"] != split["splits"]["val"]["sample_count"]):
        raise ValueError("Training or validation source coverage is incomplete")
    unit, main, early = observation_windows(dataset)
    required = {("main", main), *(("early", window) for window in early)}
    observed = [(record["kind"], record["window"]) for record in index["runs"]]
    if len(observed) != len(set(observed)) or set(observed) != required:
        raise ValueError("D2 must have exactly one main and three early runs per checkpoint")
    result = []
    for record in index["runs"]:
        run = Path(record["run_dir"])
        manifest = validate_run_directory(run)
        variant = "independent" if record["kind"] == "main" else "independent_early"
        if (manifest["run_id"] != record["run_id"] or manifest["dataset"] != dataset or manifest["model"] != {"name": MODEL_NAME, "variant": variant}
                or manifest["horizon"] != {"unit": unit, "value": record["window"]}
                or manifest["seeds"] != {"split": 42, "ctdm": seed, "arbiter": seed, "inference": seed}
                or manifest["checkpoint"]["sha256"] != checksum or manifest["checkpoint"]["metadata"] != meta
                or manifest["code"]["git_commit"] != meta["code"]["git_commit"]):
            raise ValueError("D2 run does not match its frozen checkpoint and declared observation")
        rows = read_artifact_rows(run / "predictions.parquet")
        if [r["source_id"] for r in rows] != split["splits"]["test"]["source_ids"]:
            raise ValueError("D2 test prediction membership/order is incomplete")
        if any(r["config_sha256"] != meta["config_hash"] for r in rows):
            raise ValueError("D2 prediction configuration binding changed")
        witness = meta["calibration"][str(record["window"])]
        if witness["source_ids"] != split["splits"]["val"]["source_ids"] or witness["fit_split"] != "val":
            raise ValueError("D2 calibration did not use the complete validation split")
        recomputed = aggregate_prediction_rows(rows, num_classes=meta["num_classes"])
        stored = json.loads((run / "metrics.json").read_text())
        if recomputed["metrics"] != stored["metrics"]:
            raise ValueError("D2 metrics do not recompute from predictions")
        result.append({"dataset": dataset, "seed": seed, "kind": record["kind"], "window": record["window"],
                       "unit": unit, "run_id": record["run_id"], "sample_count": len(rows),
                       "metrics": recomputed["metrics"], "architecture": architecture,
                       "code_commit": meta["code"]["git_commit"], "config_hash": meta["config_hash"]})
    return result


def report(root: Path, cases: list[tuple[str, int]]) -> dict:
    rows, missing, failed = [], [], []
    for dataset, seed in cases:
        directory = root / "exploratory" / PROTOCOL / MODEL_NAME / dataset / f"seed_{seed}"
        if not (directory / "published.json").is_file():
            missing.append({"dataset": dataset, "seed": seed})
            continue
        try:
            rows.extend(qualify_case(directory, dataset, seed))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failed.append({"dataset": dataset, "seed": seed, "error": str(exc)})
    aggregates = []
    for dataset in DATASETS:
        keys = {(row["kind"], row["window"]) for row in rows if row["dataset"] == dataset}
        for kind, window in sorted(keys):
            group = [row for row in rows if (row["dataset"], row["kind"], row["window"]) == (dataset, kind, window)]
            if sorted(row["seed"] for row in group) != list(SEEDS):
                continue
            if len({(row["code_commit"], row["config_hash"]) for row in group}) != 1:
                failed.append({"dataset": dataset, "kind": kind, "window": window,
                               "error": "Three-seed group mixes source code or resolved configurations"})
                continue
            aggregates.append({"dataset": dataset, "kind": kind, "window": window, "n_seeds": 3,
                               "metrics": {metric: {"mean": mean(row["metrics"][metric] for row in group),
                                                    "std": stdev(row["metrics"][metric] for row in group)}
                                           for metric in ("accuracy", "macro_f1", "ece_calibrated")}})
    return {"status": "FAILED" if failed else "PARTIAL" if missing else "COMPLETE",
            "scope": "additional_independent_D2_cohort", "expected_main": len(cases), "expected_early": 3 * len(cases),
            "qualified_main": sum(row["kind"] == "main" for row in rows),
            "qualified_early": sum(row["kind"] == "early" for row in rows),
            "missing": missing, "failed": failed, "runs": rows, "aggregates": aggregates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", nargs=2, action="append", metavar=("DATASET", "SEED"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cases = [(dataset, int(seed)) for dataset, seed in args.case] if args.case else [(d, s) for d in DATASETS for s in SEEDS]
    if len(cases) != len(set(cases)) or any(d not in DATASETS or s not in SEEDS for d, s in cases):
        parser.error("Cases must be unique members of the four-dataset three-seed cohort")
    result = report(args.output_root, cases)
    write_json(args.report, result)
    print(json.dumps({key: result[key] for key in ("status", "qualified_main", "expected_main", "qualified_early", "expected_early")}))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
