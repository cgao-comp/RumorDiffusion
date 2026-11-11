"""Independent D2 training and prediction-first publication entry point."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import socket
import sys
from datetime import datetime

import torch

from Libs.exp.canonical_artifacts import canonical_run_id, prediction_rows_from_logits, write_canonical_run
from Libs.exp.d2_training import (
    DATASETS, SEEDS, CompletionCache, WindowData, classifier_outputs,
    observation_windows, save_checkpoint, train_pipeline,
)
from Libs.exp.run import read_yaml_like
from Libs.models.baselines.D2 import D2Classifier, D2Forecaster
from Libs.scripts.validate_canonical_results import validate_run_directory
from Libs.utils.artifact_schema import canonical_json_hash, file_sha256, omit_optional_none_fields, read_artifact_rows
from Libs.utils.calibration import fit_temperature
from Libs.utils.provenance import environment_snapshot, git_state, write_json
from Libs.utils.selective_metrics import selective_metrics

MODEL_NAME = "d2_forecast_nas"
PROTOCOL = "rumordiffusion_canonical_v1"


def input_records(root: Path, cache_name: str) -> dict:
    names = ("raw_manifest.json", "processed_manifest.json", "split_manifest.json",
             "label_schema.json", "splits.json", "pyg/data.pt", "pyg/slices.pt", cache_name)
    return {name: {"path": str((root / name).resolve()), "sha256": file_sha256(root / name)} for name in names}


def _validate_inputs(records: dict) -> None:
    for record in records.values():
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Input changed since training: {record['path']}")


def _working_data_root(args: argparse.Namespace, records: dict, authoritative: Path) -> Path:
    if args.working_data_root is None:
        return authoritative
    working = args.working_data_root / args.dataset
    for name, record in records.items():
        if file_sha256(working / name) != record["sha256"]:
            raise ValueError(f"Staged dataset file differs from its authoritative input: {name}")
    return working


def _validate_full_outputs(outputs: dict, manifest: dict, split: str) -> None:
    if outputs["source_ids"] != manifest["splits"][split]["source_ids"]:
        raise ValueError(f"{split} predictions do not cover the immutable source order")
    if not torch.isfinite(outputs["logits"]).all():
        raise ValueError("Non-finite prediction logits")


def _load_pipeline(payload: dict) -> tuple:
    meta = payload["metadata"]
    config = meta["configuration"]["model"]
    kwargs = meta["model_kwargs"]
    forecaster = D2Forecaster(max_completion_nodes=config["model"]["max_completion_nodes"],
                             loss_weights=config["forecast"]["loss_weights"], **kwargs)
    classifier = D2Classifier(num_classes=meta["num_classes"],
                              architecture=meta["training"]["selected_architecture"], **kwargs)
    for name, model in (("forecaster", forecaster), ("classifier", classifier)):
        prefix = name + "."
        model.load_state_dict({key[len(prefix):]: value for key, value in payload["state_dict"].items()
                               if key.startswith(prefix)}, strict=True)
        model.eval().requires_grad_(False)
    return forecaster, classifier


def publish(args: argparse.Namespace, log) -> None:
    checkpoint = args.run_dir / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    state = git_state(args.source_root)
    if state["dirty"] or state["git_commit"] != meta["code"]["git_commit"]:
        raise ValueError("Publication requires the exact clean training source")
    if meta["training"]["diagnostic"]:
        raise ValueError("Diagnostic checkpoints cannot publish canonical evidence")
    if meta["dataset"] != args.dataset or meta["seed"] != args.seed:
        raise ValueError("Checkpoint does not match the requested dataset and seed")
    _validate_inputs(meta["inputs"])
    forecaster, classifier = _load_pipeline(payload)
    exp, config = meta["configuration"]["experiment"], meta["configuration"]["model"]
    data_root = Path(meta["data_root"])
    split_path = data_root / "split_manifest.json"
    split_manifest = json.loads(split_path.read_text())
    test = WindowData(_working_data_root(args, meta["inputs"], data_root), "test", args.dataset, targets=False,
                      text_config=exp["dataset"]["text_encoder"])
    # No cache during timing: each measurement includes completion + classification.
    cache = CompletionCache(forecaster, 0)
    unit, main_window, early = observation_windows(args.dataset)
    results = {window: classifier_outputs(classifier, test, cache, window, measure_latency=True)
               for window in test.windows}
    for outputs in results.values():
        _validate_full_outputs(outputs, split_manifest, "test")
    log({"stage": "test_inference_complete", "sources": test.size, "windows": list(test.windows)})
    checkpoint_hash = file_sha256(checkpoint)
    records = []
    for kind, window in (("main", main_window), *(("early", w) for w in early)):
        output = results[window]
        witness = meta["calibration"][str(window)]
        if witness["source_ids"] != split_manifest["splits"]["val"]["source_ids"]:
            raise ValueError("Calibration witness has incomplete validation membership")
        seeds = {"split": 42, "ctdm": args.seed, "arbiter": args.seed, "inference": args.seed}
        run_inputs = {
            "dataset": args.dataset, "split": "test", "horizon": {"unit": unit, "value": float(window)},
            "model": {"name": MODEL_NAME, "variant": "independent" if kind == "main" else "independent_early"},
            "seeds": seeds, "calibration_tag": "val_temperature", "sampling_S": 1,
        }
        arguments = dict(run_inputs=run_inputs, exp_config=exp, model_config=config,
                         raw_manifest_path=data_root / "raw_manifest.json",
                         processed_manifest_path=data_root / "processed_manifest.json", split_manifest_path=split_path)
        run_id = canonical_run_id(**arguments)
        expected_dir = args.output_root / "canonical" / PROTOCOL / "runs" / run_id
        rows = prediction_rows_from_logits(
            run_id=run_id, dataset=args.dataset, split="test", source_ids=output["source_ids"],
            logits=output["logits"], targets=output["targets"], horizon_unit=unit, horizon_value=window,
            model=MODEL_NAME, model_variant=run_inputs["model"]["variant"], seeds=seeds,
            calibration_tag="val_temperature", sampling_S=1, checkpoint_sha256=checkpoint_hash,
            config_sha256=meta["config_hash"], split_sha256=file_sha256(split_path),
            temperature=witness["temperature"], compute_latency_ms=output["latency_ms"],
        )
        if expected_dir.exists():
            manifest = validate_run_directory(expected_dir)
            existing = read_artifact_rows(expected_dir / "predictions.parquet")
            if manifest["checkpoint"]["sha256"] != checkpoint_hash or any(
                [{k: r[k] for k in ("source_id", "logits", "y_true", "probs_calibrated")} for r in data] !=
                [{k: r[k] for k in ("source_id", "logits", "y_true", "probs_calibrated")} for r in rows]
                for data in (existing,)
            ):
                raise ValueError("Existing D2 result conflicts with the frozen checkpoint")
            records.append({"kind": kind, "window": window, "run_dir": str(expected_dir), "run_id": run_id})
            continue
        selective = selective_metrics(output["targets"].tolist(), [r["probs_calibrated"] for r in rows],
                                      threshold=0.0, num_classes=meta["num_classes"],
                                      detection_delay=[float(window)] * len(rows))
        curve = selective["risk_coverage_curve"]
        selective["risk_coverage_curve"] = [
            {"coverage": float(c), "risk": float(r), **({"threshold": float(t)} if math.isfinite(float(t))
                                                        else {"threshold_policy": "above_max_confidence"})}
            for c, r, t in zip(curve["coverage"], curve["risk"], curve["threshold"])]
        selective.update(schema_version="1", policy="fixed_accept_all", threshold_fit_split="val")
        times = torch.tensor(output["latency_ms"], dtype=torch.float64)
        latency = {"schema_version": "1", "measured": True,
                   "scope": "CPU completion and classification; prefix construction excluded; no forecast cache",
                   "compute_latency_ms": {"n": len(rows), "mean_ms": float(times.mean()),
                                          "p50_ms": float(times.quantile(0.5)), "p95_ms": float(times.quantile(0.95))},
                   "detection_latency": {"measured": False, "unit": unit, "value": float(window)}}
        calibration = {"schema_version": "1", "method": "temperature", "temperature": witness["temperature"],
                       "fit_split": "val", "threshold_fit_split": "val", "threshold_policy": "fixed_accept_all",
                       "reject_threshold": 0.0, "status": "fitted", "validation_witness": witness}
        directory = write_canonical_run(
            args.output_root, protocol_id=PROTOCOL, checkpoint_path=checkpoint,
            **{**arguments, "run_inputs": {**run_inputs, "environment": environment_snapshot(args.source_root),
                  "execution": {"start_time": args.started_at, "end_time": datetime.now().astimezone().isoformat(),
                                "exit_code": 0, "command": shlex.join(sys.argv), "log_path": str(args.log_path)}}},
            prediction_rows=rows, num_classes=meta["num_classes"], calibration=calibration,
            selective_metrics=selective, latency=latency, repo_root=args.source_root,
        )
        records.append({"kind": kind, "window": window, "run_dir": str(directory), "run_id": run_id})
    write_json(args.run_dir / "published.json", {"dataset": args.dataset, "seed": args.seed,
                                                "checkpoint_sha256": checkpoint_hash, "runs": records})
    log({"stage": "publication_complete", "main": 1, "early": 3})


def train(args: argparse.Namespace, exp: dict, config: dict, log) -> None:
    data_root = args.processed_root / args.dataset
    state = git_state(args.source_root)
    if not args.diagnostic and state["dirty"]:
        raise ValueError("Production fitting requires a clean committed source")
    records = input_records(data_root, exp["dataset"]["text_encoder"]["cache_name"])
    limit = args.diagnostic_samples if args.diagnostic else 0
    working_root = _working_data_root(args, records, data_root)
    training = WindowData(working_root, "train", args.dataset, targets=True, limit=limit,
                          text_config=exp["dataset"]["text_encoder"])
    validation = WindowData(working_root, "val", args.dataset, targets=True, limit=limit,
                            text_config=exp["dataset"]["text_encoder"])
    forecaster, classifier, resolved, history, outputs = train_pipeline(
        training, validation, config, args.seed, args.run_dir, log, diagnostic=args.diagnostic)
    split_manifest = json.loads((data_root / "split_manifest.json").read_text())
    calibration = {}
    for window, output in outputs.items():
        if not args.diagnostic:
            _validate_full_outputs(output, split_manifest, "val")
        temperature = fit_temperature(output["logits"], output["targets"], split="val")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Validation calibration produced an invalid temperature")
        calibration[window] = {"source_ids": output["source_ids"], "logits": output["logits"].tolist(),
                               "targets": output["targets"].tolist(), "temperature": temperature,
                               "fit_split": "val", "algorithm": "scalar_temperature_lbfgs", "max_iter": 100}
    exp = omit_optional_none_fields(exp)
    resolved = omit_optional_none_fields(resolved)
    metadata = {
        "format_version": 1, "model_name": MODEL_NAME, "dataset": args.dataset, "seed": args.seed,
        "independent_implementation": True, "published_equivalence_claimed": False,
        "num_classes": training.num_classes, "configuration": {"experiment": exp, "model": resolved},
        "config_hash": canonical_json_hash({"experiment": exp, "model": resolved}),
        "inputs": records, "data_root": str(data_root), "code": state, "training": history,
        "calibration": calibration, "seed_roles": {"ctdm": "forecaster", "arbiter": "classifier"},
        "model_kwargs": {**{key: resolved["model"][key] for key in ("hidden", "num_layers", "dropout", "attention_heads")},
                         "text_dim": training.example(0, min(training.windows)).observed.text.numel()},
    }
    _validate_inputs(records)
    states = {f"{name}.{key}": value.detach().cpu().clone()
              for name, model in (("forecaster", forecaster), ("classifier", classifier))
              for key, value in model.state_dict().items()}
    payload = {"state_dict": states, "metadata": metadata}
    save_checkpoint(args.run_dir / "checkpoint.pt", payload)
    write_json(args.run_dir / "training_summary.json", metadata)
    # Exercise the same checkpoint loader used by production publication on real val data.
    restored_forecaster, restored_classifier = _load_pipeline(torch.load(
        args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False))
    restored_cache = CompletionCache(restored_forecaster, 0)
    example = validation.example(0, min(validation.windows))
    with torch.no_grad():
        actual = restored_classifier(example.observed, restored_cache.get(example))
    expected = outputs[str(min(validation.windows))]["logits"][:1]
    if not torch.equal(actual, expected):
        raise AssertionError("Checkpoint roundtrip changed validation predictions")
    log({"stage": "training_complete", "diagnostic": args.diagnostic,
         "checkpoint_roundtrip": "passed", "selected_architecture": history["selected_architecture"]})


def main() -> None:
    source_root = Path(__file__).resolve().parents[2]
    project_root = Path(os.environ.get("DPLUS_REPO_ROOT", "/share/home/hwxu/Researches/D2Plus"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "publish"))
    parser.add_argument("--dataset", choices=DATASETS, default="pheme")
    parser.add_argument("--seed", choices=SEEDS, type=int, default=42)
    parser.add_argument("--exp", type=Path, default=source_root / "Libs/configs/exp_d2_forecast_nas.yaml")
    parser.add_argument("--config", type=Path, default=source_root / "Libs/configs/baselines/d2_forecast_nas.yaml")
    parser.add_argument("--processed-root", type=Path, default=project_root / "Input/processed_canonical_v1")
    parser.add_argument("--working-data-root", type=Path, default=os.environ.get("DPLUS_D2_DATA_ROOT"))
    parser.add_argument("--output-root", type=Path, default=project_root / "Output")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-samples", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.source_root = source_root
    args.processed_root = args.processed_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.diagnostic and (args.action != "train" or args.run_dir is None or args.diagnostic_samples < 1):
        parser.error("Diagnostics require train, a distinct --run-dir and positive real-sample count")
    if not args.diagnostic and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Production train/publish must execute inside the authorized Slurm allocation")
    if socket.gethostname().split(".")[0] != "cpu04":
        parser.error("Current execution authorization is cpu04 only")
    if not 1 <= args.threads <= 8:
        parser.error("CPU thread count must be within the authorized 8 CPUs")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("DPLUS_OUTPUT_ROOT", str(args.output_root))
    os.environ.setdefault("RUMORDIFFUSION_PROCESSED_ROOT", str(args.processed_root))
    os.environ.setdefault("RUMORDIFFUSION_TEXT_ENCODER", str(project_root / "Input/sentence_transformers_all_mpnet_base_v2"))
    exp, config = read_yaml_like(str(args.exp)), read_yaml_like(str(args.config))
    if exp["experiment"]["seeds"] != list(SEEDS) or config["name"] != MODEL_NAME:
        raise ValueError("D2 requires the active three-seed contract and its own configuration")
    if exp["split"]["seed"] != 42 or any(exp["dataset"]["observability"].get(key, 0.0) != 0.0
                                         for key in ("drop_node", "drop_edge")):
        raise ValueError("D2 requires the immutable split42 and complete registered observation views")
    if not exp["dataset"]["text_encoder"]["normalize_embeddings"]:
        raise ValueError("D2 requires the fixed normalized text embedding cache")
    exp["dataset"]["name"] = args.dataset
    exp["model"] = {"name": MODEL_NAME, "variant": "independent"}
    args.run_dir = (args.run_dir or args.output_root / "exploratory" / PROTOCOL / MODEL_NAME /
                    args.dataset / f"seed_{args.seed}").resolve()
    if args.diagnostic and "canonical" in args.run_dir.parts:
        raise ValueError("Diagnostics cannot write under canonical output")
    if args.action == "train":
        args.run_dir.mkdir(parents=True, exist_ok=False)
    elif not (args.run_dir / "checkpoint.pt").is_file():
        raise FileNotFoundError("Publication requires the completed training checkpoint")
    args.log_path = args.run_dir / (args.action + ".log")
    args.started_at = datetime.now().astimezone().isoformat()
    with args.log_path.open("a", encoding="utf-8") as handle:
        def log(row: dict) -> None:
            message = json.dumps(row, allow_nan=False, sort_keys=True)
            handle.write(message + "\n")
            handle.flush()
            print(message, flush=True)
        log({"stage": "start", "action": args.action, "time": args.started_at, "seed": args.seed,
             "dataset": args.dataset, "threads": args.threads, "diagnostic": args.diagnostic})
        if args.action == "train":
            train(args, exp, config, log)
        else:
            publish(args, log)


if __name__ == "__main__":
    main()
