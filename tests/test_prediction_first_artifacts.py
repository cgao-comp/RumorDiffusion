from __future__ import annotations

import json

import torch
import pytest
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Libs.exp.aggregate_results import aggregate_canonical_protocol, aggregate_prediction_rows, build_table_cell_registry
from Libs.exp.canonical_artifacts import canonical_run_id, write_canonical_run
from Libs.exp.evaluate_latency import evaluate as evaluate_latency
from Libs.exp.evaluate_selective import evaluate as evaluate_selective
from Libs.exp.run import collect_eval_outputs
from Libs.exp.trainer import build_checkpoint_payload
from Libs.scripts.generate_paper_assets import validate_canonical_table_registry
from Libs.scripts.validate_canonical_results import validate_run_directory
from Libs.utils.artifact_schema import read_predictions_parquet, write_predictions_parquet


def _rows(*, compute_latency: bool = False):
    base = {
        "run_id": "run-1", "dataset": "twitter15", "split": "test", "horizon_unit": "minutes", "horizon_value": 10,
        "model": "rumordiffusion", "model_variant": "canonical", "split_seed": 42, "ctdm_seed": 42,
        "arbiter_seed": 42, "inference_seed": 42, "calibration_tag": "val_temperature", "sampling_S": 1,
        "checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64, "split_sha256": "c" * 64,
        "accepted": True, "decision": "accept", "uncertainty_score": 0.0,
    }
    rows = [
        {**base, "source_id": "s1", "y_true": 0, "logits": [2.0, 0.0], "probs_uncalibrated": [0.88, 0.12], "probs_calibrated": [0.88, 0.12], "y_pred": 0, "confidence": 0.88},
        {**base, "source_id": "s2", "y_true": 1, "logits": [0.0, 2.0], "probs_uncalibrated": [0.12, 0.88], "probs_calibrated": [0.12, 0.88], "y_pred": 1, "confidence": 0.88},
    ]
    if compute_latency:
        for index, row in enumerate(rows):
            row["compute_latency_ms"] = float(index + 1)
    return rows


def _run_inputs(log_path):
    return {
        "dataset": "twitter15", "split": "test", "horizon": {"unit": "minutes", "value": 10},
        "model": {"name": "rumordiffusion", "variant": "canonical"},
        "seeds": {"split": 42, "ctdm": 42, "arbiter": 42, "inference": 42, "bootstrap": 42, "missingness": 42},
        "calibration_tag": "val_temperature", "sampling_S": 1,
        "execution": {"start_time": "2026-08-06T00:00:00Z", "end_time": "2026-08-06T00:00:01Z", "exit_code": 0, "command": "synthetic smoke", "log_path": str(log_path)},
    }


def _manifests(tmp_path):
    paths = []
    for name in ("raw_manifest.json", "processed_manifest.json", "split_manifest.json"):
        path = tmp_path / name
        path.write_text('{"schema_version": "1"}', encoding="utf-8")
        paths.append(path)
    return paths


def _calibration():
    return {
        "schema_version": "1",
        "fit_split": "val",
        "threshold_fit_split": "val",
        "method": "temperature",
        "temperature": 1.0,
        "reject_threshold": 0.5,
    }


def _selective():
    return {
        "schema_version": "1",
        "threshold_fit_split": "val",
        "coverage": 1.0,
        "selective_risk": 0.0,
        "resolved_accuracy": 1.0,
        "resolved_macro_f1": 1.0,
    }


def _latency(*, measured: bool = True):
    if measured:
        return {
            "schema_version": "1",
            "measured": True,
            "compute_latency_ms": {"n": 2, "mean_ms": 1.5, "p50_ms": 1.5, "p95_ms": 1.95},
            "detection_latency": {"measured": False, "unit": "minutes", "value": 10},
        }
    return {
        "schema_version": "1",
        "measured": False,
        "status": "not_profiled",
        "compute_latency_ms": {"n": 0},
        "detection_latency": {"measured": False, "unit": "minutes", "value": 10},
    }


class _LatencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return torch.zeros((int(batch.y.view(-1).numel()), 2), dtype=torch.float32, device=batch.y.device)


def test_eval_latency_rows_align_with_prediction_rows():
    graphs = []
    for index in range(3):
        data = Data(
            x=torch.ones((1, 1), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor([index % 2], dtype=torch.long),
        )
        data.source_id = f"s{index}"
        graphs.append(data)
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    model = _LatencyModel()

    outputs = collect_eval_outputs(
        model,
        loader,
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(),
        profile=True,
        profile_repeat=2,
        profile_warmup_batches=1,
    )

    assert outputs["source_ids"] == ["s0", "s1", "s2"]
    assert len(outputs["latency_ms"]) == len(outputs["source_ids"])
    assert all(float(value) >= 0.0 for value in outputs["latency_ms"])
    assert model.calls == 5


def test_prediction_parquet_roundtrip_and_metrics(tmp_path):
    path = tmp_path / "predictions.parquet"
    record = write_predictions_parquet(_rows(), path)
    assert record["rows"] == 2
    restored = read_predictions_parquet(path)
    assert [row["source_id"] for row in restored] == ["s1", "s2"]
    metrics = aggregate_prediction_rows(restored, num_classes=2)
    assert metrics["metrics"]["accuracy"] == 1.0
    assert metrics["metrics"]["macro_f1"] == 1.0
    assert metrics["metrics"]["brier_calibrated"] > 0.0


def test_prediction_aggregation_rejects_duplicate_source_ids():
    rows = _rows()
    rows[1]["source_id"] = "s1"
    with pytest.raises(ValueError, match="duplicate source IDs"):
        aggregate_prediction_rows(rows, num_classes=2)


def test_small_canonical_smoke_is_traceable(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(build_checkpoint_payload({}, {"format_version": 1, "kind": "synthetic"}), checkpoint)
    log_path = tmp_path / "run.log"
    log_path.write_text("synthetic canonical smoke\n", encoding="utf-8")
    manifests = _manifests(tmp_path)
    run_inputs = _run_inputs(log_path)
    rows = _rows(compute_latency=True)
    run_id = canonical_run_id(
        run_inputs=run_inputs,
        exp_config={"dataset": "twitter15"},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
    )
    for row in rows:
        row["run_id"] = run_id
    monkeypatch.setattr(
        "Libs.exp.canonical_artifacts.git_state",
        lambda _: {"git_commit": "f" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64},
    )
    run_dir = write_canonical_run(
        tmp_path / "Output",
        protocol_id="smoke",
        run_inputs=run_inputs,
        checkpoint_path=checkpoint,
        exp_config={"dataset": "twitter15"},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
        prediction_rows=rows,
        num_classes=2,
        calibration=_calibration(),
        selective_metrics=_selective(),
        latency=_latency(),
    )
    validate_run_directory(run_dir)
    assert (run_dir / "predictions.parquet").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "resolved_exp.yaml").is_file()
    assert (run_dir / "resolved_model.yaml").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()


def test_canonical_writer_omits_optional_null_config_fields(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(build_checkpoint_payload({}, {"format_version": 1, "kind": "synthetic"}), checkpoint)
    log_path = tmp_path / "run.log"
    log_path.write_text("canonical config null regression\n", encoding="utf-8")
    manifests = _manifests(tmp_path)
    run_inputs = _run_inputs(log_path)
    exp_config = {
        "dataset": {"prefix": {"unit": "minutes", "window": 10, "k": None}},
        "training": {"max_batches_per_epoch": None},
        "rumordiffusion": {"explain": {"source_ids": ""}},
        "sbatch": {"profiles": {"benchmark": {"gres": "", "cuda_devices": [""]}}},
    }
    model_config = {"name": "rumordiffusion", "variant": "canonical"}
    run_id = canonical_run_id(
        run_inputs=run_inputs,
        exp_config=exp_config,
        model_config=model_config,
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
    )
    rows = _rows(compute_latency=True)
    for row in rows:
        row["run_id"] = run_id
    monkeypatch.setattr(
        "Libs.exp.canonical_artifacts.git_state",
        lambda _: {"git_commit": "f" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64},
    )

    run_dir = write_canonical_run(
        tmp_path / "Output",
        protocol_id="null-config-smoke",
        run_inputs=run_inputs,
        checkpoint_path=checkpoint,
        exp_config=exp_config,
        model_config=model_config,
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
        prediction_rows=rows,
        num_classes=2,
        calibration=_calibration(),
        selective_metrics=_selective(),
        latency=_latency(),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "k" not in manifest["resolved_config"]["payload"]["dataset"]["prefix"]
    assert "max_batches_per_epoch" not in manifest["resolved_config"]["payload"]["training"]
    assert "source_ids" not in manifest["resolved_config"]["payload"]["rumordiffusion"]["explain"]
    assert "gres" not in manifest["resolved_config"]["payload"]["sbatch"]["profiles"]["benchmark"]
    assert "cuda_devices" not in manifest["resolved_config"]["payload"]["sbatch"]["profiles"]["benchmark"]
    validate_run_directory(run_dir)


def test_canonical_writer_moves_missing_metadata_run_to_failed(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(build_checkpoint_payload({}, None), checkpoint)
    log_path = tmp_path / "run.log"
    log_path.write_text("missing metadata\n", encoding="utf-8")
    manifests = _manifests(tmp_path)
    run_inputs = _run_inputs(log_path)
    run_id = canonical_run_id(
        run_inputs=run_inputs,
        exp_config={"dataset": "twitter15"},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
    )
    rows = _rows()
    for row in rows:
        row["run_id"] = run_id
    monkeypatch.setattr(
        "Libs.exp.canonical_artifacts.git_state",
        lambda _: {"git_commit": "f" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="metadata"):
        write_canonical_run(
            tmp_path / "Output",
            protocol_id="smoke",
            run_inputs=run_inputs,
            checkpoint_path=checkpoint,
            exp_config={"dataset": "twitter15"},
            model_config={"name": "rumordiffusion", "variant": "canonical"},
            raw_manifest_path=manifests[0],
            processed_manifest_path=manifests[1],
            split_manifest_path=manifests[2],
            prediction_rows=rows,
            num_classes=2,
            calibration=_calibration(),
            selective_metrics=_selective(),
            latency=_latency(measured=False),
        )
    assert not (tmp_path / "Output" / "canonical" / "smoke" / "runs" / run_id).exists()
    assert (tmp_path / "Output" / "failed" / "smoke" / "runs" / run_id).is_dir()


def test_canonical_writer_rejects_duplicate_key_before_publish(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(build_checkpoint_payload({}, {"format_version": 1, "kind": "synthetic"}), checkpoint)
    log_path = tmp_path / "run.log"
    log_path.write_text("duplicate key\n", encoding="utf-8")
    manifests = _manifests(tmp_path)
    run_inputs = _run_inputs(log_path)
    monkeypatch.setattr(
        "Libs.exp.canonical_artifacts.git_state",
        lambda _: {"git_commit": "f" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64},
    )

    first_run_id = canonical_run_id(
        run_inputs=run_inputs,
        exp_config={"dataset": "twitter15", "version": 1},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
    )
    first_rows = _rows(compute_latency=True)
    for row in first_rows:
        row["run_id"] = first_run_id
    write_canonical_run(
        tmp_path / "Output",
        protocol_id="smoke",
        run_inputs=run_inputs,
        checkpoint_path=checkpoint,
        exp_config={"dataset": "twitter15", "version": 1},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
        prediction_rows=first_rows,
        num_classes=2,
        calibration=_calibration(),
        selective_metrics=_selective(),
        latency=_latency(),
    )

    second_run_id = canonical_run_id(
        run_inputs=run_inputs,
        exp_config={"dataset": "twitter15", "version": 2},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
    )
    second_rows = _rows(compute_latency=True)
    for row in second_rows:
        row["run_id"] = second_run_id
    with pytest.raises(FileExistsError, match="duplicate canonical key"):
        write_canonical_run(
            tmp_path / "Output",
            protocol_id="smoke",
            run_inputs=run_inputs,
            checkpoint_path=checkpoint,
            exp_config={"dataset": "twitter15", "version": 2},
            model_config={"name": "rumordiffusion", "variant": "canonical"},
            raw_manifest_path=manifests[0],
            processed_manifest_path=manifests[1],
            split_manifest_path=manifests[2],
            prediction_rows=second_rows,
            num_classes=2,
            calibration=_calibration(),
            selective_metrics=_selective(),
            latency=_latency(),
        )
    assert not (tmp_path / "Output" / "canonical" / "smoke" / "runs" / second_run_id).exists()
    assert (tmp_path / "Output" / "failed" / "smoke" / "runs" / second_run_id).is_dir()


def test_focused_selective_and_latency_runners_write_json_artifacts(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(build_checkpoint_payload({}, {"format_version": 1, "kind": "synthetic"}), checkpoint)
    log_path = tmp_path / "run.log"
    log_path.write_text("focused runners\n", encoding="utf-8")
    manifests = _manifests(tmp_path)
    run_inputs = _run_inputs(log_path)
    rows = _rows(compute_latency=True)
    run_id = canonical_run_id(
        run_inputs=run_inputs,
        exp_config={"dataset": "twitter15"},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
    )
    for row in rows:
        row["run_id"] = run_id
    monkeypatch.setattr(
        "Libs.exp.canonical_artifacts.git_state",
        lambda _: {"git_commit": "f" * 40, "dirty": False, "dirty_diff_sha256": "0" * 64},
    )
    run_dir = write_canonical_run(
        tmp_path / "Output",
        protocol_id="smoke",
        run_inputs=run_inputs,
        checkpoint_path=checkpoint,
        exp_config={"dataset": "twitter15"},
        model_config={"name": "rumordiffusion", "variant": "canonical"},
        raw_manifest_path=manifests[0],
        processed_manifest_path=manifests[1],
        split_manifest_path=manifests[2],
        prediction_rows=rows,
        num_classes=2,
        calibration=_calibration(),
        selective_metrics=_selective(),
        latency=_latency(),
    )

    selective_payload = evaluate_selective(run_dir, num_classes=2)
    latency_payload = evaluate_latency(run_dir)
    assert selective_payload["threshold_fit_split"] == "val"
    assert latency_payload["measured"] is True
    assert json.loads((run_dir / "derived" / "selective_metrics.json").read_text(encoding="utf-8"))[
        "source_predictions"
    ]["sha256"]
    assert json.loads((run_dir / "derived" / "latency.json").read_text(encoding="utf-8"))["compute_latency_ms"]["n"] == 2
    validate_run_directory(run_dir)
    metrics_path = run_dir / "metrics.json"
    registry_path = run_dir.parent.parent / "aggregates" / "table_cell_registry.json"
    build_table_cell_registry(
        [
            {
                "cell_key": "twitter15.rumordiffusion.accuracy",
                "artifact_path": str(metrics_path),
                "metric_pointer": "/metrics/accuracy",
                "value": 1.0,
            }
        ],
        registry_path,
    )
    report = validate_canonical_table_registry(run_dir.parent.parent, registry_path)
    assert report["verified_cells"]["twitter15.rumordiffusion.accuracy"]["value"] == 1.0
    aggregate = aggregate_canonical_protocol(run_dir.parent.parent, num_classes=2, expected_seeds=[42])
    assert aggregate["groups"][0]["seed_coverage"]["complete"] is True
    assert aggregate["groups"][0]["metrics"]["accuracy"]["mean"] == 1.0
