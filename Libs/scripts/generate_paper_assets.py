#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validate_canonical_results import validate_canonical_root  # noqa: E402
from Libs.utils.artifact_schema import file_sha256, reject_placeholders, validate_metrics_artifact  # noqa: E402


SELECTED_MODELS = [
    "textonly",
    "bigcn",
    "tgat",
    "lp_gcn",
    "dga_fake",
    "agad",
    "hpcl_cascade",
    "tpso",
    "rumordiffusion",
]

MODEL_LABELS = {
    "textonly": "TextOnly (Sentence Emb.) \\cite{reimers2019sentence}",
    "bigcn": "BiGCN \\cite{bai2020rumor}",
    "tgat": "TGAT \\cite{xu2020tgat}",
    "lp_gcn": "LP-GCN (D$^2$)",
    "dga_fake": "DGA-Fake \\cite{zhang2024dgafake}",
    "agad": "AGAD \\cite{zhang2024agad}",
    "hpcl_cascade": "HPCL-cascade \\cite{zheng2024hpcl}",
    "tpso": "TPSO \\cite{peng2025tpso}",
    "rumordiffusion": "RumorDiffusion (Ours)",
}

MODEL_CONFIGS = {
    "textonly": "Libs/configs/baselines/textonly.yaml",
    "bigcn": "Libs/configs/baselines/bigcn.yaml",
    "tgat": "Libs/configs/baselines/tgat.yaml",
    "lp_gcn": "Libs/configs/baselines/lp_gcn.yaml",
    "dga_fake": "Libs/configs/baselines/dga_fake.yaml",
    "agad": "Libs/configs/baselines/agad.yaml",
    "hpcl_cascade": "Libs/configs/baselines/hpcl_cascade.yaml",
    "tpso": "Libs/configs/baselines/tpso.yaml",
    "rumordiffusion": "Libs/configs/baselines/rumordiffusion.yaml",
}

ABLATION_ROWS = [
    ("rumordiffusion", "RumorDiffusion (full)", "calibrated"),
    ("rumordiffusion_without_adversarial_regularization", "w/o reg", "calibrated"),
    ("rumordiffusion_without_pair_awareness", "w/o pair", "calibrated"),
    ("rumordiffusion_without_temporal_augmentation", "w/o temporal", "calibrated"),
    ("rumordiffusion_without_text_fusion", "w/o text", "calibrated"),
    ("rumordiffusion", "w/o temperature", "uncalibrated"),
]

DATASET_LABELS = {
    "twitter15": "Twitter15",
    "twitter16": "Twitter16",
    "weibo": "Weibo",
    "pheme": "PHEME",
}

PALETTE = {
    "ink": "#263238",
    "muted": "#607D8B",
    "blue": "#2A6FBB",
    "blue_light": "#E7F0FA",
    "teal": "#2A9D8F",
    "teal_light": "#E7F4F1",
    "gold": "#E9A93A",
    "gold_light": "#FFF6DF",
    "rose": "#C94C4C",
    "rose_light": "#FBEAEA",
    "green": "#3B8F4F",
    "green_light": "#EAF6ED",
    "gray": "#F4F6F8",
    "edge": "#B7C3CC",
}

TABLE_HIGHLIGHT_COLOR = "gray!20"


def highlight_best_cell(cell: str) -> str:
    return rf"\textbf{{\cellcolor{{{TABLE_HIGHLIGHT_COLOR}}}{{{cell}}}}}"


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_yaml_config(yaml_path: Path) -> Dict[str, Any]:
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing YAML config: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must contain a mapping: {yaml_path}")
    return loaded


def parse_float(value: str) -> float | None:
    cell = str(value).strip()
    if cell == "" or "±" in cell:
        return None
    try:
        value = float(cell)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def latest_rows_by_key(rows: Iterable[Dict[str, str]], key_fields: Sequence[str]) -> List[Dict[str, str]]:
    """Returns unique rows and fails on conflicting duplicate keys.

    Historical callers used timestamp-based latest-row selection. Canonical
    evidence must not silently overwrite a conflicting result, so duplicates
    are now an explicit validation error.
    """
    unique: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        if key in unique:
            raise ValueError(f"Duplicate result key {key!r}; timestamp conflict resolution is disabled")
        unique[key] = row
    return list(unique.values())


def mean_confidence(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values) - 1)
    std_value = math.sqrt(max(variance, 0.0))
    ci95 = 1.96 * std_value / math.sqrt(float(len(values)))
    return float(mean_value), float(ci95)


def bounded_confidence_interval(
    mean_value: float,
    ci95: float,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> Tuple[float, float]:
    lower_value = float(mean_value) - float(ci95)
    upper_value = float(mean_value) + float(ci95)
    if lower_bound is not None:
        lower_value = max(float(lower_bound), lower_value)
    if upper_bound is not None:
        upper_value = min(float(upper_bound), upper_value)
    return lower_value, upper_value


def bounded_confidence_errors(
    means: Sequence[float],
    cis: Sequence[float],
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> List[List[float]]:
    lower_errors: List[float] = []
    upper_errors: List[float] = []
    for mean_value, ci95 in zip(means, cis):
        lower_value, upper_value = bounded_confidence_interval(mean_value, ci95, lower_bound, upper_bound)
        lower_errors.append(max(float(mean_value) - lower_value, 0.0))
        upper_errors.append(max(upper_value - float(mean_value), 0.0))
    return [lower_errors, upper_errors]


def bounded_axis_limits(
    means: Sequence[float],
    cis: Sequence[float],
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    padding_fraction: float = 0.08,
    minimum_span: float = 0.01,
) -> Tuple[float, float]:
    intervals = [
        bounded_confidence_interval(mean_value, ci95, lower_bound, upper_bound)
        for mean_value, ci95 in zip(means, cis)
    ]
    lower_value = min(interval[0] for interval in intervals)
    upper_value = max(interval[1] for interval in intervals)
    span = max(upper_value - lower_value, minimum_span)
    padded_lower = lower_value - span * padding_fraction
    padded_upper = upper_value + span * padding_fraction
    if lower_bound is not None:
        padded_lower = max(float(lower_bound), padded_lower)
    if upper_bound is not None:
        padded_upper = min(float(upper_bound), padded_upper)
    return padded_lower, padded_upper


def format_stat(values: Sequence[float]) -> str:
    mean_value, ci95 = mean_confidence(values)
    return f"{mean_value:.3f}$\\pm${ci95:.3f}"


def normalize_int_cell(value: str, field_name: str) -> int:
    parsed = parse_float(value)
    if parsed is None:
        raise ValueError(f"Invalid numeric {field_name}: {value}")
    rounded = int(parsed)
    if float(rounded) != float(parsed):
        raise ValueError(f"{field_name} must be integer-valued: {value}")
    return rounded


def resolve_formal_sampling_contract(exp_cfg_path: Path, model_cfg_path: Path) -> Tuple[List[int], List[int]]:
    exp_cfg = read_yaml_config(exp_cfg_path)
    model_cfg = read_yaml_config(model_cfg_path)

    seeds = exp_cfg.get("experiment", {}).get("seeds", [])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"Formal seed list is missing from {exp_cfg_path}")
    seed_values = [int(seed) for seed in seeds]

    sampling_cfg = model_cfg.get("evaluation", {}).get("sampling_sweep", {})
    if not isinstance(sampling_cfg, dict):
        raise ValueError(f"Sampling sweep config is missing from {model_cfg_path}")
    samples = sampling_cfg.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"Sampling-budget list is missing from {model_cfg_path}")
    sample_values = [int(sample) for sample in samples]
    return seed_values, sample_values


def resolve_generation_contract(exp_cfg_path: Path, model_cfg_path: Path) -> Dict[str, Any]:
    exp_cfg = read_yaml_config(exp_cfg_path)
    model_cfg = read_yaml_config(model_cfg_path)
    seed_values, sample_values = resolve_formal_sampling_contract(exp_cfg_path, model_cfg_path)
    eval_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    model_eval_cfg = model_cfg.get("evaluation", {}) if isinstance(model_cfg.get("evaluation", {}), dict) else {}
    calibration_cfg = model_eval_cfg.get("calibration", {}) if isinstance(model_eval_cfg.get("calibration", {}), dict) else {}
    ece_bins = int(eval_cfg.get("ece_bins", calibration_cfg.get("ece_bins", 15)))
    early_cfg = eval_cfg.get("early_observation", {}) if isinstance(eval_cfg.get("early_observation", {}), dict) else {}
    threshold_cfg = model_eval_cfg.get("threshold_sweep", {}) if isinstance(model_eval_cfg.get("threshold_sweep", {}), dict) else {}
    display_thresholds = threshold_cfg.get("display", []) if isinstance(threshold_cfg, dict) else []
    if not isinstance(display_thresholds, list) or len(display_thresholds) == 0:
        raise ValueError(f"evaluation.threshold_sweep.display must be declared in {model_cfg_path}")
    transfer_cfg = exp_cfg.get("transfer", {}) if isinstance(exp_cfg.get("transfer", {}), dict) else {}
    transfer_model_paths = transfer_cfg.get("models", [])
    if not isinstance(transfer_model_paths, list) or len(transfer_model_paths) == 0:
        raise ValueError(f"transfer.models must be declared in {exp_cfg_path}")
    transfer_models: List[str] = []
    for transfer_model_path in transfer_model_paths:
        transfer_model_cfg = read_yaml_config(Path(str(transfer_model_path)))
        transfer_model_name = str(transfer_model_cfg.get("name", "")).lower()
        if len(transfer_model_name) == 0:
            raise ValueError(f"Transfer model config lacks a name: {transfer_model_path}")
        transfer_models.append(transfer_model_name)
    transfer_pairs_cfg = transfer_cfg.get("pairs", [])
    if not isinstance(transfer_pairs_cfg, list) or len(transfer_pairs_cfg) == 0:
        raise ValueError(f"transfer.pairs must be declared in {exp_cfg_path}")
    transfer_pairs: List[Tuple[str, str]] = []
    for pair in transfer_pairs_cfg:
        if not isinstance(pair, dict):
            raise ValueError(f"Transfer pair must be a mapping in {exp_cfg_path}")
        source = str(pair.get("source", ""))
        target = str(pair.get("target", ""))
        if len(source) == 0 or len(target) == 0:
            raise ValueError(f"Transfer pair lacks source/target in {exp_cfg_path}: {pair}")
        transfer_pairs.append((source, target))
    return {
        "seeds": seed_values,
        "seed_keys": {str(int(seed)) for seed in seed_values},
        "samples": sample_values,
        "ece_bins": ece_bins,
        "early_observation": early_cfg,
        "threshold_display": display_thresholds,
        "transfer_models": transfer_models,
        "transfer_pairs": transfer_pairs,
    }


def require_finite(value: Optional[float], context: str) -> float:
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError(f"Missing or non-finite value for {context}")
    return float(value)


def require_seed_set(rows: Sequence[Dict[str, str]], expected_seed_keys: Set[str], context: str) -> List[Dict[str, str]]:
    latest = latest_rows_by_key(rows, ["seed"])
    observed = {row.get("seed", "") for row in latest}
    if observed != expected_seed_keys:
        missing = sorted(expected_seed_keys - observed, key=int)
        unexpected = sorted(observed - expected_seed_keys)
        raise RuntimeError(f"{context} does not match formal seeds: missing={missing}, unexpected={unexpected}")
    return latest


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def compute_macro_f1_from_labels(preds: Sequence[int], targets: Sequence[int]) -> float:
    classes = sorted(set(int(v) for v in preds) | set(int(v) for v in targets))
    if len(classes) == 0:
        raise RuntimeError("Cannot compute Macro-F1 for an empty prediction set")
    f1_sum = 0.0
    for class_id in classes:
        tp = sum(1 for pred, target in zip(preds, targets) if pred == class_id and target == class_id)
        fp = sum(1 for pred, target in zip(preds, targets) if pred == class_id and target != class_id)
        fn = sum(1 for pred, target in zip(preds, targets) if pred != class_id and target == class_id)
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1_sum += 2.0 * precision * recall / float(precision + recall) if (precision + recall) > 0 else 0.0
    return f1_sum / float(len(classes))


def compute_ece_from_probabilities(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    ece_bins: int,
) -> float:
    if len(probabilities) == 0:
        raise RuntimeError("Cannot compute ECE for an empty prediction set")
    total = float(len(probabilities))
    ece = 0.0
    for bin_idx in range(int(ece_bins)):
        lower = float(bin_idx) / float(ece_bins)
        upper = float(bin_idx + 1) / float(ece_bins)
        members: List[Tuple[float, int, int]] = []
        for probs, target in zip(probabilities, targets):
            confidence = max(float(value) for value in probs)
            pred = max(range(len(probs)), key=lambda idx: float(probs[idx]))
            if confidence > lower and confidence <= upper:
                members.append((confidence, pred, int(target)))
        if len(members) == 0:
            continue
        avg_conf = sum(item[0] for item in members) / float(len(members))
        avg_acc = sum(1.0 for _, pred, target in members if pred == target) / float(len(members))
        ece += (float(len(members)) / total) * abs(avg_conf - avg_acc)
    return float(ece)


def metrics_from_prediction_rows(rows: Sequence[Dict[str, str]], ece_bins: int, context: str) -> Dict[str, float]:
    by_source: Dict[str, Dict[str, str]] = {}
    for row in rows:
        source_id = row.get("source_id", "")
        if len(source_id) == 0:
            raise RuntimeError(f"{context} has prediction rows with empty source_id")
        by_source[source_id] = row
    if len(by_source) == 0:
        raise RuntimeError(f"{context} has no prediction rows")
    targets = [int(row["target"]) for row in by_source.values()]
    preds = [int(row["pred"]) for row in by_source.values()]
    probabilities = [json.loads(row["probabilities"]) for row in by_source.values()]
    return {
        "acc": sum(1.0 for pred, target in zip(preds, targets) if pred == target) / float(len(targets)),
        "macro_f1": compute_macro_f1_from_labels(preds, targets),
        "ece": compute_ece_from_probabilities(probabilities, targets, int(ece_bins)),
    }


def collect_prediction_metric_values(
    prediction_rows: Sequence[Dict[str, str]],
    *,
    dataset: str,
    source_dataset: str,
    run_kind: str,
    model: str,
    tag: str,
    expected_seed_keys: Set[str],
    ece_bins: int,
) -> Dict[str, List[float]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        if row.get("dataset", "") != dataset:
            continue
        if row.get("source_dataset", "") != source_dataset:
            continue
        if row.get("run_kind", "") != run_kind:
            continue
        if row.get("model", "") != model:
            continue
        if row.get("tag", "") != tag:
            continue
        grouped[row.get("seed", "")].append(row)
    observed = set(grouped)
    if observed != expected_seed_keys:
        missing = sorted(expected_seed_keys - observed, key=int)
        unexpected = sorted(observed - expected_seed_keys)
        raise RuntimeError(
            f"Prediction artifact for {dataset}/{source_dataset}/{model}/{tag} does not match formal seeds: "
            f"missing={missing}, unexpected={unexpected}"
        )
    values = {"acc": [], "macro_f1": [], "ece": []}
    for seed in sorted(expected_seed_keys, key=int):
        metrics = metrics_from_prediction_rows(
            grouped[seed],
            ece_bins=ece_bins,
            context=f"{dataset}/{source_dataset}/{model}/{tag}/seed{seed}",
        )
        for metric_name in values:
            values[metric_name].append(metrics[metric_name])
    return values


def require_main_result_provenance(
    result_rows: Sequence[Dict[str, str]],
    *,
    dataset: str,
    source_dataset: str,
    run_kind: str,
    model: str,
    model_cfg: Optional[str],
    expected_seed_keys: Set[str],
) -> None:
    candidates = []
    for row in result_rows:
        if row.get("dataset", "") != dataset:
            continue
        if row.get("source_dataset", "") != source_dataset:
            continue
        if row.get("run_kind", "") != run_kind:
            continue
        if row.get("transfer_mode", "") not in {"", "main", "source_only"}:
            continue
        if row.get("model", "") != model:
            continue
        if model_cfg and row.get("model_cfg", "") != model_cfg:
            continue
        if row.get("seed", "") == "agg":
            continue
        candidates.append(row)
    seed_rows = require_seed_set(candidates, expected_seed_keys, f"Result provenance for {dataset}/{source_dataset}/{model}")
    for row in seed_rows:
        for field in ("model_cfg", "exp_cfg", "log_path"):
            if len(row.get(field, "")) == 0:
                raise RuntimeError(f"Result provenance for {dataset}/{source_dataset}/{model}/seed{row.get('seed')} lacks {field}")


def filter_seed_rows(
    rows: Iterable[Dict[str, str]],
    *,
    dataset: str,
    source_dataset: str,
    model: str,
    exp_cfg_suffix: str | None = None,
    model_cfg: str | None = None,
) -> List[Dict[str, str]]:
    filtered = []
    for row in rows:
        if row.get("dataset", "") != dataset:
            continue
        if row.get("source_dataset", "") != source_dataset:
            continue
        if row.get("run_kind", "") not in {"", "main"}:
            continue
        if row.get("transfer_mode", "") not in {"", "main"}:
            continue
        if row.get("model", "") != model:
            continue
        if model_cfg and row.get("model_cfg", "") != model_cfg:
            continue
        if row.get("seed", "") == "agg":
            continue
        if exp_cfg_suffix and not row.get("exp_cfg", "").endswith(exp_cfg_suffix):
            continue
        filtered.append(row)
    return latest_rows_by_key(filtered, ["seed"])


def build_main_table(
    results_dir: Path,
    datasets: Sequence[str],
    caption: str,
    label: str,
    expected_seed_keys: Set[str],
    ece_bins: int,
) -> str:
    rows_by_dataset = {dataset: read_rows(results_dir / f"{dataset}.csv") for dataset in datasets}
    prediction_rows = read_rows(results_dir / "predictions.csv")
    row_entries: List[Dict[str, Any]] = []
    body_lines: List[str] = []
    for model in SELECTED_MODELS:
        values: List[str] = [MODEL_LABELS[model]]
        metric_means: List[float] = []
        for dataset in datasets:
            require_main_result_provenance(
                rows_by_dataset[dataset],
                dataset=dataset,
                source_dataset=dataset,
                run_kind="main",
                model=model,
                model_cfg=MODEL_CONFIGS.get(model),
                expected_seed_keys=expected_seed_keys,
            )
            metrics = collect_prediction_metric_values(
                prediction_rows,
                dataset=dataset,
                source_dataset=dataset,
                run_kind="main",
                model=model,
                tag="main",
                expected_seed_keys=expected_seed_keys,
                ece_bins=ece_bins,
            )
            acc_mean, _ = mean_confidence(metrics["acc"])
            f1_mean, _ = mean_confidence(metrics["macro_f1"])
            values.append(format_stat(metrics["acc"]))
            values.append(format_stat(metrics["macro_f1"]))
            metric_means.extend([acc_mean, f1_mean])
        row_entries.append({"values": values, "metric_means": metric_means})
    best_values = [max(entry["metric_means"][index] for entry in row_entries) for index in range(len(row_entries[0]["metric_means"]))]
    for entry in row_entries:
        values = [entry["values"][0]]
        for index, cell in enumerate(entry["values"][1:]):
            if math.isclose(entry["metric_means"][index], best_values[index], rel_tol=1e-12, abs_tol=1e-12):
                values.append(highlight_best_cell(cell))
            else:
                values.append(cell)
        body_lines.append(" & ".join(values) + r" \\")
    header = [
        r"\begin{table*}[!h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{tabular}{l" + "cc" * len(datasets) + r"}",
        r"\toprule",
    ]
    if len(datasets) == 2:
        header.extend(
            [
                rf"& \multicolumn{{2}}{{c}}{{{DATASET_LABELS.get(datasets[0], datasets[0])}}} & \multicolumn{{2}}{{c}}{{{DATASET_LABELS.get(datasets[1], datasets[1])}}} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                r"Method & Acc & Macro-F1 & Acc & Macro-F1 \\",
                r"\midrule",
            ]
        )
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(header + body_lines + footer) + "\n"


def build_transfer_table(
    results_dir: Path,
    expected_seed_keys: Set[str],
    ece_bins: int,
    transfer_pairs: Sequence[Tuple[str, str]],
    transfer_models: Sequence[str],
) -> str:
    rows = read_rows(results_dir / "transfer.csv")
    prediction_rows = read_rows(results_dir / "predictions.csv")
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("seed", "") == "agg":
            continue
        key = (row.get("train_dataset", ""), row.get("test_dataset", ""), row.get("model", ""))
        grouped[key].append(row)
    body_lines: List[str] = []

    for pair_index, (train_dataset, test_dataset) in enumerate(transfer_pairs):
        entries: List[Dict[str, Any]] = []
        for model_name in transfer_models:
            model_rows = grouped.get((train_dataset, test_dataset, model_name), [])
            model_rows = require_seed_set(
                model_rows,
                expected_seed_keys,
                f"Transfer results for {train_dataset}->{test_dataset}/{model_name}",
            )
            model_rows_by_seed = {row.get("seed", ""): row for row in model_rows}
            source_metrics = collect_prediction_metric_values(
                prediction_rows,
                dataset=train_dataset,
                source_dataset=train_dataset,
                run_kind="main",
                model=model_name,
                tag="main",
                expected_seed_keys=expected_seed_keys,
                ece_bins=ece_bins,
            )
            ordered_seeds = sorted(expected_seed_keys, key=int)
            ordered_transfer_rows = [model_rows_by_seed[seed] for seed in ordered_seeds]
            f1_values = [
                require_finite(parse_float(row.get("macro_f1", "")), f"{train_dataset}->{test_dataset}/{model_name} macro_f1")
                for row in ordered_transfer_rows
            ]
            ece_values = [
                require_finite(parse_float(row.get("ece", "")), f"{train_dataset}->{test_dataset}/{model_name} ece")
                for row in ordered_transfer_rows
            ]
            retention_values = []
            for transfer_f1, source_f1 in zip(f1_values, source_metrics["macro_f1"]):
                if source_f1 <= 0.0 or not math.isfinite(source_f1):
                    raise RuntimeError(f"Invalid source Macro-F1 for {train_dataset}/{model_name}: {source_f1}")
                retention_values.append(transfer_f1 / source_f1)
            source_f1_mean, source_f1_ci = mean_confidence(source_metrics["macro_f1"])
            transfer_f1_mean, transfer_f1_ci = mean_confidence(f1_values)
            transfer_ece_mean, transfer_ece_ci = mean_confidence(ece_values)
            retention_mean, retention_ci = mean_confidence(retention_values)
            entries.append(
                {
                    "model": model_name,
                    "source_f1_mean": source_f1_mean,
                    "source_f1": f"{source_f1_mean:.3f}$\\pm${source_f1_ci:.3f}",
                    "transfer_f1_mean": transfer_f1_mean,
                    "transfer_f1": f"{transfer_f1_mean:.3f}$\\pm${transfer_f1_ci:.3f}",
                    "retention_mean": retention_mean,
                    "retention": f"{100.0 * retention_mean:.1f}$\\pm${100.0 * retention_ci:.1f}\\%",
                    "ece_mean": transfer_ece_mean,
                    "ece": f"{transfer_ece_mean:.3f}$\\pm${transfer_ece_ci:.3f}",
                }
            )
        best_source_f1 = max(entry["source_f1_mean"] for entry in entries)
        best_transfer_f1 = max(entry["transfer_f1_mean"] for entry in entries)
        best_retention = max(entry["retention_mean"] for entry in entries)
        best_ece = min(entry["ece_mean"] for entry in entries)
        if pair_index > 0:
            body_lines.append(r"\midrule")
        for entry in entries:
            source_f1_cell = entry["source_f1"]
            transfer_f1_cell = entry["transfer_f1"]
            retention_cell = entry["retention"]
            ece_cell = entry["ece"]
            if math.isclose(float(entry["source_f1_mean"]), best_source_f1, rel_tol=1e-12, abs_tol=1e-12):
                source_f1_cell = highlight_best_cell(source_f1_cell)
            if math.isclose(float(entry["transfer_f1_mean"]), best_transfer_f1, rel_tol=1e-12, abs_tol=1e-12):
                transfer_f1_cell = highlight_best_cell(transfer_f1_cell)
            if math.isclose(float(entry["retention_mean"]), best_retention, rel_tol=1e-12, abs_tol=1e-12):
                retention_cell = highlight_best_cell(retention_cell)
            if math.isclose(float(entry["ece_mean"]), best_ece, rel_tol=1e-12, abs_tol=1e-12):
                ece_cell = highlight_best_cell(ece_cell)
            body_lines.append(
                " & ".join(
                    [
                        f"{train_dataset}$\\rightarrow${test_dataset}",
                        MODEL_LABELS.get(entry["model"], latex_escape(entry["model"])),
                        source_f1_cell,
                        transfer_f1_cell,
                        retention_cell,
                        ece_cell,
                    ]
                )
                + r" \\"
            )
    return "\n".join(
        [
            r"\begin{table*}[!h]",
            r"\centering",
            r"\caption{Source-only cross-dataset transfer comparison against strong in-scope baselines. Source-domain Macro-F1 is shown for reference, F1 retention is computed seed-wise as transfer/source, and the best value in each metric column within a transfer pair is highlighted.}",
            r"\label{tab:paper_ready_transfer}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"Transfer pair & Method & Source F1 & Transfer F1 & F1 retained & Transfer ECE \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def build_rebuttal_tradeoff_figure(
    results_dir: Path,
    figure_path: Path,
    expected_seeds: Sequence[int],
    expected_samples: Sequence[int],
) -> None:
    revision_rows = read_rows(results_dir / "revision_artifacts.csv")
    expected_seed_keys = {str(int(seed)) for seed in expected_seeds}
    expected_sample_values = {int(samples) for samples in expected_samples}

    def is_twitter15_to_twitter16_ten_minutes(row: Dict[str, str]) -> bool:
        if row.get("run_kind", "") != "transfer":
            return False
        if row.get("source_dataset", "") != "twitter15" or row.get("dataset", "") != "twitter16":
            return False
        observation_window = row.get("observation_window", row.get("obs_minutes", ""))
        observation_unit = row.get("observation_unit", "minutes")
        return str(observation_window) in {"10", "10.0", "10.000000"} and str(observation_unit) == "minutes"

    sampling_rows = [
        row for row in revision_rows
        if row.get("artifact_type", "") == "sampling_sweep"
        and is_twitter15_to_twitter16_ten_minutes(row)
    ]
    sampling_rows = latest_rows_by_key(sampling_rows, ["seed", "samples"])
    sampling_rows = [
        row for row in sampling_rows
        if row.get("seed", "") in expected_seed_keys
        and normalize_int_cell(row.get("samples", ""), "samples") in expected_sample_values
    ]
    if len(sampling_rows) == 0:
        raise FileNotFoundError("Missing transfer sampling_sweep rows in revision_artifacts.csv")

    grouped_sampling: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in sampling_rows:
        mean_var = parse_float(row.get("mean_var", ""))
        if mean_var is None:
            continue
        grouped_sampling[normalize_int_cell(row["samples"], "samples")].append(row)
    if len(grouped_sampling) == 0:
        raise RuntimeError("Sampling sweep rows do not contain mean_var; rerun stochastic sampling-budget evaluation.")
    observed_sample_values = set(grouped_sampling)
    if observed_sample_values != expected_sample_values:
        missing_samples = sorted(expected_sample_values - observed_sample_values)
        extra_samples = sorted(observed_sample_values - expected_sample_values)
        raise RuntimeError(
            "Sampling sweep rows do not match the YAML-declared sampling budgets: "
            f"missing={missing_samples}, unexpected={extra_samples}."
        )

    def collect_metric(rows_for_sample: Sequence[Dict[str, str]], field: str) -> Tuple[float, float]:
        values = [parse_float(row.get(field, "")) for row in rows_for_sample]
        clean_values = [value for value in values if value is not None]
        if len(clean_values) == 0:
            raise RuntimeError(f"Sampling sweep rows do not contain finite {field} values.")
        return mean_confidence(clean_values)

    sampling_points = []
    for samples, rows_for_sample in sorted(grouped_sampling.items()):
        observed_seed_keys = {row.get("seed", "") for row in rows_for_sample}
        if observed_seed_keys != expected_seed_keys:
            missing_seeds = sorted(expected_seed_keys - observed_seed_keys, key=int)
            extra_seeds = sorted(observed_seed_keys - expected_seed_keys, key=int)
            raise RuntimeError(
                f"Sampling sweep for S={samples} does not match the YAML-declared formal seeds: "
                f"missing={missing_seeds}, unexpected={extra_seeds}."
            )
        latency_mean, latency_ci = collect_metric(rows_for_sample, "latency_mean_ms")
        acc_mean, acc_ci = collect_metric(rows_for_sample, "acc_resolved")
        f1_mean, f1_ci = collect_metric(rows_for_sample, "resolved_macro_f1")
        coverage_mean, coverage_ci = collect_metric(rows_for_sample, "coverage")
        var_mean, var_ci = collect_metric(rows_for_sample, "mean_var")
        sampling_points.append(
            {
                "samples": samples,
                "seed_count": len({row.get("seed", "") for row in rows_for_sample}),
                "latency_mean": latency_mean,
                "latency_ci": latency_ci,
                "acc_mean": acc_mean,
                "acc_ci": acc_ci,
                "f1_mean": f1_mean,
                "f1_ci": f1_ci,
                "coverage_mean": coverage_mean,
                "coverage_ci": coverage_ci,
                "var_mean": var_mean,
                "var_ci": var_ci,
            }
        )
    if not any(point["samples"] > 1 and point["var_mean"] > 0.0 for point in sampling_points):
        raise RuntimeError(
            "Sampling sweep rows still have zero stochastic variance for S>1; "
            "rerun with stochastic_reverse and sampled_full_graph before generating the figure."
        )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

    x_positions = list(range(len(sampling_points)))
    sample_values = [point["samples"] for point in sampling_points]
    latency_means = [point["latency_mean"] for point in sampling_points]
    latency_cis = [point["latency_ci"] for point in sampling_points]
    variance_means = [point["var_mean"] for point in sampling_points]
    variance_cis = [point["var_ci"] for point in sampling_points]
    axes[0].bar(
        x_positions,
        latency_means,
        yerr=bounded_confidence_errors(latency_means, latency_cis, lower_bound=0.0),
        capsize=3,
        width=0.54,
        color="#D8E6F3",
        edgecolor=PALETTE["blue"],
        linewidth=1.1,
        label="Latency",
    )
    axes[0].set_xlabel("Stochastic samples S")
    axes[0].set_ylabel("Mean latency per cascade (ms)")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([str(value) for value in sample_values])
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_ylim(*bounded_axis_limits(latency_means, latency_cis, lower_bound=0.0, padding_fraction=0.05))
    axes[0].set_title("Sampling Cost and Variance")
    variance_axis = axes[0].twinx()
    variance_axis.errorbar(
        x_positions,
        variance_means,
        yerr=bounded_confidence_errors(variance_means, variance_cis, lower_bound=0.0),
        marker="D",
        markersize=4,
        linestyle=":",
        color=PALETTE["muted"],
        linewidth=1.6,
        capsize=3,
        label="Mean variance",
    )
    variance_axis.set_ylabel("Mean stochastic variance")
    variance_axis.set_ylim(*bounded_axis_limits(variance_means, variance_cis, lower_bound=0.0, padding_fraction=0.10))
    variance_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    variance_axis.spines["top"].set_visible(False)
    handles_left, labels_left = axes[0].get_legend_handles_labels()
    handles_right, labels_right = variance_axis.get_legend_handles_labels()
    axes[0].legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", frameon=False)

    metric_specs = [
        ("acc_mean", "acc_ci", "s", PALETTE["rose"], 2.0, "Resolved accuracy"),
        ("f1_mean", "f1_ci", "^", PALETTE["blue"], 1.8, "Resolved Macro-F1"),
        ("coverage_mean", "coverage_ci", "o", PALETTE["green"], 1.8, "Coverage"),
    ]
    decision_means: List[float] = []
    decision_cis: List[float] = []
    for mean_field, ci_field, marker, color, linewidth, label in metric_specs:
        means = [point[mean_field] for point in sampling_points]
        cis = [point[ci_field] for point in sampling_points]
        decision_means.extend(means)
        decision_cis.extend(cis)
        axes[1].errorbar(
            x_positions,
            means,
            yerr=bounded_confidence_errors(means, cis, lower_bound=0.0, upper_bound=1.0),
            marker=marker,
            color=color,
            linewidth=linewidth,
            capsize=3,
            label=label,
        )
    metric_axis_lower, metric_axis_upper = bounded_axis_limits(
        decision_means,
        decision_cis,
        lower_bound=0.0,
        upper_bound=1.0,
        padding_fraction=0.04,
    )
    metric_axis_span = metric_axis_upper - metric_axis_lower
    for x_value, point in zip(x_positions, sampling_points):
        upper_edges = [
            bounded_confidence_interval(point[mean_field], point[ci_field], 0.0, 1.0)[1]
            for mean_field, ci_field, _, _, _, _ in metric_specs
        ]
        axes[1].text(
            x_value,
            min(metric_axis_upper - 0.02 * metric_axis_span, max(upper_edges) + 0.018),
            f"n={point['seed_count']}",
            ha="center",
            fontsize=7,
            color=PALETTE["muted"],
        )
    axes[1].set_xlabel("Stochastic samples S")
    axes[1].set_ylabel("Decision metric")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([str(value) for value in sample_values])
    axes[1].set_ylim(metric_axis_lower, metric_axis_upper)
    axes[1].set_title("Decision Trade-off")
    axes[1].grid(alpha=0.22)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def resolve_windows_for_dataset(contract: Dict[str, Any], dataset: str) -> Tuple[List[int], str]:
    early_cfg = contract.get("early_observation", {})
    if not isinstance(early_cfg, dict):
        raise ValueError("evaluation.early_observation must be declared in the experiment YAML")
    windows = early_cfg.get("default_windows", [])
    unit = str(early_cfg.get("default_unit", "minutes"))
    windows_overrides = early_cfg.get("windows_overrides", {})
    unit_overrides = early_cfg.get("unit_overrides", {})
    if isinstance(windows_overrides, dict) and dataset in windows_overrides:
        windows = windows_overrides[dataset]
    if isinstance(unit_overrides, dict) and dataset in unit_overrides:
        unit = str(unit_overrides[dataset])
    if not isinstance(windows, list) or len(windows) == 0:
        raise ValueError(f"No early-observation windows declared for {dataset}")
    return [int(window) for window in windows], unit.lower()


def build_early_table(
    results_dir: Path,
    datasets: Sequence[str],
    caption: str,
    label: str,
    expected_seed_keys: Set[str],
    ece_bins: int,
    contract: Dict[str, Any],
) -> str:
    prediction_rows = read_rows(results_dir / "predictions.csv")
    windows_by_dataset: Dict[str, List[int]] = {}
    unit_by_dataset: Dict[str, str] = {}
    for dataset in datasets:
        windows, unit = resolve_windows_for_dataset(contract, dataset)
        windows_by_dataset[dataset] = windows
        unit_by_dataset[dataset] = unit
    if len({tuple(windows_by_dataset[dataset]) for dataset in datasets}) != 1:
        raise RuntimeError(f"Early table {label} requires matching window counts across {datasets}")
    window_labels = [
        f"{window} min" if unit_by_dataset[datasets[0]] == "minutes" else f"{window} hop"
        for window in windows_by_dataset[datasets[0]]
    ]
    metric_values_by_model: Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]] = {}
    for model in SELECTED_MODELS:
        metric_values_by_model[model] = {}
        for dataset in datasets:
            metric_values_by_model[model][dataset] = {"acc": {}, "macro_f1": {}, "ece": {}}
            for window in windows_by_dataset[dataset]:
                tag = f"early_{int(window)}_{unit_by_dataset[dataset]}"
                metrics = collect_prediction_metric_values(
                    prediction_rows,
                    dataset=dataset,
                    source_dataset=dataset,
                    run_kind="main",
                    model=model,
                    tag=tag,
                    expected_seed_keys=expected_seed_keys,
                    ece_bins=ece_bins,
                )
                for metric_name in metric_values_by_model[model][dataset]:
                    metric_values_by_model[model][dataset][metric_name][window] = metrics[metric_name]

    best_values: Dict[Tuple[str, int, str], float] = {}
    for dataset in datasets:
        for window in windows_by_dataset[dataset]:
            for metric_name in ("acc", "macro_f1", "ece"):
                means = [
                    mean_confidence(metric_values_by_model[model][dataset][metric_name][window])[0]
                    for model in SELECTED_MODELS
                ]
                if metric_name == "ece":
                    best_values[(dataset, window, metric_name)] = min(means)
                else:
                    best_values[(dataset, window, metric_name)] = max(means)

    body_lines: List[str] = []
    for model in SELECTED_MODELS:
        for metric_label, metric_name in [("Acc", "acc"), ("Macro-F1", "macro_f1"), ("ECE", "ece")]:
            row_label = MODEL_LABELS[model] if metric_name == "acc" else ""
            values = [row_label, metric_label]
            for dataset in datasets:
                for window in windows_by_dataset[dataset]:
                    cell = format_stat(metric_values_by_model[model][dataset][metric_name][window])
                    mean_value, _ = mean_confidence(metric_values_by_model[model][dataset][metric_name][window])
                    if math.isclose(mean_value, best_values[(dataset, window, metric_name)], rel_tol=1e-12, abs_tol=1e-12):
                        cell = highlight_best_cell(cell)
                    values.append(cell)
            body_lines.append(" & ".join(values) + r" \\")
        body_lines.append(r"\midrule")
    if body_lines:
        body_lines.pop()
    column_spec = "ll" + "c" * sum(len(windows_by_dataset[dataset]) for dataset in datasets)
    dataset_header: List[str] = ["&"]
    cmidrules: List[str] = []
    start_col = 3
    for dataset in datasets:
        width = len(windows_by_dataset[dataset])
        dataset_header.append(rf"& \multicolumn{{{width}}}{{c}}{{{DATASET_LABELS.get(dataset, dataset)}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{start_col + width - 1}}}")
        start_col += width
    method_header = ["Method", "Metric"]
    for dataset in datasets:
        method_header.extend(window_labels)
    return "\n".join(
        [
            r"\begin{table*}[!h]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            " ".join(dataset_header) + r" \\",
            "".join(cmidrules),
            " & ".join(method_header) + r" \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def collect_revision_rows(
    revision_rows: Sequence[Dict[str, str]],
    *,
    artifact_type: str,
    dataset: str,
    source_dataset: str,
    run_kind: str,
    model: str,
    tag: Optional[str] = None,
    extra_filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    filtered = []
    for row in revision_rows:
        if row.get("artifact_type", "") != artifact_type:
            continue
        if row.get("dataset", "") != dataset:
            continue
        if row.get("source_dataset", "") != source_dataset:
            continue
        if row.get("run_kind", "") != run_kind:
            continue
        if row.get("model", "") != model:
            continue
        if tag is not None and row.get("tag", "") != tag:
            continue
        if extra_filters:
            matched = True
            for field, expected_value in extra_filters.items():
                if str(row.get(field, "")) != str(expected_value):
                    matched = False
                    break
            if not matched:
                continue
        filtered.append(row)
    return latest_rows_by_key(filtered, ["seed"])


def collect_revision_metric_values(rows: Sequence[Dict[str, str]], fields: Sequence[str], context: str) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {field: [] for field in fields}
    for row in rows:
        for field in fields:
            values[field].append(require_finite(parse_float(row.get(field, "")), f"{context}/{field}/seed{row.get('seed')}"))
    return values


def build_latency_table(results_dir: Path, expected_seed_keys: Set[str]) -> str:
    revision_rows = read_rows(results_dir / "revision_artifacts.csv")
    result_rows = read_rows(results_dir / "twitter16.csv")
    entries: List[Tuple[str, str, str, str, str, Dict[str, str]]] = [
        ("BiGCN", "latency_profile", "twitter16", "twitter16", "bigcn", {}),
        ("TGAT", "latency_profile", "twitter16", "twitter16", "tgat", {}),
        ("D$^2$", "latency_profile", "twitter16", "twitter16", "lp_gcn", {}),
        ("DGA-Fake", "latency_profile", "twitter16", "twitter16", "dga_fake", {}),
        ("RumorDiffusion ($S=1$)", "sampling_sweep", "twitter16", "twitter16", "rumordiffusion", {"tag": "early_10_minutes_samples_1", "samples": "1"}),
        ("RumorDiffusion (early-stop, $S=2$)", "early_stop", "twitter16", "twitter16", "rumordiffusion", {"tag": "early_10_minutes", "samples": "2"}),
    ]
    row_entries: List[Dict[str, Any]] = []
    for label, artifact_type, dataset, source_dataset, model, filters in entries:
        tag = filters.get("tag")
        extra_filters = {key: value for key, value in filters.items() if key != "tag"}
        rows = collect_revision_rows(
            revision_rows,
            artifact_type=artifact_type,
            dataset=dataset,
            source_dataset=source_dataset,
            run_kind="main",
            model=model,
            tag=tag,
            extra_filters=extra_filters,
        )
        rows = require_seed_set(rows, expected_seed_keys, f"Latency artifact {label}")
        require_main_result_provenance(
            result_rows,
            dataset=dataset,
            source_dataset=source_dataset,
            run_kind="main",
            model=model,
            model_cfg=MODEL_CONFIGS.get(model),
            expected_seed_keys=expected_seed_keys,
        )
        values = collect_revision_metric_values(rows, ["latency_mean_ms", "latency_p95_ms"], f"Latency artifact {label}")
        row_entries.append(
            {
                "label": label,
                "mean_ms": mean_confidence(values["latency_mean_ms"])[0],
                "p95_ms": mean_confidence(values["latency_p95_ms"])[0],
            }
        )
    best_mean_ms = min(entry["mean_ms"] for entry in row_entries)
    best_p95_ms = min(entry["p95_ms"] for entry in row_entries)
    body_lines = []
    for entry in row_entries:
        mean_cell = f"{entry['mean_ms']:.2f}"
        p95_cell = f"{entry['p95_ms']:.2f}"
        if math.isclose(float(entry["mean_ms"]), best_mean_ms, rel_tol=1e-12, abs_tol=1e-12):
            mean_cell = highlight_best_cell(mean_cell)
        if math.isclose(float(entry["p95_ms"]), best_p95_ms, rel_tol=1e-12, abs_tol=1e-12):
            p95_cell = highlight_best_cell(p95_cell)
        body_lines.append(f"{entry['label']} & {mean_cell} & {p95_cell} " + r"\\")
    return "\n".join(
        [
            r"\begin{table}[!h]",
            r"\centering",
            r"\caption{Wall-clock inference latency on the Twitter16 test set (mean and P95 milliseconds per cascade). RumorDiffusion sampling rows use the 10-minute operating point with $(\tau_{\mathrm{low}},\tau_{\mathrm{high}},\gamma)=(0.3,0.7,0.02)$.}",
            r"\label{tab:latency_profile_revision}",
            r"\small",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Method & Mean ms & P95 ms \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_sampling_budget_table(
    results_dir: Path,
    expected_seed_keys: Set[str],
    expected_samples: Sequence[int],
) -> str:
    revision_rows = read_rows(results_dir / "revision_artifacts.csv")
    row_entries: List[Dict[str, Any]] = []
    variance_values = []
    for samples in expected_samples:
        rows = collect_revision_rows(
            revision_rows,
            artifact_type="sampling_sweep",
            dataset="twitter16",
            source_dataset="twitter15",
            run_kind="transfer",
            model="rumordiffusion",
            tag=f"early_10_minutes_samples_{int(samples)}",
            extra_filters={"samples": str(int(samples))},
        )
        rows = require_seed_set(rows, expected_seed_keys, f"Sampling-budget artifact S={samples}")
        values = collect_revision_metric_values(
            rows,
            ["coverage", "acc_resolved", "resolved_macro_f1", "resolved_ece", "latency_mean_ms", "latency_p95_ms", "mean_var"],
            f"Sampling-budget artifact S={samples}",
        )
        variance_values.append((int(samples), mean_confidence(values["mean_var"])[0]))
        row_entries.append(
            {
                "samples": int(samples),
                "coverage": mean_confidence(values["coverage"])[0],
                "acc_resolved": mean_confidence(values["acc_resolved"])[0],
                "resolved_macro_f1": mean_confidence(values["resolved_macro_f1"])[0],
                "resolved_ece": mean_confidence(values["resolved_ece"])[0],
                "latency_mean_ms": mean_confidence(values["latency_mean_ms"])[0],
                "latency_p95_ms": mean_confidence(values["latency_p95_ms"])[0],
            }
        )
    if not any(samples > 1 and value > 0.0 for samples, value in variance_values):
        raise RuntimeError("Sampling-budget table lacks nonzero stochastic variance for S>1.")
    higher_is_better = ["coverage", "acc_resolved", "resolved_macro_f1"]
    lower_is_better = ["resolved_ece", "latency_mean_ms", "latency_p95_ms"]
    best_values = {field: max(entry[field] for entry in row_entries) for field in higher_is_better}
    best_values.update({field: min(entry[field] for entry in row_entries) for field in lower_is_better})
    body_lines = []
    for entry in row_entries:
        cells = [
            str(int(entry["samples"])),
            f"{entry['coverage']:.3f}",
            f"{entry['acc_resolved']:.3f}",
            f"{entry['resolved_macro_f1']:.3f}",
            f"{entry['resolved_ece']:.3f}",
            f"{entry['latency_mean_ms']:.2f}",
            f"{entry['latency_p95_ms']:.2f}",
        ]
        for index, field in enumerate(higher_is_better + lower_is_better, start=1):
            if math.isclose(float(entry[field]), best_values[field], rel_tol=1e-12, abs_tol=1e-12):
                cells[index] = highlight_best_cell(cells[index])
        body_lines.append(" & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[!h]",
            r"\centering",
            r"\caption{Sampling-budget trade-off on Twitter15$\rightarrow$Twitter16 at 10 minutes under the default threshold policy.}",
            r"\label{tab:sampling_budget_revision}",
            r"\small",
            r"\begin{tabular}{ccccccc}",
            r"\toprule",
            r"$S$ & Coverage & Resolved Acc & Resolved Macro-F1 & ECE & Mean ms & P95 ms \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_threshold_table(results_dir: Path, expected_seed_keys: Set[str], display_thresholds: Sequence[Dict[str, Any]]) -> str:
    revision_rows = read_rows(results_dir / "revision_artifacts.csv")
    row_entries: List[Dict[str, Any]] = []
    for setting in display_thresholds:
        tau_low = float(setting["tau_low"])
        tau_high = float(setting["tau_high"])
        gamma = float(setting["gamma"])
        rows = collect_revision_rows(
            revision_rows,
            artifact_type="threshold_sweep",
            dataset="twitter16",
            source_dataset="twitter15",
            run_kind="transfer",
            model="rumordiffusion",
            tag=f"early_10_minutes_tau{tau_low:.2f}_{tau_high:.2f}_gamma{gamma:.2f}",
            extra_filters={
                "tau_low": f"{tau_low:.6f}",
                "tau_high": f"{tau_high:.6f}",
                "gamma": f"{gamma:.6f}",
            },
        )
        rows = require_seed_set(rows, expected_seed_keys, f"Threshold artifact tau={tau_low},{tau_high},gamma={gamma}")
        values = collect_revision_metric_values(
            rows,
            ["coverage", "acc_resolved", "resolved_macro_f1", "resolved_ece", "latency_mean_ms"],
            f"Threshold artifact tau={tau_low},{tau_high},gamma={gamma}",
        )
        row_entries.append(
            {
                "threshold": rf"$({tau_low:.1f}, {tau_high:.1f}, {gamma:.2f})$",
                "coverage": mean_confidence(values["coverage"])[0],
                "acc_resolved": mean_confidence(values["acc_resolved"])[0],
                "resolved_macro_f1": mean_confidence(values["resolved_macro_f1"])[0],
                "resolved_ece": mean_confidence(values["resolved_ece"])[0],
                "latency_mean_ms": mean_confidence(values["latency_mean_ms"])[0],
            }
        )
    higher_is_better = ["coverage", "acc_resolved", "resolved_macro_f1"]
    lower_is_better = ["resolved_ece", "latency_mean_ms"]
    best_values = {field: max(entry[field] for entry in row_entries) for field in higher_is_better}
    best_values.update({field: min(entry[field] for entry in row_entries) for field in lower_is_better})
    body_lines = []
    for entry in row_entries:
        cells = [
            entry["threshold"],
            f"{entry['coverage']:.3f}",
            f"{entry['acc_resolved']:.3f}",
            f"{entry['resolved_macro_f1']:.3f}",
            f"{entry['resolved_ece']:.3f}",
            f"{entry['latency_mean_ms']:.2f}",
        ]
        for index, field in enumerate(higher_is_better + lower_is_better, start=1):
            if math.isclose(float(entry[field]), best_values[field], rel_tol=1e-12, abs_tol=1e-12):
                cells[index] = highlight_best_cell(cells[index])
        body_lines.append(" & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[!h]",
            r"\centering",
            r"\caption{Representative threshold settings on Twitter15$\rightarrow$Twitter16 at 10 minutes under source-only transfer.}",
            r"\label{tab:threshold_guidance_revision}",
            r"\small",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"$(\tau_{\mathrm{low}},\tau_{\mathrm{high}},\gamma)$ & Coverage & Resolved Acc & Resolved Macro-F1 & ECE & Mean ms \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def build_ablation_table(
    results_dir: Path,
    dataset: str,
    expected_seed_keys: Set[str],
    ece_bins: int,
    contract: Dict[str, Any],
) -> str:
    prediction_rows = read_rows(results_dir / "predictions.csv")
    windows, unit = resolve_windows_for_dataset(contract, dataset)
    row_entries: List[Dict[str, Any]] = []
    for model, label, calibration_state in ABLATION_ROWS:
        values = [label]
        metric_means: List[float] = []
        metric_directions: List[str] = []
        for window in windows:
            tag = f"early_{int(window)}_{unit}"
            if calibration_state == "uncalibrated":
                tag = f"{tag}_uncalibrated"
            metrics = collect_prediction_metric_values(
                prediction_rows,
                dataset=dataset,
                source_dataset=dataset,
                run_kind="main",
                model=model,
                tag=tag,
                expected_seed_keys=expected_seed_keys,
                ece_bins=ece_bins,
            )
            values.append(format_stat(metrics["macro_f1"]))
            values.append(format_stat(metrics["ece"]))
            metric_means.append(mean_confidence(metrics["macro_f1"])[0])
            metric_directions.append("max")
            metric_means.append(mean_confidence(metrics["ece"])[0])
            metric_directions.append("min")
        row_entries.append({"values": values, "metric_means": metric_means, "metric_directions": metric_directions})
    best_values = []
    for index, direction in enumerate(row_entries[0]["metric_directions"]):
        values = [entry["metric_means"][index] for entry in row_entries]
        best_values.append(max(values) if direction == "max" else min(values))
    body_lines = []
    for entry in row_entries:
        values = [entry["values"][0]]
        for index, cell in enumerate(entry["values"][1:]):
            if math.isclose(entry["metric_means"][index], best_values[index], rel_tol=1e-12, abs_tol=1e-12):
                values.append(highlight_best_cell(cell))
            else:
                values.append(cell)
        body_lines.append(" & ".join(values) + r" \\")
    header_cells = ["Variant"]
    for window in windows:
        label = f"{window} min" if unit == "minutes" else f"{window} hop"
        header_cells.extend([rf"\multicolumn{{2}}{{c}}{{{label}}}"])
    metric_cells = [""]
    for _ in windows:
        metric_cells.extend(["Macro-F1", "ECE"])
    cmidrules = []
    start_col = 2
    for _ in windows:
        cmidrules.append(rf"\cmidrule(lr){{{start_col}-{start_col + 1}}}")
        start_col += 2
    return "\n".join(
        [
            r"\begin{table*}[!h]",
            r"\centering",
            rf"\caption{{Ablation results on {DATASET_LABELS.get(dataset, dataset)} at 10/20/30 minutes (Macro-F1 and ECE, mean $\pm$ 95\% CI over 5 runs).}}",
            rf"\label{{tab:ablation_{dataset}}}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{0.95}",
            r"\begin{adjustbox}{width=\textwidth}",
            r"\begin{tabular}{l" + "cc" * len(windows) + r"}",
            r"\toprule",
            " & ".join(header_cells) + r" \\",
            "".join(cmidrules),
            " & ".join(metric_cells) + r" \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
            "",
        ]
    )


def read_source_texts(raw_root: Path) -> Dict[str, Tuple[str, str]]:
    source_texts: Dict[str, Tuple[str, str]] = {}
    for dataset in ("twitter15", "twitter16"):
        source_path = raw_root / dataset / "source_tweets.txt"
        if not source_path.is_file():
            continue
        with source_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                source_texts[parts[0]] = (dataset, parts[1])
    return source_texts


def read_raw_labels(raw_root: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for dataset in ("twitter15", "twitter16"):
        label_path = raw_root / dataset / "label.txt"
        if not label_path.is_file():
            continue
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" not in line:
                    continue
                label, source_id = line.split(":", 1)
                labels[source_id] = label
    return labels


def format_decision_trace(trace: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for item in trace:
        window = int(item.get("obs_minutes", item.get("observation_window", 0)))
        decision = str(item.get("decision", ""))
        rumor_prob = float(item.get("rumor_prob", 0.0))
        parts.append(f"{window} min: {latex_escape(decision.replace('_', ' '))} ($p_r={rumor_prob:.3f}$)")
    return "; ".join(parts)


def build_case_study_table(results_dir: Path, raw_root: Path) -> str:
    case_path = results_dir / "twitter16_selected_cases.json"
    if not case_path.is_file():
        raise FileNotFoundError(f"Missing case-study artifact: {case_path}")
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(cases, dict) or len(cases) == 0:
        raise RuntimeError(f"Case-study artifact is empty: {case_path}")
    source_texts = read_source_texts(raw_root)
    raw_labels = read_raw_labels(raw_root)
    body_lines = []
    for case_name, case in cases.items():
        source_id = str(case.get("source_id", ""))
        if source_id not in source_texts:
            raise RuntimeError(f"Case-study source id {source_id} is absent from raw source_tweets.txt")
        dataset, source_text = source_texts[source_id]
        label = raw_labels.get(source_id, str(case.get("label", "")))
        trace = case.get("trace", [])
        if not isinstance(trace, list) or len(trace) == 0:
            raise RuntimeError(f"Case-study source id {source_id} has no decision trace")
        body_lines.append(
            " & ".join(
                [
                    latex_escape(str(case_name).replace("_", " ")),
                    DATASET_LABELS.get(dataset, dataset),
                    latex_escape(source_id),
                    latex_escape(source_text),
                    latex_escape(label),
                    format_decision_trace(trace),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table*}[!h]",
            r"\centering",
            r"\caption{Qualitative decision traces selected by the explain pipeline.}",
            r"\resizebox{\textwidth}{!}{%",
            r"\label{tab:case_study_twitter16}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{p{1.7cm}p{1.2cm}p{2.4cm}p{4.5cm}p{1.3cm}p{4.2cm}}",
            r"\toprule",
            r"Case & Dataset & Source id & Source post & Gold label & Decision trace \\",
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def add_round_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str,
    edgecolor: str = PALETTE["edge"],
    fontsize: int = 10,
    weight: str = "semibold",
    radius: float = 0.06,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=PALETTE["ink"],
        linespacing=1.12,
    )
    return box


def add_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    color: str = PALETTE["muted"],
    linewidth: float = 1.7,
    rad: float = 0.0,
    linestyle: str = "-",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def draw_prefix_graph(ax: plt.Axes, x0: float, y0: float, scale: float = 1.0) -> None:
    coords = [
        (x0 + 0.00 * scale, y0 + 0.30 * scale),
        (x0 + 0.26 * scale, y0 + 0.50 * scale),
        (x0 + 0.52 * scale, y0 + 0.32 * scale),
        (x0 + 0.18 * scale, y0 + 0.02 * scale),
        (x0 + 0.45 * scale, y0 + 0.02 * scale),
    ]
    edges = [(0, 1), (1, 2), (0, 3), (0, 4), (3, 4)]
    for source, target in edges:
        ax.plot(
            [coords[source][0], coords[target][0]],
            [coords[source][1], coords[target][1]],
            color=PALETTE["muted"],
            linewidth=1.2,
            zorder=2,
        )
    for x_value, y_value in coords:
        ax.add_patch(Circle((x_value, y_value), 0.034 * scale, color=PALETTE["blue"], zorder=3))
    dashed = Circle((x0 + 0.66 * scale, y0 + 0.45 * scale), 0.045 * scale, fill=False, linestyle="--", linewidth=1.0, edgecolor=PALETTE["muted"])
    ax.add_patch(dashed)
    ax.plot([x0 + 0.52 * scale, x0 + 0.62 * scale], [y0 + 0.32 * scale, y0 + 0.42 * scale], color=PALETTE["muted"], linestyle="--", linewidth=1.0)


def build_method_overview_figure(figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.6, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sections = [
        (0.02, 0.12, 0.27, 0.78, "1. Observed Prefix", PALETTE["blue_light"], PALETTE["blue"]),
        (0.34, 0.12, 0.30, 0.78, "2. Conditional Latent\nDiffusion", PALETTE["teal_light"], PALETTE["teal"]),
        (0.68, 0.12, 0.30, 0.78, "3. Predictive Distribution & Decision", PALETTE["gold_light"], PALETTE["gold"]),
    ]
    for x_value, y_value, width, height, title, fill, accent in sections:
        ax.add_patch(
            FancyBboxPatch(
                (x_value, y_value),
                width,
                height,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                linewidth=1.4,
                edgecolor=accent,
                facecolor=fill,
            )
        )
        ax.add_patch(Rectangle((x_value, y_value), 0.012, height, color=accent, alpha=0.95))
        ax.text(x_value + 0.022, y_value + height - 0.065, title, fontsize=12, fontweight="bold", color=PALETTE["ink"], va="center", linespacing=0.95)

    graph_box = add_round_box(
        ax,
        (0.055, 0.53),
        0.20,
        0.22,
        "",
        facecolor="white",
        fontsize=10,
    )
    del graph_box
    draw_prefix_graph(ax, 0.082, 0.590, scale=0.14)
    ax.text(
        0.172,
        0.640,
        "Prefix-truncated\ncascade graph",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="semibold",
        color=PALETTE["ink"],
        linespacing=1.12,
    )
    add_round_box(ax, (0.055, 0.36), 0.20, 0.10, "Fixed text\nembedding", facecolor="white", fontsize=10)
    add_round_box(ax, (0.055, 0.23), 0.20, 0.10, "Temporal /\nstructural features", facecolor="white", fontsize=10)

    _encoder = add_round_box(ax, (0.355, 0.42), 0.095, 0.20, "Prefix\nencoder", facecolor="white", edgecolor=PALETTE["teal"], fontsize=10)
    _condition = add_round_box(ax, (0.482, 0.42), 0.100, 0.20, "Condition\nvector $c^{(k)}$", facecolor="white", edgecolor=PALETTE["teal"], fontsize=10)
    add_round_box(ax, (0.390, 0.69), 0.16, 0.10, "Gaussian noise\n$z_T \\sim \\mathcal{N}(0, I)$", facecolor="white", edgecolor=PALETTE["teal"], fontsize=9)
    _denoise = add_round_box(ax, (0.390, 0.22), 0.17, 0.11, "Reverse denoising\nchain", facecolor="white", edgecolor=PALETTE["teal"], fontsize=10)
    samples = add_round_box(ax, (0.592, 0.22), 0.032, 0.53, "$S$\nlatent\nsamples", facecolor="#D4EFEA", edgecolor=PALETTE["teal"], fontsize=8)
    del samples
    for y_value in [0.31, 0.43, 0.55, 0.67]:
        ax.add_patch(Rectangle((0.599, y_value), 0.017, 0.055, color=PALETTE["teal"], alpha=0.65))

    _dist = add_round_box(ax, (0.705, 0.53), 0.22, 0.20, "Aggregate class\nprobabilities", facecolor="white", edgecolor=PALETTE["gold"], fontsize=10)
    ax.plot([0.735, 0.775, 0.815, 0.855, 0.895], [0.575, 0.66, 0.61, 0.69, 0.59], color=PALETTE["blue"], linewidth=2.0)
    ax.fill_between([0.735, 0.775, 0.815, 0.855, 0.895], [0.555, 0.62, 0.58, 0.63, 0.56], [0.595, 0.70, 0.64, 0.73, 0.62], color=PALETTE["blue"], alpha=0.13)
    add_round_box(ax, (0.705, 0.36), 0.22, 0.10, "Estimate uncertainty\nfrom sample variance", facecolor="white", edgecolor=PALETTE["gold"], fontsize=10)
    add_round_box(ax, (0.715, 0.21), 0.09, 0.10, "Predict", facecolor=PALETTE["green_light"], edgecolor=PALETTE["green"], fontsize=10)
    add_round_box(ax, (0.835, 0.21), 0.09, 0.10, "Defer", facecolor=PALETTE["rose_light"], edgecolor=PALETTE["rose"], fontsize=10)

    for start_y in [0.64, 0.41, 0.28]:
        add_arrow(ax, (0.255, start_y), (0.355, 0.52), color=PALETTE["blue"], rad=0.03)
    add_arrow(ax, (0.450, 0.52), (0.482, 0.52), color=PALETTE["teal"])
    add_arrow(ax, (0.532, 0.42), (0.430, 0.33), color=PALETTE["teal"], rad=0.10)
    add_arrow(ax, (0.470, 0.70), (0.470, 0.62), color=PALETTE["teal"])
    add_arrow(ax, (0.560, 0.275), (0.592, 0.48), color=PALETTE["teal"], rad=-0.12)
    add_arrow(ax, (0.624, 0.49), (0.705, 0.63), color=PALETTE["gold"], rad=0.05)
    add_arrow(ax, (0.815, 0.53), (0.815, 0.46), color=PALETTE["gold"])
    add_arrow(ax, (0.775, 0.36), (0.760, 0.31), color=PALETTE["green"], rad=0.0)
    add_arrow(ax, (0.865, 0.36), (0.880, 0.31), color=PALETTE["rose"], rad=0.0)
    ax.text(0.815, 0.485, "confidence + uncertainty rule", fontsize=9, color=PALETTE["muted"], ha="center")

    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def _json_pointer(payload: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError(f"table registry metric pointer must start with '/': {pointer}")
    value = payload
    for part in pointer.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise ValueError(f"table registry pointer cannot traverse {pointer}")
    return value


def validate_canonical_table_registry(canonical_root: Path, registry_path: Path) -> Dict[str, Any]:
    """Validate registry cells against validated metric artifacts, without writing paper assets."""
    validate_canonical_root(canonical_root)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1" or not isinstance(payload.get("cells"), dict):
        raise ValueError("table_cell_registry.json must be schema version 1 with a cells object")
    runs_root = canonical_root / "runs" if (canonical_root / "runs").is_dir() else canonical_root
    approved_metrics: Dict[Path, str] = {}
    for manifest_path in runs_root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["metrics"]
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        approved_metrics[path.resolve()] = str(record["sha256"])
    verified: Dict[str, Any] = {}
    for cell_key, cell in payload["cells"].items():
        if not isinstance(cell, dict):
            raise ValueError(f"table cell {cell_key} must be an object")
        for field in ("artifact_path", "artifact_sha256", "metric_pointer", "value"):
            if field not in cell:
                raise ValueError(f"table cell {cell_key} is missing {field}")
        artifact = Path(str(cell["artifact_path"]))
        if not artifact.is_absolute():
            artifact = registry_path.parent / artifact
        artifact = artifact.resolve()
        if artifact not in approved_metrics:
            raise ValueError(f"table cell {cell_key} points outside validated canonical metrics: {artifact}")
        observed_hash = file_sha256(artifact)
        if observed_hash != approved_metrics[artifact] or observed_hash != str(cell["artifact_sha256"]):
            raise ValueError(f"table cell {cell_key} metric artifact hash mismatch")
        metrics = json.loads(artifact.read_text(encoding="utf-8"))
        validate_metrics_artifact(metrics)
        pointed_value = _json_pointer(metrics, str(cell["metric_pointer"]))
        reject_placeholders(pointed_value, f"table cell {cell_key}")
        if pointed_value != cell["value"]:
            raise ValueError(f"table cell {cell_key} value differs from metric artifact")
        verified[cell_key] = {"artifact": str(artifact), "metric_pointer": cell["metric_pointer"], "value": pointed_value}
    return {"schema_version": "1", "canonical_root": str(canonical_root), "registry_sha256": file_sha256(registry_path), "verified_cells": verified}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical table evidence. This command never writes manuscript assets.")
    parser.add_argument("--canonical-root", type=Path, help="Output/canonical/<protocol_id> root")
    parser.add_argument("--canonical-registry", type=Path, help="Validated table_cell_registry.json")
    parser.add_argument("--validation-report", type=Path, help="Optional machine-readable validation report outside Docs/paper")
    parser.add_argument("--exp", default="Libs/configs/exp_paper_ready.yaml", help="Formal experiment config that declares the seed list")
    parser.add_argument("--sampling-model-config", default=MODEL_CONFIGS["rumordiffusion"], help="RumorDiffusion config that declares the sampling-budget sweep")
    parser.add_argument("--results-dir", default="Output/results/paper_aligned", help="Canonical result directory")
    parser.add_argument("--tables-dir", default="Docs/paper/Generated", help="Directory for generated LaTeX tables")
    parser.add_argument("--figures-dir", default="Docs/paper/Figs/generated", help="Directory for generated figures")
    parser.add_argument("--overview-figure", default="Docs/paper/Figs/overview.pdf", help="Path for the method overview figure")
    parser.add_argument("--rebuttal-tradeoff-copy", default="Docs/rebuttal/responses/round1/Figs/deployment_tradeoff.pdf", help="Reader-facing response copy of the deployment trade-off figure")
    parser.add_argument("--transfer-table-only", action="store_true", help="Only regenerate the source-only transfer table")
    args = parser.parse_args()

    if args.canonical_root is not None or args.canonical_registry is not None:
        if args.canonical_root is None or args.canonical_registry is None:
            parser.error("--canonical-root and --canonical-registry must be supplied together")
        report = validate_canonical_table_registry(args.canonical_root, args.canonical_registry)
        if args.validation_report:
            if "Docs/paper" in args.validation_report.as_posix():
                parser.error("canonical validation reports must not be written under Docs/paper")
            write_text(args.validation_report, json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(f"canonical table evidence passed: {args.canonical_registry}")
        return

    parser.error("legacy CSV/paper-figure generation is disabled; use --canonical-root and --canonical-registry")

    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")
    contract = resolve_generation_contract(
        Path(args.exp),
        Path(args.sampling_model_config),
    )
    expected_seeds = contract["seeds"]
    expected_seed_keys = contract["seed_keys"]
    expected_samples = contract["samples"]
    ece_bins = int(contract["ece_bins"])

    if bool(args.transfer_table_only):
        transfer_table = build_transfer_table(
            results_dir,
            expected_seed_keys,
            ece_bins,
            contract["transfer_pairs"],
            contract["transfer_models"],
        )
        write_text(tables_dir / "transfer_results.tex", transfer_table)
        return

    twitter_table = build_main_table(
        results_dir=results_dir,
        datasets=["twitter15", "twitter16"],
        caption="Main results for Twitter15/16 under the observed-prefix protocol (mean $\\pm$ 95\\% CI over 5 runs).",
        label="tab:twitter_results",
        expected_seed_keys=expected_seed_keys,
        ece_bins=ece_bins,
    )
    binary_table = build_main_table(
        results_dir=results_dir,
        datasets=["weibo", "pheme"],
        caption="Main results for Weibo/PHEME under the observed-prefix protocol (mean $\\pm$ 95\\% CI over 5 runs).",
        label="tab:weibo_pheme_results",
        expected_seed_keys=expected_seed_keys,
        ece_bins=ece_bins,
    )
    transfer_table = build_transfer_table(
        results_dir,
        expected_seed_keys,
        ece_bins,
        contract["transfer_pairs"],
        contract["transfer_models"],
    )
    early_twitter_table = build_early_table(
        results_dir=results_dir,
        datasets=["twitter15", "twitter16"],
        caption="Early detection on Twitter15 and Twitter16 (Acc / Macro-F1 / ECE, mean $\\pm$ 95\\% CI over 5 runs) at 10, 20, and 30 minutes.",
        label="tab:early_results_twitter",
        expected_seed_keys=expected_seed_keys,
        ece_bins=ece_bins,
        contract=contract,
    )
    early_binary_table = build_early_table(
        results_dir=results_dir,
        datasets=["weibo", "pheme"],
        caption="Early detection on Weibo and PHEME (Acc / Macro-F1 / ECE, mean $\\pm$ 95\\% CI over 5 runs) at hop-based windows 1, 2, and 3.",
        label="tab:early_results_weibo_pheme",
        expected_seed_keys=expected_seed_keys,
        ece_bins=ece_bins,
        contract=contract,
    )
    latency_table = build_latency_table(results_dir, expected_seed_keys)
    sampling_budget_table = build_sampling_budget_table(results_dir, expected_seed_keys, expected_samples)
    threshold_table = build_threshold_table(results_dir, expected_seed_keys, contract["threshold_display"])
    case_study_table = build_case_study_table(results_dir, Path("Input/raw"))
    ablation_twitter15 = build_ablation_table(
        results_dir=results_dir,
        dataset="twitter15",
        expected_seed_keys=expected_seed_keys,
        ece_bins=ece_bins,
        contract=contract,
    )
    ablation_twitter16 = build_ablation_table(
        results_dir=results_dir,
        dataset="twitter16",
        expected_seed_keys=expected_seed_keys,
        ece_bins=ece_bins,
        contract=contract,
    )

    write_text(tables_dir / "main_results_twitter.tex", twitter_table)
    write_text(tables_dir / "main_results_binary.tex", binary_table)
    write_text(tables_dir / "transfer_results.tex", transfer_table)
    write_text(tables_dir / "early_results_twitter.tex", early_twitter_table)
    write_text(tables_dir / "early_results_binary.tex", early_binary_table)
    write_text(tables_dir / "latency_profile.tex", latency_table)
    write_text(tables_dir / "sampling_budget.tex", sampling_budget_table)
    write_text(tables_dir / "threshold_guidance.tex", threshold_table)
    write_text(tables_dir / "case_study.tex", case_study_table)
    write_text(tables_dir / "ablation_twitter15.tex", ablation_twitter15)
    write_text(tables_dir / "ablation_twitter16.tex", ablation_twitter16)
    tradeoff_path = figures_dir / "rebuttal_tradeoff.pdf"
    build_method_overview_figure(Path(args.overview_figure))
    build_rebuttal_tradeoff_figure(results_dir, tradeoff_path, expected_seeds, expected_samples)
    rebuttal_tradeoff_copy = Path(args.rebuttal_tradeoff_copy)
    rebuttal_tradeoff_copy.parent.mkdir(parents=True, exist_ok=True)
    rebuttal_tradeoff_copy.write_bytes(tradeoff_path.read_bytes())


if __name__ == "__main__":
    main()
