#!/bin/bash
set -euo pipefail

# Aggregate mean±std for sweeps, supporting both rd_sweeps.csv 和各 dataset 结果表。
# 可选环境变量：
#   SEEDS_OVERRIDE        (默认 "42,3047,126")
#   DATASETS_OVERRIDE     (默认 "pheme,weibo,twitter15,twitter16")
#   MODELS_OVERRIDE       (默认 "rumordiffusion")
#   CSV_PATH              (默认 "Output/results/rd_sweeps.csv"，向后兼容)
#   INPUT_FILES           (逗号分隔，默认使用 dataset 结果文件与 CSV_PATH 中存在的文件)
#   SUMMARY_OUT           (默认 "Output/results/rd_param_sweeps_summary.csv")

set -e

SEEDS="${SEEDS_OVERRIDE:-42,3047,126}"
DATASETS="${DATASETS_OVERRIDE:-pheme,weibo,twitter15,twitter16}"
MODELS="${MODELS_OVERRIDE:-rumordiffusion}"
CSV_PATH="${CSV_PATH:-Output/results/rd_sweeps.csv}"
SUMMARY_OUT="${SUMMARY_OUT:-Output/results/rd_param_sweeps_summary.csv}"
DEFAULT_INPUTS="Output/results/twitter15.csv,Output/results/twitter16.csv,Output/results/weibo.csv,Output/results/pheme.csv,${CSV_PATH}"
INPUT_FILES="${INPUT_FILES:-$DEFAULT_INPUTS}"

python3 - <<'PY'
import csv
import math
import os
import re
from datetime import datetime

input_env = os.environ.get("INPUT_FILES", "")
default_inputs = os.environ.get("DEFAULT_INPUTS", "")
paths_env = input_env if input_env.strip() else default_inputs
all_paths = [p.strip() for p in paths_env.split(",") if p.strip()]
existing_paths = [p for p in all_paths if os.path.isfile(p)]
if len(existing_paths) == 0:
    raise SystemExit("No input files found for summarization.")

summary_out = os.environ.get("SUMMARY_OUT", "Output/results/rd_param_sweeps_summary.csv")
seeds_env = os.environ.get("SEEDS_OVERRIDE", "42,3047,126")
datasets_env = os.environ.get("DATASETS_OVERRIDE", "pheme,weibo,twitter15,twitter16")
models_env = os.environ.get("MODELS_OVERRIDE", "rumordiffusion")

seeds = [s.strip() for s in seeds_env.split(",") if s.strip()]
datasets = [d.strip() for d in datasets_env.split(",") if d.strip()]
models = [m.strip() for m in models_env.split(",") if m.strip()]
seeds_tag = "_".join(seeds) + "_meanstd"

numeric_cols = [
    "best_val_macro_f1",
    "test_loss",
    "test_nll",
    "test_acc",
    "test_macro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "test_weighted_f1",
    "test_weighted_precision",
    "test_weighted_recall",
    "ece",
    "calibration_temp",
]

def parse_float(text: str):
    s = str(text).strip()
    if s == "":
        return None
    if not re.match(r"^-?\d+(\.\d+)?(e-?\d+)?$", s, re.IGNORECASE):
        return None
    return float(s)

def mean_std(values):
    if len(values) == 0:
        return ""
    mean_val = sum(values) / float(len(values))
    var = sum((v - mean_val) ** 2 for v in values) / float(len(values))
    std_val = math.sqrt(var)
    return f"{mean_val:.6f}±{std_val:.6f}"

def collect_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        hdrs = reader.fieldnames or []
        data = list(reader)
    return hdrs, data

headers_union = []
data_all = []
for p in existing_paths:
    hdrs, data = collect_rows(p)
    if len(headers_union) == 0:
        headers_union = hdrs
    data_all.extend(data)

groups = {}
for r in data_all:
    ds = r.get("dataset", "").strip()
    md = r.get("model", "").strip()
    if ds not in datasets or (len(models) > 0 and md not in models):
        continue
    if seeds and r.get("seed", "").strip() not in seeds:
        continue
    key = (ds, r.get("model_cfg", "").strip(), md)
    if key not in groups:
        groups[key] = []
    groups[key].append(r)

rows_out = []
now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for (ds, model_cfg, md) in groups:
    filtered = groups[(ds, model_cfg, md)]
    agg = {}
    for col in numeric_cols:
        vals = [parse_float(r.get(col, "")) for r in filtered]
        vals = [v for v in vals if v is not None]
        agg[col] = mean_std(vals)
    row = {
        "timestamp": now_ts,
        "dataset": ds,
        "model": md,
        "model_cfg": model_cfg,
        "exp_cfg": filtered[0].get("exp_cfg", ""),
        "ckpt_path": "aggregate",
        "log_path": "aggregate_meanstd",
        "seed": seeds_tag,
        "best_val_macro_f1": agg.get("best_val_macro_f1", ""),
        "test_loss": agg.get("test_loss", ""),
        "test_nll": agg.get("test_nll", ""),
        "test_acc": agg.get("test_acc", ""),
        "test_macro_f1": agg.get("test_macro_f1", ""),
        "test_macro_precision": agg.get("test_macro_precision", ""),
        "test_macro_recall": agg.get("test_macro_recall", ""),
        "test_weighted_f1": agg.get("test_weighted_f1", ""),
        "test_weighted_precision": agg.get("test_weighted_precision", ""),
        "test_weighted_recall": agg.get("test_weighted_recall", ""),
        "ece": agg.get("ece", ""),
        "calibration_temp": agg.get("calibration_temp", ""),
    }
    rows_out.append(row)

if len(rows_out) > 0:
    out_dir = os.path.dirname(summary_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(summary_out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers_union, extrasaction="ignore")
        if f.tell() == 0 and headers_union:
            writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
PY
