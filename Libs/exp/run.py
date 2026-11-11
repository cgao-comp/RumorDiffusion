import argparse
import collections
import copy
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from Libs.data.processor import (build_text_embedding_cache,
                                 build_text_embedding_cache_from_mapping,
                                 collect_json_source_texts,
                                 resolve_text_model_path)
from Libs.data.dataset import GraphDataset
from Libs.data.potential_cache import build_potential_cache_metadata, potential_cache_dir
from Libs.exp.canonical_artifacts import canonical_run_id, prediction_rows_from_logits, write_canonical_run
from Libs.exp.trainer import (CGVAEConfig, CGVAETrainer, LinkPredConfig,
                              LinkPredTrainer, Trainer, TrainerConfig,
                              build_checkpoint_payload,
                              unpack_checkpoint_payload)
from Libs.models.baselines.AGAD import AGAD
from Libs.models.baselines.APPNPNet import APPNPNet
from Libs.models.baselines.BiGCN import BiGCN
from Libs.models.baselines.CTDM_syn import CTDMSyn
from Libs.models.baselines.DGAFake import DGAFakeConfig, DGAFakeModel
from Libs.models.baselines.GAT import GAT
from Libs.models.baselines.GATv2 import GATv2
from Libs.models.baselines.GCN import GCN
from Libs.models.baselines.GCN2Net import GCN2Net
from Libs.models.baselines.GraphSAGE import GraphSAGE
from Libs.models.baselines.HPCLCascade import HPCLCascade
from Libs.models.baselines.ResGCN import ResGCN
from Libs.models.baselines.RootPooling import RootPooling
from Libs.models.baselines.RumorDiffusion import RumorDiffusion
from Libs.models.baselines.TPSO import TPSO
from Libs.models.baselines.TemporalGCN import TemporalGCN
from Libs.models.baselines.TextOnly import TextOnlyClassifier
from Libs.models.baselines.TGAT import TGAT
from Libs.models.baselines.TGN import TGN
from Libs.models.modules.rumor_diffusion import _extract_source_ids_from_batch, generate_potentials
from Libs.utils.metrics import compute_accuracy, compute_ece, compute_macro_f1, metrics_from_logits
from Libs.utils.seed import set_global_seed
from Libs.utils.artifact_schema import (
    canonical_json_hash as artifact_canonical_json_hash,
    file_sha256,
    omit_optional_none_fields,
)

# 屏蔽警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def read_yaml_like(path: str) -> Dict[str, Any]:
    def expand_env(value: Any) -> Any:
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            if "${" in expanded:
                raise ValueError(f"Unresolved environment variable in config {path}: {value}")
            return expanded
        if isinstance(value, list):
            return [expand_env(item) for item in value]
        if isinstance(value, dict):
            return {key: expand_env(item) for key, item in value.items()}
        return value

    with open(path, "r", encoding="utf-8") as f:
        payload = expand_env(yaml.safe_load(f))
    if not isinstance(payload, dict):
        raise TypeError(f"YAML config must be a mapping: {path}")
    base_config = payload.pop("base_config", None)
    if base_config is None:
        return payload
    if not isinstance(base_config, str) or len(base_config.strip()) == 0:
        raise TypeError(f"base_config must be a non-empty relative path in {path}")
    base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_config))
    if not os.path.isfile(base_path):
        raise FileNotFoundError(f"base_config does not exist: {base_path}")
    return _deep_merge(read_yaml_like(base_path), payload)


def sha256_text(text: str) -> str:
    """Returns the SHA256 digest for a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(payload: Dict[str, Any]) -> str:
    """Hashes a JSON-serializable mapping with stable key ordering."""
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def compute_split_signature(dataset_root: str) -> str:
    """Computes a stable signature for the split definition of a processed dataset."""
    splits_path = os.path.join(dataset_root, "splits.json")
    if not os.path.isfile(splits_path):
        raise FileNotFoundError(f"Missing split definition: {splits_path}")
    with open(splits_path, "r", encoding="utf-8") as f:
        return sha256_text(f.read())


def resolve_dataset_label_mapping(dataset_name: str) -> Dict[str, int]:
    """Returns the label mapping configured for a dataset in `dataset.yaml`."""
    dataset_cfg = read_yaml_like(os.path.join("Libs", "configs", "dataset.yaml"))
    label_cfg = dataset_cfg.get("dataset", {}).get("labels", {})
    overrides = label_cfg.get("overrides", {}) if isinstance(label_cfg.get("overrides", {}), dict) else {}
    mapping = overrides.get(dataset_name, {}).get("map", label_cfg.get("map", {}))
    if not isinstance(mapping, dict):
        raise TypeError(f"Invalid label mapping for dataset={dataset_name}")
    return {str(key): int(value) for key, value in mapping.items()}


def build_checkpoint_metadata(
    model_name: str,
    stage: str,
    dataset_name: str,
    dataset_root: str,
    seed: int,
    label_mapping: Dict[str, int],
    config_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds a strict metadata contract for serialized checkpoints."""
    resolved_config_payload = omit_optional_none_fields(config_payload)
    return {
        "format_version": 1,
        "model_name": str(model_name),
        "stage": str(stage),
        "dataset_name": str(dataset_name),
        "seed": int(seed),
        "split_signature": compute_split_signature(dataset_root),
        "label_mapping": {str(key): int(value) for key, value in label_mapping.items()},
        "config_hash": canonical_json_hash(resolved_config_payload),
        "config_payload": copy.deepcopy(resolved_config_payload),
    }


def validate_checkpoint_metadata(
    ckpt_path: str,
    metadata: Dict[str, Any],
    expected_metadata: Dict[str, Any],
    ignored_keys: Optional[Sequence[str]] = None,
) -> None:
    """Validates a loaded checkpoint against the expected dataset/model contract."""
    ignored = {str(key) for key in (ignored_keys or [])}
    optional_keys = {"config_payload"}
    missing_keys = [
        key for key in expected_metadata.keys()
        if key not in ignored and key not in optional_keys and key not in metadata
    ]
    if missing_keys:
        raise ValueError(
            f"Checkpoint {ckpt_path} is missing metadata keys: {', '.join(sorted(missing_keys))}"
        )
    mismatches: List[str] = []
    for key, expected_value in expected_metadata.items():
        if key in ignored:
            continue
        observed_value = metadata.get(key)
        if key in optional_keys and observed_value is None:
            continue
        if observed_value != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value!r}, observed={observed_value!r}"
            )
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"Checkpoint metadata mismatch for {ckpt_path}: {mismatch_text}")


def load_checkpoint_state(
    ckpt_path: str,
    device: str,
    expected_metadata: Optional[Dict[str, Any]] = None,
    validate_metadata: bool = False,
    allow_legacy: bool = True,
    ignored_metadata_keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Loads a checkpoint payload and optionally enforces metadata validation."""
    payload = torch.load(ckpt_path, map_location=device)
    state_dict, metadata_raw = unpack_checkpoint_payload(payload)
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    if validate_metadata:
        if len(metadata) == 0:
            if allow_legacy:
                return state_dict, metadata
            raise ValueError(
                f"Checkpoint {ckpt_path} has no metadata payload; "
                "recreate or migrate the checkpoint before evaluation."
            )
        if expected_metadata is None:
            raise ValueError("Expected checkpoint metadata must be provided when validation is enabled.")
        validate_checkpoint_metadata(
            ckpt_path,
            metadata,
            expected_metadata,
            ignored_keys=ignored_metadata_keys,
        )
    return state_dict, metadata


def rumordiffusion_metadata_ignored_keys(
    require_config_hash: bool,
    transfer_run: bool,
) -> Optional[List[str]]:
    """Returns metadata keys that should not block RumorDiffusion checkpoint loading."""
    ignored: List[str] = []
    if (not bool(require_config_hash)) or bool(transfer_run):
        ignored.extend(["config_hash", "config_payload"])
    if len(ignored) == 0:
        return None
    return list(dict.fromkeys(ignored))


def resolve_max_nodes_setting(scanned_max_nodes: int, configured_max_nodes: int, context: str) -> int:
    """Resolves `max_nodes` without silent truncation."""
    if int(scanned_max_nodes) <= 0:
        raise ValueError(f"{context}: scanned max_nodes must be positive")
    if int(configured_max_nodes) <= 0:
        return int(scanned_max_nodes)
    if int(scanned_max_nodes) > int(configured_max_nodes):
        raise ValueError(
            f"{context}: scanned max_nodes={int(scanned_max_nodes)} exceeds configured max_nodes={int(configured_max_nodes)}. "
            "Use max_nodes: 0 to resolve from active dataset splits or raise the configured budget."
        )
    return int(configured_max_nodes)


def checkpoint_relevant_rumordiffusion_cfg(rd_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Drops evaluation-only fields before hashing RumorDiffusion checkpoints."""
    filtered = copy.deepcopy(rd_cfg)
    for key in ("checkpoint", "evaluation", "explain"):
        if key in filtered:
            filtered.pop(key)
    text_cfg = filtered.get("text", {})
    if isinstance(text_cfg, dict) and "robust_text" in text_cfg:
        text_cfg = dict(text_cfg)
        text_cfg.pop("robust_text")
        filtered["text"] = text_cfg
    return filtered


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two mappings with override taking precedence.

    - Only merges when both sides are dict; other types are replaced by override.
    - Leaves input objects unmodified by constructing a new dict.
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return dict(base)

    merged: Dict[str, Any] = {}
    # Keys from both dicts
    for k in set(base.keys()) | set(override.keys()):
        if k in override and isinstance(override[k], dict) and isinstance(base.get(k), dict):
            merged[k] = _deep_merge(base[k], override[k])
        elif k in override:
            merged[k] = override[k]
        else:
            merged[k] = base[k]
    return merged


def apply_dataset_overrides(cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Applies dataset-specific overrides stored under ``dataset_overrides``."""
    overrides = cfg.get("dataset_overrides", {}) if isinstance(cfg.get("dataset_overrides", {}), dict) else {}
    dataset_override = overrides.get(dataset_name, {}) if isinstance(overrides.get(dataset_name, {}), dict) else {}
    if len(dataset_override) == 0:
        return cfg
    merged = _deep_merge(cfg, dataset_override)
    merged["dataset_overrides"] = overrides
    return merged


def validate_class_pair(pair_ids: Sequence[int], num_classes: int, context: str, dataset_name: str) -> List[int]:
    """Validates that a configured class pair exists for the target dataset."""
    resolved = [int(v) for v in pair_ids]
    if len(resolved) != 2:
        raise ValueError(
            f"{context} requires exactly two class ids for dataset '{dataset_name}', got {resolved}."
        )
    if min(resolved) < 0 or max(resolved) >= int(num_classes):
        raise ValueError(
            f"{context} pair_ids {resolved} invalid for dataset '{dataset_name}' with {int(num_classes)} classes."
        )
    return resolved


def validate_rumordiffusion_pair_config(rd_cfg: Dict[str, Any], dataset_name: str, num_classes: int) -> Dict[str, Any]:
    """Validates and normalizes RumorDiffusion class-pair settings."""
    normalized = _deep_merge({}, rd_cfg)
    ctdm_cfg = normalized.get("ctdm", {}) if isinstance(normalized.get("ctdm", {}), dict) else {}
    reg_cfg = ctdm_cfg.get("regularizers", {}) if isinstance(ctdm_cfg.get("regularizers", {}), dict) else {}
    if "pair_ids" in reg_cfg:
        reg_cfg["pair_ids"] = validate_class_pair(
            pair_ids=reg_cfg["pair_ids"],
            num_classes=num_classes,
            context="RumorDiffusion CTDM regularizers",
            dataset_name=dataset_name,
        )
        ctdm_cfg["regularizers"] = reg_cfg
        normalized["ctdm"] = ctdm_cfg
    arb_cfg = normalized.get("arbiter", {}) if isinstance(normalized.get("arbiter", {}), dict) else {}
    if "pair_ids" in arb_cfg:
        arb_cfg["pair_ids"] = validate_class_pair(
            pair_ids=arb_cfg["pair_ids"],
            num_classes=num_classes,
            context="RumorDiffusion arbiter",
            dataset_name=dataset_name,
        )
        normalized["arbiter"] = arb_cfg
    return normalized


def resolve_rumordiffusion_potentials_pair(
    configured_pair: Sequence[int],
    num_classes: int,
    dataset_name: str,
) -> List[int]:
    """Normalizes cached-potential token selection to the dataset label space."""
    if int(num_classes) <= 0:
        raise ValueError(f"RumorDiffusion requires a positive num_classes for dataset '{dataset_name}'.")
    normalized: List[int] = []
    for raw_value in configured_pair:
        value = int(raw_value)
        if 0 <= value < int(num_classes) and value not in normalized:
            normalized.append(value)
    if len(normalized) == int(num_classes):
        return normalized
    if len(normalized) == 0:
        return list(range(int(num_classes)))
    raise ValueError(
        f"RumorDiffusion cached potentials for dataset '{dataset_name}' require one unique class id per label. "
        f"Configured={list(configured_pair)}, resolved_valid={normalized}, num_classes={int(num_classes)}."
    )


def collect_logits(model: torch.nn.Module, loader: DataLoader, device: str, amp: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and targets from a loader without altering model state."""
    model.eval()
    logits_list = []
    target_list = []
    use_amp = amp and str(device).startswith("cuda")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(batch)
            else:
                out = model(batch)
            logits_list.append(out.detach().to(torch.float32).cpu())
            target_list.append(batch.y.view(-1).cpu())
    if len(logits_list) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(logits_list, dim=0), torch.cat(target_list, dim=0)


def calibration_preserves_non_tied_argmax(
    original_logits: torch.Tensor,
    scaled_logits: torch.Tensor,
    atol: float = 1e-6,
) -> bool:
    """Checks whether temperature scaling only changes predictions on tied logits."""
    original_logits = original_logits.to(torch.float32)
    scaled_logits = scaled_logits.to(torch.float32)
    original_pred = torch.argmax(original_logits, dim=1)
    scaled_pred = torch.argmax(scaled_logits, dim=1)
    changed_mask = original_pred != scaled_pred
    if not bool(changed_mask.any()):
        return True
    topk_vals = torch.topk(original_logits, k=min(2, int(original_logits.size(1))), dim=1).values
    if int(topk_vals.size(1)) < 2:
        return True
    top_margin = torch.abs(topk_vals[:, 0] - topk_vals[:, 1])
    return bool(torch.all(top_margin[changed_mask] <= float(atol)).item())


def tune_temperature(logits: torch.Tensor, targets: torch.Tensor, init_temp: float = 1.0, max_iter: int = 100, lr: float = 0.01) -> float:
    """Optimize a single temperature parameter on validation logits."""
    if logits.numel() == 0:
        return float(init_temp)
    device = logits.device
    temp = torch.tensor([init_temp], device=device,
                        requires_grad=True, dtype=logits.dtype)
    optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits / temp.clamp_min(1e-3), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temp.detach().clamp_min(1e-3).item())


def append_results_csv(csv_path: str, headers: list[str], row: list[str]) -> None:
    """Appends one row to a CSV, creating header if missing."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(headers)
        w.writerow(row)


def append_rows_csv(csv_path: str, headers: list[str], rows: list[list[str]]) -> None:
    """Appends multiple rows to a CSV, creating header if missing."""
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)
    if exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_headers = reader.fieldnames or []
            if existing_headers == headers:
                with open(csv_path, "a", encoding="utf-8", newline="") as fa:
                    w = csv.writer(fa)
                    w.writerows(rows)
                return
            existing_rows = [{key: row.get(key, "") for key in headers} for row in reader]
        new_rows = [
            {headers[idx]: value for idx, value in enumerate(row) if idx < len(headers)}
            for row in rows
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(new_rows)
        return
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def append_dict_rows_csv(csv_path: str, headers: List[str], rows: List[Dict[str, str]]) -> None:
    """Appends multiple dict rows to a CSV, creating header if missing."""
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    existing_rows: List[Dict[str, str]] = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_headers = reader.fieldnames or []
            if existing_headers == headers:
                with open(csv_path, "a", encoding="utf-8", newline="") as fa:
                    writer = csv.DictWriter(fa, fieldnames=headers, extrasaction="ignore")
                    writer.writerows(rows)
                return
            for row in reader:
                normalized = {key: row.get(key, "") for key in headers}
                existing_rows.append(normalized)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        if existing_rows:
            writer.writerows(existing_rows)
        writer.writerows(rows)


def parse_seeds(arg_seeds: str, exp_cfg: Dict[str, Any]) -> List[int]:
    """Resolves seed list from CLI or exp config."""
    if arg_seeds:
        return [int(s.strip()) for s in arg_seeds.split(",") if s.strip()]
    exp_block = exp_cfg.get("experiment", {})
    if isinstance(exp_block, dict) and "seeds" in exp_block:
        return [int(s) for s in exp_block.get("seeds", [])]
    return [int(exp_block.get("seed", 42)) if isinstance(exp_block, dict) else 42]


def parse_transfer_pair_overrides(raw_pairs: str) -> Optional[List[Dict[str, str]]]:
    """Parses transfer pair overrides encoded as comma-separated source:target entries."""
    pair_text = str(raw_pairs or "").strip()
    if len(pair_text) == 0:
        return None
    parsed_pairs: List[Dict[str, str]] = []
    for pair_token in pair_text.split(","):
        normalized_token = pair_token.strip()
        if len(normalized_token) == 0:
            continue
        parts = [part.strip() for part in normalized_token.split(":")]
        if len(parts) != 2 or len(parts[0]) == 0 or len(parts[1]) == 0:
            raise ValueError(
                "Transfer pair overrides must use comma-separated source:target entries."
            )
        parsed_pairs.append({"source": parts[0], "target": parts[1]})
    if len(parsed_pairs) == 0:
        raise ValueError("Transfer pair override string did not contain any valid source:target entries.")
    return parsed_pairs


def sanitize_cache_token(text: str) -> str:
    """Returns a filesystem-safe token for derived cache names."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    return safe or "default"


def resolve_dataset_text_runtime_cfg(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves runtime text encoder/cache configuration from the experiment config."""
    dataset_cfg = exp_cfg.get("dataset", {}) if isinstance(exp_cfg.get("dataset", {}), dict) else {}
    text_encoder_cfg = dataset_cfg.get("text_encoder", {}) if isinstance(dataset_cfg.get("text_encoder", {}), dict) else {}
    normalization_cfg = dataset_cfg.get("text_normalization", {}) if isinstance(dataset_cfg.get("text_normalization", {}), dict) else {}
    raw_root = str(
        text_encoder_cfg.get("raw_root")
        or dataset_cfg.get("raw_root")
        or ""
    ).strip()
    processed_root = str(
        text_encoder_cfg.get("processed_root")
        or dataset_cfg.get("root")
        or dataset_cfg.get("processed_root")
        or ""
    ).strip()
    model_name_or_path = str(text_encoder_cfg.get("model_name_or_path", "")).strip()
    enabled = len(model_name_or_path) > 0
    cache_tag = str(text_encoder_cfg.get("cache_tag", "")).strip()
    if len(cache_tag) == 0 and enabled:
        cache_tag = sanitize_cache_token(os.path.basename(model_name_or_path) or model_name_or_path)
    cache_name = str(text_encoder_cfg.get("cache_name", "")).strip()
    if len(cache_name) == 0:
        cache_name = "text_embs.pt" if not enabled else f"text_embs_{cache_tag}.pt"
    return {
        "enabled": enabled,
        "model_name_or_path": model_name_or_path,
        "batch_size": int(text_encoder_cfg.get("batch_size", 64)),
        "cache_name": cache_name,
        "force_text_cache": bool(text_encoder_cfg.get("force_text_cache", enabled)),
        "normalize_embeddings": bool(text_encoder_cfg.get("normalize_embeddings", False)),
        "rebuild_cache": bool(text_encoder_cfg.get("rebuild_cache", False)),
        "normalization": normalization_cfg,
        "raw_root": raw_root,
        "processed_root": processed_root,
    }


def resolve_local_text_model_path(text_runtime_cfg: Dict[str, Any]) -> str:
    """Resolves a configured local text model path and rejects missing local models."""
    dataset_cfg = read_yaml_like(os.path.join("Libs", "configs", "dataset.yaml")).get("dataset", {})
    raw_root = str(text_runtime_cfg.get("raw_root") or dataset_cfg.get("raw_root", os.path.join("Input", "raw")))
    model_name_or_path = str(text_runtime_cfg.get("model_name_or_path", "")).strip()
    model_path = resolve_text_model_path(model_name_or_path, raw_root)
    if len(model_name_or_path) == 0:
        raise ValueError("text_encoder.model_name_or_path must be set when runtime text encoder is enabled.")
    if os.path.isabs(model_name_or_path) or os.sep in model_name_or_path:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Configured local text model directory does not exist: {model_path}"
            )
        return model_path
    return model_path


def ensure_text_embedding_cache(dataset_name: str, text_runtime_cfg: Dict[str, Any]) -> None:
    """Ensures the configured text embedding cache exists for a dataset."""
    if not bool(text_runtime_cfg.get("enabled", False)):
        return
    dataset_cfg = read_yaml_like(os.path.join("Libs", "configs", "dataset.yaml")).get("dataset", {})
    raw_root = str(text_runtime_cfg.get("raw_root") or dataset_cfg.get("raw_root", os.path.join("Input", "raw")))
    processed_root = str(text_runtime_cfg.get("processed_root") or dataset_cfg.get("processed_root", os.path.join("Input", "processed")))
    cache_name = str(text_runtime_cfg.get("cache_name", "text_embs.pt"))
    cache_path = os.path.join(processed_root, dataset_name, cache_name)
    if os.path.isfile(cache_path) and not bool(text_runtime_cfg.get("rebuild_cache", False)):
        return
    model_path = resolve_local_text_model_path(text_runtime_cfg)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if dataset_name in {"weibo", "pheme"}:
        mapping = collect_json_source_texts(os.path.join(raw_root, dataset_name))
        build_text_embedding_cache_from_mapping(
            dataset=dataset_name,
            processed_root=processed_root,
            mapping=mapping,
            model_path=model_path,
            batch_size=int(text_runtime_cfg.get("batch_size", 64)),
            normalization_cfg=text_runtime_cfg.get("normalization", {}),
            cache_name=cache_name,
        )
    else:
        build_text_embedding_cache(
            dataset=dataset_name,
            raw_root=raw_root,
            processed_root=processed_root,
            model_name=model_path,
            batch_size=int(text_runtime_cfg.get("batch_size", 64)),
            normalization_cfg=text_runtime_cfg.get("normalization", {}),
            cache_name=cache_name,
        )


def make_graph_dataset(
    root: str,
    split: str,
    use_lp_edges: bool,
    prefix_cfg: Dict[str, Any],
    observability_cfg: Dict[str, Any],
    text_runtime_cfg: Optional[Dict[str, Any]] = None,
    use_potentials: bool = False,
    potentials_dir: Optional[str] = None,
    potentials_pair: Optional[List[int]] = None,
    potential_cache_metadata: Optional[Dict[str, Any]] = None,
    prefix_consistency_views: Optional[List[Dict[str, Any]]] = None,
    prefix_consistency_potential_specs: Optional[List[Dict[str, Any]]] = None,
    mode: str = "legacy",
    require_lp_edges: bool = False,
    label_schema: Optional[Dict[str, int]] = None,
) -> GraphDataset:
    """Builds a GraphDataset with consistent runtime text-cache overrides."""
    # Canonical protocol settings are carried through the resolved prefix
    # contract so every runner call site receives the same data mode and LP
    # policy without silently changing historical legacy runs.
    if mode == "legacy" and isinstance(prefix_cfg, dict) and prefix_cfg.get("_runtime_mode"):
        requested_mode = str(prefix_cfg["_runtime_mode"])
        split_modes = prefix_cfg.get("_runtime_modes", {}) if isinstance(prefix_cfg.get("_runtime_modes", {}), dict) else {}
        mode = str(split_modes.get(split, requested_mode))
        if requested_mode == "eval_prefix" and split == "train" and split not in split_modes:
            mode = "train_arbiter"
    if not require_lp_edges and isinstance(prefix_cfg, dict):
        require_lp_edges = bool(prefix_cfg.get("_require_lp_edges", False))
    text_cfg = text_runtime_cfg if isinstance(text_runtime_cfg, dict) else {}
    return GraphDataset(
        root=root,
        split=split,
        use_lp_edges=use_lp_edges,
        use_potentials=use_potentials,
        potentials_dir=potentials_dir,
        potentials_pair=potentials_pair,
        potential_cache_metadata=potential_cache_metadata,
        prefix_cfg=prefix_cfg,
        prefix_consistency_views=prefix_consistency_views,
        prefix_consistency_potential_specs=prefix_consistency_potential_specs,
        observability_cfg=observability_cfg,
        text_cache_name=str(text_cfg.get("cache_name", "text_embs.pt")),
        force_text_cache=bool(text_cfg.get("force_text_cache", False)),
        text_normalize=bool(text_cfg.get("normalize_embeddings", False)),
        mode=str(mode),
        require_lp_edges=bool(require_lp_edges),
        label_schema=label_schema,
    )


def resolve_dataset_prefix_cfg(prefix_cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Resolves a dataset-specific observed-prefix policy from YAML settings."""
    resolved = dict(prefix_cfg) if isinstance(prefix_cfg, dict) else {}
    dataset_key = str(dataset_name)
    window = resolved.get("window", resolved.get("minutes", None))
    window_overrides = resolved.get("window_overrides", resolved.get("windows_overrides", {}))
    if isinstance(window_overrides, dict) and dataset_key in window_overrides:
        window = window_overrides[dataset_key]
    if window not in (None, ""):
        resolved["window"] = int(window)
        resolved["minutes"] = int(window)

    strategy_overrides = resolved.get("strategy_overrides", resolved.get("unit_overrides", {}))
    if isinstance(strategy_overrides, dict) and dataset_key in strategy_overrides:
        resolved_unit = str(strategy_overrides[dataset_key]).lower()
        resolved["strategy"] = resolved_unit
        resolved["unit"] = resolved_unit
    resolved["include_root"] = bool(resolved.get("include_root", True))
    return resolved


def resolve_early_observation_spec(
    eval_cfg: Dict[str, Any],
    model_eval_cfg: Dict[str, Any],
    dataset_name: str,
) -> Tuple[List[int], str]:
    """Resolves the configured early-observation windows and their semantic unit."""
    observation_cfg = eval_cfg.get("early_observation", {}) if isinstance(eval_cfg.get("early_observation", {}), dict) else {}
    windows = observation_cfg.get("default_windows", []) if isinstance(observation_cfg, dict) else []
    unit = str(observation_cfg.get("default_unit", "minutes")) if isinstance(observation_cfg, dict) else "minutes"
    windows_overrides = observation_cfg.get("windows_overrides", {}) if isinstance(observation_cfg, dict) else {}
    unit_overrides = observation_cfg.get("unit_overrides", {}) if isinstance(observation_cfg, dict) else {}
    if isinstance(windows_overrides, dict) and dataset_name in windows_overrides:
        windows = windows_overrides[dataset_name]
    if isinstance(unit_overrides, dict) and dataset_name in unit_overrides:
        unit = str(unit_overrides[dataset_name])
    if len(windows) == 0 and isinstance(model_eval_cfg, dict):
        model_observation_cfg = model_eval_cfg.get("early_observation", {}) if isinstance(model_eval_cfg.get("early_observation", {}), dict) else {}
        if len(model_observation_cfg) > 0:
            windows = model_observation_cfg.get("default_windows", [])
            unit = str(model_observation_cfg.get("default_unit", unit))
            model_windows_overrides = model_observation_cfg.get("windows_overrides", {})
            model_unit_overrides = model_observation_cfg.get("unit_overrides", {})
            if isinstance(model_windows_overrides, dict) and dataset_name in model_windows_overrides:
                windows = model_windows_overrides[dataset_name]
            if isinstance(model_unit_overrides, dict) and dataset_name in model_unit_overrides:
                unit = str(model_unit_overrides[dataset_name])
    if len(windows) == 0 and isinstance(model_eval_cfg, dict):
        windows = model_eval_cfg.get("early_obs_minutes", [])
        legacy_overrides = model_eval_cfg.get("early_obs_minutes_overrides", {})
        if isinstance(legacy_overrides, dict) and dataset_name in legacy_overrides:
            windows = legacy_overrides[dataset_name]
        if dataset_name in {"weibo", "pheme"}:
            unit = "hop"
    return [int(window) for window in windows], str(unit).lower()


def build_observation_prefix_cfg(
    base_prefix_cfg: Dict[str, Any],
    observation_window: int,
    observation_unit: str,
) -> Dict[str, Any]:
    """Builds a prefix config that materializes an early-window dataset view."""
    resolved = dict(base_prefix_cfg) if isinstance(base_prefix_cfg, dict) else {}
    resolved["strategy"] = str(observation_unit).lower()
    resolved["window"] = int(observation_window)
    resolved["minutes"] = int(observation_window)
    resolved["include_root"] = bool(resolved.get("include_root", True))
    return resolved


def resolve_prefix_consistency_views(
    consistency_cfg: Dict[str, Any],
    dataset_name: str,
    base_prefix_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Resolves auxiliary observed-prefix views for multi-view RD training."""
    if not isinstance(consistency_cfg, dict) or not bool(consistency_cfg.get("enabled", False)):
        return []
    windows = consistency_cfg.get("windows", [])
    unit = str(consistency_cfg.get("unit", base_prefix_cfg.get("strategy", "minutes"))).lower()
    window_overrides = consistency_cfg.get("windows_overrides", {})
    unit_overrides = consistency_cfg.get("unit_overrides", {})
    if isinstance(window_overrides, dict) and dataset_name in window_overrides:
        windows = window_overrides[dataset_name]
    if isinstance(unit_overrides, dict) and dataset_name in unit_overrides:
        unit = str(unit_overrides[dataset_name]).lower()
    if not isinstance(windows, (list, tuple)) or len(windows) == 0:
        raise ValueError("Enabled prefix consistency requires at least one configured observation window.")

    main_unit = str(base_prefix_cfg.get("strategy", base_prefix_cfg.get("unit", "none"))).lower()
    main_window = base_prefix_cfg.get("window", base_prefix_cfg.get("minutes", None))
    views: List[Dict[str, Any]] = []
    seen = set()
    for window in windows:
        resolved_window = int(window)
        key = (unit, resolved_window)
        if key in seen:
            continue
        seen.add(key)
        # The main loader already supplies this supervised view.
        if unit == main_unit and main_window not in (None, "") and resolved_window == int(main_window):
            continue
        views.append(build_observation_prefix_cfg(base_prefix_cfg, resolved_window, unit))
    if len(views) == 0:
        raise ValueError("Prefix consistency requires at least one auxiliary view distinct from the main prefix.")
    return views


def build_prefix_consistency_potential_specs(
    views: Sequence[Dict[str, Any]],
    potentials_root: str,
    dataset_name: str,
    seed: int,
    ctdm_config_payload: Dict[str, Any],
    potentials_pair: Sequence[int],
) -> List[Dict[str, Any]]:
    """Binds every auxiliary prefix view to its own validated cache namespace."""
    specs: List[Dict[str, Any]] = []
    for view_cfg in views:
        resolved_view = dict(view_cfg)
        specs.append(
            {
                "potentials_dir": potential_cache_dir(potentials_root, resolved_view),
                "potentials_pair": [int(index) for index in potentials_pair],
                "potential_cache_metadata": build_potential_cache_metadata(
                    dataset_name=dataset_name,
                    seed=int(seed),
                    prefix_cfg=resolved_view,
                    ctdm_config_payload=ctdm_config_payload,
                ),
            }
        )
    return specs


def aggregate_numeric(values: List[float]) -> tuple[float, float]:
    """Returns (mean, std) for numeric list; std=0 if len<2."""
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def scan_max_nodes_across_splits(datasets: Sequence[GraphDataset]) -> int:
    """Returns the padded node budget required by active prefix views and preserved targets."""
    max_nodes_scan = 0
    for ds in datasets:
        for split_idx in range(len(ds)):
            sample = ds.get(split_idx)
            active_nodes = int(sample.num_nodes)
            full_nodes = int(sample.full_x.size(0)) if hasattr(sample, "full_x") else active_nodes
            max_nodes_scan = max(max_nodes_scan, active_nodes, full_nodes)
    if int(max_nodes_scan) <= 0:
        raise ValueError("Failed to scan a positive max_nodes budget from dataset splits.")
    return int(max_nodes_scan)


def infer_text_channels(dataset: GraphDataset) -> int:
    """Returns graph-level text embedding width when the active dataset provides it."""
    if len(dataset) == 0:
        raise ValueError("Cannot infer text embedding width from an empty dataset")
    sample = dataset.get(0)
    if not hasattr(sample, "text_emb"):
        return 0
    text_emb = sample.text_emb
    if text_emb.dim() != 2:
        raise ValueError(f"text_emb must be [batch, dim], got {tuple(text_emb.shape)}")
    return int(text_emb.size(-1))


def parse_numeric_cell(text: str) -> Optional[float]:
    """Parses a numeric CSV cell, ignoring aggregate strings like mean±std."""
    cell = str(text).strip()
    if cell == "" or "±" in cell:
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def read_csv_rows(csv_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Loads CSV headers and rows, returning empty structures if missing."""
    if not os.path.isfile(csv_path):
        return [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = [{key: row.get(key, "") for key in headers} for row in reader]
    return headers, rows


def rewrite_csv_rows(csv_path: str, headers: List[str], rows: List[Dict[str, str]]) -> None:
    """Rewrites a CSV file with normalized rows."""
    if len(headers) == 0:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dedupe_csv_rows(rows: List[Dict[str, str]], key_fields: Sequence[str]) -> List[Dict[str, str]]:
    """Validate legacy rows have unique composite keys.

    Historical code selected the last timestamped row.  Legacy CSV remains
    exploratory only, and a conflicting row must be repaired explicitly rather
    than silently selected.
    """
    seen = set()
    unique: List[Dict[str, str]] = []
    for index, row in enumerate(rows):
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key in seen:
            raise ValueError(f"Duplicate legacy result key at row {index}: {key!r}; latest-row resolution is disabled")
        seen.add(key)
        unique.append(row)
    return unique


def refresh_result_aggregate_row(
    csv_path: str,
    dataset_name: str,
    source_dataset_name: str,
    transfer_mode: str,
    model_name: str,
    model_cfg_path: str,
    exp_cfg_path: str,
    seeds: Sequence[int],
) -> None:
    """Refreshes the aggregate row for one dataset/model/config slice."""
    headers, rows = read_csv_rows(csv_path)
    if len(headers) == 0:
        return
    numeric_fields = [
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
    deduped_rows = dedupe_csv_rows(
        rows,
        key_fields=["dataset", "source_dataset", "run_kind", "transfer_mode", "model", "model_cfg", "exp_cfg", "seed"],
    )
    seed_keys = {str(int(seed)) for seed in seeds}
    matched_rows = [
        row for row in deduped_rows
        if row.get("dataset", "") == dataset_name
        and row.get("source_dataset", "") == source_dataset_name
        and row.get("run_kind", "") == ("transfer" if dataset_name != source_dataset_name else "main")
        and row.get("transfer_mode", "") == transfer_mode
        and row.get("model", "") == model_name
        and row.get("model_cfg", "") == model_cfg_path
        and row.get("exp_cfg", "") == exp_cfg_path
        and row.get("seed", "") in seed_keys
    ]
    if len(matched_rows) == 0:
        return
    preserved_rows = [
        row for row in deduped_rows
        if not (
            row.get("dataset", "") == dataset_name
            and row.get("source_dataset", "") == source_dataset_name
            and row.get("run_kind", "") == ("transfer" if dataset_name != source_dataset_name else "main")
            and row.get("transfer_mode", "") == transfer_mode
            and row.get("model", "") == f"{model_name}_agg"
            and row.get("model_cfg", "") == model_cfg_path
            and row.get("exp_cfg", "") == exp_cfg_path
            and row.get("seed", "") == "agg"
        )
    ]
    agg_row = {key: "" for key in headers}
    agg_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agg_row["dataset"] = dataset_name
    if "source_dataset" in headers:
        agg_row["source_dataset"] = source_dataset_name
    if "run_kind" in headers:
        agg_row["run_kind"] = "transfer" if dataset_name != source_dataset_name else "main"
    if "transfer_mode" in headers:
        agg_row["transfer_mode"] = transfer_mode
    agg_row["model"] = f"{model_name}_agg"
    agg_row["model_cfg"] = model_cfg_path
    agg_row["exp_cfg"] = exp_cfg_path
    agg_row["seed"] = "agg"
    for field in numeric_fields:
        values = [parse_numeric_cell(row.get(field, "")) for row in matched_rows]
        values = [value for value in values if value is not None]
        if len(values) == 0:
            continue
        mean_value, std_value = aggregate_numeric(values)
        agg_row[field] = f"{mean_value:.6f}±{std_value:.6f}"
    preserved_rows.append(agg_row)
    rewrite_csv_rows(csv_path, headers, preserved_rows)


def refresh_transfer_aggregate_row(
    csv_path: str,
    train_dataset: str,
    test_dataset: str,
    transfer_mode: str,
    model_name: str,
    seeds: Sequence[int],
) -> None:
    """Refreshes the aggregate row for one transfer slice."""
    headers, rows = read_csv_rows(csv_path)
    if len(headers) == 0:
        return
    numeric_fields = ["acc", "macro_f1", "nll", "ece", "calibration_temp"]
    deduped_rows = dedupe_csv_rows(
        rows,
        key_fields=["train_dataset", "test_dataset", "transfer_mode", "model", "seed"],
    )
    seed_keys = {str(int(seed)) for seed in seeds}
    matched_rows = [
        row for row in deduped_rows
        if row.get("train_dataset", "") == train_dataset
        and row.get("test_dataset", "") == test_dataset
        and row.get("transfer_mode", "") == transfer_mode
        and row.get("model", "") == model_name
        and row.get("seed", "") in seed_keys
    ]
    if len(matched_rows) == 0:
        return
    preserved_rows = [
        row for row in deduped_rows
        if not (
            row.get("train_dataset", "") == train_dataset
            and row.get("test_dataset", "") == test_dataset
            and row.get("transfer_mode", "") == transfer_mode
            and row.get("model", "") == f"{model_name}_agg"
            and row.get("seed", "") == "agg"
        )
    ]
    agg_row = {key: "" for key in headers}
    agg_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agg_row["train_dataset"] = train_dataset
    agg_row["test_dataset"] = test_dataset
    if "transfer_mode" in headers:
        agg_row["transfer_mode"] = transfer_mode
    agg_row["model"] = f"{model_name}_agg"
    agg_row["seed"] = "agg"
    for field in numeric_fields:
        values = [parse_numeric_cell(row.get(field, "")) for row in matched_rows]
        values = [value for value in values if value is not None]
        if len(values) == 0:
            continue
        mean_value, std_value = aggregate_numeric(values)
        agg_row[field] = f"{mean_value:.6f}±{std_value:.6f}"
    preserved_rows.append(agg_row)
    rewrite_csv_rows(csv_path, headers, preserved_rows)


def freeze_ctdm_for_target_adaptation(model: RumorDiffusion, unfreeze_last_layers: int) -> None:
    """Freezes the CTDM backbone except for the highest layers and heads."""
    if model.ctdm is None:
        return
    for param in model.ctdm.parameters():
        param.requires_grad = False
    for attr_name in ("film", "readout", "full_to_latent", "eps_x", "eps_a", "time_embed", "class_embed"):
        module = getattr(model.ctdm, attr_name, None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True
    for attr_name in ("text_proj", "text_fusion_layer", "gated_text_fusion", "multiscale_proj", "forecast_head"):
        module = getattr(model.ctdm, attr_name, None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True
    backbone = getattr(model.ctdm, "backbone", None)
    if isinstance(backbone, nn.Module) and hasattr(backbone, "layers"):
        layers = list(getattr(backbone, "layers"))
        depth = max(0, int(unfreeze_last_layers))
        if depth > 0:
            for layer in layers[-depth:]:
                if isinstance(layer, nn.Module):
                    for param in layer.parameters():
                        param.requires_grad = True


def configure_ctdm_joint_finetuning(model: RumorDiffusion, finetune_cfg: Dict[str, Any]) -> None:
    """Configures trainable CTDM parameters for joint RumorDiffusion fine-tuning."""
    if model.ctdm is None:
        return
    for param in model.ctdm.parameters():
        param.requires_grad = False
    trainable_heads = finetune_cfg.get(
        "trainable_heads",
        ["film", "readout", "full_to_latent", "eps_x", "eps_a", "time_embed", "class_embed", "text_proj", "text_fusion_layer", "gated_text_fusion", "multiscale_proj", "forecast_head"],
    )
    if not isinstance(trainable_heads, list):
        raise TypeError("training.joint_finetuning.trainable_heads must be a list")
    for attr_name in trainable_heads:
        module = getattr(model.ctdm, str(attr_name), None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = True
    backbone = getattr(model.ctdm, "backbone", None)
    if isinstance(backbone, nn.Module) and hasattr(backbone, "layers"):
        layers = list(getattr(backbone, "layers"))
        depth = max(0, int(finetune_cfg.get("ctdm_unfreeze_last_layers", 0)))
        if depth > 0:
            for layer in layers[-depth:]:
                if isinstance(layer, nn.Module):
                    for param in layer.parameters():
                        param.requires_grad = True


def build_adaptation_optimizer(model: RumorDiffusion, adaptation_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Builds a target-adaptation optimizer with distinct LR groups."""
    arbiter_lr = float(adaptation_cfg.get("arbiter_lr", adaptation_cfg.get("lr", 1e-4)))
    ctdm_lr = float(adaptation_cfg.get("ctdm_lr", arbiter_lr * 0.3))
    weight_decay = float(adaptation_cfg.get("weight_decay", 1e-4))
    arbiter_params: List[nn.Parameter] = []
    ctdm_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("ctdm."):
            ctdm_params.append(param)
        elif "arbiter" in name or "tf_head" in name or "class_embedding" in name or "class_embed" in name:
            arbiter_params.append(param)
        else:
            other_params.append(param)
    param_groups = []
    if arbiter_params:
        param_groups.append({"params": arbiter_params, "lr": arbiter_lr})
    if ctdm_params:
        param_groups.append({"params": ctdm_params, "lr": ctdm_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": arbiter_lr})
    if len(param_groups) == 0:
        raise ValueError("No trainable parameters available for target adaptation.")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def set_model_mode_for_adaptation(model: RumorDiffusion, adaptation_cfg: Dict[str, Any]) -> None:
    """Sets trainable modules to training mode while keeping frozen CTDM parts stable."""
    model.train()
    if model.ctdm is not None:
        model.ctdm.eval()
        backbone = getattr(model.ctdm, "backbone", None)
        if isinstance(backbone, nn.Module) and hasattr(backbone, "layers"):
            layers = list(getattr(backbone, "layers"))
            depth = max(0, int(adaptation_cfg.get("ctdm_unfreeze_last_layers", 1)))
            if depth > 0:
                for layer in layers[-depth:]:
                    if isinstance(layer, nn.Module):
                        layer.train()
        for attr_name in ("film", "readout", "full_to_latent", "eps_x", "eps_a", "time_embed", "class_embed"):
            module = getattr(model.ctdm, attr_name, None)
            if isinstance(module, nn.Module):
                module.train()
        for attr_name in ("text_proj", "text_fusion_layer", "gated_text_fusion", "multiscale_proj", "forecast_head"):
            module = getattr(model.ctdm, attr_name, None)
            if isinstance(module, nn.Module):
                module.train()


def update_ema_teacher(student: RumorDiffusion, teacher: RumorDiffusion, momentum: float) -> None:
    """Updates EMA teacher weights from the student model."""
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(float(momentum)).add_(student_param.data, alpha=1.0 - float(momentum))
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.data.copy_(student_buffer.data)


def symmetric_kl_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Computes symmetric KL divergence between two logits tensors."""
    probs_a = torch.softmax(logits_a, dim=1)
    probs_b = torch.softmax(logits_b, dim=1)
    return 0.5 * (
        F.kl_div(torch.log(probs_a.clamp_min(1e-8)), probs_b, reduction="batchmean")
        + F.kl_div(torch.log(probs_b.clamp_min(1e-8)), probs_a, reduction="batchmean")
    )


def apply_transfer_logit_adjustment(logits: torch.Tensor, class_priors: torch.Tensor, strength: float) -> torch.Tensor:
    """Applies logit adjustment using class priors."""
    if float(strength) <= 0.0:
        return logits
    priors = class_priors.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-8)
    return logits - float(strength) * torch.log(priors).unsqueeze(0)


def compute_target_prototype_alignment_loss(
    model: RumorDiffusion,
    pseudo_labels: torch.Tensor,
    confidence_mask: torch.Tensor,
    min_samples_per_class: int,
) -> torch.Tensor:
    """Aligns target pseudo-label prototypes with source prototypes."""
    if model.ctdm is None or not hasattr(model, "_last_potentials"):
        return torch.zeros((), device=pseudo_labels.device)
    source_prototypes = getattr(model.ctdm, "prototypes", None)
    if source_prototypes is None:
        return torch.zeros((), device=pseudo_labels.device)
    potentials = getattr(model, "_last_potentials")
    if not isinstance(potentials, torch.Tensor) or potentials.dim() != 3:
        return torch.zeros((), device=pseudo_labels.device)
    confident_idx = torch.where(confidence_mask)[0]
    if confident_idx.numel() == 0:
        return torch.zeros((), device=pseudo_labels.device)
    confident_potentials = potentials[confident_idx]
    confident_labels = pseudo_labels[confident_idx]
    losses: List[torch.Tensor] = []
    for class_idx in range(int(model.num_classes)):
        class_mask = confident_labels == int(class_idx)
        if int(class_mask.sum().item()) < int(min_samples_per_class):
            continue
        target_proto = confident_potentials[class_mask, int(class_idx), :].mean(dim=0)
        source_proto = source_prototypes[int(class_idx)].to(target_proto.device, dtype=target_proto.dtype)
        losses.append(1.0 - F.cosine_similarity(target_proto.unsqueeze(0), source_proto.unsqueeze(0)).mean())
    if len(losses) == 0:
        return torch.zeros((), device=pseudo_labels.device)
    return torch.stack(losses).mean()


def adapt_rumordiffusion_to_target(
    model: RumorDiffusion,
    trainer: Trainer,
    source_checkpoint_path: str,
    target_train_loader: DataLoader,
    target_val_loader: DataLoader,
    adaptation_cfg: Dict[str, Any],
    class_names: Dict[int, str],
    device: str,
    save_path: str,
) -> Dict[str, Any]:
    """Performs unsupervised target adaptation with pseudo-labeling and consistency."""
    if model.ctdm is None:
        raise ValueError("RumorDiffusion target adaptation requires a CTDM module.")
    freeze_ctdm_for_target_adaptation(model, int(adaptation_cfg.get("ctdm_unfreeze_last_layers", 1)))
    optimizer = build_adaptation_optimizer(model, adaptation_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(adaptation_cfg.get("epochs", 5))),
    )
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    ema_decay = float(adaptation_cfg.get("ema_decay", 0.995))
    pseudo_weight = float(adaptation_cfg.get("pseudo_label_weight", 1.0))
    consistency_weight = float(adaptation_cfg.get("consistency_weight", 0.1))
    prototype_weight = float(adaptation_cfg.get("prototype_align_weight", 0.1))
    confidence_threshold = float(adaptation_cfg.get("confidence_threshold", 0.8))
    min_samples_per_class = int(adaptation_cfg.get("min_samples_per_class", 2))
    grad_clip = float(adaptation_cfg.get("grad_clip", trainer.cfg.grad_clip))
    grad_accum_steps = max(1, int(adaptation_cfg.get("gradient_accumulation_steps", 1)))
    logit_adjust_strength = float(adaptation_cfg.get("logit_adjustment_strength", 0.0))
    adapt_text_dropout = float(adaptation_cfg.get("consistency_text_dropout", 0.1))
    adapt_text_noise = float(adaptation_cfg.get("consistency_text_noise_std", 0.0))
    adapt_edge_drop = float(adaptation_cfg.get("consistency_edge_drop_rate", 0.1))
    obs_min = int(adaptation_cfg.get("consistency_obs_min", getattr(model.ctdm, "cutoff_min", 10)))
    obs_max = int(adaptation_cfg.get("consistency_obs_max", getattr(model.ctdm, "cutoff_max", getattr(model.ctdm, "obs_minutes", 30))))
    use_class_reweight = bool(adaptation_cfg.get("use_class_reweighting", False))
    use_amp = bool(trainer.cfg.amp) and str(device).startswith("cuda")
    source_priors = model.class_priors.detach().clone() if hasattr(model, "class_priors") else torch.ones(model.num_classes, device=device) / float(model.num_classes)
    class_weight_tensor = None
    if use_class_reweight:
        class_weight_tensor = (1.0 / source_priors.clamp_min(1e-6))
        class_weight_tensor = class_weight_tensor / class_weight_tensor.mean()
        class_weight_tensor = class_weight_tensor.to(device=device, dtype=torch.float32)

    model.eval()
    baseline_val_metrics = trainer.evaluate(target_val_loader, detailed=True, class_names=class_names)
    baseline_val_macro_f1 = float(baseline_val_metrics.get("macro_f1", 0.0))
    logging.info(f"[TargetAdapt] source-only target-val macro_f1={baseline_val_macro_f1:.4f}")

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_metric = float(baseline_val_macro_f1)
    best_pseudo_coverage = 0.0
    best_epoch = 0
    for epoch in range(int(adaptation_cfg.get("epochs", 5))):
        set_model_mode_for_adaptation(model, adaptation_cfg)
        epoch_batches = 0
        epoch_confident = 0
        epoch_seen = 0
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch in target_train_loader:
            batch = batch.to(device)
            batch_size = int(batch.y.view(-1).size(0))
            epoch_seen += batch_size
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        teacher_logits = teacher(batch)
                else:
                    teacher_logits = teacher(batch)
                teacher_probs = torch.softmax(teacher_logits, dim=1)
                teacher_confidence, pseudo_labels = torch.max(teacher_probs, dim=1)
                confident_mask = teacher_confidence >= float(confidence_threshold)
            model.train()
            set_model_mode_for_adaptation(model, adaptation_cfg)
            autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                clean_logits = model(batch)
                adjusted_logits = apply_transfer_logit_adjustment(clean_logits, source_priors, logit_adjust_strength)
                if confident_mask.any():
                    loss_pseudo = F.cross_entropy(
                        adjusted_logits[confident_mask],
                        pseudo_labels[confident_mask],
                        weight=class_weight_tensor,
                    )
                    epoch_confident += int(confident_mask.sum().item())
                else:
                    loss_pseudo = clean_logits.new_zeros(())

                text_noise_backup = float(getattr(model.ctdm, "text_noise_std", 0.0))
                text_drop_backup = float(getattr(model.ctdm, "text_dropout_p", 0.0))
                edge_drop_backup = float(getattr(model.ctdm, "edge_drop_rate", 0.0))
                obs_backup = int(getattr(model.ctdm, "obs_minutes", obs_max))
                model.ctdm.text_noise_std = float(adapt_text_noise)
                model.ctdm.text_dropout_p = float(adapt_text_dropout)
                model.ctdm.edge_drop_rate = float(adapt_edge_drop)
                model.ctdm.obs_minutes = int(torch.randint(low=min(obs_min, obs_max), high=max(obs_min, obs_max) + 1, size=(1,)).item())
                noisy_logits = model(batch)
                model.ctdm.text_noise_std = text_noise_backup
                model.ctdm.text_dropout_p = text_drop_backup
                model.ctdm.edge_drop_rate = edge_drop_backup
                model.ctdm.obs_minutes = obs_backup

                loss_consistency = symmetric_kl_from_logits(clean_logits, noisy_logits)
                loss_proto = compute_target_prototype_alignment_loss(
                    model=model,
                    pseudo_labels=pseudo_labels,
                    confidence_mask=confident_mask,
                    min_samples_per_class=min_samples_per_class,
                )
                total_loss = (
                    pseudo_weight * loss_pseudo
                    + consistency_weight * loss_consistency
                    + prototype_weight * loss_proto
                )
                total_loss = total_loss / float(grad_accum_steps)
            if use_amp and trainer.scaler is not None:
                trainer.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            if (epoch_batches + 1) % grad_accum_steps == 0:
                if grad_clip > 0.0:
                    if use_amp and trainer.scaler is not None:
                        trainer.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if use_amp and trainer.scaler is not None:
                    trainer.scaler.step(optimizer)
                    trainer.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema_teacher(model, teacher, ema_decay)
            epoch_loss += float(total_loss.detach().item())
            epoch_batches += 1
        if epoch_batches % grad_accum_steps != 0:
            if grad_clip > 0.0:
                if use_amp and trainer.scaler is not None:
                    trainer.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_amp and trainer.scaler is not None:
                trainer.scaler.step(optimizer)
                trainer.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema_teacher(model, teacher, ema_decay)
        scheduler.step()
        model.eval()
        val_metrics = trainer.evaluate(target_val_loader, detailed=True, class_names=class_names)
        current_metric = float(val_metrics.get("macro_f1", 0.0))
        pseudo_coverage = float(epoch_confident / max(1, epoch_seen))
        logging.info(
            f"[TargetAdapt] epoch={epoch + 1}, loss={(epoch_loss * float(grad_accum_steps)) / max(1, epoch_batches):.4f}, "
            f"pseudo_coverage={pseudo_coverage:.4f}, val_macro_f1={current_metric:.4f}"
        )
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_pseudo_coverage = pseudo_coverage
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint_obj = build_checkpoint_payload(
                best_state,
                metadata={
                    "format_version": 1,
                    "model_name": "rumordiffusion",
                    "stage": "target_adapt",
                    "source_checkpoint_path": source_checkpoint_path,
                    "adapt_epoch": int(best_epoch),
                },
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint_obj, save_path)
    model.load_state_dict(best_state)
    return {
        "best_val_macro_f1": float(best_val_metric),
        "best_val_metric": float(best_val_metric),
        "adapt_epochs": int(best_epoch),
        "pseudo_label_coverage": float(best_pseudo_coverage),
        "adapt_ckpt_path": save_path,
        "baseline_target_val_macro_f1": float(baseline_val_macro_f1),
    }


def derive_ctdm_checkpoint_from_arbiter_checkpoint(
    arbiter_ckpt_path: str,
    ctdm_subdir: str = "rumordiffusion_ctdm",
) -> Optional[str]:
    """Derives a sibling CTDM checkpoint path from a RumorDiffusion arbiter checkpoint."""
    normalized = os.path.normpath(str(arbiter_ckpt_path))
    parts = normalized.split(os.sep)
    if "rumordiffusion" not in parts:
        return None
    idx = parts.index("rumordiffusion")
    parts[idx] = str(ctdm_subdir)
    return os.sep.join(parts)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Returns the q-th percentile from a sorted numeric sequence."""
    assert 0.0 <= float(q) <= 1.0
    if len(sorted_values) == 0:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = int(round(float(q) * float(len(sorted_values) - 1)))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def summarize_latency_ms(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Summarizes a latency distribution in milliseconds."""
    if len(latencies_ms) == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(float(v) for v in latencies_ms)
    return {
        "mean_ms": float(sum(ordered) / float(len(ordered))),
        "p95_ms": percentile(ordered, 0.95),
    }


def profile_cuda_boundary(device: str) -> None:
    """Synchronizes CUDA when profiling GPU latency."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_eval_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    amp: bool = False,
    detailed: bool = True,
    class_names: Optional[Dict[int, str]] = None,
    compute_calibration: bool = False,
    ece_bins: int = 15,
    profile: bool = False,
    profile_repeat: int = 1,
    profile_warmup_batches: int = 0,
    profile_max_batches: int = 0,
) -> Dict[str, Any]:
    """Collects logits, targets, source ids, and optional latency profile."""
    model.eval()
    use_amp = bool(amp) and str(device).startswith("cuda")
    logits_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []
    source_ids: List[str] = []
    per_sample_latency_ms: List[float] = []
    total_loss = 0.0
    total = 0
    profiled_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            source_ids.extend(_extract_source_ids_from_batch(batch))
            logits = None
            should_profile = bool(profile) and (
                int(profile_max_batches) <= 0 or profiled_batches < int(profile_max_batches)
            )
            if should_profile:
                repeat_count = max(1, int(profile_repeat))
                if batch_idx < int(profile_warmup_batches):
                    profile_cuda_boundary(device)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            _ = model(batch)
                    else:
                        _ = model(batch)
                    profile_cuda_boundary(device)
                batch_size = max(1, int(batch.y.view(-1).size(0)))
                repeated_latency_ms: List[float] = []
                for _ in range(repeat_count):
                    profile_cuda_boundary(device)
                    start = time.perf_counter()
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            logits_profile = model(batch)
                    else:
                        logits_profile = model(batch)
                    profile_cuda_boundary(device)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    repeated_latency_ms.append(elapsed_ms / float(batch_size))
                    logits = logits_profile
                batch_latency_ms = sum(repeated_latency_ms) / max(1, len(repeated_latency_ms))
                per_sample_latency_ms.extend([batch_latency_ms] * batch_size)
                profiled_batches += 1
            else:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(batch)
                else:
                    logits = model(batch)

            assert logits is not None
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError(
                    "Model produced non-finite logits during evaluation."
                )
            loss = criterion(logits, batch.y.view(-1))
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError(
                    "Evaluation loss is non-finite for finite logits."
                )
            logits_list.append(logits.detach().to(torch.float32).cpu())
            target_list.append(batch.y.view(-1).cpu())
            bsz = int(batch.y.view(-1).size(0))
            total_loss += float(loss.item()) * float(bsz)
            total += bsz

    if len(logits_list) == 0:
        return {
            "metrics": {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0},
            "logits": torch.empty((0, 0)),
            "targets": torch.empty((0,), dtype=torch.long),
            "source_ids": [],
            "latency_ms": [],
        }

    logits_cat = torch.cat(logits_list, dim=0)
    targets_cat = torch.cat(target_list, dim=0)
    if len(source_ids) != int(targets_cat.size(0)):
        raise ValueError(
            f"Prediction export alignment mismatch: collected {len(source_ids)} source ids "
            f"for {int(targets_cat.size(0))} targets."
        )
    metrics = metrics_from_logits(
        logits_cat,
        targets_cat,
        num_classes=int(logits_cat.size(1)),
        detailed=bool(detailed),
        class_names=class_names,
        compute_calibration=bool(compute_calibration),
        ece_bins=int(ece_bins),
    )
    metrics["loss"] = total_loss / max(1, total)
    return {
        "metrics": metrics,
        "logits": logits_cat,
        "targets": targets_cat,
        "source_ids": source_ids,
        "latency_ms": per_sample_latency_ms,
    }


def export_prediction_rows(
    csv_path: str,
    dataset_name: str,
    source_dataset_name: str,
    run_kind: str,
    model_name: str,
    seed: int,
    tag: str,
    source_ids: Sequence[str],
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Exports per-sample predictions for later significance testing and case analysis."""
    headers = [
        "dataset",
        "source_dataset",
        "run_kind",
        "model",
        "seed",
        "tag",
        "source_id",
        "target",
        "pred",
        "confidence",
        "probabilities",
    ]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    if len(source_ids) != int(targets.size(0)):
        raise ValueError(
            f"Cannot export predictions: {len(source_ids)} source ids vs {int(targets.size(0))} targets."
        )
    rows: List[Dict[str, str]] = []
    for idx, source_id in enumerate(source_ids):
        rows.append(
            {
                "dataset": dataset_name,
                "source_dataset": source_dataset_name,
                "run_kind": run_kind,
                "model": model_name,
                "seed": str(seed),
                "tag": tag,
                "source_id": str(source_id),
                "target": str(int(targets[idx].item())),
                "pred": str(int(preds[idx].item())),
                "confidence": f"{float(probs[idx, preds[idx]].item()):.6f}",
                "probabilities": json.dumps([float(v) for v in probs[idx].tolist()]),
            }
        )
    append_dict_rows_csv(csv_path, headers, rows)


def load_prediction_rows(
    csv_path: str,
    dataset_name: str,
    source_dataset_name: str,
    run_kind: str,
    model_name: str,
    seed: int,
    tag: str,
) -> Dict[str, Dict[str, Any]]:
    """Loads exported per-sample predictions keyed by source id."""
    if not os.path.isfile(csv_path):
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset", "") != dataset_name:
                continue
            if row.get("source_dataset", "") != source_dataset_name:
                continue
            if row.get("run_kind", "") != run_kind:
                continue
            if row.get("model", "") != model_name:
                continue
            if row.get("seed", "") != str(seed):
                continue
            if row.get("tag", "") != tag:
                continue
            source_id = row.get("source_id", "")
            rows[source_id] = {
                "target": int(row["target"]),
                "pred": int(row["pred"]),
                "confidence": float(row["confidence"]),
            }
    return rows


def bootstrap_compare_predictions(
    current_rows: Dict[str, Dict[str, Any]],
    baseline_rows: Dict[str, Dict[str, Any]],
    metric_name: str,
    n_bootstrap: int,
    random_seed: int,
) -> Dict[str, float]:
    """Computes paired bootstrap significance between two aligned prediction sets."""
    shared_ids = sorted(set(current_rows.keys()) & set(baseline_rows.keys()))
    if len(shared_ids) == 0:
        return {"delta": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0, "count": 0.0}
    targets = torch.tensor([int(current_rows[sid]["target"]) for sid in shared_ids], dtype=torch.long)
    preds_current = torch.tensor([int(current_rows[sid]["pred"]) for sid in shared_ids], dtype=torch.long)
    preds_baseline = torch.tensor([int(baseline_rows[sid]["pred"]) for sid in shared_ids], dtype=torch.long)
    metric_name = str(metric_name).lower()

    def metric_value(preds: torch.Tensor, target_subset: torch.Tensor) -> float:
        if metric_name == "acc":
            return compute_accuracy(preds, target_subset)
        if metric_name == "macro_f1":
            num_classes = int(
                max(
                    int(target_subset.max().item()),
                    int(preds.max().item()),
                ) + 1
            )
            return compute_macro_f1(preds, target_subset, num_classes=num_classes)
        raise ValueError(f"Unsupported significance metric: {metric_name}")

    delta_full = float(metric_value(preds_current, targets) - metric_value(preds_baseline, targets))
    generator = torch.Generator()
    generator.manual_seed(int(random_seed))
    sample_count = int(targets.numel())
    deltas: List[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sample_idx = torch.randint(low=0, high=sample_count, size=(sample_count,), generator=generator)
        sampled_targets = targets[sample_idx]
        deltas.append(
            float(
                metric_value(preds_current[sample_idx], sampled_targets)
                - metric_value(preds_baseline[sample_idx], sampled_targets)
            )
        )
    ordered = sorted(deltas)
    non_positive = sum(1 for delta in ordered if delta <= 0.0)
    non_negative = sum(1 for delta in ordered if delta >= 0.0)
    tail_count = min(non_positive, non_negative)
    return {
        "delta": delta_full,
        "p_value": float(min(1.0, 2.0 * (tail_count + 1) / float(len(ordered) + 1))),
        "ci_low": percentile(ordered, 0.025),
        "ci_high": percentile(ordered, 0.975),
        "count": float(sample_count),
    }


def build_revision_row(
    timestamp: str,
    dataset_name: str,
    source_dataset_name: str,
    run_kind: str,
    model_name: str,
    seed: int,
    artifact_type: str,
    tag: str,
    values: Dict[str, Any],
    model_cfg_path: str = "",
    exp_cfg_path: str = "",
    log_path: str = "",
) -> Dict[str, str]:
    """Builds one normalized revision-artifact row."""
    row = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "source_dataset": source_dataset_name,
        "run_kind": run_kind,
        "model": model_name,
        "model_cfg": model_cfg_path,
        "exp_cfg": exp_cfg_path,
        "log_path": log_path,
        "seed": str(seed),
        "artifact_type": artifact_type,
        "tag": tag,
        "observation_window": "",
        "observation_unit": "",
        "obs_minutes": "",
        "samples": "",
        "tau_low": "",
        "tau_high": "",
        "gamma": "",
        "coverage": "",
        "acc_resolved": "",
        "resolved_macro_f1": "",
        "resolved_ece": "",
        "mean_prob": "",
        "mean_var": "",
        "latency_mean_ms": "",
        "latency_p95_ms": "",
        "compare_model": "",
        "metric": "",
        "delta": "",
        "p_value": "",
        "ci_low": "",
        "ci_high": "",
        "notes": "",
    }
    for key, value in values.items():
        if key not in row:
            continue
        if isinstance(value, float):
            if key == "mean_var":
                row[key] = f"{value:.8g}"
            else:
                row[key] = f"{value:.6f}"
        else:
            row[key] = str(value)
    return row


def revision_artifact_headers() -> List[str]:
    """Returns the stable column order for revision artifact CSV files."""
    return [
        "timestamp",
        "dataset",
        "source_dataset",
        "run_kind",
        "model",
        "model_cfg",
        "exp_cfg",
        "log_path",
        "seed",
        "artifact_type",
        "tag",
        "observation_window",
        "observation_unit",
        "obs_minutes",
        "samples",
        "tau_low",
        "tau_high",
        "gamma",
        "coverage",
        "acc_resolved",
        "resolved_macro_f1",
        "resolved_ece",
        "mean_prob",
        "mean_var",
        "latency_mean_ms",
        "latency_p95_ms",
        "compare_model",
        "metric",
        "delta",
        "p_value",
        "ci_low",
        "ci_high",
        "notes",
    ]


def persist_significance_rows(
    prediction_csv: str,
    revision_artifacts_csv: str,
    revision_headers: List[str],
    timestamp: str,
    dataset_name: str,
    source_dataset_name: str,
    run_kind: str,
    reference_model: str,
    compare_model: str,
    seed: int,
    metric_names: Sequence[str],
    bootstrap_rounds: int,
    model_cfg_path: str = "",
    exp_cfg_path: str = "",
    log_path: str = "",
) -> int:
    """Writes paired-bootstrap rows for one reference-vs-comparison pair."""
    reference_name = str(reference_model).lower()
    compare_name = str(compare_model).lower()
    if reference_name == compare_name:
        return 0
    reference_rows = load_prediction_rows(
        csv_path=prediction_csv,
        dataset_name=dataset_name,
        source_dataset_name=source_dataset_name,
        run_kind=run_kind,
        model_name=reference_name,
        seed=int(seed),
        tag="main",
    )
    compare_rows = load_prediction_rows(
        csv_path=prediction_csv,
        dataset_name=dataset_name,
        source_dataset_name=source_dataset_name,
        run_kind=run_kind,
        model_name=compare_name,
        seed=int(seed),
        tag="main",
    )
    if len(reference_rows) == 0 or len(compare_rows) == 0:
        return 0
    rows: List[Dict[str, str]] = []
    for metric_name in metric_names:
        stats = bootstrap_compare_predictions(
            current_rows=reference_rows,
            baseline_rows=compare_rows,
            metric_name=str(metric_name),
            n_bootstrap=int(bootstrap_rounds),
            random_seed=int(seed),
        )
        rows.append(
            build_revision_row(
                timestamp=timestamp,
                dataset_name=dataset_name,
                source_dataset_name=source_dataset_name,
                run_kind=run_kind,
                model_name=reference_name,
                seed=int(seed),
                artifact_type="significance",
                tag="main",
                model_cfg_path=model_cfg_path,
                exp_cfg_path=exp_cfg_path,
                log_path=log_path,
                values={
                    "compare_model": compare_name,
                    "metric": str(metric_name),
                    "delta": stats["delta"],
                    "p_value": stats["p_value"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "notes": f"paired_bootstrap_n={int(bootstrap_rounds)} shared={int(stats['count'])}",
                },
            )
        )
    append_dict_rows_csv(revision_artifacts_csv, revision_headers, rows)
    return len(rows)


def refresh_significance_artifacts(
    exp_cfg: Dict[str, Any],
    args: argparse.Namespace,
    model_cfgs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
) -> None:
    """Refreshes paired-bootstrap revision rows from exported prediction CSV files."""
    eval_cfg = exp_cfg.get("evaluation", {}) if isinstance(exp_cfg.get("evaluation", {}), dict) else {}
    significance_cfg = eval_cfg.get("significance", {}) if isinstance(eval_cfg.get("significance", {}), dict) else {}
    if not bool(significance_cfg.get("enabled", False)):
        raise ValueError("evaluation.significance.enabled must be true for --refresh_significance.")
    outputs_cfg = exp_cfg.get("outputs", {}) if isinstance(exp_cfg.get("outputs", {}), dict) else {}
    base_results_dir = outputs_cfg.get("results_dir", os.path.join("Output", "results"))
    prediction_csv = outputs_cfg.get("prediction_csv", os.path.join(base_results_dir, "predictions.csv"))
    revision_artifacts_csv = outputs_cfg.get("revision_artifacts_csv", os.path.join(base_results_dir, "revision_artifacts.csv"))
    dataset_cfg = exp_cfg.get("dataset", {}) if isinstance(exp_cfg.get("dataset", {}), dict) else {}
    config_dataset_name = str(dataset_cfg.get("name", ""))
    dataset_name = str(args.dataset_override if args.dataset_override else config_dataset_name)
    source_dataset_name = str(args.source_dataset_override if args.source_dataset_override else dataset_name)
    run_kind = "transfer" if dataset_name != source_dataset_name else "main"
    reference_model = str(significance_cfg.get("reference_model", "")).lower()
    if len(reference_model) == 0:
        raise ValueError("evaluation.significance.reference_model must be configured for --refresh_significance.")
    metric_names = [str(metric) for metric in significance_cfg.get("metrics", ["macro_f1", "acc"])]
    bootstrap_rounds = int(significance_cfg.get("n_bootstrap", 1000))
    compare_models = [str(model_cfg["name"]).lower() for model_cfg in model_cfgs]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    revision_headers = revision_artifact_headers()
    written = 0
    for seed in seeds:
        for compare_model in compare_models:
            written += persist_significance_rows(
                prediction_csv=prediction_csv,
                revision_artifacts_csv=revision_artifacts_csv,
                revision_headers=revision_headers,
                timestamp=timestamp,
                dataset_name=dataset_name,
                source_dataset_name=source_dataset_name,
                run_kind=run_kind,
                reference_model=reference_model,
                compare_model=compare_model,
                seed=int(seed),
                metric_names=metric_names,
                bootstrap_rounds=bootstrap_rounds,
            )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info(
        f"Refreshed {written} significance rows for reference_model={reference_model}, "
        f"dataset={dataset_name}, source_dataset={source_dataset_name}."
    )


def run_multi_seed_jobs(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    """Runs one child process per seed to avoid mixed-state multi-seed execution."""
    for seed in seeds:
        cmd = [
            sys.executable,
            "-m",
            "Libs.exp.run",
            "--exp",
            args.exp,
            "--model",
            args.model,
            "--seeds",
            str(int(seed)),
        ]
        if args.eval_only:
            cmd.append("--eval_only")
        if args.ckpt:
            cmd.extend(["--ckpt", args.ckpt])
        if args.dataset_override:
            cmd.extend(["--dataset_override", args.dataset_override])
        if args.source_dataset_override:
            cmd.extend(["--source_dataset_override", args.source_dataset_override])
        if args.lp_edges_policy:
            cmd.extend(["--lp_edges_policy", args.lp_edges_policy])
        if getattr(args, "transfer_adapt", False):
            cmd.append("--transfer_adapt")
        if getattr(args, "sampling_budget_only", False):
            cmd.append("--sampling_budget_only")
        subprocess.run(cmd, check=True)
    exp_cfg = read_yaml_like(args.exp)
    outputs_cfg = exp_cfg.get("outputs", {})
    results_dir = outputs_cfg.get("results_dir", os.path.join("Output", "results"))
    dataset_cfg = exp_cfg.get("dataset", {})
    config_dataset_name = str(dataset_cfg.get("name", ""))
    dataset_name = str(args.dataset_override or config_dataset_name)
    source_dataset_name = str(args.source_dataset_override or dataset_name)
    model_paths = [p.strip() for p in args.model.split(",") if p.strip()]
    csv_path = os.path.join(results_dir, f"{dataset_name}.csv")
    for model_path in model_paths:
        model_cfg = read_yaml_like(model_path)
        refresh_result_aggregate_row(
            csv_path=csv_path,
            dataset_name=dataset_name,
            source_dataset_name=source_dataset_name,
            transfer_mode=("source_only" if dataset_name != source_dataset_name else "main"),
            model_name=str(model_cfg["name"]).lower(),
            model_cfg_path=model_path,
            exp_cfg_path=args.exp,
            seeds=seeds,
        )


def run_transfer_eval(exp_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Orchestrates transfer evaluation by spawning eval-only runs per pair/model/seed."""
    transfer_cfg = exp_cfg.get("transfer", {})
    transfer_pairs = parse_transfer_pair_overrides(args.transfer_pairs)
    if transfer_pairs is None:
        transfer_pairs = transfer_cfg.get("pairs", []) if isinstance(transfer_cfg, dict) else []
    if not isinstance(transfer_cfg, dict) or len(transfer_pairs) == 0:
        raise ValueError("Transfer config is missing pairs.")

    outputs_cfg = exp_cfg.get("outputs", {})
    base_ckpt_dir = outputs_cfg.get("ckpt_dir", os.path.join("Output", "ckpts"))
    results_dir = outputs_cfg.get("results_dir", os.path.join("Output", "results"))
    transfer_csv_default = os.path.join(results_dir, "transfer.csv")
    transfer_csv = exp_cfg.get("transfer", {}).get("results_path", transfer_csv_default)
    ckpt_name = str(transfer_cfg.get("ckpt_name", "best.pt"))
    lp_policy = str(transfer_cfg.get("lp_edges_policy", args.lp_edges_policy))
    seeds = parse_seeds(args.seeds, exp_cfg)

    transfer_models = transfer_cfg.get("models")
    if not transfer_models:
        transfer_models = [p.strip() for p in args.model.split(",") if p.strip()]
    model_cfgs = [read_yaml_like(p) for p in transfer_models]
    metric_only = bool(transfer_cfg.get("metric_only", False))
    skip_existing = bool(transfer_cfg.get("skip_existing", False))
    existing_transfer_rows: List[Dict[str, str]] = []
    if skip_existing:
        _, existing_transfer_rows = read_csv_rows(transfer_csv)
        existing_transfer_rows = dedupe_csv_rows(
            existing_transfer_rows,
            key_fields=["train_dataset", "test_dataset", "transfer_mode", "model", "seed"],
        )

    def has_existing_transfer_metric(source: str, target: str, model_name: str, seed: int) -> bool:
        for row in existing_transfer_rows:
            if row.get("train_dataset", "") != source:
                continue
            if row.get("test_dataset", "") != target:
                continue
            if row.get("transfer_mode", "") != "source_only":
                continue
            if row.get("model", "") != model_name:
                continue
            if row.get("seed", "") != str(int(seed)):
                continue
            return all(parse_numeric_cell(row.get(field, "")) is not None for field in ("acc", "macro_f1", "ece"))
        return False

    for pair in transfer_pairs:
        source = str(pair["source"])
        target = str(pair["target"])
        for model_path, model_cfg in zip(transfer_models, model_cfgs):
            model_name = str(model_cfg["name"]).lower()
            for seed in seeds:
                if skip_existing and has_existing_transfer_metric(source, target, model_name, int(seed)):
                    print(f"[transfer] skip existing {source}->{target} model={model_name} seed={seed}")
                    continue
                ckpt_path = os.path.join(
                    base_ckpt_dir,
                    source,
                    model_name,
                    f"seed{seed}",
                    ckpt_name,
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "Libs.exp.run",
                    "--exp",
                    args.exp,
                    "--model",
                    model_path,
                    "--seeds",
                    str(seed),
                    "--eval_only",
                    "--ckpt",
                    ckpt_path,
                    "--dataset_override",
                    target,
                    "--source_dataset_override",
                    source,
                    "--lp_edges_policy",
                    lp_policy,
                ]
                if getattr(args, "sampling_budget_only", False):
                    cmd.append("--sampling_budget_only")
                if metric_only:
                    cmd.append("--transfer_metric_only")
                print(f"[transfer] {source}->{target} model={model_name} seed={seed} ckpt={ckpt_path}")
                subprocess.run(cmd, check=True)
            refresh_transfer_aggregate_row(
                csv_path=transfer_csv,
                train_dataset=source,
                test_dataset=target,
                transfer_mode="source_only",
                model_name=model_name,
                seeds=seeds,
            )
            refresh_result_aggregate_row(
                csv_path=os.path.join(results_dir, f"{target}.csv"),
                dataset_name=target,
                source_dataset_name=source,
                transfer_mode="source_only",
                model_name=model_name,
                model_cfg_path=model_path,
                exp_cfg_path=args.exp,
                seeds=seeds,
            )


def run_transfer_adaptation(exp_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """Orchestrates target adaptation runs across configured transfer pairs."""
    transfer_cfg = exp_cfg.get("transfer", {})
    if not isinstance(transfer_cfg, dict) or len(transfer_cfg.get("pairs", [])) == 0:
        raise ValueError("Transfer config is missing pairs.")
    outputs_cfg = exp_cfg.get("outputs", {})
    base_ckpt_dir = outputs_cfg.get("ckpt_dir", os.path.join("Output", "ckpts"))
    results_dir = outputs_cfg.get("results_dir", os.path.join("Output", "results"))
    transfer_csv_default = os.path.join(results_dir, "transfer.csv")
    transfer_csv = exp_cfg.get("transfer", {}).get("results_path", transfer_csv_default)
    ckpt_name = str(transfer_cfg.get("ckpt_name", "best.pt"))
    lp_policy = str(transfer_cfg.get("lp_edges_policy", args.lp_edges_policy))
    seeds = parse_seeds(args.seeds, exp_cfg)
    transfer_models = transfer_cfg.get("models")
    if not transfer_models:
        transfer_models = [p.strip() for p in args.model.split(",") if p.strip()]
    model_cfgs = [read_yaml_like(p) for p in transfer_models]
    for pair in transfer_cfg.get("pairs", []):
        source = str(pair["source"])
        target = str(pair["target"])
        for model_path, model_cfg in zip(transfer_models, model_cfgs):
            model_name = str(model_cfg["name"]).lower()
            for seed in seeds:
                ckpt_path = os.path.join(
                    base_ckpt_dir,
                    source,
                    model_name,
                    f"seed{seed}",
                    ckpt_name,
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "Libs.exp.run",
                    "--exp",
                    args.exp,
                    "--model",
                    model_path,
                    "--seeds",
                    str(seed),
                    "--transfer_adapt",
                    "--ckpt",
                    ckpt_path,
                    "--dataset_override",
                    target,
                    "--source_dataset_override",
                    source,
                    "--lp_edges_policy",
                    lp_policy,
                ]
                print(f"[transfer_adapt] {source}->{target} model={model_name} seed={seed} ckpt={ckpt_path}")
                subprocess.run(cmd, check=True)
                transfer_adapt_cfg = transfer_cfg.get("adaptation", {}) if isinstance(transfer_cfg.get("adaptation", {}), dict) else {}
                adapt_ckpt_name = str(transfer_adapt_cfg.get("adapt_ckpt_name", "best_target_adapt.pt"))
                adapt_ckpt_path = os.path.join(
                    base_ckpt_dir,
                    source,
                    model_name,
                    f"seed{seed}",
                    adapt_ckpt_name,
                )
                if not os.path.isfile(adapt_ckpt_path):
                    raise RuntimeError(
                        f"Target adaptation child run did not create checkpoint: {adapt_ckpt_path}"
                    )
            refresh_transfer_aggregate_row(
                csv_path=transfer_csv,
                train_dataset=source,
                test_dataset=target,
                transfer_mode="target_adapt",
                model_name=model_name,
                seeds=seeds,
            )
            refresh_result_aggregate_row(
                csv_path=os.path.join(results_dir, f"{target}.csv"),
                dataset_name=target,
                source_dataset_name=source,
                transfer_mode="target_adapt",
                model_name=model_name,
                model_cfg_path=model_path,
                exp_cfg_path=args.exp,
                seeds=seeds,
            )


def build_model(
    name: str,
    cfg: Dict[str, Any],
    num_features: int,
    num_classes: int,
    text_channels: int = 0,
):
    name = name.lower()
    if name == "textonly":
        m = TextOnlyClassifier(
            in_channels=int(
                cfg.get("text_dim", cfg.get("hidden", num_features))),
            num_classes=num_classes,
            hidden=int(cfg.get("hidden", 256)),
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "dga_fake":
        cfg_struct = DGAFakeConfig.from_dict(cfg)
        m = DGAFakeModel(cfg_struct, num_classes=num_classes)
        return m
    if name == "agad":
        m = AGAD(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            heads=int(cfg["heads"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
            projection_dim=int(cfg["projection_dim"]),
            contrastive_weight=float(cfg["contrastive_weight"]),
            prototype_weight=float(cfg["prototype_weight"]),
            adversarial_weight=float(cfg["adversarial_weight"]),
            adversarial_epsilon=float(cfg["adversarial_epsilon"]),
            contrastive_temperature=float(cfg["contrastive_temperature"]),
            prototype_temperature=float(cfg["prototype_temperature"]),
            prototype_logit_weight=float(cfg["prototype_logit_weight"]),
        )
        return m
    if name == "hpcl_cascade":
        configured_text_channels = int(cfg["text_dim"])
        resolved_text_channels = text_channels if configured_text_channels <= 0 else configured_text_channels
        m = HPCLCascade(
            in_channels=num_features,
            text_channels=int(resolved_text_channels),
            hidden_channels=int(cfg["hidden"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
            projection_dim=int(cfg["projection_dim"]),
            prototypes_per_class=int(cfg["prototypes_per_class"]),
            contrastive_weight=float(cfg["contrastive_weight"]),
            prototype_weight=float(cfg["prototype_weight"]),
            contrastive_temperature=float(cfg["contrastive_temperature"]),
            prototype_temperature=float(cfg["prototype_temperature"]),
            prototype_logit_weight=float(cfg["prototype_logit_weight"]),
            curriculum_warmup_epochs=int(cfg["curriculum_warmup_epochs"]),
            curriculum_min_weight=float(cfg["curriculum_min_weight"]),
            use_text=bool(cfg["use_text"]),
        )
        return m
    if name == "tpso":
        m = TPSO(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
            temporal_decay=float(cfg["temporal_decay"]),
            readout_decay=float(cfg["readout_decay"]),
        )
        return m
    if name == "sage":
        m = GraphSAGE(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "appnp":
        m = APPNPNet(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
            appnp_alpha=float(cfg.get("alpha", 0.1)),
            appnp_k=int(cfg.get("k", 10)),
        )
        return m
    if name == "gcn2":
        m = GCN2Net(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 4)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
            alpha=float(cfg.get("alpha", 0.1)),
            theta=float(cfg.get("theta", 0.5)),
        )
        return m
    if name == "gatv2":
        m = GATv2(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            heads=int(cfg.get("heads", 4)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "temporalgcn":
        m = TemporalGCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "root_pool":
        m = RootPooling(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "gcn":
        m = GCN(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
        )
        return m
    if name == "ctdm_syn":
        m = CTDMSyn(
            in_channels=num_features,
            num_classes=num_classes,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            dropout=float(cfg.get("dropout", 0.3)),
            T=int(cfg.get("T", 200)),
            obs_minutes=int(cfg.get("obs_minutes", 30)),
            t_embed_dim=int(cfg.get("t_embed_dim", cfg.get("hidden", 128))),
            film_hidden=int(cfg.get("film_hidden", 128)),
            max_nodes=int(cfg["max_nodes"]),
        )
        # Optional inference stride
        if "t_stride" in cfg:
            setattr(m, "t_stride", int(cfg["t_stride"]))
        if "sampling_eta" in cfg:
            m.sampling_eta = float(cfg["sampling_eta"])
        return m
    if name == "gat":
        m = GAT(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            heads=int(cfg["heads"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
        )
        return m
    if name == "tgat":
        m = TGAT(
            in_channels=num_features,
            hidden_channels=int(cfg["hidden"]),
            heads=int(cfg["heads"]),
            num_layers=int(cfg["num_layers"]),
            num_classes=num_classes,
            dropout=float(cfg["dropout"]),
            time_normalization=str(cfg["time_normalization"]),
        )
        return m
    if name == "bigcn":
        m = BiGCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "resgcn":
        m = ResGCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    if name == "lp_gcn":
        # LP+GCN is just GCN with LP edges enabled
        m = GCN(
            in_channels=num_features,
            hidden_channels=int(cfg.get("hidden", 128)),
            num_layers=int(cfg.get("num_layers", 3)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.3)),
        )
        return m
    raise ValueError(f"Unknown model name: {name}")


def setup_logging(log_dir: str, dataset: str, model_name: str, seed: int | None = None) -> str:
    os.makedirs(log_dir, exist_ok=True)
    job_name = os.environ.get("SLURM_JOB_NAME", "").strip()
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if job_name and job_id:
        log_filename = f"{job_name}_{job_id}.log"
    else:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        seed_suffix = f"seed{int(seed)}" if seed is not None else "seedna"
        log_filename = f"{dataset}_{model_name}_{seed_suffix}_{stamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger()
    for h in list(logger.handlers):
        if hasattr(h, "flush"):
            h.flush()
        if hasattr(h, "close"):
            h.close()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return log_path


def resolve_canonical_output_cfg(exp_cfg: Dict[str, Any]) -> Dict[str, Any]:
    outputs_cfg = exp_cfg.get("outputs", {}) if isinstance(exp_cfg, dict) else {}
    canonical_cfg = outputs_cfg.get("canonical", {}) if isinstance(outputs_cfg, dict) else {}
    if not isinstance(canonical_cfg, dict) or not bool(canonical_cfg.get("enabled", False)):
        return {}
    return canonical_cfg


def resolve_canonical_stage_seeds(exp_cfg: Dict[str, Any], fallback_seed: int) -> Dict[str, int]:
    experiment_cfg = exp_cfg.get("experiment", {}) if isinstance(exp_cfg, dict) else {}
    seeds_cfg = experiment_cfg.get("seeds", []) if isinstance(experiment_cfg, dict) else []
    split_seed = ctdm_seed = arbiter_seed = inference_seed = int(fallback_seed)
    if isinstance(seeds_cfg, (list, tuple)) and len(seeds_cfg) >= 5:
        bootstrap_seed = int(seeds_cfg[4])
    else:
        bootstrap_seed = int(fallback_seed)
    missingness_seed = int(bootstrap_seed)
    return {
        "split": split_seed,
        "ctdm": ctdm_seed,
        "arbiter": arbiter_seed,
        "inference": inference_seed,
        "bootstrap": bootstrap_seed,
        "missingness": missingness_seed,
    }


def resolve_canonical_horizon(prefix_cfg: Dict[str, Any]) -> Dict[str, Any]:
    unit = str(prefix_cfg.get("unit", prefix_cfg.get("strategy", "minutes"))).lower()
    value = prefix_cfg.get("window", prefix_cfg.get("minutes", prefix_cfg.get("k", prefix_cfg.get("horizon_value", 0))))
    if unit not in {"minutes", "hop", "count"}:
        raise ValueError(f"Unsupported canonical horizon unit: {unit}")
    if float(value) <= 0:
        raise ValueError("Canonical horizon value must be positive")
    return {"unit": unit, "value": float(value)}


def main() -> None:
    parser = argparse.ArgumentParser(description="D2Plus baseline runner")
    parser.add_argument("--exp", type=str, required=True,
                        help="Path to exp.yaml")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to baseline model yaml")
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma separated seeds to override config (e.g., 42,43,44)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and only run evaluation using provided checkpoint(s)")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint path override for eval_only or transfer runs")
    parser.add_argument("--dataset_override", type=str, default="",
                        help="Override dataset name from exp.yaml for transfer evaluation")
    parser.add_argument("--source_dataset_override", type=str, default="",
                        help="Override dataset.name for transfer source dataset")
    parser.add_argument("--transfer_eval", action="store_true",
                        help="Run transfer evaluation across dataset pairs defined in exp transfer config")
    parser.add_argument("--transfer_pairs", type=str, default="",
                        help="Comma-separated source:target transfer pairs to run instead of exp transfer pairs")
    parser.add_argument("--transfer_adapt", action="store_true",
                        help="Run target adaptation transfer evaluation across dataset pairs or for a single child run")
    parser.add_argument("--lp_edges_policy", type=str, default="auto",
                        choices=["auto", "require_existing"],
                        help="LP edges handling: auto-generate when missing, or require existing edges")
    parser.add_argument("--refresh_significance", action="store_true",
                        help="Refresh paired-bootstrap revision artifact rows from exported predictions")
    parser.add_argument("--sampling_budget_only", action="store_true",
                        help="Refresh stochastic sampling-budget revision rows after loading checkpoints")
    parser.add_argument("--transfer_metric_only", action="store_true",
                        help="For source-only transfer child runs, stop after exporting main transfer metrics")
    args = parser.parse_args()

    exp_cfg = read_yaml_like(args.exp)
    model_paths: List[str] = [p.strip() for p in args.model.split(",")]
    model_cfgs = [read_yaml_like(p) for p in model_paths]
    seeds = parse_seeds(args.seeds, exp_cfg)
    train_max_samples = int(exp_cfg.get("training", {}).get("max_samples", 0))

    if args.transfer_eval:
        run_transfer_eval(exp_cfg, args)
        return
    if args.transfer_adapt and not args.ckpt:
        run_transfer_adaptation(exp_cfg, args)
        return
    if args.refresh_significance:
        refresh_significance_artifacts(exp_cfg, args, model_cfgs, seeds)
        return
    if len(seeds) > 1:
        run_multi_seed_jobs(args, seeds)
        return

    dataset_cfg = exp_cfg["dataset"]
    config_dataset_name = str(dataset_cfg["name"])
    dataset_name = str(args.dataset_override if args.dataset_override else config_dataset_name)
    source_dataset_name = str(args.source_dataset_override if args.source_dataset_override else dataset_name)
    dataset_root = os.path.join(dataset_cfg["root"], dataset_name)
    source_dataset_root = os.path.join(dataset_cfg["root"], source_dataset_name)
    transfer_run = bool(dataset_name != source_dataset_name)
    text_runtime_cfg = resolve_dataset_text_runtime_cfg(exp_cfg)
    ensure_text_embedding_cache(dataset_name, text_runtime_cfg)
    if source_dataset_name != dataset_name:
        ensure_text_embedding_cache(source_dataset_name, text_runtime_cfg)
    use_lp_edges = bool(dataset_cfg.get("use_lp_edges", False))
    lp_edges_policy = str(args.lp_edges_policy)
    prefix_cfg = resolve_dataset_prefix_cfg(dataset_cfg.get("prefix", {}), dataset_name)
    prefix_cfg["_runtime_mode"] = str(dataset_cfg.get("mode", "legacy"))
    prefix_cfg["_runtime_modes"] = dict(dataset_cfg.get("modes", {})) if isinstance(dataset_cfg.get("modes", {}), dict) else {}
    prefix_cfg["_require_lp_edges"] = bool(
        dataset_cfg.get("require_lp_edges", False)
        or (prefix_cfg["_runtime_mode"] != "legacy" and use_lp_edges)
    )
    observability_cfg = dataset_cfg.get("observability", {})
    observability_sweep_cfg = dataset_cfg.get("observability_sweep", {})
    lp_cfg_dict = exp_cfg.get("link_prediction", {})
    outputs_cfg = exp_cfg.get("outputs", {})
    canonical_outputs_cfg = resolve_canonical_output_cfg(exp_cfg)
    base_ckpt_dir = outputs_cfg.get(
        "ckpt_dir", os.path.join("Output", "ckpts"))
    base_log_dir = outputs_cfg.get("log_dir", os.path.join("Output", "logs"))
    base_results_dir = outputs_cfg.get(
        "results_dir", os.path.join("Output", "results"))
    sweeps_csv = os.path.join(base_results_dir, "rd_sweeps.csv")
    transfer_csv_default = os.path.join(base_results_dir, "transfer.csv")
    transfer_csv = exp_cfg.get("transfer", {}).get("results_path", transfer_csv_default)
    prediction_csv = outputs_cfg.get(
        "prediction_csv", os.path.join(base_results_dir, "predictions.csv"))
    revision_artifacts_csv = outputs_cfg.get(
        "revision_artifacts_csv", os.path.join(base_results_dir, "revision_artifacts.csv"))
    revision_headers = revision_artifact_headers()

    # Device configuration
    device_cfg = exp_cfg.get("device", {})
    force_cpu = device_cfg.get("force_cpu", False)
    if force_cpu:
        device = "cpu"
    else:
        requested_device = str(
            device_cfg.get("type", "cuda" if torch.cuda.is_available() else "cpu")
        ).lower()
        device = "cpu" if requested_device == "cuda" and not torch.cuda.is_available() else requested_device

    # Set CUDA device if specified
    if device == "cuda" and torch.cuda.is_available():
        gpu_id = device_cfg.get("gpu_id", 0)
        torch.cuda.set_device(gpu_id)
        logging.info(f"Using CUDA device {gpu_id}")
    else:
        logging.info(f"Using device: {device}")

    ds_train = make_graph_dataset(
        root=dataset_root,
        split="train",
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
        text_runtime_cfg=text_runtime_cfg,
    )
    ds_val = make_graph_dataset(
        root=dataset_root,
        split="val",
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
        text_runtime_cfg=text_runtime_cfg,
    )
    ds_test = make_graph_dataset(
        root=dataset_root,
        split="test",
        use_lp_edges=use_lp_edges,
        prefix_cfg=prefix_cfg,
        observability_cfg=observability_cfg,
        text_runtime_cfg=text_runtime_cfg,
    )

    num_features = ds_train.num_features
    num_classes = ds_train.num_classes
    dataset_label_mapping = resolve_dataset_label_mapping(source_dataset_name)

    os.makedirs(base_results_dir, exist_ok=True)
    csv_path = os.path.join(base_results_dir, f"{dataset_name}.csv")
    sweeps_headers = [
        "timestamp",
        "dataset",
        "source_dataset",
        "run_kind",
        "model",
        "model_cfg",
        "exp_cfg",
        "log_path",
        "seed",
        "tag",
        "observation_window",
        "observation_unit",
        "obs_minutes",
        "drop_node",
        "drop_edge",
        "edge_drop_rate",
        "text_noise_std",
        "text_dropout",
        "train_seconds",
        "eval_seconds",
        "loss",
        "nll",
        "acc",
        "macro_f1",
        "ece",
        "calibration_temp",
    ]
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ckpt_filename = str(exp_cfg.get("transfer", {}).get("ckpt_name", "best.pt"))

    for model_cfg, model_path in zip(model_cfgs, model_paths):
        name = model_cfg["name"].lower()
        is_rumordiffusion = name == "rumordiffusion" or name.startswith("rumordiffusion_")
        active_training_cfg = _deep_merge(
            exp_cfg.get("training", {}),
            model_cfg.get("training", {}) if isinstance(model_cfg.get("training", {}), dict) else {},
        )
        train_max_samples = int(active_training_cfg.get("max_samples", 0))
        lp_forced = name == "lp_gcn"
        use_lp = True if lp_forced else use_lp_edges
        robust_text_cfg: Dict[str, Any] = {}  # default to avoid undefined access
        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            set_global_seed(int(seed))

            ds_train = make_graph_dataset(
                root=dataset_root,
                split="train",
                use_lp_edges=use_lp,
                prefix_cfg=prefix_cfg,
                observability_cfg=observability_cfg,
                text_runtime_cfg=text_runtime_cfg,
            )
            ds_val = make_graph_dataset(
                root=dataset_root,
                split="val",
                use_lp_edges=use_lp,
                prefix_cfg=prefix_cfg,
                observability_cfg=observability_cfg,
                text_runtime_cfg=text_runtime_cfg,
            )
            ds_test = make_graph_dataset(
                root=dataset_root,
                split="test",
                use_lp_edges=use_lp,
                prefix_cfg=prefix_cfg,
                observability_cfg=observability_cfg,
                text_runtime_cfg=text_runtime_cfg,
            )

            num_features = ds_train.num_features
            num_classes = ds_train.num_classes
            text_channels = infer_text_channels(ds_train)

            log_path = setup_logging(
                log_dir=base_log_dir, dataset=dataset_name, model_name=name, seed=int(seed))
            run_started_at = datetime.now().astimezone().isoformat()
            logging.info(
                f"Dataset: {dataset_name}; Source dataset: {source_dataset_name}; Model: {name}; LP: {use_lp}; Seed: {seed}")

            lp_require_existing = (lp_edges_policy == "require_existing" and name == "lp_gcn")
            # Optional pre-step: if LP is required but lp_edges are missing, train LP and generate edges
            if use_lp:
                lp_dir = os.path.join(dataset_root, "lp_edges")
                need_generate = not os.path.isdir(lp_dir) or len(
                    [p for p in os.listdir(lp_dir) if p.endswith('.pt')]) == 0
                if lp_require_existing and need_generate:
                    raise ValueError(f"LP edges required for LP-GCN transfer, but missing in {lp_dir}")
                if not lp_require_existing and need_generate:
                    logging.info(
                        "LP edges missing; training link predictor and generating edges...")
                    # Build TGN from model_cfg if provided; else from lp_cfg_dict
                    tgn_hidden = int(lp_cfg_dict.get(
                        "hidden", model_cfg["model"].get("hidden", 128)))
                    tgn_layers = int(lp_cfg_dict.get(
                        "num_layers", model_cfg["model"].get("num_layers", 3)))
                    tgn_heads = int(lp_cfg_dict.get(
                        "heads", model_cfg["model"].get("heads", 2)))
                    tgn_dropout = float(lp_cfg_dict.get(
                        "dropout", model_cfg["model"].get("dropout", 0.3)))

                    tgn = TGN(
                        in_channels=num_features,
                        hidden_channels=tgn_hidden,
                        num_layers=tgn_layers,
                        heads=tgn_heads,
                        dropout=tgn_dropout,
                    )

                    lp_cfg = LinkPredConfig(
                        epochs=int(lp_cfg_dict.get(
                            "epochs", exp_cfg["training"].get("epochs", 20))),
                        learning_rate=float(lp_cfg_dict.get(
                            "lr", exp_cfg["training"].get("lr", 3e-4))),
                        weight_decay=float(lp_cfg_dict.get(
                            "weight_decay", exp_cfg["training"].get("weight_decay", 0.0))),
                        batch_size=int(lp_cfg_dict.get(
                            "batch_size", exp_cfg["training"].get("batch_size", 1))),
                        grad_clip=float(lp_cfg_dict.get(
                            "grad_clip", exp_cfg["training"].get("grad_clip", 1.0))),
                        amp=bool(lp_cfg_dict.get(
                            "amp", exp_cfg["training"].get("amp", True))),
                        early_stopping_patience=int(lp_cfg_dict.get(
                            "early_stopping_patience", exp_cfg["training"].get("early_stopping_patience", 15))),
                        num_workers=int(lp_cfg_dict.get(
                            "num_workers", exp_cfg["training"].get("num_workers", 0))),
                        device=device,
                        heads=tgn_heads,
                        hidden=tgn_hidden,
                        num_layers=tgn_layers,
                        dropout=tgn_dropout,
                        topk_per_node=int(lp_cfg_dict.get("topk_per_node", 2)),
                    )

                    def make_loader(ds: GraphDataset, shuffle: bool) -> DataLoader:
                        return DataLoader(
                            ds,
                            batch_size=lp_cfg.batch_size,
                            shuffle=shuffle,
                            num_workers=lp_cfg.num_workers,
                            pin_memory=(device == "cuda"),
                            persistent_workers=lp_cfg.num_workers > 0,
                        )

                    lp_trainer = LinkPredTrainer(model=tgn, cfg=lp_cfg)
                    lp_ckpt_dir = os.path.join(
                        "Output", "ckpts", dataset_name, "tgn", f"seed{seed}")
                    lp_ckpt_path = os.path.join(lp_ckpt_dir, "best.pt")
                    lp_trainer.fit(make_loader(ds_train, shuffle=True), save_path=lp_ckpt_path,
                                   epochs=lp_cfg.epochs, early_stopping_patience=lp_cfg.early_stopping_patience)

                    for split_name, ds_split in ("train", ds_train), ("val", ds_val), ("test", ds_test):
                        LinkPredTrainer.generate_edges_for_split(
                            lp_trainer.model, ds_split, topk_per_node=lp_cfg.topk_per_node, out_dir=lp_dir, device=torch.device(device), num_workers=lp_cfg.num_workers)

                    # Reload datasets with new LP edges
                    ds_train = make_graph_dataset(
                        root=dataset_root,
                        split="train",
                        use_lp_edges=use_lp,
                        prefix_cfg=prefix_cfg,
                        observability_cfg=observability_cfg,
                        text_runtime_cfg=text_runtime_cfg,
                    )
                    ds_val = make_graph_dataset(
                        root=dataset_root,
                        split="val",
                        use_lp_edges=use_lp,
                        prefix_cfg=prefix_cfg,
                        observability_cfg=observability_cfg,
                        text_runtime_cfg=text_runtime_cfg,
                    )
                    ds_test = make_graph_dataset(
                        root=dataset_root,
                        split="test",
                        use_lp_edges=use_lp,
                        prefix_cfg=prefix_cfg,
                        observability_cfg=observability_cfg,
                        text_runtime_cfg=text_runtime_cfg,
                    )
                    num_features = ds_train.num_features
                    num_classes = ds_train.num_classes

            # If CTDM-syn, scan max nodes from train split and cap at configured max_nodes, then inject into model config
            if name == "ctdm_syn":
                max_nodes_scan = scan_max_nodes_across_splits([ds_train, ds_val, ds_test])
                max_nodes_config = int(model_cfg["model"].get("max_nodes", 0) or 0)
                max_nodes_final = resolve_max_nodes_setting(
                    scanned_max_nodes=max_nodes_scan,
                    configured_max_nodes=max_nodes_config,
                    context="CTDM-syn",
                )
                model_cfg["model"]["max_nodes"] = int(max_nodes_final)
                logging.info(
                    f"CTDM-syn: scanned max_nodes={max_nodes_scan}, resolved to {max_nodes_final} "
                    f"(config value: {max_nodes_config})")

            # Build model for classification or generative baseline
            m = None
            class_weights = None
            model_use_potentials = False
            model_potentials_dir: Optional[str] = None
            model_potentials_root: Optional[str] = None
            model_potentials_pair: List[int] = []
            model_potential_cache_metadata: Optional[Dict[str, Any]] = None
            model_potential_cache_ctdm_payload: Optional[Dict[str, Any]] = None
            rd_prefix_consistency_cfg: Dict[str, Any] = {}
            rd_prefix_consistency_views: List[Dict[str, Any]] = []
            rd_prefix_consistency_potential_specs: List[Dict[str, Any]] = []
            rd_source_robustness_cfg: Dict[str, Any] = {}
            rd_ignored_metadata_keys: Optional[List[str]] = None
            if is_rumordiffusion:
                # Stage 1: train CTDM if needed; then generate potentials
                logging.info("Running RumorDiffusion two-stage training...")

                # 1) Scan max_nodes from train split
                rd_cfg_exp = exp_cfg.get("rumordiffusion", {})
                rd_cfg_model = model_cfg.get("model", {})
                rd_cfg = _deep_merge(rd_cfg_model, rd_cfg_exp)
                rd_variant_overrides = model_cfg.get("rumordiffusion_overrides", {})
                if isinstance(rd_variant_overrides, dict) and len(rd_variant_overrides) > 0:
                    rd_cfg = _deep_merge(rd_cfg, rd_variant_overrides)
                rd_cfg = apply_dataset_overrides(rd_cfg, dataset_name)
                rd_cfg = validate_rumordiffusion_pair_config(rd_cfg, dataset_name, int(num_classes))
                ctdm_cfg = rd_cfg.get("ctdm", {})
                checkpoint_cfg = rd_cfg.get("checkpoint", {}) if isinstance(rd_cfg.get("checkpoint", {}), dict) else {}
                validate_ckpt_metadata = bool(checkpoint_cfg.get("validate_metadata", False))
                require_config_hash = bool(checkpoint_cfg.get("require_config_hash", True))
                allow_legacy_ckpt = bool(checkpoint_cfg.get("allow_legacy", True))
                rd_ignored_metadata_keys = rumordiffusion_metadata_ignored_keys(
                    require_config_hash=require_config_hash,
                    transfer_run=transfer_run,
                )
                ctdm_ckpt_root = str(checkpoint_cfg.get("ctdm_dir", base_ckpt_dir))
                ctdm_subdir = str(checkpoint_cfg.get("ctdm_subdir", "rumordiffusion_ctdm"))
                use_potentials = bool(rd_cfg.get("use_potentials", False))
                potentials_pair = resolve_rumordiffusion_potentials_pair(
                    configured_pair=rd_cfg.get("potentials_pair", list(range(int(num_classes)))),
                    num_classes=int(num_classes),
                    dataset_name=dataset_name,
                )
                potentials_dir_cfg = str(rd_cfg.get("potentials_dir", "rd_potentials"))
                potentials_root = potentials_dir_cfg if os.path.isabs(potentials_dir_cfg) else os.path.join(dataset_root, potentials_dir_cfg)
                if bool(rd_cfg.get("seed_scoped_potentials", False)):
                    potentials_root = os.path.join(potentials_root, f"seed{int(seed)}")
                cache_potentials_after_ctdm = bool(rd_cfg.get("cache_potentials_after_ctdm", False))
                if cache_potentials_after_ctdm:
                    use_potentials = True
                if args.sampling_budget_only:
                    use_potentials = False
                model_potentials_pair = list(potentials_pair)
                rd_prefix_consistency_cfg = active_training_cfg.get("prefix_consistency", {})
                rd_prefix_consistency_cfg = rd_prefix_consistency_cfg if isinstance(rd_prefix_consistency_cfg, dict) else {}
                rd_source_robustness_cfg = active_training_cfg.get("source_robustness", {})
                rd_source_robustness_cfg = rd_source_robustness_cfg if isinstance(rd_source_robustness_cfg, dict) else {}
                rd_prefix_consistency_views = resolve_prefix_consistency_views(
                    rd_prefix_consistency_cfg,
                    dataset_name,
                    prefix_cfg,
                )
                ctdm_scan_train_ds = make_graph_dataset(
                    root=dataset_root,
                    split="train",
                    use_lp_edges=use_lp,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                    text_runtime_cfg=text_runtime_cfg,
                    mode="train_ctdm",
                )
                ctdm_scan_val_ds = make_graph_dataset(
                    root=dataset_root,
                    split="val",
                    use_lp_edges=use_lp,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                    text_runtime_cfg=text_runtime_cfg,
                    mode="train_ctdm",
                )
                max_nodes_scan = scan_max_nodes_across_splits([ctdm_scan_train_ds, ctdm_scan_val_ds, ds_test])
                max_nodes_config = int(ctdm_cfg.get("max_nodes", 0) or 0)
                max_nodes_final = resolve_max_nodes_setting(
                    scanned_max_nodes=max_nodes_scan,
                    configured_max_nodes=max_nodes_config,
                    context="RumorDiffusion CTDM",
                )
                logging.info(
                    f"RumorDiffusion CTDM: scanned max_nodes={max_nodes_scan}, resolved to {max_nodes_final} "
                    f"(config value: {max_nodes_config})")
                model_potential_cache_ctdm_payload = {
                    "ctdm": _deep_merge(ctdm_cfg, {"max_nodes": int(max_nodes_final)}),
                    "text": rd_cfg.get("text", {}),
                    "prefix": prefix_cfg,
                    "observability": observability_cfg,
                }
                model_potential_cache_metadata = build_potential_cache_metadata(
                    dataset_name=dataset_name,
                    seed=int(seed),
                    prefix_cfg=prefix_cfg,
                    ctdm_config_payload=model_potential_cache_ctdm_payload,
                )
                potentials_dir = potential_cache_dir(potentials_root, prefix_cfg)
                rd_prefix_consistency_potential_specs = build_prefix_consistency_potential_specs(
                    views=rd_prefix_consistency_views,
                    potentials_root=potentials_root,
                    dataset_name=dataset_name,
                    seed=int(seed),
                    ctdm_config_payload=model_potential_cache_ctdm_payload,
                    potentials_pair=potentials_pair,
                )
                model_use_potentials = use_potentials
                model_potentials_root = potentials_root
                model_potentials_dir = potentials_dir

                # 2) Build CTDM
                bb_cfg = ctdm_cfg.get("backbone", {}) if isinstance(
                    ctdm_cfg.get("backbone", {}), dict) else {}
                # Optionally compute degree histogram for PNA
                deg_histogram = None
                if str(bb_cfg.get("name", "gcn")).lower() == "pna":
                    logging.info("Computing degree histogram for PNA...")
                    from torch_geometric.utils import degree as pyg_degree
                    deg_counter = collections.Counter()
                    for raw_idx in range(len(ds_train)):
                        data = ds_train.get(raw_idx)
                        if hasattr(data, "edge_index") and data.edge_index is not None and int(data.edge_index.numel()) > 0:
                            deg = pyg_degree(data.edge_index[1].to(
                                torch.long), num_nodes=int(data.num_nodes), dtype=torch.long)
                            deg_counter.update([int(v.item()) for v in deg])
                    max_degree = max(deg_counter.keys()) if len(
                        deg_counter) > 0 else 0
                    deg_hist = torch.zeros(max_degree + 1, dtype=torch.long)
                    for d, c in deg_counter.items():
                        if d <= max_degree:
                            deg_hist[d] = int(c)
                    deg_histogram = deg_hist
                    logging.info(
                        f"Degree histogram computed: max_deg={int(deg_histogram.argmax().item())}")

                # Optional text fusion from exp config
                text_cfg = rd_cfg.get("text", {})
                text_enabled = bool(text_cfg.get("enabled", False)) if isinstance(
                    text_cfg, dict) else False
                text_dim = int(text_cfg.get("text_dim", 0)) if text_enabled else 0
                text_fuse = str(text_cfg.get("fuse_mode", "add")
                                ) if text_enabled else "add"
                robust_text_cfg = text_cfg.get(
                    "robust_text", {}) if isinstance(text_cfg, dict) else {}

                ctdm = CTDMSyn(
                    in_channels=num_features,
                    num_classes=num_classes,
                    hidden_channels=int(ctdm_cfg.get("hidden", 128)),
                    num_layers=int(ctdm_cfg.get("num_layers", 3)),
                    dropout=float(ctdm_cfg.get("dropout", 0.3)),
                    T=int(ctdm_cfg.get("T", 200)),
                    obs_minutes=int(ctdm_cfg.get("obs_minutes", 30)),
                    t_embed_dim=int(ctdm_cfg.get(
                        "t_embed_dim", ctdm_cfg.get("hidden", 128))),
                    film_hidden=int(ctdm_cfg.get("film_hidden", 128)),
                    max_nodes=int(max_nodes_final),
                    backbone_name=str(bb_cfg.get("name", "gcn")).lower(),
                    backbone_heads=int(bb_cfg.get("heads", 1)),
                    backbone_dropout=float(bb_cfg.get(
                        "dropout", ctdm_cfg.get("dropout", 0.3))),
                    deg_histogram=deg_histogram,
                    pna_aggregators=bb_cfg.get("aggregators", None),
                    pna_scalers=bb_cfg.get("scalers", None),
                    text_dim=(text_dim if text_dim > 0 else None),
                    text_fuse=text_fuse,
                )
                potential_adapter_cfg = ctdm_cfg.get("potential_adapter", {})
                if not isinstance(potential_adapter_cfg, dict):
                    raise TypeError("rumordiffusion.ctdm.potential_adapter must be a mapping")
                potential_adapter_enabled = bool(potential_adapter_cfg.get("enabled", False))
                potential_adapter_bottleneck = int(
                    potential_adapter_cfg.get("bottleneck_dim", max(1, ctdm.hidden_channels // 4))
                )
                ctdm.configure_potential_adapter(
                    enabled=potential_adapter_enabled,
                    bottleneck_dim=potential_adapter_bottleneck,
                )
                if potential_adapter_enabled:
                    logging.info(
                        "RD-CTDM post-readout potential adapter enabled: "
                        f"bottleneck_dim={potential_adapter_bottleneck}"
                    )
                if isinstance(robust_text_cfg, dict):
                    noise_list = robust_text_cfg.get("noise_std", [])
                    drop_list = robust_text_cfg.get("dropout_p", [])
                    if len(noise_list) > 0:
                        ctdm.text_noise_std = float(noise_list[0])
                    if len(drop_list) > 0:
                        ctdm.text_dropout_p = float(drop_list[0])
                if "t_stride" in ctdm_cfg:
                    setattr(ctdm, "t_stride", int(ctdm_cfg["t_stride"]))
                # Inject CTDM auxiliary configs
                if "parameterization" in ctdm_cfg:
                    ctdm.parameterization = str(
                        ctdm_cfg["parameterization"]).lower()
                # Defaults per plan: guidance_p=0.2, guidance_scale=2.0
                ctdm.guidance_p = float(ctdm_cfg.get("guidance_p", 0.2))
                ctdm.guidance_scale = float(ctdm_cfg.get("guidance_scale", 2.0))
                ctdm.sampling_eta = float(ctdm_cfg.get("sampling_eta", 0.0))
                logging.info(
                    f"RD-CTDM cfg -> T={ctdm.T}, stride={getattr(ctdm, 't_stride', 1)}, "
                    f"guidance_scale={ctdm.guidance_scale}, guidance_p={ctdm.guidance_p}, "
                    f"sampling_eta={ctdm.sampling_eta}"
                )
                if "use_attn_readout" in ctdm_cfg:
                    ctdm.use_attn_readout = bool(ctdm_cfg["use_attn_readout"])
                # Multi-window training block
                mw_cfg = ctdm_cfg.get("multiwindow", {})
                if isinstance(mw_cfg, dict):
                    # Defaults per plan: enabled=True, min=10, max=30
                    ctdm.multiwindow_enabled = bool(mw_cfg.get("enabled", True))
                    ctdm.cutoff_min = int(mw_cfg.get("min", 10))
                    ctdm.cutoff_max = int(mw_cfg.get("max", 30))
                # Sparse candidate pairs block
                sp_cfg = ctdm_cfg.get("sparse_pairs", {})
                if isinstance(sp_cfg, dict):
                    # Defaults per plan: enabled=True, strategy=b, topk=8
                    ctdm.sparse_pairs_enabled = bool(sp_cfg.get("enabled", True))
                    ctdm.sparse_pairs_strategy = str(
                        sp_cfg.get("strategy", "b")).lower()
                    ctdm.sparse_pairs_topk = int(sp_cfg.get("topk", 8))
                    if "lambda_time" in sp_cfg:
                        ctdm.sparse_pairs_lambda_time = float(
                            sp_cfg["lambda_time"])
                    if "lambda_degree" in sp_cfg:
                        ctdm.sparse_pairs_lambda_degree = float(
                            sp_cfg["lambda_degree"])

                # CTDM regularizer weights and prototype settings (optional overrides)
                reg_cfg = ctdm_cfg.get("regularizers", {})
                if isinstance(reg_cfg, dict):
                    if "lambda_proto" in reg_cfg:
                        ctdm.lambda_proto = float(reg_cfg["lambda_proto"])
                    if "temperature" in reg_cfg:
                        ctdm.prototype_temperature = float(reg_cfg["temperature"])
                    if "momentum" in reg_cfg:
                        ctdm.prototype_momentum = float(reg_cfg["momentum"])
                    if "lambda_topo_bce" in reg_cfg:
                        ctdm.lambda_topo_bce = float(reg_cfg["lambda_topo_bce"])
                    if "lambda_deg_kl" in reg_cfg:
                        ctdm.lambda_deg_kl = float(reg_cfg["lambda_deg_kl"])
                    # New regularizer parameters for text semantic integration
                    if "lambda_proto_orth" in reg_cfg:
                        ctdm.lambda_proto_orth = float(
                            reg_cfg["lambda_proto_orth"])
                    if "lambda_supcon_pair" in reg_cfg:
                        ctdm.lambda_supcon_pair = float(
                            reg_cfg["lambda_supcon_pair"])
                    if "pair_ids" in reg_cfg:
                        ctdm.pair_ids = tuple(reg_cfg["pair_ids"])
                # Optional forecast head for future statistics
                forecast_cfg = ctdm_cfg.get("forecast", {})
                if isinstance(forecast_cfg, dict) and forecast_cfg.get("enabled", False):
                    forecast_dim = int(forecast_cfg.get("dim", 2))
                    ctdm.forecast_enabled = True
                    ctdm.lambda_forecast = float(forecast_cfg.get("weight", 0.0))
                    ctdm.forecast_head = nn.Linear(
                        ctdm.hidden_channels, forecast_dim)
                struct_cfg = ctdm_cfg.get("structure_dropout", {})
                if isinstance(struct_cfg, dict):
                    ctdm.edge_drop_rate = float(struct_cfg.get("edge", 0.0))
                struct_cfg = ctdm_cfg.get("structure_dropout", {})
                if isinstance(struct_cfg, dict):
                    ctdm.edge_drop_rate = float(struct_cfg.get("edge", 0.0))

                ctdm_train_cfg = TrainerConfig(
                    epochs=int(active_training_cfg.get("epochs_ctdm", active_training_cfg["epochs"])),
                    learning_rate=float(active_training_cfg.get("lr_ctdm", active_training_cfg.get("lr", 3e-4))),
                    weight_decay=float(active_training_cfg.get("weight_decay_ctdm", active_training_cfg.get("weight_decay", 0.0))),
                    batch_size=int(active_training_cfg.get("batch_size_ctdm", 1)),
                    grad_clip=float(active_training_cfg.get("grad_clip", 0.0)),
                    amp=bool(active_training_cfg.get("amp", True)),
                    early_stopping_patience=int(
                        active_training_cfg.get(
                            "early_stopping_patience_ctdm",
                            active_training_cfg.get("early_stopping_patience", 20),
                        )
                    ),
                    num_workers=int(active_training_cfg.get("num_workers", 4)),
                    device=device,
                    use_tqdm=bool(active_training_cfg.get("use_tqdm", True)),
                    class_weights=None,
                    max_batches_per_epoch=active_training_cfg.get(
                        "max_batches_per_epoch_ctdm",
                        active_training_cfg.get("max_batches_per_epoch"),
                    ),
                    min_epochs=int(
                        active_training_cfg.get(
                            "min_epochs_ctdm",
                            active_training_cfg.get("min_epochs", 30),
                        )
                    ),
                    selection_metric="objective_loss",
                    selection_mode="objective",
                    fail_on_nonfinite=bool(active_training_cfg.get("fail_on_nonfinite", False)),
                )
                ctdm_eval_cfg = exp_cfg.get("evaluation", {})
                ctdm_max_eval_samples = int(
                    ctdm_eval_cfg.get("max_samples", 0)
                ) if isinstance(ctdm_eval_cfg, dict) else 0

                def _maybe_subset_ctdm(ds):
                    if ctdm_max_eval_samples > 0:
                        return Subset(ds, list(range(min(len(ds), ctdm_max_eval_samples))))
                    return ds

                def _make_ctdm_loader(ds, shuffle: bool) -> DataLoader:
                    return DataLoader(
                        ds,
                        batch_size=ctdm_train_cfg.batch_size,
                        shuffle=shuffle,
                        num_workers=ctdm_train_cfg.num_workers,
                        pin_memory=(device == "cuda"),
                        persistent_workers=ctdm_train_cfg.num_workers > 0,
                    )

                ctdm_train_ds = ctdm_scan_train_ds if train_max_samples <= 0 else Subset(
                    ctdm_scan_train_ds, list(range(min(len(ctdm_scan_train_ds), train_max_samples)))
                )
                ctdm_val_ds = _maybe_subset_ctdm(ctdm_scan_val_ds)
                ctdm_metadata_model_name = "rumordiffusion"
                ctdm_pretrain_dir = os.path.join(
                    ctdm_ckpt_root, source_dataset_name, ctdm_subdir, f"seed{seed}"
                )
                os.makedirs(ctdm_pretrain_dir, exist_ok=True)
                ctdm_pretrain_path = os.path.join(ctdm_pretrain_dir, ckpt_filename)
                skip_ctdm_pretraining = bool(args.transfer_adapt)
                if skip_ctdm_pretraining and (not os.path.isfile(ctdm_pretrain_path)) and args.ckpt:
                    derived_ctdm_path = derive_ctdm_checkpoint_from_arbiter_checkpoint(args.ckpt, ctdm_subdir=ctdm_subdir)
                    if derived_ctdm_path is not None and os.path.isfile(derived_ctdm_path):
                        ctdm_pretrain_path = derived_ctdm_path
                reuse_ctdm_pretrain = bool(active_training_cfg.get("reuse_ctdm_pretrain", False))
                ctdm_checkpoint_metadata = build_checkpoint_metadata(
                    model_name=ctdm_metadata_model_name,
                    stage="ctdm",
                    dataset_name=source_dataset_name,
                    dataset_root=source_dataset_root,
                    seed=int(seed),
                    label_mapping=dataset_label_mapping,
                    config_payload={
                        "ctdm": _deep_merge(ctdm_cfg, {"max_nodes": int(max_nodes_final)}),
                        "text": rd_cfg.get("text", {}),
                        "dataset_text_runtime": text_runtime_cfg,
                        "training": {
                            "epochs_ctdm": int(active_training_cfg.get("epochs_ctdm", active_training_cfg["epochs"])),
                            "lr_ctdm": float(active_training_cfg.get("lr_ctdm", active_training_cfg.get("lr", 3e-4))),
                            "weight_decay_ctdm": float(active_training_cfg.get("weight_decay_ctdm", active_training_cfg.get("weight_decay", 0.0))),
                            "batch_size_ctdm": int(active_training_cfg.get("batch_size_ctdm", 1)),
                            "early_stopping_patience_ctdm": int(
                                active_training_cfg.get(
                                    "early_stopping_patience_ctdm",
                                    active_training_cfg.get("early_stopping_patience", 20),
                                )
                            ),
                            "min_epochs_ctdm": int(
                                active_training_cfg.get(
                                    "min_epochs_ctdm",
                                    active_training_cfg.get("min_epochs", 30),
                                )
                            ),
                        },
                        "prefix": prefix_cfg,
                        "observability": observability_cfg,
                    },
                )
                if skip_ctdm_pretraining:
                    logging.info(f"Target adaptation: reusing source CTDM checkpoint {ctdm_pretrain_path}")
                elif (not args.eval_only) and not (reuse_ctdm_pretrain and os.path.isfile(ctdm_pretrain_path)):
                    logging.info("CTDM pretraining started")
                    ctdm_trainer = Trainer(model=ctdm, cfg=ctdm_train_cfg)
                    ctdm_fit_out = ctdm_trainer.fit(
                        train_loader=_make_ctdm_loader(ctdm_train_ds, shuffle=True),
                        val_loader=_make_ctdm_loader(ctdm_val_ds, shuffle=False),
                        save_path=ctdm_pretrain_path,
                        checkpoint_metadata=ctdm_checkpoint_metadata,
                    )
                    logging.info(
                        "CTDM pretraining best val diffusion loss: "
                        f"{ctdm_fit_out.get('best_val_objective_loss', ctdm_fit_out.get('best_val_metric', 0.0)):.6f}"
                    )
                elif (not args.eval_only) and reuse_ctdm_pretrain and os.path.isfile(ctdm_pretrain_path):
                    logging.info(f"Reusing existing CTDM pretraining checkpoint: {ctdm_pretrain_path}")
                if (skip_ctdm_pretraining or not args.eval_only) and os.path.isfile(ctdm_pretrain_path):
                    ctdm_state, _ = load_checkpoint_state(
                        ckpt_path=ctdm_pretrain_path,
                        device=device,
                        expected_metadata=ctdm_checkpoint_metadata,
                        validate_metadata=validate_ckpt_metadata,
                        allow_legacy=allow_legacy_ckpt,
                        ignored_metadata_keys=rd_ignored_metadata_keys,
                    )
                    ctdm.load_state_dict(ctdm_state)
                elif skip_ctdm_pretraining or not args.eval_only:
                    raise FileNotFoundError(
                        f"Missing CTDM pretraining checkpoint at {ctdm_pretrain_path}"
                    )

                if use_potentials:
                    if cache_potentials_after_ctdm and not args.eval_only:
                        logging.info(f"Generating RumorDiffusion cached potentials in {potentials_dir}")
                        for split_dataset in (ds_train, ds_val, ds_test):
                            generate_potentials(
                                model=ctdm,
                                dataset=split_dataset,
                                out_dir=potentials_dir,
                                batch_size=int(active_training_cfg.get("potentials_batch_size", 1)),
                                num_workers=int(active_training_cfg.get("potentials_num_workers", 0)),
                                device=device,
                                stride=int(ctdm_cfg.get("t_stride", 20)),
                                skip_existing=bool(active_training_cfg.get("reuse_cached_potentials", True)),
                                cache_metadata=model_potential_cache_metadata,
                            )
                        # Auxiliary consistency views must use independent
                        # prefix-aware namespaces. A full-prefix cache is never
                        # reused for an early window.
                        for view_cfg, view_spec in zip(
                            rd_prefix_consistency_views,
                            rd_prefix_consistency_potential_specs,
                        ):
                            view_dir = str(view_spec["potentials_dir"])
                            view_metadata = dict(view_spec["potential_cache_metadata"])
                            logging.info(
                                "Generating RumorDiffusion prefix-view cache "
                                f"unit={view_metadata['prefix_unit']} window={view_metadata['prefix_window']} "
                                f"in {view_dir}"
                            )
                            view_datasets = [
                                make_graph_dataset(
                                    root=dataset_root,
                                    split=split_name,
                                    use_lp_edges=use_lp,
                                    prefix_cfg=view_cfg,
                                    observability_cfg=observability_cfg,
                                    text_runtime_cfg=text_runtime_cfg,
                                )
                                for split_name in ("train", "val", "test")
                            ]
                            for view_dataset in view_datasets:
                                generate_potentials(
                                    model=ctdm,
                                    dataset=view_dataset,
                                    out_dir=view_dir,
                                    batch_size=int(active_training_cfg.get("potentials_batch_size", 1)),
                                    num_workers=int(active_training_cfg.get("potentials_num_workers", 0)),
                                    device=device,
                                    stride=int(ctdm_cfg.get("t_stride", 20)),
                                    skip_existing=bool(active_training_cfg.get("reuse_cached_potentials", True)),
                                    cache_metadata=view_metadata,
                                )
                    if not os.path.isdir(potentials_dir):
                        raise FileNotFoundError(f"Missing RumorDiffusion potentials directory: {potentials_dir}")
                    ds_train = make_graph_dataset(
                        root=dataset_root,
                        split="train",
                        use_lp_edges=use_lp_edges,
                        use_potentials=True,
                        potentials_dir=potentials_dir,
                        potentials_pair=potentials_pair,
                        potential_cache_metadata=model_potential_cache_metadata,
                        prefix_cfg=prefix_cfg,
                        observability_cfg=observability_cfg,
                        text_runtime_cfg=text_runtime_cfg,
                    )
                    ds_val = make_graph_dataset(
                        root=dataset_root,
                        split="val",
                        use_lp_edges=use_lp_edges,
                        use_potentials=True,
                        potentials_dir=potentials_dir,
                        potentials_pair=potentials_pair,
                        potential_cache_metadata=model_potential_cache_metadata,
                        prefix_cfg=prefix_cfg,
                        observability_cfg=observability_cfg,
                        text_runtime_cfg=text_runtime_cfg,
                    )
                    ds_test = make_graph_dataset(
                        root=dataset_root,
                        split="test",
                        use_lp_edges=use_lp_edges,
                        use_potentials=True,
                        potentials_dir=potentials_dir,
                        potentials_pair=potentials_pair,
                        potential_cache_metadata=model_potential_cache_metadata,
                        prefix_cfg=prefix_cfg,
                        observability_cfg=observability_cfg,
                        text_runtime_cfg=text_runtime_cfg,
                    )

                # RumorDiffusion arbiter and loss config overrides
                rd_loss_cfg = rd_cfg.get("loss", {})
                # Build RumorDiffusion arbiter
                arb_cfg = rd_cfg.get("arbiter", {})
                rd_eval_cfg_for_model = rd_cfg.get("evaluation", model_cfg.get("evaluation", {}))
                uncertainty_cfg = (
                    rd_eval_cfg_for_model.get("uncertainty", {})
                    if isinstance(rd_eval_cfg_for_model, dict)
                    else {}
                )
                uncertainty_sampling = str(
                    uncertainty_cfg.get(
                        "sampling",
                        arb_cfg.get("uncertainty_sampling", "deterministic"),
                    )
                ).lower()
                uncertainty_readout = str(
                    uncertainty_cfg.get(
                        "potential_readout",
                        arb_cfg.get("uncertainty_readout", "observed_prefix"),
                    )
                ).lower()
                if use_potentials and str(arb_cfg.get("type", "transformer")).lower() == "transformer":
                    if len(potentials_pair) != int(num_classes):
                        raise ValueError(
                            "RumorDiffusion with cached potentials requires one token per class. "
                            f"Expected {int(num_classes)} entries in potentials_pair, got {len(potentials_pair)}."
                        )
                # Get actual potential vector dimension from loaded data
                sample_data = ds_train.get(0)
                if hasattr(sample_data, 'h0') and hasattr(sample_data, 'h1'):
                    actual_hidden_dim = int(sample_data.h0.size(-1))
                    concat_dim = int(
                        torch.cat([sample_data.h0, sample_data.h1], dim=-1).size(-1))
                elif hasattr(sample_data, 'h_cat'):
                    token_count = len(potentials_pair) if use_potentials else int(num_classes)
                    if token_count <= 0:
                        raise ValueError("RumorDiffusion cached potentials require a positive token count.")
                    concat_dim = int(sample_data.h_cat.size(-1))
                    if concat_dim % int(token_count) != 0:
                        raise ValueError(
                            f"Invalid cached potentials width {concat_dim} for token_count={token_count}."
                        )
                    actual_hidden_dim = int(concat_dim // int(token_count))
                else:
                    # Fallback to CTDM inference if potentials not loaded
                    sample_batch = next(
                        iter(DataLoader(ds_train, batch_size=1, shuffle=False)))
                    sample_batch = sample_batch.to(device)
                    with torch.no_grad():
                        sample_potentials = ctdm.infer_potentials_all(sample_batch)
                        actual_hidden_dim = int(sample_potentials.size(-1))
                        h0 = sample_potentials[:, 0, :]
                        h1 = sample_potentials[:, 1, :]
                        concat_dim = int(torch.cat([h0, h1], dim=-1).size(-1))
                logging.info(
                    f"Single potential vector dimension: {actual_hidden_dim}")
                logging.info(f"Concatenated input dimension: {concat_dim}")

                # Build RumorDiffusion with Transformer arbiter; pass dd_gan config if provided
                dd_cfg = rd_cfg.get("dd_gan", {})
                m = RumorDiffusion(
                    hidden_channels=actual_hidden_dim,
                    num_classes=num_classes,
                    arbiter_hidden=int(arb_cfg.get("hidden", 256)),
                    dropout=float(arb_cfg.get("dropout", 0.3)),
                    ctdm=ctdm,
                    freeze_ctdm=bool(arb_cfg.get("freeze_ctdm", True)),
                    arbiter_type=str(arb_cfg.get("type", "transformer")).lower(),
                    arbiter_readout=str(arb_cfg.get("readout", "cls")).lower(),
                    token_norm=str(arb_cfg.get("token_norm", "layernorm")).lower(),
                    token_dropout=float(arb_cfg.get(
                        "token_dropout", arb_cfg.get("dropout", 0.3))),
                    transformer_nhead=int(arb_cfg.get(
                        "transformer", {}).get("nhead", 4)),
                    transformer_nlayers=int(arb_cfg.get(
                        "transformer", {}).get("nlayers", 2)),
                    transformer_dim_feedforward=int(arb_cfg.get(
                        "transformer", {}).get("dim_feedforward", 512)),
                    transformer_dropout=float(arb_cfg.get("transformer", {}).get(
                        "dropout", arb_cfg.get("dropout", 0.3))),
                    dd_gan_enabled=bool(dd_cfg.get("enabled", False)),
                    enable_dd_gan=bool(dd_cfg.get(
                        "enable", dd_cfg.get("enabled", False))),
                    lambda_gan=float(dd_cfg.get("lambda_gan", 0.0)),
                    lr_d=float(dd_cfg.get("lr_d", 1.0e-4)),
                    lambda_r1=float(dd_cfg.get("lambda_r1", 10.0)),
                    # New parameters for text semantic integration and refined loss
                    use_pair_token=bool(arb_cfg.get("use_pair_token", False)),
                    pair_token_mode=str(arb_cfg.get("pair_token_mode", "diff")),
                    pair_ids=tuple(arb_cfg.get("pair_ids", [1, 2])),
                    tf_head_enabled=bool(arb_cfg.get("tf_head_enabled", False)),
                    tf_infer_adjust_alpha=float(
                        arb_cfg.get("tf_infer_adjust_alpha", 0.0)),
                    pair_margin_enabled=bool(
                        rd_loss_cfg.get("pair_margin_enabled", False)),
                    pair_margin_m=float(rd_loss_cfg.get("pair_margin_m", 0.5)),
                    lambda_pair_margin=float(
                        rd_loss_cfg.get("lambda_pair_margin", 0.0)),
                    enable_pair_margin_loss=bool(
                        rd_loss_cfg.get("enable_pair_margin_loss", True)),
                    lambda_tf_bce=float(rd_loss_cfg.get("lambda_tf_bce", 0.0)),
                    use_class_embedding=bool(
                        arb_cfg.get("use_class_embedding", False)),
                    class_embed_mode=str(arb_cfg.get(
                        "class_embed_mode", "concat_proj")),
                    cache_potentials=bool(arb_cfg.get("cache_potentials", False)),
                    infer_amp=bool(arb_cfg.get("infer_amp", False)),
                    uncertainty_force_ctdm=bool(arb_cfg.get("uncertainty_force_ctdm", False)),
                    uncertainty_sampling=uncertainty_sampling,
                    uncertainty_readout=uncertainty_readout,
                    loss_logging=bool(rd_loss_cfg.get("loss_logging", False)),
                    lambda_supcon=float(rd_loss_cfg.get("lambda_supcon", 0.0)),
                    supcon_temperature=float(rd_loss_cfg.get("supcon_temperature", 0.07)),
                    prototype_residual_cfg=arb_cfg.get("prototype_residual", {}),
                    semantic_residual_cfg=arb_cfg.get("semantic_residual", {}),
                    prefix_graph_expert_cfg=arb_cfg.get("prefix_graph_expert", {}),
                    gated_fusion_cfg=arb_cfg.get("gated_fusion", {}),
                    posterior_aggregation_cfg=arb_cfg.get("posterior_aggregation", {}),
                    prefix_consistency_cfg=rd_prefix_consistency_cfg,
                    source_robustness_cfg=rd_source_robustness_cfg,
                    root_feature_dim=int(num_features),
                    text_feature_dim=int(text_channels),
                )
                logging.info(
                    "RD uncertainty cfg -> "
                    f"sampling={uncertainty_sampling}, "
                    f"potential_readout={uncertainty_readout}, "
                    f"force_ctdm={bool(arb_cfg.get('uncertainty_force_ctdm', False))}"
                )
                # Inject loss hyperparameters and class priors
                if "class_priors" in rd_loss_cfg:
                    pri = torch.tensor(
                        rd_loss_cfg["class_priors"], dtype=torch.float32)
                    pri = pri / pri.sum()
                    m.class_priors = pri.to(device)
                else:
                    counts = [0 for _ in range(int(num_classes))]
                    for split_idx in range(len(ds_train)):
                        yv = int(ds_train.get(split_idx).y.view(-1)[0].item())
                        counts[yv] += 1
                    total = sum(counts)
                    pri = torch.tensor([c/float(total)
                                       for c in counts], dtype=torch.float32)
                    m.class_priors = pri.to(device)
                if m.class_priors.sum().abs().item() <= 0.0 or torch.isnan(m.class_priors.sum()).item():
                    raise ValueError("RumorDiffusion class priors must be positive and finite.")
                # Loss weights
                m.lambda_diffusion = float(
                    rd_loss_cfg.get("lambda_diffusion", 0.0))
                m.la_tau = float(rd_loss_cfg.get("la_tau", 1.0))
                m.la_gamma = float(rd_loss_cfg.get("la_gamma", 1.5))
                m.la_smooth = float(rd_loss_cfg.get("la_smooth", 0.05))
                m.lambda_brier = float(rd_loss_cfg.get("w_brier", 0.01))
                m.lambda_consistency = float(rd_loss_cfg.get("w_cons", 0.05))
                logging.info(
                    f"RD-Loss cfg -> lambda_diffusion={m.lambda_diffusion}, "
                    f"la_tau={m.la_tau}, la_gamma={m.la_gamma}"
                )
                # Multi-pair margin loss configuration for handling multiple confusable class pairs
                mp_enabled = bool(rd_loss_cfg.get("enable_multi_pair_margin", rd_loss_cfg.get(
                    "multi_pair_margin_enabled", False)))
                m.multi_pair_margin_list = rd_loss_cfg.get(
                    "multi_pair_margin_list", []) if mp_enabled else []
                m.multi_pair_margin_values = rd_loss_cfg.get(
                    "multi_pair_margin_values", []) if mp_enabled else []
                m.lambda_multi_pair_margin = float(rd_loss_cfg.get(
                    "lambda_multi_pair_margin", 0.0)) if mp_enabled else 0.0
                m.enable_multi_pair_margin = mp_enabled and len(
                    m.multi_pair_margin_list) > 0 and float(m.lambda_multi_pair_margin) > 0.0
                if m.enable_multi_pair_margin:
                    logging.info(
                        f"Multi-pair margin loss enabled with pairs: {m.multi_pair_margin_list}, margins: {m.multi_pair_margin_values}, lambda: {m.lambda_multi_pair_margin}")
                # Optional toggles for auxiliary losses
                m.enable_pair_margin_loss = bool(rd_loss_cfg.get(
                    "enable_pair_margin_loss", rd_loss_cfg.get("pair_margin_enabled", False)))
                if not m.enable_pair_margin_loss:
                    m.lambda_pair_margin = 0.0
                m.enable_tf_head_loss = bool(
                    rd_loss_cfg.get("enable_tf_head_loss", True))
                if not m.enable_tf_head_loss:
                    m.lambda_tf_bce = 0.0
                # Summarize key loss hyperparameters for reproducibility
                logging.info(
                    f"Loss cfg -> la_gamma={m.la_gamma}, la_tau={m.la_tau}, "
                    f"pair_margin(enabled={m.enable_pair_margin_loss}, lambda={m.lambda_pair_margin}, m={m.pair_margin_m}), "
                    f"multi_pair(enabled={m.enable_multi_pair_margin}, lambda={m.lambda_multi_pair_margin}), "
                    f"tf_head(enabled={m.enable_tf_head_loss}, lambda={m.lambda_tf_bce}), "
                    f"supcon={m.lambda_supcon}, brier={m.lambda_brier}, "
                    f"consistency={m.lambda_consistency}, dd_gan={m.dd_gan_enabled}"
                )
            elif name == "cgvae_gcn":
                # CGVAE-GCN two-stage pipeline
                logging.info("Running CGVAE-GCN two-stage training...")

                # Stage 1: Train CVAE models and generate augmented edges
                logging.info("Stage 1: Training class-conditional VGAE models...")
                cgvae_cfg_dict = model_cfg.get("cvae", {})
                print("Creating CGVAEConfig...")
                print(f"cgvae_cfg_dict: {cgvae_cfg_dict}")
                print(f"model_cfg: {model_cfg}")
                cg_cfg = CGVAEConfig(
                    epochs=int(cgvae_cfg_dict.get(
                        "epochs", exp_cfg["training"].get("epochs", 50))),
                    learning_rate=float(cgvae_cfg_dict.get(
                        "lr", exp_cfg["training"].get("lr", 1e-3))),
                    weight_decay=float(cgvae_cfg_dict.get(
                        "weight_decay", exp_cfg["training"].get("weight_decay", 1e-5))),
                    batch_size=int(cgvae_cfg_dict.get(
                        "batch_size", exp_cfg["training"].get("batch_size", 8))),
                    grad_clip=float(cgvae_cfg_dict.get(
                        "grad_clip", exp_cfg["training"].get("grad_clip", 1.0))),
                    amp=bool(cgvae_cfg_dict.get(
                        "amp", exp_cfg["training"].get("amp", True))),
                    num_workers=int(cgvae_cfg_dict.get(
                        "num_workers", exp_cfg["training"].get("num_workers", 0))),
                    early_stopping_patience=int(cgvae_cfg_dict.get(
                        "early_stopping_patience", exp_cfg["training"].get("early_stopping_patience", 15))),
                    device=device,
                    hidden=int(cgvae_cfg_dict.get(
                        "hidden", model_cfg["model"].get("hidden", 256))),
                    latent=int(cgvae_cfg_dict.get(
                        "latent", model_cfg["model"].get("latent", 128))),
                    num_layers=int(cgvae_cfg_dict.get("num_layers", 3)),
                    dropout=float(cgvae_cfg_dict.get("dropout", 0.3)),
                    use_bn=bool(cgvae_cfg_dict.get("use_bn", True)),
                    kl_anneal=bool(cgvae_cfg_dict.get("kl_anneal", True)),
                )
                print("Creating CGVAETrainer...")
                cg_trainer = CGVAETrainer(cfg=cg_cfg)
                print("CGVAETrainer created successfully")
                cg_ckpt_root = os.path.join(
                    "Output", "ckpts", dataset_name, "cgvae_gcn_stage1")
                print(f"Checkpoint root: {cg_ckpt_root}")
                print(
                    f"Number of classes: {num_classes}, Number of features: {num_features}")
                print("Starting train_all_classes...")
                cg_trainer.train_all_classes(
                    ds_train, cg_ckpt_root, num_classes, num_features)

                # Generate augmented edges for all splits
                logging.info(
                    "Generating augmented edges using trained CVAE models...")
                aug_edge_dir = os.path.join(
                    "Input", "processed", dataset_name, "cgvae_edges")
                cg_trainer.generate_edges_for_split(
                    ds_train, cg_ckpt_root, num_classes, num_features, aug_edge_dir, "train")
                cg_trainer.generate_edges_for_split(
                    ds_val, cg_ckpt_root, num_classes, num_features, aug_edge_dir, "val")
                cg_trainer.generate_edges_for_split(
                    ds_test, cg_ckpt_root, num_classes, num_features, aug_edge_dir, "test")

                # Stage 2: Train GCN on augmented graphs
                logging.info(
                    "Stage 2: Training GCN classifier on augmented graphs...")

                # Reload datasets with augmented edges
                ds_train = GraphDataset(
                    root=dataset_root,
                    split="train",
                    use_lp_edges=True,
                    lp_dir=aug_edge_dir,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                    text_cache_name=str(text_runtime_cfg.get("cache_name", "text_embs.pt")),
                    force_text_cache=bool(text_runtime_cfg.get("force_text_cache", False)),
                    text_normalize=bool(text_runtime_cfg.get("normalize_embeddings", False)),
                )
                ds_val = GraphDataset(
                    root=dataset_root,
                    split="val",
                    use_lp_edges=True,
                    lp_dir=aug_edge_dir,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                    text_cache_name=str(text_runtime_cfg.get("cache_name", "text_embs.pt")),
                    force_text_cache=bool(text_runtime_cfg.get("force_text_cache", False)),
                    text_normalize=bool(text_runtime_cfg.get("normalize_embeddings", False)),
                )
                ds_test = GraphDataset(
                    root=dataset_root,
                    split="test",
                    use_lp_edges=True,
                    lp_dir=aug_edge_dir,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                    text_cache_name=str(text_runtime_cfg.get("cache_name", "text_embs.pt")),
                    force_text_cache=bool(text_runtime_cfg.get("force_text_cache", False)),
                    text_normalize=bool(text_runtime_cfg.get("normalize_embeddings", False)),
                )

                # Build GCN model for stage 2
                from Libs.models.baselines.GCN import GCN
                gcn_hidden = int(model_cfg["model"].get("gcn_hidden", 128))
                gcn_num_layers = int(model_cfg["model"].get("gcn_num_layers", 3))
                gcn_dropout = float(model_cfg["model"].get("gcn_dropout", 0.3))

                m = GCN(
                    in_channels=num_features,
                    hidden_channels=gcn_hidden,
                    num_layers=gcn_num_layers,
                    num_classes=num_classes,
                    dropout=gcn_dropout
                ).to(device)

                # Continue with standard GCN training flow
            else:
                m = build_model(
                    model_cfg["name"], model_cfg["model"], num_features, num_classes, text_channels=text_channels)

            if is_rumordiffusion and rd_prefix_consistency_views:
                ds_train = make_graph_dataset(
                    root=dataset_root,
                    split="train",
                    use_lp_edges=use_lp,
                    use_potentials=model_use_potentials,
                    potentials_dir=model_potentials_dir,
                    potentials_pair=model_potentials_pair,
                    potential_cache_metadata=model_potential_cache_metadata,
                    prefix_cfg=prefix_cfg,
                    prefix_consistency_views=rd_prefix_consistency_views,
                    prefix_consistency_potential_specs=rd_prefix_consistency_potential_specs,
                    observability_cfg=observability_cfg,
                    text_runtime_cfg=text_runtime_cfg,
                )

            trainer_cfg = TrainerConfig(
                epochs=int(active_training_cfg["epochs"]),
                learning_rate=float(active_training_cfg.get("lr", 3e-4)),
                weight_decay=float(active_training_cfg.get("weight_decay", 0.0)),
                batch_size=int(active_training_cfg["batch_size"]),
                grad_clip=float(active_training_cfg.get("grad_clip", 0.0)),
                amp=bool(active_training_cfg.get("amp", True)),
                early_stopping_patience=int(
                    active_training_cfg.get("early_stopping_patience", 20)),
                num_workers=int(active_training_cfg.get("num_workers", 4)),
                device=device,
                use_tqdm=bool(active_training_cfg.get("use_tqdm", True)),
                class_weights=class_weights,
                max_batches_per_epoch=active_training_cfg.get(
                    "max_batches_per_epoch"),
                min_epochs=int(active_training_cfg.get("min_epochs", 30)),
                fail_on_nonfinite=bool(active_training_cfg.get("fail_on_nonfinite", False)),
            )

            eval_cfg = exp_cfg.get("evaluation", {})
            eval_batch_size = int(eval_cfg.get("batch_size", trainer_cfg.batch_size)) if isinstance(eval_cfg, dict) else int(trainer_cfg.batch_size)

            def make_loader(ds: GraphDataset, shuffle: bool, sampler=None, batch_size: Optional[int] = None) -> DataLoader:
                resolved_batch_size = int(batch_size) if batch_size is not None else int(trainer_cfg.batch_size)
                return DataLoader(
                    ds,
                    batch_size=resolved_batch_size,
                    shuffle=(False if sampler is not None else shuffle),
                    sampler=sampler,
                    num_workers=trainer_cfg.num_workers,
                    pin_memory=(device == "cuda"),
                    persistent_workers=trainer_cfg.num_workers > 0,
                )

            # Use regular shuffle instead of WeightedRandomSampler for balanced datasets
            # (Data is already balanced, so WeightedRandomSampler may introduce noise)
            max_eval_samples = int(eval_cfg.get(
                "max_samples", 0)) if isinstance(eval_cfg, dict) else 0

            def _maybe_subset(ds):
                if max_eval_samples > 0:
                    return Subset(ds, list(range(min(len(ds), max_eval_samples))))
                return ds
            ds_val_eval = _maybe_subset(ds_val)
            ds_test_eval = _maybe_subset(ds_test)

            ds_train_eff = ds_train if train_max_samples <= 0 else Subset(
                ds_train, list(range(min(len(ds_train), train_max_samples))))
            train_loader = make_loader(ds_train_eff, shuffle=True, batch_size=trainer_cfg.batch_size)
            val_loader = make_loader(ds_val_eval, shuffle=False, batch_size=eval_batch_size)
            test_loader = make_loader(ds_test_eval, shuffle=False, batch_size=eval_batch_size)
            observation_loader_cache: Dict[Tuple[str, int, str], DataLoader] = {}

            def get_observation_loader(split_name: str, observation_window: int, observation_unit: str) -> DataLoader:
                cache_key = (str(split_name), int(observation_window), str(observation_unit))
                if cache_key in observation_loader_cache:
                    return observation_loader_cache[cache_key]
                prefix_view_cfg = build_observation_prefix_cfg(
                    base_prefix_cfg=prefix_cfg,
                    observation_window=int(observation_window),
                    observation_unit=str(observation_unit),
                )
                observation_potentials_dir = model_potentials_dir
                observation_potential_metadata = model_potential_cache_metadata
                if model_use_potentials:
                    if model_potentials_root is None or model_potential_cache_ctdm_payload is None:
                        raise RuntimeError("RumorDiffusion potential-cache context is missing for early evaluation.")
                    observation_potentials_dir = potential_cache_dir(model_potentials_root, prefix_view_cfg)
                    observation_potential_metadata = build_potential_cache_metadata(
                        dataset_name=dataset_name,
                        seed=int(seed),
                        prefix_cfg=prefix_view_cfg,
                        ctdm_config_payload=model_potential_cache_ctdm_payload,
                    )
                split_dataset = make_graph_dataset(
                    root=dataset_root,
                    split=str(split_name),
                    use_lp_edges=use_lp,
                    use_potentials=model_use_potentials,
                    potentials_dir=observation_potentials_dir,
                    potentials_pair=model_potentials_pair,
                    potential_cache_metadata=observation_potential_metadata,
                    prefix_cfg=prefix_view_cfg,
                    observability_cfg=observability_cfg,
                    text_runtime_cfg=text_runtime_cfg,
                )
                if str(split_name) == "val":
                    split_dataset_eval = _maybe_subset(split_dataset)
                elif str(split_name) == "test":
                    split_dataset_eval = _maybe_subset(split_dataset)
                else:
                    split_dataset_eval = split_dataset
                loader = make_loader(split_dataset_eval, shuffle=False, batch_size=eval_batch_size)
                observation_loader_cache[cache_key] = loader
                return loader

            ckpt_dir = os.path.join(
                base_ckpt_dir, source_dataset_name, name, f"seed{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
            actual_ckpt_path = args.ckpt if args.ckpt else ckpt_path

            wb_cfg = active_training_cfg.get("wandb", {})
            wb_enabled = bool(wb_cfg.get("enabled", False))
            if wb_enabled:
                import wandb

                _wandb_run = wandb.init(
                    project=str(wb_cfg.get("project", "D2Plus")),
                    name=str(wb_cfg.get(
                        "run_name", f"{dataset_name}-{name}-seed{seed}")),
                    config={
                        "dataset": dataset_name,
                        "model": name,
                        "exp": args.exp,
                        "model_cfg": model_path,
                        "seed": seed,
                    },
                )
            trainer = Trainer(model=m, cfg=trainer_cfg)
            if is_rumordiffusion:
                joint_cfg = active_training_cfg.get("joint_finetuning", {})
                if isinstance(joint_cfg, dict) and bool(joint_cfg.get("enabled", False)):
                    configure_ctdm_joint_finetuning(trainer.model, joint_cfg)
                    arbiter_lr = float(joint_cfg.get("arbiter_lr", active_training_cfg.get("lr", 3e-4)))
                    ctdm_lr = float(joint_cfg.get("ctdm_lr", active_training_cfg.get("lr_ctdm", arbiter_lr * 0.1)))
                    weight_decay_joint = float(joint_cfg.get("weight_decay", active_training_cfg.get("weight_decay", 0.0)))
                    arbiter_params: List[nn.Parameter] = []
                    ctdm_params: List[nn.Parameter] = []
                    other_params: List[nn.Parameter] = []
                    for param_name, param in trainer.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        if param_name.startswith("ctdm."):
                            ctdm_params.append(param)
                        elif "arbiter" in param_name or "tf_head" in param_name or "class_embed" in param_name:
                            arbiter_params.append(param)
                        else:
                            other_params.append(param)
                    param_groups = []
                    if arbiter_params:
                        param_groups.append({"params": arbiter_params, "lr": arbiter_lr})
                    if ctdm_params:
                        param_groups.append({"params": ctdm_params, "lr": ctdm_lr})
                    if other_params:
                        param_groups.append({"params": other_params, "lr": arbiter_lr})
                    if len(param_groups) == 0:
                        raise ValueError("RumorDiffusion joint fine-tuning has no trainable parameters.")
                    trainer.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay_joint)
                    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        trainer.optimizer,
                        T_max=max(1, int(trainer_cfg.epochs)),
                    )
                    logging.info(
                        "RumorDiffusion joint fine-tuning enabled: "
                        f"arbiter_lr={arbiter_lr}, ctdm_lr={ctdm_lr}, "
                        f"trainable_arbiter_params={sum(p.numel() for p in arbiter_params)}, "
                        f"trainable_ctdm_params={sum(p.numel() for p in ctdm_params)}, "
                        f"trainable_other_params={sum(p.numel() for p in other_params)}"
                    )
            fit_out = {"best_val_macro_f1": 0.0, "best_val_metric": 0.0}
            train_seconds = 0.0
            transfer_mode = "source_only" if transfer_run else "main"
            source_checkpoint_path = actual_ckpt_path if transfer_run else ""
            adapt_ckpt_path = ""
            adapt_epochs = 0
            pseudo_label_coverage = 0.0
            checkpoint_metadata: Optional[Dict[str, Any]] = build_checkpoint_metadata(
                model_name=name,
                stage="classifier",
                dataset_name=source_dataset_name,
                dataset_root=source_dataset_root,
                seed=int(seed),
                label_mapping=dataset_label_mapping,
                config_payload={
                    "model": copy.deepcopy(model_cfg),
                    "training": {
                        "epochs": int(active_training_cfg["epochs"]),
                        "lr": float(active_training_cfg.get("lr", active_training_cfg.get("learning_rate", 3e-4))),
                        "weight_decay": float(active_training_cfg.get("weight_decay", 0.0)),
                        "batch_size": int(active_training_cfg["batch_size"]),
                        "early_stopping_patience": int(active_training_cfg.get("early_stopping_patience", 20)),
                        "min_epochs": int(active_training_cfg.get("min_epochs", 30)),
                    },
                    "prefix": prefix_cfg,
                    "observability": observability_cfg,
                },
            )
            if is_rumordiffusion:
                checkpoint_metadata = build_checkpoint_metadata(
                    model_name=name,
                    stage="arbiter",
                    dataset_name=source_dataset_name,
                    dataset_root=source_dataset_root,
                    seed=int(seed),
                    label_mapping=dataset_label_mapping,
                    config_payload={
                        "rumordiffusion": checkpoint_relevant_rumordiffusion_cfg(rd_cfg),
                        "training": {
                            "epochs": int(active_training_cfg["epochs"]),
                            "lr": float(active_training_cfg.get("lr", 3e-4)),
                            "weight_decay": float(active_training_cfg.get("weight_decay", 0.0)),
                            "batch_size": int(active_training_cfg["batch_size"]),
                            "early_stopping_patience": int(active_training_cfg.get("early_stopping_patience", 20)),
                            "min_epochs": int(active_training_cfg.get("min_epochs", 30)),
                            "joint_finetuning": active_training_cfg.get("joint_finetuning", {}),
                        },
                        "prefix": prefix_cfg,
                        "observability": observability_cfg,
                    },
                )
            eval_cfg = exp_cfg.get("evaluation", {})
            class_names = eval_cfg.get(
                "class_names", {0: "non-rumor", 1: "true", 2: "false", 3: "unverified"})
            class_names_overrides = eval_cfg.get("class_names_overrides", {})
            if isinstance(class_names_overrides, dict) and dataset_name in class_names_overrides:
                class_names = class_names_overrides[dataset_name]
            detailed_eval = True
            rd_eval_cfg = rd_cfg.get("evaluation", model_cfg.get(
                "evaluation", {})) if is_rumordiffusion else {}
            cal_cfg_for_ece = rd_eval_cfg.get("calibration", {}) if isinstance(
                rd_eval_cfg, dict) else {}
            active_ece_bins = int(
                eval_cfg.get("ece_bins", cal_cfg_for_ece.get("ece_bins", 15))
            ) if isinstance(eval_cfg, dict) else int(cal_cfg_for_ece.get("ece_bins", 15))
            export_pred_cfg = eval_cfg.get("export_predictions", {}) if isinstance(
                eval_cfg, dict) else {}
            latency_cfg = eval_cfg.get("latency", {}) if isinstance(
                eval_cfg, dict) else {}
            significance_cfg = eval_cfg.get("significance", {}) if isinstance(
                eval_cfg, dict) else {}
            if args.transfer_adapt:
                if not is_rumordiffusion:
                    raise ValueError("Target adaptation is only implemented for RumorDiffusion.")
                logging.info("Target-adaptation run: loading source checkpoint before unsupervised adaptation")
                state, _ = load_checkpoint_state(
                    ckpt_path=actual_ckpt_path,
                    device=device,
                    expected_metadata=checkpoint_metadata,
                    validate_metadata=bool(checkpoint_metadata is not None and validate_ckpt_metadata),
                    allow_legacy=bool(allow_legacy_ckpt),
                    ignored_metadata_keys=rd_ignored_metadata_keys,
                )
                trainer.model.load_state_dict(state)
                transfer_mode = "target_adapt"
                transfer_adapt_cfg = exp_cfg.get("transfer", {}).get("adaptation", {}) if isinstance(exp_cfg.get("transfer", {}).get("adaptation", {}), dict) else {}
                rd_adapt_cfg = rd_cfg.get("adaptation", {}) if isinstance(rd_cfg.get("adaptation", {}), dict) else {}
                adapt_cfg = _deep_merge(transfer_adapt_cfg, rd_adapt_cfg)
                if len(adapt_cfg) == 0:
                    raise ValueError("Missing rumordiffusion.adaptation config for --transfer_adapt.")
                target_split = str(adapt_cfg.get("target_split", "train"))
                selection_split = str(adapt_cfg.get("selection_split", "val"))
                if target_split != "train":
                    raise ValueError(f"Unsupported target adaptation split: {target_split}")
                if selection_split != "val":
                    raise ValueError(f"Unsupported target selection split: {selection_split}")
                target_obs_cfg = adapt_cfg.get("target_observability", {}) if isinstance(adapt_cfg.get("target_observability", {}), dict) else {}
                target_prefix_cfg = prefix_cfg if not isinstance(adapt_cfg.get("target_prefix", {}), dict) else {**prefix_cfg, **adapt_cfg.get("target_prefix", {})}
                target_potentials_dir = model_potentials_dir
                target_potential_metadata = model_potential_cache_metadata
                if model_use_potentials:
                    if model_potentials_root is None or model_potential_cache_ctdm_payload is None:
                        raise RuntimeError("RumorDiffusion potential-cache context is missing for target adaptation.")
                    target_potentials_dir = potential_cache_dir(model_potentials_root, target_prefix_cfg)
                    target_potential_metadata = build_potential_cache_metadata(
                        dataset_name=dataset_name,
                        seed=int(seed),
                        prefix_cfg=target_prefix_cfg,
                        ctdm_config_payload=model_potential_cache_ctdm_payload,
                    )
                target_train_ds = make_graph_dataset(
                    root=dataset_root,
                    split=target_split,
                    use_lp_edges=use_lp,
                    prefix_cfg=target_prefix_cfg,
                    observability_cfg={**observability_cfg, **target_obs_cfg},
                    text_runtime_cfg=text_runtime_cfg,
                    use_potentials=model_use_potentials,
                    potentials_dir=target_potentials_dir,
                    potentials_pair=model_potentials_pair,
                    potential_cache_metadata=target_potential_metadata,
                )
                target_val_ds = make_graph_dataset(
                    root=dataset_root,
                    split=selection_split,
                    use_lp_edges=use_lp,
                    prefix_cfg=prefix_cfg,
                    observability_cfg=observability_cfg,
                    text_runtime_cfg=text_runtime_cfg,
                    use_potentials=model_use_potentials,
                    potentials_dir=model_potentials_dir,
                    potentials_pair=model_potentials_pair,
                    potential_cache_metadata=model_potential_cache_metadata,
                )
                target_train_max_samples = int(adapt_cfg.get("target_max_samples", train_max_samples))
                target_selection_max_samples = int(adapt_cfg.get("selection_max_samples", max_eval_samples))
                target_train_eff = target_train_ds if target_train_max_samples <= 0 else Subset(
                    target_train_ds,
                    list(range(min(len(target_train_ds), target_train_max_samples))),
                )
                target_val_eff = target_val_ds if target_selection_max_samples <= 0 else Subset(
                    target_val_ds,
                    list(range(min(len(target_val_ds), target_selection_max_samples))),
                )
                logging.info(
                    f"Target-adapt data sizes -> train={len(target_train_eff)}, val={len(target_val_eff)}, "
                    f"batch_size={int(adapt_cfg.get('batch_size', trainer_cfg.batch_size))}, "
                    f"eval_batch_size={int(adapt_cfg.get('eval_batch_size', eval_batch_size))}"
                )
                target_train_loader = make_loader(target_train_eff, shuffle=True, batch_size=int(adapt_cfg.get("batch_size", trainer_cfg.batch_size)))
                target_val_loader = make_loader(
                    target_val_eff,
                    shuffle=False,
                    batch_size=int(adapt_cfg.get("eval_batch_size", eval_batch_size)),
                )
                adapt_ckpt_dir = os.path.join(base_ckpt_dir, source_dataset_name, name, f"seed{seed}")
                adapt_ckpt_name = str(adapt_cfg.get("adapt_ckpt_name", "best_target_adapt.pt"))
                adapt_ckpt_path = os.path.join(adapt_ckpt_dir, adapt_ckpt_name)
                train_start = time.time()
                fit_out = adapt_rumordiffusion_to_target(
                    model=trainer.model,
                    trainer=trainer,
                    source_checkpoint_path=actual_ckpt_path,
                    target_train_loader=target_train_loader,
                    target_val_loader=target_val_loader,
                    adaptation_cfg=adapt_cfg,
                    class_names=class_names,
                    device=device,
                    save_path=adapt_ckpt_path,
                )
                train_seconds = time.time() - train_start
                adapt_epochs = int(fit_out.get("adapt_epochs", 0))
                pseudo_label_coverage = float(fit_out.get("pseudo_label_coverage", 0.0))
                logging.info(
                    f"Best target-val macro F1 after adaptation: {fit_out.get('best_val_macro_f1', 0.0):.4f}; "
                    f"pseudo_label_coverage={pseudo_label_coverage:.4f}"
                )
            elif not args.eval_only:
                logging.info("Training started")
                train_start = time.time()
                fit_out = trainer.fit(train_loader=train_loader,
                                      val_loader=val_loader, save_path=ckpt_path, checkpoint_metadata=checkpoint_metadata)
                train_seconds = time.time() - train_start
                logging.info(
                    f"Best val macro F1: {fit_out.get('best_val_macro_f1', fit_out.get('best_val_metric', 0.0)):.4f}")
                if wb_enabled:
                    wandb.log({"best_val_macro_f1": float(fit_out.get(
                        'best_val_macro_f1', fit_out.get('best_val_metric', 0.0)))})
            else:
                logging.info("Evaluation-only run: loading checkpoint without optimizer updates")
                state, _ = load_checkpoint_state(
                    ckpt_path=actual_ckpt_path,
                    device=device,
                    expected_metadata=checkpoint_metadata,
                    validate_metadata=bool(is_rumordiffusion and checkpoint_metadata is not None and validate_ckpt_metadata),
                    allow_legacy=bool((not is_rumordiffusion) or allow_legacy_ckpt),
                    ignored_metadata_keys=rd_ignored_metadata_keys,
                )
                trainer.model.load_state_dict(state)
                logging.info(f"Checkpoint loaded from {actual_ckpt_path}")
            def persist_revision_row(row: Dict[str, str]) -> None:
                row["model_cfg"] = model_path
                row["exp_cfg"] = args.exp
                row["log_path"] = log_path
                append_dict_rows_csv(revision_artifacts_csv, revision_headers, [row])

            def persist_sweep_row(row: List[str]) -> None:
                append_rows_csv(sweeps_csv, sweeps_headers, [row])

            eval_start = time.time()
            logging.info("Main test evaluation started")
            test_outputs_base = collect_eval_outputs(
                trainer.model,
                test_loader,
                device=device,
                criterion=trainer.criterion,
                amp=trainer_cfg.amp,
                detailed=detailed_eval,
                class_names=class_names,
                profile=bool(isinstance(latency_cfg, dict) and latency_cfg.get("enabled", False)),
                profile_repeat=int(latency_cfg.get("repeat", 1)) if isinstance(latency_cfg, dict) else 1,
                profile_warmup_batches=int(latency_cfg.get("warmup_batches", 0)) if isinstance(latency_cfg, dict) else 0,
                profile_max_batches=int(latency_cfg.get("max_batches", 0)) if isinstance(latency_cfg, dict) else 0,
                compute_calibration=True,
                ece_bins=int(active_ece_bins),
            )
            test_metrics_base = test_outputs_base["metrics"]
            reported_logits = test_outputs_base["logits"]
            reported_targets = test_outputs_base["targets"]
            eval_seconds = time.time() - eval_start
            logging.info(f"Test loss: {test_metrics_base['loss']:.4f}")
            logging.info(f"Test accuracy: {test_metrics_base['acc']:.4f}")
            logging.info(f"Test macro F1: {test_metrics_base['macro_f1']:.4f}")
            if "report" in test_metrics_base:
                logging.info("\n=== Test Set Evaluation Report ===")
                logging.info(f"\n{test_metrics_base['report']}")
            if "per_class" in test_metrics_base:
                per_class_metrics = test_metrics_base["per_class"]
                if isinstance(per_class_metrics, dict):
                    for class_idx, class_metrics in per_class_metrics.items():
                        class_name = class_names.get(int(class_idx), str(class_idx)) if isinstance(class_names, dict) else str(class_idx)
                        if isinstance(class_metrics, dict):
                            persist_revision_row(
                                build_revision_row(
                                    timestamp=run_ts,
                                    dataset_name=dataset_name,
                                    source_dataset_name=source_dataset_name,
                                    run_kind="transfer" if dataset_name != source_dataset_name else "main",
                                    model_name=name,
                                    seed=int(seed),
                                    artifact_type="per_class_metric",
                                    tag=f"class_{int(class_idx)}_{class_name}",
                                    values={
                                        "metric": "precision",
                                        "delta": float(class_metrics.get("precision", 0.0)),
                                        "p_value": float(class_metrics.get("recall", 0.0)),
                                        "ci_low": float(class_metrics.get("f1", 0.0)),
                                        "ci_high": float(class_metrics.get("support", 0.0)),
                                        "notes": f"class_name={class_name}",
                                    },
                                )
                            )
            if "confusion_matrix" in test_metrics_base and isinstance(test_metrics_base["confusion_matrix"], torch.Tensor):
                persist_revision_row(
                    build_revision_row(
                        timestamp=run_ts,
                        dataset_name=dataset_name,
                        source_dataset_name=source_dataset_name,
                        run_kind="transfer" if dataset_name != source_dataset_name else "main",
                        model_name=name,
                        seed=int(seed),
                        artifact_type="confusion_matrix",
                        tag="main",
                        values={
                            "notes": json.dumps(test_metrics_base["confusion_matrix"].tolist()),
                        },
                    )
                )
            if len(test_outputs_base["latency_ms"]) > 0:
                latency_stats = summarize_latency_ms(test_outputs_base["latency_ms"])
                persist_revision_row(
                    build_revision_row(
                        timestamp=run_ts,
                        dataset_name=dataset_name,
                        source_dataset_name=source_dataset_name,
                        run_kind="transfer" if dataset_name != source_dataset_name else "main",
                        model_name=name,
                        seed=int(seed),
                        artifact_type="latency_profile",
                        tag="main",
                        values={
                            "latency_mean_ms": latency_stats["mean_ms"],
                            "latency_p95_ms": latency_stats["p95_ms"],
                        },
                    )
                )

            def push_sweep(tag: str, metrics: Dict[str, Any], extras: Dict[str, Any]) -> None:
                persist_sweep_row([
                    run_ts,
                    dataset_name,
                    source_dataset_name,
                    "transfer" if dataset_name != source_dataset_name else "main",
                    name,
                    model_path,
                    args.exp,
                    log_path,
                    str(seed),
                    tag,
                    str(extras.get("observation_window", "")),
                    str(extras.get("observation_unit", "")),
                    str(extras.get("obs_minutes", "")),
                    str(extras.get("drop_node", "")),
                    str(extras.get("drop_edge", "")),
                    str(extras.get("edge_drop_rate", "")),
                    str(extras.get("text_noise_std", "")),
                    str(extras.get("text_dropout", "")),
                    f"{extras.get('train_seconds', 0.0):.4f}",
                    f"{extras.get('eval_seconds', 0.0):.4f}",
                    f"{metrics.get('loss', 0.0):.6f}",
                    f"{metrics.get('nll', 0.0):.6f}",
                    f"{metrics.get('acc', 0.0):.6f}",
                    f"{metrics.get('macro_f1', 0.0):.6f}",
                    f"{metrics.get('ece', 0.0):.6f}",
                    "" if extras.get(
                        "calibration_temp", None) is None else f"{extras['calibration_temp']:.6f}",
                ])

            def evaluate_early_stop_setting(
                observation_window: int,
                observation_unit: str,
                loader: DataLoader,
                num_samples: int,
                tau_low_value: float,
                tau_high_value: float,
                gamma_value: float,
                rumor_classes_value: Sequence[int],
                max_batches_value: int,
            ) -> Dict[str, float]:
                total_seen = 0
                resolved = 0
                correct = 0
                mean_prob_accum = 0.0
                mean_uncert_accum = 0.0
                batch_count = 0
                latency_ms: List[float] = []
                resolved_preds: List[Tensor] = []
                resolved_targets: List[Tensor] = []
                resolved_probs: List[Tensor] = []
                for batch in loader:
                    if int(max_batches_value) > 0 and batch_count >= int(max_batches_value):
                        break
                    batch_count += 1
                    profile_cuda_boundary(device)
                    start = time.perf_counter()
                    rumor_prob, rumor_var = trainer.model.predict_with_uncertainty(
                        batch,
                        num_samples=int(num_samples),
                        rumor_classes=list(rumor_classes_value),
                    )
                    profile_cuda_boundary(device)
                    latency_ms.append((time.perf_counter() - start) * 1000.0 / max(1, int(rumor_prob.numel())))
                    decision = torch.full_like(rumor_prob, -1, dtype=torch.long)
                    decision = torch.where(
                        (rumor_prob > float(tau_high_value)) & (rumor_var < float(gamma_value)),
                        torch.ones_like(decision),
                        decision,
                    )
                    decision = torch.where(
                        (rumor_prob < float(tau_low_value)) & (rumor_var < float(gamma_value)),
                        torch.zeros_like(decision),
                        decision,
                    )
                    target_bin = torch.isin(
                        batch.y.view(-1),
                        torch.tensor(list(rumor_classes_value), device=decision.device),
                    ).long().view_as(decision)
                    resolved_mask = decision >= 0
                    resolved += int(resolved_mask.sum().item())
                    total_seen += int(decision.numel())
                    if resolved_mask.any():
                        resolved_decision = decision[resolved_mask]
                        resolved_target = target_bin[resolved_mask]
                        correct += int((resolved_decision == resolved_target).sum().item())
                        resolved_prob = rumor_prob[resolved_mask].clamp(0.0, 1.0)
                        resolved_preds.append(resolved_decision.detach().cpu())
                        resolved_targets.append(resolved_target.detach().cpu())
                        resolved_probs.append(
                            torch.stack(
                                [1.0 - resolved_prob, resolved_prob],
                                dim=1,
                            ).detach().cpu()
                        )
                    mean_prob_accum += float(rumor_prob.mean().item())
                    mean_uncert_accum += float(rumor_var.mean().item())
                latency_stats = summarize_latency_ms(latency_ms)
                if resolved_preds:
                    preds_tensor = torch.cat(resolved_preds, dim=0)
                    targets_tensor = torch.cat(resolved_targets, dim=0)
                    probs_tensor = torch.cat(resolved_probs, dim=0)
                    resolved_macro_f1 = compute_macro_f1(preds_tensor, targets_tensor, num_classes=2)
                    resolved_ece = compute_ece(probs_tensor, targets_tensor, n_bins=int(active_ece_bins))
                else:
                    resolved_macro_f1 = 0.0
                    resolved_ece = 0.0
                return {
                    "coverage": float(resolved / max(1, total_seen)),
                    "acc_resolved": float(correct / max(1, resolved)),
                    "resolved_macro_f1": float(resolved_macro_f1),
                    "resolved_ece": float(resolved_ece),
                    "mean_prob": float(mean_prob_accum / max(1, batch_count)),
                    "mean_var": float(mean_uncert_accum / max(1, batch_count)),
                    "latency_mean_ms": latency_stats["mean_ms"],
                    "latency_p95_ms": latency_stats["p95_ms"],
                    "observation_window": float(observation_window),
                    "observation_unit": str(observation_unit),
                    "obs_minutes": float(observation_window) if str(observation_unit) == "minutes" else float(observation_window),
                }

            if args.sampling_budget_only:
                if not is_rumordiffusion or not hasattr(trainer.model, "predict_with_uncertainty"):
                    raise ValueError("--sampling_budget_only requires a RumorDiffusion checkpoint.")
                early_windows, early_unit = resolve_early_observation_spec(
                    eval_cfg=eval_cfg,
                    model_eval_cfg=rd_eval_cfg,
                    dataset_name=dataset_name,
                )
                early_stop_cfg = rd_eval_cfg.get(
                    "early_stop", {}) if isinstance(rd_eval_cfg, dict) else {}
                early_stop_over = rd_eval_cfg.get(
                    "early_stop_overrides", {}) if isinstance(rd_eval_cfg, dict) else {}
                if isinstance(early_stop_over, dict) and dataset_name in early_stop_over:
                    cfg_over = early_stop_over[dataset_name]
                    if isinstance(cfg_over, dict):
                        early_stop_cfg = {**early_stop_cfg, **cfg_over}
                if not bool(early_stop_cfg.get("enabled", False)):
                    raise ValueError("RumorDiffusion early_stop must be enabled for --sampling_budget_only.")
                sampling_sweep_cfg = rd_eval_cfg.get(
                    "sampling_sweep", {}) if isinstance(rd_eval_cfg, dict) else {}
                if not isinstance(sampling_sweep_cfg, dict) or not bool(sampling_sweep_cfg.get("enabled", False)):
                    raise ValueError("RumorDiffusion sampling_sweep must be enabled for --sampling_budget_only.")
                sampling_windows = sampling_sweep_cfg.get("windows", early_windows)
                sampling_window_overrides = sampling_sweep_cfg.get("windows_overrides", {})
                if isinstance(sampling_window_overrides, dict) and dataset_name in sampling_window_overrides:
                    sampling_windows = sampling_window_overrides[dataset_name]
                if not isinstance(sampling_windows, (list, tuple)) or len(sampling_windows) == 0:
                    raise ValueError("RumorDiffusion sampling_sweep windows are empty.")
                sweep_samples_list = sampling_sweep_cfg.get("samples", [])
                if not isinstance(sweep_samples_list, (list, tuple)) or len(sweep_samples_list) == 0:
                    raise ValueError("RumorDiffusion sampling_sweep samples are empty.")
                tau_low = float(early_stop_cfg.get("tau_low", 0.3))
                tau_high = float(early_stop_cfg.get("tau_high", 0.7))
                gamma_uncert = float(early_stop_cfg.get("gamma", 0.02))
                rumor_classes_cfg = early_stop_cfg.get("rumor_classes", [1, 2, 3])
                early_stop_max_batches = int(early_stop_cfg.get("max_batches", 0))
                sampling_max_batches = int(sampling_sweep_cfg.get("max_batches", early_stop_max_batches))
                for observation_window in sampling_windows:
                    early_test_loader = get_observation_loader("test", int(observation_window), early_unit)
                    legacy_obs_value = int(observation_window) if str(early_unit) == "minutes" else ""
                    early_tag = f"early_{int(observation_window)}_{early_unit}"
                    for sweep_samples in sweep_samples_list:
                        sweep_result = evaluate_early_stop_setting(
                            observation_window=int(observation_window),
                            observation_unit=early_unit,
                            loader=early_test_loader,
                            num_samples=int(sweep_samples),
                            tau_low_value=float(tau_low),
                            tau_high_value=float(tau_high),
                            gamma_value=float(gamma_uncert),
                            rumor_classes_value=rumor_classes_cfg,
                            max_batches_value=sampling_max_batches,
                        )
                        logging.info(
                            f"[SamplingBudget] observation_window={int(observation_window)} {early_unit}, "
                            f"samples={int(sweep_samples)}, coverage={sweep_result['coverage']:.4f}, "
                            f"acc_resolved={sweep_result['acc_resolved']:.4f}, "
                            f"macro_f1_resolved={sweep_result['resolved_macro_f1']:.4f}, "
                            f"mean_var={sweep_result['mean_var']:.8g}, "
                            f"latency_mean_ms={sweep_result['latency_mean_ms']:.4f}"
                        )
                        persist_revision_row(
                            build_revision_row(
                                timestamp=run_ts,
                                dataset_name=dataset_name,
                                source_dataset_name=source_dataset_name,
                                run_kind="transfer" if dataset_name != source_dataset_name else "main",
                                model_name=name,
                                seed=int(seed),
                                artifact_type="sampling_sweep",
                                tag=f"{early_tag}_samples_{int(sweep_samples)}",
                                values={
                                    "observation_window": int(observation_window),
                                    "observation_unit": early_unit,
                                    "obs_minutes": legacy_obs_value,
                                    "samples": int(sweep_samples),
                                    "tau_low": float(tau_low),
                                    "tau_high": float(tau_high),
                                    "gamma": float(gamma_uncert),
                                    **sweep_result,
                                },
                            )
                        )
                continue

            push_sweep("main", test_metrics_base, {
                       "train_seconds": train_seconds, "eval_seconds": eval_seconds})

            test_metrics = test_metrics_base
            calibration_temp = None
            cal_cfg = rd_eval_cfg.get("calibration", {}) if isinstance(
                rd_eval_cfg, dict) else {}
            if (
                is_rumordiffusion
                and isinstance(cal_cfg, dict)
                and cal_cfg.get("enabled", False)
                and not transfer_run
            ):
                calibration_fit_split = str(cal_cfg.get("fit_split", "val")).lower()
                if calibration_fit_split != "val":
                    raise ValueError(
                        f"Unsupported calibration fit split: {calibration_fit_split}. Only 'val' is allowed."
                    )
                val_logits, val_targets = collect_logits(
                    trainer.model, val_loader, device, amp=trainer_cfg.amp)
                temp_opt = tune_temperature(
                    val_logits.to(device),
                    val_targets.to(device),
                    init_temp=float(cal_cfg.get("init_temp", 1.0)),
                    max_iter=int(cal_cfg.get("max_iter", 50)),
                    lr=float(cal_cfg.get("lr", 0.01)),
                )
                test_logits = test_outputs_base["logits"]
                test_targets = test_outputs_base["targets"]
                logits_scaled = (test_logits.to(device=device, dtype=torch.float32) / float(temp_opt)).cpu()
                if not calibration_preserves_non_tied_argmax(
                    original_logits=test_logits,
                    scaled_logits=logits_scaled,
                ):
                    raise RuntimeError(
                        "Temperature scaling changed predicted labels on non-tied stored test logits."
                    )
                test_metrics_cal = metrics_from_logits(
                    logits_scaled,
                    test_targets,
                    num_classes=num_classes,
                    detailed=detailed_eval,
                    class_names=class_names,
                    compute_calibration=True,
                    ece_bins=int(active_ece_bins),
                )
                logging.info(
                    f"[Calibration] Temperature={temp_opt:.3f}; "
                    f"Test acc={test_metrics_cal.get('acc', 0.0):.4f}, "
                    f"macro_f1={test_metrics_cal.get('macro_f1', 0.0):.4f}, "
                    f"ECE={test_metrics_cal.get('ece', 0.0):.4f}"
                )
                test_metrics = test_metrics_cal
                reported_logits = logits_scaled
                reported_targets = test_targets
                calibration_temp = float(temp_opt)
                push_sweep("calibrated", test_metrics_cal, {
                           "calibration_temp": calibration_temp})
            elif (
                is_rumordiffusion
                and isinstance(cal_cfg, dict)
                and cal_cfg.get("enabled", False)
                and transfer_run
            ):
                logging.info(
                    "Source-only transfer disables temperature fitting because target validation labels "
                    "must not influence reported target metrics."
                )
            if canonical_outputs_cfg:
                if transfer_run:
                    raise ValueError("canonical publishing is disabled for transfer runs; use the transfer runner")
                canonical_output_root = str(canonical_outputs_cfg.get("root", "Output"))
                canonical_protocol_id = str(canonical_outputs_cfg.get("protocol_id", "rumordiffusion_canonical_v1"))
                canonical_horizon = resolve_canonical_horizon(prefix_cfg)
                canonical_stage_seeds = resolve_canonical_stage_seeds(exp_cfg, int(seed))
                resolved_canonical_checkpoint = actual_ckpt_path if len(actual_ckpt_path) > 0 else ckpt_path
                if not os.path.isfile(resolved_canonical_checkpoint):
                    raise FileNotFoundError(f"Canonical checkpoint missing: {resolved_canonical_checkpoint}")
                raw_manifest_path = os.path.join(dataset_root, "raw_manifest.json")
                processed_manifest_path = os.path.join(dataset_root, "processed_manifest.json")
                split_manifest_path = os.path.join(dataset_root, "split_manifest.json")
                canonical_temperature = float(calibration_temp if calibration_temp is not None else 1.0)
                canonical_rows = prediction_rows_from_logits(
                    run_id=canonical_run_id(
                        run_inputs={
                            "dataset": dataset_name,
                            "split": "test",
                            "horizon": canonical_horizon,
                            "model": {"name": name, "variant": str(model_cfg.get("variant", name))},
                            "seeds": canonical_stage_seeds,
                            "calibration_tag": "val_temperature" if calibration_temp is not None else "identity",
                            "sampling_S": 1,
                        },
                        exp_config=exp_cfg,
                        model_config=model_cfg,
                        raw_manifest_path=raw_manifest_path,
                        processed_manifest_path=processed_manifest_path,
                        split_manifest_path=split_manifest_path,
                    ),
                    dataset=dataset_name,
                    split="test",
                    source_ids=test_outputs_base["source_ids"],
                    logits=test_outputs_base["logits"],
                    targets=test_outputs_base["targets"],
                    horizon_unit=str(canonical_horizon["unit"]),
                    horizon_value=float(canonical_horizon["value"]),
                    model=name,
                    model_variant=str(model_cfg.get("variant", name)),
                    seeds=canonical_stage_seeds,
                    calibration_tag="val_temperature" if calibration_temp is not None else "identity",
                    sampling_S=1,
                    checkpoint_sha256=file_sha256(resolved_canonical_checkpoint),
                    config_sha256=artifact_canonical_json_hash(
                        omit_optional_none_fields({"experiment": exp_cfg, "model": model_cfg})
                    ),
                    split_sha256=file_sha256(split_manifest_path),
                    temperature=canonical_temperature,
                    compute_latency_ms=test_outputs_base["latency_ms"],
                )
                latency_values = [float(value) for value in test_outputs_base["latency_ms"]]
                if latency_values:
                    latency_summary = summarize_latency_ms(latency_values)
                    latency_payload = {
                        "schema_version": "1",
                        "measured": True,
                        "compute_latency_ms": {
                            "n": len(latency_values),
                            "mean_ms": float(latency_summary["mean_ms"]),
                            "p50_ms": percentile(sorted(latency_values), 0.50),
                            "p95_ms": float(latency_summary["p95_ms"]),
                        },
                        "detection_latency": {
                            "measured": False,
                            "unit": canonical_horizon["unit"],
                            "value": float(canonical_horizon["value"]),
                        },
                    }
                else:
                    latency_payload = {
                        "schema_version": "1",
                        "measured": False,
                        "status": "not_profiled",
                        "compute_latency_ms": {"n": 0},
                        "detection_latency": {
                            "measured": False,
                            "unit": canonical_horizon["unit"],
                            "value": float(canonical_horizon["value"]),
                        },
                    }
                canonical_probability = float(test_metrics.get("acc", 0.0))
                selective_payload = {
                    "schema_version": "1",
                    "policy": "accept_all",
                    "threshold_fit_split": "val",
                    "coverage": 1.0,
                    "selective_risk": float(1.0 - canonical_probability),
                    "resolved_accuracy": canonical_probability,
                    "resolved_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
                    "risk_coverage_curve": [{"coverage": 1.0, "risk": float(1.0 - canonical_probability)}],
                }
                calibration_payload = {
                    "schema_version": "1",
                    "method": "temperature" if calibration_temp is not None else "identity",
                    "fit_split": "val",
                    "threshold_fit_split": "val",
                    "temperature": canonical_temperature,
                    "status": "fitted" if calibration_temp is not None else "identity",
                }
                canonical_run_finished_at = datetime.now().astimezone().isoformat()
                write_canonical_run(
                    canonical_output_root,
                    protocol_id=canonical_protocol_id,
                    run_inputs={
                        "dataset": dataset_name,
                        "split": "test",
                        "horizon": canonical_horizon,
                        "model": {"name": name, "variant": str(model_cfg.get("variant", name))},
                        "seeds": canonical_stage_seeds,
                        "calibration_tag": "val_temperature" if calibration_temp is not None else "identity",
                        "sampling_S": 1,
                        "execution": {
                            "start_time": run_started_at,
                            "end_time": canonical_run_finished_at,
                            "exit_code": 0,
                            "command": " ".join(sys.argv),
                            "log_path": log_path,
                        },
                        "environment": {
                            "batch_size": int(eval_batch_size),
                            "train_batch_size": int(trainer_cfg.batch_size),
                            "amp": bool(trainer_cfg.amp),
                            "device": device,
                            "model_path": model_path,
                            "checkpoint_path": resolved_canonical_checkpoint,
                            "dataset_root": dataset_root,
                        },
                    },
                    checkpoint_path=resolved_canonical_checkpoint,
                    exp_config=exp_cfg,
                    model_config=model_cfg,
                    split_manifest_path=split_manifest_path,
                    raw_manifest_path=raw_manifest_path,
                    processed_manifest_path=processed_manifest_path,
                    prediction_rows=canonical_rows,
                    num_classes=num_classes,
                    calibration=calibration_payload,
                    selective_metrics=selective_payload,
                    latency=latency_payload,
                    repo_root=".",
                )
            if bool(isinstance(export_pred_cfg, dict) and export_pred_cfg.get("enabled", False)):
                export_prediction_rows(
                    csv_path=prediction_csv,
                    dataset_name=dataset_name,
                    source_dataset_name=source_dataset_name,
                    run_kind="transfer" if dataset_name != source_dataset_name else "main",
                    model_name=name,
                    seed=int(seed),
                    tag="main",
                    source_ids=test_outputs_base["source_ids"],
                    logits=reported_logits,
                    targets=reported_targets,
                )
            if bool(isinstance(significance_cfg, dict) and significance_cfg.get("enabled", False)) and bool(isinstance(export_pred_cfg, dict) and export_pred_cfg.get("enabled", False)):
                compare_models = [str(model).lower() for model in significance_cfg.get("baseline_models", [])]
                reference_model = str(significance_cfg.get("reference_model", "")).lower()
                metric_names = significance_cfg.get("metrics", ["macro_f1", "acc"])
                bootstrap_rounds = int(significance_cfg.get("n_bootstrap", 1000))
                active_run_kind = "transfer" if dataset_name != source_dataset_name else "main"
                if len(reference_model) > 0 and name != reference_model:
                    persist_significance_rows(
                        prediction_csv=prediction_csv,
                        revision_artifacts_csv=revision_artifacts_csv,
                        revision_headers=revision_headers,
                        timestamp=run_ts,
                        dataset_name=dataset_name,
                        source_dataset_name=source_dataset_name,
                        run_kind=active_run_kind,
                        reference_model=reference_model,
                        compare_model=name,
                        seed=int(seed),
                        metric_names=metric_names,
                        bootstrap_rounds=bootstrap_rounds,
                        model_cfg_path=model_path,
                        exp_cfg_path=args.exp,
                        log_path=log_path,
                    )
                else:
                    for compare_model in compare_models:
                        persist_significance_rows(
                            prediction_csv=prediction_csv,
                            revision_artifacts_csv=revision_artifacts_csv,
                            revision_headers=revision_headers,
                            timestamp=run_ts,
                            dataset_name=dataset_name,
                            source_dataset_name=source_dataset_name,
                            run_kind=active_run_kind,
                            reference_model=name,
                            compare_model=str(compare_model),
                            seed=int(seed),
                            metric_names=metric_names,
                            bootstrap_rounds=bootstrap_rounds,
                            model_cfg_path=model_path,
                            exp_cfg_path=args.exp,
                            log_path=log_path,
                        )

            result_headers = [
                "timestamp",
                "dataset",
                "source_dataset",
                "run_kind",
                "transfer_mode",
                "model",
                "model_cfg",
                "exp_cfg",
                "ckpt_path",
                "source_ckpt_path",
                "adapt_ckpt_path",
                "log_path",
                "seed",
                "adapt_epochs",
                "pseudo_label_coverage",
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
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            resolved_ckpt_path = adapt_ckpt_path if transfer_mode == "target_adapt" and len(adapt_ckpt_path) > 0 else actual_ckpt_path
            if transfer_mode == "target_adapt":
                if len(adapt_ckpt_path) == 0 or not os.path.isfile(adapt_ckpt_path):
                    raise RuntimeError(
                        f"Target adaptation completed without producing checkpoint: {adapt_ckpt_path}"
                    )
            append_dict_rows_csv(
                csv_path=csv_path,
                headers=result_headers,
                rows=[{
                    "timestamp": ts,
                    "dataset": dataset_name,
                    "source_dataset": source_dataset_name,
                    "run_kind": "transfer" if dataset_name != source_dataset_name else "main",
                    "transfer_mode": transfer_mode,
                    "model": name,
                    "model_cfg": model_path,
                    "exp_cfg": args.exp,
                    "ckpt_path": resolved_ckpt_path,
                    "source_ckpt_path": source_checkpoint_path,
                    "adapt_ckpt_path": adapt_ckpt_path,
                    "log_path": log_path,
                    "seed": str(seed),
                    "adapt_epochs": "" if int(adapt_epochs) <= 0 else str(int(adapt_epochs)),
                    "pseudo_label_coverage": "" if float(pseudo_label_coverage) <= 0.0 else f"{float(pseudo_label_coverage):.6f}",
                    "best_val_macro_f1": f"{fit_out.get('best_val_macro_f1', fit_out.get('best_val_metric', 0.0)):.6f}",
                    "test_loss": f"{test_metrics.get('loss', 0.0):.6f}",
                    "test_nll": f"{test_metrics.get('nll', 0.0):.6f}",
                    "test_acc": f"{test_metrics.get('acc', 0.0):.6f}",
                    "test_macro_f1": f"{test_metrics.get('macro_f1', 0.0):.6f}",
                    "test_macro_precision": f"{test_metrics.get('macro_precision', 0.0):.6f}",
                    "test_macro_recall": f"{test_metrics.get('macro_recall', 0.0):.6f}",
                    "test_weighted_f1": f"{test_metrics.get('weighted_f1', 0.0):.6f}",
                    "test_weighted_precision": f"{test_metrics.get('weighted_precision', 0.0):.6f}",
                    "test_weighted_recall": f"{test_metrics.get('weighted_recall', 0.0):.6f}",
                    "ece": f"{test_metrics.get('ece', 0.0):.6f}",
                    "calibration_temp": "" if calibration_temp is None else f"{calibration_temp:.6f}",
                }],
            )
            if dataset_name != source_dataset_name:
                transfer_headers = [
                    "timestamp",
                    "train_dataset",
                    "test_dataset",
                    "transfer_mode",
                    "model",
                    "seed",
                    "ckpt_path",
                    "source_ckpt_path",
                    "adapt_ckpt_path",
                    "adapt_epochs",
                    "pseudo_label_coverage",
                    "acc",
                    "macro_f1",
                    "nll",
                    "ece",
                    "calibration_temp",
                ]
                append_dict_rows_csv(
                    csv_path=transfer_csv,
                    headers=transfer_headers,
                    rows=[{
                        "timestamp": ts,
                        "train_dataset": source_dataset_name,
                        "test_dataset": dataset_name,
                        "transfer_mode": transfer_mode,
                        "model": name,
                        "seed": str(seed),
                        "ckpt_path": resolved_ckpt_path,
                        "source_ckpt_path": source_checkpoint_path,
                        "adapt_ckpt_path": adapt_ckpt_path,
                        "adapt_epochs": "" if int(adapt_epochs) <= 0 else str(int(adapt_epochs)),
                        "pseudo_label_coverage": "" if float(pseudo_label_coverage) <= 0.0 else f"{float(pseudo_label_coverage):.6f}",
                        "acc": f"{test_metrics.get('acc', 0.0):.6f}",
                        "macro_f1": f"{test_metrics.get('macro_f1', 0.0):.6f}",
                        "nll": f"{test_metrics.get('nll', 0.0):.6f}",
                        "ece": f"{test_metrics.get('ece', 0.0):.6f}",
                        "calibration_temp": "" if calibration_temp is None else f"{calibration_temp:.6f}",
                    }],
                )
            if wb_enabled:
                import wandb

                wandb.log({
                    "test/loss": float(test_metrics.get("loss", 0.0)),
                    "test/acc": float(test_metrics.get("acc", 0.0)),
                    "test/macro_f1": float(test_metrics.get("macro_f1", 0.0)),
                })
                wandb.finish()

            if bool(args.transfer_metric_only) and dataset_name != source_dataset_name:
                logging.info(
                    "Transfer metric-only evaluation completed after result and transfer CSV export."
                )
                continue

            early_windows, early_unit = resolve_early_observation_spec(
                eval_cfg=eval_cfg,
                model_eval_cfg=rd_eval_cfg,
                dataset_name=dataset_name,
            )
            early_stop_cfg = rd_eval_cfg.get(
                "early_stop", {}) if isinstance(rd_eval_cfg, dict) else {}
            early_stop_over = rd_eval_cfg.get(
                "early_stop_overrides", {}) if isinstance(rd_eval_cfg, dict) else {}
            if isinstance(early_stop_over, dict) and dataset_name in early_stop_over:
                cfg_over = early_stop_over[dataset_name]
                if isinstance(cfg_over, dict):
                    early_stop_cfg = {**early_stop_cfg, **cfg_over}
            early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
            early_stop_samples = int(early_stop_cfg.get(
                "samples", 4)) if early_stop_enabled else 0
            tau_low = float(early_stop_cfg.get("tau_low", 0.3)
                            ) if early_stop_enabled else 0.0
            tau_high = float(early_stop_cfg.get("tau_high", 0.7)
                             ) if early_stop_enabled else 0.0
            gamma_uncert = float(early_stop_cfg.get(
                "gamma", 0.02)) if early_stop_enabled else 0.0
            rumor_classes_cfg = early_stop_cfg.get(
                "rumor_classes", [1, 2, 3]) if early_stop_enabled else []
            early_stop_max_batches = int(early_stop_cfg.get(
                "max_batches", 0)) if isinstance(early_stop_cfg, dict) else 0
            sampling_sweep_cfg = rd_eval_cfg.get(
                "sampling_sweep", {}) if isinstance(rd_eval_cfg, dict) else {}
            threshold_sweep_cfg = rd_eval_cfg.get(
                "threshold_sweep", {}) if isinstance(rd_eval_cfg, dict) else {}
            if isinstance(early_windows, (list, tuple)) and len(early_windows) > 0:
                for observation_window in early_windows:
                    early_test_loader = get_observation_loader("test", int(observation_window), early_unit)
                    early_outputs = collect_eval_outputs(
                        trainer.model,
                        early_test_loader,
                        device=device,
                        criterion=trainer.criterion,
                        amp=trainer_cfg.amp,
                        detailed=detailed_eval,
                        class_names=class_names,
                        compute_calibration=True,
                        ece_bins=int(active_ece_bins),
                    )
                    metrics_early = early_outputs["metrics"]
                    early_reported_logits = early_outputs["logits"]
                    early_reported_targets = early_outputs["targets"]
                    early_calibration_temp = None
                    early_tag = f"early_{int(observation_window)}_{early_unit}"
                    legacy_obs_value = int(observation_window) if str(early_unit) == "minutes" else ""
                    if is_rumordiffusion and isinstance(cal_cfg, dict) and cal_cfg.get("enabled", False):
                        push_sweep(
                            f"{early_tag}_uncalibrated",
                            metrics_early,
                            {
                                "observation_window": int(observation_window),
                                "observation_unit": early_unit,
                                "obs_minutes": legacy_obs_value,
                            },
                        )
                        if bool(isinstance(export_pred_cfg, dict) and export_pred_cfg.get("enabled", False)):
                            export_prediction_rows(
                                csv_path=prediction_csv,
                                dataset_name=dataset_name,
                                source_dataset_name=source_dataset_name,
                                run_kind="transfer" if dataset_name != source_dataset_name else "main",
                                model_name=name,
                                seed=int(seed),
                                tag=f"{early_tag}_uncalibrated",
                                source_ids=early_outputs["source_ids"],
                                logits=early_reported_logits,
                                targets=early_reported_targets,
                            )
                        early_val_loader = get_observation_loader("val", int(observation_window), early_unit)
                        val_logits_early, val_targets_early = collect_logits(
                            trainer.model, early_val_loader, device, amp=trainer_cfg.amp
                        )
                        temp_opt_early = tune_temperature(
                            val_logits_early.to(device),
                            val_targets_early.to(device),
                            init_temp=float(cal_cfg.get("init_temp", 1.0)),
                            max_iter=int(cal_cfg.get("max_iter", 50)),
                            lr=float(cal_cfg.get("lr", 0.01)),
                        )
                        scaled_logits_early = (early_outputs["logits"].to(device=device, dtype=torch.float32) / float(temp_opt_early)).cpu()
                        metrics_early_cal = metrics_from_logits(
                            scaled_logits_early,
                            early_outputs["targets"],
                            num_classes=num_classes,
                            detailed=detailed_eval,
                            class_names=class_names,
                            compute_calibration=True,
                            ece_bins=int(active_ece_bins),
                        )
                        metrics_early_cal["loss"] = float(metrics_early.get("loss", 0.0))
                        metrics_early = metrics_early_cal
                        early_reported_logits = scaled_logits_early
                        early_reported_targets = early_outputs["targets"]
                        early_calibration_temp = float(temp_opt_early)
                    push_sweep(
                        early_tag,
                        metrics_early,
                        {
                            "observation_window": int(observation_window),
                            "observation_unit": early_unit,
                            "obs_minutes": legacy_obs_value,
                            "calibration_temp": early_calibration_temp,
                        },
                    )
                    if bool(isinstance(export_pred_cfg, dict) and export_pred_cfg.get("enabled", False)):
                        export_prediction_rows(
                            csv_path=prediction_csv,
                            dataset_name=dataset_name,
                            source_dataset_name=source_dataset_name,
                            run_kind="transfer" if dataset_name != source_dataset_name else "main",
                            model_name=name,
                            seed=int(seed),
                            tag=early_tag,
                            source_ids=early_outputs["source_ids"],
                            logits=early_reported_logits,
                            targets=early_reported_targets,
                        )
                    if is_rumordiffusion and early_stop_enabled and hasattr(trainer.model, "predict_with_uncertainty"):
                        early_stop_result = evaluate_early_stop_setting(
                            observation_window=int(observation_window),
                            observation_unit=early_unit,
                            loader=early_test_loader,
                            num_samples=int(early_stop_samples),
                            tau_low_value=float(tau_low),
                            tau_high_value=float(tau_high),
                            gamma_value=float(gamma_uncert),
                            rumor_classes_value=rumor_classes_cfg,
                            max_batches_value=int(early_stop_max_batches),
                        )
                        logging.info(
                            f"[EarlyStop] observation_window={int(observation_window)} {early_unit}, samples={early_stop_samples}, "
                            f"coverage={early_stop_result['coverage']:.4f}, acc_resolved={early_stop_result['acc_resolved']:.4f}, "
                            f"macro_f1_resolved={early_stop_result['resolved_macro_f1']:.4f}, "
                            f"ece_resolved={early_stop_result['resolved_ece']:.4f}, "
                            f"mean_prob={early_stop_result['mean_prob']:.4f}, mean_var={early_stop_result['mean_var']:.6g}, "
                            f"latency_mean_ms={early_stop_result['latency_mean_ms']:.4f}, latency_p95_ms={early_stop_result['latency_p95_ms']:.4f}"
                        )
                        persist_revision_row(
                            build_revision_row(
                                timestamp=run_ts,
                                dataset_name=dataset_name,
                                source_dataset_name=source_dataset_name,
                                run_kind="transfer" if dataset_name != source_dataset_name else "main",
                                model_name=name,
                                seed=int(seed),
                                artifact_type="early_stop",
                                tag=early_tag,
                                values={
                                    "observation_window": int(observation_window),
                                    "observation_unit": early_unit,
                                    "obs_minutes": legacy_obs_value,
                                    "samples": int(early_stop_samples),
                                    "tau_low": float(tau_low),
                                    "tau_high": float(tau_high),
                                    "gamma": float(gamma_uncert),
                                    **early_stop_result,
                                },
                            )
                        )
                        if isinstance(sampling_sweep_cfg, dict) and bool(sampling_sweep_cfg.get("enabled", False)):
                            for sweep_samples in sampling_sweep_cfg.get("samples", []):
                                sweep_result = evaluate_early_stop_setting(
                                    observation_window=int(observation_window),
                                    observation_unit=early_unit,
                                    loader=early_test_loader,
                                    num_samples=int(sweep_samples),
                                    tau_low_value=float(tau_low),
                                    tau_high_value=float(tau_high),
                                    gamma_value=float(gamma_uncert),
                                    rumor_classes_value=rumor_classes_cfg,
                                    max_batches_value=int(sampling_sweep_cfg.get("max_batches", early_stop_max_batches)),
                                )
                                persist_revision_row(
                                    build_revision_row(
                                        timestamp=run_ts,
                                        dataset_name=dataset_name,
                                        source_dataset_name=source_dataset_name,
                                        run_kind="transfer" if dataset_name != source_dataset_name else "main",
                                        model_name=name,
                                        seed=int(seed),
                                        artifact_type="sampling_sweep",
                                        tag=f"{early_tag}_samples_{int(sweep_samples)}",
                                        values={
                                            "observation_window": int(observation_window),
                                            "observation_unit": early_unit,
                                            "obs_minutes": legacy_obs_value,
                                            "samples": int(sweep_samples),
                                            "tau_low": float(tau_low),
                                            "tau_high": float(tau_high),
                                            "gamma": float(gamma_uncert),
                                            **sweep_result,
                                        },
                                    )
                                )
                        if isinstance(threshold_sweep_cfg, dict) and bool(threshold_sweep_cfg.get("enabled", False)):
                            tau_low_list = threshold_sweep_cfg.get("tau_low", [])
                            tau_high_list = threshold_sweep_cfg.get("tau_high", [])
                            gamma_list = threshold_sweep_cfg.get("gamma", [])
                            for tau_low_value in tau_low_list:
                                for tau_high_value in tau_high_list:
                                    if float(tau_low_value) >= float(tau_high_value):
                                        continue
                                    for gamma_value in gamma_list:
                                        sweep_result = evaluate_early_stop_setting(
                                            observation_window=int(observation_window),
                                            observation_unit=early_unit,
                                            loader=early_test_loader,
                                            num_samples=int(threshold_sweep_cfg.get("samples", early_stop_samples)),
                                            tau_low_value=float(tau_low_value),
                                            tau_high_value=float(tau_high_value),
                                            gamma_value=float(gamma_value),
                                            rumor_classes_value=rumor_classes_cfg,
                                            max_batches_value=int(threshold_sweep_cfg.get("max_batches", early_stop_max_batches)),
                                        )
                                        persist_revision_row(
                                            build_revision_row(
                                                timestamp=run_ts,
                                                dataset_name=dataset_name,
                                                source_dataset_name=source_dataset_name,
                                                run_kind="transfer" if dataset_name != source_dataset_name else "main",
                                                model_name=name,
                                                seed=int(seed),
                                                artifact_type="threshold_sweep",
                                                tag=f"{early_tag}_tau{float(tau_low_value):.2f}_{float(tau_high_value):.2f}_gamma{float(gamma_value):.2f}",
                                                values={
                                                    "observation_window": int(observation_window),
                                                    "observation_unit": early_unit,
                                                    "obs_minutes": legacy_obs_value,
                                                    "samples": int(threshold_sweep_cfg.get("samples", early_stop_samples)),
                                                    "tau_low": float(tau_low_value),
                                                    "tau_high": float(tau_high_value),
                                                    "gamma": float(gamma_value),
                                                    **sweep_result,
                                                },
                                            )
                                        )
            robust_cfg = rd_eval_cfg.get("robustness", {}) if isinstance(
                rd_eval_cfg, dict) else {}
            drop_sweep = robust_cfg.get("edge_dropout", []) if isinstance(
                robust_cfg, dict) else []
            if is_rumordiffusion and hasattr(m, "ctdm") and isinstance(drop_sweep, (list, tuple)) and len(drop_sweep) > 0:
                edge_backup = float(getattr(m.ctdm, "edge_drop_rate", 0.0))
                for p in drop_sweep:
                    m.ctdm.edge_drop_rate = float(p)
                    metrics_drop = trainer.evaluate(
                        test_loader, detailed=True, class_names=class_names)
                    push_sweep(f"robust_edge_{p}", metrics_drop, {
                               "edge_drop_rate": p})
                m.ctdm.edge_drop_rate = edge_backup

            text_noise_list = robust_text_cfg.get(
                "noise_std", []) if isinstance(robust_text_cfg, dict) else []
            text_drop_list = robust_text_cfg.get(
                "dropout_p", []) if isinstance(robust_text_cfg, dict) else []
            if is_rumordiffusion and hasattr(m, "ctdm") and (len(text_noise_list) > 0 or len(text_drop_list) > 0):
                noise_candidates = text_noise_list if len(
                    text_noise_list) > 0 else [0.0]
                drop_candidates = text_drop_list if len(
                    text_drop_list) > 0 else [0.0]
                noise_backup = float(getattr(m.ctdm, "text_noise_std", 0.0))
                drop_backup = float(getattr(m.ctdm, "text_dropout_p", 0.0))
                for n_std in noise_candidates:
                    for d_p in drop_candidates:
                        m.ctdm.text_noise_std = float(n_std)
                        m.ctdm.text_dropout_p = float(d_p)
                        metrics_text_rob = trainer.evaluate(
                            test_loader, detailed=True, class_names=class_names)
                        push_sweep(f"text_robust_noise{n_std}_drop{d_p}", metrics_text_rob, {
                                   "text_noise_std": n_std, "text_dropout": d_p})
                m.ctdm.text_noise_std = noise_backup
                m.ctdm.text_dropout_p = drop_backup

            drop_node_list = observability_sweep_cfg.get(
                "drop_node", []) if isinstance(observability_sweep_cfg, dict) else []
            drop_edge_list = observability_sweep_cfg.get(
                "drop_edge", []) if isinstance(observability_sweep_cfg, dict) else []
            if len(drop_node_list) > 0 or len(drop_edge_list) > 0:
                dn_candidates = drop_node_list if len(drop_node_list) > 0 else [
                    observability_cfg.get("drop_node", 0.0)]
                de_candidates = drop_edge_list if len(drop_edge_list) > 0 else [
                    observability_cfg.get("drop_edge", 0.0)]
                for dn in dn_candidates:
                    for de in de_candidates:
                        ds_test_obs = make_graph_dataset(
                            root=dataset_root,
                            split="test",
                            use_lp_edges=use_lp,
                            use_potentials=False,
                            prefix_cfg=prefix_cfg,
                            observability_cfg={"drop_node": float(dn), "drop_edge": float(
                                de), "seed": observability_cfg.get("seed", 0)},
                            text_runtime_cfg=text_runtime_cfg,
                        )
                        test_loader_obs = make_loader(ds_test_obs, shuffle=False)
                        metrics_obs = trainer.evaluate(
                            test_loader_obs, detailed=True, class_names=class_names)
                        push_sweep(f"robust_obs_node{dn}_edge{de}", metrics_obs, {
                                   "drop_node": dn, "drop_edge": de})
            seed_results.append(
                {
                    "val_macro_f1": float(fit_out.get("best_val_macro_f1", fit_out.get("best_val_metric", 0.0))),
                    "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
                    "test_acc": float(test_metrics.get("acc", 0.0)),
                    "test_nll": float(test_metrics.get("nll", 0.0)),
                }
            )


if __name__ == "__main__":
    main()
