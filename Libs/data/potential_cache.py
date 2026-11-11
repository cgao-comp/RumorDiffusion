"""Strict cache contract for RumorDiffusion class-potential tensors."""

import hashlib
import json
import os
import re
from typing import Any, Dict, Mapping

import torch
from torch import Tensor


POTENTIAL_CACHE_SCHEMA_VERSION = 2
_METADATA_FIELDS = (
    "schema_version",
    "dataset",
    "source_id",
    "seed",
    "prefix_unit",
    "prefix_window",
    "ctdm_config_hash",
)


def _canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prefix_identity(prefix_cfg: Mapping[str, Any]) -> tuple[str, int | str]:
    unit = str(prefix_cfg.get("strategy", prefix_cfg.get("unit", "none"))).lower()
    window = prefix_cfg.get("window", prefix_cfg.get("minutes", prefix_cfg.get("k", "all")))
    if window in (None, ""):
        window = "all"
    elif isinstance(window, bool):
        raise ValueError("Potential-cache prefix window must not be boolean.")
    else:
        window = int(window)
    return unit, window


def build_potential_cache_metadata(
    dataset_name: str,
    seed: int,
    prefix_cfg: Mapping[str, Any],
    ctdm_config_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Returns metadata shared by every cache entry for one CTDM/prefix run."""
    prefix_unit, prefix_window = _prefix_identity(prefix_cfg)
    return {
        "schema_version": POTENTIAL_CACHE_SCHEMA_VERSION,
        "dataset": str(dataset_name),
        "seed": int(seed),
        "prefix_unit": prefix_unit,
        "prefix_window": prefix_window,
        "ctdm_config_hash": _canonical_json_hash(ctdm_config_payload),
    }


def potential_cache_dir(root_dir: str, prefix_cfg: Mapping[str, Any]) -> str:
    """Returns a deterministic, prefix-specific subdirectory below ``root_dir``."""
    prefix_unit, prefix_window = _prefix_identity(prefix_cfg)
    safe_unit = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix_unit).strip("_") or "none"
    safe_window = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix_window)).strip("_") or "all"
    return os.path.join(str(root_dir), f"prefix_{safe_unit}_{safe_window}")


def potential_cache_payload(h: Tensor, metadata: Mapping[str, Any], source_id: object) -> Dict[str, Any]:
    """Builds a schema-v2 cache payload after validating the potential shape."""
    if not torch.is_tensor(h) or h.dim() != 2 or int(h.size(0)) <= 0 or int(h.size(1)) <= 0:
        raise ValueError("RumorDiffusion potential cache requires h with shape [C, H].")
    resolved_metadata = dict(metadata)
    resolved_metadata["source_id"] = str(source_id)
    missing = [field for field in _METADATA_FIELDS if field not in resolved_metadata]
    if missing:
        raise ValueError(f"RumorDiffusion potential cache metadata is incomplete: missing {missing}.")
    if int(resolved_metadata["schema_version"]) != POTENTIAL_CACHE_SCHEMA_VERSION:
        raise ValueError("RumorDiffusion potential cache metadata has an invalid schema version.")
    return {"h": h.detach().cpu(), "metadata": resolved_metadata}


def load_validated_potential_cache(
    payload: object,
    expected_metadata: Mapping[str, Any],
    source_id: object,
    cache_path: str,
) -> Tensor:
    """Validates and returns one cache tensor; legacy or mismatched entries fail closed."""
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy potential cache is not accepted: {cache_path}")
    if "h" not in payload or "metadata" not in payload:
        raise ValueError(f"Potential cache must contain h and metadata: {cache_path}")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Potential cache metadata must be a mapping: {cache_path}")
    resolved_expected = dict(expected_metadata)
    resolved_expected["source_id"] = str(source_id)
    missing = [field for field in _METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"Potential cache metadata is incomplete at {cache_path}: missing {missing}.")
    for field in _METADATA_FIELDS:
        if metadata[field] != resolved_expected.get(field):
            raise ValueError(
                "Potential cache metadata mismatch at "
                f"{cache_path}: {field}={metadata[field]!r}, expected {resolved_expected.get(field)!r}."
            )
    h = payload["h"]
    if not torch.is_tensor(h) or h.dim() != 2 or int(h.size(0)) <= 0 or int(h.size(1)) <= 0:
        raise ValueError(f"Potential cache h must have shape [C, H]: {cache_path}")
    if not bool(torch.isfinite(h).all().item()):
        raise ValueError(f"Potential cache contains non-finite values: {cache_path}")
    return h.to(torch.float32)
