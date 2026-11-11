import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str, payload: object) -> None:
    """Writes a deterministic UTF-8 JSON artifact and creates its parent."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _class_distribution(labels: List[int]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for value in labels:
        key = str(int(value))
        distribution[key] = distribution.get(key, 0) + 1
    return dict(sorted(distribution.items()))


def _label_schema_payload(dataset_name: str, label_cfg: "LabelConfig") -> Dict[str, object]:
    mapping = {str(key): int(value) for key, value in label_cfg.mapping.items()}
    values = sorted(set(mapping.values()))
    if values != list(range(len(values))):
        raise ValueError(f"Label mapping for {dataset_name} must use contiguous IDs from zero: {mapping}")
    return {
        "schema_version": "1",
        "dataset": str(dataset_name),
        "allowed": [str(value) for value in label_cfg.allowed],
        "mapping": mapping,
        "num_classes": len(values),
        "skip_unknown": bool(label_cfg.skip_unknown),
        "numeric": bool(label_cfg.numeric),
        "mapping_sha256": _sha256_json(mapping),
    }


def _write_dataset_manifests(
    dataset_name: str,
    out_root: str,
    raw_dir: str,
    data_list: List[Data],
    labels: List[int],
    splits: Dict[str, List[int]],
    split_seed: int,
    split_strategy: str,
    label_cfg: "LabelConfig",
    time_unit: str,
    generation_command: str,
) -> None:
    """Writes immutable raw/processed/split manifests for one processed dataset."""
    dataset_dir = os.path.join(out_root, dataset_name)
    data_path = os.path.join(dataset_dir, "pyg", "data.pt")
    slices_path = os.path.join(dataset_dir, "pyg", "slices.pt")
    splits_path = os.path.join(dataset_dir, "splits.json")
    source_ids = [str(getattr(data, "source_id")) for data in data_list]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"Duplicate source_id while processing {dataset_name}")
    validate_split_integrity(splits, source_ids)

    label_schema = _label_schema_payload(dataset_name, label_cfg)
    _write_json(os.path.join(dataset_dir, "label_schema.json"), label_schema)

    raw_manifest = {
        "schema_version": "1",
        "dataset": str(dataset_name),
        "raw_root": os.path.abspath(raw_dir),
        "files": _directory_manifest(raw_dir),
        "source_id_sha256": _sha256_json(source_ids),
        "label_schema_sha256": _sha256_json(label_schema),
    }
    raw_manifest_path = os.path.join(dataset_dir, "raw_manifest.json")
    _write_json(raw_manifest_path, raw_manifest)

    processed_manifest = {
        "schema_version": "1",
        "dataset": str(dataset_name),
        "processed_root": os.path.abspath(dataset_dir),
        "data_sha256": _sha256_file(data_path),
        "slices_sha256": _sha256_file(slices_path),
        "sample_count": len(data_list),
        "source_id_sha256": _sha256_json(source_ids),
        "class_distribution": _class_distribution(labels),
        "time_unit": str(time_unit),
        "label_schema_sha256": _sha256_json(label_schema),
        "raw_manifest_sha256": _sha256_file(raw_manifest_path),
    }
    processed_manifest_path = os.path.join(dataset_dir, "processed_manifest.json")
    _write_json(processed_manifest_path, processed_manifest)

    split_manifest = {
        "schema_version": "1",
        "dataset": str(dataset_name),
        "split_seed": int(split_seed),
        "split_strategy": str(split_strategy),
        "generation_command": str(generation_command),
        "splits_sha256": _sha256_file(splits_path),
        "raw_manifest_sha256": _sha256_file(raw_manifest_path),
        "processed_manifest_sha256": _sha256_file(processed_manifest_path),
        "sample_count": len(source_ids),
        "source_id_sha256": _sha256_json(source_ids),
        "splits": {
            name: {
                "sample_count": len(splits[name]),
                "source_ids": [source_ids[index] for index in splits[name]],
                "source_id_sha256": _sha256_json([source_ids[index] for index in splits[name]]),
                "class_distribution": _class_distribution([labels[index] for index in splits[name]]),
            }
            for name in ("train", "val", "test")
        },
    }
    _write_json(os.path.join(dataset_dir, "split_manifest.json"), split_manifest)


def _directory_manifest(root: str) -> Dict[str, object]:
    files: List[Dict[str, object]] = []
    if os.path.isdir(root):
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in sorted(filenames):
                path = os.path.join(dirpath, filename)
                rel = os.path.relpath(path, root)
                files.append({"path": rel, "sha256": _sha256_file(path), "size": os.path.getsize(path)})
    return {
        "root": os.path.abspath(root),
        "file_count": len(files),
        "files": files,
        "sha256": _sha256_json(files),
    }


def validate_split_integrity(splits: Dict[str, List[int]], source_ids: List[str]) -> None:
    """Validates disjoint, exhaustive index and source-id split membership."""
    if len(source_ids) != len(set(str(value) for value in source_ids)):
        raise ValueError("Processed samples contain duplicate source IDs")
    expected = set(range(len(source_ids)))
    seen_indices: set[int] = set()
    seen_sources: set[str] = set()
    for split_name in ("train", "val", "test"):
        if split_name not in splits:
            raise ValueError(f"Missing split: {split_name}")
        split_indices = {int(index) for index in splits[split_name]}
        if not split_indices <= expected:
            raise ValueError(f"Split {split_name} contains an out-of-range index")
        if seen_indices.intersection(split_indices):
            raise ValueError(f"Split {split_name} overlaps another split")
        split_sources = {str(source_ids[index]) for index in split_indices}
        if seen_sources.intersection(split_sources):
            raise ValueError(f"Split {split_name} contains a duplicate source_id")
        seen_indices.update(split_indices)
        seen_sources.update(split_sources)
    if seen_indices != expected:
        raise ValueError("Splits do not cover every processed sample exactly once")


@dataclass
class DatasetConfig:
    """Holds dataset processing configuration loaded from YAML-equivalent JSON.

    Attributes:
        datasets: Names of datasets to process, e.g., ["twitter15", "twitter16"].
        raw_root: Absolute path to raw datasets directory.
        processed_root: Absolute path to save processed outputs.
        split_type: Strategy for split (currently only random).
        split_ratios: Fractions for train/val/test splits.
        split_seed: Random seed to make splits reproducible.
        label_allowed: Labels allowed to include.
        label_map: Mapping from string label to integer class id.
        label_skip_unknown: If true, skip samples with unknown labels.
        truncation_mode: Processing-time truncation policy.
        truncation_window: Optional processing-time observation window.
        truncation_unit: Semantic unit for the processing-time window.
        cutoff_minutes: Legacy processing cutoff retained for backward compatibility.
        enforce_root_component: Keep only the connected component containing the root.
        features_structural: List of structural feature names in order.
        augment_enabled: Whether to enable augmentation.
        augment_method: Augmentation method name.
        saving_format: Output saving format identifier.
        text_encoder_path: Optional path to a sentence transformer checkpoint.
        text_normalize: Whether to L2-normalize loaded text embeddings.
        text_noise_std: Std of Gaussian noise added to text embeddings when augmentation is enabled.
        text_dropout: Feature dropout probability applied to text embeddings when augmentation is enabled.
    """

    datasets: List[str]
    raw_root: str
    processed_root: str
    split_type: str
    split_ratios: Dict[str, float]
    split_seed: int
    label_allowed: List[str]
    label_map: Dict[str, int]
    label_skip_unknown: bool
    label_overrides: Dict[str, Dict[str, object]]
    truncation_mode: str
    truncation_window: Optional[int]
    truncation_unit: str
    cutoff_minutes: int
    enforce_root_component: bool
    features_structural: List[str]
    augment_enabled: bool
    augment_method: str
    text_encoder_path: Optional[str]
    text_normalize: bool
    text_cache_name: str
    text_batch_size: int
    text_normalization: Dict[str, object]
    text_noise_std: float
    text_dropout: float
    saving_format: str
    time_source_default: str
    time_source_overrides: Dict[str, str]


def load_yaml(config_path: str) -> Dict:
    """Loads full YAML using yaml.safe_load for robust parsing."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> DatasetConfig:
    cfg = load_yaml(config_path)["dataset"]
    # Validate presence
    required_keys = [
        "datasets",
        "raw_root",
        "processed_root",
        "split",
        "labels",
        "early",
        "features",
        "augment",
        "saving",
    ]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing config section: {k}")

    split = cfg["split"]
    labels = cfg["labels"]
    early = cfg["early"]
    features = cfg["features"]
    augment = cfg["augment"]
    saving = cfg["saving"]
    text_cfg = cfg.get("text", {})
    time_cfg = cfg.get("time_source", {})
    truncation_cfg = early.get("truncation", {})
    if len(truncation_cfg) == 0:
        truncation_mode = "window"
        truncation_window = int(early["cutoff_minutes"])
        truncation_unit = "minutes"
    else:
        truncation_mode = str(truncation_cfg.get("mode", "window")).lower()
        truncation_window_raw = truncation_cfg.get("window", None)
        truncation_window = None if truncation_window_raw in (None, "") else int(truncation_window_raw)
        truncation_unit = str(truncation_cfg.get("unit", "auto")).lower()

    def _path_value(value: object) -> str:
        resolved = os.path.expandvars(str(value))
        if "$" in resolved:
            raise ValueError(f"Unresolved environment variable in dataset config path: {value}")
        return resolved

    return DatasetConfig(
        datasets=list(cfg["datasets"]),
        raw_root=_path_value(cfg["raw_root"]),
        processed_root=_path_value(cfg["processed_root"]),
        split_type=str(split["type"]),
        split_ratios={
            "train": float(split["ratios"]["train"]),
            "val": float(split["ratios"]["val"]),
            "test": float(split["ratios"]["test"]),
        },
        split_seed=int(split["seed"]),
        label_allowed=list(labels["allowed"]),
        label_map={str(k): int(v) for k, v in labels["map"].items()},
        label_skip_unknown=bool(labels["skip_unknown"]),
        label_overrides={str(k): dict(v) for k, v in labels.get("overrides", {}).items()},
        truncation_mode=truncation_mode,
        truncation_window=truncation_window,
        truncation_unit=truncation_unit,
        cutoff_minutes=int(early["cutoff_minutes"]),
        enforce_root_component=bool(early["enforce_root_component"]),
        features_structural=list(features["structural"]),
        augment_enabled=bool(augment["enabled"]),
        augment_method=str(augment["method"]),
        text_encoder_path=_path_value(text_cfg["encoder_path"]) if text_cfg.get("encoder_path") else None,
        text_normalize=bool(text_cfg.get("normalize", False)),
        text_cache_name=str(text_cfg.get("cache_name", "text_embs.pt")),
        text_batch_size=int(text_cfg.get("batch_size", 64)),
        text_normalization=dict(text_cfg.get("normalization", {})),
        text_noise_std=float(augment.get("text_noise_std", 0.0)),
        text_dropout=float(augment.get("text_dropout", 0.0)),
        saving_format=str(saving["format"]),
        time_source_default=str(time_cfg.get("default", "minutes")),
        time_source_overrides={str(k): str(v) for k, v in time_cfg.get("overrides", {}).items()},
    )


@dataclass
class LabelConfig:
    allowed: List[str]
    mapping: Dict[str, int]
    skip_unknown: bool
    numeric: bool


def resolve_label_config(dataset: str, cfg: DatasetConfig) -> LabelConfig:
    """Returns dataset-specific label mapping configuration."""
    override = cfg.label_overrides.get(dataset, {})
    allowed = list(override.get("allowed", cfg.label_allowed))
    mapping = {str(k): int(v) for k, v in override.get("map", cfg.label_map).items()}
    skip_unknown = bool(override.get("skip_unknown", cfg.label_skip_unknown))
    numeric = bool(override.get("numeric", False))
    return LabelConfig(allowed=allowed, mapping=mapping, skip_unknown=skip_unknown, numeric=numeric)


def resolve_time_source(dataset: str, cfg: DatasetConfig) -> str:
    """Returns dataset-specific time source selection."""
    return cfg.time_source_overrides.get(dataset, cfg.time_source_default)


def normalize_tweet_text(text: str, normalization_cfg: Optional[Dict[str, object]] = None) -> str:
    """Normalizes tweet text using configurable, non-destructive substitutions."""
    cfg = normalization_cfg if isinstance(normalization_cfg, dict) else {}
    normalized = str(text)
    if bool(cfg.get("lowercase", False)):
        normalized = normalized.lower()
    if bool(cfg.get("replace_url", False)):
        normalized = re.sub(r"https?://\S+|www\.\S+", " <url> ", normalized)
    if bool(cfg.get("replace_mention", False)):
        normalized = re.sub(r"@\w+", " <user> ", normalized)
    if bool(cfg.get("replace_number", False)):
        normalized = re.sub(r"\b\d+(?:[\.,]\d+)?\b", " <num> ", normalized)
    if bool(cfg.get("keep_hashtag_text", False)):
        normalized = re.sub(r"#([A-Za-z0-9_]+)", r" \1 ", normalized)
    if bool(cfg.get("compress_repeated_chars", False)):
        normalized = re.sub(r"(.)\1{2,}", r"\1\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def resolve_text_model_path(model_name_or_path: str, raw_root: str) -> str:
    """Resolves a configured model path, preferring explicit local paths when present."""
    model_text = str(model_name_or_path).strip()
    if len(model_text) == 0:
        return os.path.join(os.path.dirname(raw_root), "sentence_transformers_all_MiniLM_L6v2")
    if os.path.isdir(model_text):
        return model_text
    local_candidate = os.path.join(os.path.dirname(raw_root), model_text)
    if os.path.isdir(local_candidate):
        return local_candidate
    return model_text


def _text_tokenizer_hash(model_path: str) -> str:
    candidates = ("tokenizer_config.json", "tokenizer.json", "vocab.txt", "spiece.model", "merges.txt")
    payload = []
    for name in candidates:
        path = os.path.join(model_path, name)
        if os.path.isfile(path):
            payload.append({"name": name, "sha256": _sha256_file(path)})
    return _sha256_json(payload)


def _text_cache_metadata(
    model_path: str,
    encoder_name: str,
    ids: List[str],
    normalized_texts: List[str],
    normalization_cfg: Optional[Dict[str, object]],
    dimension: int,
    dtype: str,
) -> Dict[str, object]:
    return {
        "encoder_name": str(encoder_name),
        "weights_sha256": _directory_manifest(model_path).get("sha256") if os.path.isdir(model_path) else "",
        "tokenizer_sha256": _text_tokenizer_hash(model_path) if os.path.isdir(model_path) else "",
        "normalization_config_sha256": _sha256_json(normalization_cfg or {}),
        "source_id_sha256": _sha256_json(ids),
        "normalized_text_sha256": _sha256_json(normalized_texts),
        "embedding_dimension": int(dimension),
        "dtype": str(dtype),
    }


def parse_source_texts(raw_dir: str) -> Dict[str, str]:
    """Parses `source_tweets.txt` into a mapping from source_id to raw text.

    The file format is tab-separated: `<source_id>\t<text>` per line.
    """
    path = os.path.join(raw_dir, "source_tweets.txt")
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            assert len(parts) == 2, f"Bad line without TAB: {line}"
            sid = parts[0].strip()
            text = parts[1].strip()
            assert len(sid) > 0, "Empty source id"
            mapping[sid] = text
    return mapping


def collect_json_source_texts(raw_dir: str) -> Dict[str, str]:
    """Collects source texts from JSON-format datasets (e.g., weibo, pheme).

    JSON raw files are keyed by the file stem, not by the mutable `source`
    payload field. Some releases repeat the payload source identifier across
    distinct cascades; file stems remain the stable per-cascade identity.
    """
    mapping: Dict[str, str] = {}
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(raw_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        source = obj.get("source", {})
        if not isinstance(source, dict):
            continue
        sid = os.path.splitext(fname)[0]
        text = str(source.get("content", "")).strip()
        if len(text) == 0:
            continue
        if sid in mapping:
            raise ValueError(f"Duplicate JSON cascade file stem while collecting texts: {sid}")
        mapping[sid] = text
    assert len(mapping) > 0, f"No JSON sources found in {raw_dir}"
    return mapping


def try_reuse_text_embedding_cache(
    out_path: str,
    processed_root: str,
    dataset: str,
    cache_name: str,
    expected_ids: List[str],
    expected_metadata: Optional[Dict[str, object]] = None,
) -> bool:
    """Copies an identical existing text cache from a sibling processed root when available."""
    root_dir = os.path.abspath(os.path.dirname(processed_root))
    candidate_roots = [
        os.path.join(root_dir, "processed"),
        os.path.join(root_dir, "processed_paper_aligned"),
    ]
    normalized_out = os.path.abspath(out_path)
    for candidate_root in candidate_roots:
        candidate_path = os.path.join(candidate_root, dataset, str(cache_name))
        if os.path.abspath(candidate_path) == normalized_out or not os.path.isfile(candidate_path):
            continue
        obj = torch.load(candidate_path, weights_only=False)
        candidate_ids = list(obj.get("ids", []))
        candidate_metadata = obj.get("manifest", {})
        if candidate_ids != expected_ids or not isinstance(candidate_metadata, dict):
            continue
        if expected_metadata is not None:
            # Compare every declared value. Empty values are intentionally not
            # treated as wildcards except for dimension/dtype, which can only
            # be known after loading the candidate tensor.
            mismatch = False
            for key, expected_value in expected_metadata.items():
                if key in {"embedding_dimension", "dtype"} and expected_value in {None, "", 0}:
                    continue
                if candidate_metadata.get(key) != expected_value:
                    mismatch = True
                    break
            if mismatch:
                continue
        if not isinstance(obj.get("emb"), Tensor) or obj["emb"].dim() != 2:
            continue
        if int(candidate_metadata.get("embedding_dimension", -1)) != int(obj["emb"].size(-1)):
            continue
        if str(candidate_metadata.get("dtype", "")) != str(obj["emb"].dtype).replace("torch.", ""):
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(obj, out_path)
        return True
    return False


def build_text_embedding_cache(dataset: str,
                               raw_root: str,
                               processed_root: str,
                               model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                               batch_size: int = 64,
                               normalization_cfg: Optional[Dict[str, object]] = None,
                               cache_name: str = "text_embs.pt") -> str:
    """Builds and saves `{source_id -> text_emb}` cache to processed dir.

    The output file path is `{processed_root}/{dataset}/text_embs.pt` and contains:
      {"model": str, "dim": int, "ids": List[str], "emb": FloatTensor[N, D]}
    """
    raw_dir = os.path.join(raw_root, dataset)
    out_dir = os.path.join(processed_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, str(cache_name))

    # Parse all available source texts
    sid_to_text = parse_source_texts(raw_dir)
    ids = sorted(sid_to_text.keys())
    texts = [normalize_tweet_text(sid_to_text[sid], normalization_cfg) for sid in ids]
    model_path = resolve_text_model_path(model_name, raw_root)
    expected_metadata = _text_cache_metadata(
        model_path=model_path,
        encoder_name=str(model_name),
        ids=ids,
        normalized_texts=texts,
        normalization_cfg=normalization_cfg,
        dimension=0,
        dtype="",
    )
    if try_reuse_text_embedding_cache(
        out_path,
        processed_root,
        dataset,
        cache_name,
        ids,
        expected_metadata=expected_metadata,
    ):
        return out_path

    emb = _encode_texts_transformer_mean_pool(texts, model_path=model_path, batch_size=batch_size)

    metadata = _text_cache_metadata(
        model_path=model_path,
        encoder_name=str(model_name),
        ids=ids,
        normalized_texts=texts,
        normalization_cfg=normalization_cfg,
        dimension=int(emb.size(-1)),
        dtype=str(emb.dtype).replace("torch.", ""),
    )

    obj = {
        "model": str(model_name),
        "dim": int(emb.size(-1)),
        "ids": ids,
        "emb": emb,
        "manifest": metadata,
    }
    torch.save(obj, out_path)
    with open(f"{out_path}.manifest.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return out_path


def _encode_texts_sentence_transformer(texts: List[str],
                                       model_path: str,
                                       batch_size: int = 64) -> Tensor:
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model_path, device=device)
    emb = encoder.encode(
        texts,
        batch_size=int(batch_size),
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    return emb.to(torch.float32).cpu()


def _encode_texts_transformer_mean_pool(texts: List[str],
                                        model_path: str,
                                        batch_size: int = 64) -> Tensor:
    """Encodes texts with a PyTorch HuggingFace encoder and mean pooling."""
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content in ("", "{}"):
            os.remove(adapter_cfg_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    all_emb: List[Tensor] = []
    for start in range(0, len(texts), int(batch_size)):
        batch_texts = texts[start:start + int(batch_size)]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1).type_as(token_embeddings)
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-6)
            pooled = summed / counts
        all_emb.append(pooled.cpu())
    return torch.cat(all_emb, dim=0).to(torch.float32)


def build_text_embedding_cache_from_mapping(dataset: str,
                                            processed_root: str,
                                            mapping: Dict[str, str],
                                            model_path: str,
                                            batch_size: int = 64,
                                            normalization_cfg: Optional[Dict[str, object]] = None,
                                            cache_name: str = "text_embs.pt") -> str:
    """Builds text embedding cache from provided id->text mapping."""
    out_dir = os.path.join(processed_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, str(cache_name))
    ids = sorted(mapping.keys())
    texts = [normalize_tweet_text(mapping[i], normalization_cfg) for i in ids]
    model_path = resolve_text_model_path(model_path, os.path.dirname(processed_root))
    expected_metadata = _text_cache_metadata(
        model_path=model_path,
        encoder_name=str(model_path),
        ids=ids,
        normalized_texts=texts,
        normalization_cfg=normalization_cfg,
        dimension=0,
        dtype="",
    )
    if try_reuse_text_embedding_cache(
        out_path,
        processed_root,
        dataset,
        cache_name,
        ids,
        expected_metadata=expected_metadata,
    ):
        return out_path
    emb = _encode_texts_transformer_mean_pool(texts, model_path=model_path, batch_size=batch_size)
    metadata = _text_cache_metadata(
        model_path=model_path,
        encoder_name=str(model_path),
        ids=ids,
        normalized_texts=texts,
        normalization_cfg=normalization_cfg,
        dimension=int(emb.size(-1)),
        dtype=str(emb.dtype).replace("torch.", ""),
    )
    obj = {
        "model": str(model_path),
        "dim": int(emb.size(-1)),
        "ids": ids,
        "emb": emb,
        "manifest": metadata,
    }
    torch.save(obj, out_path)
    with open(f"{out_path}.manifest.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return out_path


def _load_text_embedding_cache(
    processed_root: str,
    dataset: str,
    cache_name: str = "text_embs.pt",
    expected_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, int], Tensor]:
    """Loads text embedding cache and returns (id_to_index, emb_matrix)."""
    path = os.path.join(processed_root, dataset, str(cache_name))
    obj = torch.load(path, weights_only=False)
    ids: List[str] = list(obj["ids"])  # type: ignore[index]
    emb: Tensor = obj["emb"].to(torch.float32)  # type: ignore[index]
    manifest = obj.get("manifest") if isinstance(obj, dict) else None
    required = {
        "encoder_name", "weights_sha256", "tokenizer_sha256", "normalization_config_sha256",
        "source_id_sha256", "normalized_text_sha256", "embedding_dimension", "dtype",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError(f"Text embedding cache lacks strict manifest metadata: {path}")
    if int(manifest["embedding_dimension"]) != int(emb.size(-1)):
        raise ValueError(f"Text embedding cache dimension mismatch: {path}")
    if str(manifest["source_id_sha256"]) != _sha256_json(ids):
        raise ValueError(f"Text embedding cache source ID hash mismatch: {path}")
    if expected_metadata is not None:
        for key, expected_value in expected_metadata.items():
            if key in {"embedding_dimension", "dtype"} and expected_value in {None, "", 0}:
                continue
            if manifest.get(key) != expected_value:
                raise ValueError(f"Text embedding cache manifest mismatch for {key}: {path}")
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    return id_to_idx, emb

def parse_node(token: str) -> Tuple[str, str, int]:
    """Parses a node token like ["uid","tweet","delay"].

    Returns:
        (uid, tweet_id, delay_minutes)
    """
    s = token.strip()
    assert s.startswith("[") and s.endswith("]"), f"Bad node token: {token}"
    inner = s[1:-1].strip()
    parts = [p.strip() for p in inner.split(",")]
    assert len(parts) == 3, f"Node parts != 3: {token}"

    def strip_quotes(x: str) -> str:
        x = x.strip()
        if x.startswith("'") and x.endswith("'"):
            return x[1:-1]
        if x.startswith('"') and x.endswith('"'):
            return x[1:-1]
        return x

    uid = strip_quotes(parts[0])
    tweet_id = strip_quotes(parts[1])
    delay_raw = strip_quotes(parts[2])
    # allow optional leading '-' and decimal digits
    assert re.match(r"^-?\d+(?:\.\d+)?$",
                    delay_raw) is not None, f"Delay not numeric: {delay_raw}"
    delay_minutes = int(round(float(delay_raw)))
    return uid, tweet_id, delay_minutes


def parse_tree_file(path: str) -> Tuple[List[Tuple[str, str, int]], List[Tuple[int, int]]]:
    """Parses one tree file into node list and edge index pairs.

    Returns:
        nodes: list of (uid, tweet_id, delay_minutes) in order of first appearance
        edges: list of (src_idx, dst_idx)
    """
    nodes: List[Tuple[str, str, int]] = []
    node_to_idx: Dict[Tuple[str, str, int], int] = {}
    edges: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "->" not in line:
                raise ValueError(f"Edge line missing '->': {line}")
            lhs, rhs = line.split("->", 1)
            u = parse_node(lhs.strip())
            v = parse_node(rhs.strip())

            if u not in node_to_idx:
                node_to_idx[u] = len(nodes)
                nodes.append(u)
            if v not in node_to_idx:
                node_to_idx[v] = len(nodes)
                nodes.append(v)
            edges.append((node_to_idx[u], node_to_idx[v]))

    return nodes, edges


def select_observation_view(nodes: List[Tuple[str, str, int]],
                            edges: List[Tuple[int, int]],
                            truncation_mode: str,
                            truncation_window: Optional[int],
                            enforce_root_component: bool,
                            truncation_unit: str = "auto",
                            source_time_unit: str = "minutes") -> Tuple[List[int], List[Tuple[int, int]], int, List[int]]:
    """Applies an explicit processing-time observation policy.

    Returns:
        keep_nodes: indices of nodes kept after filtering
        keep_edges: edges reindexed within kept nodes
        root_index_new: index of root node in the reindexed space
    """
    # shift times so that the earliest time becomes 0
    original_times = torch.tensor([t for (_, _, t) in nodes], dtype=torch.long)
    min_time = int(original_times.min().item())
    shifted_times = original_times - min_time
    mode = str(truncation_mode).lower()
    unit = str(truncation_unit).lower()
    native_unit = str(source_time_unit).lower()
    if unit not in {"auto", "minutes", "hop"}:
        raise ValueError(f"Unsupported truncation unit: {truncation_unit}")
    if unit != "auto" and unit != native_unit:
        raise ValueError(
            f"Truncation unit {unit} does not match the sample time unit {native_unit}; "
            "minutes and hop horizons must use separate fields."
        )
    if mode == "none":
        keep_mask = torch.ones_like(shifted_times, dtype=torch.bool)
    elif mode in {"window", "minutes", "hop"}:
        if truncation_window is None or int(truncation_window) <= 0:
            raise ValueError(
                f"Processing truncation mode '{mode}' requires a positive window, got {truncation_window}."
            )
        keep_mask = shifted_times <= int(truncation_window)
    else:
        raise ValueError(f"Unsupported processing truncation mode: {truncation_mode}")
    keep_nodes = [i for i, k in enumerate(keep_mask.tolist()) if k]
    if not keep_nodes:
        raise ValueError("No nodes kept after applying the observation policy")

    # Identify root as node with minimal time (prefer time==0)
    candidate_indices = [i for i in keep_nodes if shifted_times[i].item() == 0]
    if candidate_indices:
        root_idx = min(candidate_indices)
    else:
        # fallback to earliest time among kept nodes
        root_idx = min((i for i in keep_nodes),
                       key=lambda x: shifted_times[x].item())

    # Filter edges by kept nodes
    kept_set = set(keep_nodes)
    filtered_edges = [(u, v)
                      for (u, v) in edges if u in kept_set and v in kept_set]

    if enforce_root_component:
        # BFS/DFS from root within filtered edges
        adj: Dict[int, List[int]] = {}
        for u, v in filtered_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        stack = [root_idx]
        visited = set([root_idx])
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        keep_nodes = sorted(list(visited))
        kept_set = set(keep_nodes)
        filtered_edges = [(u, v) for (
            u, v) in filtered_edges if u in kept_set and v in kept_set]

    # Reindex nodes to 0..N-1
    old_to_new = {old: new for new, old in enumerate(keep_nodes)}
    reindexed_edges = [(old_to_new[u], old_to_new[v])
                       for (u, v) in filtered_edges]
    root_index_new = old_to_new[root_idx]
    # provide shifted times for feature computation
    shifted_times_list = [int(shifted_times[i].item()) for i in keep_nodes]
    return keep_nodes, reindexed_edges, root_index_new, shifted_times_list


def resolve_feature_time_scale(shifted_times_list: List[int], explicit_window: Optional[int] = None) -> int:
    """Resolves the normalization scale used by time-derived structural features."""
    if explicit_window is not None and int(explicit_window) > 0:
        return int(explicit_window)
    if len(shifted_times_list) == 0:
        return 1
    return max(1, max(int(value) for value in shifted_times_list))


def compute_structural_features(nodes: List[Tuple[str, str, int]],
                                edges: List[Tuple[int, int]],
                                keep_nodes: List[int],
                                reindexed_edges: List[Tuple[int, int]],
                                root_index_new: int,
                                feature_time_scale: Optional[int],
                                shifted_times_list: List[int]) -> Tensor:
    """Computes structural node features.

    Returns features in the order:
    [norm_time, depth, in_degree, out_degree, parent_delta_norm, time_speed_norm, 
     time_gradient_norm, root_time_delta_norm].
    """
    num_nodes = len(keep_nodes)
    times = torch.tensor(shifted_times_list, dtype=torch.float32)
    time_scale = float(resolve_feature_time_scale(shifted_times_list, feature_time_scale))
    norm_time = times / time_scale
    norm_time = torch.clamp(norm_time, 0.0, 1.0)

    # Build undirected adjacency for BFS depth
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in reindexed_edges:
        adj[u].append(v)
        adj[v].append(u)

    depth = torch.full((num_nodes,), float("inf"))
    depth[root_index_new] = 0.0
    queue = [root_index_new]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        for nb in adj.get(cur, []):
            if math.isinf(depth[nb].item()):
                depth[nb] = depth[cur] + 1.0
                queue.append(nb)
    # disconnected should not happen if filtered
    depth[torch.isinf(depth)] = float(num_nodes)

    in_degree = torch.zeros(num_nodes, dtype=torch.float32)
    out_degree = torch.zeros(num_nodes, dtype=torch.float32)
    for u, v in reindexed_edges:
        out_degree[u] += 1.0
        in_degree[v] += 1.0

    # Parent-child time delta, normalized by the feature time scale.
    # For nodes without an incoming edge, the delta is 0.
    parents: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for u, v in reindexed_edges:
        parents[v].append(u)

    parent_delta = torch.zeros(num_nodes, dtype=torch.float32)
    for v in range(num_nodes):
        if parents[v]:
            p_idx = torch.tensor(parents[v], dtype=torch.long)
            p_times = times[p_idx]
            delta = times[v] - torch.min(p_times)
            parent_delta[v] = torch.clamp(delta, min=0.0)
        else:
            parent_delta[v] = 0.0
    parent_delta_norm = torch.clamp(parent_delta / time_scale, 0.0, 1.0)

    # Time speed: arrival time relative to depth (captures propagation velocity)
    # Nodes with high time but low depth have fast propagation
    time_speed = torch.where(
        depth > 0,
        times / (depth + 1.0),
        times
    )
    time_speed_norm = torch.clamp(time_speed / time_scale, 0.0, 1.0)

    # Time gradient: average time difference with neighbors
    time_gradient = torch.zeros(num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        neighbors = adj.get(i, [])
        if neighbors:
            neighbor_times = times[torch.tensor(neighbors, dtype=torch.long)]
            time_gradient[i] = torch.abs(times[i] - torch.mean(neighbor_times))
        else:
            time_gradient[i] = 0.0
    time_gradient_norm = torch.clamp(time_gradient / time_scale, 0.0, 1.0)

    # Root time delta: direct time difference with root node
    root_time = times[root_index_new]
    root_time_delta = torch.abs(times - root_time)
    root_time_delta_norm = torch.clamp(root_time_delta / time_scale, 0.0, 1.0)

    x = torch.stack([
        norm_time, depth, in_degree, out_degree, parent_delta_norm,
        time_speed_norm, time_gradient_norm, root_time_delta_norm
    ], dim=1)
    return x


def parse_json_tree(obj: Dict[str, Any], time_source: str) -> Tuple[List[Tuple[str, str, int]], List[Tuple[int, int]]]:
    """Parses JSON-format cascades with hop-based timing."""
    assert "source" in obj, "source field missing in JSON object"
    source = obj["source"]
    comments = obj.get("comment", [])

    source_uid = str(source.get("user id", source.get("user_id", "")))
    source_tid = (
        source.get("tweet id")
        or source.get("tweet_id")
        or source.get("source id")
        or source.get("source_id")
        or source.get("id")
    )
    assert source_tid is not None, "Source tweet id missing"
    nodes: List[Tuple[str, str, int]] = [(source_uid, str(source_tid), 0)]

    comment_id_to_idx: Dict[int, int] = {}
    edges: List[Tuple[int, int]] = []

    for comment in comments:
        cid_raw = comment.get("comment id", comment.get("comment_id"))
        assert cid_raw is not None, "comment id missing"
        cid = int(cid_raw)
        delay = int(comment.get("hop", 1)) if time_source == "hop" else int(comment.get("time", 1))
        user = str(comment.get("user id", comment.get("user_id", "")))
        idx = len(nodes)
        nodes.append((user, str(cid), delay))
        comment_id_to_idx[cid] = idx

    for comment in comments:
        cid_raw = comment.get("comment id", comment.get("comment_id"))
        cid = int(cid_raw)
        parent_raw = comment.get("parent", -1)
        parent_val = int(parent_raw) if parent_raw is not None else -1
        parent_idx = 0 if parent_val == -1 else comment_id_to_idx[parent_val]
        child_idx = comment_id_to_idx[cid]
        edges.append((parent_idx, child_idx))

    return nodes, edges


def build_pyg_data(nodes: List[Tuple[str, str, int]],
                   edges: List[Tuple[int, int]],
                   keep_nodes: List[int],
                   reindexed_edges: List[Tuple[int, int]],
                   root_index_new: int,
                   feature_time_scale: Optional[int],
                   shifted_times_list: List[int],
                   label_id: int,
                   source_id: str,
                   time_unit: str) -> Data:
    num_nodes = len(keep_nodes)
    if num_nodes == 0:
        raise ValueError("No nodes after filtering")

    x = compute_structural_features(
        nodes,
        edges,
        keep_nodes,
        reindexed_edges,
        root_index_new,
        feature_time_scale,
        shifted_times_list,
    )
    edge_index = torch.tensor(
        reindexed_edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([label_id], dtype=torch.long)

    node_uid = [nodes[i][0] for i in keep_nodes]
    node_tid = [nodes[i][1] for i in keep_nodes]
    node_time = torch.tensor([nodes[i][2] for i in keep_nodes], dtype=torch.long)
    node_time_shifted = torch.tensor(shifted_times_list, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        root_index=torch.tensor([root_index_new], dtype=torch.long),
    )
    # Attach metadata as plain attributes (PyG tolerant)
    data.source_id = source_id
    data.node_uid = node_uid
    data.node_tweet_id = node_tid
    unit = str(time_unit).lower()
    if unit == "minutes":
        data.node_time_minutes = node_time
        data.node_time_minutes_shifted = node_time_shifted
        data.node_time_hops = torch.zeros_like(node_time)
        data.node_time_hops_shifted = torch.zeros_like(node_time_shifted)
    elif unit == "hop":
        # Hop depth is kept in its own fields. Minutes tensors are zero-filled
        # rather than silently reusing hop values under a minutes name.
        data.node_time_minutes = torch.zeros_like(node_time)
        data.node_time_minutes_shifted = torch.zeros_like(node_time_shifted)
        data.node_time_hops = node_time
        data.node_time_hops_shifted = node_time_shifted
    else:
        raise ValueError(f"Unsupported time unit for processed sample: {time_unit}")
    data.time_unit = str(time_unit)
    return data


def read_labels(label_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Bad label line: {line}")
            label, sid = line.split(":", 1)
            mapping[sid.strip()] = label.strip()
    return mapping


def generate_stratified_splits(labels: List[int], ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    assert len(labels) > 0, "Labels must be non-empty for stratified split"
    cls_to_indices: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        y_int = int(y)
        cls_to_indices.setdefault(y_int, []).append(idx)

    g = torch.Generator()
    g.manual_seed(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls, idx_list in cls_to_indices.items():
        order = torch.randperm(len(idx_list), generator=g).tolist()
        shuffled = [idx_list[i] for i in order]
        n = len(shuffled)
        n_train = int(math.floor(n * ratios["train"]))
        n_val = int(math.floor(n * ratios["val"]))
        # Ensure minimal coverage when samples exist
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_val = min(n_val, max(0, n - n_train))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            if n_val > 0 and n_train + n_val > n:
                n_val = max(0, n - n_train)
        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:])

    def _shuffle(arr: List[int]) -> List[int]:
        if len(arr) == 0:
            return arr
        order = torch.randperm(len(arr), generator=g).tolist()
        return [arr[i] for i in order]

    return {
        "train": _shuffle(train_idx),
        "val": _shuffle(val_idx),
        "test": _shuffle(test_idx),
    }


def generate_splits(labels: List[int], ratios: Dict[str, float], seed: int, strategy: str) -> Dict[str, List[int]]:
    strategy_norm = str(strategy).lower()
    if strategy_norm == "stratified":
        return generate_stratified_splits(labels, ratios, seed)
    return generate_random_splits(len(labels), ratios, seed)


def generate_random_splits(num_samples: int, ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    assert abs(sum(ratios.values()) -
               1.0) < 1e-6, "Split ratios must sum to 1.0"
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(num_samples, generator=g).tolist()
    n_train = int(num_samples * ratios["train"])
    n_val = int(num_samples * ratios["val"])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def collate_and_save(dataset_name: str,
                     data_list: List[Data],
                     labels: List[int],
                     out_root: str,
                     split_ratios: Dict[str, float],
                     split_seed: int,
                     split_strategy: str,
                     raw_dir: Optional[str] = None,
                     label_cfg: Optional[LabelConfig] = None,
                     time_unit: str = "minutes",
                     generation_command: str = "processor.py") -> None:
    out_dir = os.path.join(out_root, dataset_name, "pyg")
    os.makedirs(out_dir, exist_ok=True)
    # Use collate without touching internal storage attributes to avoid warnings

    class _Collater(InMemoryDataset):
        def __init__(self, root: str):
            super().__init__(root)
    collater = _Collater(out_dir)
    data_obj, slices = collater.collate(data_list)
    torch.save(data_obj, os.path.join(out_dir, "data.pt"))
    torch.save(slices, os.path.join(out_dir, "slices.pt"))

    splits = generate_splits(labels, split_ratios, split_seed, split_strategy)
    with open(os.path.join(out_root, dataset_name, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    if raw_dir is None:
        raw_dir = os.path.join(out_root, dataset_name)
    if label_cfg is None:
        inferred = sorted(set(int(value) for value in labels))
        if inferred != list(range(len(inferred))):
            raise ValueError("A declared label schema is required when labels are not contiguous")
        label_cfg = LabelConfig(
            allowed=[str(value) for value in inferred],
            mapping={str(value): int(value) for value in inferred},
            skip_unknown=False,
            numeric=True,
        )
    _write_dataset_manifests(
        dataset_name=dataset_name,
        out_root=out_root,
        raw_dir=raw_dir,
        data_list=data_list,
        labels=labels,
        splits=splits,
        split_seed=split_seed,
        split_strategy=split_strategy,
        label_cfg=label_cfg,
        time_unit=time_unit,
        generation_command=generation_command,
    )


def load_labels_from_processed(data_path: str, slices_path: str) -> List[int]:
    data_obj = torch.load(data_path, weights_only=False)
    slices = torch.load(slices_path, weights_only=False)
    y_slices = slices["y"]
    labels: List[int] = []
    for i in range(len(y_slices) - 1):
        start = int(y_slices[i])
        end = int(y_slices[i + 1])
        y_tensor = data_obj.y[start:end]
        labels.append(int(y_tensor.view(-1)[0].item()))
    return labels


def write_manifests_from_processed(
    ds: str,
    cfg: DatasetConfig,
    *,
    generation_command: str,
    splits_override: Optional[Dict[str, List[int]]] = None,
) -> None:
    """Write canonical manifests from existing processed data and splits.

    This path does not fabricate samples or reorder an existing split. It only
    binds already materialized PyG tensors, the current splits.json, declared
    label schema, and raw input tree into hash-bearing manifests.
    """
    data_path = os.path.join(cfg.processed_root, ds, "pyg", "data.pt")
    slices_path = os.path.join(cfg.processed_root, ds, "pyg", "slices.pt")
    splits_path = os.path.join(cfg.processed_root, ds, "splits.json")
    assert os.path.isfile(data_path), f"Missing {data_path}"
    assert os.path.isfile(slices_path), f"Missing {slices_path}"
    if splits_override is None:
        assert os.path.isfile(splits_path), f"Missing {splits_path}"
        with open(splits_path, "r", encoding="utf-8") as handle:
            splits = json.load(handle)
    else:
        splits = splits_override
    labels = load_labels_from_processed(data_path, slices_path)
    data_obj = torch.load(data_path, map_location="cpu", weights_only=False)
    source_ids = [str(value) for value in getattr(data_obj, "source_id", [])]
    if len(source_ids) != len(labels):
        raise ValueError(f"Processed source_id/label count mismatch for {ds}: {len(source_ids)} vs {len(labels)}")
    data_list = [Data(source_id=source_id) for source_id in source_ids]
    _write_dataset_manifests(
        dataset_name=ds,
        out_root=cfg.processed_root,
        raw_dir=os.path.join(cfg.raw_root, ds),
        data_list=data_list,
        labels=labels,
        splits={str(name): [int(index) for index in values] for name, values in splits.items()},
        split_seed=cfg.split_seed,
        split_strategy=cfg.split_type,
        label_cfg=resolve_label_config(ds, cfg),
        time_unit=resolve_time_source(ds, cfg),
        generation_command=generation_command,
    )


def process_dataset(ds: str, cfg: DatasetConfig) -> None:
    label_cfg = resolve_label_config(ds, cfg)
    time_source = resolve_time_source(ds, cfg)
    raw_dir = os.path.join(cfg.raw_root, ds)
    text_model_path = resolve_text_model_path(
        cfg.text_encoder_path or "sentence_transformers_all_MiniLM_L6v2",
        cfg.raw_root,
    )

    # JSON-format datasets (weibo, pheme)
    if ds in {"weibo", "pheme"}:
        assert os.path.isdir(raw_dir), f"Missing raw dir: {raw_dir}"
        source_texts = collect_json_source_texts(raw_dir)
        cache_path = os.path.join(cfg.processed_root, ds, cfg.text_cache_name)
        if not os.path.isfile(cache_path):
            build_text_embedding_cache_from_mapping(
                dataset=ds,
                processed_root=cfg.processed_root,
                mapping=source_texts,
                model_path=text_model_path,
                batch_size=cfg.text_batch_size,
                normalization_cfg=cfg.text_normalization,
                cache_name=cfg.text_cache_name,
            )
        expected_cache_metadata = _text_cache_metadata(
            model_path=text_model_path,
            encoder_name=str(text_model_path),
            ids=sorted(source_texts.keys()),
            normalized_texts=[normalize_tweet_text(source_texts[sid], cfg.text_normalization) for sid in sorted(source_texts.keys())],
            normalization_cfg=cfg.text_normalization,
            dimension=0,
            dtype="",
        )
        id_to_idx, emb_matrix = _load_text_embedding_cache(
            cfg.processed_root,
            ds,
            cfg.text_cache_name,
            expected_metadata=expected_cache_metadata,
        )

        data_list: List[Data] = []
        labels: List[int] = []
        for fname in sorted(os.listdir(raw_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(raw_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            source = obj.get("source", {})
            source_id = os.path.splitext(fname)[0]
            raw_source_id = (
                source.get("tweet id")
                or source.get("tweet_id")
                or source.get("source id")
                or source.get("source_id")
                or source.get("id")
            )
            if source_id not in source_texts:
                continue

            label_raw = source.get("label")
            assert label_raw is not None, f"Missing label for source {source_id}"
            if label_cfg.numeric:
                label_value = int(label_raw)
                label_name = "rumor" if label_value == 1 else "non-rumor"
            else:
                label_name = str(label_raw).strip()
            if label_name not in label_cfg.allowed:
                if label_cfg.skip_unknown:
                    continue
                raise ValueError(f"Disallowed label {label_raw} for source {source_id}")
            label_id = label_cfg.mapping[label_name]

            nodes, edges = parse_json_tree(obj, time_source=time_source)
            keep_nodes, re_e, root_new, shifted_times = select_observation_view(
                nodes,
                edges,
                cfg.truncation_mode,
                cfg.truncation_window,
                cfg.enforce_root_component,
                truncation_unit=cfg.truncation_unit,
                source_time_unit=time_source,
            )

            data = build_pyg_data(
                nodes,
                edges,
                keep_nodes,
                re_e,
                root_new,
                resolve_feature_time_scale(shifted_times, cfg.truncation_window),
                shifted_times,
                label_id,
                str(source_id),
                time_source,
            )
            data.raw_source_id = "" if raw_source_id is None else str(raw_source_id)
            assert source_id in id_to_idx, f"Text embedding missing for source {source_id}"
            row = int(id_to_idx[source_id])
            vec = emb_matrix[row].to(torch.float32).unsqueeze(0)
            if cfg.text_normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
            if cfg.augment_enabled and cfg.augment_method in ["text_noise", "text_aug", "text_dropout"]:
                if cfg.text_noise_std > 0:
                    vec = vec + torch.randn_like(vec) * float(cfg.text_noise_std)
                if cfg.text_dropout > 0:
                    vec = torch.nn.functional.dropout(vec, p=min(0.9, float(cfg.text_dropout)), training=True)
            data.text_emb = vec
            data_list.append(data)
            labels.append(int(label_id))

        assert len(data_list) > 0, f"No samples produced for {ds}"
        collate_and_save(
            dataset_name=ds,
            data_list=data_list,
            labels=labels,
            out_root=cfg.processed_root,
            split_ratios=cfg.split_ratios,
            split_seed=cfg.split_seed,
            split_strategy=cfg.split_type,
            raw_dir=raw_dir,
            label_cfg=label_cfg,
            time_unit=time_source,
            generation_command=f"processor.py --config <resolved> --dataset {ds}",
        )
        return

    # Twitter-style datasets (tree + label.txt)
    tree_dir = os.path.join(raw_dir, "tree")
    label_path = os.path.join(raw_dir, "label.txt")
    if not os.path.isdir(tree_dir):
        raise FileNotFoundError(f"Missing tree dir: {tree_dir}")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Missing label file: {label_path}")

    sid_to_text = parse_source_texts(raw_dir)
    cache_path = os.path.join(cfg.processed_root, ds, cfg.text_cache_name)
    if not os.path.isfile(cache_path):
        build_text_embedding_cache(
            ds,
            cfg.raw_root,
            cfg.processed_root,
            model_name=text_model_path,
            batch_size=cfg.text_batch_size,
            normalization_cfg=cfg.text_normalization,
            cache_name=cfg.text_cache_name,
        )
    expected_cache_metadata = _text_cache_metadata(
        model_path=text_model_path,
        encoder_name=str(text_model_path),
        ids=sorted(sid_to_text.keys()),
        normalized_texts=[normalize_tweet_text(sid_to_text[sid], cfg.text_normalization) for sid in sorted(sid_to_text.keys())],
        normalization_cfg=cfg.text_normalization,
        dimension=0,
        dtype="",
    )
    id_to_idx, emb_matrix = _load_text_embedding_cache(
        cfg.processed_root,
        ds,
        cfg.text_cache_name,
        expected_metadata=expected_cache_metadata,
    )
    label_map_raw = read_labels(label_path)
    data_list: List[Data] = []
    labels: List[int] = []
    for fname in sorted(os.listdir(tree_dir)):
        fpath = os.path.join(tree_dir, fname)
        if not os.path.isfile(fpath):
            continue
        source_id = os.path.splitext(os.path.basename(fpath))[0]
        label_str = label_map_raw.get(source_id)
        if label_str is None:
            if label_cfg.skip_unknown:
                continue
            raise ValueError(f"Missing label for source {source_id}")
        norm_label = label_str.strip()

        if norm_label not in label_cfg.allowed:
            if label_cfg.skip_unknown:
                continue
            raise ValueError(f"Disallowed label {label_str}")
        label_id = label_cfg.mapping[norm_label]

        assert source_id in sid_to_text, f"Missing text for source {source_id}"

        nodes, edges = parse_tree_file(fpath)
        keep_nodes, re_e, root_new, shifted_times = select_observation_view(
            nodes,
            edges,
            cfg.truncation_mode,
            cfg.truncation_window,
            cfg.enforce_root_component,
            truncation_unit=cfg.truncation_unit,
            source_time_unit=time_source,
        )

        data = build_pyg_data(
            nodes,
            edges,
            keep_nodes,
            re_e,
            root_new,
            resolve_feature_time_scale(shifted_times, cfg.truncation_window),
            shifted_times,
            label_id,
            source_id,
            time_source,
        )
        assert source_id in id_to_idx, f"Text embedding missing for source {source_id}"
        row = int(id_to_idx[source_id])
        vec = emb_matrix[row].to(torch.float32).unsqueeze(0)
        if cfg.text_normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
        if cfg.augment_enabled and cfg.augment_method in ["text_noise", "text_aug", "text_dropout"]:
            if cfg.text_noise_std > 0:
                vec = vec + torch.randn_like(vec) * float(cfg.text_noise_std)
            if cfg.text_dropout > 0:
                vec = torch.nn.functional.dropout(vec, p=min(0.9, float(cfg.text_dropout)), training=True)
        data.text_emb = vec
        data_list.append(data)
        labels.append(int(label_id))

    if not data_list:
        raise ValueError(f"No samples produced for {ds}")

    collate_and_save(
        dataset_name=ds,
        data_list=data_list,
        labels=labels,
        out_root=cfg.processed_root,
        split_ratios=cfg.split_ratios,
        split_seed=cfg.split_seed,
        split_strategy=cfg.split_type,
        raw_dir=raw_dir,
        label_cfg=label_cfg,
        time_unit=time_source,
        generation_command=f"processor.py --config <resolved> --dataset {ds}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Early propagation PyG preprocessor")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str,
                        choices=["twitter15", "twitter16", "weibo", "pheme"], required=True)
    parser.add_argument("--regen_splits_only", action="store_true",
                        help="Recompute splits.json using existing processed data without rebuilding features")
    parser.add_argument("--write_manifests_only", action="store_true",
                        help="Write canonical manifests for existing processed data and splits without changing splits")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset not in cfg.datasets:
        raise ValueError(
            f"Dataset {args.dataset} not in configured list: {cfg.datasets}")
    if args.regen_splits_only and args.write_manifests_only:
        raise ValueError("--regen_splits_only and --write_manifests_only are mutually exclusive")
    if args.regen_splits_only:
        data_path = os.path.join(cfg.processed_root, args.dataset, "pyg", "data.pt")
        slices_path = os.path.join(cfg.processed_root, args.dataset, "pyg", "slices.pt")
        assert os.path.isfile(data_path), f"Missing {data_path}"
        assert os.path.isfile(slices_path), f"Missing {slices_path}"
        labels = load_labels_from_processed(data_path, slices_path)
        splits = generate_splits(labels, cfg.split_ratios, cfg.split_seed, cfg.split_type)
        splits_path = os.path.join(cfg.processed_root, args.dataset, "splits.json")
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, ensure_ascii=False, indent=2)
        write_manifests_from_processed(
            args.dataset,
            cfg,
            generation_command=f"processor.py --config {args.config} --dataset {args.dataset} --regen_splits_only",
            splits_override=splits,
        )
        print(f"Regenerated stratified splits for {args.dataset} at {splits_path}")
        return
    if args.write_manifests_only:
        write_manifests_from_processed(
            args.dataset,
            cfg,
            generation_command=f"processor.py --config {args.config} --dataset {args.dataset} --write_manifests_only",
        )
        print(f"Wrote canonical manifests for {args.dataset} under {os.path.join(cfg.processed_root, args.dataset)}")
        return
    process_dataset(args.dataset, cfg)


if __name__ == "__main__":
    main()
