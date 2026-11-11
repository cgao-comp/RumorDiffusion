from __future__ import annotations

import json

import pytest

from Libs.data.dataset import GraphDataset
from Libs.data.processor import (
    DatasetConfig,
    LabelConfig,
    collate_and_save,
    collect_json_source_texts,
    write_manifests_from_processed,
)


def test_collate_writes_manifests_and_canonical_dataset_loads(tmp_path, synthetic_graph):
    raw_dir = tmp_path / "raw" / "twitter15"
    out_root = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    (raw_dir / "source.txt").write_text("fixture\n", encoding="utf-8")
    data_list = [synthetic_graph]
    for index in (2, 3):
        clone = synthetic_graph.clone()
        clone.source_id = f"synthetic-{index}"
        data_list.append(clone)
    collate_and_save(
        dataset_name="twitter15",
        data_list=data_list,
        labels=[1, 0, 1],
        out_root=str(out_root),
        split_ratios={"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
        split_seed=42,
        split_strategy="random",
        raw_dir=str(raw_dir),
        label_cfg=LabelConfig(["non-rumor", "rumor"], {"non-rumor": 0, "rumor": 1}, False, False),
        time_unit="minutes",
    )
    dataset_dir = out_root / "twitter15"
    for name in ("raw_manifest.json", "processed_manifest.json", "split_manifest.json", "label_schema.json"):
        assert (dataset_dir / name).is_file()
    loaded = GraphDataset(
        root=str(dataset_dir),
        split="test",
        mode="eval_prefix",
        prefix_cfg={"strategy": "minutes", "unit": "minutes", "window": 10},
        label_schema={"non-rumor": 0, "rumor": 1},
    )
    assert loaded.num_classes == 2
    assert not hasattr(loaded[0], "full_x")


def test_canonical_dataset_rejects_stale_split_manifest(tmp_path, synthetic_graph):
    raw_dir = tmp_path / "raw" / "twitter15"
    out_root = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    data_list = [synthetic_graph]
    for index in (2, 3):
        clone = synthetic_graph.clone()
        clone.source_id = f"synthetic-{index}"
        data_list.append(clone)
    collate_and_save(
        "twitter15", data_list, [0, 1, 0],
        str(out_root), {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3}, 42, "random",
        raw_dir=str(raw_dir),
        label_cfg=LabelConfig(["non-rumor", "rumor"], {"non-rumor": 0, "rumor": 1}, False, False),
    )
    splits_path = out_root / "twitter15" / "splits.json"
    splits_path.write_text(json.dumps({"train": [0], "val": [1], "test": [2]}), encoding="utf-8")
    with pytest.raises(ValueError, match="split_manifest"):
        GraphDataset(root=str(out_root / "twitter15"), split="test", mode="eval_prefix")


def test_manifest_only_rebuild_preserves_existing_splits(tmp_path, synthetic_graph):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "twitter15"
    out_root = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    (raw_dir / "source.txt").write_text("fixture\n", encoding="utf-8")
    data_list = [synthetic_graph]
    for index in (2, 3):
        clone = synthetic_graph.clone()
        clone.source_id = f"synthetic-{index}"
        data_list.append(clone)
    collate_and_save(
        "twitter15",
        data_list,
        [0, 1, 0],
        str(out_root),
        {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
        42,
        "random",
        raw_dir=str(raw_dir),
        label_cfg=LabelConfig(["non-rumor", "rumor"], {"non-rumor": 0, "rumor": 1}, False, False),
    )
    dataset_dir = out_root / "twitter15"
    original_splits = (dataset_dir / "splits.json").read_text(encoding="utf-8")
    for name in ("raw_manifest.json", "processed_manifest.json", "split_manifest.json", "label_schema.json"):
        (dataset_dir / name).unlink()
    cfg = DatasetConfig(
        datasets=["twitter15"],
        raw_root=str(raw_root),
        processed_root=str(out_root),
        split_type="random",
        split_ratios={"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
        split_seed=42,
        label_allowed=["non-rumor", "rumor"],
        label_map={"non-rumor": 0, "rumor": 1},
        label_skip_unknown=False,
        label_overrides={},
        truncation_mode="window",
        truncation_window=10,
        truncation_unit="minutes",
        cutoff_minutes=10,
        enforce_root_component=True,
        features_structural=[],
        augment_enabled=False,
        augment_method="none",
        text_encoder_path=None,
        text_normalize=False,
        text_cache_name="text_embs.pt",
        text_batch_size=4,
        text_normalization={},
        text_noise_std=0.0,
        text_dropout=0.0,
        saving_format="pyg",
        time_source_default="minutes",
        time_source_overrides={},
    )

    write_manifests_from_processed("twitter15", cfg, generation_command="test --write_manifests_only")

    assert (dataset_dir / "splits.json").read_text(encoding="utf-8") == original_splits
    for name in ("raw_manifest.json", "processed_manifest.json", "split_manifest.json", "label_schema.json"):
        assert (dataset_dir / name).is_file()


def test_json_source_texts_use_file_stem_keys(tmp_path):
    raw_dir = tmp_path / "weibo"
    raw_dir.mkdir()
    first = {"source": {"id": "duplicate-source", "content": "alpha"}}
    second = {"source": {"id": "duplicate-source", "content": "beta"}}
    (raw_dir / "alpha.json").write_text(json.dumps(first), encoding="utf-8")
    (raw_dir / "beta.json").write_text(json.dumps(second), encoding="utf-8")

    mapping = collect_json_source_texts(str(raw_dir))

    assert mapping == {"alpha": "alpha", "beta": "beta"}
