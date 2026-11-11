from __future__ import annotations

import pytest
import torch

from Libs.data.processor import _load_text_embedding_cache
from Libs.exp.run import resolve_dataset_text_runtime_cfg


def test_text_cache_requires_complete_manifest(tmp_path):
    dataset_dir = tmp_path / "twitter15"
    dataset_dir.mkdir()
    torch.save({"ids": ["s1"], "emb": torch.ones(1, 2)}, dataset_dir / "text_embs.pt")
    with pytest.raises(ValueError, match="strict manifest"):
        _load_text_embedding_cache(str(tmp_path), "twitter15")


def test_text_cache_rejects_dimension_mismatch(tmp_path):
    dataset_dir = tmp_path / "twitter15"
    dataset_dir.mkdir()
    manifest = {
        "encoder_name": "fixture", "weights_sha256": "a", "tokenizer_sha256": "b",
        "normalization_config_sha256": "c", "source_id_sha256": "d",
        "normalized_text_sha256": "e", "embedding_dimension": 3, "dtype": "float32",
    }
    torch.save({"ids": ["s1"], "emb": torch.ones(1, 2), "manifest": manifest}, dataset_dir / "text_embs.pt")
    with pytest.raises(ValueError, match="dimension mismatch"):
        _load_text_embedding_cache(str(tmp_path), "twitter15")


def test_runtime_text_cache_uses_experiment_processed_root(tmp_path):
    cfg = {
        "dataset": {
            "root": str(tmp_path / "processed_canonical"),
            "text_encoder": {
                "model_name_or_path": "fixture-encoder",
                "cache_name": "text_embs_fixture.pt",
            },
        }
    }

    runtime = resolve_dataset_text_runtime_cfg(cfg)

    assert runtime["processed_root"] == str(tmp_path / "processed_canonical")
    assert runtime["cache_name"] == "text_embs_fixture.pt"
