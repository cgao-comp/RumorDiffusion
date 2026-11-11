from __future__ import annotations

import pytest

from Libs.exp.run import build_checkpoint_metadata, canonical_json_hash, read_yaml_like, validate_checkpoint_metadata
from Libs.utils.artifact_schema import omit_optional_none_fields, reject_placeholders


def test_checkpoint_metadata_mismatch_fails_closed(tmp_path):
    expected = {"format_version": 1, "dataset_name": "twitter15", "seed": 42}
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_checkpoint_metadata(str(tmp_path / "ckpt.pt"), {**expected, "seed": 91}, expected)


def test_checkpoint_metadata_omits_optional_null_config_fields(tmp_path):
    (tmp_path / "splits.json").write_text('{"train": ["a"]}', encoding="utf-8")
    payload = {
        "prefix": {"unit": "minutes", "window": 10, "k": None},
        "training": {"epochs": 1, "max_batches_per_epoch": None},
    }

    metadata = build_checkpoint_metadata(
        model_name="textonly",
        stage="classifier",
        dataset_name="twitter15",
        dataset_root=str(tmp_path),
        seed=42,
        label_mapping={"non-rumor": 0, "rumor": 1},
        config_payload=payload,
    )

    assert metadata["config_payload"] == {
        "prefix": {"unit": "minutes", "window": 10},
        "training": {"epochs": 1},
    }
    assert metadata["config_hash"] == canonical_json_hash(metadata["config_payload"])
    with pytest.raises(ValueError, match="null/placeholder"):
        reject_placeholders({"required": None})


def test_config_sanitization_only_omits_known_optional_empty_fields():
    payload = {
        "sbatch": {"profiles": {"benchmark": {"gres": "", "cuda_devices": [""]}}},
        "rumordiffusion": {"explain": {"source_ids": ""}},
        "required_value": "",
    }

    sanitized = omit_optional_none_fields(payload)

    assert "gres" not in sanitized["sbatch"]["profiles"]["benchmark"]
    assert "cuda_devices" not in sanitized["sbatch"]["profiles"]["benchmark"]
    assert "source_ids" not in sanitized["rumordiffusion"]["explain"]
    assert sanitized["required_value"] == ""
    with pytest.raises(ValueError, match="placeholder"):
        reject_placeholders(sanitized)


def test_canonical_resolved_config_has_no_placeholders(monkeypatch):
    monkeypatch.setenv("RUMORDIFFUSION_PROCESSED_ROOT", "/tmp/processed-canonical")
    monkeypatch.setenv("RUMORDIFFUSION_TEXT_ENCODER", "/tmp/text-encoder")

    resolved = omit_optional_none_fields(read_yaml_like("Libs/configs/exp_canonical.yaml"))

    reject_placeholders(resolved)
