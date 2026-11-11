from __future__ import annotations

import pytest

from Libs.utils.artifact_schema import ensure_unique_canonical_keys


def _key(source_id: str):
    return {
        "dataset": "twitter15", "split": "test", "horizon_unit": "minutes", "horizon_value": 10,
        "model": "rumordiffusion", "model_variant": "canonical", "split_seed": 42,
        "ctdm_seed": 42, "arbiter_seed": 42, "inference_seed": 42,
        "calibration_tag": "val_temperature", "sampling_S": 1, "source_id": source_id,
    }


def test_duplicate_canonical_key_fails():
    with pytest.raises(ValueError, match="Duplicate canonical key"):
        ensure_unique_canonical_keys([_key("s1"), _key("s2")])


def test_distinct_canonical_keys_pass():
    second = _key("s2")
    second["horizon_value"] = 20
    ensure_unique_canonical_keys([_key("s1"), second])
