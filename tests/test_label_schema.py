from __future__ import annotations

import pytest

from Libs.data.processor import LabelConfig, _label_schema_payload


def test_label_schema_is_declared_and_contiguous():
    payload = _label_schema_payload(
        "twitter15",
        LabelConfig(["false", "true"], {"false": 0, "true": 1}, False, False),
    )
    assert payload["num_classes"] == 2
    assert payload["mapping_sha256"]


def test_label_schema_rejects_gapped_ids():
    with pytest.raises(ValueError):
        _label_schema_payload(
            "twitter15",
            LabelConfig(["false", "true"], {"false": 0, "true": 2}, False, False),
        )
