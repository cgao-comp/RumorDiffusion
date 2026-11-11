from __future__ import annotations

import pytest

from Libs.data.processor import validate_split_integrity


def test_split_integrity_rejects_overlap_and_missing_ids():
    with pytest.raises(ValueError):
        validate_split_integrity({"train": [0, 1], "val": [1], "test": [2]}, ["a", "b", "c"])
    with pytest.raises(ValueError):
        validate_split_integrity({"train": [0], "val": [1], "test": []}, ["a", "b", "c"])


def test_split_integrity_accepts_disjoint_cascade_ids():
    validate_split_integrity({"train": [0], "val": [1], "test": [2]}, ["a", "b", "c"])
