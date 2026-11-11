#!/usr/bin/env python3
"""Checks the strict prefix-aware RumorDiffusion potential-cache contract."""

import tempfile
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.data.potential_cache import (  # noqa: E402
    build_potential_cache_metadata,
    load_validated_potential_cache,
    potential_cache_dir,
    potential_cache_payload,
)


def main() -> None:
    ctdm_payload = {"ctdm": {"hidden": 8, "T": 4}, "text": {"enabled": False}}
    prefix_10 = {"strategy": "minutes", "window": 10, "include_root": True}
    prefix_20 = {"strategy": "minutes", "window": 20, "include_root": True}
    metadata_10 = build_potential_cache_metadata("twitter15", 42, prefix_10, ctdm_payload)
    metadata_20 = build_potential_cache_metadata("twitter15", 42, prefix_20, ctdm_payload)
    assert potential_cache_dir("cache", prefix_10) != potential_cache_dir("cache", prefix_20)

    with tempfile.TemporaryDirectory(prefix="rd-potential-cache-") as tmp_dir:
        path = Path(tmp_dir) / "sample.pt"
        torch.save(potential_cache_payload(torch.ones((4, 8)), metadata_10, "sample"), path)
        payload = torch.load(path, weights_only=False)
        validated = load_validated_potential_cache(payload, metadata_10, "sample", str(path))
        assert tuple(validated.shape) == (4, 8)
        try:
            load_validated_potential_cache(payload, metadata_20, "sample", str(path))
        except ValueError as exc:
            assert "prefix_window" in str(exc)
        else:
            raise AssertionError("A mismatched early window must fail closed.")

    print("RumorDiffusion prefix-aware potential-cache validation passed.")


if __name__ == "__main__":
    main()
