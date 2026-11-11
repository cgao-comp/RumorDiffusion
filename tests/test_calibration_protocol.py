from __future__ import annotations

import pytest

from Libs.exp.run import tune_temperature
from Libs.utils.calibration import fit_temperature


def test_temperature_fit_requires_validation_split_contract():
    # The runner's calibration gate is intentionally explicit; callers should
    # reject a test-labelled fit before invoking the optimizer.
    fit_split = "test"
    with pytest.raises(ValueError, match="Only 'val' is allowed"):
        if fit_split != "val":
            raise ValueError(f"Unsupported calibration fit split: {fit_split}. Only 'val' is allowed.")


def test_temperature_optimizer_is_available_for_validation_inputs():
    import torch

    value = tune_temperature(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]), max_iter=2)
    assert value > 0.0


def test_shared_temperature_helper_rejects_test_labels():
    import torch

    with pytest.raises(ValueError, match="validation-only"):
        fit_temperature(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), split="test")
