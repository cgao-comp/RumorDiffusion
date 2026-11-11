from __future__ import annotations

import torch

from Libs.utils.metrics import compute_ece, compute_macro_f1


def test_macro_f1_uses_declared_complete_class_space():
    preds = torch.tensor([0, 0, 0])
    targets = torch.tensor([0, 0, 0])
    assert compute_macro_f1(preds, targets, num_classes=2) == 0.5


def test_ece_includes_confidence_zero_in_first_bin():
    probs = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    targets = torch.tensor([0, 1])
    # The half-confidence incorrect example belongs to the first bin and
    # contributes 0.25 to the two-example ECE.
    assert compute_ece(probs, targets, n_bins=2) == 0.25
