"""Validation-only calibration and decision-threshold helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def require_validation_split(split: str, *, operation: str) -> None:
    if split != "val":
        raise ValueError(f"{operation} is validation-only; test split is report-only")


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, *, split: str, max_iter: int = 100) -> float:
    """Fit one positive scalar temperature using validation labels only."""
    require_validation_split(split, operation="temperature fitting")
    if logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.size(0) or logits.size(0) == 0:
        raise ValueError("logits and targets must be non-empty and aligned")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    log_temperature = torch.zeros((), dtype=logits.dtype, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=int(max_iter))

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp().clamp_min(1e-6), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp_min(1e-6).cpu())
    if temperature <= 0.0:
        raise ValueError("temperature fitting produced a non-positive value")
    return temperature


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits / float(temperature), dim=-1)


def calibration_record(*, split: str, temperature: float, source_logits_sha256: str, fit_command: str) -> dict[str, Any]:
    require_validation_split(split, operation="calibration recording")
    if not source_logits_sha256 or len(source_logits_sha256) != 64:
        raise ValueError("calibration record requires a source artifact SHA-256")
    return {"schema_version": "1", "fit_split": split, "method": "temperature", "temperature": float(temperature), "source_logits_sha256": source_logits_sha256, "fit_command": fit_command}
