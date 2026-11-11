"""Prediction-set metrics shared by selective and stochastic evaluations.

The functions operate on one row per example.  They never infer a threshold
from test labels; callers must pass a threshold selected on validation data.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from Libs.utils.metrics import compute_macro_f1


def _arrays(probs: Sequence[Sequence[float]], y_true: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.ndim != 2 or p.shape[1] < 2 or y.ndim != 1 or len(y) != len(p):
        raise ValueError("probs must be [N,C] and y_true must be [N]")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probabilities must be finite, non-negative and row-normalized")
    return p, y


def max_probability_confidence(probs: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("probs must be a matrix")
    return p.max(axis=1)


def predictive_entropy(probs: Sequence[Sequence[float]], *, base: float = math.e) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probs must be finite non-negative matrix")
    if not math.isfinite(float(base)) or float(base) <= 0.0 or math.isclose(float(base), 1.0):
        raise ValueError("entropy base must be finite, positive and not one")
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1) / math.log(float(base))


def trajectory_variance(trajectories: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """Per-example mean variance over stochastic trajectory probabilities."""
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 2:
        raise ValueError("trajectories must have shape [N,S,C]")
    if not np.isfinite(values).all():
        raise ValueError("trajectory probabilities must be finite")
    return values.var(axis=1, ddof=0).mean(axis=1)


def coverage(accepted: Sequence[bool]) -> float:
    values = np.asarray(accepted, dtype=bool)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("accepted must be a non-empty vector")
    return float(values.mean())


def selective_risk(y_true: Sequence[int], y_pred: Sequence[int], accepted: Sequence[bool]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    a = np.asarray(accepted, dtype=bool)
    if y.ndim != 1 or len(y) == 0 or p.shape != y.shape or a.shape != y.shape:
        raise ValueError("y_true, y_pred and accepted must have matching shape")
    if not a.any():
        return 0.0
    return float(np.mean(y[a] != p[a]))


def resolved_accuracy(y_true: Sequence[int], y_pred: Sequence[int], accepted: Sequence[bool]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    a = np.asarray(accepted, dtype=bool)
    if y.ndim != 1 or len(y) == 0 or p.shape != y.shape or a.shape != y.shape:
        raise ValueError("y_true, y_pred and accepted must have matching non-empty shape")
    return float(np.mean(y[a] == p[a])) if a.any() else 0.0


def resolved_macro_f1(y_true: Sequence[int], y_pred: Sequence[int], accepted: Sequence[bool], num_classes: int | None = None, *, labels: Sequence[int] | None = None) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    a = np.asarray(accepted, dtype=bool)
    if y.ndim != 1 or len(y) == 0 or p.shape != y.shape or a.shape != y.shape:
        raise ValueError("y_true, y_pred and accepted must have matching non-empty shape")
    if labels is not None:
        num_classes = len(labels)
    if num_classes is None or num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if not a.any():
        return 0.0
    import torch

    return compute_macro_f1(torch.as_tensor(p[a]), torch.as_tensor(y[a]), int(num_classes))


def risk_coverage_curve(y_true: Sequence[int], y_pred: Sequence[int], confidence: Sequence[float]) -> dict[str, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    c = np.asarray(confidence, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or c.shape != y.shape or len(y) == 0:
        raise ValueError("curve inputs must have matching non-empty shape")
    if not np.isfinite(c).all():
        raise ValueError("confidence must be finite")
    order = np.argsort(-c, kind="mergesort")
    errors = (y[order] != p[order]).astype(float)
    cumulative = np.cumsum(errors)
    coverage_values = np.concatenate(([0.0], (np.arange(len(y)) + 1) / len(y)))
    risk_values = np.concatenate(([0.0], cumulative / (np.arange(len(y)) + 1)))
    threshold_values = np.concatenate(([float("inf")], c[order]))
    return {"coverage": coverage_values, "risk": risk_values, "threshold": threshold_values}


def _curve_arrays(curve: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(curve, Mapping):
        coverage_values = np.asarray(curve["coverage"], dtype=float)
        risk_values = np.asarray(curve["risk"], dtype=float)
    else:
        coverage_values = np.asarray([float(row["coverage"]) for row in curve], dtype=float)
        risk_values = np.asarray([float(row["risk"]) for row in curve], dtype=float)
    if coverage_values.ndim != 1 or risk_values.shape != coverage_values.shape or len(coverage_values) == 0:
        raise ValueError("invalid risk-coverage curve")
    return coverage_values, risk_values


def aurc(curve_or_y_true: Any, y_pred: Sequence[int] | None = None, confidence: Sequence[float] | None = None) -> float:
    if y_pred is not None and confidence is not None:
        curve = risk_coverage_curve(curve_or_y_true, y_pred, confidence)
    else:
        curve = curve_or_y_true
    coverage_values, risk_values = _curve_arrays(curve)
    if len(coverage_values) == 1:
        raise ValueError("risk-coverage curve is empty")
    order = np.argsort(coverage_values)
    return float(np.trapezoid(risk_values[order], coverage_values[order]))


def eaurc(curve_or_y_true: Any, y_pred: Sequence[int] | None = None, confidence: Sequence[float] | None = None) -> float:
    """Excess AURC over the oracle risk-coverage curve for this error count."""
    curve = risk_coverage_curve(curve_or_y_true, y_pred, confidence) if y_pred is not None and confidence is not None else curve_or_y_true
    coverage_values, risk_values = _curve_arrays(curve)
    n = len(coverage_values) - 1
    errors = int(round(float(risk_values[-1]) * n))
    oracle_risk = np.concatenate(([0.0], [max(0, index - (n - errors)) / index for index in range(1, n + 1)]))
    return max(0.0, aurc(curve) - float(np.trapezoid(oracle_risk, coverage_values)))


def e_aurc(curve_or_y_true: Any, y_pred: Sequence[int] | None = None, confidence: Sequence[float] | None = None) -> float:
    return eaurc(curve_or_y_true, y_pred, confidence)


def risk_at_coverage(curve_or_y_true: Any, y_pred_or_target: Any, confidence: Sequence[float] | None = None, target_coverage: float | None = None) -> float:
    if confidence is not None:
        curve, target_coverage = risk_coverage_curve(curve_or_y_true, y_pred_or_target, confidence), target_coverage
    else:
        curve, target_coverage = curve_or_y_true, y_pred_or_target
    if target_coverage is None:
        raise ValueError("target_coverage is required")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0,1]")
    cover, risk = _curve_arrays(curve)
    index = int(np.searchsorted(cover, target_coverage, side="left"))
    return float(risk[min(index, len(risk) - 1)])


def coverage_at_target_risk(curve_or_y_true: Any, y_pred_or_target: Any, confidence: Sequence[float] | None = None, target_risk: float | None = None) -> float:
    if confidence is not None:
        curve, target_risk = risk_coverage_curve(curve_or_y_true, y_pred_or_target, confidence), target_risk
    else:
        curve, target_risk = curve_or_y_true, y_pred_or_target
    if target_risk is None:
        raise ValueError("target_risk is required")
    if not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be in [0,1]")
    cover, risk = _curve_arrays(curve)
    eligible = cover[risk <= target_risk]
    return float(eligible.max()) if len(eligible) else 0.0


def deployment_cost(
    y_true: Sequence[int], y_pred: Sequence[int], accepted: Sequence[bool],
    *, fp_cost: float = 1.0, fn_cost: float = 1.0, reject_cost: float = 0.0, delay_cost: float = 0.0,
    detection_delay: Sequence[float] | None = None,
    delay: Sequence[float] | None = None,
    return_breakdown: bool = False,
) -> float | dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    a = np.asarray(accepted, dtype=bool)
    if y.shape != p.shape or y.shape != a.shape:
        raise ValueError("deployment cost inputs must have matching shape")
    # For multiclass reporting, FP/FN are counted at the example level.
    fp = np.logical_and(a, p != y).sum()
    fn = np.logical_and(a, p != y).sum()
    rejects = (~a).sum()
    delay_values = detection_delay if detection_delay is not None else delay
    delay_total = 0.0 if delay_values is None else float(np.asarray(delay_values, dtype=float).sum())
    # For multiclass data, a wrong accepted decision contributes one FP and one FN
    # at the example level.  Keep the explicit breakdown for auditability.
    breakdown = {"false_positive": float(fp_cost * fp), "false_negative": float(fn_cost * fn), "reject": float(reject_cost * rejects), "delay": float(delay_cost * delay_total)}
    breakdown["total"] = float(sum(breakdown.values()))
    return breakdown if return_breakdown else breakdown["total"]


def selective_metrics(
    y_true: Sequence[int], probs: Sequence[Sequence[float]], *, threshold: float,
    num_classes: int | None = None, y_pred: Sequence[int] | None = None,
    detection_delay: Sequence[float] | None = None,
    cost: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    p, y = _arrays(probs, y_true)
    pred = p.argmax(axis=1) if y_pred is None else np.asarray(y_pred, dtype=int)
    if pred.shape != y.shape:
        raise ValueError("y_pred shape does not match y_true")
    conf = max_probability_confidence(p)
    accepted = conf >= float(threshold)
    curve = risk_coverage_curve(y, pred, conf)
    costs = dict(cost or {})
    return {
        "threshold": float(threshold),
        "coverage": coverage(accepted),
        "selective_risk": selective_risk(y, pred, accepted),
        "resolved_accuracy": resolved_accuracy(y, pred, accepted),
        "resolved_macro_f1": resolved_macro_f1(y, pred, accepted, int(num_classes or p.shape[1])),
        "risk_coverage_curve": curve,
        "aurc": aurc(curve),
        "eaurc": eaurc(curve),
        "risk_at_80_coverage": risk_at_coverage(curve, 0.8),
        "coverage_at_target_risk": coverage_at_target_risk(curve, float(costs.pop("target_risk", 0.2))),
        "predictive_entropy_mean": float(predictive_entropy(p).mean()),
        "max_probability_confidence_mean": float(conf.mean()),
        "deployment_cost": deployment_cost(y, pred, accepted, detection_delay=detection_delay, **costs),
        "accepted_count": int(accepted.sum()),
        "total_count": int(len(y)),
    }


def select_reject_threshold(
    y_true: Sequence[int], probs: Sequence[Sequence[float]], *, target_risk: float, split: str,
) -> float:
    """Select a reject threshold on validation only."""
    if split != "val":
        raise ValueError("reject threshold fitting is validation-only; test split is report-only")
    if not 0.0 <= float(target_risk) <= 1.0:
        raise ValueError("target_risk must lie in [0,1]")
    p, y = _arrays(probs, y_true)
    curve = risk_coverage_curve(y, p.argmax(axis=1), max_probability_confidence(p))
    eligible = [
        (float(coverage_value), float(threshold_value))
        for coverage_value, risk_value, threshold_value in zip(curve["coverage"], curve["risk"], curve["threshold"])
        if float(coverage_value) > 0.0
        and float(risk_value) <= float(target_risk)
        and math.isfinite(float(threshold_value))
    ]
    if not eligible:
        return 1.0
    return float(max(eligible, key=lambda item: (item[0], -item[1]))[1])
