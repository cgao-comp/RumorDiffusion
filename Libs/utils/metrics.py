import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch

import torch.nn.functional as F


def _as_probabilities(probs: torch.Tensor) -> torch.Tensor:
    if probs.dim() != 2 or probs.size(1) < 2:
        raise ValueError("probs must have shape [N, C] with C >= 2")
    if not torch.isfinite(probs).all():
        raise ValueError("probabilities must be finite")
    if torch.any(probs < 0):
        raise ValueError("probabilities must be non-negative")
    row_sums = probs.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5):
        raise ValueError("probability rows must sum to one")
    return probs


def compute_brier(probs: torch.Tensor, targets: torch.Tensor, num_classes: Optional[int] = None) -> float:
    """Multiclass Brier score over the declared complete class space."""
    probs = _as_probabilities(probs)
    if targets.dim() != 1 or targets.numel() != probs.size(0):
        raise ValueError("targets must have shape [N] matching probs")
    classes = int(num_classes or probs.size(1))
    if classes != probs.size(1) or torch.any((targets < 0) | (targets >= classes)):
        raise ValueError("targets are outside the declared class space")
    one_hot = F.one_hot(targets.to(torch.long), num_classes=classes).to(probs.dtype)
    return float(torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item())


def metrics_from_probabilities(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    ece_bins: int = 15,
) -> Dict[str, Any]:
    """Compute report metrics without reconstructing logits or scanning labels."""
    probs = _as_probabilities(probs)
    classes = int(num_classes or probs.size(1))
    if classes != probs.size(1):
        raise ValueError("num_classes must match probability dimension")
    logits = torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny))
    result = metrics_from_logits(logits, targets, classes, detailed=True, compute_calibration=True, ece_bins=ece_bins)
    result["brier"] = compute_brier(probs, targets, classes)
    return result


def seed_level_summary(values: Sequence[float], *, ci: str = "student_t", confidence: float = 0.95) -> Dict[str, Any]:
    """Summarise independent seed values; pooled-example intervals are disallowed."""
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("seed values must be non-empty and finite")
    n = len(numbers)
    mean = sum(numbers) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in numbers) / (n - 1)) if n > 1 else 0.0
    if ci not in {"student_t", "none"}:
        raise ValueError(f"Unsupported CI method: {ci}")
    if ci == "none" or n == 1:
        half_width = 0.0
    else:
        try:
            from scipy.stats import t

            critical = float(t.ppf((1.0 + confidence) / 2.0, df=n - 1))
        except Exception as exc:  # pragma: no cover - scipy is an optional runtime dependency
            raise RuntimeError("Student-t CI requires scipy") from exc
        half_width = critical * sd / math.sqrt(n)
    return {
        "values": numbers,
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "ci_method": ci,
        "confidence": confidence,
        "df": n - 1 if n > 1 else 0,
        "ci95": [mean - half_width, mean + half_width],
    }


def paired_bootstrap(
    reference: Mapping[str, float], comparison: Mapping[str, float],
    *, n_bootstrap: int = 2000, seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap of per-example scalar scores keyed by immutable IDs."""
    if set(reference) != set(comparison) or not reference:
        raise ValueError("paired comparison requires identical non-empty sample IDs")
    import random

    ids = sorted(reference)
    deltas = [float(comparison[item]) - float(reference[item]) for item in ids]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("paired scores must be finite")
    rng = random.Random(int(seed))
    observed = sum(deltas) / len(deltas)
    samples = []
    for _ in range(max(1, int(n_bootstrap))):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    p = 2.0 * min(sum(value <= 0.0 for value in samples), sum(value >= 0.0 for value in samples)) / len(samples)
    return {"estimate": observed, "ci95": [lo, hi], "p_value": min(1.0, p), "n": len(ids), "n_bootstrap": len(samples), "seed": int(seed)}


def holm_bonferroni(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values, preserving comparison names."""
    ordered = sorted(((name, float(value)) for name, value in p_values.items()), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, max(running, value * (total - rank)))
        running = candidate
        adjusted[name] = candidate
    return adjusted


def hierarchical_bootstrap(
    grouped_values: Mapping[Any, Sequence[float]],
    *, n_bootstrap: int = 2000, seed: int = 42,
) -> Dict[str, Any]:
    """Outer-seed/inner-example bootstrap; every group must be non-empty."""
    groups = {key: [float(value) for value in values] for key, values in grouped_values.items()}
    if not groups or any(not values for values in groups.values()):
        raise ValueError("hierarchical bootstrap requires non-empty groups")
    if any(not math.isfinite(value) for values in groups.values() for value in values):
        raise ValueError("bootstrap values must be finite")
    import random

    keys = list(groups)
    rng = random.Random(int(seed))
    observed = sum(sum(values) / len(values) for values in groups.values()) / len(groups)
    samples = []
    for _ in range(max(1, int(n_bootstrap))):
        outer = [keys[rng.randrange(len(keys))] for _ in keys]
        means = []
        for key in outer:
            vals = groups[key]
            means.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
        samples.append(sum(means) / len(means))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {"estimate": observed, "ci95": [lo, hi], "n_outer": len(keys), "n_bootstrap": len(samples), "seed": int(seed), "stratification": "outer_seed_inner_example"}


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes accuracy given predicted class indices and targets.

    Args:
        preds: Predicted class indices of shape [N].
        targets: Ground-truth class indices of shape [N].

    Returns:
        Scalar accuracy in [0, 1].
    """
    assert preds.dim() == 1 and targets.dim() == 1
    assert preds.size(0) == targets.size(0)
    return (preds == targets).float().mean().item()


def compute_macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Computes macro-averaged F1 score without external dependencies.

    Args:
        preds: Predicted class indices [N].
        targets: Ground-truth class indices [N].
        num_classes: Total number of classes.

    Returns:
        Macro F1 over all classes.
    """
    assert preds.dim() == 1 and targets.dim() == 1
    assert preds.size(0) == targets.size(0)
    assert num_classes >= 2

    f1_sum = 0.0
    for c in range(num_classes):
        tp = torch.sum((preds == c) & (targets == c)).item()
        fp = torch.sum((preds == c) & (targets != c)).item()
        fn = torch.sum((preds != c) & (targets == c)).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_sum += f1
    return f1_sum / float(num_classes)


def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Computes confusion matrix for multi-class classification.

    Args:
        preds: Predicted class indices of shape [N].
        targets: Ground-truth class indices of shape [N].
        num_classes: Total number of classes.

    Returns:
        Confusion matrix of shape [num_classes, num_classes] where entry [i, j]
        is the count of samples with true label i predicted as j.
    """
    assert preds.dim() == 1 and targets.dim() == 1
    assert preds.size(0) == targets.size(0)
    assert num_classes >= 2

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def compute_per_class_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[int, Dict[str, float]]:
    """Computes precision, recall, and F1 score for each class.

    Args:
        preds: Predicted class indices of shape [N].
        targets: Ground-truth class indices of shape [N].
        num_classes: Total number of classes.

    Returns:
        Dictionary mapping class index to metrics dict containing:
        - 'precision': Precision for the class
        - 'recall': Recall for the class
        - 'f1': F1 score for the class
        - 'support': Number of true instances for the class
    """
    assert preds.dim() == 1 and targets.dim() == 1
    assert preds.size(0) == targets.size(0)
    assert num_classes >= 2

    per_class = {}
    for c in range(num_classes):
        tp = torch.sum((preds == c) & (targets == c)).item()
        fp = torch.sum((preds == c) & (targets != c)).item()
        fn = torch.sum((preds != c) & (targets == c)).item()
        support = torch.sum(targets == c).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return per_class


def compute_weighted_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Computes weighted average of precision, recall, and F1 score.

    Weighted average accounts for class imbalance by weighting each class
    metric by its support (number of true instances).

    Args:
        preds: Predicted class indices of shape [N].
        targets: Ground-truth class indices of shape [N].
        num_classes: Total number of classes.

    Returns:
        Dictionary containing:
        - 'weighted_precision': Support-weighted precision
        - 'weighted_recall': Support-weighted recall
        - 'weighted_f1': Support-weighted F1 score
    """
    assert preds.dim() == 1 and targets.dim() == 1
    assert preds.size(0) == targets.size(0)
    assert num_classes >= 2

    per_class = compute_per_class_metrics(preds, targets, num_classes)
    total_support = sum(m["support"] for m in per_class.values())

    if total_support == 0:
        return {"weighted_precision": 0.0, "weighted_recall": 0.0, "weighted_f1": 0.0}

    weighted_precision = sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
    weighted_recall = sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support

    return {
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def format_metrics_report(
    per_class: Dict[int, Dict[str, float]],
    macro_avg: Dict[str, float],
    weighted_avg: Dict[str, float],
    accuracy: float,
    class_names: Optional[Union[Dict[int, str], List[str]]] = None,
) -> str:
    """Formats classification metrics into a human-readable table.

    Args:
        per_class: Per-class metrics from compute_per_class_metrics.
        macro_avg: Macro-averaged metrics (precision, recall, f1).
        weighted_avg: Weighted-averaged metrics (precision, recall, f1).
        accuracy: Overall accuracy.
        class_names: Optional mapping from class index to name, or list of names.

    Returns:
        Formatted string report similar to sklearn's classification_report.
    """
    if class_names is None:
        class_names = {i: str(i) for i in per_class.keys()}
    elif isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    lines = []
    lines.append("              precision    recall  f1-score   support")
    lines.append("")

    total_support = sum(m["support"] for m in per_class.values())
    max_name_len = max(len(class_names.get(i, str(i))) for i in per_class.keys())
    max_name_len = max(max_name_len, len("weighted avg"))

    for class_idx in sorted(per_class.keys()):
        m = per_class[class_idx]
        name = class_names.get(class_idx, str(class_idx))
        lines.append(
            f"{name:>{max_name_len}}       {m['precision']:0.2f}      {m['recall']:0.2f}      "
            f"{m['f1']:0.2f}   {m['support']:7d}"
        )

    lines.append("")
    lines.append(
        f"{'accuracy':>{max_name_len}}                           {accuracy:0.2f}   {total_support:7d}"
    )
    lines.append(
        f"{'macro avg':>{max_name_len}}       {macro_avg['precision']:0.2f}      "
        f"{macro_avg['recall']:0.2f}      {macro_avg['f1']:0.2f}   {total_support:7d}"
    )
    lines.append(
        f"{'weighted avg':>{max_name_len}}       {weighted_avg['weighted_precision']:0.2f}      "
        f"{weighted_avg['weighted_recall']:0.2f}      {weighted_avg['weighted_f1']:0.2f}   {total_support:7d}"
    )

    return "\n".join(lines)


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width bins."""
    assert probs.dim() == 2
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)
    bins = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        # Include the left edge of the first bin so confidence=0.0 is covered.
        if i == 0:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].float().mean()
        ece += (mask.float().mean()) * torch.abs(avg_conf - avg_acc)
    return float(ece.detach().cpu())


def metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    detailed: bool = False,
    class_names: Optional[Union[Dict[int, str], List[str]]] = None,
    compute_calibration: bool = False,
    ece_bins: int = 15,
) -> Dict[str, Union[float, str, Dict]]:
    """Returns classification metrics from raw logits.

    Args:
        logits: Raw model outputs [N, C].
        targets: Ground-truth labels [N].
        num_classes: Number of classes.
        detailed: If True, return comprehensive metrics including per-class,
                  macro/weighted averages, and formatted report.
        class_names: Optional class names for formatted report.

    Returns:
        Dict with keys: 'acc', 'macro_f1'.
        If detailed=True, additionally includes:
        - 'per_class': Per-class metrics
        - 'macro_precision', 'macro_recall': Macro averages
        - 'weighted_precision', 'weighted_recall', 'weighted_f1': Weighted averages
        - 'confusion_matrix': Confusion matrix as tensor
        - 'report': Formatted string report
    """
    assert logits.dim() == 2 and logits.size(1) == num_classes
    preds = torch.argmax(logits, dim=-1)
    acc = compute_accuracy(preds, targets)
    macro_f1 = compute_macro_f1(preds, targets, num_classes)
    probs = torch.softmax(logits, dim=1)

    nll = float(F.cross_entropy(logits, targets).item())
    result: Dict[str, Union[float, str, Dict]] = {
        "acc": acc,
        "macro_f1": macro_f1,
        "loss": nll,
        "nll": nll,
    }
    if compute_calibration:
        result["ece"] = compute_ece(probs, targets, n_bins=ece_bins)

    if detailed:
        per_class = compute_per_class_metrics(preds, targets, num_classes)
        weighted = compute_weighted_metrics(preds, targets, num_classes)
        cm = compute_confusion_matrix(preds, targets, num_classes)

        macro_precision = sum(m["precision"] for m in per_class.values()) / num_classes
        macro_recall = sum(m["recall"] for m in per_class.values()) / num_classes

        macro_avg = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1}

        report = format_metrics_report(per_class, macro_avg, weighted, acc, class_names)

        result.update(
            {
                "per_class": per_class,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "weighted_precision": weighted["weighted_precision"],
                "weighted_recall": weighted["weighted_recall"],
                "weighted_f1": weighted["weighted_f1"],
                "confusion_matrix": cm,
                "report": report,
            }
        )

    return result
