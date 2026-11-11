from typing import List

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Computes supervised contrastive loss over graph-level representations."""
    if features.dim() != 2:
        raise ValueError(f"features must be [batch, dim], got {tuple(features.shape)}")
    labels = labels.view(-1).to(torch.long)
    if labels.numel() != features.size(0):
        raise ValueError("labels length must match the feature batch size")
    if float(temperature) <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    if int(features.size(0)) <= 1:
        return features.sum() * 0.0

    normed = F.normalize(features, p=2, dim=-1)
    logits = torch.matmul(normed, normed.transpose(0, 1)) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(int(features.size(0)), dtype=torch.bool, device=features.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & self_mask.logical_not()
    valid = positive_mask.sum(dim=1) > 0
    if not bool(valid.any().item()):
        return features.sum() * 0.0

    logits_mask = self_mask.logical_not().to(features.dtype)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(torch.finfo(features.dtype).eps))
    positive_log_prob = (positive_mask.to(features.dtype) * log_prob).sum(dim=1)
    positive_count = positive_mask.sum(dim=1).clamp_min(1).to(features.dtype)
    return -(positive_log_prob[valid] / positive_count[valid]).mean()


def prototype_class_logits(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
    num_classes: int,
    temperature: float,
) -> torch.Tensor:
    """Aggregates prototype similarities into class logits with log-sum-exp pooling."""
    if features.dim() != 2 or prototypes.dim() != 2:
        raise ValueError("features and prototypes must be rank-2 tensors")
    if features.size(1) != prototypes.size(1):
        raise ValueError(
            f"prototype width mismatch: features={features.size(1)}, prototypes={prototypes.size(1)}"
        )
    if prototype_labels.numel() != prototypes.size(0):
        raise ValueError("prototype_labels length must match number of prototypes")
    if int(num_classes) < 2:
        raise ValueError("num_classes must be at least 2")
    if float(temperature) <= 0.0:
        raise ValueError("prototype temperature must be positive")

    normalized_features = F.normalize(features, p=2, dim=-1)
    normalized_prototypes = F.normalize(prototypes, p=2, dim=-1)
    similarities = torch.matmul(normalized_features, normalized_prototypes.transpose(0, 1)) / float(temperature)
    class_logits: List[torch.Tensor] = []
    for class_idx in range(int(num_classes)):
        mask = prototype_labels.to(device=features.device).eq(int(class_idx))
        if not bool(mask.any().item()):
            raise ValueError(f"No prototype is assigned to class {int(class_idx)}")
        class_logits.append(torch.logsumexp(similarities[:, mask], dim=1))
    return torch.stack(class_logits, dim=1)
