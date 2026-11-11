from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from Libs.models.baselines.ContrastiveObjectives import prototype_class_logits, supervised_contrastive_loss
from Libs.models.modules.root_aware_cascade import RootAwareCascadeEncoder


class HPCLCascade(nn.Module):
    """HPCL cascade-protocol adaptation without user-history side information."""

    def __init__(
        self,
        in_channels: int,
        text_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        projection_dim: int,
        prototypes_per_class: int,
        contrastive_weight: float,
        prototype_weight: float,
        contrastive_temperature: float,
        prototype_temperature: float,
        prototype_logit_weight: float,
        curriculum_warmup_epochs: int,
        curriculum_min_weight: float,
        use_text: bool,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0 or int(hidden_channels) <= 0 or int(projection_dim) <= 0:
            raise ValueError("HPCL-cascade channel dimensions must be positive")
        if int(num_layers) < 1 or int(num_classes) < 2 or int(prototypes_per_class) < 1:
            raise ValueError("HPCL-cascade layers, classes, and prototypes are outside the valid range")
        if bool(use_text) and int(text_channels) <= 0:
            raise ValueError("HPCL-cascade text_channels must be positive when text fusion is enabled")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("HPCL-cascade dropout must be in [0, 1)")

        self.graph_encoder = RootAwareCascadeEncoder(
            in_channels=int(in_channels),
            hidden_channels=int(hidden_channels),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.use_text = bool(use_text)
        text_width = int(hidden_channels) if self.use_text else 0
        if self.use_text:
            self.text_projection = nn.Sequential(
                nn.Linear(int(text_channels), int(hidden_channels)),
                nn.LayerNorm(int(hidden_channels)),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
            )
        fusion_width = int(self.graph_encoder.output_channels) + text_width
        self.feature_projection = nn.Sequential(
            nn.Linear(fusion_width, int(projection_dim)),
            nn.LayerNorm(int(projection_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.classifier = nn.Linear(int(projection_dim), int(num_classes))
        prototype_count = int(num_classes) * int(prototypes_per_class)
        prototype_labels = torch.arange(int(num_classes), dtype=torch.long).repeat_interleave(int(prototypes_per_class))
        self.class_prototypes = nn.Parameter(torch.empty(prototype_count, int(projection_dim)))
        self.register_buffer("prototype_labels", prototype_labels)
        self.register_buffer("curriculum_weight", torch.tensor(float(curriculum_min_weight), dtype=torch.float32))

        self.num_classes = int(num_classes)
        self.contrastive_weight = float(contrastive_weight)
        self.prototype_weight = float(prototype_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_logit_weight = float(prototype_logit_weight)
        self.curriculum_warmup_epochs = int(curriculum_warmup_epochs)
        self.curriculum_min_weight = float(curriculum_min_weight)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.class_prototypes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_training_progress(self, epoch_index: int, total_epochs: int) -> None:
        warmup = max(1, int(self.curriculum_warmup_epochs))
        progress = min(1.0, float(epoch_index + 1) / float(warmup))
        weight = float(self.curriculum_min_weight) + (1.0 - float(self.curriculum_min_weight)) * progress
        self.curriculum_weight.fill_(float(weight))

    def encode(self, data) -> torch.Tensor:
        graph_features = self.graph_encoder(data.x, data.edge_index, data.batch, data.root_index)
        if self.use_text:
            if not hasattr(data, "text_emb"):
                raise AttributeError("HPCL-cascade text fusion requires data.text_emb")
            text_features = self.text_projection(data.text_emb.to(graph_features.dtype))
            graph_features = torch.cat([graph_features, text_features], dim=-1)
        return self.feature_projection(graph_features)

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        base_logits = self.classifier(features)
        proto_logits = prototype_class_logits(
            features=features,
            prototypes=self.class_prototypes,
            prototype_labels=self.prototype_labels,
            num_classes=self.num_classes,
            temperature=self.prototype_temperature,
        )
        return base_logits + float(self.prototype_logit_weight) * proto_logits

    def forward_with_features(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(data)
        return self.logits_from_features(features), features

    def forward(self, data) -> torch.Tensor:
        logits, _ = self.forward_with_features(data)
        return logits

    def training_step(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = data.y.view(-1).to(torch.long)
        logits, features = self.forward_with_features(data)
        classification_loss = F.cross_entropy(logits, labels)
        contrastive_loss = supervised_contrastive_loss(features, labels, self.contrastive_temperature)
        proto_logits = prototype_class_logits(
            features=features,
            prototypes=self.class_prototypes,
            prototype_labels=self.prototype_labels,
            num_classes=self.num_classes,
            temperature=self.prototype_temperature,
        )
        prototype_loss = F.cross_entropy(proto_logits, labels)
        auxiliary_loss = (
            float(self.contrastive_weight) * contrastive_loss
            + float(self.prototype_weight) * prototype_loss
        )
        return classification_loss + self.curriculum_weight.to(features.device) * auxiliary_loss, logits
