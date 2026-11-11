from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.utils import softmax

from Libs.models.baselines.ContrastiveObjectives import prototype_class_logits, supervised_contrastive_loss


class _DirectionalAttentionEncoder(nn.Module):
    """GAT encoder followed by attention pooling for one propagation direction."""

    def __init__(self, in_channels: int, hidden_channels: int, heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.convs: List[GATConv] = nn.ModuleList()
        self.norms: List[nn.LayerNorm] = nn.ModuleList()
        last_channels = int(in_channels)
        for _ in range(int(num_layers)):
            self.convs.append(
                GATConv(
                    in_channels=last_channels,
                    out_channels=int(hidden_channels),
                    heads=int(heads),
                    concat=True,
                    dropout=float(dropout),
                )
            )
            last_channels = int(hidden_channels) * int(heads)
            self.norms.append(nn.LayerNorm(last_channels))
        self.pool_gate = nn.Linear(last_channels, 1)
        self.output_channels = int(last_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
        weights = softmax(self.pool_gate(x).squeeze(-1), batch).unsqueeze(-1)
        return global_add_pool(x * weights, batch)


class AGAD(nn.Module):
    """Cascade-protocol AGAD baseline with attention, prototypes, and adversarial graph features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        projection_dim: int,
        contrastive_weight: float,
        prototype_weight: float,
        adversarial_weight: float,
        adversarial_epsilon: float,
        contrastive_temperature: float,
        prototype_temperature: float,
        prototype_logit_weight: float,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0 or int(hidden_channels) <= 0 or int(projection_dim) <= 0:
            raise ValueError("AGAD channel dimensions must be positive")
        if int(heads) < 1 or int(num_layers) < 1 or int(num_classes) < 2:
            raise ValueError("AGAD heads, layers, and classes are outside the valid range")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("AGAD dropout must be in [0, 1)")

        self.forward_encoder = _DirectionalAttentionEncoder(
            in_channels=int(in_channels),
            hidden_channels=int(hidden_channels),
            heads=int(heads),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.reverse_encoder = _DirectionalAttentionEncoder(
            in_channels=int(in_channels),
            hidden_channels=int(hidden_channels),
            heads=int(heads),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        encoder_width = self.forward_encoder.output_channels + self.reverse_encoder.output_channels
        self.projection = nn.Sequential(
            nn.Linear(encoder_width, int(projection_dim)),
            nn.LayerNorm(int(projection_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.classifier = nn.Linear(int(projection_dim), int(num_classes))
        self.class_prototypes = nn.Parameter(torch.empty(int(num_classes), int(projection_dim)))
        self.register_buffer("prototype_labels", torch.arange(int(num_classes), dtype=torch.long))

        self.num_classes = int(num_classes)
        self.contrastive_weight = float(contrastive_weight)
        self.prototype_weight = float(prototype_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.adversarial_epsilon = float(adversarial_epsilon)
        self.contrastive_temperature = float(contrastive_temperature)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_logit_weight = float(prototype_logit_weight)
        self.dropout = nn.Dropout(p=float(dropout))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.class_prototypes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        forward_graph = self.forward_encoder(x, edge_index, batch)
        reverse_graph = self.reverse_encoder(x, reverse_edge_index, batch)
        return self.projection(torch.cat([forward_graph, reverse_graph], dim=-1))

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        base_logits = self.classifier(self.dropout(features))
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
        prototype_logits = prototype_class_logits(
            features=features,
            prototypes=self.class_prototypes,
            prototype_labels=self.prototype_labels,
            num_classes=self.num_classes,
            temperature=self.prototype_temperature,
        )
        prototype_loss = F.cross_entropy(prototype_logits, labels)
        clean_loss = (
            classification_loss
            + float(self.contrastive_weight) * contrastive_loss
            + float(self.prototype_weight) * prototype_loss
        )
        if float(self.adversarial_weight) <= 0.0:
            return clean_loss, logits
        feature_grad = torch.autograd.grad(
            clean_loss,
            features,
            retain_graph=True,
            create_graph=False,
        )[0].detach()
        perturbation = float(self.adversarial_epsilon) * F.normalize(feature_grad, p=2, dim=-1)
        adversarial_logits = self.logits_from_features(features + perturbation)
        adversarial_loss = F.cross_entropy(adversarial_logits, labels)
        return clean_loss + float(self.adversarial_weight) * adversarial_loss, logits
