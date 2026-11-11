from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """Graph Convolutional Network for graph classification.

    This model stacks multiple GCNConv layers with BatchNorm and ReLU, followed
    by global mean pooling and a linear classifier.

    Args:
        in_channels: Number of input node features.
        hidden_channels: Number of hidden units per layer.
        num_layers: Number of graph convolution layers.
        num_classes: Number of classes for graph classification.
        dropout: Dropout probability applied after pooling.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert in_channels > 0
        assert hidden_channels > 0
        assert num_layers >= 1
        assert num_classes >= 2
        assert 0.0 <= dropout < 1.0

        self.convs: List[GCNConv] = nn.ModuleList()
        self.bns: List[nn.LayerNorm] = nn.ModuleList()
        last_c = in_channels
        for _ in range(num_layers):
            self.convs.append(GCNConv(last_c, hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))
            last_c = hidden_channels

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

