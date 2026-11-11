from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCN2Conv, global_mean_pool


class GCN2Net(nn.Module):
    """GCNII for graph classification with residual propagation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        alpha: float = 0.1,
        theta: float = 0.5,
    ) -> None:
        super().__init__()
        assert in_channels > 0
        assert hidden_channels > 0
        assert num_layers >= 1
        assert num_classes >= 2

        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.convs: List[GCN2Conv] = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha=float(alpha), theta=float(theta), layer=layer + 1))
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h0 = self.input_lin(x)
        h = h0
        for conv in self.convs:
            h = conv(h, h0, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)
        h = global_mean_pool(h, batch)
        h = self.dropout(h)
        return self.classifier(h)
