from typing import List

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGE(nn.Module):
    """GraphSAGE for graph classification with mean pooling."""

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
        self.convs: List[SAGEConv] = nn.ModuleList()
        self.bns: List[nn.LayerNorm] = nn.ModuleList()
        last_c = in_channels
        for _ in range(num_layers):
            self.convs.append(SAGEConv(last_c, hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))
            last_c = hidden_channels

        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.classifier(x)
