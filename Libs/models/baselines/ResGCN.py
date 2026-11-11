from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class _ResBlock(nn.Module):
    """Residual GCN block: BN -> ReLU -> GCNConv -> (+skip)."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.bn = nn.LayerNorm(hidden_channels)
        self.conv = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = torch.relu(out)
        out = self.conv(out, edge_index)
        return x + out


class ResGCN(nn.Module):
    """Static Residual GCN for graph classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert in_channels > 0 and hidden_channels > 0 and num_layers >= 1 and num_classes >= 2
        assert 0.0 <= dropout < 1.0
        self.feat_bn = nn.LayerNorm(in_channels)
        self.feat_conv = GCNConv(in_channels, hidden_channels)
        self.blocks: List[_ResBlock] = nn.ModuleList([
            _ResBlock(hidden_channels) for _ in range(max(0, num_layers - 1))
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.feat_bn(x)
        x = torch.relu(self.feat_conv(x, edge_index))
        for block in self.blocks:
            x = block(x, edge_index)
        g = global_mean_pool(x, batch)
        g = self.dropout(g)
        logits = self.classifier(g)
        return logits
