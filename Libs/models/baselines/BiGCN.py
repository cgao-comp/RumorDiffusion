from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class _GCNEncoder(nn.Module):
    """Stacked GCN encoder with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int) -> None:
        super().__init__()
        assert num_layers >= 1
        self.convs: List[GCNConv] = nn.ModuleList()
        self.bns: List[nn.LayerNorm] = nn.ModuleList()
        last_c = in_channels
        for _ in range(num_layers):
            self.convs.append(GCNConv(last_c, hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))
            last_c = hidden_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        return x


class BiGCN(nn.Module):
    """Static Bi-directional GCN for graph classification.

    Two GCN encoders are applied on forward and reversed edges respectively.
    Pooled representations are concatenated and passed to a classifier.
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
        assert in_channels > 0 and hidden_channels > 0 and num_layers >= 1 and num_classes >= 2
        assert 0.0 <= dropout < 1.0
        self.enc_fwd = _GCNEncoder(in_channels, hidden_channels, num_layers)
        self.enc_bwd = _GCNEncoder(in_channels, hidden_channels, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h_f = self.enc_fwd(x, edge_index)
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        h_b = self.enc_bwd(x, rev_edge_index)
        g_f = global_mean_pool(h_f, batch)
        g_b = global_mean_pool(h_b, batch)
        g = torch.cat([g_f, g_b], dim=-1)
        g = self.dropout(g)
        logits = self.classifier(g)
        return logits
