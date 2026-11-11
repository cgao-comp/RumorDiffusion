"""Root-aware graph encoding for observed-prefix rumor cascades."""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class RootAwareCascadeEncoder(nn.Module):
    """Propagates the observed root context through each cascade GCN layer."""

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if int(in_channels) <= 0 or int(hidden_channels) <= 0 or int(num_layers) < 1:
            raise ValueError("Root-aware cascade encoder dimensions are invalid.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Root-aware cascade encoder dropout must be in [0, 1).")
        self.convs: List[GCNConv] = nn.ModuleList()
        self.norms: List[nn.LayerNorm] = nn.ModuleList()
        last_channels = int(in_channels)
        for _ in range(int(num_layers)):
            self.convs.append(GCNConv(last_channels * 2, int(hidden_channels)))
            self.norms.append(nn.LayerNorm(int(hidden_channels)))
            last_channels = int(hidden_channels)
        self.output_channels = int(hidden_channels) * 2
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        root_index: torch.Tensor,
    ) -> torch.Tensor:
        root_index = root_index.view(-1).to(torch.long)
        if int(root_index.numel()) == 0:
            raise ValueError("Root-aware cascade encoder received no graph roots.")
        for conv, norm in zip(self.convs, self.norms):
            root_features = x[root_index]
            expanded_root = root_features[batch]
            x = conv(torch.cat([x, expanded_root], dim=-1), edge_index)
            x = self.dropout(F.relu(norm(x)))
        graph_features = global_mean_pool(x, batch)
        return torch.cat([graph_features, x[root_index]], dim=-1)
