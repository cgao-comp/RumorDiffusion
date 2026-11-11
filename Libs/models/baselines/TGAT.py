from typing import List

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool


class TGAT(nn.Module):
    """Temporal Graph Attention Network for graph classification.

    This is a simplified TGAT implementation using GATConv with temporal features.
    The dataset must provide `data.t` as node timestamps (float).

    Args:
        in_channels: Number of node features.
        hidden_channels: Hidden size per head output.
        heads: Attention heads per layer.
        num_layers: Number of GAT layers.
        num_classes: Number of classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        time_normalization: str,
    ) -> None:
        super().__init__()
        assert in_channels > 0
        assert hidden_channels > 0
        assert heads >= 1
        assert num_layers >= 1
        assert num_classes >= 2
        assert 0.0 <= dropout < 1.0
        if time_normalization not in {"none", "per_graph_max"}:
            raise ValueError(f"Unsupported TGAT time_normalization: {time_normalization}")
        self.time_normalization = time_normalization

        # Add temporal feature dimension
        self.temporal_embed = nn.Linear(1, hidden_channels // 4)
        in_channels_with_time = in_channels + hidden_channels // 4

        self.convs: List[GATConv] = nn.ModuleList()
        self.bns: List[nn.LayerNorm] = nn.ModuleList()
        last_c = in_channels_with_time
        for _ in range(num_layers):
            conv = GATConv(in_channels=last_c, out_channels=hidden_channels, heads=heads, dropout=dropout)
            self.convs.append(conv)
            out_c = hidden_channels * heads
            self.bns.append(nn.LayerNorm(out_c))
            last_c = out_c

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(last_c, num_classes)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _normalize_time(self, t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.time_normalization == "none":
            return t
        t_float = t.to(torch.float32)
        out = torch.empty_like(t_float)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        eps = torch.finfo(t_float.dtype).eps
        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            graph_t = t_float[mask]
            scale = graph_t.max().clamp_min(eps)
            out[mask] = graph_t / scale
        return out

    def forward(self, data):
        x, edge_index, batch, t = data.x, data.edge_index, data.batch, data.t
        
        # Add temporal features
        t_input = self._normalize_time(t, batch)
        t_emb = self.temporal_embed(t_input.unsqueeze(-1))
        x = torch.cat([x, t_emb], dim=-1)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
