from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool


def _normalize_time_per_graph(t: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    values = t.view(-1).to(torch.float32)
    output = torch.empty_like(values)
    graph_count = int(batch.max().item()) + 1 if int(batch.numel()) > 0 else 0
    eps = torch.finfo(values.dtype).eps
    for graph_idx in range(graph_count):
        mask = batch.eq(int(graph_idx))
        graph_values = values[mask]
        shifted = graph_values - graph_values.min()
        scale = shifted.max().clamp_min(eps)
        output[mask] = shifted / scale
    return output


def _degree_features(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    device = edge_index.device
    in_degree = torch.zeros(int(num_nodes), device=device, dtype=dtype)
    out_degree = torch.zeros(int(num_nodes), device=device, dtype=dtype)
    if int(edge_index.numel()) > 0:
        ones = torch.ones(edge_index.size(1), device=device, dtype=dtype)
        out_degree.index_add_(0, edge_index[0], ones)
        in_degree.index_add_(0, edge_index[1], ones)
    denom = torch.tensor(max(1, int(num_nodes) - 1), device=device, dtype=dtype)
    return torch.stack([in_degree / denom, out_degree / denom], dim=1)


class _TemporalPropagationLayer(nn.Module):
    """Message-passing layer with learned temporal edge gates."""

    def __init__(self, hidden_channels: int, dropout: float, temporal_decay: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(int(hidden_channels), int(hidden_channels))
        self.message_linear = nn.Linear(int(hidden_channels), int(hidden_channels))
        self.edge_gate = nn.Sequential(
            nn.Linear(5, int(hidden_channels)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_channels), 1),
        )
        self.norm = nn.LayerNorm(int(hidden_channels))
        self.dropout = nn.Dropout(p=float(dropout))
        self.temporal_decay = float(temporal_decay)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        normalized_time: torch.Tensor,
        degree_features: torch.Tensor,
    ) -> torch.Tensor:
        if int(edge_index.numel()) == 0:
            return self.norm(F.relu(self.self_linear(x)))
        src, dst = edge_index[0], edge_index[1]
        delta_forward = (normalized_time[dst] - normalized_time[src]).clamp_min(0.0)
        delta_abs = torch.abs(normalized_time[dst] - normalized_time[src])
        edge_features = torch.stack(
            [
                delta_forward,
                delta_abs,
                degree_features[src, 0],
                degree_features[src, 1],
                degree_features[dst, 0],
            ],
            dim=1,
        )
        temporal_weight = torch.exp(-float(self.temporal_decay) * delta_forward)
        gate = torch.sigmoid(self.edge_gate(edge_features).squeeze(-1))
        edge_weight = temporal_weight * gate
        messages = x[src] * edge_weight.unsqueeze(-1)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        degree_weight = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        degree_weight.index_add_(0, dst, edge_weight.to(x.dtype))
        aggregated = aggregated / degree_weight.clamp_min(torch.finfo(x.dtype).eps).unsqueeze(-1)
        updated = self.self_linear(x) + self.message_linear(aggregated)
        return self.dropout(F.relu(self.norm(updated)))


class TPSO(nn.Module):
    """Temporal propagation structure optimizer for prefix cascade graphs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        temporal_decay: float,
        readout_decay: float,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0 or int(hidden_channels) <= 0:
            raise ValueError("TPSO channel dimensions must be positive")
        if int(num_layers) < 1 or int(num_classes) < 2:
            raise ValueError("TPSO layers and classes are outside the valid range")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("TPSO dropout must be in [0, 1)")
        self.input_projection = nn.Sequential(
            nn.Linear(int(in_channels), int(hidden_channels)),
            nn.LayerNorm(int(hidden_channels)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.layers: List[_TemporalPropagationLayer] = nn.ModuleList(
            [
                _TemporalPropagationLayer(
                    hidden_channels=int(hidden_channels),
                    dropout=float(dropout),
                    temporal_decay=float(temporal_decay),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_channels) * 3, int(hidden_channels)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_channels), int(num_classes)),
        )
        self.readout_decay = float(readout_decay)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _time_weighted_pool(self, x: torch.Tensor, batch: torch.Tensor, normalized_time: torch.Tensor) -> torch.Tensor:
        graph_count = int(batch.max().item()) + 1 if int(batch.numel()) > 0 else 0
        pooled: List[torch.Tensor] = []
        eps = torch.finfo(x.dtype).eps
        for graph_idx in range(graph_count):
            mask = batch.eq(int(graph_idx))
            weights = torch.exp(-float(self.readout_decay) * normalized_time[mask]).to(x.dtype)
            pooled.append((x[mask] * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(eps))
        return torch.stack(pooled, dim=0)

    def forward(self, data) -> torch.Tensor:
        if not hasattr(data, "t"):
            raise AttributeError("TPSO requires node timestamps in data.t")
        x = self.input_projection(data.x)
        normalized_time = _normalize_time_per_graph(data.t, data.batch)
        degree_features = _degree_features(data.edge_index, int(data.num_nodes), x.dtype)
        for layer in self.layers:
            x = layer(x, data.edge_index, normalized_time, degree_features)
        mean_pool = global_mean_pool(x, data.batch)
        root_pool = x[data.root_index.view(-1).to(torch.long)]
        temporal_pool = self._time_weighted_pool(x, data.batch, normalized_time)
        return self.classifier(torch.cat([mean_pool, root_pool, temporal_pool], dim=-1))
