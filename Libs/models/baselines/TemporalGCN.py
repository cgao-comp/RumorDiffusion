import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class TemporalGCN(nn.Module):
    """GCN with time concatenation for graph classification."""

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

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_c = in_channels + 1  # plus time feature
        for _ in range(num_layers):
            self.convs.append(GCNConv(last_c, hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))
            last_c = hidden_channels
        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        assert hasattr(data, "node_time_minutes_shifted"), "TemporalGCN requires node_time_minutes_shifted"
        t = data.node_time_minutes_shifted.view(-1, 1).to(torch.float32)
        x = torch.cat([data.x, t], dim=1)
        edge_index, batch = data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.classifier(x)
