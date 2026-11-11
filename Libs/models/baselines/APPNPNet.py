from typing import List

from torch import nn
from torch_geometric.nn import APPNP, global_mean_pool


class APPNPNet(nn.Module):
    """APPNP graph classifier: MLP encoder + APPNP propagation + pooling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        appnp_alpha: float = 0.1,
        appnp_k: int = 10,
    ) -> None:
        super().__init__()
        assert in_channels > 0
        assert hidden_channels > 0
        assert num_layers >= 1
        assert num_classes >= 2

        layers: List[nn.Module] = []
        last = in_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(last, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=float(dropout)))
            last = hidden_channels
        self.encoder = nn.Sequential(*layers)
        self.propagation = APPNP(K=int(appnp_k), alpha=float(appnp_alpha), dropout=float(dropout))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.encoder(x)
        h = self.propagation(h, edge_index)
        h = global_mean_pool(h, batch)
        return self.classifier(h)
