from torch import nn


class RootPooling(nn.Module):
    """Root-focused classifier: use root node embedding for graph label."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        assert in_channels > 0 and hidden_channels > 0 and num_classes >= 2
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, data):
        assert hasattr(data, "root_index"), "RootPooling requires root_index"
        root_idx = data.root_index.view(-1)
        h_root = data.x[root_idx]
        return self.net(h_root)
