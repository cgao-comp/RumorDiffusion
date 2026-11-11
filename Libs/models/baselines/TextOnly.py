import torch
from torch import nn


class TextOnlyClassifier(nn.Module):
    """Graph-level text-only classifier using cached text embeddings.

    Expects `data.text_emb` of shape [B, D] (graph-level embedding). Falls back
    to `data.x` pooled if text_emb is missing.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        assert in_channels > 0 and num_classes >= 2
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        assert hasattr(data, "text_emb"), "text_emb is required for text-only baseline"
        h = data.text_emb
        return self.net(h)
