from typing import List

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv


class TGN(nn.Module):
    """Temporal Graph Network for link prediction on per-graph samples.

    This module builds time-aware node embeddings using stacked GAT layers with
    simple temporal embeddings and predicts link existence via a bilinear scorer.

    Expected input `data` fields:
      - `x`: Node features of shape [num_nodes, in_channels]
      - `edge_index`: Directed edges of shape [2, num_edges]
      - `t`: Node timestamps of shape [num_nodes] (float minutes)

    Args:
        in_channels: Number of input node features.
        hidden_channels: Hidden size per layer (before heads aggregation).
        num_layers: Number of GAT layers.
        heads: Number of attention heads per layer.
        dropout: Dropout probability applied to hidden representations.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        assert in_channels > 0
        assert hidden_channels > 0
        assert num_layers >= 1
        assert heads >= 1
        assert 0.0 <= dropout < 1.0

        time_dim = max(1, hidden_channels // 4)
        self.time_embed = nn.Linear(1, time_dim)

        self.convs: List[GATConv] = nn.ModuleList()
        self.norms: List[nn.LayerNorm] = nn.ModuleList()

        last_c = in_channels + time_dim
        for _ in range(num_layers):
            conv = GATConv(in_channels=last_c, out_channels=hidden_channels, heads=heads, dropout=dropout)
            self.convs.append(conv)
            out_c = hidden_channels * heads
            self.norms.append(nn.LayerNorm(out_c))
            last_c = out_c

        self.dropout = nn.Dropout(p=dropout)
        self.link_scorer = nn.Bilinear(last_c, last_c, 1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> Tensor:
        """Returns node embeddings for the input graph.

        Args:
            data: PyG Data with `x`, `edge_index`, and `t`.

        Returns:
            Tensor of shape [num_nodes, hidden] representing node embeddings.
        """
        assert hasattr(data, "x") and hasattr(data, "edge_index")
        assert hasattr(data, "t"), "TGN requires temporal feature 't' on data"

        x: Tensor = data.x
        edge_index: Tensor = data.edge_index
        t: Tensor = data.t

        t_in = t.view(-1, 1).to(dtype=x.dtype)
        t_emb = self.time_embed(t_in)
        h = torch.cat([x, t_emb], dim=-1)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = torch.relu(h)

        h = self.dropout(h)
        return h

    def link_logits(self, z: Tensor, edge_index_pairs: Tensor) -> Tensor:
        """Computes logits for candidate edges.

        Args:
            z: Node embeddings of shape [N, D].
            edge_index_pairs: Candidate edges [2, E].

        Returns:
            Logits of shape [E].
        """
        assert z.dim() == 2 and edge_index_pairs.dim() == 2 and edge_index_pairs.size(0) == 2
        src = edge_index_pairs[0].to(torch.long)
        dst = edge_index_pairs[1].to(torch.long)
        logits = self.link_scorer(z[src], z[dst]).view(-1)
        return logits

    @staticmethod
    def all_missing_directed_pairs(num_nodes: int, existing_edge_index: Tensor, device: torch.device) -> Tensor:
        """Returns all directed non-self pairs not present in `existing_edge_index`.

        This computes a boolean adjacency mask for existing edges and returns
        stacked indices for all missing directed pairs (i, j), i != j.
        """
        assert num_nodes > 0
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        if existing_edge_index.numel() > 0:
            adj[existing_edge_index[0].long(), existing_edge_index[1].long()] = True
        mask = ~adj
        mask.fill_diagonal_(False)
        src, dst = torch.where(mask)
        pairs = torch.stack([src, dst], dim=0)
        return pairs

