import math
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, PNAConv


def _assert_shape(tensor: Tensor, dims: Tuple[int, ...]) -> None:
    """Asserts a tensor has the expected number of dimensions.

    Args:
        tensor: Input tensor to validate.
        dims: Expected shape pattern using -1 for wildcard per dimension length.
    """
    assert tensor.dim() == len(dims), f"Expected {len(dims)} dims, got {tensor.dim()}"
    for i, (got, exp) in enumerate(zip(tensor.shape, dims)):
        if exp != -1:
            assert int(got) == int(exp), f"Dim {i} expected {exp}, got {got}"


def _build_adj_from_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Builds a dense adjacency matrix from 2xE edge_index (directed, binary).

    Args:
        edge_index: Long tensor of shape [2, E].
        num_nodes: Number of nodes in the original graph.

    Returns:
        Dense adjacency tensor of shape [num_nodes, num_nodes] with 0/1 values.
    """
    _assert_shape(edge_index, (2, -1))
    assert edge_index.dtype in (torch.long, torch.int64)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() > 0:
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        assert int(src.max().item()) < num_nodes and int(dst.max().item()) < num_nodes
        adj[src, dst] = 1.0
    return adj


def _extract_obs_mask(times_shifted: Tensor, obs_minutes: int) -> Tensor:
    """Creates a boolean mask for observed nodes given shifted minutes.

    Args:
        times_shifted: Tensor [N] of shifted minutes per node (>=0).
        obs_minutes: Observation cutoff in minutes.

    Returns:
        Boolean mask [N] where True indicates observed node.
    """
    _assert_shape(times_shifted, (-1,))
    min_v = int(times_shifted.min().item())
    assert min_v >= 0
    mask = times_shifted.to(torch.long) <= int(obs_minutes)
    return mask


def _reindex_edges_for_mask(edge_index: Tensor, mask: Tensor) -> Tuple[Tensor, int]:
    """Reindexes edges to the compact index space of kept nodes under a mask.

    Args:
        edge_index: Tensor [2, E] original directed edges.
        mask: Boolean mask [N] of kept nodes.

    Returns:
        (edge_index_new [2, E'], num_nodes_new)
    """
    _assert_shape(edge_index, (2, -1))
    _assert_shape(mask, (-1,))
    kept_indices = torch.nonzero(mask, as_tuple=False).view(-1)
    assert kept_indices.numel() > 0, "No observed nodes under the given cutoff"
    old_to_new = torch.full((int(mask.numel()),), -1, dtype=torch.long, device=mask.device)
    old_to_new[kept_indices] = torch.arange(kept_indices.numel(), device=mask.device, dtype=torch.long)
    src = edge_index[0]
    dst = edge_index[1]
    keep_e = mask[src] & mask[dst]
    src_new = old_to_new[src[keep_e]]
    dst_new = old_to_new[dst[keep_e]]
    edge_new = torch.stack([src_new, dst_new], dim=0).contiguous()
    return edge_new, int(kept_indices.numel())


def _pad_adj(adj: Tensor, N: int) -> Tensor:
    """Pads adjacency to [N, N] with zeros in bottom-right block.

    Args:
        adj: Tensor [n, n].
        N: Target padded size.
    """
    _assert_shape(adj, (-1, -1))
    n = int(adj.size(0))
    assert int(adj.size(1)) == n
    assert n <= N
    if n == N:
        return adj
    pad_rows = N - n
    top = nn.functional.pad(adj, (0, N - n), value=0.0)
    bottom = torch.zeros((pad_rows, N), dtype=adj.dtype, device=adj.device)
    out = torch.cat([top, bottom], dim=0)
    return out


def _pad_x(x: Tensor, N: int) -> Tensor:
    """Pads node features to [N, D].

    Args:
        x: Tensor [n, D].
        N: Target number of nodes.
    """
    _assert_shape(x, (-1, -1))
    n, d = int(x.size(0)), int(x.size(1))
    assert n <= N
    if n == N:
        return x
    pad_rows = N - n
    pad = torch.zeros((pad_rows, d), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


def _flatten_AX(adj: Tensor, x: Tensor) -> Tensor:
    """Flattens adjacency and features to a single vector [N*N + N*D].

    Args:
        adj: Tensor [N, N].
        x: Tensor [N, D].
    """
    _assert_shape(adj, (-1, -1))
    _assert_shape(x, (-1, -1))
    N = int(adj.size(0))
    assert int(adj.size(1)) == N
    assert int(x.size(0)) == N
    return torch.cat([adj.reshape(N * N), x.reshape(N * int(x.size(1)))], dim=0)


def _unflatten_AX(vec: Tensor, N: int, D: int) -> Tuple[Tensor, Tensor]:
    """Inverse of _flatten_AX.

    Args:
        vec: Tensor [N*N + N*D].
        N: Nodes.
        D: Features.
    """
    _assert_shape(vec, (-1,))
    expected = N * N + N * D
    assert int(vec.numel()) == expected, f"Bad vector length {vec.numel()} != {expected}"
    a = vec[: N * N].view(N, N)
    x = vec[N * N :].view(N, D)
    return a, x


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion time steps.

    Produces a fixed, non-trainable embedding from discrete t in [1, T].
    """

    def __init__(self, embedding_dim: int, max_steps: int) -> None:
        super().__init__()
        assert embedding_dim % 2 == 0
        self.embedding_dim = embedding_dim
        self.max_steps = int(max_steps)

        half = embedding_dim // 2
        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / float(half)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        _assert_shape(t, (-1,))
        t = t.to(self.inv_freq.device).to(torch.float32).view(-1, 1)
        angles = t * self.inv_freq.view(1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation from a condition vector.

    Generates gamma and beta to modulate features: out = (1+gamma) * h + beta.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, film_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, film_dim),
            nn.SiLU(),
            nn.Linear(film_dim, hidden_dim * 2),
        )

    def forward(self, cond: Tensor, h: Tensor) -> Tensor:
        _assert_shape(cond, (-1, -1))
        _assert_shape(h, (-1, -1, -1))
        b, n, c = h.shape
        gamma_beta = self.net(cond)
        _assert_shape(gamma_beta, (b, c * 2))
        gamma, beta = gamma_beta[:, :c], gamma_beta[:, c:]
        gamma = gamma.view(b, 1, c)
        beta = beta.view(b, 1, c)
        return (1.0 + gamma) * h + beta


class DenoiserBackbone(nn.Module):
    """GNN backbone to encode node states from noisy X_t with observed edges.

    Uses stacked residual message passing with linear transformations and
    adjacency aggregation at each layer.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float, normalize_adj: bool = True) -> None:
        super().__init__()
        self.normalize_adj = bool(normalize_adj)
        assert in_channels > 0 and hidden_channels > 0 and num_layers >= 1
        assert 0.0 <= dropout < 1.0
        self.layers = nn.ModuleList()
        last = in_channels
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(last, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
            )
            self.layers.append(block)
            last = hidden_channels

    def forward(self, x_t: Tensor, adj_obs: Tensor, return_intermediate: bool = False) -> Tensor:
        _assert_shape(x_t, (-1, -1, -1))
        _assert_shape(adj_obs, (-1, -1, -1))
        h = x_t
        intermediate_states = [h]
        for block in self.layers:
            h = block(h)
            # Residual message passing with optional row-normalized adjacency
            if self.normalize_adj:
                deg = adj_obs.sum(dim=-1, keepdim=True)
                A_hat = adj_obs / (deg + 1e-6)
            else:
                A_hat = adj_obs
            h = torch.matmul(A_hat, h) + h
            intermediate_states.append(h)
        return intermediate_states if return_intermediate else h


class GATv2Backbone(nn.Module):
    """Multi-layer GATv2 backbone over per-graph edge_index.

    Expects batched node features and a Python list of edge indices, one per
    sample in the batch. Returns batched node embeddings with the same padded
    node dimension.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        assert hidden_channels % int(heads) == 0
        self.layers = nn.ModuleList()
        last_dim = int(in_channels)
        out_dim_per_head = int(hidden_channels) // int(heads)
        for _ in range(int(num_layers)):
            self.layers.append(
                GATv2Conv(
                    in_channels=last_dim,
                    out_channels=out_dim_per_head,
                    heads=int(heads),
                    dropout=float(dropout),
                    concat=True,
                    share_weights=False,
                )
            )
            last_dim = int(hidden_channels)
        self.out_norm = nn.LayerNorm(int(hidden_channels))
        self.out_drop = nn.Dropout(p=float(dropout))

    def forward(self, x_t: Tensor, edge_index_list: List[Tensor], return_intermediate: bool = False) -> Tensor:
        _assert_shape(x_t, (-1, -1, -1))
        B = int(x_t.size(0))
        if return_intermediate:
            all_intermediate: List[List[Tensor]] = []
            for b in range(B):
                h = x_t[b]
                ei = edge_index_list[b]
                intermediate_states = [h]
                for conv in self.layers:
                    h = conv(h, ei)
                    h = F.gelu(h)
                    intermediate_states.append(h)
                h = self.out_drop(self.out_norm(h))
                intermediate_states[-1] = h
                all_intermediate.append(intermediate_states)
            return all_intermediate
        outs: List[Tensor] = []
        for b in range(B):
            h = x_t[b]
            ei = edge_index_list[b]
            for conv in self.layers:
                h = conv(h, ei)
                h = F.gelu(h)
            h = self.out_drop(self.out_norm(h))
            outs.append(h)
        return torch.stack(outs, dim=0)


class PNABackbone(nn.Module):
    """Multi-layer PNA backbone over per-graph edge_index.

    Applies a sequence of PNAConv layers to each graph in the batch using its
    own edge_index constructed from observed adjacency.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        deg: Tensor,
        aggregators: List[str],
        scalers: List[str],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        last_dim = int(in_channels)
        for _ in range(int(num_layers)):
            self.layers.append(
                PNAConv(
                    in_channels=last_dim,
                    out_channels=int(hidden_channels),
                    aggregators=list(aggregators),
                    scalers=list(scalers),
                    deg=deg,
                )
            )
            last_dim = int(hidden_channels)
        self.out_norm = nn.LayerNorm(int(hidden_channels))
        self.out_drop = nn.Dropout(p=float(dropout))

    def forward(self, x_t: Tensor, edge_index_list: List[Tensor], return_intermediate: bool = False) -> Tensor:
        _assert_shape(x_t, (-1, -1, -1))
        B = int(x_t.size(0))
        if return_intermediate:
            all_intermediate: List[List[Tensor]] = []
            for b in range(B):
                h = x_t[b]
                ei = edge_index_list[b]
                intermediate_states = [h]
                for conv in self.layers:
                    h = conv(h, ei)
                    h = F.gelu(h)
                    intermediate_states.append(h)
                h = self.out_drop(self.out_norm(h))
                intermediate_states[-1] = h
                all_intermediate.append(intermediate_states)
            return all_intermediate
        outs: List[Tensor] = []
        for b in range(B):
            h = x_t[b]
            ei = edge_index_list[b]
            for conv in self.layers:
                h = conv(h, ei)
                h = F.gelu(h)
            h = self.out_drop(self.out_norm(h))
            outs.append(h)
        return torch.stack(outs, dim=0)


class GatedTextFusion(nn.Module):
    """Gated fusion mechanism for dynamically combining graph and text features.

    Uses a learned gating mechanism to adaptively weight graph and text embeddings:
    z = sigmoid(Linear(concat(h_graph, h_text)))
    h_out = z * h_graph + (1 - z) * h_text
    """

    def __init__(self, hidden_channels: int, text_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.text_dim = int(text_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(int(hidden_channels) + int(text_dim), int(hidden_channels)),
            nn.LayerNorm(int(hidden_channels)),
            nn.SiLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_channels), int(hidden_channels)),
            nn.Sigmoid(),
        )
        self.text_proj = nn.Linear(int(text_dim), int(hidden_channels))

    def forward(self, h_graph: Tensor, h_text: Tensor) -> Tensor:
        """Fuses graph and text features using learned gating.

        Args:
            h_graph: Graph-level representation [B, H]
            h_text: Text embedding [B, text_dim]

        Returns:
            Fused representation [B, H]
        """
        _assert_shape(h_graph, (-1, -1))
        _assert_shape(h_text, (-1, -1))
        assert int(h_graph.size(1)) == self.hidden_channels
        assert int(h_text.size(1)) == self.text_dim

        h_text_proj = self.text_proj(h_text)
        concat_feat = torch.cat([h_graph, h_text], dim=-1)
        gate = self.gate_net(concat_feat)
        h_out = gate * h_graph + (1.0 - gate) * h_text_proj
        return h_out


class AttentionReadout(nn.Module):
    """Attention-based graph readout with optional node mask.

    Computes scores = (tanh(W h) q), applies softmax over valid nodes, and
    returns the weighted sum of node embeddings per graph.
    """

    def __init__(self, hidden_channels: int, attn_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_channels, attn_dim)
        self.query = nn.Parameter(torch.randn(attn_dim))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.query, std=0.02)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, H: Tensor, node_mask: Optional[Tensor]) -> Tensor:
        # H: [B, N, C]; node_mask: [B, N] bool or None
        _assert_shape(H, (-1, -1, -1))
        B, N, C = H.shape
        Hp = torch.tanh(self.proj(H))  # [B, N, A]
        q = self.query.view(1, 1, -1).expand(B, N, -1)
        scores = (Hp * q).sum(dim=-1)  # [B, N]
        if node_mask is None:
            attn = torch.softmax(scores, dim=1)
            attn = self.dropout(attn)
            return (attn.view(B, N, 1) * H).sum(dim=1)

        # Ensure boolean mask
        node_mask = node_mask.to(torch.bool)
        # Identify samples with no valid nodes
        valid_counts = node_mask.sum(dim=1)
        all_empty = valid_counts == 0

        # Mask invalid positions with a large negative value (not -inf) for stable softmax
        scores_masked = scores.masked_fill(~node_mask, -1e9)
        attn_all = torch.softmax(scores_masked, dim=1)
        attn_all = self.dropout(attn_all)

        # Aggregate with attention; handle empty graphs by returning zero vectors
        out = torch.zeros(B, C, dtype=H.dtype, device=H.device)
        if (~all_empty).any():
            keep = ~all_empty
            out[keep] = (attn_all[keep].unsqueeze(-1) * H[keep]).sum(dim=1)
        # For all_empty samples, out rows remain zeros
        return out


class PotentialResidualAdapter(nn.Module):
    """Zero-initialized residual adapter applied after CTDM potential readout.

    Keeping this adapter outside the reverse denoising loop allows candidate
    fine-tuning to improve downstream potentials without destabilizing the
    diffusion trajectory or changing its cached-state semantics.
    """

    def __init__(self, hidden_channels: int, bottleneck_dim: int) -> None:
        super().__init__()
        if hidden_channels <= 0 or bottleneck_dim <= 0:
            raise ValueError("Potential adapter dimensions must be positive.")
        self.norm = nn.LayerNorm(int(hidden_channels))
        self.down = nn.Linear(int(hidden_channels), int(bottleneck_dim))
        self.up = nn.Linear(int(bottleneck_dim), int(hidden_channels))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: Tensor) -> Tensor:
        _assert_shape(h, (-1, -1))
        if int(h.size(-1)) != int(self.norm.normalized_shape[0]):
            raise ValueError(
                "Potential adapter hidden dimension mismatch: "
                f"got {int(h.size(-1))}, expected {int(self.norm.normalized_shape[0])}."
            )
        residual = self.up(F.gelu(self.down(self.norm(h))))
        return h + residual


class CTDMSyn(nn.Module):
    """Conditional Topology Diffusion Model (synchronous variant).

    The module implements a FiLM-conditioned denoiser over flattened [A, X]
    vectors and ELBO-like scoring for classification under class hypotheses.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        T: int,
        obs_minutes: int,
        t_embed_dim: int,
        film_hidden: int,
        max_nodes: int,
        backbone_name: str = "gcn",
        backbone_heads: int = 1,
        backbone_dropout: Optional[float] = None,
        deg_histogram: Optional[Tensor] = None,
        pna_aggregators: Optional[List[str]] = None,
        pna_scalers: Optional[List[str]] = None,
        text_dim: Optional[int] = None,
        text_fuse: str = "add",
        multiscale: bool = False,
    ) -> None:
        super().__init__()
        assert in_channels > 0 and num_classes >= 2
        assert hidden_channels > 0 and num_layers >= 1
        assert 0.0 <= dropout < 1.0
        assert T >= 2 and max_nodes >= 1
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.T = int(T)
        self.obs_minutes = int(obs_minutes)
        self.max_nodes = int(max_nodes)
        # Parameterization and guidance (can be overridden by config/external setters)
        self.parameterization = "v"  # one of {"v", "eps"}
        self.guidance_p = 0.1        # classifier-free guidance dropout prob during training
        self.guidance_scale = 1.5    # classifier-free guidance scale during inference
        self.sampling_eta = 0.0      # DDIM/DDPM interpolation factor for reverse sampling
        # Readout and training policy toggles
        self.use_attn_readout = True
        self.readout = AttentionReadout(hidden_channels=self.hidden_channels, attn_dim=self.hidden_channels, dropout=self.dropout)
        self.multiwindow_enabled = False
        self.cutoff_min = max(1, self.obs_minutes // 2)
        self.cutoff_max = self.obs_minutes
        self.sparse_pairs_enabled = False
        self.sparse_pairs_strategy = "a"  # one of {"a","b","c"}
        self.sparse_pairs_topk = 8
        self.sparse_pairs_lambda_time = 1.0
        self.sparse_pairs_lambda_degree = 1.0
        # Multiscale aggregation
        self.multiscale_enabled = bool(multiscale)
        if self.multiscale_enabled:
            total_dim = int(hidden_channels) * (int(num_layers) + 1)
            self.multiscale_proj = nn.Sequential(
                nn.Linear(total_dim, int(hidden_channels)),
                nn.LayerNorm(int(hidden_channels)),
                nn.SiLU(),
                nn.Dropout(p=float(dropout)),
            )
        # Optional text fusion
        self.text_dim = int(text_dim) if text_dim is not None else None
        self.text_fuse = str(text_fuse).lower() if text_fuse is not None else "add"
        assert self.text_fuse in ["add", "concat", "gate"], f"text_fuse must be 'add', 'concat', or 'gate', got {self.text_fuse}"
        self.text_noise_std = 0.0
        self.text_dropout_p = 0.0
        
        if self.text_dim is not None and self.text_dim > 0:
            if self.text_fuse == "add":
                # Add mode: project to hidden_channels
                self.text_proj = nn.Linear(self.text_dim, self.hidden_channels)
            elif self.text_fuse == "concat":
                # Concat mode: project to hidden_channels, then fuse back to hidden_channels
                self.text_proj = nn.Linear(self.text_dim, self.hidden_channels)
                # Additional fusion layer to reduce 2*hidden back to hidden
                self.text_fusion_layer = nn.Sequential(
                    nn.Linear(self.hidden_channels * 2, self.hidden_channels),
                    nn.LayerNorm(self.hidden_channels),
                    nn.SiLU()
                )
            elif self.text_fuse == "gate":
                # Gated fusion mode: use learned gating mechanism
                self.gated_text_fusion = GatedTextFusion(
                    hidden_channels=self.hidden_channels,
                    text_dim=self.text_dim,
                    dropout=float(dropout)
                )

        # Prototype contrastive and topology regularization hyperparameters
        self.prototype_momentum = 0.9
        self.prototype_temperature = 0.07
        self.lambda_proto = 0.2
        self.lambda_topo_bce = 0.5
        self.lambda_deg_kl = 0.1
        # Full-graph projection to latent x0
        self.full_to_latent = nn.Linear(self.hidden_channels, self.hidden_channels)
        # Optional future-forecast head over graph-level representation
        self.forecast_enabled = False
        self.lambda_forecast = 0.0
        self.forecast_head: nn.Module = nn.Identity()
        # Structure dropout for robustness
        self.edge_drop_rate = 0.0
        # The default identity preserves pretrained CTDM behavior. Candidate
        # configurations can replace it before checkpoint loading/pretraining.
        self.potential_adapter: nn.Module = nn.Identity()
        self.register_buffer("prototypes", torch.zeros((self.num_classes, self.hidden_channels), dtype=torch.float32), persistent=True)
        self._eps = 1e-8
        self._objective_eval_target_access_depth = 0
        # Additional regularizers for class-pair separation
        self.lambda_proto_orth = 0.0
        self.lambda_supcon_pair = 0.0
        self.pair_ids = (1, 2)

        # Denoiser components
        self.backbone_type = str(backbone_name).lower() if backbone_name is not None else "gcn"
        if self.backbone_type == "gatv2":
            heads = int(backbone_heads)
            assert int(hidden_channels) % max(1, heads) == 0
            gat_drop = float(dropout if backbone_dropout is None else backbone_dropout)
            self.backbone = GATv2Backbone(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                heads=heads,
                dropout=gat_drop,
            )
        elif self.backbone_type == "pna":
            assert deg_histogram is not None, "deg_histogram required for PNA"
            agg = pna_aggregators if pna_aggregators is not None else ["mean", "min", "max", "std"]
            scl = pna_scalers if pna_scalers is not None else ["identity", "amplification", "attenuation"]
            self.backbone = PNABackbone(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=float(dropout if backbone_dropout is None else backbone_dropout if backbone_dropout is not None else dropout),
                deg=deg_histogram,
                aggregators=agg,
                scalers=scl,
            )
        else:
            self.backbone = DenoiserBackbone(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                normalize_adj=True,
            )
        self.time_embed = SinusoidalTimeEmbedding(embedding_dim=int(t_embed_dim), max_steps=T)
        self.class_embed = nn.Embedding(num_embeddings=num_classes, embedding_dim=int(t_embed_dim))
        self.film = FiLMLayer(cond_dim=int(t_embed_dim * 2), hidden_dim=hidden_channels, film_dim=int(film_hidden))

        # Heads for predicting epsilon for X and A
        self.eps_x = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, in_channels),
        )
        # Directed low-rank bilinear projections for adjacency epsilon
        self.edge_q = nn.Linear(hidden_channels, hidden_channels)
        self.edge_k = nn.Linear(hidden_channels, hidden_channels)

        # Diffusion schedule buffers
        betas = self._beta_schedule_cosine(self.T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        self._reset_parameters()
        # Runtime state used for potential extraction
        self._last_hidden: Optional[Tensor] = None
        self._last_xobs: Optional[Tensor] = None
        # Saved adjacency scores from last call to _pred_theta
        self._last_scores: Optional[Tensor] = None

    def configure_potential_adapter(self, enabled: bool, bottleneck_dim: Optional[int] = None) -> None:
        """Configures the post-readout residual adapter before checkpoint I/O.

        The adapter is intentionally constructed after ``_reset_parameters``
        so its output projection remains exactly zero at initialization.
        """
        if not bool(enabled):
            self.potential_adapter = nn.Identity()
            return
        width = int(bottleneck_dim if bottleneck_dim is not None else max(1, self.hidden_channels // 4))
        self.potential_adapter = PotentialResidualAdapter(
            hidden_channels=self.hidden_channels,
            bottleneck_dim=width,
        )

    def _adapt_potential(self, h: Tensor) -> Tensor:
        """Applies the optional residual adapter outside reverse denoising."""
        _assert_shape(h, (-1, -1))
        return self.potential_adapter(h)

    def update_prototypes(self, h_graph: Tensor, labels: Tensor) -> None:
        """Updates class prototypes using EMA with per-batch class means.

        Args:
            h_graph: Tensor [B, H] graph-level representations.
            labels: Long tensor [B] class ids in [0, C).
        """
        _assert_shape(h_graph, (-1, -1))
        assert labels.dim() == 1 and int(labels.size(0)) == int(h_graph.size(0))
        num_classes = int(self.num_classes)
        momentum = float(self.prototype_momentum)
        with torch.no_grad():
            for cls in range(num_classes):
                mask = (labels == int(cls))
                if mask.any():
                    mean_vec = h_graph[mask].mean(dim=0)
                    proto = self.prototypes[cls]
                    if float(proto.abs().sum().item()) == 0.0:
                        self.prototypes[cls].copy_(mean_vec)
                    else:
                        self.prototypes[cls].mul_(momentum).add_((1.0 - momentum) * mean_vec)

    def _compute_prototype_contrastive_loss(self, h_graph: Tensor, labels: Tensor) -> Tensor:
        """Computes supervised prototype-contrastive loss via cosine similarity.

        Args:
            h_graph: [B, H] graph-level embeddings.
            labels: [B] int64 class ids.

        Returns:
            Scalar tensor loss.
        """
        _assert_shape(h_graph, (-1, -1))
        assert labels.dim() == 1 and int(labels.size(0)) == int(h_graph.size(0))
        h_norm = F.normalize(h_graph, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(h_norm, proto_norm.t())
        loss = F.cross_entropy(logits / max(self._eps, float(self.prototype_temperature)), labels)
        return loss

    def _compute_proto_orth_loss(self) -> Tensor:
        """Computes orthogonality regularization between prototypes of a class pair.

        Returns:
            Scalar tensor equal to squared cosine similarity between the two prototypes.
        """
        p0, p1 = int(self.pair_ids[0]), int(self.pair_ids[1])
        assert 0 <= p0 < int(self.num_classes) and 0 <= p1 < int(self.num_classes)
        v0 = self.prototypes[p0]
        v1 = self.prototypes[p1]
        n0 = v0.norm(p=2).clamp_min(self._eps)
        n1 = v1.norm(p=2).clamp_min(self._eps)
        v0n = v0 / n0
        v1n = v1 / n1
        cos = (v0n * v1n).sum()
        return (cos * cos)

    def _compute_supcon_pair_loss(self, h_graph: Tensor, labels: Tensor) -> Tensor:
        """Computes supervised contrastive loss over a specific class pair within batch.

        Args:
            h_graph: [B, H] graph-level embeddings.
            labels: [B] ground-truth class ids.

        Returns:
            Scalar tensor loss; zero if fewer than two samples in the pair are present.
        """
        _assert_shape(h_graph, (-1, -1))
        assert labels.dim() == 1 and int(labels.size(0)) == int(h_graph.size(0))
        p0, p1 = int(self.pair_ids[0]), int(self.pair_ids[1])
        pair_mask = (labels == p0) | (labels == p1)
        if not bool(pair_mask.any()):
            return h_graph.new_zeros(())
        h = h_graph[pair_mask]
        y = labels[pair_mask]
        K = int(h.size(0))
        if K <= 1:
            return h_graph.new_zeros(())
        h = F.normalize(h, dim=-1)
        sim = torch.matmul(h, h.t())  # [K, K]
        temp = max(self._eps, float(self.prototype_temperature))
        logits = sim / temp
        eye = torch.eye(K, dtype=torch.bool, device=logits.device)
        # Numerical stability: subtract row-wise max
        row_max = logits.max(dim=1, keepdim=True).values
        logits_stable = logits - row_max
        exp_logits = torch.exp(logits_stable) * (~eye)
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(self._eps)  # [K, 1]
        log_prob_matrix = logits_stable - torch.log(denom)  # [K, K]
        pos_mask = (y.view(-1, 1) == y.view(1, -1)) & (~eye)
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not bool(valid.any()):
            return h_graph.new_zeros(())
        # Average log-prob over positives for each anchor
        log_prob_pos = (log_prob_matrix * pos_mask).sum(dim=1)
        loss_per_anchor = -log_prob_pos / pos_count.clamp_min(1)
        loss = loss_per_anchor[valid].mean()
        return loss

    def _build_pair_mask(self, scores: Tensor, X_obs: Tensor) -> Tensor:
        """Builds candidate edge pair mask according to sparse pair strategy.

        Args:
            scores: [B, N, N] adjacency logits.
            X_obs: [B, N, D] observed node features (padded zeros for missing).

        Returns:
            Boolean tensor [B, N, N] pair mask over valid candidate edges.
        """
        _assert_shape(scores, (-1, -1, -1))
        _assert_shape(X_obs, (-1, -1, -1))
        node_mask = X_obs.abs().sum(dim=-1) > 0  # [B, N]
        base_mask = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1))
        if self.sparse_pairs_enabled:
            if self.sparse_pairs_strategy == "a":
                return base_mask
            if self.sparse_pairs_strategy == "b":
                k = int(max(1, min(self.sparse_pairs_topk, scores.size(-1))))
                topk_idx = torch.topk(scores, k=k, dim=-1).indices  # [B, N, k]
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_idx, True)
                return base_mask & topk_mask
            if self.sparse_pairs_strategy == "c":
                # Destination-only top-k by time/degree heuristic from X_obs
                norm_time = X_obs[..., 0]
                in_deg = X_obs[..., 2]
                out_deg = X_obs[..., 3]
                saliency = self.sparse_pairs_lambda_time * (1.0 - norm_time) + self.sparse_pairs_lambda_degree * (in_deg + out_deg)
                saliency = saliency.masked_fill(~node_mask, float('-inf'))
                k = int(max(1, min(self.sparse_pairs_topk, saliency.size(-1))))
                topk_dst = torch.topk(saliency, k=k, dim=-1).indices  # [B, k]
                dst_mask = torch.zeros_like(saliency, dtype=torch.bool)
                dst_mask.scatter_(dim=-1, index=topk_dst, src=torch.ones_like(topk_dst, dtype=torch.bool))
                return base_mask & dst_mask.unsqueeze(1)
        return base_mask

    def _topology_regularizers(self, scores: Tensor, A_obs: Tensor, X_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes masked BCE and degree-distribution KL regularizers.

        Args:
            scores: [B, N, N] adjacency logits from _pred_theta.
            A_obs: [B, N, N] observed binary adjacency (padded zeros for missing nodes).
            X_obs: [B, N, D] observed features used to derive node masks.

        Returns:
            (bce_loss, deg_kl_loss)
        """
        _assert_shape(scores, (-1, -1, -1))
        _assert_shape(A_obs, (-1, -1, -1))
        _assert_shape(X_obs, (-1, -1, -1))
        pair_mask = self._build_pair_mask(scores, X_obs)
        # Masked BCE over candidate pairs
        mask = pair_mask
        logits_m = scores[mask]
        target_m = A_obs[mask]
        if logits_m.numel() == 0:
            bce = torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
            kl_deg = torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
            return bce, kl_deg
        bce = F.binary_cross_entropy_with_logits(logits_m, target_m, reduction="mean")

        # Degree distribution KL over observed nodes only
        A_pred = torch.sigmoid(scores) * mask.to(scores.dtype)
        A_true = A_obs * mask.to(A_obs.dtype)
        deg_pred = A_pred.sum(dim=-1)  # [B, N]
        deg_true = A_true.sum(dim=-1)
        node_mask = X_obs.abs().sum(dim=-1) > 0
        # Normalize per-sample distributions over valid nodes
        deg_pred = deg_pred * node_mask.to(deg_pred.dtype)
        deg_true = deg_true * node_mask.to(deg_true.dtype)
        sum_pred = deg_pred.sum(dim=-1, keepdim=True).clamp_min(self._eps)
        sum_true = deg_true.sum(dim=-1, keepdim=True).clamp_min(self._eps)
        p = deg_true / sum_true
        q = deg_pred / sum_pred
        kl = (p * (torch.log(p.clamp_min(self._eps)) - torch.log(q.clamp_min(self._eps)))).sum(dim=-1)
        kl_deg = kl.mean()
        return bce, kl_deg

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _beta_schedule_cosine(T: int) -> Tensor:
        steps = int(T)
        s = 0.008
        x = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 1e-5, 0.999)
        return betas

    @contextmanager
    def objective_eval_target_access(self):
        """Allows CTDM validation-objective code to read held-out mature targets.

        This is intentionally narrower than `eval()`: public prefix inference
        must never receive or consume `full_*` mature-state fields, while stage
        one CTDM objective validation needs the full graph as the prediction
        target for validation-only model selection.
        """
        self._objective_eval_target_access_depth += 1
        try:
            yield
        finally:
            self._objective_eval_target_access_depth = max(
                0,
                self._objective_eval_target_access_depth - 1,
            )

    def _mature_state_target_access_allowed(self) -> bool:
        return bool(self.training) or int(self._objective_eval_target_access_depth) > 0

    def _tensorize_full(self, data) -> Tuple[Tensor, Tensor]:
        """Builds padded [A, X] tensors for the full graph.

        Returns:
            (A_full [N,N], X_full [N,D])
        """
        if not self._mature_state_target_access_allowed() and any(
            hasattr(data, field)
            for field in (
                "full_x", "full_edge_index", "full_node_time_minutes", "full_node_time_minutes_shifted",
                "full_node_time_hops", "full_node_time_hops_shifted",
            )
        ):
            raise RuntimeError(
                "CTDM eval received mature-state attributes; use an eval_prefix batch without full_* fields."
            )
        full_edge_index = data.full_edge_index.to(torch.long) if hasattr(data, "full_edge_index") else data.edge_index.to(torch.long)
        full_x = data.full_x.to(torch.float32) if hasattr(data, "full_x") else data.x.to(torch.float32)
        n = int(full_x.size(0))
        if n > self.max_nodes:
            raise ValueError(
                f"Full graph has {n} nodes, which exceeds configured max_nodes={self.max_nodes}. "
                "Increase max_nodes or resolve it automatically from active dataset splits."
            )
        A = _build_adj_from_edge_index(full_edge_index, n)
        X = full_x
        A = _pad_adj(A, self.max_nodes)
        X = _pad_x(X, self.max_nodes)
        return A, X

    def _tensorize_obs(self, data, obs_minutes_override: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Builds padded [A_obs, X_obs] and corresponding masks for observed subgraph.

        Returns:
            A_obs [N,N], X_obs [N,D], M_A [N,N], M_X [N,D]
        """
        time_unit = str(getattr(data, "time_unit", "minutes")).lower()
        if time_unit == "hop":
            if not hasattr(data, "node_time_hops_shifted"):
                raise RuntimeError("CTDM hop input lacks node_time_hops_shifted")
            times = data.node_time_hops_shifted.to(torch.long)
        else:
            times = data.node_time_minutes_shifted.to(torch.long)
        obs_m = int(self.obs_minutes if obs_minutes_override is None else obs_minutes_override)
        if hasattr(data, "prefix_node_mask"):
            mask = data.prefix_node_mask.to(torch.bool)
        else:
            mask = _extract_obs_mask(times, obs_m)
        edge_src = data.edge_index.to(torch.long)
        if hasattr(data, "prefix_edge_index"):
            edge_src = data.prefix_edge_index.to(torch.long)
        edge_obs, n_obs = _reindex_edges_for_mask(edge_src, mask)
        if n_obs > self.max_nodes:
            raise ValueError(
                f"Observed prefix has {n_obs} nodes, which exceeds configured max_nodes={self.max_nodes}. "
                "Increase max_nodes or resolve it automatically from active dataset splits."
            )
        A_obs = _build_adj_from_edge_index(edge_obs, n_obs)
        X_obs = data.x[mask].to(torch.float32)
        if float(getattr(self, "edge_drop_rate", 0.0)) > 0.0:
            drop_mask = torch.rand_like(A_obs, dtype=torch.float32) >= float(self.edge_drop_rate)
            A_obs = A_obs * drop_mask
        # Build masks before padding
        D = int(data.x.size(1))
        M_A_small = torch.ones((n_obs, n_obs), dtype=torch.float32, device=A_obs.device)
        M_X_small = torch.ones((n_obs, D), dtype=torch.float32, device=X_obs.device)
        # Pad tensors and masks to [N,N] and [N,D]
        A_obs = _pad_adj(A_obs, self.max_nodes)
        X_obs = _pad_x(X_obs, self.max_nodes)
        M_A = _pad_adj(M_A_small, self.max_nodes)
        M_X = _pad_x(M_X_small, self.max_nodes)
        return A_obs, X_obs, M_A, M_X

    def _backbone_forward_from_AX(self, A: Tensor, X: Tensor) -> Tensor:
        """Runs configured backbone on dense adjacency and features."""
        _assert_shape(A, (-1, -1, -1))
        _assert_shape(X, (-1, -1, -1))
        if self.backbone_type in ("gatv2", "pna"):
            edge_index_list: List[Tensor] = []
            for b in range(int(A.size(0))):
                ei = (A[b] > 0.5).nonzero(as_tuple=False).t().contiguous()
                edge_index_list.append(ei)
            return self.backbone(X, edge_index_list)
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        A_norm = A / deg
        return self.backbone(X, A_norm)

    @torch.no_grad()
    def encode_prefix(self, batch, obs_minutes_override: Optional[int] = None) -> Tensor:
        """Encodes observed/prefix subgraph into graph-level representation."""
        device = next(self.parameters()).device
        _, Gobs, _, _ = self._pack_batch_G(batch, obs_minutes_override=obs_minutes_override)
        B = int(Gobs.size(0))
        N = self.max_nodes
        D = self.in_channels
        A_list: List[Tensor] = []
        X_list: List[Tensor] = []
        for b in range(B):
            A_o, X_o = _unflatten_AX(Gobs[b], N, D)
            A_list.append(A_o)
            X_list.append(X_o)
        A_obs = torch.stack(A_list, dim=0).to(device)
        X_obs = torch.stack(X_list, dim=0).to(device)
        H = self._backbone_forward_from_AX(A_obs, X_obs)
        h_graph = self._graph_readout(H, X_obs)
        return self._fuse_text_embedding(h_graph, batch)

    @torch.no_grad()
    def encode_full(self, batch) -> Tensor:
        """Encodes full graph into graph-level representation h0."""
        device = next(self.parameters()).device
        data_list = batch.to_data_list()
        A_list: List[Tensor] = []
        X_list: List[Tensor] = []
        for d in data_list:
            A, X = self._tensorize_full(d)
            A_list.append(A)
            X_list.append(X)
        A_full = torch.stack(A_list, dim=0).to(device)
        X_full = torch.stack(X_list, dim=0).to(device)
        H = self._backbone_forward_from_AX(A_full, X_full)
        h_graph = self._graph_readout(H, X_full)
        return self._fuse_text_embedding(h_graph, batch)

    @torch.no_grad()
    def to_x0(self, h_full: Tensor) -> Tensor:
        """Projects full-graph representation to latent x0 space."""
        return self.full_to_latent(h_full)

    def _forecast_targets(self, batch) -> Tensor:
        """Builds simple structural targets [num_nodes_norm, num_edges_norm]."""
        data_list = batch.to_data_list()
        vals: List[List[float]] = []
        max_nodes_sq = float(max(1, self.max_nodes * self.max_nodes))
        for d in data_list:
            n = float(d.num_nodes)
            e = float(d.edge_index.size(1))
            vals.append([n / float(max(1, self.max_nodes)), e / max_nodes_sq])
        device = batch.x.device if hasattr(batch, "x") else None
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def _pack_batch_G(self, batch, obs_minutes_override: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Packs a PyG Batch into flattened G_full, G_obs, observed mask and labels.

        Returns:
            G0: [B, L], Gobs: [B, L], Mobs: [B, L], y: [B]
        """
        data_list = batch.to_data_list()
        G0_list: List[Tensor] = []
        Gobs_list: List[Tensor] = []
        Mobs_list: List[Tensor] = []
        y_list: List[int] = []
        for d in data_list:
            A, X = self._tensorize_full(d)
            Ao, Xo, M_A, M_X = self._tensorize_obs(d, obs_minutes_override=obs_minutes_override)
            G0_list.append(_flatten_AX(A, X))
            Gobs_list.append(_flatten_AX(Ao, Xo))
            Mobs_list.append(_flatten_AX(M_A, M_X))
            y_list.append(int(d.y.view(-1)[0].item()))
        G0 = torch.stack(G0_list, dim=0)
        Gobs = torch.stack(Gobs_list, dim=0)
        Mobs = torch.stack(Mobs_list, dim=0)
        y = torch.tensor(y_list, dtype=torch.long, device=G0.device)
        # Stash last observed adjacency (unpadded reconstructed from stacks)
        try:
            N = self.max_nodes
            Aobs_batch = []
            for g in Gobs:
                A, _X = _unflatten_AX(g, N, self.in_channels)
                Aobs_batch.append(A)
            self._last_Aobs_batch = torch.stack(Aobs_batch, dim=0)
        except Exception:
            pass
        return G0, Gobs, Mobs, y

    def _pred_theta(self, G_t: Tensor, t: Tensor, y: Optional[Tensor], G_obs: Tensor, drop_mask: Optional[Tensor] = None) -> Tensor:
        """Predicts parameterization vector ('eps' or 'v') over flattened [A, X].

        Args:
            G_t: [B, L] noisy vector at time t.
            t: [B] diffusion step indices.
            y: [B] class ids or None for unconditional branch.
            G_obs: [B, L] observed flattened vector.
            drop_mask: Optional [B] boolean mask; when True zeros out class embedding.

        Returns:
            Tensor [B, L] predicted parameterization aligned with self.parameterization.
        """
        _assert_shape(G_t, (-1, -1))
        _assert_shape(t, (-1,))
        _assert_shape(G_obs, (-1, -1))
        B, L = G_t.shape
        N = self.max_nodes
        D = self.in_channels
        assert int(L) == N * N + N * D

        # Unflatten both noisy graph and observed graph
        A_t_list: List[Tensor] = []
        X_t_list: List[Tensor] = []
        A_obs_list: List[Tensor] = []
        X_obs_list: List[Tensor] = []
        for b in range(B):
            A_t, X_t = _unflatten_AX(G_t[b], N, D)
            A_o, X_o = _unflatten_AX(G_obs[b], N, D)
            A_t_list.append(A_t)
            X_t_list.append(X_t)
            A_obs_list.append(A_o)
            X_obs_list.append(X_o)
        A_t = torch.stack(A_t_list, dim=0)
        X_t = torch.stack(X_t_list, dim=0)
        A_obs = torch.stack(A_obs_list, dim=0)
        X_obs = torch.stack(X_obs_list, dim=0)

        # Backbone forward per configured type
        if self.multiscale_enabled:
            if getattr(self, "backbone_type", "gcn") in ("gatv2", "pna"):
                edge_index_list: List[Tensor] = []
                for b in range(B):
                    A_bin = (A_obs[b] > 0.5)
                    ei = A_bin.nonzero(as_tuple=False).t().contiguous()
                    edge_index_list.append(ei)
                intermediate_list = self.backbone(X_t, edge_index_list, return_intermediate=True)
                H_list: List[Tensor] = []
                for b in range(B):
                    intermediate_states = intermediate_list[b]
                    multiscale_feat = torch.cat(intermediate_states, dim=-1)
                    H_b = self.multiscale_proj(multiscale_feat)
                    H_list.append(H_b)
                H = torch.stack(H_list, dim=0)
            else:
                deg = A_obs.sum(dim=-1, keepdim=True).clamp(min=1.0)
                A_obs_norm = A_obs / deg
                intermediate_states = self.backbone(X_t, A_obs_norm, return_intermediate=True)
                multiscale_feat = torch.cat(intermediate_states, dim=-1)
                H = self.multiscale_proj(multiscale_feat)
        else:
            if getattr(self, "backbone_type", "gcn") in ("gatv2", "pna"):
                edge_index_list: List[Tensor] = []
                for b in range(B):
                    A_bin = (A_obs[b] > 0.5)
                    ei = A_bin.nonzero(as_tuple=False).t().contiguous()
                    edge_index_list.append(ei)
                H = self.backbone(X_t, edge_index_list)
            else:
                deg = A_obs.sum(dim=-1, keepdim=True).clamp(min=1.0)
                A_obs_norm = A_obs / deg
                H = self.backbone(X_t, A_obs_norm)

        # FiLM conditioning with optional guidance dropout
        t_emb = self.time_embed(t.to(self.alphas.device))
        if y is None:
            y_emb = torch.zeros_like(t_emb)
        else:
            y_emb = self.class_embed(y.to(self.class_embed.weight.device))
        if drop_mask is not None:
            _assert_shape(drop_mask, (-1,))
            drop_f = drop_mask.view(-1, 1).to(t_emb.dtype)
            y_emb = y_emb * (1.0 - drop_f)
        cond = torch.cat([t_emb, y_emb], dim=-1)
        H = self.film(cond, H)
        # Stash final hidden for downstream potential extraction
        self._last_hidden = H
        self._last_xobs = X_obs

        # Heads
        H_cat = torch.cat([H, X_obs], dim=-1)
        pred_x = self.eps_x(H_cat)
        Q = self.edge_q(H)
        K = self.edge_k(H)
        scores = torch.matmul(Q, K.transpose(1, 2))  # [B, N, N]
        # Stash adjacency scores for topology regularizers
        self._last_scores = scores
        # Build base observed-node mask
        node_mask = X_obs.abs().sum(dim=-1) > 0  # [B, N] bool
        base_mask = (node_mask.unsqueeze(2) & node_mask.unsqueeze(1))  # [B,N,N]
        if self.sparse_pairs_enabled:
            if self.sparse_pairs_strategy == "a":
                pair_mask = base_mask
            elif self.sparse_pairs_strategy == "b":
                # Per-row top-k by similarity
                k = int(max(1, min(self.sparse_pairs_topk, scores.size(-1))))
                topk_idx = torch.topk(scores, k=k, dim=-1).indices  # [B, N, k]
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.scatter_(-1, topk_idx, True)
                pair_mask = base_mask & topk_mask
            elif self.sparse_pairs_strategy == "c":
                # Destination-only top-k by time/degree heuristic from X_obs
                # X_obs feature order: [norm_time, depth, in_degree, out_degree]
                norm_time = X_obs[..., 0]
                in_deg = X_obs[..., 2]
                out_deg = X_obs[..., 3]
                saliency = self.sparse_pairs_lambda_time * (1.0 - norm_time) + self.sparse_pairs_lambda_degree * (in_deg + out_deg)
                saliency = saliency.masked_fill(~node_mask, float('-inf'))
                k = int(max(1, min(self.sparse_pairs_topk, saliency.size(-1))))
                topk_dst = torch.topk(saliency, k=k, dim=-1).indices  # [B, k]
                dst_mask = torch.zeros_like(saliency, dtype=torch.bool)
                dst_mask.scatter_(dim=-1, index=topk_dst, src=torch.ones_like(topk_dst, dtype=torch.bool))
                pair_mask = base_mask & dst_mask.unsqueeze(1)
            else:
                pair_mask = base_mask
        else:
            pair_mask = base_mask
        pred_a = scores * pair_mask.to(scores.dtype)

        pred = torch.cat([pred_a.reshape(B, N * N), pred_x.reshape(B, N * D)], dim=-1)
        return pred

    def _graph_readout(self, H: Tensor, X_obs: Optional[Tensor]) -> Tensor:
        """Aggregates node embeddings to a graph-level vector.

        Uses attention with node mask if enabled; otherwise mean pooling.
        """
        _assert_shape(H, (-1, -1, -1))
        if self.use_attn_readout and X_obs is not None:
            node_mask = X_obs.abs().sum(dim=-1) > 0
            return self.readout(H, node_mask)
        return H.mean(dim=1)

    def _unflatten_batch_AX(self, G: Tensor) -> Tuple[Tensor, Tensor]:
        N = int(self.max_nodes)
        D = int(self.in_channels)
        adj_list: List[Tensor] = []
        x_list: List[Tensor] = []
        for b in range(int(G.size(0))):
            A_b, X_b = _unflatten_AX(G[b], N, D)
            adj_list.append(A_b)
            x_list.append(X_b)
        return torch.stack(adj_list, dim=0), torch.stack(x_list, dim=0)

    def _potential_readout_mask(self, G0: Tensor, X_obs_last: Tensor, readout_source: str) -> Tensor:
        source = str(readout_source).lower()
        if source in {"observed", "observed_prefix"}:
            return X_obs_last
        if source != "sampled_full_graph":
            raise ValueError(
                "CTDM potential readout_source must be one of "
                "'observed_prefix' or 'sampled_full_graph'."
            )
        _A_full, X_full = self._unflatten_batch_AX(G0)
        return X_full.to(device=X_obs_last.device, dtype=X_obs_last.dtype)

    def _potential_readout(
        self,
        H: Tensor,
        G_state: Tensor,
        G0: Tensor,
        X_obs_last: Tensor,
        readout_source: str,
    ) -> Tensor:
        source = str(readout_source).lower()
        readout_mask = self._potential_readout_mask(G0, X_obs_last, source)
        if source in {"observed", "observed_prefix"}:
            return self._graph_readout(H, readout_mask)
        return self._graph_readout(H, readout_mask)

    def _fuse_text_embedding(self, h_graph: Tensor, batch) -> Tensor:
        """Fuse text embeddings into graph-level representation.
        
        Args:
            h_graph: Graph-level representation [B, H]
            batch: Batch object containing text_emb attribute
            
        Returns:
            Fused representation [B, H]
        """
        if self.text_dim is None or not hasattr(batch, 'text_emb'):
            return h_graph
        
        # Ensure text_emb is on the same device as h_graph
        text_emb = batch.text_emb.to(device=h_graph.device, dtype=h_graph.dtype)  # [B, text_dim]
        drop_p = float(getattr(self, "text_dropout_p", 0.0))
        if drop_p > 0.0:
            mask = torch.rand_like(text_emb) > drop_p
            text_emb = text_emb * mask
        noise_std = float(getattr(self, "text_noise_std", 0.0))
        if noise_std > 0.0:
            text_emb = text_emb + noise_std * torch.randn_like(text_emb)
        
        if self.text_fuse == "gate":
            # Gated fusion: use learned gating mechanism
            return self.gated_text_fusion(h_graph, text_emb)
        elif self.text_fuse == "add":
            # Add fusion: keep dimension as H
            text_proj = self.text_proj(text_emb)  # [B, H]
            return h_graph + text_proj
        elif self.text_fuse == "concat":
            # Concat fusion: [B, H] + [B, H] -> [B, 2H] -> [B, H]
            text_proj = self.text_proj(text_emb)  # [B, H]
            fused = torch.cat([h_graph, text_proj], dim=-1)  # [B, 2H]
            return self.text_fusion_layer(fused)  # [B, H]
        else:
            return h_graph

    @torch.no_grad()
    def infer_potential_single(
        self,
        batch,
        class_id: int,
        stride: Optional[int] = None,
        readout_source: str = "observed_prefix",
        apply_potential_adapter: bool = True,
    ) -> Tensor:
        """Runs a reverse denoising trajectory under a class hypothesis and
        returns graph-level potential embeddings from the final hidden states.

        Args:
            batch: PyG Batch for a mini-batch of graphs.
            class_id: Integer class hypothesis in [0, C).
            stride: Optional step stride for faster inference.
            readout_source: Node mask source for the final potential readout.

        Returns:
            Tensor [B, hidden_channels] potential vectors.
        """
        device = next(self.parameters()).device
        G0, Gobs, Mobs, _ = self._pack_batch_G(batch)
        G0 = G0.to(device)
        Gobs = Gobs.to(device)
        Mobs = Mobs.to(device)
        B, L = int(Gobs.size(0)), int(Gobs.size(1))
        y_h = torch.full((B,), int(class_id), dtype=torch.long, device=device)
        Gt = torch.randn((B, L), device=device)

        step_stride = int(stride) if stride is not None else int(getattr(self, "t_stride", self.T // 20 if self.T >= 20 else 1))
        step_stride = max(1, step_stride)
        for step in range(self.T, 0, -step_stride):
            t = torch.full((B,), int(step), dtype=torch.long, device=device)
            prev_t = torch.full((B,), max(0, int(step) - step_stride), dtype=torch.long, device=device)
            # CFG combine conditional/unconditional predictions
            pred_c = self._pred_theta(Gt, t, y_h, Gobs, drop_mask=None)
            pred_u = self._pred_theta(Gt, t, None, Gobs, drop_mask=None)
            pred = pred_u + self.guidance_scale * (pred_c - pred_u)
            eps_pred = self._to_eps(Gt, t, pred)
            Gt = self._reverse_inference_update(Gt, t, eps_pred, prev_t)
            Gt = self._mask_observed(Gt, Gobs, Mobs)
        assert self._last_hidden is not None
        H = self._last_hidden
        X_obs_last = self._last_xobs
        assert X_obs_last is not None
        h = self._potential_readout(H, Gt, G0, X_obs_last, readout_source)
        h = self._fuse_text_embedding(h, batch)
        return self._adapt_potential(h) if apply_potential_adapter else h

    @torch.no_grad()
    def infer_potentials_all(
        self,
        batch,
        stride: Optional[int] = None,
        readout_source: str = "observed_prefix",
        apply_potential_adapter: bool = True,
    ) -> Tensor:
        """Computes potential embeddings for all classes.

        Args:
            batch: PyG Batch.
            stride: Optional step stride.
            readout_source: Node mask source for the final potential readout.

        Returns:
            Tensor [B, C, hidden_channels].
        """
        B = int(batch.y.size(0)) if hasattr(batch, "y") else None
        outs: List[Tensor] = []
        for cls in range(self.num_classes):
            h = self.infer_potential_single(
                batch,
                class_id=int(cls),
                stride=stride,
                readout_source=readout_source,
                apply_potential_adapter=apply_potential_adapter,
            )
            outs.append(h)
        Hc = torch.stack(outs, dim=1)
        if B is not None:
            assert int(Hc.size(0)) == B
        return Hc

    def potentials_all_adapter_grad(self, batch, stride: Optional[int] = None) -> Tensor:
        """Returns adapter-differentiable potentials without trajectory gradients.

        Reverse diffusion remains under ``no_grad`` so a post-readout adapter
        cannot make the iterative trajectory part of an AMP backward graph.
        Only the residual adapter receives gradients from the returned tensor.
        """
        with torch.no_grad():
            base_potentials = self.infer_potentials_all(
                batch,
                stride=stride,
                readout_source="observed_prefix",
                apply_potential_adapter=False,
            )
        batch_size, num_classes, hidden = base_potentials.shape
        if int(hidden) != self.hidden_channels:
            raise ValueError(
                f"Unexpected potential width {int(hidden)}; expected {self.hidden_channels}."
            )
        adapted = self._adapt_potential(base_potentials.reshape(batch_size * num_classes, hidden))
        return adapted.reshape(batch_size, num_classes, hidden)

    @torch.no_grad()
    def sample_potentials_all(
        self,
        batch,
        stride: Optional[int] = None,
        readout_source: str = "sampled_full_graph",
    ) -> Tensor:
        return self.infer_potentials_all(
            batch,
            stride=stride,
            readout_source=readout_source,
        )

    def potentials_all_grad(self, batch, stride: Optional[int] = None) -> Tensor:
        """Gradient-enabled version of infer_potentials_all.

        Runs reverse denoising trajectories for all classes while keeping
        autograd enabled, then returns [B, C, hidden_channels] potentials.
        """
        device = next(self.parameters()).device
        G0, Gobs, Mobs, _ = self._pack_batch_G(batch)
        Gobs = Gobs.to(device)
        Mobs = Mobs.to(device)
        B, L = int(Gobs.size(0)), int(Gobs.size(1))
        outs: List[Tensor] = []
        step_stride = int(stride) if stride is not None else int(getattr(self, "t_stride", self.T // 20 if self.T >= 20 else 1))
        step_stride = max(1, step_stride)
        for cls in range(self.num_classes):
            y_h = torch.full((B,), int(cls), dtype=torch.long, device=device)
            Gt = torch.randn((B, L), device=device)
            for step in range(self.T, 0, -step_stride):
                t = torch.full((B,), int(step), dtype=torch.long, device=device)
                prev_t = torch.full((B,), max(0, int(step) - step_stride), dtype=torch.long, device=device)
                pred_c = self._pred_theta(Gt, t, y_h, Gobs, drop_mask=None)
                pred_u = self._pred_theta(Gt, t, None, Gobs, drop_mask=None)
                pred = pred_u + self.guidance_scale * (pred_c - pred_u)
                eps_pred = self._to_eps(Gt, t, pred)
                Gt = self._reverse_inference_update(Gt, t, eps_pred, prev_t)
                Gt = self._mask_observed(Gt, Gobs, Mobs)
            H = self._last_hidden
            X_obs_last = self._last_xobs
            h = self._graph_readout(H, X_obs_last)
            h = self._fuse_text_embedding(h, batch)
            outs.append(self._adapt_potential(h))
        Hc = torch.stack(outs, dim=1)  # [B, C, H]
        return Hc

    def _q_sample(self, G0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        _assert_shape(G0, (-1, -1))
        _assert_shape(t, (-1,))
        if noise is None:
            noise = torch.randn_like(G0)
        alpha_bar = self.alphas_cumprod[(t - 1).to(torch.long)].view(-1, 1)
        Gt = (alpha_bar.sqrt() * G0) + ((1.0 - alpha_bar).sqrt() * noise)
        return Gt, noise

    def _objective_loss(self, batch, update_prototypes: bool) -> Tensor:
        """Computes the diffusion pretraining objective used by stage one."""
        device = next(self.parameters()).device
        if self.multiwindow_enabled:
            obs_override = int(
                torch.randint(
                    low=int(self.cutoff_min),
                    high=int(self.cutoff_max) + 1,
                    size=(1,),
                ).item()
            )
            G0, Gobs, Mobs, y = self._pack_batch_G(batch, obs_minutes_override=obs_override)
        else:
            G0, Gobs, Mobs, y = self._pack_batch_G(batch)
        G0 = G0.to(device)
        Gobs = Gobs.to(device)
        y = y.to(device)
        B = int(G0.size(0))
        t = torch.randint(low=1, high=self.T + 1, size=(B,), device=device)
        Gt, noise = self._q_sample(G0, t)
        if self.guidance_p > 0.0:
            drop_mask = (torch.rand((B,), device=device) < float(self.guidance_p))
        else:
            drop_mask = None
        pred = self._pred_theta(Gt, t, y, Gobs, drop_mask=drop_mask)
        target = self._v_target(G0, t, noise) if self.parameterization == "v" else noise
        _assert_shape(target, (-1, -1))
        _assert_shape(pred, (-1, -1))
        assert int(target.shape[1]) == int(pred.shape[1])
        loss = ((target - pred) ** 2).mean()

        H = self._last_hidden
        X_obs_last = self._last_xobs
        assert H is not None and X_obs_last is not None
        h_graph = self._graph_readout(H, X_obs_last)
        h_graph = self._fuse_text_embedding(h_graph, batch)
        proto_loss = self._compute_prototype_contrastive_loss(h_graph, y)
        if float(self.lambda_proto) > 0.0:
            loss = loss + float(self.lambda_proto) * proto_loss
        if float(self.lambda_supcon_pair) > 0.0:
            supcon_loss = self._compute_supcon_pair_loss(h_graph, y)
            loss = loss + float(self.lambda_supcon_pair) * supcon_loss
        if float(self.lambda_proto_orth) > 0.0:
            ortho_loss = self._compute_proto_orth_loss()
            loss = loss + float(self.lambda_proto_orth) * ortho_loss
        if self.forecast_enabled and float(self.lambda_forecast) > 0.0 and not isinstance(self.forecast_head, nn.Identity):
            targets_forecast = self._forecast_targets(batch).to(device)
            preds_forecast = self.forecast_head(h_graph)
            loss_forecast = F.mse_loss(preds_forecast, targets_forecast)
            loss = loss + float(self.lambda_forecast) * loss_forecast
        if update_prototypes:
            self.update_prototypes(h_graph.detach(), y.detach())

        scores = self._last_scores
        assert scores is not None
        N = self.max_nodes
        D = self.in_channels
        A_obs_list: List[Tensor] = []
        for bidx in range(B):
            A_o, _X_o = _unflatten_AX(Gobs[bidx], N, D)
            A_obs_list.append(A_o)
        A_obs_batch = torch.stack(A_obs_list, dim=0)
        topo_bce, topo_kl = self._topology_regularizers(scores, A_obs_batch, X_obs_last)
        if float(self.lambda_topo_bce) > 0.0:
            loss = loss + float(self.lambda_topo_bce) * topo_bce
        if float(self.lambda_deg_kl) > 0.0:
            loss = loss + float(self.lambda_deg_kl) * topo_kl
        return loss

    def training_step(self, batch) -> Tuple[Tensor, float]:
        """Returns (loss, aux_acc) for a training step using diffusion objective."""
        return self._objective_loss(batch, update_prototypes=True), 0.0

    def validation_step(self, batch) -> Tensor:
        """Returns the stage-one validation objective without mutating prototypes."""
        with self.objective_eval_target_access():
            return self._objective_loss(batch, update_prototypes=False)

    def _posterior_mean_variance(self, Gt: Tensor, t: Tensor, eps_pred: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = self.betas[(t - 1).to(torch.long)].view(-1, 1)
        alpha_t = self.alphas[(t - 1).to(torch.long)].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[(t - 1).to(torch.long)].view(-1, 1)
        alpha_bar_prev = torch.ones_like(alpha_bar_t)
        prev_mask = t > 1
        if prev_mask.any():
            alpha_bar_prev[prev_mask] = self.alphas_cumprod[(t[prev_mask] - 2).to(torch.long)].view(-1, 1)
        coef1 = (1.0 / alpha_t.sqrt())
        coef2 = ((1.0 - alpha_t) / (1.0 - alpha_bar_t).sqrt())
        mean = coef1 * (Gt - coef2 * eps_pred)
        var = (beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp_min(self._eps)).clamp(min=0.0)
        var = torch.where((t > 1).view(-1, 1), var, torch.zeros_like(var))
        return mean, var

    def _alpha_bar_at(self, t: Tensor) -> Tensor:
        alpha_bar = torch.ones((int(t.size(0)), 1), dtype=self.alphas_cumprod.dtype, device=t.device)
        valid = t > 0
        if valid.any():
            alpha_bar[valid] = self.alphas_cumprod[(t[valid] - 1).to(torch.long)].view(-1, 1)
        return alpha_bar

    def _reverse_step(self, Gt: Tensor, t: Tensor, eps_pred: Tensor, prev_t: Tensor) -> Tensor:
        alpha_bar_t = self._alpha_bar_at(t)
        alpha_bar_prev = self._alpha_bar_at(prev_t)
        x0_pred = (Gt - (1.0 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt().clamp_min(self._eps)
        eta = max(0.0, float(self.sampling_eta))
        sigma = torch.zeros_like(alpha_bar_t)
        if eta > 0.0:
            ratio = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp_min(self._eps)).clamp(min=0.0)
            step_gap = (1.0 - (alpha_bar_t / alpha_bar_prev.clamp_min(self._eps))).clamp(min=0.0)
            sigma = eta * torch.sqrt(ratio * step_gap)
        direction = torch.sqrt((1.0 - alpha_bar_prev - sigma.pow(2)).clamp(min=0.0)) * eps_pred
        G_prev = alpha_bar_prev.sqrt() * x0_pred + direction
        if eta > 0.0:
            noise = torch.randn_like(Gt)
            nonzero_mask = (prev_t > 0).to(Gt.dtype).view(-1, 1)
            G_prev = G_prev + nonzero_mask * sigma * noise
        return G_prev

    def _reverse_inference_update(self, Gt: Tensor, t: Tensor, eps_pred: Tensor, prev_t: Tensor) -> Tensor:
        if float(self.sampling_eta) > 0.0:
            return self._reverse_step(Gt, t, eps_pred, prev_t)
        mean, _var = self._posterior_mean_variance(Gt, t, eps_pred)
        return mean

    def _get_c1_c2(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (c1, c2) where x_t = c1*x0 + c2*eps and v = c1*eps - c2*x0."""
        alpha_bar_t = self.alphas_cumprod[(t - 1).to(torch.long)].view(-1, 1)
        c1 = alpha_bar_t.sqrt()
        c2 = (1.0 - alpha_bar_t).sqrt()
        return c1, c2

    def _v_target(self, G0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Computes v target for v-parameterization training."""
        c1, c2 = self._get_c1_c2(t)
        return c1 * noise - c2 * G0

    def _to_eps(self, Gt: Tensor, t: Tensor, pred: Tensor) -> Tensor:
        """Converts predicted parameterization to epsilon for posterior update."""
        if self.parameterization == "eps":
            return pred
        c1, c2 = self._get_c1_c2(t)
        return c2 * Gt + c1 * pred

    def _mask_observed(self, G: Tensor, Gobs: Tensor, Mobs: Tensor) -> Tensor:
        """Overwrites observed entries in G with values from Gobs using explicit mask."""
        _assert_shape(G, (-1, -1))
        _assert_shape(Gobs, (-1, -1))
        _assert_shape(Mobs, (-1, -1))
        mask = Mobs.to(G.dtype)
        return G * (1.0 - mask) + Gobs * mask

    def forward(self, batch) -> Tensor:
        """Computes ELBO-like scores per class and returns logits [B, C]."""
        device = next(self.parameters()).device
        G0, Gobs, Mobs, _y = self._pack_batch_G(batch)
        Gobs = Gobs.to(device)
        Mobs = Mobs.to(device)
        B = int(G0.size(0))
        scores: List[Tensor] = []
        for cls in range(self.num_classes):
            y_h = torch.full((B,), int(cls), dtype=torch.long, device=device)
            # Initialize from standard normal
            L = int(Gobs.size(1))
            Gt = torch.randn((B, L), device=device)
            total_score = torch.zeros((B,), device=device)
            # Strided reverse steps
            stride = max(1, getattr(self, "t_stride", self.T // 20 if self.T >= 20 else 1))
            for step in range(self.T, 0, -stride):
                t = torch.full((B,), int(step), dtype=torch.long, device=device)
                pred_c = self._pred_theta(Gt, t, y_h, Gobs, drop_mask=None)
                pred_u = self._pred_theta(Gt, t, None, Gobs, drop_mask=None)
                pred = pred_u + self.guidance_scale * (pred_c - pred_u)
                eps_pred = self._to_eps(Gt, t, pred)
                mean, _var = self._posterior_mean_variance(Gt, t, eps_pred)
                Gt = mean
                # Data consistency on observed entries
                Gt = self._mask_observed(Gt, Gobs, Mobs)
                # Accumulate negative reconstruction error only on observed entries
                obs_err = (((Gobs - Gt) ** 2) * Mobs).sum(dim=1)
                total_score = total_score - obs_err
            scores.append(total_score)
        logits = torch.stack(scores, dim=1)
        return logits
