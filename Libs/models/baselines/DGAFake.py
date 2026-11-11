from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _hash_user_ids(uids: Sequence[str], vocab_size: int) -> Tensor:
    """Maps arbitrary user identifiers to hashed embedding indices."""
    assert vocab_size > 0
    hashed: List[int] = []
    for uid in uids:
        h = hashlib.sha1(uid.encode("utf-8")).digest()
        val = int.from_bytes(h[:8], byteorder="big", signed=False)
        hashed.append(val % vocab_size)
    return torch.tensor(hashed, dtype=torch.long)


def _sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> Tensor:
    """Builds sinusoidal position encodings of shape [length, dim]."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


@dataclass
class DGAFakeConfig:
    """Container for DGA-Fake hyperparameters."""

    hidden: int
    text_dim: int
    content_dropout: float

    seq_layers: int
    seq_heads: int
    seq_dropout: float
    max_seq_len: int

    T: int
    beta_start: float
    beta_end: float
    cond_drop: float
    guidance_scale: float
    lambda_diff: float
    step_embed_type: str

    generator_loops: int
    max_gen_per_graph: int
    topk: int
    max_chain_len: int

    num_users: int

    hypergat_hidden: int
    hypergat_layers: int
    gat_hidden: int
    gat_layers: int
    detector_dropout: float
    cls_hidden: int

    grad_clip: float
    use_amp: bool

    @staticmethod
    def from_dict(cfg: Dict) -> "DGAFakeConfig":
        """Instantiates configuration from YAML; accepts outer{'model':{...}} or inner dict."""
        m = cfg["model"] if "model" in cfg else cfg
        return DGAFakeConfig(
            hidden=int(m["hidden"]),
            text_dim=int(m["text_dim"]),
            content_dropout=float(m.get("content_dropout", 0.1)),
            seq_layers=int(m.get("seq_layers", 2)),
            seq_heads=int(m.get("seq_heads", 4)),
            seq_dropout=float(m.get("seq_dropout", 0.1)),
            max_seq_len=int(m.get("max_seq_len", 64)),
            T=int(m.get("T", 200)),
            beta_start=float(m.get("beta_start", 1e-4)),
            beta_end=float(m.get("beta_end", 2e-2)),
            cond_drop=float(m.get("cond_drop", 0.1)),
            guidance_scale=float(m.get("guidance_scale", 3.0)),
            lambda_diff=float(m.get("lambda_diff", 0.1)),
            step_embed_type=str(m.get("step_embed_type", "learned")),
            generator_loops=int(m.get("generator_loops", 4)),
            max_gen_per_graph=int(m.get("max_gen_per_graph", 200)),
            topk=int(m.get("topk", 5)),
            max_chain_len=int(m.get("max_chain_len", 6)),
            num_users=int(m.get("num_users", 500000)),
            hypergat_hidden=int(m.get("hypergat_hidden", 128)),
            hypergat_layers=int(m.get("hypergat_layers", 1)),
            gat_hidden=int(m.get("gat_hidden", 128)),
            gat_layers=int(m.get("gat_layers", 1)),
            detector_dropout=float(m.get("detector_dropout", 0.2)),
            cls_hidden=int(m.get("cls_hidden", 256)),
            grad_clip=float(m.get("grad_clip", 1.0)),
            use_amp=bool(m.get("use_amp", False)),
        )


class ContentEncoder(nn.Module):
    """Projects precomputed sentence embeddings to model dimension."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, text_emb: Tensor) -> Tensor:
        assert text_emb.dim() == 2, "text_emb must be [B, D]"
        h = self.proj(text_emb)
        return self.dropout(h)


class SequenceEncoder(nn.Module):
    """Encodes user history prefixes with positional encoding."""

    def __init__(self, dim: int, layers: int, heads: int, dropout: float, max_len: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.max_len = int(max_len)

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        assert tokens.dim() == 3, "tokens must be [B, L, D]"
        bsz, seqlen, dim = tokens.shape
        device = tokens.device
        pe = _sinusoidal_position_encoding(seqlen, dim, device=device)
        h = tokens + pe.unsqueeze(0)
        out = self.encoder(h, src_key_padding_mask=padding_mask)
        last_index = (padding_mask.logical_not().sum(dim=1) - 1).clamp(min=0)
        batch_idx = torch.arange(bsz, device=device)
        gathered = out[batch_idx, last_index]
        return gathered


class DualConditionalDenoiser(nn.Module):
    """Implements dual-conditional denoising with self and cross attention."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim * 2, dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.cross_hist = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.cross_content = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, u_t: Tensor, step_emb: Tensor, c_hist: Tensor, r_content: Tensor) -> Tensor:
        assert u_t.shape == step_emb.shape == c_hist.shape == r_content.shape
        x = torch.cat([u_t, step_emb], dim=-1)
        x = self.in_proj(x).unsqueeze(1)

        attn_self, _ = self.self_attn(x, x, x)
        x1 = self.norm1(attn_self + x)

        hist = c_hist.unsqueeze(1)
        attn_hist, _ = self.cross_hist(x1, hist, hist)
        x2 = self.norm2(attn_hist + x1)

        content = r_content.unsqueeze(1)
        attn_content, _ = self.cross_content(x2, content, content)
        x3 = self.norm3(attn_content + x2)

        ff_out = self.ff(x3)
        out = self.norm4(ff_out + x3)
        return out.squeeze(1)


class DiffusionScheduler(nn.Module):
    """Precomputes diffusion coefficients and step embeddings."""

    def __init__(self, T: int, beta_start: float, beta_end: float, dim: int, embed_type: str = "learned") -> None:
        super().__init__()
        self.T = int(T)
        betas = torch.linspace(beta_start, beta_end, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        if embed_type == "learned":
            self.step_embed = nn.Embedding(self.T, dim)
        else:
            pe = _sinusoidal_position_encoding(self.T, dim, device=torch.device("cpu"))
            self.register_buffer("step_embed_buffer", pe)
            self.step_embed = None
        self.embed_type = embed_type

    def step_embedding(self, t: Tensor) -> Tensor:
        assert t.dim() == 1, "t must be [B]"
        if self.step_embed is not None:
            return self.step_embed(t - 1)
        emb = self.step_embed_buffer.to(t.device)
        return emb[t - 1]


class HyperGATLayer(nn.Module):
    """Two-phase hypergraph attention (node->edge, edge->node)."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.node_lin = nn.Linear(dim, dim)
        self.edge_lin = nn.Linear(dim, dim)
        self.attn_node = nn.Linear(dim, 1, bias=False)
        self.attn_edge = nn.Linear(dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, node_feats: Tensor, hyperedges: List[List[int]]) -> Tensor:
        assert node_feats.dim() == 2
        if len(hyperedges) == 0:
            return node_feats

        device = node_feats.device
        node_proj = self.node_lin(node_feats)

        edge_feats: List[Tensor] = []
        for edge_nodes in hyperedges:
            idx = torch.tensor(edge_nodes, device=device, dtype=torch.long)
            tokens = node_proj.index_select(0, idx)
            scores = self.attn_node(tokens)
            weights = torch.softmax(scores, dim=0)
            aggregated = torch.sum(weights * tokens, dim=0)
            edge_feats.append(self.act(aggregated))
        edges_tensor = torch.stack(edge_feats, dim=0)

        updated_nodes = []
        for node_idx in range(node_feats.size(0)):
            carriers: List[int] = []
            for e_idx, e_nodes in enumerate(hyperedges):
                if node_idx in e_nodes:
                    carriers.append(e_idx)
            if len(carriers) == 0:
                updated_nodes.append(node_feats[node_idx])
                continue
            e_tensor = edges_tensor.index_select(0, torch.tensor(carriers, device=device, dtype=torch.long))
            tgt = node_feats[node_idx].unsqueeze(0).expand_as(e_tensor)
            joint = torch.cat([self.edge_lin(e_tensor), tgt], dim=-1)
            scores = self.attn_edge(joint)
            weights = torch.softmax(scores, dim=0)
            msg = torch.sum(weights * e_tensor, dim=0)
            updated_nodes.append(self.act(msg))
        out = torch.stack(updated_nodes, dim=0)
        return self.dropout(out)


class GraphAttentionLayer(nn.Module):
    """Single-head GAT layer for directed edges."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Linear(dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        assert edge_index.dim() == 2 and edge_index.size(0) == 2
        if edge_index.size(1) == 0:
            return x
        h = self.lin(x)
        src, dst = edge_index[0], edge_index[1]
        src_h = h.index_select(0, src)
        dst_h = h.index_select(0, dst)
        cat = torch.cat([src_h, dst_h], dim=-1)
        scores = F.leaky_relu(self.attn(cat).squeeze(-1))
        scores_exp = torch.exp(scores - scores.max())
        norm_denom = torch.zeros(h.size(0), device=h.device, dtype=scores_exp.dtype)
        norm_denom.index_add_(0, dst, scores_exp)
        norm = scores_exp / norm_denom.index_select(0, dst).clamp(min=1e-6)
        agg = torch.zeros(h.size(0), h.size(1), device=h.device, dtype=src_h.dtype)
        agg.index_add_(0, dst, norm.to(dtype=src_h.dtype).unsqueeze(-1) * src_h)
        out = self.act(agg)
        return self.dropout(out)


class DGAFakeDetector(nn.Module):
    """Hyper-GAT + GAT encoder with content fusion for classification."""

    def __init__(
        self,
        dim: int,
        hyper_layers: int,
        gat_layers: int,
        hyper_hidden: int,
        gat_hidden: int,
        dropout: float,
        cls_hidden: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.hyper_layers = nn.ModuleList(
            [HyperGATLayer(dim if i == 0 else hyper_hidden, dropout=dropout) for i in range(hyper_layers)]
        )
        self.hyper_proj = (
            nn.Identity()
            if hyper_layers == 0
            else nn.Linear(dim if hyper_layers == 0 else hyper_hidden, dim)
        )

        self.gat_layers = nn.ModuleList(
            [GraphAttentionLayer(dim if i == 0 else gat_hidden, dropout=dropout) for i in range(gat_layers)]
        )
        self.gat_proj = (
            nn.Identity()
            if gat_layers == 0
            else nn.Linear(dim if gat_layers == 0 else gat_hidden, dim)
        )

        self.dropout = nn.Dropout(p=dropout)
        self.cls = nn.Sequential(
            nn.Linear(dim * 3, cls_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, node_feats: Tensor, hyperedges: List[List[int]], edge_index: Tensor, content: Tensor) -> Tensor:
        h_hyper = node_feats
        for layer in self.hyper_layers:
            h_hyper = layer(h_hyper, hyperedges)
        h_hyper = self.hyper_proj(h_hyper)

        h_graph = node_feats
        for layer in self.gat_layers:
            h_graph = layer(h_graph, edge_index)
        h_graph = self.gat_proj(h_graph)

        u_H = h_hyper.mean(dim=0, keepdim=True)
        u_G = h_graph.mean(dim=0, keepdim=True)
        fused = torch.cat([u_H, u_G, content], dim=-1)
        fused = self.dropout(fused)
        return self.cls(fused)


class DGAFakeModel(nn.Module):
    """End-to-end DGA-Fake with diffusion generator and graph detector."""

    def __init__(self, cfg: DGAFakeConfig, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.content_encoder = ContentEncoder(cfg.text_dim, cfg.hidden, cfg.content_dropout)
        self.start_proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.user_embedding = nn.Embedding(cfg.num_users, cfg.hidden)
        self.seq_encoder = SequenceEncoder(cfg.hidden, cfg.seq_layers, cfg.seq_heads, cfg.seq_dropout, cfg.max_seq_len)
        self.scheduler = DiffusionScheduler(cfg.T, cfg.beta_start, cfg.beta_end, cfg.hidden, cfg.step_embed_type)
        self.denoiser = DualConditionalDenoiser(cfg.hidden, cfg.seq_heads, cfg.seq_dropout)
        self.detector = DGAFakeDetector(
            dim=cfg.hidden,
            hyper_layers=cfg.hypergat_layers,
            gat_layers=cfg.gat_layers,
            hyper_hidden=cfg.hypergat_hidden,
            gat_hidden=cfg.gat_hidden,
            dropout=cfg.detector_dropout,
            cls_hidden=cfg.cls_hidden,
            num_classes=num_classes,
        )
        self.cls_loss = nn.CrossEntropyLoss()

    def _flatten_node_uid(self, batch) -> List[str]:
        """Flattens batched node_uid attribute into a per-node list of strings."""
        raw = batch.node_uid
        if isinstance(raw, list):
            if len(raw) == batch.ptr.numel() - 1 and isinstance(raw[0], list):
                flat: List[str] = []
                for sub in raw:
                    flat.extend([str(v) for v in sub])
                return flat
            return [str(v) for v in raw]
        return [str(v) for v in list(raw)]

    def _extract_sequences(self, batch) -> Tuple[List[List[int]], List[int]]:
        """Builds root-to-node user index sequences for each graph in the batch."""
        num_graphs = int(batch.ptr.numel()) - 1
        flat_uids = self._flatten_node_uid(batch)
        uid_idx = _hash_user_ids(flat_uids, self.cfg.num_users).to(batch.edge_index.device)

        sequences: List[List[int]] = []
        graph_ids: List[int] = []

        edge_index = batch.edge_index

        for g in range(num_graphs):
            start = int(batch.ptr[g].item())
            end = int(batch.ptr[g + 1].item())
            mask = (
                (edge_index[0] >= start)
                & (edge_index[0] < end)
                & (edge_index[1] >= start)
                & (edge_index[1] < end)
            )
            edges_local = edge_index[:, mask] - start
            num_nodes = end - start
            parent = torch.full((num_nodes,), -1, device=edge_index.device, dtype=torch.long)
            for e_idx in range(edges_local.size(1)):
                u = int(edges_local[0, e_idx].item())
                v = int(edges_local[1, e_idx].item())
                if parent[v] == -1:
                    parent[v] = u
            root_rel = int(batch.root_index[g].item())
            for node in range(num_nodes):
                if node == root_rel:
                    continue
                path: List[int] = []
                cur = node
                chain_len = 0
                while cur >= 0 and chain_len < self.cfg.max_seq_len:
                    path.append(cur)
                    if cur == root_rel or parent[cur] == -1:
                        break
                    cur = int(parent[cur].item())
                    chain_len += 1
                path = list(reversed(path))
                token_ids = [int(uid_idx[start + p].item()) for p in path]
                sequences.append(token_ids)
                graph_ids.append(g)
        return sequences, graph_ids

    def _pad_and_encode_history(self, embeddings: List[Tensor], device: torch.device) -> Tuple[Tensor, Tensor]:
        """Pads variable-length prefix embeddings and encodes them via Transformer."""
        max_len = max(e.size(0) for e in embeddings)
        bsz = len(embeddings)
        dim = embeddings[0].size(-1)
        tokens = torch.zeros((bsz, max_len, dim), device=device)
        mask = torch.ones((bsz, max_len), device=device, dtype=torch.bool)
        for i, emb in enumerate(embeddings):
            length = emb.size(0)
            tokens[i, :length] = emb
            mask[i, :length] = False
        summary = self.seq_encoder(tokens, mask)
        return summary, mask

    def _sample_diffusion_loss(
        self, sequences: List[List[int]], graph_ids: List[int], content: Tensor, device: torch.device
    ) -> Tensor:
        """Computes diffusion noise-prediction loss over sampled prefixes."""
        if len(sequences) == 0:
            return torch.tensor(0.0, device=device)

        start_embeddings = self.start_proj(content)
        emb_list: List[Tensor] = []
        target_list: List[Tensor] = []
        cond_r_list: List[Tensor] = []

        for seq, gid in zip(sequences, graph_ids):
            if len(seq) < 2:
                continue
            pos = torch.randint(1, len(seq), (1,), device=device).item()
            prefix_ids = seq[:pos]
            target_id = seq[pos]

            prefix_emb = self.user_embedding(torch.tensor(prefix_ids, device=device))
            start_token = start_embeddings[gid].unsqueeze(0)
            tokens = torch.cat([start_token, prefix_emb], dim=0)
            emb_list.append(tokens)
            target_list.append(self.user_embedding(torch.tensor([target_id], device=device)).squeeze(0))
            cond_r_list.append(content[gid])

        if len(target_list) == 0:
            return torch.tensor(0.0, device=device)

        c_summary, _ = self._pad_and_encode_history(emb_list, device)
        r_stack = torch.stack(cond_r_list, dim=0)
        u0 = torch.stack(target_list, dim=0)

        t = torch.randint(1, self.cfg.T + 1, (u0.size(0),), device=device)
        eps = torch.randn_like(u0)
        sqrt_ab = self.scheduler.sqrt_alpha_bar.index_select(0, t - 1)
        sqrt_omb = self.scheduler.sqrt_one_minus_alpha_bar.index_select(0, t - 1)
        u_t = sqrt_ab.unsqueeze(1) * u0 + sqrt_omb.unsqueeze(1) * eps

        drop_mask = torch.rand(u0.size(0), device=device) < self.cfg.cond_drop
        c_cond = torch.where(drop_mask.unsqueeze(1), torch.zeros_like(c_summary), c_summary)
        r_cond = torch.where(drop_mask.unsqueeze(1), torch.zeros_like(r_stack), r_stack)
        step_emb = self.scheduler.step_embedding(t)

        eps_pred = self.denoiser(u_t, step_emb, c_cond, r_cond)
        loss = F.mse_loss(eps_pred, eps, reduction="mean")
        return loss

    def _generate_candidates(
        self,
        sequences: List[List[int]],
        graph_ids: List[int],
        content: Tensor,
        device: torch.device,
    ) -> Tuple[List[List[int]], List[int]]:
        """Generates new user sequences via classifier-free guided diffusion."""
        if len(sequences) == 0:
            return sequences, graph_ids

        start_embeddings = self.start_proj(content)
        generated_seq: List[List[int]] = list(sequences)
        generated_gid: List[int] = list(graph_ids)
        loops = min(self.cfg.generator_loops, self.cfg.max_gen_per_graph)

        for _ in range(loops):
            idx = torch.randint(0, len(sequences), (1,), device=device).item()
            base_seq = sequences[idx]
            gid = graph_ids[idx]
            if len(base_seq) >= self.cfg.max_chain_len:
                continue
            prefix_ids = base_seq
            prefix_emb = self.user_embedding(torch.tensor(prefix_ids, device=device))
            start_token = start_embeddings[gid].unsqueeze(0)
            tokens = torch.cat([start_token, prefix_emb], dim=0)
            tokens_pad, _ = self._pad_and_encode_history([tokens], device)
            c_hist = tokens_pad[0]
            r_content = content[gid]

            u_t = torch.randn((1, self.cfg.hidden), device=device)
            t_steps = torch.arange(self.cfg.T, 0, -1, device=device)
            for t_val in t_steps:
                t_batch = torch.tensor([t_val], device=device)
                step_emb = self.scheduler.step_embedding(t_batch)
                eps_cond = self.denoiser(u_t, step_emb, c_hist.unsqueeze(0), r_content.unsqueeze(0))
                eps_uncond = self.denoiser(
                    u_t, step_emb, torch.zeros_like(c_hist).unsqueeze(0), torch.zeros_like(r_content).unsqueeze(0)
                )
                eps_hat = (1.0 + self.cfg.guidance_scale) * eps_cond - self.cfg.guidance_scale * eps_uncond

                alpha = self.scheduler.alphas[t_val - 1]
                alpha_bar = self.scheduler.alpha_bar[t_val - 1]
                sqrt_alpha = torch.sqrt(alpha)
                beta = self.scheduler.betas[t_val - 1]
                coef_eps = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
                mean = (1.0 / sqrt_alpha) * (u_t - coef_eps * eps_hat)
                if t_val > 1:
                    noise = torch.randn_like(u_t)
                    u_t = mean + torch.sqrt(beta) * noise
                else:
                    u_t = mean

            scores = torch.matmul(self.user_embedding.weight, u_t[0])
            topk = torch.topk(scores, k=min(self.cfg.topk, scores.numel()))
            for cand in topk.indices.tolist():
                new_seq = base_seq + [cand]
                generated_seq.append(new_seq)
                generated_gid.append(gid)
                if len(generated_seq) >= self.cfg.max_gen_per_graph:
                    return generated_seq, generated_gid
        return generated_seq, generated_gid

    def _build_graph_inputs(
        self, sequences: List[List[int]], graph_ids: List[int], content: Tensor
    ) -> List[Tuple[Tensor, List[List[int]], Tensor, Tensor, int]]:
        """Constructs per-graph node features, hyperedges, and edges for detection."""
        device = content.device
        grouped: Dict[int, List[List[int]]] = {}
        for seq, gid in zip(sequences, graph_ids):
            grouped.setdefault(gid, []).append(seq)

        outputs: List[Tuple[Tensor, List[List[int]], Tensor, Tensor, int]] = []
        for gid, seqs in grouped.items():
            content_node = content[gid].unsqueeze(0)
            node_pool: Dict[int, int] = {}
            node_feats = [content_node]
            for seq in seqs:
                for uid in seq:
                    if uid not in node_pool:
                        node_pool[uid] = len(node_feats)
                        node_feats.append(self.user_embedding(torch.tensor([uid], device=device)))
            node_mat = torch.cat(node_feats, dim=0)
            hyperedges: List[List[int]] = []
            edge_src: List[int] = []
            edge_dst: List[int] = []
            for seq in seqs:
                mapped = [0] + [node_pool[uid] for uid in seq]
                hyperedges.append(mapped)
                for i in range(len(mapped) - 1):
                    edge_src.append(mapped[i])
                    edge_dst.append(mapped[i + 1])
            if len(edge_src) == 0:
                edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            else:
                edge_index = torch.stack(
                    [
                        torch.tensor(edge_src, device=device, dtype=torch.long),
                        torch.tensor(edge_dst, device=device, dtype=torch.long),
                    ],
                    dim=0,
                )
            outputs.append((node_mat, hyperedges, edge_index, content[gid].unsqueeze(0), gid))
        return outputs

    def _forward_with_content(self, batch, content: Tensor) -> Tensor:
        """Runs detection using provided content embeddings, including generation."""
        device = batch.edge_index.device
        sequences, graph_ids = self._extract_sequences(batch)
        if len(sequences) == 0:
            empty_edge = torch.zeros((2, 0), device=device, dtype=torch.long)
            logits_list: List[Tensor] = []
            for g in range(content.size(0)):
                logits_list.append(
                    self.detector(
                        node_feats=content[g].unsqueeze(0),
                        hyperedges=[],
                        edge_index=empty_edge,
                        content=content[g].unsqueeze(0),
                    )
                )
            return torch.cat(logits_list, dim=0)

        sequences_all, graph_ids_all = self._generate_candidates(sequences, graph_ids, content, batch.edge_index.device)
        graph_inputs = self._build_graph_inputs(sequences_all, graph_ids_all, content)
        logits_list: List[Tensor] = []
        covered = set()
        for node_feats, hyperedges, edge_index, content_vec, gid in graph_inputs:
            covered.add(gid)
            logits_list.append(self.detector(node_feats=node_feats, hyperedges=hyperedges, edge_index=edge_index, content=content_vec))
        if len(covered) < content.size(0):
            empty_edge = torch.zeros((2, 0), device=device, dtype=torch.long)
            for gid in range(content.size(0)):
                if gid in covered:
                    continue
                logits_list.append(
                    self.detector(
                        node_feats=content[gid].unsqueeze(0),
                        hyperedges=[],
                        edge_index=empty_edge,
                        content=content[gid].unsqueeze(0),
                    )
                )
        if len(logits_list) == 0:
            return torch.zeros((content.size(0), self.num_classes), device=device)
        return torch.cat(logits_list, dim=0)

    def forward(self, batch) -> Tensor:
        assert hasattr(batch, "text_emb"), "text_emb is required for DGA-Fake"
        device = batch.edge_index.device
        content = self.content_encoder(batch.text_emb.to(device))
        return self._forward_with_content(batch, content)

    def training_step(self, batch) -> Tuple[Tensor, Dict[str, float]]:
        """Custom training step combining classification and diffusion objectives."""
        device = batch.edge_index.device
        content = self.content_encoder(batch.text_emb.to(device))
        logits = self._forward_with_content(batch, content)
        cls_loss = self.cls_loss(logits, batch.y.view(-1))

        sequences, graph_ids = self._extract_sequences(batch)
        diff_loss = self._sample_diffusion_loss(sequences, graph_ids, content, device)

        total_loss = cls_loss + self.cfg.lambda_diff * diff_loss
        metrics = {"cls_loss": float(cls_loss.detach().cpu()), "diff_loss": float(diff_loss.detach().cpu())}
        return total_loss, metrics
