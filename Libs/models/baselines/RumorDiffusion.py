import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda.amp import autocast

from Libs.models.baselines.ContrastiveObjectives import supervised_contrastive_loss
from Libs.models.modules.root_aware_cascade import RootAwareCascadeEncoder


def _resolve_global_root_index(batch) -> Tensor:
    """Normalizes PyG root indices to global node indices for a graph batch."""
    if not hasattr(batch, "x") or not hasattr(batch, "root_index"):
        raise ValueError("Root-aware branches require batch.x and batch.root_index.")
    root_index = batch.root_index.view(-1).to(device=batch.x.device, dtype=torch.long)
    if hasattr(batch, "ptr"):
        ptr = batch.ptr.to(device=batch.x.device, dtype=torch.long)
        graph_count = int(ptr.numel()) - 1
        if int(root_index.numel()) != graph_count:
            raise ValueError(
                f"root_index length {int(root_index.numel())} does not match graph count {graph_count}."
            )
        global_valid = bool(((root_index >= ptr[:-1]) & (root_index < ptr[1:])).all().item())
        local_index = root_index + ptr[:-1]
        local_valid = bool(((local_index >= ptr[:-1]) & (local_index < ptr[1:])).all().item())
        if global_valid:
            root_global = root_index
        elif local_valid:
            root_global = local_index
        else:
            raise ValueError("root_index is neither a valid global nor local batch index.")
    else:
        root_global = root_index
    if int(root_global.numel()) == 0:
        raise ValueError("Root-aware branches received an empty root_index tensor.")
    if int(root_global.min().item()) < 0 or int(root_global.max().item()) >= int(batch.x.size(0)):
        raise ValueError("root_index is outside the node-feature tensor.")
    return root_global


class ArbiterMLP(nn.Module):
    """Two-branch arbiter MLP operating on concatenated potentials.

    Input shape: [B, 2*H] where H is hidden_channels of CTDM backbone.
    Output: [B, C] logits.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PotentialDiscriminator(nn.Module):
    """MLP discriminator operating on single-class potential vectors.

    Input shape: [B, H] where H is hidden_channels of CTDM backbone.
    Output: [B, C] logits.
    """

    def __init__(self, hidden_channels: int, num_classes: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PrototypeResidualBranch(nn.Module):
    """Class-prototype residual logits over arbiter representations."""

    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        temperature: float,
        momentum: float,
        logit_weight: float,
        min_count: int,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.logit_weight = float(logit_weight)
        self.min_count = int(min_count)
        self.register_buffer(
            "prototypes",
            torch.zeros((self.num_classes, self.hidden_channels), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "prototype_counts",
            torch.zeros((self.num_classes,), dtype=torch.float32),
            persistent=True,
        )

    @torch.no_grad()
    def update(self, features: Tensor, targets: Tensor) -> None:
        assert features.dim() == 2 and int(features.size(1)) == self.hidden_channels
        assert targets.dim() == 1 and int(targets.size(0)) == int(features.size(0))
        for class_id in range(self.num_classes):
            mask = targets == int(class_id)
            if mask.any():
                class_feature = features[mask].mean(dim=0)
                previous_count = float(self.prototype_counts[class_id].item())
                if previous_count <= 0.0:
                    self.prototypes[class_id].copy_(class_feature)
                else:
                    self.prototypes[class_id].mul_(self.momentum).add_((1.0 - self.momentum) * class_feature)
                self.prototype_counts[class_id].add_(float(mask.sum().item()))

    def forward(self, features: Tensor) -> Tensor:
        assert features.dim() == 2 and int(features.size(1)) == self.hidden_channels
        feature_norm = F.normalize(features, dim=-1)
        proto_norm = F.normalize(self.prototypes.to(device=features.device, dtype=features.dtype), dim=-1)
        logits = torch.matmul(feature_norm, proto_norm.t())
        valid = (self.prototype_counts.to(device=features.device) >= float(self.min_count)).to(features.dtype)
        logits = logits * valid.view(1, -1)
        return float(self.logit_weight) * logits / max(float(self.temperature), 1e-6)


class SemanticResidualBranch(nn.Module):
    """Root/text semantic residual logits for observed-prefix classification."""

    def __init__(
        self,
        root_dim: int,
        text_dim: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float,
        logit_weight: float,
        use_root: bool,
        use_text: bool,
    ) -> None:
        super().__init__()
        self.root_dim = int(root_dim)
        self.text_dim = int(text_dim)
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)
        self.logit_weight = float(logit_weight)
        self.use_root = bool(use_root)
        self.use_text = bool(use_text)
        if not self.use_root and not self.use_text:
            raise ValueError("Semantic residual branch requires root, text, or both inputs.")
        if self.use_root and self.root_dim <= 0:
            raise ValueError("Semantic residual root input is enabled but root_dim is not positive.")
        if self.use_text and self.text_dim <= 0:
            raise ValueError("Semantic residual text input is enabled but text_dim is not positive.")
        if self.use_root:
            self.root_proj = nn.Sequential(
                nn.Linear(self.root_dim, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
            )
        if self.use_text:
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_dim, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
            )
        fusion_inputs = int(self.use_root) + int(self.use_text)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_channels * fusion_inputs, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.classifier = nn.Linear(self.hidden_channels, self.num_classes)

    @staticmethod
    def _extract_text_embedding(batch, expected_batch_size: int) -> Tensor:
        if not hasattr(batch, "text_emb"):
            raise ValueError("Semantic residual branch requires batch.text_emb.")
        text_emb = batch.text_emb
        if text_emb.dim() == 3 and int(text_emb.size(1)) == 1:
            text_emb = text_emb.squeeze(1)
        elif text_emb.dim() == 1 and int(expected_batch_size) == 1:
            text_emb = text_emb.view(1, -1)
        if text_emb.dim() != 2 or int(text_emb.size(0)) != int(expected_batch_size):
            raise ValueError(
                f"Semantic residual text_emb must be [B, D], got {tuple(text_emb.shape)} for B={expected_batch_size}."
            )
        return text_emb

    @staticmethod
    def _extract_root_features(batch) -> Tensor:
        return batch.x[_resolve_global_root_index(batch)].to(torch.float32)

    def forward(self, batch, expected_batch_size: int) -> Tuple[Tensor, Tensor]:
        parts: List[Tensor] = []
        if self.use_root:
            root_features = self._extract_root_features(batch).to(
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )
            if int(root_features.size(0)) != int(expected_batch_size):
                raise ValueError(
                    f"Semantic residual root features must have B={expected_batch_size}, got {int(root_features.size(0))}."
                )
            parts.append(self.root_proj(root_features))
        if self.use_text:
            text_emb = self._extract_text_embedding(batch, expected_batch_size).to(
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )
            parts.append(self.text_proj(text_emb))
        semantic_feature = self.fusion(torch.cat(parts, dim=-1))
        return float(self.logit_weight) * self.classifier(semantic_feature), semantic_feature


class PrefixGraphExpertBranch(nn.Module):
    """Observed-prefix root-aware graph expert fused inside RumorDiffusion."""

    def __init__(
        self,
        in_channels: int,
        text_dim: int,
        hidden_channels: int,
        graph_hidden: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        logit_weight: float,
        use_text: bool,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0 or int(hidden_channels) <= 0 or int(graph_hidden) <= 0:
            raise ValueError("Prefix graph expert channel dimensions must be positive.")
        if bool(use_text) and int(text_dim) <= 0:
            raise ValueError("Prefix graph expert text_dim must be positive when text is enabled.")
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.use_text = bool(use_text)
        self.logit_weight = float(logit_weight)
        self.graph_encoder = RootAwareCascadeEncoder(
            in_channels=int(in_channels),
            hidden_channels=int(graph_hidden),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        feature_width = int(self.graph_encoder.output_channels)
        if self.use_text:
            self.text_projection = nn.Sequential(
                nn.Linear(int(text_dim), int(graph_hidden)),
                nn.LayerNorm(int(graph_hidden)),
                nn.GELU(),
                nn.Dropout(p=float(dropout)),
            )
            feature_width += int(graph_hidden)
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_width, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.classifier = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, batch, expected_batch_size: int) -> Tuple[Tensor, Tensor]:
        if not hasattr(batch, "edge_index") or not hasattr(batch, "batch"):
            raise ValueError("Prefix graph expert requires batched graph edges and assignments.")
        root_index = _resolve_global_root_index(batch)
        graph_features = self.graph_encoder(
            batch.x.to(dtype=next(self.parameters()).dtype),
            batch.edge_index,
            batch.batch,
            root_index,
        )
        if int(graph_features.size(0)) != int(expected_batch_size):
            raise ValueError(
                f"Prefix graph expert expected B={expected_batch_size}, got {int(graph_features.size(0))}."
            )
        parts = [graph_features]
        if self.use_text:
            text_emb = SemanticResidualBranch._extract_text_embedding(batch, expected_batch_size).to(
                device=graph_features.device,
                dtype=graph_features.dtype,
            )
            parts.append(self.text_projection(text_emb))
        feature = self.feature_projection(torch.cat(parts, dim=-1))
        return float(self.logit_weight) * self.classifier(feature), feature


class GatedLogitFusion(nn.Module):
    """Learns sample-specific weights for diffusion, semantic, and prototype logits."""

    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        num_branches: int,
        hidden: int,
        dropout: float,
        temperature: float,
        init_diffusion_bias: float,
        init_branch_biases: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if int(num_branches) < 2:
            raise ValueError("Gated logit fusion requires diffusion plus at least one residual branch.")
        if float(temperature) <= 0.0:
            raise ValueError("Gated fusion temperature must be positive.")
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)
        self.num_branches = int(num_branches)
        self.temperature = float(temperature)
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_channels + self.num_branches * self.num_classes, int(hidden)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden), self.num_branches),
        )
        with torch.no_grad():
            final = self.gate[-1]
            assert isinstance(final, nn.Linear)
            final.weight.zero_()
            final.bias.zero_()
            if init_branch_biases is None:
                final.bias[0] = float(init_diffusion_bias)
            else:
                if len(init_branch_biases) != self.num_branches:
                    raise ValueError(
                        "gated_fusion.init_branch_biases must match the number of enabled logit branches."
                    )
                final.bias.copy_(torch.tensor(init_branch_biases, dtype=final.bias.dtype))

    def forward(
        self,
        logit_sources: List[Tensor],
        representation: Optional[Tensor],
        semantic_feature: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if len(logit_sources) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} logit sources, got {len(logit_sources)}.")
        base_logits = logit_sources[0]
        if base_logits.dim() != 2 or int(base_logits.size(1)) != self.num_classes:
            raise ValueError("Gated fusion logits must have shape [B, C].")
        batch_size = int(base_logits.size(0))
        device, dtype = base_logits.device, base_logits.dtype
        if representation is None:
            representation = torch.zeros((batch_size, self.hidden_channels), device=device, dtype=dtype)
        if semantic_feature is None:
            semantic_feature = torch.zeros((batch_size, self.hidden_channels), device=device, dtype=dtype)
        if representation.shape != semantic_feature.shape or tuple(representation.shape) != (batch_size, self.hidden_channels):
            raise ValueError("Gated fusion features must have shape [B, hidden_channels].")
        stacked = torch.stack(logit_sources, dim=1)
        gate_input = torch.cat(
            [representation, semantic_feature, stacked.reshape(batch_size, -1)],
            dim=-1,
        )
        weights = F.softmax(self.gate(gate_input) / self.temperature, dim=-1)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1), weights


class RumorDiffusion(nn.Module):
    """RumorDiffusion with Transformer-based arbiter over class potentials.

    The module consumes per-sample cached potentials (h_cat or h0/h1) when
    provided, or will query CTDM to obtain all-class potentials Hc.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        arbiter_hidden: int = 256,
        dropout: float = 0.3,
        ctdm: Optional[nn.Module] = None,
        freeze_ctdm: bool = True,
        arbiter_type: str = "transformer",
        arbiter_readout: str = "cls",
        token_norm: str = "layernorm",
        token_dropout: float = 0.3,
        transformer_nhead: int = 4,
        transformer_nlayers: int = 2,
        transformer_dim_feedforward: int = 512,
        transformer_dropout: float = 0.3,
        # DD-GAN configuration
        dd_gan_enabled: bool = False,
        lambda_gan: float = 0.0,
        lr_d: float = 1.0e-4,
        lambda_r1: float = 10.0,
        # Pair/token and auxiliary head configuration
        use_pair_token: bool = False,
        pair_ids: Tuple[int, int] = (1, 2),
        pair_token_mode: str = "diff",
        # True/False margin loss
        pair_margin_enabled: bool = False,
        pair_margin_m: float = 0.5,
        lambda_pair_margin: float = 0.0,
        enable_pair_margin_loss: bool = True,
        # Auxiliary TF binary head
        tf_head_enabled: bool = False,
        lambda_tf_bce: float = 0.0,
        tf_infer_adjust_alpha: float = 0.0,
        enable_tf_head_loss: bool = True,
        # Class embedding configuration
        use_class_embedding: bool = False,
        class_embed_mode: str = "concat_proj",
        # DD-GAN gating
        enable_dd_gan: bool = True,
        # Inference/efficiency
        cache_potentials: bool = False,
        infer_amp: bool = False,
        uncertainty_force_ctdm: bool = False,
        uncertainty_sampling: str = "deterministic",
        uncertainty_readout: str = "observed_prefix",
        # Diagnostics
        loss_logging: bool = False,
        # Representation objectives
        lambda_supcon: float = 0.0,
        supcon_temperature: float = 0.07,
        prototype_residual_cfg: Optional[Dict[str, object]] = None,
        semantic_residual_cfg: Optional[Dict[str, object]] = None,
        prefix_graph_expert_cfg: Optional[Dict[str, object]] = None,
        gated_fusion_cfg: Optional[Dict[str, object]] = None,
        posterior_aggregation_cfg: Optional[Dict[str, object]] = None,
        prefix_consistency_cfg: Optional[Dict[str, object]] = None,
        source_robustness_cfg: Optional[Dict[str, object]] = None,
        root_feature_dim: int = 0,
        text_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.ctdm = ctdm
        if freeze_ctdm and self.ctdm is not None:
            for p in self.ctdm.parameters():
                p.requires_grad = False
        # Token-level normalization for potential vectors
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)
        self.arbiter_type = str(arbiter_type).lower()
        self.arbiter_readout = str(arbiter_readout).lower()
        assert self.arbiter_readout in ["cls", "token"], f"arbiter_readout must be 'cls' or 'token', got {self.arbiter_readout}"
        self.cache_potentials = bool(cache_potentials)
        self.infer_amp = bool(infer_amp)
        self.uncertainty_force_ctdm = bool(uncertainty_force_ctdm)
        self.uncertainty_sampling = str(uncertainty_sampling).lower()
        self.uncertainty_readout = str(uncertainty_readout).lower()
        if self.uncertainty_sampling not in {"deterministic", "stochastic_reverse"}:
            raise ValueError(
                "uncertainty_sampling must be 'deterministic' or 'stochastic_reverse'."
            )
        if self.uncertainty_readout not in {"observed", "observed_prefix", "sampled_full_graph"}:
            raise ValueError(
                "uncertainty_readout must be 'observed_prefix' or 'sampled_full_graph'."
            )
        self.loss_logging = bool(loss_logging)
        norm_choices = {
            "layernorm": nn.LayerNorm(self.hidden_channels),
            "none": nn.Identity(),
        }
        self.token_norm = norm_choices.get(str(token_norm).lower(), norm_choices["layernorm"])
        self.token_drop = nn.Dropout(p=float(token_dropout))
        # Pair/token and aux-head flags
        self.use_pair_token = bool(use_pair_token)
        self.pair_ids = (int(pair_ids[0]), int(pair_ids[1]))
        self.pair_token_mode = str(pair_token_mode).lower()
        self.pair_margin_enabled = bool(pair_margin_enabled)
        self.pair_margin_m = float(pair_margin_m)
        self.lambda_pair_margin = float(lambda_pair_margin)
        self.enable_pair_margin_loss = bool(enable_pair_margin_loss)
        self.tf_head_enabled = bool(tf_head_enabled)
        self.lambda_tf_bce = float(lambda_tf_bce)
        self.tf_infer_adjust_alpha = float(tf_infer_adjust_alpha)
        self.enable_tf_head_loss = bool(enable_tf_head_loss)
        self.lambda_supcon = float(lambda_supcon)
        self.supcon_temperature = float(supcon_temperature)
        prototype_residual_cfg = prototype_residual_cfg if isinstance(prototype_residual_cfg, dict) else {}
        semantic_residual_cfg = semantic_residual_cfg if isinstance(semantic_residual_cfg, dict) else {}
        prefix_graph_expert_cfg = prefix_graph_expert_cfg if isinstance(prefix_graph_expert_cfg, dict) else {}
        gated_fusion_cfg = gated_fusion_cfg if isinstance(gated_fusion_cfg, dict) else {}
        posterior_aggregation_cfg = posterior_aggregation_cfg if isinstance(posterior_aggregation_cfg, dict) else {}
        prefix_consistency_cfg = prefix_consistency_cfg if isinstance(prefix_consistency_cfg, dict) else {}
        source_robustness_cfg = source_robustness_cfg if isinstance(source_robustness_cfg, dict) else {}
        self.prototype_residual_enabled = bool(prototype_residual_cfg.get("enabled", False))
        self.semantic_residual_enabled = bool(semantic_residual_cfg.get("enabled", False))
        self.prefix_graph_expert_enabled = bool(prefix_graph_expert_cfg.get("enabled", False))
        self.posterior_aggregation_enabled = bool(posterior_aggregation_cfg.get("enabled", False))
        if self.prototype_residual_enabled:
            self.prototype_residual = PrototypeResidualBranch(
                hidden_channels=self.hidden_channels,
                num_classes=self.num_classes,
                temperature=float(prototype_residual_cfg["temperature"]),
                momentum=float(prototype_residual_cfg["momentum"]),
                logit_weight=float(prototype_residual_cfg["logit_weight"]),
                min_count=int(prototype_residual_cfg["min_count"]),
            )
        if self.semantic_residual_enabled:
            self.semantic_residual = SemanticResidualBranch(
                root_dim=int(root_feature_dim),
                text_dim=int(text_feature_dim),
                hidden_channels=self.hidden_channels,
                num_classes=self.num_classes,
                dropout=float(semantic_residual_cfg["dropout"]),
                logit_weight=float(semantic_residual_cfg["logit_weight"]),
                use_root=bool(semantic_residual_cfg["use_root"]),
                use_text=bool(semantic_residual_cfg["use_text"]),
            )
        if self.prefix_graph_expert_enabled:
            self.prefix_graph_expert = PrefixGraphExpertBranch(
                in_channels=int(root_feature_dim),
                text_dim=int(text_feature_dim),
                hidden_channels=self.hidden_channels,
                graph_hidden=int(prefix_graph_expert_cfg.get("hidden", max(1, self.hidden_channels // 2))),
                num_layers=int(prefix_graph_expert_cfg.get("num_layers", 2)),
                num_classes=self.num_classes,
                dropout=float(prefix_graph_expert_cfg.get("dropout", dropout)),
                logit_weight=float(prefix_graph_expert_cfg.get("logit_weight", 1.0)),
                use_text=bool(prefix_graph_expert_cfg.get("use_text", True)),
            )
        self.gated_fusion_enabled = bool(gated_fusion_cfg.get("enabled", False))
        fusion_branches = (
            1
            + int(self.semantic_residual_enabled)
            + int(self.prefix_graph_expert_enabled)
            + int(self.prototype_residual_enabled)
        )
        if self.gated_fusion_enabled:
            self.gated_fusion = GatedLogitFusion(
                hidden_channels=self.hidden_channels,
                num_classes=self.num_classes,
                num_branches=fusion_branches,
                hidden=int(gated_fusion_cfg.get("hidden", self.hidden_channels)),
                dropout=float(gated_fusion_cfg.get("dropout", dropout)),
                temperature=float(gated_fusion_cfg.get("temperature", 1.0)),
                init_diffusion_bias=float(gated_fusion_cfg.get("init_diffusion_bias", 1.5)),
                init_branch_biases=gated_fusion_cfg.get("init_branch_biases", None),
            )
        self.prefix_consistency_enabled = bool(prefix_consistency_cfg.get("enabled", False))
        self.prefix_consistency_supervised_weight = float(prefix_consistency_cfg.get("supervised_weight", 0.0))
        self.prefix_consistency_kl_weight = float(prefix_consistency_cfg.get("symmetric_kl_weight", 0.0))
        self.prefix_consistency_temperature = float(prefix_consistency_cfg.get("temperature", 1.0))
        if self.prefix_consistency_enabled and self.prefix_consistency_temperature <= 0.0:
            raise ValueError("Prefix-consistency temperature must be positive.")
        self.source_robustness_enabled = bool(source_robustness_cfg.get("enabled", False))
        self.source_robustness_weight = float(source_robustness_cfg.get("weight", 0.0))
        self.source_robustness_edge_dropout = float(source_robustness_cfg.get("edge_dropout", 0.0))
        self.source_robustness_text_dropout = float(source_robustness_cfg.get("text_dropout", 0.0))
        self.source_robustness_text_noise_std = float(source_robustness_cfg.get("text_noise_std", 0.0))
        self.source_robustness_prefix_weight = float(source_robustness_cfg.get("prefix_weight", 0.0))
        for name, value in (
            ("edge_dropout", self.source_robustness_edge_dropout),
            ("text_dropout", self.source_robustness_text_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"Source-robustness {name} must be in [0, 1).")
        if self.posterior_aggregation_enabled:
            self.posterior_aggregation_samples = int(posterior_aggregation_cfg["samples"])
            self.posterior_aggregation_mode = str(posterior_aggregation_cfg["mode"]).lower()
            self.posterior_aggregation_force_ctdm = bool(posterior_aggregation_cfg["force_ctdm"])
            self.posterior_aggregation_stochastic = bool(posterior_aggregation_cfg["stochastic"])
            if self.posterior_aggregation_samples < 1:
                raise ValueError("posterior_aggregation.samples must be positive.")
            if self.posterior_aggregation_mode not in {"mean_logits", "logmeanexp", "mean_prob"}:
                raise ValueError("posterior_aggregation.mode must be mean_logits, logmeanexp, or mean_prob.")
            if self.posterior_aggregation_samples > 1 and not self.posterior_aggregation_stochastic:
                warnings.warn(
                    "Deterministic posterior aggregation repeats identical logits; reducing samples to 1.",
                    RuntimeWarning,
                )
                self.posterior_aggregation_samples = 1
        self._posterior_repeat_warned = False
        # Multi-pair margin loss configuration (for handling multiple confusable class pairs)
        self.multi_pair_margin_list = []
        self.multi_pair_margin_values = []
        self.lambda_multi_pair_margin = 0.0
        self.enable_multi_pair_margin = True
        # [CLS] token for Transformer arbiter
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_channels))
        # Learnable class embeddings to enhance class-awareness in potential vectors (optional)
        self.use_class_embedding = bool(use_class_embedding)
        self.class_embed_mode = str(class_embed_mode).lower()
        if self.use_class_embedding:
            self.class_embedding = nn.Embedding(num_classes, self.hidden_channels)
            # Projection layer to fuse class embedding with potential vector
            self.class_embed_fusion = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
            if self.class_embed_mode == "gated":
                self.class_embed_gate = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        if self.pair_token_mode == "learnable":
            self.learnable_pair_token = nn.Parameter(torch.zeros(1, 1, self.hidden_channels))
        if self.arbiter_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_channels,
                nhead=int(transformer_nhead),
                dim_feedforward=int(transformer_dim_feedforward),
                dropout=float(transformer_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(transformer_nlayers))
            if self.arbiter_readout == "token":
                self.token_arbiter = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_channels, arbiter_hidden),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=dropout),
                        nn.Linear(arbiter_hidden, 1),
                    ) for _ in range(num_classes)
                ])
            else:
                self.arbiter = ArbiterMLP(
                    in_dim=self.hidden_channels,
                    num_classes=num_classes,
                    hidden=arbiter_hidden,
                    dropout=dropout,
                )
            if self.tf_head_enabled:
                self.tf_head = nn.Linear(self.hidden_channels, 1)
        else:
            # Fallback MLP over concatenated first two potentials if explicitly requested
            self.input_dim = self.hidden_channels * 2
            self.input_norm = nn.LayerNorm(self.input_dim)
            self.input_drop = nn.Dropout(p=dropout)
            self.arbiter = ArbiterMLP(in_dim=self.input_dim, num_classes=num_classes, hidden=arbiter_hidden, dropout=dropout)
        # Loss hyperparameters (can be overridden externally)
        self.la_tau = 1.0
        self.la_gamma = 1.5
        self.la_smooth = 0.05
        self.lambda_brier = 0.01
        self.lambda_consistency = 0.05
        self.lambda_diffusion = 0.0
        # Class priors for logit-adjusted losses
        self.register_buffer("class_priors", torch.ones((num_classes,), dtype=torch.float32) / float(num_classes), persistent=True)
        # Diagnostics buffer
        self._last_loss_breakdown = {}

        # DD-GAN components
        self.dd_gan_enabled = bool(dd_gan_enabled) and bool(enable_dd_gan)
        if self.dd_gan_enabled:
            self.discriminator = PotentialDiscriminator(
                hidden_channels=self.hidden_channels,
                num_classes=self.num_classes,
                hidden=int(arbiter_hidden),
                dropout=float(dropout),
            )
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(), lr=float(lr_d), weight_decay=1.0e-5
            )
            self.lambda_gan = float(lambda_gan)
            self.lambda_r1 = float(lambda_r1)

    @staticmethod
    def _effective_pair_ids(pair_ids: Tuple[int, int], num_classes: int) -> Tuple[int, int] | Tuple[()]:
        valid = tuple(i for i in pair_ids if 0 <= int(i) < int(num_classes))
        if len(valid) >= 2:
            return (int(valid[0]), int(valid[1]))
        if num_classes >= 2:
            return (0, 1)
        return tuple()

    def _encode_from_tokens(self, tokens: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Encodes a sequence of tokens via Transformer and returns (logits, cls_out).

        If pair token injection is enabled and compatible with token layout,
        appends a differential token between configured class indices.
        Optionally fuses learnable class embeddings with each potential vector to enhance class-awareness.
        """
        B = int(tokens.size(0))
        H = int(tokens.size(-1))
        # Optional fusion of learnable class embeddings with each potential vector
        if self.use_class_embedding:
            num_classes = int(self.num_classes)
            class_embs = self.class_embedding(torch.arange(num_classes, device=tokens.device)).unsqueeze(0).expand(B, -1, -1)
            if int(tokens.size(1)) >= num_classes:
                tokens_with_class = tokens[:, :num_classes, :]
                class_embs = class_embs[:, :int(tokens_with_class.size(1)), :]
            else:
                class_embs = class_embs[:, :int(tokens.size(1)), :]
            tokens_slice = tokens[:, :int(class_embs.size(1)), :]
            fused_tokens = torch.cat([tokens_slice, class_embs], dim=-1)
            if self.class_embed_mode == "gated":
                proj = self.class_embed_fusion(fused_tokens)
                gate = torch.sigmoid(self.class_embed_gate(fused_tokens))
                tokens = gate * proj + (1.0 - gate) * tokens_slice
            else:
                tokens = self.class_embed_fusion(fused_tokens)
        # Optional pair token: requires tokens to include class-aligned positions
        pair_ids_eff = self._effective_pair_ids(self.pair_ids, int(self.num_classes))
        if self.use_pair_token and len(pair_ids_eff) == 2:
            p0, p1 = pair_ids_eff
            pair_token = None
            if self.pair_token_mode == "learnable":
                pair_token = self.learnable_pair_token.expand(B, 1, H)
            else:
                if int(tokens.size(1)) > max(p0, p1):
                    d_tf = tokens[:, p0, :] - tokens[:, p1, :]
                    pair_token = d_tf.unsqueeze(1)
            if pair_token is not None:
                tokens = torch.cat([tokens, pair_token], dim=1)
        cls = self.cls_token.expand(B, 1, H)
        seq = torch.cat([cls, self.token_drop(self.token_norm(tokens))], dim=1)
        out = self.transformer_encoder(seq)
        
        if self.arbiter_readout == "token":
            cls_out = out[:, 0, :]
            token_out = out[:, 1:, :]
            num_token_classes = min(int(token_out.size(1)), int(self.num_classes))
            logits_list = []
            for c in range(int(self.num_classes)):
                if c < num_token_classes:
                    logit_c = self.token_arbiter[c](token_out[:, c, :]).view(B, 1)
                else:
                    logit_c = torch.zeros(B, 1, device=token_out.device, dtype=token_out.dtype)
                logits_list.append(logit_c)
            logits = torch.cat(logits_list, dim=1)
        else:
            cls_out = out[:, 0, :]
            logits = self.arbiter(cls_out)
        
        # Optional inference-time adjustment for class pair using TF head
        if self.tf_head_enabled and self.tf_infer_adjust_alpha > 0.0 and len(pair_ids_eff) == 2:
            tf_logit = self.tf_head(cls_out).view(-1)
            p = torch.sigmoid(tf_logit).view(-1, 1)
            p0, p1 = pair_ids_eff
            logits[:, p0] = logits[:, p0] + float(self.tf_infer_adjust_alpha) * p.view(-1)
            logits[:, p1] = logits[:, p1] + float(self.tf_infer_adjust_alpha) * (1.0 - p.view(-1))
        return logits, cls_out

    def _build_tokens_from_batch(self, batch) -> Tensor:
        """Builds per-sample token sequence [B, S, H] from batch or CTDM."""
        if hasattr(batch, "h_cat"):
            x = batch.h_cat
            B = int(x.size(0))
            H = self.hidden_channels
            K = int(x.size(1)) // H
            tokens = x.view(B, K, H)
            self._last_potentials = tokens
            return tokens
        if hasattr(batch, "h0") and hasattr(batch, "h1"):
            tokens = torch.stack([batch.h0, batch.h1], dim=1)
            self._last_potentials = tokens
            return tokens
        assert self.ctdm is not None, "CTDM is required when potentials are not present in the batch"
        Hc = self._compute_ctdm_potentials(batch)
        self._last_potentials = Hc
        return Hc

    def _logits_from_tokens(self, tokens: Tensor) -> Tensor:
        """Maps potential tokens to class logits for both arbiter variants."""
        if self.arbiter_type == "transformer":
            logits, _cls = self._encode_from_tokens(tokens)
            return logits
        x = torch.cat([tokens[:, 0, :], tokens[:, 1, :]], dim=-1)
        x = self.input_drop(self.input_norm(x))
        return self.arbiter(x)

    def _apply_residual_branches(
        self,
        logits: Tensor,
        representation: Optional[Tensor],
        batch,
        targets: Optional[Tensor] = None,
        update_prototypes: bool = False,
    ) -> Tensor:
        logit_sources = [logits]
        residual_features = representation
        semantic_feature: Optional[Tensor] = None
        if self.semantic_residual_enabled:
            semantic_logits, semantic_feature = self.semantic_residual(batch, expected_batch_size=int(logits.size(0)))
            logit_sources.append(semantic_logits)
            if residual_features is None:
                residual_features = semantic_feature
        if self.prefix_graph_expert_enabled:
            graph_logits, graph_feature = self.prefix_graph_expert(
                batch,
                expected_batch_size=int(logits.size(0)),
            )
            logit_sources.append(graph_logits)
            if residual_features is None:
                residual_features = graph_feature
        if self.prototype_residual_enabled and residual_features is not None:
            logit_sources.append(self.prototype_residual(residual_features))
            if bool(update_prototypes):
                if targets is None:
                    raise ValueError("Prototype residual update requires training targets.")
                self.prototype_residual.update(residual_features.detach(), targets.detach())
        if self.gated_fusion_enabled:
            if len(logit_sources) != self.gated_fusion.num_branches:
                raise RuntimeError("Gated fusion branch configuration does not match available residual logits.")
            fused_logits, fusion_gates = self.gated_fusion(
                logit_sources,
                representation=residual_features,
                semantic_feature=semantic_feature,
            )
            self._last_fusion_gates = fusion_gates.detach()
            return fused_logits
        return sum(logit_sources)

    def _logits_from_tokens_with_residuals(
        self,
        tokens: Tensor,
        batch,
        targets: Optional[Tensor] = None,
        update_prototypes: bool = False,
    ) -> Tensor:
        if self.arbiter_type == "transformer":
            logits, cls_out = self._encode_from_tokens(tokens)
            return self._apply_residual_branches(
                logits,
                cls_out,
                batch,
                targets=targets,
                update_prototypes=update_prototypes,
            )
        logits = self._logits_from_tokens(tokens)
        return self._apply_residual_branches(
            logits,
            None,
            batch,
            targets=targets,
            update_prototypes=update_prototypes,
        )

    def _forward_from_ctdm(self, batch, stochastic: bool = False) -> Tensor:
        """Forces on-the-fly CTDM potential inference regardless of cached fields."""
        assert self.ctdm is not None, "CTDM is required for sampled uncertainty inference"
        Hc = self._compute_ctdm_potentials(batch, stochastic=stochastic)
        self._last_potentials = Hc
        return self._logits_from_tokens_with_residuals(Hc, batch)

    def _ctdm_trainable_parameter_names(self) -> List[str]:
        if self.ctdm is None:
            return []
        return [name for name, param in self.ctdm.named_parameters() if param.requires_grad]

    def _ctdm_adapter_only_trainable(self) -> bool:
        trainable_names = self._ctdm_trainable_parameter_names()
        return len(trainable_names) > 0 and all(
            name.startswith("potential_adapter.") for name in trainable_names
        )

    def _ctdm_reverse_trajectory_trainable(self) -> bool:
        trainable_names = self._ctdm_trainable_parameter_names()
        return any(not name.startswith("potential_adapter.") for name in trainable_names)

    def _compute_ctdm_potentials(self, batch, stochastic: bool = False) -> Tensor:
        assert self.ctdm is not None, "CTDM is required when potentials are not present in the batch"
        ctdm_trainable = self.training and len(self._ctdm_trainable_parameter_names()) > 0
        if bool(stochastic) and ctdm_trainable:
            raise ValueError("Stochastic CTDM uncertainty sampling is inference-only.")
        if ctdm_trainable and self._ctdm_adapter_only_trainable():
            return self.ctdm.potentials_all_adapter_grad(batch)
        if ctdm_trainable:
            return self.ctdm.potentials_all_grad(batch)
        if bool(stochastic):
            return self.ctdm.sample_potentials_all(
                batch,
                readout_source=self.uncertainty_readout,
            )
        return self.ctdm.infer_potentials_all(batch)

    def _forward_once(self, batch, force_ctdm: bool = False, stochastic: bool = False) -> Tensor:
        if force_ctdm:
            return self._forward_from_ctdm(batch, stochastic=stochastic)
        if self.arbiter_type == "transformer":
            tokens = self._build_tokens_from_batch(batch)
            return self._logits_from_tokens_with_residuals(tokens, batch)
        if hasattr(batch, "h_cat"):
            x = batch.h_cat
            H = self.hidden_channels
            x = x[:, : 2 * H]
            x = self.input_drop(self.input_norm(x))
            return self._apply_residual_branches(self.arbiter(x), None, batch)
        if hasattr(batch, "h0") and hasattr(batch, "h1"):
            x = torch.cat([batch.h0, batch.h1], dim=-1)
            x = self.input_drop(self.input_norm(x))
            return self._apply_residual_branches(self.arbiter(x), None, batch)
        assert self.ctdm is not None, "CTDM is required when potentials are not present in the batch"
        return self._forward_from_ctdm(batch)

    def _posterior_aggregated_logits(self, batch) -> Tensor:
        logits_samples: List[Tensor] = []
        for _ in range(int(self.posterior_aggregation_samples)):
            with autocast(enabled=False):
                sample_logits = self._forward_once(
                    batch,
                    force_ctdm=bool(self.posterior_aggregation_force_ctdm),
                    stochastic=bool(self.posterior_aggregation_stochastic),
                )
            logits_samples.append(sample_logits.to(torch.float32))
        stacked = torch.stack(logits_samples, dim=0)
        if stacked.size(0) > 1:
            repeated = all(torch.allclose(stacked[0], sample, rtol=1e-5, atol=1e-6) for sample in stacked[1:])
            if repeated:
                if not self._posterior_repeat_warned:
                    warnings.warn(
                        "Posterior aggregation produced repeated logits; reducing effective samples to 1.",
                        RuntimeWarning,
                    )
                    self._posterior_repeat_warned = True
                stacked = stacked[:1]
        if self.posterior_aggregation_mode == "mean_logits":
            return stacked.mean(dim=0)
        if self.posterior_aggregation_mode == "logmeanexp":
            return torch.logsumexp(stacked, dim=0) - math.log(float(stacked.size(0)))
        probs = F.softmax(stacked, dim=-1).mean(dim=0).clamp_min(1e-8)
        return probs.log()

    def forward(self, batch) -> Tensor:
        """Build token sequence and arbitrate via Transformer [CLS] output by default."""
        if not self.training and hasattr(batch, "to_data_list"):
            mature_fields = (
                "full_x", "full_edge_index", "full_root_index", "full_node_time_minutes",
                "full_node_time_minutes_shifted", "full_node_time_hops", "full_node_time_hops_shifted",
                "full_node_uid", "full_node_tweet_id",
            )
            for sample in batch.to_data_list():
                if any(hasattr(sample, field) for field in mature_fields):
                    raise RuntimeError(
                        "RumorDiffusion eval received mature-state attributes; use eval_prefix input only."
                    )
        def _compute():
            if (not self.training) and self.posterior_aggregation_enabled:
                return self._posterior_aggregated_logits(batch)
            return self._forward_once(batch)

        if (not self.training) and self.infer_amp:
            with autocast():
                return _compute()
        return _compute()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        batch,
        num_samples: int = 4,
        rumor_classes: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Runs multi-sample inference and returns (mean_prob, var_prob).

        If rumor_classes provided, returns aggregated rumor probability/variance
        of shape [B, 1]; otherwise returns per-class tensors [B, C].
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        samples = max(1, int(num_samples))
        probs = []
        stochastic = self.uncertainty_sampling == "stochastic_reverse"
        force_ctdm = self.uncertainty_force_ctdm or stochastic
        for _ in range(samples):
            if force_ctdm:
                assert self.ctdm is not None, "CTDM is required for stochastic uncertainty sampling"
                if (not self.training) and self.infer_amp:
                    with autocast():
                        logits = self._forward_from_ctdm(batch, stochastic=stochastic)
                else:
                    logits = self._forward_from_ctdm(batch, stochastic=stochastic)
            else:
                if (not self.training) and self.infer_amp:
                    with autocast():
                        logits = self(batch)
                else:
                    logits = self(batch)
            probs.append(F.softmax(logits, dim=1))
        prob_stack = torch.stack(probs, dim=0)  # [S, B, C]
        mean_prob = prob_stack.mean(dim=0)
        var_prob = prob_stack.var(dim=0, unbiased=False)
        if rumor_classes is None:
            return mean_prob, var_prob
        mask = torch.zeros((mean_prob.size(1),), device=mean_prob.device, dtype=mean_prob.dtype)
        for cid in rumor_classes:
            if 0 <= int(cid) < mask.numel():
                mask[int(cid)] = 1.0
        rumor_prob = (mean_prob * mask.view(1, -1)).sum(dim=1, keepdim=True)
        rumor_var = (var_prob * mask.view(1, -1)).sum(dim=1, keepdim=True)
        return rumor_prob, rumor_var

    @staticmethod
    def _tf_margin_loss_static(logits: Tensor, targets: Tensor, pair_ids: Tuple[int, int], margin: float) -> Tensor:
        """Computes margin loss between logits of a configured class pair.

        Applies only to samples whose targets belong to the specified pair.
        """
        p0, p1 = int(pair_ids[0]), int(pair_ids[1])
        y = targets.view(-1)
        mask = (y == p0) | (y == p1)
        if not mask.any():
            return logits.new_zeros(())
        z = logits[mask]
        y_masked = y[mask]
        # margin for y==p0: m - (z_p0 - z_p1), for y==p1: m - (z_p1 - z_p0)
        z0 = z[:, p0]
        z1 = z[:, p1]
        diff = torch.where(y_masked == p0, z0 - z1, z1 - z0)
        loss = F.relu(float(margin) - diff)
        return loss.mean()

    @staticmethod
    def _multi_pair_margin_loss(logits: Tensor, targets: Tensor, pair_list: list, margin_list: list) -> Tensor:
        """Computes margin loss for multiple class pairs.

        Generalizes pair margin loss to handle arbitrary class pairs for better discrimination
        between confusable categories. Each pair can have its own margin threshold.

        Args:
            logits: Classification logits of shape [B, C].
            targets: Target class indices of shape [B].
            pair_list: List of tuples, each containing (class_id_0, class_id_1) pairs.
            margin_list: List of margin values corresponding to each pair.

        Returns:
            Aggregated margin loss across all pairs.
        """
        if not pair_list or len(pair_list) == 0:
            return logits.new_zeros(())
        
        total_loss = 0.0
        num_pairs = len(pair_list)
        y = targets.view(-1)
        
        for pair_ids, margin in zip(pair_list, margin_list):
            p0, p1 = int(pair_ids[0]), int(pair_ids[1])
            mask = (y == p0) | (y == p1)
            if not mask.any():
                continue
            z = logits[mask]
            y_masked = y[mask]
            z0 = z[:, p0]
            z1 = z[:, p1]
            diff = torch.where(y_masked == p0, z0 - z1, z1 - z0)
            loss = F.relu(float(margin) - diff)
            total_loss = total_loss + loss.mean()
        
        return total_loss / max(1, num_pairs)

    @staticmethod
    def _label_smoothing_targets(targets: Tensor, num_classes: int, epsilon: float) -> Tensor:
        with torch.no_grad():
            y = torch.zeros((targets.size(0), num_classes), device=targets.device, dtype=torch.float32)
            y.scatter_(1, targets.view(-1, 1), 1.0)
            y = (1.0 - epsilon) * y + epsilon / float(num_classes)
            return y

    def _la_focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        assert logits.dim() == 2 and targets.dim() == 1
        B, C = logits.shape
        priors = self.class_priors.to(logits.device).clamp_min(1e-6)
        z_tilde = logits + self.la_tau * priors.log().view(1, -1)
        log_p = F.log_softmax(z_tilde, dim=1)
        p = log_p.exp()
        y_s = self._label_smoothing_targets(targets, C, self.la_smooth)
        ce = -(y_s * log_p).sum(dim=1)
        p_y = p[torch.arange(B, device=logits.device), targets]
        mod = (1.0 - p_y).clamp_min(0.0).pow(self.la_gamma)
        return (mod * ce).mean()

    @staticmethod
    def _brier_loss(logits: Tensor, targets: Tensor) -> Tensor:
        B, C = logits.shape
        p = F.softmax(logits, dim=1)
        y = torch.zeros((B, C), device=logits.device, dtype=torch.float32)
        y.scatter_(1, targets.view(-1, 1), 1.0)
        return torch.mean((p - y) ** 2)

    @staticmethod
    def _symmetric_kl(p: Tensor, q: Tensor) -> Tensor:
        eps = 1e-8
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        kl_pq = (p * (p / q).log()).sum(dim=1)
        kl_qp = (q * (q / p).log()).sum(dim=1)
        return (kl_pq + kl_qp).mean()

    def _symmetric_kl_logits(self, logits_a: Tensor, logits_b: Tensor, temperature: float = 1.0) -> Tensor:
        temp = max(float(temperature), 1e-6)
        return self._symmetric_kl(
            F.softmax(logits_a / temp, dim=1),
            F.softmax(logits_b / temp, dim=1),
        )

    def _prefix_view_batches(self, batch) -> List:
        views = getattr(batch, "prefix_views", None)
        if not isinstance(views, (list, tuple)) or len(views) == 0:
            raise RuntimeError("Enabled prefix regularization requires batched prefix_views from GraphDataset.")
        if isinstance(views[0], (list, tuple)):
            from torch_geometric.data import Batch

            batch_size = int(batch.y.view(-1).numel())
            if len(views) == batch_size:
                # PyG preserves arbitrary nested Data attributes per source graph.
                view_count = len(views[0])
                if view_count == 0 or any(len(sample_views) != view_count for sample_views in views):
                    raise RuntimeError("Prefix views must have a consistent view count for every source graph.")
                return [Batch.from_data_list([sample_views[index] for sample_views in views]) for index in range(view_count)]
            if all(len(view_sources) == batch_size for view_sources in views):
                # Also support an already transposed [view][source] representation.
                return [Batch.from_data_list(list(view_sources)) for view_sources in views]
            raise RuntimeError("Unable to align nested prefix views with the current source batch.")
        return list(views)

    def _source_robust_batch(self, batch):
        """Applies source-only edge/text perturbations without changing labels or splits."""
        from torch_geometric.data import Batch

        robust_graphs = []
        for graph in batch.to_data_list():
            if hasattr(graph, "prefix_views"):
                delattr(graph, "prefix_views")
            # Cached potentials describe the unperturbed observed prefix. They
            # are invalid once this graph is changed, so robustness training
            # must recompute CTDM potentials instead of comparing stale logits.
            for stale_attr in ("h0", "h1", "h_cat"):
                if hasattr(graph, stale_attr):
                    delattr(graph, stale_attr)
            if self.source_robustness_edge_dropout > 0.0 and hasattr(graph, "edge_index"):
                edge_index = graph.edge_index
                if int(edge_index.size(1)) > 0:
                    keep = torch.rand((int(edge_index.size(1)),), device=edge_index.device) >= self.source_robustness_edge_dropout
                    if not bool(keep.any()):
                        keep[0] = True
                    graph.edge_index = edge_index[:, keep]
                    if hasattr(graph, "prefix_edge_index"):
                        graph.prefix_edge_index = graph.edge_index.clone()
            if hasattr(graph, "text_emb"):
                text_emb = graph.text_emb.to(torch.float32)
                if self.source_robustness_text_dropout > 0.0:
                    text_emb = F.dropout(text_emb, p=self.source_robustness_text_dropout, training=True)
                if self.source_robustness_text_noise_std > 0.0:
                    text_emb = text_emb + torch.randn_like(text_emb) * self.source_robustness_text_noise_std
                graph.text_emb = text_emb
            robust_graphs.append(graph)
        return Batch.from_data_list(robust_graphs)

    def _multiview_regularization(self, batch, main_logits: Tensor, targets: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Computes source-only prefix and perturbation consistency objectives."""
        total = main_logits.new_zeros(())
        details: Dict[str, float] = {}
        main_potentials = getattr(self, "_last_potentials", None)
        view_logits: List[Tensor] = []
        if self.prefix_consistency_enabled:
            supervised_losses: List[Tensor] = []
            consistency_losses: List[Tensor] = []
            for view_batch in self._prefix_view_batches(batch):
                view_batch = view_batch.to(main_logits.device)
                view_logits_i = self._forward_once(view_batch)
                view_targets = view_batch.y.view(-1).long()
                if not torch.equal(view_targets, targets):
                    raise RuntimeError("Prefix consistency views must preserve source-order labels.")
                view_logits.append(view_logits_i)
                supervised_losses.append(self._la_focal_loss(view_logits_i, targets))
                consistency_losses.append(
                    self._symmetric_kl_logits(
                        main_logits,
                        view_logits_i,
                        temperature=self.prefix_consistency_temperature,
                    )
                )
            supervised = torch.stack(supervised_losses).mean()
            consistency = torch.stack(consistency_losses).mean()
            total = total + self.prefix_consistency_supervised_weight * supervised
            total = total + self.prefix_consistency_kl_weight * consistency
            details["prefix_supervised"] = float(supervised.detach().item())
            details["prefix_symmetric_kl"] = float(consistency.detach().item())

        if self.source_robustness_enabled and self.source_robustness_weight > 0.0:
            robust_logits = self._forward_once(self._source_robust_batch(batch))
            robust_kl = self._symmetric_kl_logits(main_logits, robust_logits)
            total = total + self.source_robustness_weight * robust_kl
            details["source_robustness"] = float(robust_kl.detach().item())
            if self.source_robustness_prefix_weight > 0.0:
                if not view_logits:
                    view_logits = [
                        self._forward_once(view_batch.to(main_logits.device))
                        for view_batch in self._prefix_view_batches(batch)
                    ]
                prefix_robust_kl = torch.stack(
                    [self._symmetric_kl_logits(main_logits, logits) for logits in view_logits]
                ).mean()
                total = total + self.source_robustness_prefix_weight * prefix_robust_kl
                details["source_prefix_robustness"] = float(prefix_robust_kl.detach().item())

        if main_potentials is not None:
            self._last_potentials = main_potentials
        return total, details

    def training_step(self, batch) -> Tuple[Tensor, float]:
        """Performs one training step and returns (loss, aux_acc).

        Computes the primary classification objective and auxiliary
        regularization terms. When adversarial regularization is enabled,
        updates the discriminator and augments the main loss accordingly.
        """
        targets = batch.y.view(-1).long()
        loss_dict = {}
        has_cached_potentials = hasattr(batch, "h_cat") or (hasattr(batch, "h0") and hasattr(batch, "h1"))
        ctdm_trainable = self.ctdm is not None and len(self._ctdm_trainable_parameter_names()) > 0
        ctdm_reverse_trajectory_trainable = self._ctdm_reverse_trajectory_trainable()
        diffusion_loss: Optional[Tensor] = None
        if ctdm_reverse_trajectory_trainable and float(self.lambda_diffusion) > 0.0:
            diffusion_loss, _ = self.ctdm.training_step(batch)

        if self.dd_gan_enabled:
            Hc = self._build_tokens_from_batch(batch)

            B, C, H = int(Hc.size(0)), int(Hc.size(1)), int(Hc.size(2))
            Hc_detached = Hc.detach().view(B * C, H)
            Hc_detached.requires_grad_(True)
            y_disc = torch.arange(C, device=Hc.device).repeat(B)

            # Step A: discriminator update
            self.discriminator.train()
            for p in self.discriminator.parameters():
                p.requires_grad = True
            with autocast(enabled=False):
                self.optimizer_d.zero_grad(set_to_none=True)
                logits_d = self.discriminator(Hc_detached.float())
                loss_d_ce = F.cross_entropy(logits_d, y_disc)
                grad_real = torch.autograd.grad(
                    outputs=logits_d.sum(),
                    inputs=Hc_detached,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                r1_penalty = grad_real.pow(2).reshape(B * C, -1).sum(dim=1).mean()
                loss_d = loss_d_ce + (float(self.lambda_r1) / 2.0) * r1_penalty
                loss_d.backward()
                self.optimizer_d.step()

            # Step B: generator + classifier
            self.discriminator.eval()
            for p in self.discriminator.parameters():
                p.requires_grad = False

            if self.arbiter_type == "transformer":
                logits_main, cls_out = self._encode_from_tokens(Hc)
                logits_main = self._apply_residual_branches(
                    logits_main,
                    cls_out,
                    batch,
                    targets=targets,
                    update_prototypes=True,
                )
            else:
                x = torch.cat([Hc[:, 0, :], Hc[:, 1, :]], dim=-1)
                x = self.input_drop(self.input_norm(x))
                logits_main = self.arbiter(x)
                cls_out = None
                logits_main = self._apply_residual_branches(
                    logits_main,
                    None,
                    batch,
                    targets=targets,
                    update_prototypes=True,
                )

            loss_focal = self._la_focal_loss(logits_main, targets)
            loss = loss_focal
            loss_dict["la_focal"] = float(loss_focal.detach().item())

            if float(self.lambda_supcon) > 0.0 and cls_out is not None:
                loss_supcon = supervised_contrastive_loss(cls_out, targets, self.supcon_temperature)
                loss = loss + float(self.lambda_supcon) * loss_supcon
                loss_dict["supcon"] = float(loss_supcon.detach().item())

            pair_ids_eff = self._effective_pair_ids(self.pair_ids, int(self.num_classes))
            if self.enable_pair_margin_loss and self.pair_margin_enabled and float(self.lambda_pair_margin) > 0.0:
                if len(pair_ids_eff) == 2:
                    loss_tf_margin = self._tf_margin_loss_static(logits_main, targets, pair_ids_eff, self.pair_margin_m)
                    loss = loss + float(self.lambda_pair_margin) * loss_tf_margin
                    loss_dict["pair_margin"] = float(loss_tf_margin.detach().item())

            if self.enable_multi_pair_margin and len(self.multi_pair_margin_list) > 0 and float(self.lambda_multi_pair_margin) > 0.0:
                pair_filtered = []
                margin_filtered = []
                for pair_ids, margin in zip(self.multi_pair_margin_list, self.multi_pair_margin_values):
                    if len(pair_ids) == 2 and max(pair_ids) < int(self.num_classes):
                        pair_filtered.append(pair_ids)
                        margin_filtered.append(margin)
                if len(pair_filtered) > 0:
                    loss_multi_margin = self._multi_pair_margin_loss(logits_main, targets, pair_filtered, margin_filtered)
                    loss = loss + float(self.lambda_multi_pair_margin) * loss_multi_margin
                    loss_dict["multi_pair_margin"] = float(loss_multi_margin.detach().item())

            if self.enable_tf_head_loss and self.tf_head_enabled and float(self.lambda_tf_bce) > 0.0 and cls_out is not None and len(pair_ids_eff) == 2:
                tf_logit = self.tf_head(cls_out).view(-1)
                p0, p1 = pair_ids_eff
                mask = (targets == p0) | (targets == p1)
                if mask.any():
                    tf_targets = (targets[mask] == p0).float()
                    tf_loss = F.binary_cross_entropy_with_logits(tf_logit[mask], tf_targets)
                    loss = loss + float(self.lambda_tf_bce) * tf_loss
                    loss_dict["tf_bce"] = float(tf_loss.detach().item())

            logits_g = self.discriminator(Hc.view(B * C, H))
            loss_g = F.cross_entropy(logits_g, y_disc)
            loss = loss + float(self.lambda_gan) * loss_g
            loss_dict["gan"] = float(loss_g.detach().item())

            if float(self.lambda_brier) > 0.0:
                loss_brier = self._brier_loss(logits_main, targets)
                loss = loss + float(self.lambda_brier) * loss_brier
                loss_dict["brier"] = float(loss_brier.detach().item())

            if self.ctdm is not None and float(self.lambda_consistency) > 0.0 and not has_cached_potentials:
                obs_backup = int(self.ctdm.obs_minutes)
                min_c = int(getattr(self.ctdm, "cutoff_min", 10))
                max_c = int(getattr(self.ctdm, "cutoff_max", 30))
                t1 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
                t2 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
                self.ctdm.obs_minutes = t1
                logits_v1 = self(batch)
                self.ctdm.obs_minutes = t2
                logits_v2 = self(batch)
                self.ctdm.obs_minutes = obs_backup
                p1 = F.softmax(logits_v1, dim=1)
                p2 = F.softmax(logits_v2, dim=1)
                loss_cons = self._symmetric_kl(p1, p2)
                loss = loss + float(self.lambda_consistency) * loss_cons
                loss_dict["consistency"] = float(loss_cons.detach().item())

            if ctdm_trainable:
                if hasattr(self, "_last_potentials"):
                    Hc_cached = self._last_potentials  # [B, C, H]
                    idx = torch.arange(Hc_cached.size(0), device=Hc_cached.device)
                    h_y = Hc_cached[idx, targets, :]
                    if float(getattr(self.ctdm, "lambda_proto", 0.0)) > 0.0:
                        proto_loss = self.ctdm._compute_prototype_contrastive_loss(h_y, targets)
                        loss = loss + float(self.ctdm.lambda_proto) * proto_loss
                        self.ctdm.update_prototypes(h_y.detach(), targets.detach())
                        loss_dict["proto"] = float(proto_loss.detach().item())
                scores = getattr(self.ctdm, "_last_scores", None)
                A_obs = getattr(self.ctdm, "_last_Aobs_batch", None)
                X_obs_last = getattr(self.ctdm, "_last_xobs", None)
                if scores is not None and A_obs is not None and X_obs_last is not None:
                    topo_bce, topo_kl = self.ctdm._topology_regularizers(scores, A_obs, X_obs_last)
                    if float(getattr(self.ctdm, "lambda_topo_bce", 0.0)) > 0.0:
                        loss = loss + float(self.ctdm.lambda_topo_bce) * topo_bce
                        loss_dict["topo_bce"] = float(topo_bce.detach().item())
                    if float(getattr(self.ctdm, "lambda_deg_kl", 0.0)) > 0.0:
                        loss = loss + float(self.ctdm.lambda_deg_kl) * topo_kl
                        loss_dict["topo_kl"] = float(topo_kl.detach().item())

            if diffusion_loss is not None:
                loss = loss + float(self.lambda_diffusion) * diffusion_loss
                loss_dict["diffusion"] = float(diffusion_loss.detach().item())

            multiview_loss, multiview_details = self._multiview_regularization(batch, logits_main, targets)
            loss = loss + multiview_loss
            loss_dict.update(multiview_details)

            for p in self.discriminator.parameters():
                p.requires_grad = True
            self.discriminator.train()
            loss_dict["total"] = float(loss.detach().item())
            if self.loss_logging:
                self._last_loss_breakdown = loss_dict
            return loss, 0.0

        cls_out = None
        if self.arbiter_type == "transformer":
            tokens = self._build_tokens_from_batch(batch)
            logits, cls_out = self._encode_from_tokens(tokens)
            logits = self._apply_residual_branches(
                logits,
                cls_out,
                batch,
                targets=targets,
                update_prototypes=True,
            )
        else:
            logits = self(batch)
        loss_focal = self._la_focal_loss(logits, targets)
        loss = loss_focal
        loss_dict["la_focal"] = float(loss_focal.detach().item())

        if float(self.lambda_supcon) > 0.0 and cls_out is not None:
            loss_supcon = supervised_contrastive_loss(cls_out, targets, self.supcon_temperature)
            loss = loss + float(self.lambda_supcon) * loss_supcon
            loss_dict["supcon"] = float(loss_supcon.detach().item())

        pair_ids_eff = self._effective_pair_ids(self.pair_ids, int(self.num_classes))
        if self.enable_pair_margin_loss and self.pair_margin_enabled and float(self.lambda_pair_margin) > 0.0:
            if len(pair_ids_eff) == 2:
                loss_tf_margin = self._tf_margin_loss_static(logits, targets, pair_ids_eff, self.pair_margin_m)
                loss = loss + float(self.lambda_pair_margin) * loss_tf_margin
                loss_dict["pair_margin"] = float(loss_tf_margin.detach().item())

        if self.enable_multi_pair_margin and len(self.multi_pair_margin_list) > 0 and float(self.lambda_multi_pair_margin) > 0.0:
            pair_filtered = []
            margin_filtered = []
            for pair_ids, margin in zip(self.multi_pair_margin_list, self.multi_pair_margin_values):
                if len(pair_ids) == 2 and max(pair_ids) < int(self.num_classes):
                    pair_filtered.append(pair_ids)
                    margin_filtered.append(margin)
            if len(pair_filtered) > 0:
                loss_multi_margin = self._multi_pair_margin_loss(logits, targets, pair_filtered, margin_filtered)
                loss = loss + float(self.lambda_multi_pair_margin) * loss_multi_margin
                loss_dict["multi_pair_margin"] = float(loss_multi_margin.detach().item())

        if self.enable_tf_head_loss and self.tf_head_enabled and float(self.lambda_tf_bce) > 0.0 and cls_out is not None:
            p0, p1 = self.pair_ids
            mask = (targets == p0) | (targets == p1)
            if mask.any() and len(self._effective_pair_ids(self.pair_ids, int(self.num_classes))) == 2:
                tf_logit = self.tf_head(cls_out).view(-1)
                tf_targets = (targets[mask] == p0).float()
                tf_loss = F.binary_cross_entropy_with_logits(tf_logit[mask], tf_targets)
                loss = loss + float(self.lambda_tf_bce) * tf_loss
                loss_dict["tf_bce"] = float(tf_loss.detach().item())

        if float(self.lambda_brier) > 0.0:
            loss_brier = self._brier_loss(logits, targets)
            loss = loss + float(self.lambda_brier) * loss_brier
            loss_dict["brier"] = float(loss_brier.detach().item())

        if self.ctdm is not None and float(self.lambda_consistency) > 0.0 and not has_cached_potentials:
            obs_backup = int(self.ctdm.obs_minutes)
            min_c = int(getattr(self.ctdm, "cutoff_min", 10))
            max_c = int(getattr(self.ctdm, "cutoff_max", 30))
            t1 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
            t2 = int(torch.randint(low=min_c, high=max_c + 1, size=(1,)).item())
            self.ctdm.obs_minutes = t1
            logits_v1 = self(batch)
            self.ctdm.obs_minutes = t2
            logits_v2 = self(batch)
            self.ctdm.obs_minutes = obs_backup
            p1 = F.softmax(logits_v1, dim=1)
            p2 = F.softmax(logits_v2, dim=1)
            loss_cons = self._symmetric_kl(p1, p2)
            loss = loss + float(self.lambda_consistency) * loss_cons
            loss_dict["consistency"] = float(loss_cons.detach().item())

        if ctdm_trainable:
            if hasattr(self, "_last_potentials"):
                Hc = self._last_potentials  # [B, C, H]
                idx = torch.arange(Hc.size(0), device=Hc.device)
                h_y = Hc[idx, targets, :]
                if float(getattr(self.ctdm, "lambda_proto", 0.0)) > 0.0:
                    proto_loss = self.ctdm._compute_prototype_contrastive_loss(h_y, targets)
                    loss = loss + float(self.ctdm.lambda_proto) * proto_loss
                    self.ctdm.update_prototypes(h_y.detach(), targets.detach())
                    loss_dict["proto"] = float(proto_loss.detach().item())
            scores = getattr(self.ctdm, "_last_scores", None)
            A_obs = getattr(self.ctdm, "_last_Aobs_batch", None)
            X_obs_last = getattr(self.ctdm, "_last_xobs", None)
            if scores is not None and A_obs is not None and X_obs_last is not None:
                topo_bce, topo_kl = self.ctdm._topology_regularizers(scores, A_obs, X_obs_last)
                if float(getattr(self.ctdm, "lambda_topo_bce", 0.0)) > 0.0:
                    loss = loss + float(self.ctdm.lambda_topo_bce) * topo_bce
                    loss_dict["topo_bce"] = float(topo_bce.detach().item())
                if float(getattr(self.ctdm, "lambda_deg_kl", 0.0)) > 0.0:
                    loss = loss + float(self.ctdm.lambda_deg_kl) * topo_kl
                    loss_dict["topo_kl"] = float(topo_kl.detach().item())

        if diffusion_loss is not None:
            loss = loss + float(self.lambda_diffusion) * diffusion_loss
            loss_dict["diffusion"] = float(diffusion_loss.detach().item())
        multiview_loss, multiview_details = self._multiview_regularization(batch, logits, targets)
        loss = loss + multiview_loss
        loss_dict.update(multiview_details)
        loss_dict["total"] = float(loss.detach().item())
        if self.loss_logging:
            self._last_loss_breakdown = loss_dict
        return loss, 0.0
