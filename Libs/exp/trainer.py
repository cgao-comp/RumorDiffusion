import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import VGAE, GCNConv
from tqdm.auto import tqdm

from Libs.models.baselines.TGN import TGN
from Libs.utils.metrics import metrics_from_logits

CHECKPOINT_STATE_KEY = "state_dict"
CHECKPOINT_METADATA_KEY = "metadata"


def build_checkpoint_payload(
    state_dict: Dict[str, Tensor],
    metadata: Optional[Dict[str, Union[str, int, float, bool, Dict[str, int]]]] = None,
) -> Dict[str, object]:
    """Builds a checkpoint payload with explicit metadata."""
    payload: Dict[str, object] = {
        CHECKPOINT_STATE_KEY: {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        }
    }
    if metadata is not None:
        payload[CHECKPOINT_METADATA_KEY] = dict(metadata)
    return payload


def unpack_checkpoint_payload(payload: object) -> Tuple[Dict[str, Tensor], Dict[str, object]]:
    """Returns `(state_dict, metadata)` from either payload or legacy state dict."""
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload)}")
    if CHECKPOINT_STATE_KEY in payload:
        state_dict = payload[CHECKPOINT_STATE_KEY]
        metadata = payload.get(CHECKPOINT_METADATA_KEY, {})
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint state_dict must be a dict")
        if not isinstance(metadata, dict):
            raise TypeError("Checkpoint metadata must be a dict")
        return state_dict, metadata
    return payload, {}


@dataclass
class TrainerConfig:
    """Holds training hyperparameters and settings."""

    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_clip: float
    amp: bool
    early_stopping_patience: int
    num_workers: int
    device: str
    use_tqdm: bool = True
    class_weights: Optional[List[float]] = None
    max_batches_per_epoch: Optional[int] = None
    min_epochs: int = 30
    selection_metric: str = "macro_f1"
    selection_mode: str = "classification"
    fail_on_nonfinite: bool = False


class Trainer:
    """Generic trainer for graph classification baselines.

    This trainer supports AMP, gradient clipping, early stopping and returns the
    best validation metrics. The model is assumed to output raw logits.
    """

    def __init__(self, model: nn.Module, cfg: TrainerConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        if cfg.class_weights is not None:
            w = torch.tensor(cfg.class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.epochs)
        )
        self.scaler = GradScaler(enabled=cfg.amp) if cfg.device == 'cuda' else None

    def _raise_if_nonfinite(self, tensors, stage: str) -> None:
        """Fails closed before a non-finite training state reaches a checkpoint."""
        if not bool(getattr(self.cfg, "fail_on_nonfinite", False)):
            return
        invalid = [
            name for name, tensor in tensors
            if tensor is not None and not bool(torch.isfinite(tensor).all().item())
        ]
        if invalid:
            raise FloatingPointError(
                f"Non-finite {stage} tensors: {', '.join(invalid[:8])}"
            )

    def _step(self, batch: Batch, train: bool) -> Tuple[float, float]:
        batch = batch.to(self.device)
        if train:
            self.model.train()
        else:
            self.model.eval()

        logits: Optional[Tensor] = None
        if self.cfg.device == 'cuda' and self.cfg.amp:
            with autocast():
                if train and hasattr(self.model, 'training_step'):
                    loss, _ = self.model.training_step(batch)
                else:
                    logits = self.model(batch)
                    loss = self.criterion(logits, batch.y.view(-1))
        else:
            if train and hasattr(self.model, 'training_step'):
                loss, _ = self.model.training_step(batch)
            else:
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y.view(-1))
        if train:
            self._raise_if_nonfinite([("loss", loss)], stage="loss")
            self.optimizer.zero_grad(set_to_none=True)
            if self.cfg.amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._raise_if_nonfinite(
                    ((name, param.grad) for name, param in self.model.named_parameters() if param.requires_grad),
                    stage="gradient",
                )
                if self.cfg.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._raise_if_nonfinite(
                    ((name, param.grad) for name, param in self.model.named_parameters() if param.requires_grad),
                    stage="gradient",
                )
                if self.cfg.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            self._raise_if_nonfinite(
                ((name, param) for name, param in self.model.named_parameters() if param.requires_grad),
                stage="parameter",
            )

        if logits is None:
            with torch.no_grad():
                logits = self.model(batch)
        with torch.no_grad():
            metrics = metrics_from_logits(logits, batch.y.view(-1), num_classes=logits.size(1))
        return float(loss.item()), float(metrics["acc"])

    def _run_epoch(self, loader: DataLoader, train: bool, desc: str = "") -> Dict[str, float]:
        total_loss = 0.0
        total_correct = 0.0
        total = 0
        all_logits = []
        all_targets = []
        iterator = loader
        if self.cfg.use_tqdm:
            iterator = tqdm(loader, desc=desc, leave=False)
        max_steps = self.cfg.max_batches_per_epoch
        for step_idx, batch in enumerate(iterator):
            loss, acc = self._step(batch, train=train)
            bsz = batch.y.size(0)
            total_loss += loss * bsz
            total_correct += acc * bsz
            total += bsz
            if not train:
                with torch.no_grad():
                    logits = self.model(batch.to(self.device))
                    all_logits.append(logits.detach().cpu())
                    all_targets.append(batch.y.view(-1).cpu())
            if max_steps is not None and step_idx + 1 >= int(max_steps):
                break
        out = {
            "loss": total_loss / max(1, total),
            "acc": total_correct / max(1, total),
        }
        if not train:
            if len(all_logits) > 0 and len(all_targets) > 0:
                logits_cat = torch.cat(all_logits, dim=0)
                targets_cat = torch.cat(all_targets, dim=0)
                m = metrics_from_logits(logits_cat, targets_cat, num_classes=logits_cat.size(1))
                out["acc"] = m["acc"]
                out["macro_f1"] = m["macro_f1"]
        return out

    def _run_objective_eval_epoch(self, loader: DataLoader, desc: str = "") -> Dict[str, float]:
        total_loss = 0.0
        total = 0
        iterator = loader
        if self.cfg.use_tqdm:
            iterator = tqdm(loader, desc=desc, leave=False)
        max_steps = self.cfg.max_batches_per_epoch
        self.model.eval()
        with torch.no_grad():
            for step_idx, batch in enumerate(iterator):
                batch = batch.to(self.device)
                if hasattr(self.model, "validation_step"):
                    loss = self.model.validation_step(batch)
                elif hasattr(self.model, "training_step"):
                    loss, _ = self.model.training_step(batch)
                else:
                    raise AttributeError("Objective validation requires `validation_step` or `training_step`.")
                bsz = batch.y.size(0)
                total_loss += float(loss.item()) * bsz
                total += bsz
                if max_steps is not None and step_idx + 1 >= int(max_steps):
                    break
        return {"objective_loss": total_loss / max(1, total)}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[str] = None,
        checkpoint_metadata: Optional[Dict[str, Union[str, int, float, bool, Dict[str, int]]]] = None,
    ) -> Dict[str, float]:
        selection_mode = str(self.cfg.selection_mode).lower()
        selection_metric = str(self.cfg.selection_metric).lower()
        maximize_metric = selection_mode != "objective"
        best_val_metric = -1.0 if maximize_metric else float("inf")
        best_state = None
        patience = 0
        min_epochs = max(1, int(self.cfg.min_epochs))

        epoch_iter = range(self.cfg.epochs)
        if self.cfg.use_tqdm:
            epoch_iter = tqdm(epoch_iter, desc="epochs")
        for epoch in epoch_iter:
            if hasattr(self.model, "set_training_progress"):
                self.model.set_training_progress(epoch, self.cfg.epochs)
            self._run_epoch(train_loader, train=True, desc=f"train {epoch}")
            self.scheduler.step()
            if selection_mode == "objective":
                val_metrics = self._run_objective_eval_epoch(val_loader, desc=f"val objective {epoch}")
                current_metric = float(val_metrics.get(selection_metric, val_metrics.get("objective_loss", float("inf"))))
                improved = current_metric < best_val_metric
            else:
                val_metrics = self._run_epoch(val_loader, train=False, desc=f"val {epoch}")
                current_metric = float(val_metrics.get(selection_metric, val_metrics.get("macro_f1", -1.0)))
                improved = current_metric > best_val_metric

            if improved:
                best_val_metric = current_metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
                if save_path is not None:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    checkpoint_obj: object = best_state
                    if checkpoint_metadata is not None:
                        checkpoint_obj = build_checkpoint_payload(
                            best_state,
                            checkpoint_metadata,
                        )
                    torch.save(checkpoint_obj, save_path)
            else:
                patience += 1
                # Only allow early stopping after minimum epochs
                if epoch >= min_epochs and patience >= self.cfg.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        result = {"best_val_metric": best_val_metric}
        if selection_mode == "objective":
            result["best_val_objective_loss"] = best_val_metric
        else:
            result["best_val_macro_f1"] = best_val_metric
        return result

    def evaluate(self, loader: DataLoader, detailed: bool = False, class_names: Optional[Dict[int, str]] = None) -> Dict[str, Union[float, str, Dict]]:
        """Evaluates the model on the given data loader.
        
        Args:
            loader: DataLoader to evaluate on.
            detailed: If True, compute and return comprehensive per-class metrics.
            class_names: Optional mapping from class index to name for formatted report.
        
        Returns:
            Dictionary of metrics. Always includes 'acc', 'macro_f1', 'loss'.
            If detailed=True, additionally includes per-class metrics, weighted averages,
            confusion matrix, and formatted report string.
        """
        if not detailed:
            return self._run_epoch(loader, train=False)
        
        self.model.eval()
        all_logits = []
        all_targets = []
        total_loss = 0.0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y.view(-1))
                
                all_logits.append(logits.cpu())
                all_targets.append(batch.y.view(-1).cpu())
                
                bsz = batch.y.size(0)
                total_loss += loss.item() * bsz
                total += bsz
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        num_classes = all_logits.size(1)
        metrics = metrics_from_logits(all_logits, all_targets, num_classes, detailed=True, class_names=class_names)
        metrics['loss'] = total_loss / max(1, total)
        
        return metrics



@dataclass
class LinkPredConfig:
    """Hyperparameters for supervised link prediction training."""

    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_clip: float
    amp: bool
    early_stopping_patience: int
    num_workers: int
    device: str
    heads: int
    hidden: int
    num_layers: int
    dropout: float
    topk_per_node: int


class LinkPredTrainer:
    """Trainer for link prediction using a TGN backbone.

    This trainer optimizes a BCE objective on positive (observed) and
    uniformly-sampled negative directed edges, supports AMP, gradient clipping,
    cosine annealing, and early stopping. It also provides edge generation APIs
    to export top-K edges per node into a directory recognized by GraphDataset.
    """

    def __init__(self, model: nn.Module, cfg: LinkPredConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, cfg.epochs))
        self.scaler = GradScaler(enabled=cfg.amp) if cfg.device == 'cuda' else None

    @staticmethod
    def _sample_negatives(num_nodes: int, pos_edge_index: Tensor, num_samples: int, device: torch.device) -> Tensor:
        assert pos_edge_index.dim() == 2 and pos_edge_index.size(0) == 2
        if num_nodes <= 1:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        pos = pos_edge_index.to(torch.long)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        if pos.numel() > 0:
            adj[pos[0], pos[1]] = True
        mask = ~adj
        mask.fill_diagonal_(False)
        src, dst = torch.where(mask)
        if src.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        k = min(int(num_samples), src.numel())
        perm = torch.randperm(src.numel(), device=device)[:k]
        return torch.stack([src[perm], dst[perm]], dim=0)

    def _step(self, data: Batch) -> float:
        data = data.to(self.device)
        if self.cfg.amp and self.cfg.device == 'cuda':
            with autocast():
                z = self.model(data)
                pos = data.edge_index
                num_pos = pos.size(1)
                neg = self._sample_negatives(int(data.num_nodes), pos, num_pos, self.device)
                if num_pos == 0 and neg.size(1) == 0:
                    return 0.0  # Skip if no edges
                lbl = torch.cat([
                    torch.ones(num_pos, device=self.device),
                    torch.zeros(neg.size(1), device=self.device),
                ], dim=0)
                logits = torch.cat([
                    self.model.link_logits(z, pos),
                    self.model.link_logits(z, neg),
                ], dim=0)
                loss = self.criterion(logits, lbl)
        else:
            z = self.model(data)
            pos = data.edge_index
            num_pos = pos.size(1)
            neg = self._sample_negatives(int(data.num_nodes), pos, num_pos, self.device)
            if num_pos == 0 and neg.size(1) == 0:
                return 0.0  # Skip if no edges
            lbl = torch.cat([
                torch.ones(num_pos, device=self.device),
                torch.zeros(neg.size(1), device=self.device),
            ], dim=0)
            logits = torch.cat([
                self.model.link_logits(z, pos),
                self.model.link_logits(z, neg),
            ], dim=0)
            loss = self.criterion(logits, lbl)

        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0  # Skip NaN/Inf losses

        self.optimizer.zero_grad(set_to_none=True)
        if self.cfg.amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.cfg.grad_clip > 0:
                clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
        return float(loss.item())

    def fit(self, loader: DataLoader, save_path: Optional[str], epochs: int, early_stopping_patience: int) -> Dict[str, float]:
        best = float('inf')
        best_state = None
        patience = 0

        iterator = tqdm(range(epochs), desc="lp-epochs") if epochs > 1 else range(epochs)
        for _ in iterator:
            total = 0.0
            n = 0
            for batch in loader:
                total += self._step(batch)
                n += 1
            avg = total / max(1, n)
            if isinstance(iterator, tqdm):
                iterator.set_postfix(loss=f"{avg:.4f}")
            self.scheduler.step()

            if avg + 1e-6 < best:
                best = avg
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
                if save_path is not None:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(best_state, save_path)
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_loss": best}

    @staticmethod
    def generate_edges_for_split(model: TGN, ds: torch.utils.data.Dataset, topk_per_node: int, out_dir: str, device: torch.device, num_workers: int) -> None:
        from torch_geometric.loader import DataLoader as PyGDataLoader
        loader = PyGDataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for data in tqdm(loader, desc="gen-edges"):
                data = data.to(device)
                z = model(data)
                existing = data.edge_index
                num_nodes = int(data.num_nodes)
                cand = TGN.all_missing_directed_pairs(num_nodes, existing, device)
                if cand.numel() == 0:
                    continue
                scores = model.link_logits(z, cand)
                k = min(int(topk_per_node) * num_nodes, scores.numel())
                if k == 0:
                    continue
                topk = torch.topk(scores, k=k, largest=True).indices
                new_edges = cand[:, topk].detach().cpu()
                sid = str(data.source_id[0]) if hasattr(data, "source_id") and isinstance(data.source_id, torch.Tensor) else str(data.source_id)
                torch.save({"edge_index": new_edges}, os.path.join(out_dir, f"{sid}.pt"))


class GCNEncoderVAE(nn.Module):
    """Improved GCN encoder with multiple layers, dropout, and batch normalization."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        assert in_channels > 0 and hidden_channels > 0 and latent_channels > 0
        assert num_layers >= 2
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        
        # Build encoder layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layers for mu and logvar
        self.conv_mu = GCNConv(hidden_channels, latent_channels)
        self.conv_logvar = GCNConv(hidden_channels, latent_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            # Skip BatchNorm if batch size is too small (<=1 nodes)
            if self.use_bn and self.bns is not None and h.size(0) > 1:
                h = self.bns[i](h)
            h = torch.relu(h)
            h = torch.nn.functional.dropout(h, p=self.dropout, training=self.training)
        
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar


@dataclass
class CGVAEConfig:
    """Hyperparameters for class-conditional VGAE training and inference."""

    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_clip: float
    amp: bool
    num_workers: int
    early_stopping_patience: int
    device: str
    hidden: int
    latent: int
    num_layers: int = 3
    dropout: float = 0.3
    use_bn: bool = True
    kl_anneal: bool = True


class CGVAETrainer:
    """Trains one VGAE per class and classifies graphs via reconstruction score."""

    def __init__(self, cfg: CGVAEConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def _build_vgae(self, in_channels: int, hidden: int, latent: int) -> VGAE:
        enc = GCNEncoderVAE(
            in_channels=in_channels, 
            hidden_channels=hidden, 
            latent_channels=latent,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            use_bn=self.cfg.use_bn
        )
        return VGAE(enc)

    def _loss_components(self, model: VGAE, data: Batch) -> Tuple[Tensor, Tensor]:
        data = data.to(self.device)
        z = model.encode(data.x, data.edge_index)
        pos = model.recon_loss(z, data.edge_index)
        kl = (1.0 / data.num_nodes) * model.kl_loss()
        
        # Check for NaN/Inf and return zeros if found
        if torch.isnan(pos) or torch.isinf(pos):
            pos = torch.tensor(0.0, device=self.device, requires_grad=True)
        if torch.isnan(kl) or torch.isinf(kl):
            kl = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return pos, kl

    def train_per_class(self, class_id: int, ds: torch.utils.data.Dataset, ckpt_path: str, num_features: int) -> None:
        print(f"Starting training for class {class_id}")
        class_samples = []
        for local_idx in range(len(ds)):
            sample = ds.get(local_idx)
            sample_class = int(sample.y.view(-1)[0].item())
            if sample_class == class_id:
                class_samples.append(sample)
        
        print(f"Found {len(class_samples)} samples for class {class_id}")
        assert len(class_samples) > 0, f"No samples for class {class_id}"
        from torch_geometric.loader import DataLoader as PyGDataLoader
        print(f"Creating data loader for class {class_id}")
        loader = PyGDataLoader(class_samples, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        print(f"Building VGAE model for class {class_id}")
        model = self._build_vgae(num_features, self.cfg.hidden, self.cfg.latent).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

        # KL annealing setup
        kl_weight = 0.0
        kl_anneal_rate = 1.0 / (self.cfg.epochs * 0.5) if self.cfg.kl_anneal else 1.0  # Reach 1.0 at 50% epochs

        best = float('inf')
        best_state = None
        patience = 0
        iterator = range(self.cfg.epochs)
        for epoch in iterator:
            total = 0.0
            n = 0
            for data in loader:
                pos, kl = self._loss_components(model, data)
                loss = pos + kl_weight * kl  # Weighted KL loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip > 0:
                    clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                opt.step()
                total += float(loss.item())
                n += 1
            
            # Update KL weight
            if self.cfg.kl_anneal:
                kl_weight = min(1.0, kl_weight + kl_anneal_rate)
            
            sch.step(total / max(1, n))  # Use average loss for scheduler
            avg = total / max(1, n)
            print(f"Class {class_id}, Epoch {epoch+1}/{self.cfg.epochs}, Loss: {avg:.4f}, KL_weight: {kl_weight:.3f}")
            if avg + 1e-6 < best:
                best = avg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(best_state, ckpt_path)
            else:
                patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

    def train_all_classes(self, ds_train: torch.utils.data.Dataset, ckpt_root: str, num_classes: int, num_features: int) -> None:
        print(f"Starting train_all_classes with {num_classes} classes")
        print(f"Dataset has {len(ds_train)} samples")
        for k in range(num_classes):
            print(f"Processing class {k}/{num_classes-1}")
            ckpt_path = os.path.join(ckpt_root, f"class_{k}", "best.pt")
            self.train_per_class(k, ds_train, ckpt_path, num_features)

    def _gaussian_likelihood(self, z: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute Gaussian likelihood of z given mu and logvar."""
        # Compute log probability density for each node
        log_prob = -0.5 * (torch.sum(logvar, dim=-1) + 
                           torch.sum(((z - mu) ** 2) / torch.exp(logvar), dim=-1) + 
                           z.size(-1) * torch.log(torch.tensor(2 * torch.pi, device=z.device)))
        # Return mean log probability across all nodes in the graph
        return torch.mean(log_prob)

    def logits_for_dataset(self, ds: torch.utils.data.Dataset, ckpt_root: str, num_classes: int, num_features: int) -> Tensor:
        device = self.device
        models: List[VGAE] = []
        for k in range(num_classes):
            m = self._build_vgae(num_features, self.cfg.hidden, self.cfg.latent).to(device)
            ckpt_path = os.path.join(ckpt_root, f"class_{k}", "best.pt")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            m.load_state_dict(state)
            m.eval()
            models.append(m)

        from torch_geometric.loader import DataLoader as PyGDataLoader
        loader = PyGDataLoader(ds, batch_size=1, shuffle=False, num_workers=self.cfg.num_workers)
        out: List[Tensor] = []
        with torch.no_grad():
            for data in tqdm(loader, desc="cgvae-infer"):
                scores: List[float] = []
                for k in range(num_classes):
                    # 1. Reconstruction-based score (ELBO)
                    pos, kl = self._loss_components(models[k], data)
                    elbo = -(pos + kl)
                    
                    # 2. Latent space density (Gaussian likelihood)
                    z = models[k].encode(data.x, data.edge_index)
                    mu, logvar = models[k].encoder(data.x, data.edge_index)
                    latent_score = self._gaussian_likelihood(z, mu, logvar)
                    
                    # 3. Combined score (weighted combination)
                    combined_score = 0.7 * elbo + 0.3 * latent_score
                    scores.append(float(combined_score))
                out.append(torch.tensor(scores, dtype=torch.float32))
        return torch.stack(out, dim=0)

    def evaluate(self, ds: torch.utils.data.Dataset, ckpt_root: str, num_classes: int, num_features: int) -> Dict[str, Union[float, str, Dict]]:
        logits = self.logits_for_dataset(ds, ckpt_root, num_classes, num_features)
        targets = torch.tensor([int(ds.get(i).y.view(-1)[0].item()) for i in ds.indices()])
        metrics = metrics_from_logits(logits, targets, num_classes, detailed=True)
        # Add loss field for compatibility
        metrics['loss'] = 0.0  # CGVAE doesn't have a traditional loss during inference
        return metrics

    def generate_edges_for_split(
        self, 
        ds: torch.utils.data.Dataset, 
        ckpt_root: str, 
        num_classes: int, 
        num_features: int,
        output_dir: str,
        split_name: str
    ) -> None:
        """Generate augmented edges for a dataset split using the best-class CVAE reconstruction.
        
        For each graph:
        1. Use all class models to compute reconstruction scores
        2. Select the best class model (highest ELBO score)
        3. Use that model to reconstruct and augment the graph structure
        4. Save augmented edges to disk
        
        Args:
            ds: Dataset to generate edges for
            ckpt_root: Root directory containing class model checkpoints
            num_classes: Number of classes
            num_features: Number of input features
            output_dir: Directory to save augmented edges
            split_name: Name of the split (e.g., 'train', 'val', 'test')
        """
        device = self.device
        
        # Load all class models
        models: List[VGAE] = []
        for k in tqdm(range(num_classes), desc=f"Loading CVAE models for {split_name}"):
            m = self._build_vgae(num_features, self.cfg.hidden, self.cfg.latent).to(device)
            ckpt_path = os.path.join(ckpt_root, f"class_{k}", "best.pt")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            m.load_state_dict(state)
            m.eval()
            models.append(m)
        
        from torch_geometric.loader import DataLoader as PyGDataLoader
        loader = PyGDataLoader(ds, batch_size=1, shuffle=False, num_workers=self.cfg.num_workers)
        
        all_aug_edges: List[Tensor] = []
        
        with torch.no_grad():
            for data in tqdm(loader, desc=f"cgvae-aug-{split_name}"):
                data = data.to(device)
                
                # Find best class model for this graph
                best_score = float('-inf')
                best_model_idx = 0
                
                for k in range(num_classes):
                    pos, kl = self._loss_components(models[k], data)
                    elbo = -(pos + kl)
                    score = float(elbo)
                    if score > best_score:
                        best_score = score
                        best_model_idx = k
                
                # Use best model to reconstruct graph
                best_model = models[best_model_idx]
                z = best_model.encode(data.x, data.edge_index)
                
                # Compute reconstruction probability for all possible edges
                adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
                
                # Create augmented edges: select top-K edges not in original graph
                num_nodes = data.x.size(0)
                existing_edges = set()
                for i in range(data.edge_index.size(1)):
                    src, dst = int(data.edge_index[0, i]), int(data.edge_index[1, i])
                    existing_edges.add((src, dst))
                
                # Get all candidate edges
                candidates = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j and (i, j) not in existing_edges:
                            candidates.append((i, j, float(adj_pred[i, j])))
                
                # Sort by probability and select top edges
                candidates.sort(key=lambda x: x[2], reverse=True)
                topk = min(len(candidates), num_nodes * 2)  # Add up to 2*num_nodes new edges
                
                aug_edges = [[c[0], c[1]] for c in candidates[:topk]]
                
                if len(aug_edges) > 0:
                    all_aug_edges.append(torch.tensor(aug_edges, dtype=torch.long))
                else:
                    # If no edges to add, append empty tensor
                    all_aug_edges.append(torch.tensor([], dtype=torch.long).reshape(0, 2))
        
        # Save to disk
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{split_name}.pt")
        torch.save(all_aug_edges, output_path)
        logging.info(f"Saved augmented edges to {output_path}")
