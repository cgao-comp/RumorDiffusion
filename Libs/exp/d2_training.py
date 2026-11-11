"""Train the independent D2 forecaster, search architecture, then refit weights.

The three stages use real train/validation cascades. This module never opens the
report split. A diagnostic follows the same stages with an explicitly bounded
real subset and cannot publish canonical evidence.
"""
from __future__ import annotations

import copy
import math
import random
import time
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from Libs.data.d2 import D2Example, ObservedGraph, adapt_graph
from Libs.data.dataset import GraphDataset
from Libs.models.baselines.D2 import D2Classifier, D2Forecaster, SEARCH_SPACE as MODEL_SEARCH_SPACE
from Libs.utils.metrics import metrics_from_logits
from Libs.utils.provenance import write_json
from Libs.utils.seed import set_global_seed

SEEDS = (41, 42, 43)
DATASETS = ("pheme", "weibo", "twitter15", "twitter16")
SEARCH_SPACE = {key: list(values) for key, values in MODEL_SEARCH_SPACE.items()}


def observation_windows(dataset: str) -> tuple[str, int, tuple[int, ...]]:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return ("minutes", 60, (10, 20, 30)) if dataset.startswith("twitter") else ("hop", 3, (1, 2, 3))


class WindowData:
    """Verified immutable dataset views; labels/targets stay outside the model."""
    def __init__(self, root: Path, split: str, dataset: str, *, targets: bool,
                 limit: int = 0, text_config: dict | None = None) -> None:
        if targets and split == "test":
            raise ValueError("Test completion targets are prohibited")
        self.root, self.split, self.targets = root, split, targets
        self.unit, self.main_window, early = observation_windows(dataset)
        self.windows = tuple(sorted(set((self.main_window, *early))))
        text_config = text_config or {}
        self.datasets = {
            window: GraphDataset(
                root=str(root), split=split, use_lp_edges=False,
                prefix_cfg={"strategy": self.unit, "unit": self.unit, "window": window,
                            "include_root": True},
                observability_cfg={"drop_node": 0.0, "drop_edge": 0.0, "seed": 42},
                mode="train_ctdm" if targets else "eval_prefix",
                text_cache_name=text_config.get("cache_name", "text_embs_mpnet_canonical_v1.pt"),
                force_text_cache=True, text_normalize=True,
            ) for window in self.windows
        }
        sizes = {len(data) for data in self.datasets.values()}
        if len(sizes) != 1 or not next(iter(sizes)):
            raise ValueError("Observation views must cover the same nonempty split")
        self.full_size = next(iter(sizes))
        self.size = min(limit, self.full_size) if limit else self.full_size
        self.num_classes = len(set(next(iter(self.datasets.values())).label_schema.values()))
        if sorted(set(next(iter(self.datasets.values())).label_schema.values())) != list(range(self.num_classes)):
            raise ValueError("Class schema must be contiguous and fixed")

    def example(self, index: int, window: int) -> D2Example:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return adapt_graph(self.datasets[window][index], self.split, self.unit, window,
                           with_targets=self.targets)

    def capacity(self) -> int:
        if self.split != "train" or not self.targets:
            raise ValueError("Completion capacity must come from training targets")
        # The smallest registered prefix has the largest completion of each source.
        data = self.datasets[min(self.windows)]
        maximum = 0
        for i in range(self.size):
            sample = data[i]
            maximum = max(maximum, int(sample.full_x.size(0) - sample.x.size(0)))
        return maximum

    def batches(self, rng: random.Random, batch_size: int) -> Iterator[list[D2Example]]:
        order = list(range(self.size))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            yield [self.example(index, rng.choice(self.windows))
                   for index in order[start:start + batch_size]]


class CompletionCache:
    """Bounded in-memory cache for a frozen deterministic forecaster."""
    def __init__(self, forecaster: D2Forecaster, budget_mib: int) -> None:
        self.forecaster = forecaster
        self.budget = int(budget_mib) * 1024 * 1024
        self.used = 0
        self.cache: OrderedDict[tuple, tuple[ObservedGraph, int]] = OrderedDict()

    @torch.no_grad()
    def get(self, example: D2Example) -> ObservedGraph:
        if self.forecaster.training or any(p.requires_grad for p in self.forecaster.parameters()):
            raise ValueError("Completion cache requires an eval-mode frozen forecaster")
        key = (example.source_id, example.observed.unit, example.observed.horizon)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value[0]
        completed = self.forecaster.complete(example.observed, self.forecaster(example.observed)).detach()
        size = sum(value.numel() * value.element_size()
                   for field in fields(completed)
                   if isinstance(value := getattr(completed, field.name), Tensor))
        if 0 < size <= self.budget:
            while self.cache and self.used + size > self.budget:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.used -= old_size
            self.cache[key] = (completed, size)
            self.used += size
        return completed


def _snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _step(loss: Tensor, optimizer: torch.optim.Optimizer, parameters: list[nn.Parameter],
          clip: float) -> float:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Training loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Training gradient is not finite")
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def forecast_validation(model: D2Forecaster, views: WindowData, weights: dict) -> dict:
    """Score predicted completions against all validation targets, including overflow.

    The graph is generated before targets are inspected. Count and edge errors
    penalize all missing/extra nodes. Identity/coordinate errors use aligned slots;
    a missing predicted known author is an error. No true target is truncated to
    training capacity and no validation capacity changes the model.
    """
    model.eval()
    windows = {}
    for window in views.windows:
        totals = {"count": 0.0, "edge": 0.0, "author": 0.0, "coordinate": 0.0,
                  "count_mae": 0.0, "target_capacity_overflow": 0}
        for index in range(views.size):
            example = views.example(index, window)
            observed = example.observed
            prediction = model(observed)
            completed = model.complete(observed, prediction)
            target = example.target
            if target is None:
                raise ValueError("Forecaster validation needs separate real targets")
            actual, predicted = target.count, completed.num_posts - observed.num_posts
            totals["count"] += (float(prediction.count_log1p) - math.log1p(actual)) ** 2
            totals["count_mae"] += abs(predicted - actual)
            totals["target_capacity_overflow"] += int(actual > model.max_completion_nodes)
            n = observed.num_posts
            truth = {(int(src), int(dst)) for src, dst in target.graph_edges.t().tolist()}
            estimate = {(int(src), int(dst)) for src, dst in completed.edge_index.t().tolist() if src >= n or dst >= n}
            denominator = len(truth) + len(estimate)
            totals["edge"] += 1.0 - (2.0 * len(truth & estimate) / denominator if denominator else 1.0)
            known = target.author >= 0
            correct = 0
            aliases: dict[int, int] = {}
            for slot in range(predicted):
                actor = int(completed.post_author[n + slot])
                if actor >= observed.num_users:
                    actor = aliases.setdefault(actor, observed.num_users + slot)
                if slot < actual and bool(known[slot]) and actor == int(target.author[slot]):
                    correct += 1
            if int(known.sum()):
                totals["author"] += 1.0 - correct / int(known.sum())
            common = min(actual, predicted)
            if common:
                totals["coordinate"] += float(F.l1_loss(
                    torch.log1p(completed.coordinates[n:n + common].float()),
                    torch.log1p(target.coordinates[:common].float())))
        windows[str(window)] = {key: value / views.size if key != "target_capacity_overflow" else value
                                for key, value in totals.items()}
    score = sum(sum(weights[key] * values[key] for key in weights) for values in windows.values()) / len(windows)
    if not math.isfinite(score):
        raise FloatingPointError("Forecaster validation score is not finite")
    return {"score": score, "windows": windows,
            "criterion": "weighted_log_count_error_edge_F1_loss_author_error_log_coordinate_MAE"}


@torch.no_grad()
def classifier_outputs(model: D2Classifier, views: WindowData, cache: CompletionCache,
                       window: int, *, measure_latency: bool = False) -> dict:
    model.eval()
    ids, logits, labels, latency = [], [], [], []
    for index in range(views.size):
        example = views.example(index, window)
        started = time.perf_counter()
        completed = cache.get(example)
        output = model(example.observed, completed)
        elapsed = (time.perf_counter() - started) * 1000.0
        if output.shape != (1, views.num_classes) or not bool(torch.isfinite(output).all()):
            raise ValueError("Classifier output shape or values are invalid")
        ids.append(example.source_id)
        labels.append(example.label)
        logits.append(output.detach().cpu())
        if measure_latency:
            latency.append(elapsed)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sources in classifier evaluation")
    return {"source_ids": ids, "logits": torch.cat(logits),
            "targets": torch.tensor(labels, dtype=torch.long), "latency_ms": latency}


def classifier_validation(model: D2Classifier, views: WindowData, cache: CompletionCache) -> dict:
    results = {}
    for window in views.windows:
        outputs = classifier_outputs(model, views, cache, window)
        results[str(window)] = metrics_from_logits(outputs["logits"], outputs["targets"], views.num_classes)
    return {"score": sum(float(row["macro_f1"]) for row in results.values()) / len(results),
            "windows": results, "criterion": "mean_window_macro_f1"}


def _finish_stage(model: nn.Module, best: dict | None, history: list, best_epoch: int,
                  reason: str, path: Path, metadata: dict) -> dict:
    if best is None:
        raise ValueError("Stage produced no finite validated checkpoint")
    model.load_state_dict(best)
    record = {"history": history, "best_epoch": best_epoch, "epochs_run": len(history),
              "stop_reason": reason, "metadata": metadata}
    save_checkpoint(path, {"state_dict": best, "stage_record": record})
    write_json(path.with_suffix(".json"), record)
    return record


def fit_forecaster(model: D2Forecaster, train: WindowData, val: WindowData, config: dict,
                   seed: int, output: Path, log: Callable[[dict], None]) -> dict:
    settings = config["training"]
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    rng = random.Random(seed)
    weights = config["forecast"]["loss_weights"]
    history, best, best_score, best_epoch, stale = [], None, math.inf, 0, 0
    reason = "epoch_budget"
    for epoch in range(1, settings["forecast_epochs"] + 1):
        started, loss_sum, count = time.perf_counter(), 0.0, 0
        model.train()
        for batch in train.batches(rng, settings["batch_size"]):
            losses = []
            for example in batch:
                if example.target is None:
                    raise ValueError("Training completion target is missing")
                losses.append(model.loss(model(example.observed), example.target)["total"])
            loss = _step(torch.stack(losses).mean(), optimizer, parameters, settings["grad_clip"])
            loss_sum += loss * len(batch)
            count += len(batch)
        validation = forecast_validation(model, val, weights)
        row = {"stage": "forecaster", "epoch": epoch, "train_loss": loss_sum / count,
               "validation": validation, "seconds": time.perf_counter() - started}
        history.append(row)
        log(row)
        if validation["score"] < best_score:
            best, best_score, best_epoch, stale = _snapshot(model), validation["score"], epoch, 0
            save_checkpoint(output / "forecaster_best.pt", {"state_dict": best, "epoch": epoch})
        else:
            stale += 1
        write_json(output / "forecaster_history.json", {"history": history})
        if epoch >= settings["min_epochs"] and stale >= settings["early_stopping_patience"]:
            reason = "validation_patience"
            break
    return _finish_stage(model, best, history, best_epoch, reason, output / "forecaster.pt",
                         {"training_split": "train", "selection_split": "val", "labels_used": False})


def fit_classifier(model: D2Classifier, train: WindowData, val: WindowData,
                   cache: CompletionCache, config: dict, seed: int, output: Path,
                   log: Callable[[dict], None], *, search: bool) -> dict:
    settings = config["training"]
    parameters, architecture = list(model.weight_parameters()), list(model.architecture_parameters())
    if bool(architecture) != search or {id(p) for p in parameters} & {id(p) for p in architecture}:
        raise ValueError("Search requires disjoint network and architecture parameter sets")
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    architecture_optimizer = torch.optim.Adam(architecture, lr=settings["architecture_learning_rate"]) if search else None
    rng = random.Random(seed)
    val_order = [(i, w) for i in range(val.size) for w in val.windows]
    rng.shuffle(val_order)
    val_cursor = 0
    history, best, best_score, best_epoch, stale = [], None, -math.inf, 0, 0
    reason, stage = "epoch_budget", "search" if search else "classifier"
    for epoch in range(1, settings["search_epochs" if search else "final_epochs"] + 1):
        started, loss_sum, count, architecture_loss_sum, architecture_updates = time.perf_counter(), 0.0, 0, 0.0, 0
        for batch in train.batches(rng, settings["batch_size"]):
            model.train()
            for parameter in parameters:
                parameter.requires_grad_(True)
            for parameter in architecture:
                parameter.requires_grad_(False)
            losses = [F.cross_entropy(model(e.observed, cache.get(e)), torch.tensor([e.label])) for e in batch]
            loss = _step(torch.stack(losses).mean(), optimizer, parameters, settings["grad_clip"])
            loss_sum += loss * len(batch)
            count += len(batch)
            if search:
                model.eval()
                for parameter in parameters:
                    parameter.requires_grad_(False)
                    parameter.grad = None
                for parameter in architecture:
                    parameter.requires_grad_(True)
                validation_batch = []
                for _ in range(len(batch)):
                    if val_cursor == len(val_order):
                        rng.shuffle(val_order)
                        val_cursor = 0
                    index, window = val_order[val_cursor]
                    validation_batch.append(val.example(index, window))
                    val_cursor += 1
                losses = [F.cross_entropy(model(e.observed, cache.get(e)), torch.tensor([e.label]))
                          for e in validation_batch]
                architecture_loss_sum += _step(torch.stack(losses).mean(), architecture_optimizer,
                                               architecture, settings["grad_clip"])
                architecture_updates += 1
                if any(p.grad is not None for p in parameters):
                    raise AssertionError("Architecture update changed frozen network gradients")
        validation = classifier_validation(model, val, cache)
        row = {"stage": stage, "epoch": epoch, "train_loss": loss_sum / count,
               "validation": validation, "seconds": time.perf_counter() - started}
        if search:
            row["architecture_loss"] = architecture_loss_sum / architecture_updates
            row["architecture"] = model.selected_architecture()
        history.append(row)
        log(row)
        if validation["score"] > best_score:
            best, best_score, best_epoch, stale = _snapshot(model), validation["score"], epoch, 0
            save_checkpoint(output / f"{stage}_best.pt", {"state_dict": best, "epoch": epoch})
        else:
            stale += 1
        write_json(output / f"{stage}_history.json", {"history": history})
        if epoch >= settings["min_epochs"] and stale >= settings["early_stopping_patience"]:
            reason = "validation_patience"
            break
    for parameter in parameters + architecture:
        parameter.requires_grad_(True)
    return _finish_stage(model, best, history, best_epoch, reason, output / f"{stage}.pt",
                         {"training_split": "train", "selection_split": "val", "architecture_split": "val" if search else "frozen"})


def train_pipeline(train: WindowData, val: WindowData, config: dict, seed: int,
                   output: Path, log: Callable[[dict], None], *, diagnostic: bool = False) -> tuple:
    if seed not in SEEDS or train.split != "train" or val.split != "val":
        raise ValueError("D2 training requires a declared seed and train/validation splits")
    config = copy.deepcopy(config)
    settings = config["training"]
    if config["model"]["search_space"] != SEARCH_SPACE:
        raise ValueError("Unsupported D2 architecture search space")
    if settings["max_samples"] != 0 or (not diagnostic and (train.size != train.full_size or val.size != val.full_size)):
        raise ValueError("Production D2 fitting cannot truncate samples")
    if diagnostic:
        for key in ("forecast_epochs", "search_epochs", "final_epochs", "min_epochs"):
            settings[key] = 1
    capacity = train.capacity()
    if config["model"]["max_completion_nodes"] not in (0, capacity):
        raise ValueError("Completion capacity must match the training-derived maximum")
    config["model"]["max_completion_nodes"] = capacity
    kwargs = {key: config["model"][key] for key in ("hidden", "num_layers", "dropout", "attention_heads")}
    kwargs["text_dim"] = train.example(0, min(train.windows)).observed.text.numel()
    set_global_seed(seed)
    forecaster = D2Forecaster(max_completion_nodes=capacity, loss_weights=config["forecast"]["loss_weights"], **kwargs)
    log({"stage": "capacity", "max_completion_nodes": capacity, "training_sources": train.size,
         "training_windows": train.windows, "diagnostic": diagnostic})
    forecast_record = fit_forecaster(forecaster, train, val, config, seed, output, log)
    forecaster.eval().requires_grad_(False)
    forecast_state = _snapshot(forecaster)
    cache = CompletionCache(forecaster, settings["forecast_cache_mib"])
    set_global_seed(seed)
    supernet = D2Classifier(num_classes=train.num_classes, **kwargs)
    search_record = fit_classifier(supernet, train, val, cache, config, seed, output, log, search=True)
    architecture = supernet.selected_architecture()
    set_global_seed(seed)
    classifier = D2Classifier(num_classes=train.num_classes, architecture=architecture, **kwargs)
    # Construct new modules instead of transferring any weight-sharing parameters.
    if {p.data_ptr() for p in classifier.parameters()} & {p.data_ptr() for p in supernet.parameters()}:
        raise AssertionError("Discrete classifier unexpectedly shares supernet storage")
    del supernet
    final_record = fit_classifier(classifier, train, val, cache, config, seed, output, log, search=False)
    if any(not torch.equal(forecast_state[key], value) for key, value in forecaster.state_dict().items()):
        raise AssertionError("Frozen forecaster changed during architecture or classifier fitting")
    validation = {str(window): classifier_outputs(classifier, val, cache, window) for window in val.windows}
    return forecaster, classifier, config, {
        "forecaster": forecast_record, "search": search_record, "classifier": final_record,
        "selected_architecture": architecture, "training_sources": train.size,
        "validation_sources": val.size, "windows": list(train.windows),
        "diagnostic": diagnostic, "forecaster_frozen": True,
        "classifier_retrained_from_scratch": True,
    }, validation
