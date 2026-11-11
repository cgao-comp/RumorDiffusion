import hashlib
import json
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

from Libs.data.potential_cache import load_validated_potential_cache
from Libs.data.processor import compute_structural_features


def _assert_tensor_2_by_n(edge_index: Tensor) -> None:
    """Asserts that an edge index tensor has shape (2, E) and dtype long.

    Args:
        edge_index: Edge index tensor to validate.
    """
    assert isinstance(edge_index, Tensor), "edge_index must be a torch.Tensor"
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, (
        f"edge_index must be 2 x E, got {tuple(edge_index.shape)}"
    )
    assert edge_index.dtype in (torch.long, torch.int64), (
        f"edge_index dtype must be long, got {edge_index.dtype}"
    )


def _remove_self_loops(edge_index: Tensor) -> Tensor:
    """Removes self-loops from directed edge list.

    Args:
        edge_index: Input edges of shape (2, E).

    Returns:
        Edge index without self-loops.
    """
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    return edge_index[:, mask]


def _unique_union_edges(edge_index_a: Tensor, edge_index_b: Tensor, num_nodes: int) -> Tensor:
    """Returns the union of two directed edge sets, deduplicated and loop-free.

    The result keeps no self-loops and contains each directed edge at most once.

    Args:
        edge_index_a: First edge set (2, E1).
        edge_index_b: Second edge set (2, E2).
        num_nodes: Number of nodes, used to hash pairs.

    Returns:
        A tensor of shape (2, E) with unique edges.
    """
    _assert_tensor_2_by_n(edge_index_a)
    _assert_tensor_2_by_n(edge_index_b)

    ea = _remove_self_loops(edge_index_a)
    eb = _remove_self_loops(edge_index_b)

    # Bounds check to catch bad LP edges early.
    if ea.numel() > 0:
        assert int(ea.max().item()) < num_nodes, (
            f"edge_index_a has index >= num_nodes ({int(ea.max().item())} >= {num_nodes})"
        )
    if eb.numel() > 0:
        assert int(eb.max().item()) < num_nodes, (
            f"edge_index_b has index >= num_nodes ({int(eb.max().item())} >= {num_nodes})"
        )

    src = torch.cat([ea[0], eb[0]], dim=0)
    dst = torch.cat([ea[1], eb[1]], dim=0)

    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index_a.device)

    # Hash pairs to 1D keys and unique them.
    keys = src.to(torch.int64) * int(num_nodes) + dst.to(torch.int64)
    uniq_keys = torch.unique(keys)
    uniq_src = torch.div(uniq_keys, int(num_nodes), rounding_mode="floor")
    uniq_dst = uniq_keys % int(num_nodes)
    out = torch.stack([uniq_src.to(torch.long), uniq_dst.to(torch.long)], dim=0)
    return out


def _stable_seed(source_id: object, base: int = 0) -> int:
    """Returns a stable integer seed derived from source_id and base."""
    text = str(source_id)
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    seed_core = int(h[:8], 16)
    return int((seed_core + int(base)) % (2**31))


def _prefix_mask_from_strategy(times: Tensor, strategy: str, minutes: Optional[int], k: Optional[int], root_index: Optional[int], include_root: bool) -> Tensor:
    """Computes prefix node mask according to strategy."""
    times_long = times.to(torch.long)
    n = times_long.numel()
    if strategy in {"minutes", "hop", "window"}:
        if minutes is None:
            raise ValueError(f"Prefix strategy '{strategy}' requires an explicit horizon value.")
        cutoff = int(minutes if minutes is not None else times_long.max().item() if n > 0 else 0)
        mask = times_long <= cutoff
    elif strategy == "count":
        if n == 0:
            return torch.zeros_like(times_long, dtype=torch.bool)
        budget = int(k if k is not None else n)
        budget = max(1, min(budget, n))
        order = torch.argsort(times_long)
        chosen = order[:budget]
        mask = torch.zeros_like(times_long, dtype=torch.bool)
        mask[chosen] = True
    elif strategy == "none":
        mask = torch.ones_like(times_long, dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported prefix strategy: {strategy}")
    if include_root and root_index is not None and 0 <= int(root_index) < n:
        mask[int(root_index)] = True
    return mask


def _time_tensor_for_unit(data: Data, unit: str) -> Tensor:
    """Returns the explicitly stored observation coordinate for a horizon unit."""
    unit_norm = str(unit).lower()
    if unit_norm == "minutes":
        field = "node_time_minutes_shifted"
    elif unit_norm == "hop":
        field = "node_time_hops_shifted"
    elif unit_norm == "count":
        # Count budgets use arrival order. Prefer the declared native unit only
        # to establish a deterministic order; no count is inferred from future
        # cascade size.
        native = str(getattr(data, "time_unit", "minutes")).lower()
        field = "node_time_hops_shifted" if native == "hop" else "node_time_minutes_shifted"
    else:
        raise ValueError(f"Unsupported horizon unit: {unit}")
    if not hasattr(data, field):
        raise ValueError(f"Sample lacks explicit time field {field} for horizon unit {unit_norm}")
    values = getattr(data, field)
    if not isinstance(values, Tensor) or values.dim() != 1:
        raise ValueError(f"Sample field {field} must be a one-dimensional tensor")
    return values.to(torch.long)


def _root_connected_mask(mask: Tensor, edge_index: Tensor, root_index: int) -> Tensor:
    """Retains the root-connected component after prefix/missingness filtering."""
    if mask.numel() == 0 or not bool(mask.any()):
        raise ValueError("Prefix/missingness removed every node")
    if root_index < 0 or root_index >= int(mask.numel()) or not bool(mask[root_index]):
        raise ValueError("Prefix/missingness removed the root node")
    adjacency: Dict[int, List[int]] = {}
    for src, dst in edge_index.t().tolist():
        src_i, dst_i = int(src), int(dst)
        if bool(mask[src_i]) and bool(mask[dst_i]):
            adjacency.setdefault(src_i, []).append(dst_i)
            adjacency.setdefault(dst_i, []).append(src_i)
    visited = {int(root_index)}
    stack = [int(root_index)]
    while stack:
        current = stack.pop()
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    output = torch.zeros_like(mask, dtype=torch.bool)
    output[torch.tensor(sorted(visited), dtype=torch.long, device=mask.device)] = True
    return output


def _collated_source_ids(data_obj: Data, slices: Dict[str, Tensor]) -> List[str]:
    """Extracts per-graph source IDs from PyG InMemoryDataset storage."""
    if not hasattr(data_obj, "source_id") or "source_id" not in slices:
        raise ValueError("Processed dataset lacks collated source_id metadata")
    values = list(data_obj.source_id)
    boundaries = slices["source_id"]
    result: List[str] = []
    for index in range(len(boundaries) - 1):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        segment = values[start:end]
        if len(segment) != 1:
            raise ValueError(f"Expected one source_id per graph, got {segment!r}")
        result.append(str(segment[0]))
    return result


def _apply_missingness(mask: Tensor, edge_index: Tensor, drop_node: float, drop_edge: float, seed: int, root_index: Optional[int], include_root: bool) -> Tuple[Tensor, Tensor]:
    """Applies node/edge dropout to observed prefix."""
    keep_nodes = mask.to(torch.bool).clone()
    if drop_node > 0.0 and keep_nodes.numel() > 0:
        gen = torch.Generator(device=keep_nodes.device)
        gen.manual_seed(int(seed))
        rand_node = torch.rand(keep_nodes.shape, generator=gen, device=keep_nodes.device)
        keep_nodes = keep_nodes & (rand_node >= float(drop_node))
        if include_root and root_index is not None and 0 <= int(root_index) < keep_nodes.numel():
            keep_nodes[int(root_index)] = True
    edge_mask = keep_nodes[edge_index[0]] & keep_nodes[edge_index[1]]
    if drop_edge > 0.0 and edge_mask.numel() > 0:
        gen_edge = torch.Generator(device=edge_mask.device)
        gen_edge.manual_seed(int(seed))
        edge_mask = edge_mask & (torch.rand(edge_mask.shape, generator=gen_edge, device=edge_mask.device) >= float(drop_edge))
    edge_filtered = edge_index[:, edge_mask]
    return keep_nodes, edge_filtered


def _reindex_edge_index(edge_index: Tensor, keep_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Reindexes an edge list into the compact index space defined by `keep_mask`."""
    kept_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    old_to_new = torch.full(
        (int(keep_mask.numel()),),
        -1,
        dtype=torch.long,
        device=keep_mask.device,
    )
    old_to_new[kept_indices] = torch.arange(kept_indices.numel(), dtype=torch.long, device=keep_mask.device)
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), kept_indices
    src = old_to_new[edge_index[0].to(torch.long)]
    dst = old_to_new[edge_index[1].to(torch.long)]
    valid = (src >= 0) & (dst >= 0)
    if not bool(valid.any()):
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), kept_indices
    return torch.stack([src[valid], dst[valid]], dim=0).contiguous(), kept_indices


def _materialize_prefix_view(data: Data, keep_mask: Tensor, edge_index: Tensor, preserve_mature_state: bool = False) -> Data:
    """Materializes a prefix-only graph view.

    Mature-state tensors are retained only for CTDM training. Evaluation views
    deliberately omit them so a model cannot accidentally consume future graph
    state.
    """
    keep_mask = keep_mask.to(torch.bool)
    full_edge_index = data.edge_index.to(torch.long)
    full_x = data.x.to(torch.float32)
    full_root_index = data.root_index.view(-1).to(torch.long)
    full_shifted_times = _time_tensor_for_unit(data, str(getattr(data, "time_unit", "minutes")))
    full_times_minutes = data.node_time_minutes.to(torch.long)
    full_times_hops = data.node_time_hops.to(torch.long) if hasattr(data, "node_time_hops") else torch.zeros_like(full_times_minutes)
    full_shifted_minutes = data.node_time_minutes_shifted.to(torch.long)
    full_shifted_hops = data.node_time_hops_shifted.to(torch.long) if hasattr(data, "node_time_hops_shifted") else torch.zeros_like(full_shifted_minutes)
    full_node_uid = list(data.node_uid)
    full_node_tweet_id = list(data.node_tweet_id)

    prefix_edge_index, kept_indices = _reindex_edge_index(edge_index.to(torch.long), keep_mask)
    keep_nodes = [int(index) for index in kept_indices.tolist()]
    root_old = int(full_root_index[0].item())
    if root_old not in keep_nodes:
        raise ValueError("Prefix materialization dropped the root node.")
    root_index_new = int(keep_nodes.index(root_old))
    shifted_times_list = [int(full_shifted_times[index].item()) for index in keep_nodes]
    full_nodes = [
        (str(full_node_uid[index]), str(full_node_tweet_id[index]), int(full_shifted_times[index].item()))
        for index in range(len(full_node_uid))
    ]
    full_edges = [(int(src), int(dst)) for src, dst in full_edge_index.t().tolist()]
    reindexed_edges = [(int(src), int(dst)) for src, dst in prefix_edge_index.t().tolist()]

    if preserve_mature_state:
        data.full_x = full_x
        data.full_edge_index = full_edge_index
        data.full_root_index = full_root_index
        data.full_node_time_minutes = full_times_minutes
        data.full_node_time_minutes_shifted = full_shifted_minutes
        data.full_node_time_hops = full_times_hops
        data.full_node_time_hops_shifted = full_shifted_hops
        data.full_node_uid = full_node_uid
        data.full_node_tweet_id = full_node_tweet_id
        # Preserve the exact source-row alignment; tweet IDs can repeat.
        data.full_observed_node_indices = kept_indices.clone()
    else:
        for field in (
            "full_x", "full_edge_index", "full_root_index", "full_node_time_minutes",
            "full_node_time_minutes_shifted", "full_node_time_hops", "full_node_time_hops_shifted",
            "full_node_uid", "full_node_tweet_id", "full_observed_node_indices",
        ):
            if hasattr(data, field):
                delattr(data, field)

    data.x = compute_structural_features(
        nodes=full_nodes,
        edges=full_edges,
        keep_nodes=keep_nodes,
        reindexed_edges=reindexed_edges,
        root_index_new=root_index_new,
        feature_time_scale=max(shifted_times_list) if len(shifted_times_list) > 0 else 1,
        shifted_times_list=shifted_times_list,
    )
    data.edge_index = prefix_edge_index
    data.root_index = torch.tensor([root_index_new], dtype=torch.long)
    data.node_uid = [full_node_uid[index] for index in keep_nodes]
    data.node_tweet_id = [full_node_tweet_id[index] for index in keep_nodes]
    data.node_time_minutes = full_times_minutes[kept_indices].clone()
    data.node_time_minutes_shifted = full_shifted_minutes[kept_indices].clone()
    data.node_time_hops = full_times_hops[kept_indices].clone()
    data.node_time_hops_shifted = full_shifted_hops[kept_indices].clone()
    data.prefix_node_mask = torch.ones(int(len(keep_nodes)), dtype=torch.bool)
    data.prefix_edge_index = data.edge_index.clone()
    # This is prefix-local observability, not a fraction of the unobserved
    # mature cascade. Keep it out of the model input until explicitly declared.
    data.prefix_observed_count = int(len(keep_nodes))
    if hasattr(data, "t"):
        data.t = _time_tensor_for_unit(data, str(getattr(data, "time_unit", "minutes"))).to(torch.float32)
    for stale_attr in ("h0", "h1", "h_cat"):
        if hasattr(data, stale_attr):
            delattr(data, stale_attr)
    return data


class GraphDataset(InMemoryDataset):
    """Graph classification dataset for D2Plus processed trees.

    This dataset loads pre-collated PyG data saved by the preprocessor and
    exposes train/val/test splits. It optionally merges link-prediction edges
    into each sample and sets time features for TGAT models.

    Attributes:
        root: Path like `Input/processed/twitter15`.
        split: Split name in {"train", "val", "test"}.
        use_lp_edges: Whether to merge predicted edges from `lp_edges/`.
        lp_dir: Directory storing LP edges, defaults to `root/lp_edges`.
        set_t_for_tgat: Whether to set `data.t` from shifted minutes.
        indices: Indices of samples within the collated storage for this split.
    """

    def __init__(
        self,
        root: str,
        split: str,
        use_lp_edges: bool = False,
        lp_dir: Optional[str] = None,
        set_t_for_tgat: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        use_potentials: bool = False,
        potentials_dir: Optional[str] = None,
        potentials_pair: Optional[List[int]] = None,
        potential_cache_metadata: Optional[Dict] = None,
        prefix_cfg: Optional[Dict] = None,
        prefix_consistency_views: Optional[List[Dict]] = None,
        prefix_consistency_potential_specs: Optional[List[Dict]] = None,
        observability_cfg: Optional[Dict] = None,
        text_cache_name: Optional[str] = None,
        force_text_cache: bool = False,
        text_normalize: bool = False,
        mode: str = "eval_prefix",
        require_lp_edges: bool = False,
        label_schema: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

        assert split in {"train", "val", "test"}, f"Invalid split: {split}"

        self._processed_dir = os.path.join(self.root, "pyg")
        data_path = os.path.join(self._processed_dir, "data.pt")
        slices_path = os.path.join(self._processed_dir, "slices.pt")
        splits_path = os.path.join(self.root, "splits.json")

        assert os.path.isfile(data_path), f"Missing {data_path}"
        assert os.path.isfile(slices_path), f"Missing {slices_path}"
        assert os.path.isfile(splits_path), f"Missing {splits_path}"

        self.mode = str(mode)
        if self.mode not in {"legacy", "train_ctdm", "train_arbiter", "eval_prefix"}:
            raise ValueError(f"Unsupported dataset mode: {self.mode}")

        self.data = torch.load(data_path, weights_only=False)
        self.slices = torch.load(slices_path, weights_only=False)

        with open(splits_path, "r", encoding="utf-8") as f:
            all_splits = json.load(f)
        assert split in all_splits, f"Split '{split}' not in splits.json"
        self._indices: List[int] = list(all_splits[split])

        # Canonical modes require immutable manifests and a declared class
        # schema. Legacy callers may continue to load historical processed
        # trees, but they cannot be used as canonical evidence.
        self._canonical_mode = self.mode != "legacy"
        label_schema_path = os.path.join(self.root, "label_schema.json")
        if label_schema is None and os.path.isfile(label_schema_path):
            with open(label_schema_path, "r", encoding="utf-8") as handle:
                schema_payload = json.load(handle)
            if not isinstance(schema_payload, dict) or not isinstance(schema_payload.get("mapping"), dict):
                raise ValueError(f"Invalid label schema: {label_schema_path}")
            label_schema = {str(key): int(value) for key, value in schema_payload["mapping"].items()}
        if self._canonical_mode:
            required_manifests = ("raw_manifest.json", "processed_manifest.json", "split_manifest.json", "label_schema.json")
            missing_manifests = [name for name in required_manifests if not os.path.isfile(os.path.join(self.root, name))]
            if missing_manifests:
                raise FileNotFoundError(
                    f"Canonical dataset is missing manifests for {self.root}: {', '.join(missing_manifests)}"
                )
            with open(os.path.join(self.root, "split_manifest.json"), "r", encoding="utf-8") as handle:
                split_manifest = json.load(handle)
            with open(splits_path, "rb") as splits_artifact:
                current_splits_hash = hashlib.sha256(splits_artifact.read()).hexdigest()
            if split_manifest.get("splits_sha256") != current_splits_hash:
                raise ValueError(f"split_manifest.json is stale for {self.root}")
            with open(os.path.join(self.root, "processed_manifest.json"), "r", encoding="utf-8") as handle:
                processed_manifest = json.load(handle)
            for path_key, path in (("data_sha256", data_path), ("slices_sha256", slices_path)):
                with open(path, "rb") as artifact:
                    current_hash = hashlib.sha256(artifact.read()).hexdigest()
                if processed_manifest.get(path_key) != current_hash:
                    raise ValueError(f"processed_manifest.json is stale for {path}")
            source_ids = _collated_source_ids(self.data, self.slices)
            if processed_manifest.get("source_id_sha256") != hashlib.sha256(
                json.dumps(source_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest():
                raise ValueError(f"processed_manifest.json source IDs are stale for {self.root}")
            declared_split = split_manifest.get("splits", {})
            for split_name in ("train", "val", "test"):
                expected_ids = [source_ids[index] for index in all_splits[split_name]]
                observed = declared_split.get(split_name, {}).get("source_ids")
                if observed != expected_ids:
                    raise ValueError(f"split_manifest.json source IDs are stale for split={split_name}")

        # Optional LP edge merging settings
        self.use_lp_edges = bool(use_lp_edges)
        self.lp_dir = lp_dir if lp_dir is not None else os.path.join(self.root, "lp_edges")
        self.set_t_for_tgat = bool(set_t_for_tgat)
        self.require_lp_edges = bool(require_lp_edges)
        self.label_schema = {str(key): int(value) for key, value in (label_schema or {}).items()}

        # Optional potentials loading settings
        self.use_potentials = bool(use_potentials)
        self.split = str(split)
        self.potentials_dir = potentials_dir if potentials_dir is not None else os.path.join(self.root, "rd_potentials")
        self.potentials_pair = potentials_pair if potentials_pair is not None else [0, 1]
        self.potential_cache_metadata = dict(potential_cache_metadata) if potential_cache_metadata is not None else None
        if self.use_potentials and self.potential_cache_metadata is None:
            raise ValueError("RumorDiffusion potential loading requires expected cache metadata.")
        # Prefix and observability configuration
        prefix_cfg = prefix_cfg if prefix_cfg is not None else {}
        self._prefix_strategy = str(prefix_cfg.get("strategy", "none")).lower()
        self._prefix_unit = str(prefix_cfg.get("unit", "minutes")).lower()
        if self._prefix_unit not in {"minutes", "hop", "count"}:
            raise ValueError(f"Unsupported prefix horizon unit: {self._prefix_unit}")
        self._prefix_horizon_value = prefix_cfg.get("horizon_value", prefix_cfg.get("window", prefix_cfg.get("minutes", None)))
        self._prefix_minutes = self._prefix_horizon_value
        self._prefix_k = prefix_cfg.get("k", None)
        self._prefix_include_root = bool(prefix_cfg.get("include_root", True))
        self._prefix_consistency_views = [dict(view) for view in (prefix_consistency_views or [])]
        self._prefix_consistency_potential_specs = [
            dict(spec) for spec in (prefix_consistency_potential_specs or [])
        ]
        if self._prefix_consistency_views and self.use_potentials:
            if len(self._prefix_consistency_potential_specs) != len(self._prefix_consistency_views):
                raise ValueError(
                    "Prefix-consistency potential specs must align one-to-one with prefix views."
                )
            for index, spec in enumerate(self._prefix_consistency_potential_specs):
                if not spec.get("potentials_dir") or not isinstance(spec.get("potential_cache_metadata"), dict):
                    raise ValueError(
                        f"Prefix-consistency potential spec {index} must provide potentials_dir and metadata."
                    )

        observability_cfg = observability_cfg if observability_cfg is not None else {}
        self._drop_edge = float(observability_cfg.get("drop_edge", 0.0))
        self._drop_node = float(observability_cfg.get("drop_node", 0.0))
        self._obs_seed = int(observability_cfg.get("seed", 0))
        self._force_text_cache = bool(force_text_cache)
        self._text_normalize = bool(text_normalize)

        # Optional text embedding cache (graph-level). Load once if present.
        resolved_text_cache_name = str(text_cache_name).strip() if text_cache_name is not None else "text_embs.pt"
        text_cache_path = os.path.join(self.root, resolved_text_cache_name)
        self._text_cache_loaded = False
        self._text_id_to_idx: Optional[Dict[str, int]] = None
        self._text_emb_matrix: Optional[Tensor] = None
        if os.path.isfile(text_cache_path):
            obj = torch.load(text_cache_path, weights_only=False)
            ids: List[str] = list(obj["ids"])  # type: ignore[index]
            emb: Tensor = obj["emb"].to(torch.float32)  # type: ignore[index]
            manifest = obj.get("manifest") if isinstance(obj, dict) else None
            required_manifest = {
                "encoder_name", "weights_sha256", "tokenizer_sha256", "normalization_config_sha256",
                "source_id_sha256", "normalized_text_sha256", "embedding_dimension", "dtype",
            }
            if not isinstance(manifest, dict) or not required_manifest.issubset(manifest):
                raise ValueError(f"Text embedding cache lacks strict manifest metadata: {text_cache_path}")
            if int(manifest["embedding_dimension"]) != int(emb.size(-1)):
                raise ValueError(f"Text embedding cache dimension mismatch: {text_cache_path}")
            if str(manifest["dtype"]) != str(obj["emb"].dtype).replace("torch.", ""):
                raise ValueError(f"Text embedding cache dtype mismatch: {text_cache_path}")
            if str(manifest["source_id_sha256"]) != hashlib.sha256(
                json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest():
                raise ValueError(f"Text embedding cache source ID hash mismatch: {text_cache_path}")
            self._text_id_to_idx = {sid: i for i, sid in enumerate(ids)}
            self._text_emb_matrix = emb
            self._text_cache_loaded = True

        # Basic integrity check on the first sample
        if len(self._indices) > 0:
            sample = super().get(self._indices[0])
            self._validate_sample(sample)

        # PyG batches require every graph to expose the same tensor keys. If a
        # cache is incomplete for a split, fall back to CTDM for that entire
        # split/view rather than creating an invalid mixed cached/on-the-fly
        # batch. Present entries are still metadata-validated on access.
        self._main_potential_cache_active = (
            self.use_potentials and self._potential_cache_complete(self.potentials_dir)
        )
        if self._prefix_consistency_views and self.use_potentials:
            self._prefix_consistency_cache_active = [
                self._potential_cache_complete(str(spec["potentials_dir"]))
                for spec in self._prefix_consistency_potential_specs
            ]
        else:
            # Keep one explicit inactive entry per view so valid on-the-fly
            # prefix consistency does not index an empty spec list.
            self._prefix_consistency_cache_active = [
                False for _ in self._prefix_consistency_views
            ]

        # Initialize num_classes as None, will be computed on first access
        self._num_classes = None
        if self.label_schema:
            self._num_classes = max(self.label_schema.values()) + 1
        elif not self._canonical_mode:
            observed = []
            for raw_index in self._indices:
                observed.append(int(super().get(int(raw_index)).y.view(-1)[0].item()))
            self._num_classes = max(observed) + 1 if observed else 0

    def _potential_cache_complete(self, potentials_dir: str) -> bool:
        """Returns whether all sources in this split have one cache entry."""
        if not os.path.isdir(potentials_dir):
            warnings.warn(
                "RumorDiffusion potential cache directory is missing for "
                f"split={self.split} at {potentials_dir}; using on-the-fly CTDM for this split.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        for raw_index in self._indices:
            sample = super().get(int(raw_index))
            cache_path = os.path.join(potentials_dir, f"{sample.source_id}.pt")
            if not os.path.isfile(cache_path):
                warnings.warn(
                    "RumorDiffusion potential cache is incomplete for "
                    f"split={self.split} at {potentials_dir}; using on-the-fly CTDM for this split.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return False
        return True

    def _materialize_configured_prefix(self, data: Data, prefix_cfg: Dict) -> Data:
        """Materializes one observed-prefix view under the dataset's missingness policy."""
        strategy = str(prefix_cfg.get("strategy", "none")).lower()
        window = prefix_cfg.get("window", prefix_cfg.get("minutes", None))
        k = prefix_cfg.get("k", None)
        include_root = bool(prefix_cfg.get("include_root", True))
        root_idx = int(data.root_index.view(-1)[0].item()) if hasattr(data, "root_index") else None
        horizon_unit = str(prefix_cfg.get("unit", self._prefix_unit)).lower()
        if horizon_unit not in {"minutes", "hop", "count"}:
            raise ValueError(f"Unsupported prefix horizon unit: {horizon_unit}")
        times = _time_tensor_for_unit(data, horizon_unit)
        prefix_mask = _prefix_mask_from_strategy(
            times=times,
            strategy=strategy,
            minutes=window,
            k=k,
            root_index=root_idx,
            include_root=include_root,
        )
        seed_val = _stable_seed(getattr(data, "source_id", "0"), base=self._obs_seed)
        obs_mask, edge_prefix = _apply_missingness(
            mask=prefix_mask,
            edge_index=data.edge_index.to(torch.long),
            drop_node=self._drop_node,
            drop_edge=self._drop_edge,
            seed=seed_val,
            root_index=root_idx,
            include_root=include_root,
        )
        # Missingness can disconnect observed nodes from the root. Apply the
        # component contract before recomputing structural features.
        obs_mask = _root_connected_mask(
            obs_mask,
            edge_prefix.to(torch.long),
            root_idx if root_idx is not None else 0,
        )
        edge_prefix = edge_prefix[:, obs_mask[edge_prefix[0]] & obs_mask[edge_prefix[1]]]
        view = _materialize_prefix_view(
            data,
            obs_mask,
            edge_prefix,
            preserve_mature_state=self.mode in {"legacy", "train_ctdm"},
        )
        if self._prefix_horizon_value is not None:
            view.horizon_value = int(self._prefix_horizon_value)
        view.horizon_unit = horizon_unit
        view.dataset_mode = self.mode
        return view

    def _attach_potentials(
        self,
        data: Data,
        potentials_dir: str,
        potentials_pair: List[int],
        potential_cache_metadata: Dict,
        require_entry: bool = False,
    ) -> Data:
        """Attaches validated class potentials when a matching cache exists.

        A missing entry is intentionally left unset so RumorDiffusion can perform
        on-the-fly CTDM inference. Any present but legacy or mismatched entry
        raises from the strict cache validator instead of silently falling back.
        """
        sid = str(data.source_id)
        pot_path = os.path.join(str(potentials_dir), f"{sid}.pt")
        if not os.path.isfile(pot_path):
            if require_entry:
                raise RuntimeError(
                    "RumorDiffusion potential cache entry disappeared after split validation: "
                    f"split={self.split}, source_id={sid}, path={pot_path}"
                )
            return data
        pot_obj = torch.load(pot_path, weights_only=False)
        Hc = load_validated_potential_cache(
            payload=pot_obj,
            expected_metadata=potential_cache_metadata,
            source_id=sid,
            cache_path=pot_path,
        )
        c = int(Hc.size(0))
        sel = [int(i) for i in potentials_pair]
        if len(sel) < 2:
            raise ValueError("potentials_pair must contain at least two indices")
        for i in sel:
            if not 0 <= i < c:
                raise ValueError(f"Invalid potentials index {i} for C={c}")
        if len(sel) == 2:
            i0, i1 = sel
            if i0 == i1:
                raise ValueError("potentials_pair indices must be distinct")
            data.h0 = Hc[i0].unsqueeze(0)
            data.h1 = Hc[i1].unsqueeze(0)
        else:
            data.h_cat = torch.cat([Hc[i] for i in sel], dim=-1).unsqueeze(0)
        return data

    def _validate_sample(self, data: Data) -> None:
        """Validates a single sample from storage for structural integrity.

        Args:
            data: A PyG `Data` object from storage.
        """
        assert isinstance(data, Data), "Loaded object must be torch_geometric.data.Data"
        assert hasattr(data, "x") and data.x.dim() == 2, "x must be [N, F]"
        assert hasattr(data, "edge_index"), "edge_index missing"
        assert hasattr(data, "y") and data.y.numel() == 1, "y must be a scalar class id"
        assert hasattr(data, "root_index") and data.root_index.numel() == 1, "root_index must be scalar"
        assert hasattr(data, "node_time_minutes_shifted"), "shifted time is required"
        assert hasattr(data, "source_id"), "source_id metadata is required"
        native_unit = str(getattr(data, "time_unit", "minutes")).lower()
        active_field = "node_time_hops_shifted" if native_unit == "hop" else "node_time_minutes_shifted"
        assert hasattr(data, active_field), f"{active_field} is required for time_unit={native_unit}"
        assert getattr(data, active_field).size(0) == data.num_nodes, (
            f"{active_field} length must equal number of nodes"
        )
        _assert_tensor_2_by_n(data.edge_index)
        if data.edge_index.numel() > 0:
            assert int(data.edge_index.max().item()) < data.num_nodes, (
                "edge_index contains node id >= num_nodes"
            )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Data:
        """Override to use split-local indexing."""
        return self.get(idx)

    def indices(self) -> List[int]:
        """Returns the indices for this split."""
        return self._indices

    def get(self, idx: int) -> Data:
        """Returns the split-specific sample with optional LP edge merging.

        Args:
            idx: Index within the current split.

        Returns:
            A `Data` object with optional `t` and merged edges.
        """
        # Map split-local index to underlying storage index
        real_idx = int(self._indices[idx])
        data = super().get(real_idx)

        if self.set_t_for_tgat:
            t = _time_tensor_for_unit(data, str(getattr(data, "time_unit", "minutes"))).to(torch.float32)
            assert t.numel() == data.num_nodes, "t length must match num_nodes"
            data.t = t

        if self.use_lp_edges:
            lp_path = os.path.join(self.lp_dir, f"{data.source_id}.pt")
            if os.path.isfile(lp_path):
                lp_obj = torch.load(lp_path, weights_only=False)
                if isinstance(lp_obj, dict):
                    assert "edge_index" in lp_obj, "LP file dict must contain 'edge_index'"
                    lp_edge_index = lp_obj["edge_index"]
                else:
                    lp_edge_index = lp_obj
                _assert_tensor_2_by_n(lp_edge_index)

                merged = _unique_union_edges(
                    data.edge_index.to(torch.long),
                    lp_edge_index.to(torch.long),
                    num_nodes=data.num_nodes,
                )
                data.edge_index = merged
            elif self.require_lp_edges:
                raise FileNotFoundError(f"Required LP edges are missing: {lp_path}")

        # Lazy attach graph-level text embedding if cache is available and attribute missing
        if self._text_cache_loaded:
            assert hasattr(data, "source_id"), "source_id metadata is required for text cache lookup"
            sid = str(data.source_id)
            assert self._text_id_to_idx is not None and self._text_emb_matrix is not None
            if self._force_text_cache or not hasattr(data, "text_emb"):
                assert sid in self._text_id_to_idx, f"text_emb missing for source_id={sid}"
                row = int(self._text_id_to_idx[sid])
                vec = self._text_emb_matrix[row].to(torch.float32).unsqueeze(0)
                if self._text_normalize:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
                data.text_emb = vec

        # Build prefix view and observability-aware subgraph if configured
        use_prefix_view = (self._prefix_strategy != "none") or (self._drop_edge > 0.0) or (self._drop_node > 0.0)
        if self._prefix_consistency_views:
            prefix_views: List[Data] = []
            for view_index, view_cfg in enumerate(self._prefix_consistency_views):
                view = self._materialize_configured_prefix(data.clone(), view_cfg)
                if self._prefix_consistency_cache_active[view_index]:
                    spec = self._prefix_consistency_potential_specs[view_index]
                    view = self._attach_potentials(
                        view,
                        potentials_dir=str(spec["potentials_dir"]),
                        potentials_pair=list(spec.get("potentials_pair", self.potentials_pair)),
                        potential_cache_metadata=dict(spec["potential_cache_metadata"]),
                        require_entry=True,
                    )
                prefix_views.append(view)
            data.prefix_views = prefix_views
        if use_prefix_view:
            data = self._materialize_configured_prefix(
                data,
                {
                    "strategy": self._prefix_strategy,
                    "window": self._prefix_minutes,
                    "k": self._prefix_k,
                    "include_root": self._prefix_include_root,
                },
            )

        if self._main_potential_cache_active:
            assert self.potential_cache_metadata is not None
            data = self._attach_potentials(
                data,
                potentials_dir=self.potentials_dir,
                potentials_pair=self.potentials_pair,
                potential_cache_metadata=self.potential_cache_metadata,
                require_entry=True,
            )

        if self.mode == "eval_prefix":
            mature_fields = (
                "full_x", "full_edge_index", "full_root_index", "full_node_time_minutes",
                "full_node_time_minutes_shifted", "full_node_time_hops", "full_node_time_hops_shifted",
                "full_node_uid", "full_node_tweet_id",
            )
            for field in mature_fields:
                if hasattr(data, field):
                    delattr(data, field)
            data.dataset_mode = self.mode
        return data

    @property
    def num_features(self) -> int:
        """Number of input node features per node."""
        if len(self._indices) == 0:
            return 0
        sample = super().get(self._indices[0])
        return int(sample.x.size(1))

    @property
    def num_classes(self) -> int:
        """Number of classes from the declared label schema."""
        if self._num_classes is None:
            raise ValueError("GraphDataset requires a declared label_schema in canonical mode")
        return self._num_classes
