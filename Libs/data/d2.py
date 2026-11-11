"""Typed observed inputs and separate real completion targets for independent D2.

No label, source identifier, mature size, or withheld identity enters ObservedGraph.
Completion slots follow undirected breadth-first attachment from observed nodes.
All recorded directed arcs remain separate targets, including cycles and backlinks.
Minute completions distinguish actual future events from disconnected past history
excluded by the frozen root-connected observation rule.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ObservedGraph:
    x: Tensor
    edge_index: Tensor
    coordinates: Tensor
    post_author: Tensor
    num_users: int
    text: Tensor
    root: int
    unit: str
    horizon: int

    @property
    def num_posts(self) -> int:
        return int(self.x.shape[0])

    def to(self, device: torch.device | str) -> ObservedGraph:
        return replace(self, **{name: getattr(self, name).to(device) for name in
                               ("x", "edge_index", "coordinates", "post_author", "text")})

    def detach(self) -> ObservedGraph:
        return replace(self, **{name: getattr(self, name).detach() for name in
                               ("x", "edge_index", "coordinates", "post_author", "text")})


@dataclass(frozen=True)
class CompletionTarget:
    """Supervision only; parent destinations are completion-slot indices.

    Parent sources use observed indices followed by completion slots. An author is
    an observed-user index, U + its first completion slot, or -1 for a missing ID.
    Parent edges identify one real undirected attachment per unobserved node.
    Directions are 0=parent-to-child, 1=child-to-parent, 2=both. graph_edges
    retains every recorded directed arc incident to a unobserved node, with both
    endpoints in observed-then-completion indices. Integer author aliases encode
    supervision without publishing real identities.
    """
    count: int
    parent_edges: Tensor
    author: Tensor
    coordinates: Tensor
    parent_direction: Tensor
    graph_edges: Tensor

    def to(self, device: torch.device | str) -> CompletionTarget:
        return replace(self, parent_edges=self.parent_edges.to(device),
                       author=self.author.to(device), coordinates=self.coordinates.to(device),
                       parent_direction=self.parent_direction.to(device), graph_edges=self.graph_edges.to(device))


@dataclass(frozen=True)
class D2Example:
    observed: ObservedGraph
    target: CompletionTarget | None
    label: int
    source_id: str


def _author_identity(value: Any, tweet_id: Any) -> str:
    # The Twitter tree format retains a structural [ROOT, ROOT, time] node.
    # Keep that node without assigning it an actual user or an authorship edge.
    if value is None or (str(value) == "ROOT" and str(tweet_id) == "ROOT"):
        return ""
    return str(value)


def _authors(values: list[Any], tweet_ids: list[Any]) -> tuple[Tensor, dict[str, int]]:
    mapping: dict[str, int] = {}
    indices: list[int] = []
    if len(values) != len(tweet_ids):
        raise ValueError("Author metadata and node records must align")
    for value, tweet_id in zip(values, tweet_ids):
        uid = _author_identity(value, tweet_id)
        if not uid.strip():
            indices.append(-1)
        else:
            if uid not in mapping:
                mapping[uid] = len(mapping)
            indices.append(mapping[uid])
    return torch.tensor(indices, dtype=torch.long), mapping


def adapt_graph(data: Any, split: str, unit: str, horizon: int,
                with_targets: bool = False) -> D2Example:
    """Copy only observed tensors into model input; optionally align real targets."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if with_targets and split == "test":
        raise ValueError("D2 test completion targets are forbidden")
    unit = {"hops": "hop", "minute": "minutes"}.get(str(unit).lower(), str(unit).lower())
    if unit not in {"hop", "minutes"} or horizon < 0:
        raise ValueError("D2 requires a nonnegative minute or hop horizon")
    suffix = "hops" if unit == "hop" else "minutes"
    coordinate_name = f"node_time_{suffix}_shifted"
    x = data.x.detach().to(dtype=torch.float32).clone()
    edges = data.edge_index.detach().to(dtype=torch.long).clone()
    coordinates = getattr(data, coordinate_name).detach().to(dtype=torch.long).view(-1).clone()
    authors, author_ids = _authors(list(data.node_uid), list(data.node_tweet_id))
    text = data.text_emb.detach().to(dtype=torch.float32).reshape(1, -1).clone()
    root = int(data.root_index.view(-1)[0])
    n = int(x.shape[0])
    if n == 0 or x.shape != (n, 8) or coordinates.shape != (n,) or authors.shape != (n,):
        raise ValueError("D2 observed tensors must align and contain eight structural features")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("D2 observed edges must have shape [2,E]")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError("D2 observed edge endpoint is outside the observed graph")
    if not 0 <= root < n or bool((coordinates < 0).any()) or bool((coordinates > horizon).any()):
        raise ValueError("D2 observed root or observation coordinates violate the declared view")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(text).all()):
        raise ValueError("D2 observed features contain non-finite values")
    observed = ObservedGraph(x, edges, coordinates, authors.to(x.device), len(author_ids),
                             text, root, unit, int(horizon))
    labels = data.y.detach().view(-1)
    if labels.numel() != 1:
        raise ValueError("D2 adapter accepts one cascade at a time")
    target = None
    if with_targets:
        required = ("full_node_tweet_id", "full_node_uid", "full_edge_index",
                    "full_observed_node_indices", f"full_{coordinate_name}")
        if any(not hasattr(data, name) for name in required):
            raise ValueError("Completion supervision requires a train_ctdm train/validation view")
        full_ids = list(data.full_node_tweet_id)
        observed_ids = list(data.node_tweet_id)
        full_coordinates = getattr(data, f"full_{coordinate_name}").detach().to(dtype=torch.long).view(-1)
        full_authors = list(data.full_node_uid)
        if len(full_ids) != len(full_authors) or len(full_ids) != len(full_coordinates):
            raise ValueError("Mature target identities and coordinates do not align")
        source_indices = data.full_observed_node_indices
        if (not isinstance(source_indices, Tensor) or source_indices.dtype != torch.long
                or source_indices.shape != (n,) or len(observed_ids) != n):
            raise ValueError("Observed-to-mature source indices must align with observed nodes")
        old_observed = source_indices.detach().cpu().tolist()
        if (len(set(old_observed)) != n or min(old_observed) < 0
                or max(old_observed) >= len(full_ids)):
            raise ValueError("Observed-to-mature source indices must be unique and in bounds")
        old_to_new = {old: i for i, old in enumerate(old_observed)}
        for i, old in enumerate(old_observed):
            if (str(full_ids[old]) != str(observed_ids[i])
                    or int(full_coordinates[old]) != int(coordinates[i])
                    or str(full_authors[old]) != str(data.node_uid[i])):
                raise ValueError("Observed source-row metadata conflicts with mature target")
        unobserved = set(range(len(full_ids))) - set(old_observed)
        if any(int(full_coordinates[i]) < 0 for i in unobserved):
            raise ValueError("Completion target contains a negative native coordinate")
        full_edges = [tuple(map(int, edge)) for edge in data.full_edge_index.detach().cpu().t().tolist()]
        if any(min(edge) < 0 or max(edge) >= len(full_ids) for edge in full_edges):
            raise ValueError("Mature target edge endpoint is outside the graph")
        adjacency: dict[int, set[int]] = {i: set() for i in range(len(full_ids))}
        recorded_arcs = set(full_edges)
        for source, destination in full_edges:
            adjacency[source].add(destination)
            adjacency[destination].add(source)
        censored_history = [i for i in unobserved if int(full_coordinates[i]) <= horizon]
        if censored_history and unit == "hop":
            raise ValueError("Hop completion includes a node inside the observed horizon")
        if any(adjacency[node].intersection(old_observed) for node in censored_history):
            raise ValueError("Hidden past node has direct observed contact and violates root-connected censoring")

        def order_key(index: int) -> tuple[int, int]:
            return int(full_coordinates[index]), index

        queue = deque(sorted(old_observed, key=order_key))
        visited = set(old_observed)
        ordered: list[int] = []
        attachments: list[int] = []
        directions: list[int] = []
        while queue:
            parent = queue.popleft()
            for node in sorted(adjacency[parent], key=order_key):
                if node in visited:
                    continue
                visited.add(node)
                queue.append(node)
                ordered.append(node)
                attachments.append(parent)
                forward = (parent, node) in recorded_arcs
                reverse = (node, parent) in recorded_arcs
                directions.append(2 if forward and reverse else (0 if forward else 1))
        if len(ordered) != len(unobserved):
            raise ValueError("Completion target contains a component disconnected from every observed node")
        old_to_new.update({old: n + i for i, old in enumerate(ordered)})
        parent_edges = [(old_to_new[parent], i) for i, parent in enumerate(attachments)]
        graph_edges = [(old_to_new[source], old_to_new[destination])
                       for source, destination in full_edges if source in unobserved or destination in unobserved]
        target_authors: list[int] = []
        new_author_slots: dict[str, int] = {}
        for i, old in enumerate(ordered):
            uid = _author_identity(full_authors[old], full_ids[old])
            if not uid.strip():
                target_authors.append(-1)
            elif uid in author_ids:
                target_authors.append(author_ids[uid])
            else:
                if uid not in new_author_slots:
                    new_author_slots[uid] = len(author_ids) + i
                target_authors.append(new_author_slots[uid])
        edge_tensor = torch.tensor(parent_edges, dtype=torch.long).t().contiguous() if parent_edges else torch.empty((2, 0), dtype=torch.long)
        target = CompletionTarget(len(ordered), edge_tensor,
                                  torch.tensor(target_authors, dtype=torch.long),
                                  full_coordinates[ordered].clone().cpu(),
                                  torch.tensor(directions, dtype=torch.long),
                                  torch.tensor(graph_edges, dtype=torch.long).t().contiguous()
                                  if graph_edges else torch.empty((2, 0), dtype=torch.long))
    source_id = str(data.source_id)
    if not source_id:
        raise ValueError("D2 example requires a source identity for evidence")
    return D2Example(observed, target, int(labels.item()), source_id)
