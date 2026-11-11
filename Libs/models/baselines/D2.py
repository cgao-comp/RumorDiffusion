"""Independent cascade forecaster and heterogeneous differentiable NAS baseline.

This project implementation is not an exact reproduction of unpublished D2 code.
Hierarchical pointers use linear memory and logarithmic routing depth. Exact
directed adjacency supervision evaluates all incident-completion pairs in row blocks;
its compute and retained training activations scale quadratically with graph size.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.utils import softmax

from Libs.data.d2 import CompletionTarget, ObservedGraph
from Libs.data.processor import compute_structural_features


SEARCH_SPACE = {
    "convolution": ("gcn", "sage", "gat"),
    "fusion": ("concat", "sum", "gated"),
    "pooling": ("mean", "max", "attention"),
    "temporal": ("gru", "lstm"),
}
RELATION_COUNT = 6
GRAPH_BLOCK_ROWS = 128


def _relations(graph: ObservedGraph) -> list[Tensor]:
    """Return recorded arcs, reversed arcs, authorship, and user interaction edges."""
    posts = torch.arange(graph.num_posts, device=graph.x.device)
    known = graph.post_author >= 0
    author_edges = torch.stack((posts[known], graph.post_author[known] + graph.num_posts))
    src, dst = graph.edge_index
    known_pair = (graph.post_author[src] >= 0) & (graph.post_author[dst] >= 0)
    user_edges = torch.stack((graph.post_author[src[known_pair]] + graph.num_posts,
                              graph.post_author[dst[known_pair]] + graph.num_posts))
    if user_edges.numel():
        user_edges = torch.unique(user_edges, dim=1)
    return [graph.edge_index, graph.edge_index.flip(0), author_edges,
            author_edges.flip(0), user_edges, user_edges.flip(0)]


class _RelationConv(nn.Module):
    def __init__(self, hidden: int, operation: str, heads: int, dropout: float) -> None:
        super().__init__()
        self.operation = operation
        self.heads = heads
        self.self_linear = nn.Linear(hidden, hidden)
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(RELATION_COUNT))
        self.normalization = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        if operation == "gat":
            if hidden % heads:
                raise ValueError("D2 hidden width must be divisible by attention_heads")
            self.attention_source = nn.Parameter(torch.empty(RELATION_COUNT, heads, hidden // heads))
            self.attention_destination = nn.Parameter(torch.empty(RELATION_COUNT, heads, hidden // heads))
            nn.init.xavier_uniform_(self.attention_source)
            nn.init.xavier_uniform_(self.attention_destination)

    def forward(self, states: Tensor, relations: list[Tensor]) -> Tensor:
        size, width = states.shape
        result = self.self_linear(states)
        for relation, edges in enumerate(relations):
            if edges.shape[1] == 0:
                continue
            source, destination = edges
            transformed = self.messages[relation](states)
            messages = transformed[source]
            degree = torch.bincount(destination, minlength=size).to(states.dtype).clamp_min(1)
            if self.operation == "gcn":
                out_degree = torch.bincount(source, minlength=size).to(states.dtype).clamp_min(1)
                messages = messages / torch.sqrt(out_degree[source] * degree[destination]).unsqueeze(1)
            elif self.operation == "sage":
                messages = messages / degree[destination].unsqueeze(1)
            else:
                source_heads = messages.view(-1, self.heads, width // self.heads)
                destination_heads = transformed[destination].view_as(source_heads)
                attention = (source_heads * self.attention_source[relation]).sum(-1)
                attention = attention + (destination_heads * self.attention_destination[relation]).sum(-1)
                attention = softmax(F.leaky_relu(attention, negative_slope=0.2), destination, num_nodes=size)
                messages = (source_heads * attention.unsqueeze(-1)).reshape(-1, width)
            aggregated = states.new_zeros((size, width)).index_add(0, destination, messages)
            result = result + aggregated
        return self.normalization(states + self.dropout(F.gelu(result)))


class _HeterogeneousEncoder(nn.Module):
    def __init__(self, hidden: int, layers: int, dropout: float, heads: int,
                 operations: tuple[str, ...]) -> None:
        super().__init__()
        self.post_input = nn.Linear(8, hidden)
        self.user_type = nn.Parameter(torch.zeros(hidden))
        self.layers = nn.ModuleList(nn.ModuleDict({op: _RelationConv(hidden, op, heads, dropout)
                                                   for op in operations}) for _ in range(layers))
        self.operations = operations

    def forward(self, graph: ObservedGraph, weights: Tensor | None = None) -> Tensor:
        posts = F.gelu(self.post_input(graph.x))
        users = posts.new_zeros((graph.num_users, posts.shape[1]))
        known = graph.post_author >= 0
        if graph.num_users:
            users = users.index_add(0, graph.post_author[known], posts[known])
            degree = torch.bincount(graph.post_author[known], minlength=graph.num_users).clamp_min(1)
            users = users / degree.to(posts.dtype).unsqueeze(1) + self.user_type
        states = torch.cat((posts, users), dim=0)
        relations = _relations(graph)
        for layer in self.layers:
            outputs = [layer[op](states, relations) for op in self.operations]
            states = outputs[0] if len(outputs) == 1 else sum(weights[i] * value for i, value in enumerate(outputs))
        return states


class _HierarchicalPointer(nn.Module):
    """A normalized categorical distribution over each causal candidate prefix.

    Split decisions compare learned query/key scores of interval means. Prefix
    sums make each decision O(hidden), without a query-by-candidate score matrix.
    """
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.scale = math.sqrt(hidden)

    def _route(self, queries: Tensor, keys: Tensor, lengths: Tensor,
               target: Tensor | None) -> Tensor:
        if queries.shape[0] == 0:
            return queries.new_empty((0,)) if target is not None else lengths.new_empty((0,))
        if bool((lengths <= 0).any()) or int(lengths.max()) > keys.shape[0]:
            raise ValueError("D2 pointer has an empty or invalid candidate prefix")
        if target is not None and bool(((target < 0) | (target >= lengths)).any()):
            raise ValueError("D2 pointer target is outside the causal candidate prefix")
        q = self.query(queries)
        k = self.key(keys)
        prefix = torch.cat((k.new_zeros((1, k.shape[1])), k.cumsum(0)), dim=0)
        low, high = torch.zeros_like(lengths), lengths.clone()
        losses = queries.new_zeros((queries.shape[0],))
        for _ in range(int(lengths.max()).bit_length()):
            active = high - low > 1
            middle = (low + high) // 2
            left = (prefix[middle] - prefix[low]) / (middle - low).clamp_min(1).unsqueeze(1)
            right = (prefix[high] - prefix[middle]) / (high - middle).clamp_min(1).unsqueeze(1)
            logits = ((right - left) * q).sum(1) / self.scale
            choose_right = logits >= 0 if target is None else target >= middle
            if target is not None:
                losses = losses + F.binary_cross_entropy_with_logits(logits, choose_right.to(q.dtype), reduction="none") * active
            low = torch.where(active & choose_right, middle, low)
            high = torch.where(active & ~choose_right, middle, high)
        return losses if target is not None else low

    def loss(self, queries: Tensor, keys: Tensor, lengths: Tensor, target: Tensor) -> Tensor:
        return self._route(queries, keys, lengths, target)

    def choose(self, queries: Tensor, keys: Tensor, lengths: Tensor) -> Tensor:
        return self._route(queries, keys, lengths, None)


@dataclass(frozen=True)
class CompletionPrediction:
    observed: ObservedGraph
    query: Tensor
    parent_query: Tensor
    parent_keys: Tensor
    parent_candidates: Tensor
    author_keys: Tensor
    known_author_logits: Tensor
    coordinate_log_gap: Tensor
    count_log1p: Tensor
    expected_count: Tensor
    parent_direction_logits: Tensor
    graph_source_keys: Tensor
    graph_destination_keys: Tensor
    timing_future_logits: Tensor


class D2Forecaster(nn.Module):
    def __init__(self, max_completion_nodes: int, hidden: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, attention_heads: int = 4, text_dim: int = 768,
                 loss_weights: dict[str, float] | None = None) -> None:
        super().__init__()
        if max_completion_nodes < 0 or hidden <= 0 or num_layers <= 0:
            raise ValueError("D2 forecast capacity, width, and layer count are invalid")
        self.max_completion_nodes = int(max_completion_nodes)
        self.loss_weights = {"count": 1.0, "edge": 1.0, "author": 1.0, "coordinate": 1.0}
        if loss_weights:
            if set(loss_weights) - set(self.loss_weights):
                raise ValueError("Unknown D2 forecast loss component")
            self.loss_weights.update(loss_weights)
        self.encoder = _HeterogeneousEncoder(hidden, num_layers, dropout, attention_heads, ("sage",))
        self.text = nn.Linear(text_dim, hidden)
        self.context = nn.Sequential(nn.Linear(4 * hidden + 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.completion_queries = nn.Embedding(max(1, max_completion_nodes), hidden)
        self.query_network = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.count = nn.Linear(hidden, 1)
        self.coordinate = nn.Linear(hidden, 1)
        self.coordinate_condition = nn.Linear(1, hidden)
        self.author_known = nn.Linear(hidden, 1)
        self.anonymous_user = nn.Linear(hidden, hidden)
        self.parent_pointer = _HierarchicalPointer(hidden)
        self.author_pointer = _HierarchicalPointer(hidden)
        self.parent_direction = nn.Linear(hidden, 3)
        self.graph_source = nn.Linear(hidden, hidden, bias=False)
        self.graph_destination = nn.Linear(hidden, hidden, bias=False)
        self.graph_bias = nn.Parameter(torch.tensor(-4.0))
        self.graph_scale = math.sqrt(hidden)
        self.timing_future = nn.Linear(hidden, 1)
        nn.init.constant_(self.timing_future.bias, 4.0)

    def forward(self, observed: ObservedGraph) -> CompletionPrediction:
        if not isinstance(observed, ObservedGraph):
            raise TypeError("D2 forecaster accepts only label-free ObservedGraph inputs")
        states = self.encoder(observed)
        posts, users = states[:observed.num_posts], states[observed.num_posts:]
        user_mean = users.mean(0) if observed.num_users else states.new_zeros((states.shape[1],))
        metadata = states.new_tensor([math.log1p(observed.horizon), float(observed.unit == "hop")])
        context = self.context(torch.cat((posts.mean(0), user_mean, posts[observed.root],
                                          self.text(observed.text).view(-1), metadata)))
        query = self.query_network(self.completion_queries.weight[:self.max_completion_nodes] + context)
        coordinate_log_gap = F.softplus(self.coordinate(query).view(-1))
        parent_query = query + self.coordinate_condition(coordinate_log_gap.unsqueeze(1))
        candidates = torch.arange(observed.num_posts, device=states.device)
        if observed.unit == "hop":
            candidates = candidates[observed.coordinates == observed.horizon]
        # A connected hop completion cannot cross a boundary that does not exist.
        possible = observed.unit != "hop" or candidates.numel() > 0
        count_log1p = torch.sigmoid(self.count(context).view(())) * math.log1p(self.max_completion_nodes)
        if not possible:
            count_log1p = count_log1p * 0
        expected_count = torch.expm1(count_log1p)
        return CompletionPrediction(observed, query, parent_query,
                                    torch.cat((posts[candidates], query), dim=0), candidates,
                                    torch.cat((users, self.anonymous_user(query)), dim=0),
                                    self.author_known(query).view(-1), coordinate_log_gap,
                                    count_log1p, expected_count, self.parent_direction(query),
                                    self.graph_source(torch.cat((posts, query), dim=0)),
                                    self.graph_destination(torch.cat((posts, query), dim=0)),
                                    self.timing_future(query).view(-1))

    def _graph_loss(self, prediction: CompletionPrediction, target: CompletionTarget) -> Tensor:
        """Exact BCE against recorded directed adjacency, without pair sampling."""
        observed_count = prediction.observed.num_posts
        size = observed_count + target.count
        source_keys = prediction.graph_source_keys[:size]
        destination_keys = prediction.graph_destination_keys[:size]
        source, destination = target.graph_edges
        if (target.graph_edges.ndim != 2 or target.graph_edges.shape[0] != 2
                or bool(((source < 0) | (source >= size) | (destination < 0) | (destination >= size)).any())
                or bool(((source < observed_count) & (destination < observed_count)).any())):
            raise ValueError("Graph targets must be valid recorded arcs incident to unobserved nodes")
        total = source_keys.sum() * 0
        pair_count = size * size - observed_count * observed_count
        columns = torch.arange(size, device=source_keys.device)
        for start in range(0, size, GRAPH_BLOCK_ROWS):
            stop = min(size, start + GRAPH_BLOCK_ROWS)
            rows = torch.arange(start, stop, device=source_keys.device)
            incident = (rows[:, None] >= observed_count) | (columns[None, :] >= observed_count)
            logits = source_keys[start:stop] @ destination_keys.t() / self.graph_scale + self.graph_bias
            presence = torch.zeros_like(logits)
            in_block = (source >= start) & (source < stop)
            presence[source[in_block] - start, destination[in_block]] = 1
            total = total + F.binary_cross_entropy_with_logits(logits[incident], presence[incident], reduction="sum")
        return total / max(1, pair_count)

    def loss(self, prediction: CompletionPrediction, target: CompletionTarget) -> dict[str, Tensor]:
        count = int(target.count)
        if count > self.max_completion_nodes:
            raise ValueError(f"Real completion count {count} exceeds TRAIN capacity {self.max_completion_nodes}; targets were not truncated")
        if (count < 0 or target.author.shape != (count,) or target.coordinates.shape != (count,)
                or target.parent_direction.shape != (count,)):
            raise ValueError("D2 completion targets have inconsistent dimensions")
        observed = prediction.observed
        target = target.to(prediction.query.device)
        losses = {"count": F.smooth_l1_loss(prediction.count_log1p,
                                           prediction.count_log1p.new_tensor(math.log1p(count)))}
        zero = prediction.query.sum() * 0
        losses.update(edge=zero, author=zero, coordinate=zero)
        if count:
            if prediction.parent_candidates.numel() == 0:
                raise ValueError("Real target crosses a missing observed hop boundary")
            parent, destination = target.parent_edges
            if (target.parent_edges.shape != (2, count)
                    or not torch.equal(destination, torch.arange(count, device=destination.device))
                    or bool(((target.parent_direction < 0) | (target.parent_direction > 2)).any())):
                raise ValueError("Every real unobserved node needs one valid attachment and its recorded direction")
            parent_map = torch.full((observed.num_posts,), -1, dtype=torch.long, device=prediction.query.device)
            parent_map[prediction.parent_candidates] = torch.arange(len(prediction.parent_candidates), device=parent_map.device)
            is_observed = parent < observed.num_posts
            mapped = parent - observed.num_posts + len(prediction.parent_candidates)
            mapped = torch.where(is_observed, parent_map[parent.clamp(0, observed.num_posts - 1)], mapped)
            lengths = len(prediction.parent_candidates) + destination
            losses["edge"] = self.parent_pointer.loss(prediction.parent_query[destination], prediction.parent_keys,
                                                        lengths, mapped).mean()
            losses["edge"] = (losses["edge"]
                              + F.cross_entropy(prediction.parent_direction_logits[:count], target.parent_direction)
                              + self._graph_loss(prediction, target))
            known = target.author >= 0
            losses["author"] = F.binary_cross_entropy_with_logits(prediction.known_author_logits[:count], known.to(zero.dtype))
            if bool(known.any()):
                rows = torch.arange(count, device=known.device)[known]
                identity_loss = self.author_pointer.loss(prediction.query[rows], prediction.author_keys,
                                                         observed.num_users + rows + 1, target.author[known])
                losses["author"] = losses["author"] + identity_loss.mean()
            signed_gaps = target.coordinates.to(prediction.query.dtype) - observed.horizon
            if observed.unit == "hop" and bool((signed_gaps <= 0).any()):
                raise ValueError("Real hop completion is not beyond the observation horizon")
            losses["coordinate"] = F.smooth_l1_loss(prediction.coordinate_log_gap[:count],
                                                    torch.log1p(signed_gaps.abs()))
            if observed.unit == "minutes":
                losses["coordinate"] = (losses["coordinate"]
                                        + F.binary_cross_entropy_with_logits(
                                            prediction.timing_future_logits[:count],
                                            (signed_gaps > 0).to(prediction.query.dtype)))
        losses["total"] = sum(self.loss_weights[name] * losses[name] for name in self.loss_weights)
        return losses

    @torch.no_grad()
    def complete(self, observed: ObservedGraph, prediction: CompletionPrediction) -> ObservedGraph:
        if prediction.observed is not observed:
            raise ValueError("Completion prediction is bound to a different observed input")
        if not bool(torch.isfinite(prediction.expected_count)):
            raise ValueError("Non-finite predicted completion count")
        count = int(torch.round(prediction.expected_count).item())
        if not 0 <= count <= self.max_completion_nodes:
            raise ValueError("Predicted completion is outside its TRAIN-derived capacity")
        if count == 0:
            return observed.detach()
        rows = torch.arange(count, device=observed.x.device)
        boundary = len(prediction.parent_candidates)
        parent_choice = self.parent_pointer.choose(prediction.parent_query[:count], prediction.parent_keys, boundary + rows)
        parents = torch.where(parent_choice < boundary,
                              prediction.parent_candidates[parent_choice.clamp(max=boundary - 1)],
                              observed.num_posts + parent_choice - boundary)
        author_choice = self.author_pointer.choose(prediction.query[:count], prediction.author_keys,
                                                   observed.num_users + rows + 1)
        known_author = prediction.known_author_logits[:count] >= 0
        coordinates = observed.coordinates.detach().cpu().tolist()
        parents_cpu = parents.detach().cpu().tolist()
        gaps = torch.expm1(prediction.coordinate_log_gap[:count]).detach().cpu()
        if not bool(torch.isfinite(gaps).all()):
            raise ValueError("Non-finite predicted native coordinate")
        timing_future = (prediction.timing_future_logits[:count] >= 0).detach().cpu().tolist()
        for i, parent in enumerate(parents_cpu):
            if observed.unit == "hop":
                coordinate = int(coordinates[parent]) + 1
            else:
                # Censored history must remain behind a generated future bridge.
                # A node attached directly to an observed node is therefore future.
                if parent < observed.num_posts or timing_future[i]:
                    coordinate = observed.horizon + max(1, math.ceil(float(gaps[i])))
                else:
                    coordinate = max(0, observed.horizon - round(float(gaps[i])))
            coordinates.append(coordinate)
        new_authors: list[int] = []
        anonymous: dict[int, int] = {}
        for slot, known in zip(author_choice.detach().cpu().tolist(), known_author.detach().cpu().tolist()):
            if not known:
                new_authors.append(-1)
            elif slot < observed.num_users:
                new_authors.append(slot)
            else:
                if slot not in anonymous:
                    anonymous[slot] = observed.num_users + len(anonymous)
                new_authors.append(anonymous[slot])
        completion_nodes = observed.num_posts + rows
        direction = prediction.parent_direction_logits[:count].argmax(1)
        forward = direction != 1
        reverse = direction != 0
        generated_edges = [torch.stack((parents[forward], completion_nodes[forward])),
                           torch.stack((completion_nodes[reverse], parents[reverse]))]
        coordinate_tensor = torch.tensor(coordinates, dtype=torch.long, device=observed.x.device)
        size = observed.num_posts + count
        columns = torch.arange(size, device=observed.x.device)
        destination_keys = prediction.graph_destination_keys[:size]
        for start in range(0, size, GRAPH_BLOCK_ROWS):
            stop = min(size, start + GRAPH_BLOCK_ROWS)
            source_rows = torch.arange(start, stop, device=observed.x.device)
            allowed = (source_rows[:, None] >= observed.num_posts) | (columns[None, :] >= observed.num_posts)
            if observed.unit == "hop":
                source_coordinates = coordinate_tensor[source_rows, None]
                destination_coordinates = coordinate_tensor[None, :]
                allowed = (allowed & ((source_coordinates - destination_coordinates).abs() <= 1)
                           & (source_coordinates >= observed.horizon)
                           & (destination_coordinates >= observed.horizon))
            else:
                # Reapplying the root-connected cutoff must recover exactly the
                # observed nodes, even when completion predicts censored history.
                source_history = ((source_rows >= observed.num_posts)
                                  & (coordinate_tensor[source_rows] <= observed.horizon))
                destination_history = ((columns >= observed.num_posts)
                                       & (coordinate_tensor <= observed.horizon))
                forbidden = ((source_history[:, None] & (columns[None, :] < observed.num_posts))
                             | ((source_rows[:, None] < observed.num_posts) & destination_history[None, :]))
                allowed = allowed & ~forbidden
            logits = (prediction.graph_source_keys[start:stop] @ destination_keys.t()
                      / self.graph_scale + self.graph_bias)
            selected = torch.nonzero(allowed & (logits >= 0), as_tuple=False)
            if selected.numel():
                generated_edges.append(torch.stack((selected[:, 0] + start, selected[:, 1])))
        # Original observed edge order and multiplicity are immutable.
        new_edges = torch.unique(torch.cat(generated_edges, dim=1), dim=1)
        edges = torch.cat((observed.edge_index, new_edges), dim=1)
        edge_pairs = edges.detach().cpu().t().tolist()
        features = compute_structural_features(nodes=[], edges=[], keep_nodes=list(range(len(coordinates))),
                                                reindexed_edges=edge_pairs, root_index_new=observed.root,
                                                feature_time_scale=max(1, max(coordinates)), shifted_times_list=coordinates)
        authors = torch.cat((observed.post_author, torch.tensor(new_authors, dtype=torch.long, device=observed.x.device)))
        return ObservedGraph(features.to(observed.x.device), edges, coordinate_tensor, authors,
                             observed.num_users + len(anonymous), observed.text.detach(),
                             observed.root, observed.unit, observed.horizon)


class _Fusion(nn.Module):
    def __init__(self, hidden: int, operation: str) -> None:
        super().__init__()
        self.operation = operation
        if operation == "concat":
            self.combine = nn.Linear(3 * hidden, hidden)
        else:
            self.projections = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))
            if operation == "gated":
                self.gate = nn.Linear(3 * hidden, 3)

    def forward(self, observed: Tensor, completed: Tensor, text: Tensor) -> Tensor:
        values = [observed, completed, text]
        if self.operation == "concat":
            return F.gelu(self.combine(torch.cat(values, dim=-1)))
        transformed = torch.stack([projection(value) for projection, value in zip(self.projections, values)], dim=0)
        if self.operation == "sum":
            return F.gelu(transformed.sum(0))
        weights = torch.softmax(self.gate(torch.cat(values, dim=-1)), dim=-1).transpose(0, 1).unsqueeze(-1)
        return F.gelu((transformed * weights).sum(0))


class D2Classifier(nn.Module):
    """Weight-sharing 3x3x3x2 search, or one freshly initialized discrete model."""
    def __init__(self, num_classes: int, hidden: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, attention_heads: int = 4, text_dim: int = 768,
                 architecture: dict[str, str] | None = None) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("D2 classifier needs at least two declared classes")
        if architecture is not None:
            if set(architecture) != set(SEARCH_SPACE) or any(architecture[axis] not in choices for axis, choices in SEARCH_SPACE.items()):
                raise ValueError("Selected architecture is outside the declared 54-architecture search space")
        self.architecture = dict(architecture) if architecture is not None else None
        self.choices = {axis: (architecture[axis],) if architecture is not None else choices
                        for axis, choices in SEARCH_SPACE.items()}
        self.alphas = nn.ParameterDict({axis: nn.Parameter(torch.zeros(len(choices)))
                                        for axis, choices in SEARCH_SPACE.items()}) if architecture is None else nn.ParameterDict()
        self.encoder = _HeterogeneousEncoder(hidden, num_layers, dropout, attention_heads, self.choices["convolution"])
        self.attention_pool = nn.Linear(hidden, 1) if "attention" in self.choices["pooling"] else None
        self.temporal = nn.ModuleDict({name: (nn.GRU(hidden, hidden, batch_first=True) if name == "gru"
                                              else nn.LSTM(hidden, hidden, batch_first=True)) for name in self.choices["temporal"]})
        self.view_projection = nn.Linear(2 * hidden, hidden)
        self.text = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU())
        self.fusions = nn.ModuleDict({name: _Fusion(hidden, name) for name in self.choices["fusion"]})
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def architecture_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.alphas.parameters())

    def weight_parameters(self) -> Iterator[nn.Parameter]:
        architecture_ids = {id(parameter) for parameter in self.alphas.parameters()}
        return (parameter for parameter in self.parameters() if id(parameter) not in architecture_ids)

    def selected_architecture(self) -> dict[str, str]:
        if self.architecture is not None:
            return dict(self.architecture)
        return {axis: choices[int(self.alphas[axis].detach().argmax())] for axis, choices in SEARCH_SPACE.items()}

    def _weights(self, axis: str) -> Tensor | None:
        return torch.softmax(self.alphas[axis], dim=0) if axis in self.alphas else None

    def _mix(self, axis: str, values: list[Tensor]) -> Tensor:
        return values[0] if len(values) == 1 else sum(weight * value for weight, value in zip(self._weights(axis), values))

    def _encode(self, graph: ObservedGraph) -> Tensor:
        states = self.encoder(graph, self._weights("convolution"))
        pooled = []
        for operation in self.choices["pooling"]:
            if operation == "mean":
                pooled.append(states.mean(0, keepdim=True))
            elif operation == "max":
                pooled.append(states.max(0, keepdim=True).values)
            else:
                attention = torch.softmax(self.attention_pool(states).view(-1), dim=0)
                pooled.append((states * attention.unsqueeze(1)).sum(0, keepdim=True))
        sequence = states[:graph.num_posts][torch.argsort(graph.coordinates, stable=True)].unsqueeze(0)
        temporal = []
        for operation in self.choices["temporal"]:
            _, final_state = self.temporal[operation](sequence)
            hidden = final_state[0] if operation == "lstm" else final_state
            temporal.append(hidden[-1])
        return F.gelu(self.view_projection(torch.cat((self._mix("pooling", pooled), self._mix("temporal", temporal)), dim=-1)))

    def forward(self, observed: ObservedGraph, completed: ObservedGraph) -> Tensor:
        if not isinstance(observed, ObservedGraph) or not isinstance(completed, ObservedGraph):
            raise TypeError("D2 classifier accepts only label-free observed and predicted graphs")
        observed_state, completed_state = self._encode(observed), self._encode(completed)
        text = self.text(observed.text)
        fused = [self.fusions[name](observed_state, completed_state, text) for name in self.choices["fusion"]]
        return self.output(self._mix("fusion", fused))
