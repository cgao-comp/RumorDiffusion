from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Batch, Data

from Libs.data.dataset import _materialize_prefix_view, _root_connected_mask, _unique_union_edges
from Libs.models.baselines.CTDM_syn import CTDMSyn


def test_eval_prefix_omits_mature_state_and_is_prefix_local(synthetic_graph):
    synthetic_graph.full_x = synthetic_graph.x.clone()
    synthetic_graph.full_edge_index = synthetic_graph.edge_index.clone()
    synthetic_graph.full_node_time_minutes = synthetic_graph.node_time_minutes.clone()
    synthetic_graph.full_node_time_minutes_shifted = synthetic_graph.node_time_minutes_shifted.clone()
    view = _materialize_prefix_view(
        synthetic_graph.clone(),
        torch.tensor([True, True, False]),
        torch.tensor([[0], [1]], dtype=torch.long),
        preserve_mature_state=False,
    )
    assert not hasattr(view, "full_x")
    assert not hasattr(view, "full_edge_index")
    assert view.num_nodes == 2
    assert view.edge_index.tolist() == [[0], [1]]


def test_root_component_filter_drops_disconnected_observed_nodes():
    mask = _root_connected_mask(
        torch.tensor([True, True, True]),
        torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        root_index=0,
    )
    assert mask.tolist() == [True, True, False]


def test_root_component_uses_after_dropout_edges():
    mask = _root_connected_mask(
        torch.tensor([True, True, True]),
        torch.tensor([[0], [1]], dtype=torch.long),
        root_index=0,
    )
    assert mask.tolist() == [True, True, False]


def test_empty_edge_lists_are_valid_boundary_inputs():
    merged = _unique_union_edges(
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
        num_nodes=1,
    )
    assert tuple(merged.shape) == (2, 0)


def _ctdm_with_mature_state() -> tuple[CTDMSyn, Data]:
    model = CTDMSyn(
        in_channels=3,
        num_classes=2,
        hidden_channels=4,
        num_layers=1,
        dropout=0.0,
        T=2,
        obs_minutes=1,
        t_embed_dim=4,
        film_hidden=4,
        max_nodes=4,
    )
    data = Data(
        x=torch.ones((2, 3), dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        y=torch.tensor([0], dtype=torch.long),
    )
    data.full_x = torch.ones((3, 3), dtype=torch.float32)
    data.full_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return model, data


def test_ctdm_eval_rejects_mature_state_outside_objective_context():
    model, data = _ctdm_with_mature_state()
    model.eval()

    with pytest.raises(RuntimeError, match="mature-state attributes"):
        model._tensorize_full(data)

    with model.objective_eval_target_access():
        adjacency, features = model._tensorize_full(data)

    assert tuple(adjacency.shape) == (model.max_nodes, model.max_nodes)
    assert tuple(features.shape) == (model.max_nodes, model.in_channels)


def test_ctdm_validation_step_sets_objective_context(monkeypatch):
    model, data = _ctdm_with_mature_state()
    model.eval()
    called = {"value": False}

    def fake_objective(batch, update_prototypes):
        called["value"] = True
        assert update_prototypes is False
        assert model._mature_state_target_access_allowed()
        first = batch.to_data_list()[0]
        model._tensorize_full(first)
        return torch.tensor(0.0)

    monkeypatch.setattr(model, "_objective_loss", fake_objective)
    assert float(model.validation_step(Batch.from_data_list([data])).item()) == 0.0
    assert called["value"]
