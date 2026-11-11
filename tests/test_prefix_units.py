from __future__ import annotations

import pytest
import torch

from Libs.data.dataset import _prefix_mask_from_strategy
from Libs.data.processor import build_pyg_data, select_observation_view


def test_hop_strategy_requires_explicit_hop_value():
    times = torch.tensor([0, 1, 2], dtype=torch.long)
    assert _prefix_mask_from_strategy(times, "hop", minutes=1, k=None, root_index=0, include_root=True).tolist() == [True, True, False]


def test_unknown_prefix_strategy_fails_closed():
    with pytest.raises(ValueError):
        _prefix_mask_from_strategy(torch.tensor([0]), "future_size", 1, None, 0, True)


def test_hop_samples_do_not_reuse_minutes_tensor():
    data = build_pyg_data(
        nodes=[("u0", "t0", 0), ("u1", "t1", 1), ("u2", "t2", 2)],
        edges=[(0, 1), (1, 2)],
        keep_nodes=[0, 1, 2],
        reindexed_edges=[(0, 1), (1, 2)],
        root_index_new=0,
        feature_time_scale=2,
        shifted_times_list=[0, 1, 2],
        label_id=0,
        source_id="hop-sample",
        time_unit="hop",
    )
    assert data.node_time_minutes.tolist() == [0, 0, 0]
    assert data.node_time_hops.tolist() == [0, 1, 2]


def test_truncation_unit_mismatch_fails_closed():
    with pytest.raises(ValueError, match="does not match"):
        select_observation_view(
            [("u0", "t0", 0), ("u1", "t1", 1)],
            [(0, 1)],
            "window",
            1,
            True,
            truncation_unit="minutes",
            source_time_unit="hop",
        )
