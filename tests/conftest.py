from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture()
def synthetic_graph():
    geometric = pytest.importorskip("torch_geometric.data")
    data = geometric.Data(
        x=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.2, 0.8]]),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        y=torch.tensor([1], dtype=torch.long),
        root_index=torch.tensor([0], dtype=torch.long),
    )
    data.source_id = "synthetic-1"
    data.node_time_minutes = torch.tensor([0, 10, 20], dtype=torch.long)
    data.node_time_minutes_shifted = torch.tensor([0, 10, 20], dtype=torch.long)
    data.time_unit = "minutes"
    data.node_uid = ["u0", "u1", "u2"]
    data.node_tweet_id = ["t0", "t1", "t2"]
    return data
