import os
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from Libs.data.potential_cache import load_validated_potential_cache, potential_cache_payload


def _ensure_dir(path: str) -> None:
    """Creates a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _extract_source_ids_from_batch(batch) -> List[str]:
    """Extracts string source_ids from a PyG Batch.

    This function supports both tensor and non-tensor forms of `source_id`.
    """
    assert hasattr(batch, "source_id"), "Batch is missing source_id field"
    sid = batch.source_id
    out: List[str] = []
    if torch.is_tensor(sid):
        if sid.dim() == 0:
            out.append(str(int(sid.item())))
        else:
            for i in range(int(sid.size(0))):
                out.append(str(int(sid[i].item())))
    else:
        if isinstance(sid, (list, tuple)):
            for v in sid:
                out.append(str(v))
        else:
            out.append(str(sid))
    assert len(out) > 0
    return out


@torch.no_grad()
def generate_potentials(
    model,  # CTDMSyn instance
    dataset,
    out_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    device: Optional[str] = None,
    stride: Optional[int] = None,
    skip_existing: bool = True,
    cache_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Generates and caches class-conditional potential vectors for a dataset.

    Each sample i produces a schema-v2 payload with `h: [C, H]` and metadata
    that binds the cache to one dataset, seed, observed-prefix policy, and CTDM.
    """
    if not isinstance(cache_metadata, dict):
        raise ValueError("RumorDiffusion potential generation requires cache metadata.")
    _ensure_dir(out_dir)
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()

    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader = PyGDataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=int(num_workers) > 0,
    )

    for batch in loader:
        batch = batch.to(dev)
        sids = _extract_source_ids_from_batch(batch)
        paths = [os.path.join(out_dir, f"{sid}.pt") for sid in sids]

        write_entry = [True for _ in paths]
        if skip_existing:
            for i, (sid, path) in enumerate(zip(sids, paths)):
                if not os.path.isfile(path):
                    continue
                cached = torch.load(path, weights_only=False)
                load_validated_potential_cache(
                    payload=cached,
                    expected_metadata=cache_metadata,
                    source_id=sid,
                    cache_path=path,
                )
                write_entry[i] = False
            if not any(write_entry):
                continue

        Hc: Tensor = model.infer_potentials_all(batch, stride=stride)  # [B, C, H]
        assert Hc.dim() == 3
        assert Hc.size(0) == len(sids)
        for i, sid in enumerate(sids):
            if not write_entry[i]:
                continue
            out_path = os.path.join(out_dir, f"{sid}.pt")
            torch.save(potential_cache_payload(Hc[i], cache_metadata, sid), out_path)


@torch.no_grad()
def generate_prefix_full_embeddings(
    model,  # CTDMSyn instance
    dataset,
    out_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    device: Optional[str] = None,
    skip_existing: bool = True,
) -> None:
    """Caches prefix/full representations and x0 projections per sample.

    Saves `{out_dir}/{source_id}.pt` with keys:
      - `c`: prefix/observed graph representation
      - `h0`: full-graph representation
      - `x0`: projected latent from full graph
    """
    _ensure_dir(out_dir)
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()

    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader = PyGDataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=int(num_workers) > 0,
    )

    for batch in loader:
        batch = batch.to(dev)
        sids = _extract_source_ids_from_batch(batch)
        paths = [os.path.join(out_dir, f"{sid}.pt") for sid in sids]

        if skip_existing:
            all_exist = True
            for p in paths:
                if not os.path.isfile(p):
                    all_exist = False
                    break
            if all_exist:
                continue

        c = model.encode_prefix(batch)   # [B, H]
        h0 = model.encode_full(batch)    # [B, H]
        x0 = model.to_x0(h0)             # [B, H]

        assert c.size(0) == len(sids) == h0.size(0) == x0.size(0)
        for i, sid in enumerate(sids):
            out_path = os.path.join(out_dir, f"{sid}.pt")
            torch.save({"c": c[i].detach().cpu(), "h0": h0[i].detach().cpu(), "x0": x0[i].detach().cpu()}, out_path)

