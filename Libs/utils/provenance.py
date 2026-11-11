"""Provenance capture helpers shared by canonical runners and validators."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from Libs.utils.artifact_schema import canonical_json_hash, file_sha256


def git_state(repo_root: str | Path = ".") -> dict[str, Any]:
    root = str(Path(repo_root))

    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    diff = subprocess.run(["git", "diff", "--binary"], cwd=root, check=True, capture_output=True).stdout
    return {"git_commit": commit, "dirty": bool(status), "dirty_diff_sha256": hashlib.sha256(diff).hexdigest()}


def environment_snapshot(repo_root: str | Path = ".") -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
    }
    lock_candidates = ("requirements.txt", "requirements.lock", "poetry.lock", "uv.lock")
    snapshot["dependency_lock_hashes"] = {
        name: file_sha256(Path(repo_root) / name) for name in lock_candidates if (Path(repo_root) / name).is_file()
    }
    try:
        import torch

        snapshot["pytorch"] = torch.__version__
        snapshot["cuda"] = torch.version.cuda
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        snapshot["pytorch_error"] = type(exc).__name__
    return snapshot


def resolved_config_record(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path_obj = Path(path)
    return {"path": str(path_obj), "sha256": canonical_json_hash(payload), "payload": dict(payload)}


def artifact_record(path: str | Path, schema_version: str, **extra: Any) -> dict[str, Any]:
    record = {"path": str(path), "sha256": file_sha256(path), "schema_version": schema_version}
    record.update(extra)
    return record


def _json_safe(value: Any) -> Any:
    """Convert common scientific scalar/array objects to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return _json_safe(value.tolist())
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def deterministic_run_id(canonical_inputs: Mapping[str, Any], *, prefix: str = "run") -> str:
    """Stable run identifier from protocol inputs, never from wall-clock time."""
    if not canonical_inputs:
        raise ValueError("canonical inputs are required for a run ID")
    return f"{prefix}-{canonical_json_hash(canonical_inputs)[:20]}"


def resolved_file_record(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    return {"path": str(target), "sha256": file_sha256(target)}


def write_run_environment(path: str | Path, *, repo_root: str | Path = ".", **overrides: Any) -> dict[str, Any]:
    payload = environment_snapshot(repo_root)
    payload.update(overrides)
    write_json(path, payload)
    return {"path": str(path), "sha256": file_sha256(path), "schema_version": "1"}
