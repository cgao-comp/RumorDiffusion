from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from Libs.scripts.build_source_release import build_release

FIXED_DATE = "2026-01-02T03:04:05+00:00"


def _git(root: Path, *args: str) -> str:
    environment = {**os.environ, "GIT_AUTHOR_DATE": FIXED_DATE, "GIT_COMMITTER_DATE": FIXED_DATE}
    completed = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=environment)
    return completed.stdout.strip()


@pytest.fixture()
def release_repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    (root / "Libs/scripts").mkdir(parents=True)
    (root / "Libs/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    launcher = root / "Libs/scripts/run.sh"
    launcher.write_text("#!/bin/bash\necho run\n", encoding="utf-8")
    launcher.chmod(0o755)
    (root / "TODO.md").write_text("internal kanban\n", encoding="utf-8")
    (root / "README.md").write_text("committed readme\n", encoding="utf-8")
    (root / ".gitignore").write_text("Input/\nOutput/\nrelease.yaml\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Release Fixture")
    _git(root, "config", "user.email", "fixture@example.org")
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "fixture source")
    revision = _git(root, "rev-parse", "HEAD")

    (root / "README.md").write_text("release readme\n", encoding="utf-8")
    processed = root / "Input/processed/ds"
    processed.mkdir(parents=True)
    (processed / "split_manifest.json").write_text('{"splits": {}}\n', encoding="utf-8")
    (processed / "data.pt").write_bytes(b"\0" * 64)
    (root / "Input/raw/ds").mkdir(parents=True)
    (root / "Input/raw/ds/cascade.json").write_text("{}\n", encoding="utf-8")
    (root / "Output/canonical").mkdir(parents=True)
    return root, revision


def _spec(revision: str) -> dict[str, Any]:
    return {
        "source_release": {
            "name": "Fixture",
            "source_revision": revision,
            "archive": {
                "output_dir": "../release",
                "file_stem": "fixture_source",
                "root_dir": "Fixture",
                "manifest_name": "release_manifest.yaml",
                "compression_level": 9,
            },
            "max_file_bytes": 4096,
            "exclude": ["TODO.md"],
            "overlay": ["README.md"],
            "data_manifests": ["Input/processed/*/split_manifest.json"],
            "keep_dirs": ["Input/raw/ds", "Output/canonical"],
            "keep_file_name": ".keep",
            "forbidden_suffixes": [".pt", ".log"],
            "forbidden_path_parts": ["__pycache__", ".git"],
        }
    }


def _write_spec(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "release.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_release_archive_contains_declared_selection(release_repo: tuple[Path, str]) -> None:
    root, revision = release_repo
    result = build_release(_write_spec(root, _spec(revision)), root, dry_run=False, replace=False)

    assert result.archive_path == root.parent / "release" / f"fixture_source_{revision[:12]}.zip"
    with zipfile.ZipFile(result.archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "Fixture/.gitignore",
            "Fixture/Input/processed/ds/split_manifest.json",
            "Fixture/Input/raw/ds/.keep",
            "Fixture/Libs/model.py",
            "Fixture/Libs/scripts/run.sh",
            "Fixture/Output/canonical/.keep",
            "Fixture/README.md",
            "Fixture/release_manifest.yaml",
        ]
        assert archive.read("Fixture/README.md") == b"release readme\n"
        assert archive.read("Fixture/Input/raw/ds/.keep") == b""
        assert (archive.getinfo("Fixture/Libs/scripts/run.sh").external_attr >> 16) & 0o111
        assert not (archive.getinfo("Fixture/Libs/model.py").external_attr >> 16) & 0o111
        manifest = yaml.safe_load(archive.read("Fixture/release_manifest.yaml"))
        for record in manifest["files"]:
            assert hashlib.sha256(archive.read(f"Fixture/{record['path']}")).hexdigest() == record["sha256"]

    assert manifest["source_revision"] == revision
    assert manifest["excluded_from_revision"] == ["TODO.md"]
    assert {record["path"]: record["origin"] for record in manifest["files"]} == {
        ".gitignore": "revision",
        "Input/processed/ds/split_manifest.json": "data_manifest",
        "Input/raw/ds/.keep": "placeholder",
        "Libs/model.py": "revision",
        "Libs/scripts/run.sh": "revision",
        "Output/canonical/.keep": "placeholder",
        "README.md": "overlay",
    }
    digest = hashlib.sha256(result.archive_path.read_bytes()).hexdigest()
    assert result.archive_sha256 == digest
    sidecar = Path(f"{result.archive_path}.sha256").read_text(encoding="utf-8")
    assert sidecar == f"{digest}  {result.archive_path.name}\n"


def test_release_archive_is_reproducible_and_never_silently_overwritten(release_repo: tuple[Path, str]) -> None:
    root, revision = release_repo
    config = _write_spec(root, _spec(revision))
    first = build_release(config, root, dry_run=False, replace=False)
    with pytest.raises(FileExistsError):
        build_release(config, root, dry_run=False, replace=False)
    second = build_release(config, root, dry_run=False, replace=True)
    assert second.archive_sha256 == first.archive_sha256


def test_dry_run_writes_nothing(release_repo: tuple[Path, str]) -> None:
    root, revision = release_repo
    result = build_release(_write_spec(root, _spec(revision)), root, dry_run=True, replace=False)
    assert result.archive_path is None
    assert result.manifest["file_count"] == 7
    assert not (root.parent / "release").exists()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("exclude", ["missing.md"], ValueError),
        ("data_manifests", ["Input/processed/*/missing.json"], FileNotFoundError),
        ("data_manifests", ["Input/processed/*/data.pt"], ValueError),
        ("overlay", ["TODO.md"], ValueError),
        ("keep_dirs", ["Input/processed/ds"], ValueError),
        ("keep_dirs", ["Input/absent"], FileNotFoundError),
        ("max_file_bytes", 8, ValueError),
        ("source_revision", "HEAD", ValueError),
    ],
)
def test_release_selection_fails_closed(release_repo: tuple[Path, str], field: str, value: Any, error: type) -> None:
    root, revision = release_repo
    payload = copy.deepcopy(_spec(revision))
    payload["source_release"][field] = value
    with pytest.raises(error):
        build_release(_write_spec(root, payload), root, dry_run=True, replace=False)


def test_release_archive_must_be_written_outside_repository(release_repo: tuple[Path, str]) -> None:
    root, revision = release_repo
    payload = copy.deepcopy(_spec(revision))
    payload["source_release"]["archive"]["output_dir"] = "release"
    with pytest.raises(ValueError, match="outside the repository"):
        build_release(_write_spec(root, payload), root, dry_run=True, replace=False)
