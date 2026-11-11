#!/usr/bin/env python3
"""Build a source-only release archive from a committed source revision.

The archive holds every blob of the configured source revision except declared
exclusions, plus declared working-tree overlays, ignored input manifests, and
placeholder files for runtime directories. A release manifest at the archive
root records the source revision and the SHA-256 digest, size, and origin of
every entry. Unmatched patterns, oversized files, forbidden artifact types,
unpinned revisions, and output locations inside the repository stop the build.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from Libs.utils.io import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT_ID = re.compile(r"^[0-9a-f]{40}$")
REGULAR_FILE = 0o100644
EXECUTABLE_FILE = 0o100755


@dataclass(frozen=True)
class ReleaseSpec:
    name: str
    source_revision: str
    output_dir: Path
    file_stem: str
    root_dir: str
    manifest_name: str
    compression_level: int
    max_file_bytes: int
    exclude: tuple[str, ...]
    overlay: tuple[str, ...]
    data_manifests: tuple[str, ...]
    keep_dirs: tuple[str, ...]
    keep_file_name: str
    forbidden_suffixes: frozenset[str]
    forbidden_path_parts: frozenset[str]


@dataclass(frozen=True)
class Entry:
    path: str
    data: bytes
    mode: int
    origin: str


@dataclass(frozen=True)
class ReleaseResult:
    manifest: dict[str, Any]
    archive_path: Path | None
    archive_sha256: str | None


def _field(mapping: Mapping[str, Any], key: str, kind: type, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{context} is missing '{key}'")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, kind):
        raise TypeError(f"{context}.{key} must be {kind.__name__}, got {type(value).__name__}")
    return value


def _string_list(mapping: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    values = _field(mapping, key, list, context)
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{context}.{key} must contain non-empty strings")
    return tuple(values)


def load_spec(config_path: Path, repo_root: Path) -> ReleaseSpec:
    """Loads and type-checks the release specification."""
    payload = read_yaml(str(config_path))
    if not isinstance(payload, dict):
        raise TypeError(f"Release configuration must be a mapping: {config_path}")
    section = _field(payload, "source_release", dict, str(config_path))
    archive = _field(section, "archive", dict, "source_release")
    output_dir = Path(_field(archive, "output_dir", str, "source_release.archive"))
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    root_dir = _field(archive, "root_dir", str, "source_release.archive")
    if not root_dir or "/" in root_dir or root_dir in {".", ".."}:
        raise ValueError(f"source_release.archive.root_dir must be a single directory name: {root_dir!r}")
    return ReleaseSpec(
        name=_field(section, "name", str, "source_release"),
        source_revision=_field(section, "source_revision", str, "source_release"),
        output_dir=Path(os.path.normpath(output_dir)),
        file_stem=_field(archive, "file_stem", str, "source_release.archive"),
        root_dir=root_dir,
        manifest_name=_field(archive, "manifest_name", str, "source_release.archive"),
        compression_level=_field(archive, "compression_level", int, "source_release.archive"),
        max_file_bytes=_field(section, "max_file_bytes", int, "source_release"),
        exclude=_string_list(section, "exclude", "source_release"),
        overlay=_string_list(section, "overlay", "source_release"),
        data_manifests=_string_list(section, "data_manifests", "source_release"),
        keep_dirs=tuple(value.rstrip("/") for value in _string_list(section, "keep_dirs", "source_release")),
        keep_file_name=_field(section, "keep_file_name", str, "source_release"),
        forbidden_suffixes=frozenset(value.lower() for value in _string_list(section, "forbidden_suffixes", "source_release")),
        forbidden_path_parts=frozenset(_string_list(section, "forbidden_path_parts", "source_release")),
    )


def _git(repo_root: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def resolve_revision(repo_root: Path, revision: str) -> tuple[str, int]:
    """Returns the pinned commit and its commit timestamp."""
    if COMMIT_ID.fullmatch(revision) is None:
        raise ValueError(f"source_revision must be a full 40-character commit id: {revision!r}")
    resolved = _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    if resolved != revision:
        raise ValueError(f"source_revision resolved to a different commit: {resolved}")
    timestamp = int(_git(repo_root, "show", "-s", "--format=%ct", resolved).decode().strip())
    return resolved, timestamp


def revision_blobs(repo_root: Path, revision: str) -> dict[str, tuple[int, str]]:
    """Maps every path of the revision to its file mode and blob id."""
    blobs: dict[str, tuple[int, str]] = {}
    for record in _git(repo_root, "ls-tree", "-r", "-z", "--full-tree", revision).split(b"\0"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = meta.decode("ascii").split(" ")
        path = raw_path.decode("utf-8")
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"Unsupported tree entry in {revision}: {path} ({mode} {kind})")
        blobs[path] = (int(mode, 8), object_id)
    return blobs


def _file_mode(path: Path) -> int:
    return EXECUTABLE_FILE if path.stat().st_mode & 0o111 else REGULAR_FILE


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _excluded_paths(blobs: Mapping[str, tuple[int, str]], patterns: tuple[str, ...]) -> set[str]:
    excluded: set[str] = set()
    for pattern in patterns:
        hits = {path for path in blobs if fnmatch.fnmatchcase(path, pattern)}
        if not hits:
            raise ValueError(f"Exclude pattern matches no path in the source revision: {pattern}")
        excluded |= hits
    return excluded


def collect_entries(spec: ReleaseSpec, repo_root: Path, revision: str) -> tuple[dict[str, Entry], list[str]]:
    """Selects revision blobs, overlays, data manifests, and placeholders."""
    blobs = revision_blobs(repo_root, revision)
    excluded = _excluded_paths(blobs, spec.exclude)
    entries: dict[str, Entry] = {}
    for path in sorted(set(blobs) - excluded):
        mode, object_id = blobs[path]
        entries[path] = Entry(path, _git(repo_root, "cat-file", "blob", object_id), mode, "revision")

    for relative in spec.overlay:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Overlay file is missing from the working tree: {relative}")
        if relative in excluded:
            raise ValueError(f"Overlay file is also excluded from the source revision: {relative}")
        entries[relative] = Entry(relative, source.read_bytes(), _file_mode(source), "overlay")

    for pattern in spec.data_manifests:
        hits = sorted(path for path in repo_root.glob(pattern) if path.is_file())
        if not hits:
            raise FileNotFoundError(f"Data manifest pattern matches no working-tree file: {pattern}")
        for source in hits:
            relative = source.relative_to(repo_root).as_posix()
            if relative in entries:
                raise ValueError(f"Data manifest collides with another release entry: {relative}")
            entries[relative] = Entry(relative, source.read_bytes(), REGULAR_FILE, "data_manifest")

    for directory in spec.keep_dirs:
        if not (repo_root / directory).is_dir():
            raise FileNotFoundError(f"Keep directory does not exist in the working tree: {directory}")
        if any(path.startswith(f"{directory}/") for path in entries):
            raise ValueError(f"Keep directory already receives shipped files: {directory}")
        placeholder = f"{directory}/{spec.keep_file_name}"
        entries[placeholder] = Entry(placeholder, b"", REGULAR_FILE, "placeholder")
    return entries, sorted(excluded)


def validate_entries(entries: Mapping[str, Entry], spec: ReleaseSpec) -> None:
    """Rejects artifact types, cache paths, oversized files, and manifest collisions."""
    if spec.manifest_name in entries:
        raise ValueError(f"Release entry collides with the release manifest: {spec.manifest_name}")
    for entry in entries.values():
        path = PurePosixPath(entry.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Release entry escapes the archive root: {entry.path}")
        if spec.forbidden_path_parts.intersection(path.parts):
            raise ValueError(f"Release entry lies in a forbidden directory: {entry.path}")
        if path.suffix.lower() in spec.forbidden_suffixes:
            raise ValueError(f"Release entry has a forbidden artifact suffix: {entry.path}")
        if len(entry.data) > spec.max_file_bytes:
            raise ValueError(f"Release entry exceeds {spec.max_file_bytes} bytes: {entry.path} ({len(entry.data)})")


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(repo_root.resolve()):
        return resolved.relative_to(repo_root.resolve()).as_posix()
    return str(resolved)


def build_manifest(
    spec: ReleaseSpec,
    entries: Mapping[str, Entry],
    excluded: list[str],
    *,
    revision: str,
    timestamp: int,
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Describes the archive contents; it carries no build time so rebuilds are byte-identical."""
    return {
        "schema_version": "1",
        "name": spec.name,
        "source_revision": revision,
        "source_commit_time_utc": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
        "builder": {
            "script": _repo_relative(Path(__file__), repo_root),
            "script_sha256": _sha256(Path(__file__).read_bytes()),
            "config": _repo_relative(config_path, repo_root),
            "config_sha256": _sha256(config_path.read_bytes()),
            "repository_head": _git(repo_root, "rev-parse", "HEAD").decode().strip(),
        },
        "excluded_from_revision": excluded,
        "file_count": len(entries),
        "total_bytes": sum(len(entry.data) for entry in entries.values()),
        "files": [
            {"path": entry.path, "sha256": _sha256(entry.data), "bytes": len(entry.data), "origin": entry.origin}
            for entry in sorted(entries.values(), key=lambda item: item.path)
        ],
    }


def _require_outside_repository(output_dir: Path, repo_root: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_repo = repo_root.resolve()
    if resolved_output == resolved_repo or resolved_output.is_relative_to(resolved_repo):
        raise ValueError(f"Release archives must be written outside the repository: {output_dir}")


def write_archive(spec: ReleaseSpec, entries: Mapping[str, Entry], archive_path: Path, timestamp: int) -> None:
    """Writes sorted entries with fixed timestamps and Unix modes, then renames atomically."""
    date_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).timetuple()[:6]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    handle, staging_name = tempfile.mkstemp(prefix=f".{archive_path.name}.", suffix=".partial", dir=archive_path.parent)
    os.close(handle)
    staging = Path(staging_name)
    try:
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=spec.compression_level) as archive:
            for path in sorted(entries):
                entry = entries[path]
                info = zipfile.ZipInfo(f"{spec.root_dir}/{path}", date_time=date_time)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = entry.mode << 16
                archive.writestr(info, entry.data, compresslevel=spec.compression_level)
        os.replace(staging, archive_path)
    finally:
        if staging.exists():
            staging.unlink()


def verify_archive(archive_path: Path, entries: Mapping[str, Entry], root_dir: str) -> None:
    """Re-reads the archive and compares every member with the selected entry."""
    expected = {f"{root_dir}/{path}": _sha256(entry.data) for path, entry in entries.items()}
    with zipfile.ZipFile(archive_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive member failed its CRC check: {corrupt}")
        names = archive.namelist()
        if sorted(names) != sorted(expected) or len(names) != len(set(names)):
            raise ValueError("Archive members differ from the release selection")
        for name in names:
            if _sha256(archive.read(name)) != expected[name]:
                raise ValueError(f"Archive member content differs from the release selection: {name}")


def build_release(config_path: Path, repo_root: Path, *, dry_run: bool, replace: bool) -> ReleaseResult:
    """Selects, validates, writes, and verifies one release archive."""
    spec = load_spec(config_path, repo_root)
    _require_outside_repository(spec.output_dir, repo_root)
    revision, timestamp = resolve_revision(repo_root, spec.source_revision)
    entries, excluded = collect_entries(spec, repo_root, revision)
    validate_entries(entries, spec)
    manifest = build_manifest(
        spec, entries, excluded, revision=revision, timestamp=timestamp, config_path=config_path, repo_root=repo_root
    )
    if dry_run:
        return ReleaseResult(manifest=manifest, archive_path=None, archive_sha256=None)

    archive_path = spec.output_dir / f"{spec.file_stem}_{revision[:12]}.zip"
    if archive_path.exists() and not replace:
        raise FileExistsError(f"Release archive already exists; pass --replace to rebuild it: {archive_path}")
    manifest_bytes = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).encode("utf-8")
    archived = dict(entries)
    archived[spec.manifest_name] = Entry(spec.manifest_name, manifest_bytes, REGULAR_FILE, "manifest")
    write_archive(spec, archived, archive_path, timestamp)
    verify_archive(archive_path, archived, spec.root_dir)
    archive_sha256 = _sha256(archive_path.read_bytes())
    Path(f"{archive_path}.sha256").write_text(f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8")
    return ReleaseResult(manifest=manifest, archive_path=archive_path, archive_sha256=archive_sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a verified source-only release archive.")
    parser.add_argument("--config", type=Path, required=True, help="Release specification YAML")
    parser.add_argument("--dry-run", action="store_true", help="Report the selection without writing an archive")
    parser.add_argument("--replace", action="store_true", help="Overwrite an existing archive of the same revision")
    args = parser.parse_args()
    result = build_release(args.config, REPO_ROOT, dry_run=args.dry_run, replace=args.replace)
    origins: dict[str, int] = {}
    for record in result.manifest["files"]:
        origins[record["origin"]] = origins.get(record["origin"], 0) + 1
    print(f"source revision: {result.manifest['source_revision']}")
    print(f"entries: {result.manifest['file_count']} ({', '.join(f'{key}={value}' for key, value in sorted(origins.items()))})")
    print(f"uncompressed bytes: {result.manifest['total_bytes']}")
    print(f"excluded from revision: {', '.join(result.manifest['excluded_from_revision'])}")
    if result.archive_path is not None:
        print(f"archive: {result.archive_path} ({result.archive_path.stat().st_size} bytes)")
        print(f"sha256: {result.archive_sha256}")


if __name__ == "__main__":
    main()
