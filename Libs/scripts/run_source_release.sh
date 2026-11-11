#!/usr/bin/env bash
#SBATCH --job-name=DPlus_source_release
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --nice=10000
set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-/share/home/hwxu/miniconda3/envs/hw/bin/python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
scratch_dir="$(mktemp -d /tmp/dplus-source-release.XXXXXXXX)"
trap 'rm -rf -- "${scratch_dir}"' EXIT
export TMPDIR="${scratch_dir}"
export DPLUS_RELEASE_SCRATCH="${scratch_dir}"
export DPLUS_RELEASE_REPO="${repo_root}"
# Packaging fixtures and archive I/O remain on node-local scratch.
mkdir -p "${scratch_dir}/tests/Libs/scripts" "${scratch_dir}/tests/Libs/utils"
cp "${repo_root}/Libs/scripts/build_source_release.py" "${scratch_dir}/tests/Libs/scripts/"
cp "${repo_root}/Libs/utils/io.py" "${scratch_dir}/tests/Libs/utils/"
cp "${repo_root}/tests/test_source_release.py" "${scratch_dir}/tests/"
cd "${scratch_dir}/tests"
srun --ntasks=1 --cpus-per-task=1 "${python_bin}" -m pytest -q -p no:cacheprovider test_source_release.py
cd "${repo_root}"
srun --ntasks=1 --cpus-per-task=1 "${python_bin}" - <<'PYTHON'
import hashlib
import json
import os
import shutil
import socket
from pathlib import Path

import yaml
from Libs.scripts import build_source_release as release

repo = Path(os.environ["DPLUS_RELEASE_REPO"])
scratch = Path(os.environ["DPLUS_RELEASE_SCRATCH"])
config = repo / "Libs/configs/source_release.yaml"
spec = release.load_spec(config, repo)
release._require_outside_repository(spec.output_dir, repo)
revision, timestamp = release.resolve_revision(repo, spec.source_revision)
entries, excluded = release.collect_entries(spec, repo, revision)
release.validate_entries(entries, spec)
# These are metadata validity checks, not dataset or scientific qualification.
for entry in entries.values():
    if entry.origin == "data_manifest":
        json.loads(entry.data)
manifest = release.build_manifest(spec, entries, excluded, revision=revision,
    timestamp=timestamp, config_path=config, repo_root=repo)
archived = dict(entries)
archived[spec.manifest_name] = release.Entry(spec.manifest_name,
    yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).encode(),
    release.REGULAR_FILE, "manifest")
filename = f"{spec.file_stem}_{revision[:12]}.zip"
local_archive = scratch / filename
release.write_archive(spec, archived, local_archive, timestamp)
release.verify_archive(local_archive, archived, spec.root_dir)
digest = release._sha256(local_archive.read_bytes())
spec.output_dir.mkdir(parents=True, exist_ok=True)
destination = spec.output_dir / filename
# Refuse overwrites. Copy only verified deliverables to durable shared storage.
with destination.open("xb") as output, local_archive.open("rb") as source:
    shutil.copyfileobj(source, output)
if release._sha256(destination.read_bytes()) != digest:
    raise ValueError(f"Transferred archive checksum mismatch: {destination}")
with Path(f"{destination}.sha256").open("x", encoding="utf-8") as output:
    output.write(f"{digest}  {filename}\n")
print(json.dumps({"status": "COMPLETE", "node": socket.gethostname(),
    "job_id": os.environ.get("SLURM_JOB_ID"), "archive": str(destination),
    "archive_bytes": destination.stat().st_size, "sha256": digest,
    "source_revision": revision, "file_count": len(archived),
    "placeholder_count": sum(e.origin == "placeholder" for e in entries.values()),
    "input_manifest_count": sum(e.origin == "data_manifest" for e in entries.values()),
    "uncompressed_bytes": manifest["total_bytes"],
    "validation": "ZIP CRC, exact members, per-file SHA-256, transferred archive SHA-256"}, indent=2))
PYTHON
