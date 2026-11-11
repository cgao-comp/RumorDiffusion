# RumorDiffusion

**Prefix-Conditioned Diffusion for Early Rumor Detection Under Sparse and Incomplete Social Cascades**

Research software and source-artifact documentation prepared for ACM Transactions
on Intelligent Systems and Technology (ACM TIST). The workspace directory is named
`D2Plus`; the canonical method is named **RumorDiffusion**. This artifact does not
assert publication acceptance, ACM artifact certification, or verified superiority.

## Overview

RumorDiffusion studies early rumor detection from incomplete social cascades.
The canonical model trains a conditional temporal diffusion model (CTDM) first,
then trains an arbiter while keeping CTDM parameters frozen. The system retains
per-example predictions and provenance so that reported metrics can be recomputed.

The repository also contains graph and text baselines, ablations, and exploratory
configurations. Prototype residuals, semantic residuals, posterior aggregation,
joint CTDM fine-tuning, and SOTA-search branches are disabled in the canonical
configuration. Their presence in the source does not make them canonical methods.
The independent `d2_forecast_nas` baseline is a project-defined implementation;
exact equivalence to the published D2 implementation is not claimed. Its assumptions
and information boundaries are documented in [CONTEXT.md](CONTEXT.md).

## Artifact scope and availability

The compact source package contains committed project code, YAML configurations,
tests, evidence schemas, selected manuscript source, project documentation, and
declared working-tree overlays. Small input manifests are included separately
from the data they describe. Runtime directories are retained with empty `.keep`
files. The archive manifest records the source revision and the origin, size, and
SHA-256 digest of each packaged file.

Raw and processed datasets, pretrained encoder weights, trained checkpoints,
per-example result bundles, logs, caches, Git history, backup files, and compiled
manuscript PDFs are not included. Omitted documentation and asset directories
retain placeholders. The ZIP is a source snapshot, not a complete experimental
record or a standalone manuscript build. Included manuscript tables and prose are
not independently qualified by packaging and must not be used as performance evidence.

## Project structure

```text
RumorDiffusion/
├── README.md                         # Artifact documentation
├── AGENTS.md                         # Evidence and execution constraints
├── CONTEXT.md                        # Scientific assumptions and baseline scope
├── TODO.md                           # Project tracker and historical status
├── PROJECT_PREFIX                    # Cluster job-name prefix
├── pyproject.toml                    # Build/tool configuration
├── Libs/
│   ├── configs/                      # Dataset, model, experiment, release YAML
│   ├── data/                         # Cascade processing and dataset views
│   ├── models/                       # RumorDiffusion, modules, and baselines
│   ├── exp/                          # Training, evaluation, evidence, aggregation
│   ├── scripts/                      # Entry points and cluster launchers
│   └── utils/                        # Metrics, calibration, provenance, validation
├── schemas/                          # Evidence schemas from the source revision
├── tests/                            # Tests from the source revision and overlays
├── Docs/                             # Selected manuscript source and placeholders
├── Input/                            # Small manifests and runtime placeholders
├── Output/                           # Runtime placeholders; no reported results
└── release_manifest.yaml             # Generated during archive creation
```

The archive is assembled from a pinned commit, not a recursive copy of the live
workspace. Some schemas and tests exist in that commit but are absent from the
current working directory. Consult `release_manifest.yaml` for the exact contents.

## Experimental protocol

| Item | Active specification |
| --- | --- |
| Datasets | Twitter15, Twitter16, Weibo, PHEME |
| Experiment seeds | Exactly `41`, `42`, `43` |
| Early Twitter horizons | `10`, `20`, `30` minutes |
| Early Weibo/PHEME horizons | `1`, `2`, `3` hops |
| Registered main observation | Twitter: `60` minutes; Weibo/PHEME: `3` hops |
| Split unit | Cascade; immutable split membership |
| Model and architecture selection | Validation data only |
| Calibration and threshold selection | Validation data only |
| Test labels | Reporting only |
| Canonical training | CTDM pretraining, then a frozen-CTDM arbiter |

The active experiment-seed override is
`Libs/configs/exp_manuscript_three_seed.yaml`. Some inherited and historical YAML
files retain earlier seeds. They must not be run unchanged as the active protocol.
Fixed split and bootstrap settings are distinct from experiment seeds; preserve
registered split identities and original evidence provenance.

Observation units must remain dataset-specific. Hop completion is a structural
observation protocol and does not establish chronological forecasting. Preserve
source identities, label mappings, text normalization, and registered embeddings.
Never use test outcomes to select models, thresholds, or which results to retain.

## Software environment

Use Python 3.10 or newer. Install mutually compatible PyTorch and PyTorch Geometric
versions for the intended CPU or CUDA runtime before installing the remaining
packages. The source uses NumPy, pandas, scikit-learn, PyYAML, tqdm, Transformers,
and sentence-transformers. Weights & Biases is optional when logging is disabled.
Pytest is required for the test suite.

```bash
python -m venv .venv
source .venv/bin/activate
# Install PyTorch and PyTorch Geometric for your selected runtime first.
python -m pip install numpy pandas scikit-learn pyyaml tqdm transformers sentence-transformers pytest
```

Run module commands from the project root. `pyproject.toml` does not declare a
complete dependency set or a pinned environment lock. Record Python, package,
CUDA, hardware, and configuration versions for each experimental run. No fixed
runtime or memory requirement is certified by this source package.

## Data preparation

1. Obtain each dataset through its authorized original distribution. Review the
   dataset terms and restrictions on redistribution of social-media content.
2. Provision the fixed text encoder. The current canonical configuration uses a
   local MPNet encoder and the cache name `text_embs_mpnet_canonical_v1.pt`.
3. Set the runtime paths and review the raw-data location in the dataset YAML.
4. Validate existing manifests before processing. Reuse frozen splits and caches;
   do not regenerate authoritative inputs to resolve a mismatch.

```bash
export RUMORDIFFUSION_PROCESSED_ROOT="$PWD/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="$PWD/Input/sentence_transformers_all_mpnet_base_v2"
export DPLUS_OUTPUT_ROOT="$PWD/Output"
```

`Libs/configs/dataset_canonical.yaml` contains a workspace-specific `raw_root`.
`Libs/configs/exp_manuscript_three_seed.yaml` contains an absolute `base_config`.
For a new installation, resolve these paths in a separate local YAML using the
existing configuration loader. Preserve the original configurations and provenance.
A path-adjusted run is a new run; it does not replace historical evidence.

Inspect the preparation interface before execution:

```bash
python -m Libs.data.processor --help
python -m Libs.scripts.prepare_canonical_inputs --help
```

After provisioning inputs and resolving local paths, the preparation command is:

```bash
python -m Libs.scripts.prepare_canonical_inputs \
  --exp Libs/configs/exp_canonical.yaml \
  --dataset-config Libs/configs/dataset_canonical.yaml
```

This command creates processed input and cache artifacts. Run it on an authorized
compute node. The packaged input manifests describe omitted files; their presence
alone does not establish input availability or successful reproduction.

## Training and evaluation

Inspect the runner interface:

```bash
python -m Libs.exp.run --help
```

The following command illustrates the real entry point for one dataset and all
three experiment seeds. It requires provisioned inputs, resolved configuration
paths, a suitable execution environment, and authorization for compute use:

```bash
python -m Libs.exp.run \
  --exp Libs/configs/exp_manuscript_three_seed.yaml \
  --model Libs/configs/model_canonical/rumordiffusion.yaml \
  --dataset_override twitter15 \
  --seeds 41,42,43
```

A single dataset command does not establish complete four-dataset evidence.
The production manuscript launchers also require pinned source and qualification
checkouts, exact code commits, and immutable execution provenance. These checkouts
and Git history are not included in the ZIP. Restore them from the authorized
source repository before using those launchers. Do not bypass their checks.

On the maintained cluster, `mgmt2` is a control-plane node. Do not run training,
preprocessing, archive creation, or large tests there. Slurm submissions and other
job changes require explicit authorization. Inspect launchers before use; they
contain workspace paths and production resource settings.

## Evidence validation and interpretation

The canonical evidence chain is:

```text
raw manifest → processed manifest → split manifest → resolved configuration
→ code commit → checkpoint → per-example predictions → metrics
→ aggregates → table cell registry
```

Missing, duplicate, stale, non-finite, placeholder, or hash-mismatched artifacts
invalidate the affected result. Exploratory, diagnostic, search, and failed
outputs remain outside `Output/canonical/`. Legacy CSVs and historical tracker
entries are not substitutes for qualified prediction bundles.

Validate a provisioned run directory or canonical root with:

```bash
python -m Libs.scripts.validate_canonical_results \
  Output/canonical/rumordiffusion_canonical_v1
```

Successful input preparation, training completion, and evidence qualification are
separate outcomes. Report performance only after the relevant evidence checks
pass. Report per-seed results, the actual evaluation scope, uncertainty, and
failure cases. Do not infer complete experimental support from a diagnostic run
or from a packaged table. This README provides no verified numerical results.

## Tests

On a suitable compute node with dependencies installed, run:

```bash
python -m pytest tests -q
```

The release omits `tests/test_original_figure_immutability.py` and its protected
figure checksum registry because the protected figure assets are not shipped.
This exclusion does not authorize changes to protected manuscript assets.
Packaging tests check archive selection, integrity, placeholders, reproducibility,
and error handling. A passing software test suite does not establish scientific
performance. No full-suite pass is asserted by this README.

## Building the compact ZIP

The existing builder uses `Libs/configs/source_release.yaml`. It enforces a
2 MiB per-file limit, excludes declared artifacts, preserves executable modes,
and creates `.keep` entries without changing the source data directories. It
compares every archived file with the selected content and performs ZIP CRC checks.

Run the following commands on an authorized compute node with Git, Python, and
PyYAML installed. The local Git repository must contain the pinned source revision.
The source-release builder cannot reconstruct that revision from the ZIP alone.

```bash
python -m Libs.scripts.build_source_release \
  --config Libs/configs/source_release.yaml --dry-run
python -m Libs.scripts.build_source_release \
  --config Libs/configs/source_release.yaml
```

The default destination is `../D2Plus_release/`. The output name is
`RumorDiffusion_source_<revision-prefix>.zip`; its SHA-256 sidecar supports transfer
verification. The builder refuses an existing archive unless replacement is
explicitly requested. The internal manifest distinguishes committed files,
working-tree overlays, input manifests, and placeholders. Preserve that distinction
when describing the source version used for an experiment.

## Citation, licensing, and contact

Use the verified bibliographic record for the associated manuscript once available.
No DOI, publication year, or accepted-paper status is supplied in this artifact.
For software citation, identify RumorDiffusion and record the source revision and
archive digest from the release manifest and checksum sidecar.

No project-wide license file is present in this workspace. Public availability
does not itself grant redistribution or reuse rights. Obtain the appropriate
permission before redistribution. Dataset, pretrained-model, dependency, and
bundled ACM-template terms apply independently.

For scientific questions or access requests, consult the corresponding-author
information in `Docs/paper/Sections/Frontmatter.tex`. This package retains manuscript
author information and internal project documentation; it is not anonymized for
review. Any public release or submission requires separate authorization.
