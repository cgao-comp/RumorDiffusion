#!/bin/bash
#SBATCH --job-name=DPlus_pheme_readiness
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-0%1
#SBATCH --output=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.out
#SBATCH --error=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.err
set -euo pipefail

dry_run=false
if [[ "${1:-}" == --dry-run && $# == 1 ]]; then
  dry_run=true
elif (( $# != 0 )); then
  printf 'Use no arguments or --dry-run.\n' >&2; exit 2
fi
if [[ -n "${DPLUS_REPO_ROOT:-}" ]]; then
  repo_root="${DPLUS_REPO_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  repo_root="${PWD}"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source_root="${DPLUS_READINESS_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/manuscript-artifact-recovery}"
expected_commit="${DPLUS_READINESS_CODE_COMMIT:-}"
python_bin="${PYTHON_BIN:-/share/home/hwxu/miniconda3/envs/hw/bin/python}"
if [[ "${SLURM_ARRAY_TASK_ID:-0}" != 0 ]]; then
  printf 'PHEME readiness has exactly one task.\n' >&2; exit 2
fi
export DPLUS_OUTPUT_ROOT="${repo_root}/Output"
export RUMORDIFFUSION_PROCESSED_ROOT="${repo_root}/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="${repo_root}/Input/sentence_transformers_all_mpnet_base_v2"
export DPLUS_MANUSCRIPT_EXECUTION_PROVENANCE="${repo_root}/Output/exploratory/rumordiffusion_canonical_v1/control/manuscript_seed42_execution.json"
export PYTHONPATH="${source_root}" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${OMP_NUM_THREADS}" NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
if [[ "${dry_run}" != true ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf 'Full readiness qualification requires an authorized Slurm job.\n' >&2; exit 2
  fi
  if [[ ! "${expected_commit}" =~ ^[0-9a-f]{40}$ || "$(git -C "${source_root}" rev-parse HEAD)" != "${expected_commit}" || -n "$(git -C "${source_root}" status --porcelain --untracked-files=normal)" ]]; then
    printf 'Declare a clean exact DPLUS_READINESS_CODE_COMMIT.\n' >&2; exit 2
  fi
  task_scratch=$(mktemp -d "/tmp/dplus-pheme-readiness-${SLURM_JOB_ID}.XXXXXX")
  cleanup() { rm -rf -- "${task_scratch}"; }
  trap cleanup EXIT
  export TMPDIR="${task_scratch}" XDG_CACHE_HOME="${task_scratch}/cache"
fi
cd "${source_root}"
model_lines=$("${python_bin}" -B -c 'from Libs.exp.manuscript_main_evidence import MODEL_NAMES; print("\n".join(MODEL_NAMES))')
mapfile -t models <<< "${model_lines}"
if (( ${#models[@]} != 20 )); then
  printf 'The declared PHEME main scope must contain all 20 methods.\n' >&2; exit 2
fi
for model in "${models[@]}"; do
  command=("${python_bin}" -B -u -m Libs.exp.manuscript_main_evidence
    --canonical-root "${DPLUS_OUTPUT_ROOT}/canonical/rumordiffusion_canonical_v1"
    --config "${repo_root}/Libs/configs/exp_manuscript_three_seed.yaml" --source-repo "${source_root}"
    --training-code-commit 80c76c3f59bd34e76d795af262d8d883bb83e0b5 --case pheme "${model}" 42)
  if [[ "${model}" != rumordiffusion ]]; then
    command+=(--model-training-code-commit "${model}=f36fc8a4b415c38635a922ca8a336e0eda1c9558")
  fi
  if [[ "${dry_run}" == true ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
  fi
done
if [[ "${dry_run}" != true ]]; then
  printf 'Qualified all 20 declared PHEME seed42 main cases.\n'
fi
