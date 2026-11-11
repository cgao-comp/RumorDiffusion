#!/bin/bash
#SBATCH --job-name=DPlus_validation_diagnostic
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8-00:00:00
#SBATCH --array=0-0%1
#SBATCH --output=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.out
#SBATCH --error=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.err
set -euo pipefail

dry_run=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) dry_run=true ;;
    *) printf 'Unknown validation diagnostic argument: %s\n' "${argument}" >&2; exit 2 ;;
  esac
done
if [[ -n "${DPLUS_REPO_ROOT:-}" ]]; then
  repo_root="${DPLUS_REPO_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  repo_root="${PWD}"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source_root="${DPLUS_VALIDATION_DIAGNOSTIC_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/validation-diagnostics}"
code_commit="${DPLUS_VALIDATION_DIAGNOSTIC_CODE_COMMIT:-}"
python_bin="${PYTHON_BIN:-/share/home/hwxu/miniconda3/envs/hw/bin/python}"
case_index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${case_index}" != 0 ]]; then
  printf 'Validation diagnostic has only array task0.\n' >&2
  exit 2
fi
source_run="${repo_root}/Output/canonical/rumordiffusion_canonical_v1/runs/run-5a8808d520d941fa111e"
config="${source_run}/resolved_exp.yaml"
model="${source_run}/resolved_model.yaml"
checkpoint="${source_run}/checkpoints/best.pt"
output="${repo_root}/Output/exploratory/rumordiffusion_canonical_v1/validation_diagnostics/pheme_seed42_hop3.json"
test -x "${python_bin}"
test -f "${config}"
test -f "${model}"
test -f "${checkpoint}"
test -f "${source_root}/Libs/exp/validation_diagnostics.py"
observed_commit="$(git -C "${source_root}" rev-parse HEAD)"
source_changes="$(git -C "${source_root}" status --porcelain --untracked-files=normal)"
export PYTHONPATH="${source_root}"
export DPLUS_OUTPUT_ROOT="${repo_root}/Output"
export RUMORDIFFUSION_PROCESSED_ROOT="${repo_root}/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="${repo_root}/Input/sentence_transformers_all_mpnet_base_v2"
export DPLUS_VALIDATION_DIAGNOSTIC_CODE_COMMIT="${code_commit}"
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
preflight=("${python_bin}" -B -u -m Libs.exp.validation_diagnostics --config "${config}" --model "${model}" --check-inputs)
run_command=("${python_bin}" -B -u -m Libs.exp.run --exp "${config}" --model "${model}"
  --validation_diagnostic_only --eval_only --ckpt "${checkpoint}" --diagnostic-output "${output}"
  --dataset_override pheme --source_dataset_override pheme --seeds 42)
qualification=("${python_bin}" -B -u -m Libs.exp.validation_diagnostics --config "${config}" --model "${model}"
  --check-only --output "${output}" --code-commit "${code_commit}")
if [[ "${dry_run}" == true ]]; then
  printf 'dataset=pheme checkpoint_seed=42 diagnostic_seeds=41,42,43 split=val hop=3\n'
  printf 'source_base=%s source_changes=%s\n' "${observed_commit}" "${source_changes:+uncommitted}"
  printf '%q ' "${preflight[@]}"; printf '\n'
  printf '%q ' "${run_command[@]}"; printf '\n'
  printf '%q ' "${qualification[@]}"; printf '\n'
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Use --dry-run for preparation. Execute validation inference inside an authorized Slurm job.\n' >&2
  exit 2
fi
if [[ ! "${code_commit}" =~ ^[0-9a-f]{40}$ || "${observed_commit}" != "${code_commit}" || -n "${source_changes}" ]]; then
  printf 'Validation diagnostic requires its declared clean source commit.\n' >&2
  exit 2
fi
task_scratch=$(mktemp -d "/tmp/dplus-validation-${SLURM_JOB_ID}.XXXXXX")
cleanup() { rm -rf -- "${task_scratch}"; }
trap cleanup EXIT
export TMPDIR="${task_scratch}" XDG_CACHE_HOME="${task_scratch}/cache"
cd "${source_root}"
"${preflight[@]}"
"${run_command[@]}"
"${qualification[@]}"
