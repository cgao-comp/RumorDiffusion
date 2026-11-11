#!/bin/bash
#SBATCH --job-name=DPlus_d2_forecast_nas
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8-00:00:00
#SBATCH --array=0-11%1
#SBATCH --output=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.out
#SBATCH --error=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.err
set -euo pipefail
repo_root="${DPLUS_REPO_ROOT:-/share/home/hwxu/Researches/D2Plus}"
source_root="${DPLUS_D2_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/d2-forecast-nas}"
python_bin="${PYTHON_BIN:-/share/home/hwxu/miniconda3/envs/hw/bin/python}"
expected_commit="${DPLUS_D2_CODE_COMMIT:-}"
case_index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${case_index}" =~ ^(0|[1-9][0-9]*)$ ]] || (( case_index >= 12 )); then
  printf 'Invalid D2 task index: %s\n' "${case_index}" >&2
  exit 2
fi
datasets=(pheme weibo twitter15 twitter16)
seeds=(42 41 43)
dataset="${datasets[$((case_index / 3))]}"
seed="${seeds[$((case_index % 3))]}"
if [[ "${1:-}" == --dry-run && $# == 1 ]]; then
  printf 'dataset=%s seed=%s source=%s cpu04 8CPU 48G array concurrency1\n' "${dataset}" "${seed}" "${source_root}"
  printf 'Stages: train -> publish -> qualify; no job submitted.\n'
  exit 0
fi
if (( $# != 0 )) || [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Use --dry-run outside an authorized Slurm job.\n' >&2
  exit 2
fi
if [[ ! "${expected_commit}" =~ ^[0-9a-f]{40}$ || "$(git -C "${source_root}" rev-parse HEAD)" != "${expected_commit}" || -n "$(git -C "${source_root}" status --porcelain --untracked-files=normal)" ]]; then
  printf 'D2 requires the exact clean commit in DPLUS_D2_CODE_COMMIT.\n' >&2
  exit 2
fi
export PYTHONPATH="${source_root}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export DPLUS_OUTPUT_ROOT="${repo_root}/Output"
export RUMORDIFFUSION_PROCESSED_ROOT="${repo_root}/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="${repo_root}/Input/sentence_transformers_all_mpnet_base_v2"
scratch="$(mktemp -d /tmp/DPlus_d2_${SLURM_JOB_ID}_XXXXXX)"
trap 'rm -rf -- "${scratch}"' EXIT
export TMPDIR="${scratch}"
export DPLUS_D2_DATA_ROOT="${scratch}/processed"
mkdir -p "${DPLUS_D2_DATA_ROOT}/${dataset}/pyg"
for name in raw_manifest.json processed_manifest.json split_manifest.json label_schema.json splits.json text_embs_mpnet_canonical_v1.pt pyg/data.pt pyg/slices.pt; do
  cp -- "${RUMORDIFFUSION_PROCESSED_ROOT}/${dataset}/${name}" "${DPLUS_D2_DATA_ROOT}/${dataset}/${name}"
done
cd "${source_root}"
case_dir="${DPLUS_OUTPUT_ROOT}/exploratory/rumordiffusion_canonical_v1/d2_forecast_nas/${dataset}/seed_${seed}"
if [[ ! -f "${case_dir}/checkpoint.pt" ]]; then
  "${python_bin}" -B -u -m Libs.exp.d2_run train --dataset "${dataset}" --seed "${seed}" --threads 1
fi
# Publication validates the exact source, inputs, and diagnostic flag before reuse.
"${python_bin}" -B -u -m Libs.exp.d2_run publish --dataset "${dataset}" --seed "${seed}" --threads 1
"${python_bin}" -B -u -m Libs.exp.d2_evidence --output-root "${DPLUS_OUTPUT_ROOT}" --case "${dataset}" "${seed}" --report "${DPLUS_OUTPUT_ROOT}/exploratory/rumordiffusion_canonical_v1/d2_forecast_nas/${dataset}/seed_${seed}/qualification.json"
