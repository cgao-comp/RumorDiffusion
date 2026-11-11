#!/bin/bash
#SBATCH --job-name=DPlus_manuscript_transfer
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8-00:00:00
#SBATCH --array=0-23%1
#SBATCH --output=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.out
#SBATCH --error=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.err
set -euo pipefail

dry_run=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) dry_run=true ;;
    *) printf 'Unknown manuscript transfer runner argument: %s\n' "${argument}" >&2; exit 2 ;;
  esac
done

if [[ -n "${DPLUS_REPO_ROOT:-}" ]]; then
  repo_root="${DPLUS_REPO_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  repo_root="${PWD}"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
source_root="${DPLUS_MANUSCRIPT_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/manuscript-lp-observation}"
expected_code_commit="${DPLUS_MANUSCRIPT_CODE_COMMIT:-}"
rd_training_code_commit="${DPLUS_MANUSCRIPT_RD_TRAINING_CODE_COMMIT:-${DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT:-}}"
baseline_training_code_commit="${DPLUS_MANUSCRIPT_BASELINE_TRAINING_CODE_COMMIT:-${DPLUS_MANUSCRIPT_BASELINE_CODE_COMMIT:-}}"
python_bin="${PYTHON_BIN:-/share/home/hwxu/miniconda3/envs/hw/bin/python}"
qualification_source="${DPLUS_MANUSCRIPT_QUALIFICATION_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/training-observation}"
qualification_commit="${DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT:-}"
run_gate() { (cd "${qualification_source}" && "$@"); }
if [[ "${dry_run}" != true ]]; then
  if [[ ! "${qualification_commit}" =~ ^[0-9a-f]{40}$ || "$(git -C "${qualification_source}" rev-parse HEAD)" != "${qualification_commit}" || -n "$(git -C "${qualification_source}" status --porcelain --untracked-files=normal)" ]]; then
    printf 'Declare the exact clean qualification source in DPLUS_MANUSCRIPT_QUALIFICATION_CODE_COMMIT.\n' >&2
    exit 2
  fi
  test -f "${DPLUS_MANUSCRIPT_EXECUTION_PROVENANCE:?Declare the immutable seed42 execution provenance JSON}"
fi
runtime_exp="${repo_root}/Libs/configs/exp_manuscript_three_seed.yaml"
canonical_root="${repo_root}/Output/canonical/rumordiffusion_canonical_v1"

sources=(twitter15 weibo)
targets=(twitter16 pheme)
models=(agad hpcl_cascade tpso rumordiffusion)
seeds=(41 42 43)
case_index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${case_index}" =~ ^(0|[1-9][0-9]*)$ ]] || (( case_index >= 24 )); then
  printf 'Invalid manuscript transfer task index: %s\n' "${case_index}" >&2
  exit 2
fi
source_dataset="${sources[$((case_index / 12))]}"
target_dataset="${targets[$((case_index / 12))]}"
model="${models[$(((case_index % 12) / 3))]}"
seed="${seeds[$((case_index % 3))]}"
if [[ "${model}" == rumordiffusion ]]; then
  model_config="${source_root}/Libs/configs/model_canonical/rumordiffusion.yaml"
else
  model_config="${source_root}/Libs/configs/baselines/${model}.yaml"
fi

test -x "${python_bin}"
test -f "${runtime_exp}"
test -f "${model_config}"
observed_code_commit="$(git -C "${source_root}" rev-parse HEAD)"
source_changes="$(git -C "${source_root}" status --porcelain --untracked-files=normal)"
export PYTHONPATH="${source_root}"
export DPLUS_OUTPUT_ROOT="${repo_root}/Output"
export RUMORDIFFUSION_PROCESSED_ROOT="${repo_root}/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="${repo_root}/Input/sentence_transformers_all_mpnet_base_v2"
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

gate_command=("${python_bin}" -B -u -m Libs.exp.manuscript_transfer_evidence
  --canonical-root "${canonical_root}" --config "${runtime_exp}" --source-repo "${source_root}"
  --rd-training-code-commit "${rd_training_code_commit}"
  --baseline-training-code-commit "${baseline_training_code_commit}"
  --inference-code-commit "${expected_code_commit}")
checkpoint_command=("${gate_command[@]}" --source-checkpoint "${source_dataset}" "${model}" "${seed}")
target_gate_command=("${gate_command[@]}" --case "${source_dataset}" "${target_dataset}" "${model}" "${seed}")
execution_exp="${runtime_exp}"
if [[ "${seed}" == 42 ]]; then execution_exp="${repo_root}/Libs/configs/exp_manuscript_main.yaml"; fi
run_command=("${python_bin}" -B -u -m Libs.exp.run --exp "${execution_exp}" --model "${model_config}"
  --eval_only --transfer_metric_only --canonical_artifacts_only
  --dataset_override "${target_dataset}" --source_dataset_override "${source_dataset}" --seeds "${seed}")

if [[ "${dry_run}" == true ]]; then
  printf 'source=%s target=%s model=%s seed=%s task=%s\n' "${source_dataset}" "${target_dataset}" "${model}" "${seed}" "${case_index}"
  printf 'source_base=%s source_changes=%s\n' "${observed_code_commit}" "${source_changes:+uncommitted}"
  printf '%q ' "${checkpoint_command[@]}"
  printf '\n'
  printf '%q ' "${run_command[@]}" --ckpt '<qualified-source-checkpoint>'
  printf '\n'
  printf '%q ' "${target_gate_command[@]}"
  printf '\n'
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Use --dry-run for preparation. Execute transfer inside an authorized Slurm job.\n' >&2
  exit 2
fi
if [[ ! "${expected_code_commit}" =~ ^[0-9a-f]{40}$ || "${observed_code_commit}" != "${expected_code_commit}" || -n "${source_changes}" ]]; then
  printf 'Execution requires the approved clean inference source commit in DPLUS_MANUSCRIPT_CODE_COMMIT.\n' >&2
  exit 2
fi
if [[ ! "${rd_training_code_commit}" =~ ^[0-9a-f]{40}$ || ! "${baseline_training_code_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'Declare the exact RumorDiffusion and baseline training source commits.\n' >&2
  exit 2
fi

task_scratch=$(mktemp -d "/tmp/dplus-manuscript-transfer-${SLURM_JOB_ID}-${case_index}.XXXXXX")
cleanup() { rm -rf -- "${task_scratch}"; }
trap cleanup EXIT
export TMPDIR="${task_scratch}"
export PYTHONDONTWRITEBYTECODE=1
cd "${source_root}"
# This command qualifies the matching main source before resolving and checking
# the checkpoint in the established model checkpoint directory.
source_checkpoint="$(run_gate "${checkpoint_command[@]}")"
"${run_command[@]}" --ckpt "${source_checkpoint}"
run_gate "${target_gate_command[@]}"
