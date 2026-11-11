#!/bin/bash
#SBATCH --job-name=DPlus_manuscript_main
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8-00:00:00
#SBATCH --array=0-239%1
#SBATCH --output=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.out
#SBATCH --error=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.err
set -euo pipefail

dry_run=false
early_evaluation=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) dry_run=true ;;
    --early-evaluation) early_evaluation=true ;;
    *) printf 'Unknown manuscript runner argument: %s\n' "${argument}" >&2; exit 2 ;;
  esac
done

if [[ -n "${DPLUS_REPO_ROOT:-}" ]]; then
  repo_root="${DPLUS_REPO_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  repo_root="${PWD}"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
default_source="${repo_root}/../D2Plus-worktrees/manuscript-experiment-source"
default_commit=80c76c3f59bd34e76d795af262d8d883bb83e0b5
if [[ "${early_evaluation}" == true ]]; then
  default_source="${repo_root}/../D2Plus-worktrees/manuscript-lp-observation"
  default_commit=f36fc8a4b415c38635a922ca8a336e0eda1c9558
fi
frozen_worktree="${DPLUS_MANUSCRIPT_SOURCE_ROOT:-${default_source}}"
expected_code_commit="${DPLUS_MANUSCRIPT_CODE_COMMIT:-${default_commit}}"
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

# PHEME RD42 is task 1 and is qualified read-only. Missing RD tasks:
# 0,2,60-62,120-122,180-182.
datasets=(pheme weibo twitter15 twitter16)
models=(rumordiffusion textonly bigcn tgat lp_gcn dga_fake agad hpcl_cascade tpso gcn gat gatv2 sage appnp gcn2 resgcn root_pool temporal_gcn ctdm_syn cgvae_gcn)
seeds=(41 42 43)
case_index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${case_index}" =~ ^(0|[1-9][0-9]*)$ ]] || (( case_index >= 240 )); then
  printf 'Invalid manuscript main task index: %s\n' "${case_index}" >&2
  exit 2
fi
dataset="${datasets[$((case_index / 60))]}"
model="${models[$(((case_index % 60) / 3))]}"
seed="${seeds[$((case_index % 3))]}"
# Keep unchanged methods on their original source while repaired baselines use
# their declared source commit. Each method retains one commit across seeds.
if [[ "${model}" != rumordiffusion ]]; then
  frozen_worktree="${DPLUS_MANUSCRIPT_BASELINE_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/manuscript-lp-observation}"
  expected_code_commit="${DPLUS_MANUSCRIPT_BASELINE_CODE_COMMIT:-f36fc8a4b415c38635a922ca8a336e0eda1c9558}"
fi
if [[ "${model}" == rumordiffusion ]]; then
  model_config="${frozen_worktree}/Libs/configs/model_canonical/rumordiffusion.yaml"
else
  model_config="${frozen_worktree}/Libs/configs/baselines/${model}.yaml"
fi

observed_code_commit="$(git -C "${frozen_worktree}" rev-parse HEAD)"
source_changes="$(git -C "${frozen_worktree}" status --porcelain --untracked-files=normal)"
test -x "${python_bin}"
test -f "${runtime_exp}"
test -f "${model_config}"

export PYTHONPATH="${frozen_worktree}"
export DPLUS_OUTPUT_ROOT="${repo_root}/Output"
export RUMORDIFFUSION_PROCESSED_ROOT="${repo_root}/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="${repo_root}/Input/sentence_transformers_all_mpnet_base_v2"
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

execution_exp="${runtime_exp}"
if [[ "${seed}" == 42 ]]; then execution_exp="${repo_root}/Libs/configs/exp_manuscript_main.yaml"; fi
run_command=("${python_bin}" -u -m Libs.exp.run --exp "${execution_exp}" --model "${model_config}" --dataset_override "${dataset}" --source_dataset_override "${dataset}" --seeds "${seed}")
qualification_command=()
post_qualification_command=()
if [[ "${early_evaluation}" == true ]]; then
  training_code_commit="${DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT:-}"
  baseline_training_code_commit="${DPLUS_MANUSCRIPT_BASELINE_TRAINING_CODE_COMMIT:-}"
  if [[ ! -f "${qualification_source}/Libs/exp/manuscript_early_evidence.py" ]]; then
    printf 'Early evaluation requires a source checkout with manuscript_early_evidence.py.\n' >&2
    exit 2
  fi
  model_name="${model}"
  if [[ "${model}" == temporal_gcn ]]; then model_name=temporalgcn; fi
  qualification_arguments=(
    --canonical-root "${repo_root}/Output/canonical/rumordiffusion_canonical_v1"
    --config "${runtime_exp}" --source-repo "${frozen_worktree}"
    --training-code-commit "${training_code_commit}")
  if [[ "${model}" != rumordiffusion ]]; then
    qualification_arguments+=(--model-training-code-commit "${model_name}=${baseline_training_code_commit}")
  fi
  qualification_arguments+=(--case "${dataset}" "${model_name}" "${seed}")
  qualification_command=("${python_bin}" -u -m Libs.exp.manuscript_main_evidence "${qualification_arguments[@]}")
  post_qualification_command=("${python_bin}" -u -m Libs.exp.manuscript_early_evidence
    "${qualification_arguments[@]}" --inference-code-commit "${expected_code_commit}")
  run_command+=(--eval_only --canonical_early_only --canonical_artifacts_only)
else
  if [[ ! -f "${qualification_source}/Libs/exp/manuscript_main_evidence.py" ]]; then
    printf 'Main evaluation requires a source checkout with manuscript_main_evidence.py.\n' >&2
    exit 2
  fi
  model_name="${model}"
  if [[ "${model}" == temporal_gcn ]]; then model_name=temporalgcn; fi
  run_command+=(--canonical_main_only)
  post_qualification_command=("${python_bin}" -u -m Libs.exp.manuscript_main_evidence
    --canonical-root "${repo_root}/Output/canonical/rumordiffusion_canonical_v1"
    --config "${runtime_exp}" --source-repo "${frozen_worktree}"
    --training-code-commit "${expected_code_commit}"
    --case "${dataset}" "${model_name}" "${seed}")
fi
if [[ "${dry_run}" == true ]]; then
  printf 'dataset=%s model=%s seed=%s task=%s\n' "${dataset}" "${model}" "${seed}" "${case_index}"
  printf 'source_base=%s source_changes=%s\n' "${observed_code_commit}" "${source_changes:+uncommitted}"
  if (( ${#qualification_command[@]} > 0 )); then
    printf '%q ' "${qualification_command[@]}"
    printf '\n'
  fi
  printf '%q ' "${run_command[@]}"
  printf '\n'
  if (( ${#post_qualification_command[@]} > 0 )); then
    printf '%q ' "${post_qualification_command[@]}"
    printf '\n'
  fi
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Use --dry-run for preparation. Execute experiments inside an authorized Slurm job.\n' >&2
  exit 2
fi
if [[ -z "${expected_code_commit}" || "${observed_code_commit}" != "${expected_code_commit}" || -n "${source_changes}" ]]; then
  printf 'Execution requires the approved clean source commit in DPLUS_MANUSCRIPT_CODE_COMMIT.\n' >&2
  exit 2
fi

if [[ "${early_evaluation}" == true ]]; then
  if [[ -z "${training_code_commit}" ]]; then
    printf 'Early evaluation requires DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT for RumorDiffusion training.\n' >&2
    exit 2
  fi
  if [[ "${model}" != rumordiffusion && -z "${baseline_training_code_commit}" ]]; then
    printf 'Early baseline evaluation requires DPLUS_MANUSCRIPT_BASELINE_TRAINING_CODE_COMMIT.\n' >&2
    exit 2
  fi
fi

task_scratch=$(mktemp -d "/tmp/dplus-manuscript-${SLURM_JOB_ID}-${case_index}.XXXXXX")
cleanup() { rm -rf -- "${task_scratch}"; }
trap cleanup EXIT
export TMPDIR="${task_scratch}"
export PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
cd "${frozen_worktree}"
if (( ${#qualification_command[@]} > 0 )); then
  run_gate "${qualification_command[@]}"
fi
if [[ "${dataset}" == pheme && "${model}" == rumordiffusion && "${seed}" == 42 ]]; then
  run_gate "${post_qualification_command[@]}"
  exit 0
fi
"${run_command[@]}"
if (( ${#post_qualification_command[@]} > 0 )); then
  run_gate "${post_qualification_command[@]}"
fi
