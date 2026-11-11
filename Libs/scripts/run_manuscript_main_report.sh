#!/bin/bash
#SBATCH --job-name=DPlus_main_report
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
source_root="${DPLUS_MAIN_COMPARISON_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/lp-bundle-recovery}"
expected_commit="${DPLUS_MAIN_COMPARISON_CODE_COMMIT:-}"
python_bin="${PYTHON_BIN:-/share/home/hwxu/miniconda3/envs/hw/bin/python}"
training_commit=80c76c3f59bd34e76d795af262d8d883bb83e0b5
baseline_commit=f36fc8a4b415c38635a922ca8a336e0eda1c9558
export DPLUS_OUTPUT_ROOT="${repo_root}/Output"
export RUMORDIFFUSION_PROCESSED_ROOT="${repo_root}/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="${repo_root}/Input/sentence_transformers_all_mpnet_base_v2"
export DPLUS_MANUSCRIPT_EXECUTION_PROVENANCE="${repo_root}/Output/exploratory/rumordiffusion_canonical_v1/control/manuscript_seed42_execution.json"
export PYTHONPATH="${source_root}" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}" MKL_NUM_THREADS="${OMP_NUM_THREADS}" NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
canonical_root="${DPLUS_OUTPUT_ROOT}/canonical/rumordiffusion_canonical_v1"
aggregate_dir="${canonical_root}/aggregates"
fragment_dir="${DPLUS_OUTPUT_ROOT}/exploratory/rumordiffusion_canonical_v1/reports/manuscript_main_tables"
arguments=(--canonical-root "${canonical_root}" --config "${repo_root}/Libs/configs/exp_manuscript_three_seed.yaml"
  --source-repo "${source_root}" --training-code-commit "${training_commit}")
baselines=(textonly bigcn tgat lp_gcn dga_fake agad hpcl_cascade tpso gcn gat gatv2 sage appnp gcn2 resgcn root_pool temporalgcn ctdm_syn cgvae_gcn)
for model in "${baselines[@]}"; do arguments+=(--model-training-code-commit "${model}=${baseline_commit}"); done
recovery_command=("${python_bin}" -B -u -m Libs.scripts.recover_lp_main_publication
  --canonical-root "${canonical_root}"
  --failed-root "${DPLUS_OUTPUT_ROOT}/failed/rumordiffusion_canonical_v1"
  --config "${repo_root}/Libs/configs/exp_manuscript_three_seed.yaml"
  --source-repo "${source_root}" --training-code-commit "${training_commit}"
  --baseline-training-code-commit "${baseline_commit}" --all-cases --recover)
main_command=("${python_bin}" -B -u -m Libs.exp.manuscript_main_comparison "${arguments[@]}" --bind-main)
paired_command=("${python_bin}" -B -u -m Libs.exp.manuscript_main_comparison "${arguments[@]}")
if [[ "${dry_run}" == true ]]; then
  printf 'scope=240_main_runs,76_pairs,152_metric_comparisons seeds=41,42,43\n'
  printf '%q ' "${recovery_command[@]}"; printf '\n'
  printf '%q ' "${main_command[@]}"; printf '\n'
  printf '%q ' "${paired_command[@]}"; printf '\n'
  printf '%q ' "${paired_command[@]}" --render-dir "${fragment_dir}"; printf '\n'
  exit 0
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Full main aggregation requires an authorized Slurm job.\n' >&2; exit 2
fi
if [[ "${SLURM_ARRAY_TASK_ID:-0}" != 0 ]]; then
  printf 'The main report has only array task 0.\n' >&2; exit 2
fi
if [[ ! "${expected_commit}" =~ ^[0-9a-f]{40}$ || "$(git -C "${source_root}" rev-parse HEAD)" != "${expected_commit}" || -n "$(git -C "${source_root}" status --porcelain --untracked-files=normal)" ]]; then
  printf 'Declare a clean exact DPLUS_MAIN_COMPARISON_CODE_COMMIT.\n' >&2; exit 2
fi
test -x "${python_bin}"
test -f "${source_root}/Libs/scripts/recover_lp_main_publication.py"
test -f "${DPLUS_MANUSCRIPT_EXECUTION_PROVENANCE}"
task_scratch=$(mktemp -d "/tmp/dplus-main-report-${SLURM_JOB_ID}.XXXXXX")
cleanup() { rm -r -- "${task_scratch}"; }
trap cleanup EXIT
export TMPDIR="${task_scratch}" XDG_CACHE_HOME="${task_scratch}/cache"
cd "${source_root}"
# This report runs after the producer arrays have terminated.
"${recovery_command[@]}"
# Reuse is verified by the next source-replaying stage. Partial pairs fail closed.
if [[ ! -e "${aggregate_dir}/manuscript_main_results.json" && ! -e "${aggregate_dir}/manuscript_main_registry.json" ]]; then
  "${main_command[@]}"
fi
if [[ ! -e "${aggregate_dir}/manuscript_paired_results.json" && ! -e "${aggregate_dir}/manuscript_paired_registry.json" ]]; then
  "${paired_command[@]}"
else
  "${paired_command[@]}" --check-only
fi
if [[ ! -e "${fragment_dir}/manuscript_main.tex" && ! -e "${fragment_dir}/manuscript_paired.tex" ]]; then
  "${paired_command[@]}" --render-dir "${fragment_dir}"
else
  "${paired_command[@]}" --render-dir "${fragment_dir}" --validate-fragments
fi
