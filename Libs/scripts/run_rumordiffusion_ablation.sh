#!/bin/bash
#SBATCH --job-name=DPlus_manuscript_ablation
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8-00:00:00
#SBATCH --array=0-35%1
#SBATCH --output=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.out
#SBATCH --error=Output/exploratory/rumordiffusion_canonical_v1/logs/parent_jobs/%x_%A_%a.err
set -euo pipefail

dry_run=false
for argument in "$@"; do
  case "${argument}" in
    --dry-run) dry_run=true ;;
    *) printf 'Unknown manuscript ablation argument: %s\n' "${argument}" >&2; exit 2 ;;
  esac
done
if [[ "${dry_run}" != true && -z "${SLURM_JOB_ID:-}" ]]; then
  printf 'Use --dry-run for preparation. Execute experiments inside an authorized Slurm job.\n' >&2
  exit 2
fi

if [[ -n "${DPLUS_REPO_ROOT:-}" ]]; then
  repo_root="${DPLUS_REPO_ROOT}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  repo_root="${PWD}"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
frozen_worktree="${DPLUS_MANUSCRIPT_SOURCE_ROOT:-${repo_root}/../D2Plus-worktrees/manuscript-lp-observation}"
expected_code_commit="${DPLUS_MANUSCRIPT_CODE_COMMIT:-}"
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
datasets=(twitter15 twitter16)
variants=(full without_reg without_pair without_temporal without_text without_temperature)
model_files=(rumordiffusion_full.yaml rumordiffusion_without_regularization.yaml rumordiffusion_without_pair.yaml rumordiffusion_without_temporal.yaml rumordiffusion_without_text.yaml rumordiffusion_without_temperature.yaml)
seeds=(41 42 43)
case_index="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ! "${case_index}" =~ ^(0|[1-9][0-9]*)$ ]] || (( case_index >= 36 )); then
  printf 'Invalid manuscript ablation task index: %s\n' "${case_index}" >&2
  exit 2
fi
dataset="${datasets[$((case_index / 18))]}"
variant_index=$(((case_index % 18) / 3))
variant="${variants[${variant_index}]}"
seed="${seeds[$((case_index % 3))]}"
model_config="${frozen_worktree}/Libs/configs/model_canonical/${model_files[${variant_index}]}"
evaluation_only=false
if [[ "${variant}" == full || "${variant}" == without_temperature ]]; then
  evaluation_only=true
  checkpoint_model_config="${frozen_worktree}/Libs/configs/model_canonical/rumordiffusion.yaml"
else
  checkpoint_model_config="${model_config}"
fi
requires_full_checkpoint="${evaluation_only}"
if [[ "${variant}" == without_pair ]]; then requires_full_checkpoint=true; fi
if [[ "${requires_full_checkpoint}" == true ]]; then
  training_code_commit="${DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT:-}"
  if [[ ! "${training_code_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    printf 'Full-checkpoint reuse requires DPLUS_MANUSCRIPT_TRAINING_CODE_COMMIT.\n' >&2
    exit 2
  fi
fi

observed_code_commit="$(git -C "${frozen_worktree}" rev-parse HEAD)"
source_changes="$(git -C "${frozen_worktree}" status --porcelain --untracked-files=normal)"
test -x "${python_bin}"
test -f "${runtime_exp}"
test -f "${model_config}"
test -f "${checkpoint_model_config}"
if [[ "${dry_run}" != true && (
  ! "${expected_code_commit}" =~ ^[0-9a-f]{40}$ ||
  "${observed_code_commit}" != "${expected_code_commit}" || -n "${source_changes}"
) ]]; then
  printf 'Execution requires the approved clean source commit in DPLUS_MANUSCRIPT_CODE_COMMIT.\n' >&2
  exit 2
fi

export PYTHONPATH="${frozen_worktree}"
export DPLUS_OUTPUT_ROOT="${repo_root}/Output"
export RUMORDIFFUSION_PROCESSED_ROOT="${repo_root}/Input/processed_canonical_v1"
export RUMORDIFFUSION_TEXT_ENCODER="${repo_root}/Input/sentence_transformers_all_mpnet_base_v2"
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
if [[ "${dry_run}" != true ]]; then
  task_scratch=$(mktemp -d "/tmp/dplus-manuscript-ablation-${SLURM_JOB_ID}-${case_index}.XXXXXX")
  cleanup() { rm -rf -- "${task_scratch}"; }
  trap cleanup EXIT
  export TMPDIR="${task_scratch}" PYTHONPYCACHEPREFIX="${task_scratch}/pycache"
  export XDG_CACHE_HOME="${task_scratch}/cache" HF_HOME="${task_scratch}/huggingface" TORCH_HOME="${task_scratch}/torch"
fi

checkpoint_path="$(cd "${frozen_worktree}" && "${python_bin}" - "${runtime_exp}" "${checkpoint_model_config}" "${dataset}" "${seed}" <<'PY'
import os
import sys
from Libs.exp.run import read_yaml_like, resolve_model_checkpoint_subdir
experiment = read_yaml_like(sys.argv[1])
model = read_yaml_like(sys.argv[2])
name = str(model["name"]).lower()
variant = str(model.get("variant", name))
subdirectory = resolve_model_checkpoint_subdir(name, variant)
checkpoint_name = str(experiment.get("transfer", {}).get("ckpt_name", "best.pt"))
print(os.path.join(str(experiment["outputs"]["ckpt_dir"]), sys.argv[3], subdirectory, f"seed{sys.argv[4]}", checkpoint_name))
PY
)"
execution_exp="${runtime_exp}"
if [[ "${seed}" == 42 ]]; then execution_exp="${repo_root}/Libs/configs/exp_manuscript_main.yaml"; fi
run_command=("${python_bin}" -u -m Libs.exp.run --exp "${execution_exp}" --model "${model_config}" --dataset_override "${dataset}" --source_dataset_override "${dataset}" --seeds "${seed}" --canonical_artifacts_only)
main_command=()
qualification_command=()
if [[ "${requires_full_checkpoint}" == true ]]; then
  qualification_command=("${python_bin}" -u -m Libs.exp.manuscript_main_evidence
    --canonical-root "${DPLUS_OUTPUT_ROOT}/canonical/rumordiffusion_canonical_v1"
    --config "${runtime_exp}" --source-repo "${frozen_worktree}"
    --training-code-commit "${training_code_commit}" --case "${dataset}" rumordiffusion "${seed}")
fi
if [[ "${evaluation_only}" != true ]]; then
  # The corrected regularization ablation changes the active CTDM auxiliary
  # objective. Its YAML requires CTDM and arbiter training.
  main_command=("${run_command[@]}" --canonical_main_only)
fi
early_command=("${run_command[@]}" --eval_only --canonical_early_only --ckpt "${checkpoint_path}")
if [[ "${dry_run}" == true ]]; then
  printf 'dataset=%s variant=%s seed=%s task=%s total=36\n' "${dataset}" "${variant}" "${seed}" "${case_index}"
  printf 'source_base=%s source_changes=%s\n' "${observed_code_commit}" "${source_changes:+uncommitted}"
  if (( ${#qualification_command[@]} > 0 )); then printf '%q ' "${qualification_command[@]}"; printf '\n'; fi
  if (( ${#main_command[@]} > 0 )); then printf '%q ' "${main_command[@]}"; printf '\n'; fi
  printf '%q ' "${early_command[@]}"
  printf '\n'
  exit 0
fi

cd "${frozen_worktree}"
if (( ${#qualification_command[@]} > 0 )); then run_gate "${qualification_command[@]}"; fi
if (( ${#main_command[@]} > 0 )); then "${main_command[@]}"; fi
"${early_command[@]}"
