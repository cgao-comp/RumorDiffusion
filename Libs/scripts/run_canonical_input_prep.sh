#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp_canonical.yaml}"
DATASET_CFG_DEFAULT="${DATASET_CFG:-Libs/configs/dataset_canonical.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  JOB_PREFIX=$(resolve_project_prefix)
  LOG_DIR="${LOG_DIR:-Output/exploratory/rumordiffusion_canonical_v1/logs}"
  CONDA_ENV="${CONDA_ENV:-hw}"
  mkdir -p "${LOG_DIR}"
  JOB_NAME="${JOB_PREFIX}_canonical_prep"
  sbatch \
    --job-name="${JOB_NAME}" \
    --partition=gpu \
    --nodelist=cpu03 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK:-16}" \
    --mem="${MEM:-64G}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",DATASET_CFG="${DATASET_CFG_DEFAULT}",CONDA_ENV_OVERRIDE="${CONDA_ENV}" \
    Libs/scripts/run_canonical_input_prep.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-hw}"
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
LOG_DIR="${LOG_DIR:-Output/exploratory/rumordiffusion_canonical_v1/logs}"
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_canonical_prep}"

export RUMORDIFFUSION_PROCESSED_ROOT="${RUMORDIFFUSION_PROCESSED_ROOT:-/share/home/hwxu/Researches/D2Plus/Input/processed_canonical_v1}"
export RUMORDIFFUSION_TEXT_ENCODER="${RUMORDIFFUSION_TEXT_ENCODER:-/share/home/hwxu/Researches/D2Plus/Input/sentence_transformers_all_mpnet_base_v2}"

RUN_CMD=(
  python3
  Libs/scripts/prepare_canonical_inputs.py
  --exp "${EXP_CFG:-${EXP_CFG_DEFAULT}}"
  --dataset-config "${DATASET_CFG:-${DATASET_CFG_DEFAULT}}"
)
log_run_header "${RUN_CMD[*]}"
srun -n 1 -c "${SLURM_CPUS_PER_TASK:-16}" "${RUN_CMD[@]}"
