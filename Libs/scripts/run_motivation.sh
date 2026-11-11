#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROFILE="motivation"
EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp_paper_ready.yaml}"
CONFIG_VALUE="${CONFIG:-Libs/configs/motivation.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  JOB_PREFIX=$(resolve_project_prefix)
  PARTITION=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "partition" "cpu")
  NODELIST=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "nodelist" "cpu04")
  GRES=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "gres" "")
  CPUS_PER_TASK=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "cpus_per_task" "4")
  MEM=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "mem" "16G")
  LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
  CONDA_ENV=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")
  DEPENDENCY_VALUE="${SBATCH_DEPENDENCY:-}"
  JOB_NAME="${JOB_PREFIX}_motivation"
  mkdir -p "${LOG_DIR}"
  sinfo
  SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --output="${LOG_DIR}/${JOB_NAME}_%j.log"
    --error="${LOG_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",CONFIG="${CONFIG_VALUE}",EXPERIMENT_OVERRIDE="${EXPERIMENT_OVERRIDE:-}",CONDA_ENV_OVERRIDE="${CONDA_ENV}"
  )
  if [ -n "${NODELIST}" ]; then
    SBATCH_ARGS+=(--nodelist="${NODELIST}")
  fi
  if [ -n "${GRES}" ]; then
    SBATCH_ARGS+=(--gres="${GRES}")
  fi
  if [ -n "${DEPENDENCY_VALUE}" ]; then
    SBATCH_ARGS+=(--dependency="${DEPENDENCY_VALUE}")
  fi
  sbatch "${SBATCH_ARGS[@]}" Libs/scripts/run_motivation.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")}"
LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_motivation}"

RUN_CMD=(
  python3 -m Libs.exp.motivation_experiment
  --config "${CONFIG_VALUE}"
)
if [ -n "${EXPERIMENT_OVERRIDE:-}" ]; then
  RUN_CMD+=(--experiment "${EXPERIMENT_OVERRIDE}")
fi
log "Motivation config: ${CONFIG_VALUE}"
log_run_header "${RUN_CMD[*]}"
"${RUN_CMD[@]}"
