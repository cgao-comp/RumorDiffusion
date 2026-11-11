#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROFILE="explain_summary"
EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp_paper_ready.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  JOB_PREFIX=$(resolve_project_prefix)
  PARTITION=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "partition" "cpu")
  NODELIST=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "nodelist" "cpu04")
  GRES=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "gres" "")
  CPUS_PER_TASK=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "cpus_per_task" "2")
  MEM=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "mem" "4G")
  LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
  CONDA_ENV=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")
  DATASET_VALUE="${DATASET_NAME_OVERRIDE:-auto}"
  DEPENDENCY_VALUE="${SBATCH_DEPENDENCY:-}"
  JOB_NAME="${JOB_PREFIX}_explain_summary_${DATASET_VALUE}"
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
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",DATASET_NAME_OVERRIDE="${DATASET_NAME_OVERRIDE:-}",INPUT_PATH="${INPUT_PATH:-}",OUT_PATH="${OUT_PATH:-}",TOPK="${TOPK:-20}",CONDA_ENV_OVERRIDE="${CONDA_ENV}"
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
  sbatch "${SBATCH_ARGS[@]}" Libs/scripts/run_rumordiffusion_explain_summary.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")}"
LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_explain_summary}"

DATASET_NAME=${DATASET_NAME_OVERRIDE:-$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("dataset", {}).get("name", "twitter15"))
PY
)}
RESULTS_DIR=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("outputs", {}).get("results_dir", os.path.join("Output", "results", "paper_aligned")))
PY
)
INPUT_PATH_VALUE="${INPUT_PATH:-${RESULTS_DIR}/${DATASET_NAME}_explain.jsonl}"
OUT_PATH_VALUE="${OUT_PATH:-${RESULTS_DIR}/${DATASET_NAME}_explain_summary.csv}"
TOPK_VALUE="${TOPK:-20}"
RUN_CMD=(
  python3 -m Libs.exp.explain_summary
  --input "${INPUT_PATH_VALUE}"
  --out "${OUT_PATH_VALUE}"
  --topk "${TOPK_VALUE}"
)
log "Explain summary input: ${INPUT_PATH_VALUE}"
log "Explain summary output: ${OUT_PATH_VALUE}"
log_run_header "${RUN_CMD[*]}"
"${RUN_CMD[@]}"
