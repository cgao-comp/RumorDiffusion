#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROFILE="explain"
EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp_paper_ready.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  JOB_PREFIX=$(resolve_project_prefix)
  PARTITION=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "partition" "gpu1")
  NODELIST=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "nodelist" "gpu03")
  GRES=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "gres" "gpu:1")
  CPUS_PER_TASK=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "cpus_per_task" "8")
  MEM=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "mem" "16G")
  LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
  CONDA_ENV=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")
  CUDA_DEV=$(profile_get_cuda_device "${EXP_CFG_DEFAULT}" "${PROFILE}" "0" "0")
  DATASET_VALUE="${DATASET_NAME_OVERRIDE:-auto}"
  DEPENDENCY_VALUE="${SBATCH_DEPENDENCY:-}"
  JOB_NAME="${JOB_PREFIX}_explain_${DATASET_VALUE}"
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
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",MODEL_CFG="${MODEL_CFG:-Libs/configs/baselines/rumordiffusion.yaml}",DATASET_NAME_OVERRIDE="${DATASET_NAME_OVERRIDE:-}",SEED_OVERRIDE="${SEED_OVERRIDE:-}",CKPT_PATH_OVERRIDE="${CKPT_PATH_OVERRIDE:-}",OUT_PATH_OVERRIDE="${OUT_PATH_OVERRIDE:-}",TRACE_OUT_OVERRIDE="${TRACE_OUT_OVERRIDE:-}",SELECTED_OUT_OVERRIDE="${SELECTED_OUT_OVERRIDE:-}",SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-}",SOURCE_IDS_OVERRIDE="${SOURCE_IDS_OVERRIDE:-}",CUDA_VISIBLE_DEVICES="${CUDA_DEV}",CONDA_ENV_OVERRIDE="${CONDA_ENV}"
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
  sbatch "${SBATCH_ARGS[@]}" Libs/scripts/run_rumordiffusion_explain.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")}"
LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_explain}"

MODEL_CFG_VALUE="${MODEL_CFG:-Libs/configs/baselines/rumordiffusion.yaml}"
DATASET_NAME=${DATASET_NAME_OVERRIDE:-$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("dataset", {}).get("name", "twitter15"))
PY
)}
SEED_VALUE=${SEED_OVERRIDE:-$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42])
print(seeds[0] if seeds else 42)
PY
)}
CKPT_ROOT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("outputs", {}).get("ckpt_dir", os.path.join("Output", "ckpts", "paper_aligned")))
PY
)
RESULTS_DIR=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(cfg.get("outputs", {}).get("results_dir", os.path.join("Output", "results", "paper_aligned")))
PY
)
CKPT_PATH=${CKPT_PATH_OVERRIDE:-"${CKPT_ROOT}/${DATASET_NAME}/rumordiffusion/seed${SEED_VALUE}/best.pt"}
OUT_PATH=${OUT_PATH_OVERRIDE:-"${RESULTS_DIR}/${DATASET_NAME}_explain.jsonl"}
TRACE_OUT=${TRACE_OUT_OVERRIDE:-"${RESULTS_DIR}/${DATASET_NAME}_case_traces.jsonl"}
SELECTED_OUT=${SELECTED_OUT_OVERRIDE:-"${RESULTS_DIR}/${DATASET_NAME}_selected_cases.json"}
SPLIT=${SPLIT_OVERRIDE:-test}
SOURCE_IDS=${SOURCE_IDS_OVERRIDE:-}

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

RUN_CMD=(
  python3 -m Libs.exp.rumordiffusion_explain
  --exp "${EXP_CFG_DEFAULT}"
  --model "${MODEL_CFG_VALUE}"
  --ckpt "${CKPT_PATH}"
  --split "${SPLIT}"
  --out "${OUT_PATH}"
  --trace_out "${TRACE_OUT}"
  --selected_out "${SELECTED_OUT}"
)
if [ -n "${SOURCE_IDS}" ]; then
  RUN_CMD+=(--source_ids "${SOURCE_IDS}")
fi
log "Explain dataset: ${DATASET_NAME}"
log "Explain seed: ${SEED_VALUE}"
log "Checkpoint: ${CKPT_PATH}"
log_run_header "${RUN_CMD[*]}"
"${RUN_CMD[@]}"
