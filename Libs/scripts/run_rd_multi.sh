#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROFILE="rumordiffusion_train"
EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp_paper_ready.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  JOB_PREFIX=$(resolve_project_prefix)
  PARTITION=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "partition" "gpu1")
  NODELIST=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "nodelist" "gpu03")
  GRES=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "gres" "gpu:1")
  CPUS_PER_TASK=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "cpus_per_task" "8")
  MEM=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "mem" "48G")
  LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
  CONDA_ENV=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")
  SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3407, 126, 218, 91])
print(",".join(str(seed) for seed in seeds))
PY
)
  DATASETS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
datasets = cfg.get("multi_run", {}).get("datasets", ["twitter15", "twitter16", "weibo", "pheme"])
print(" ".join(str(dataset) for dataset in datasets))
PY
)
  SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
  DATASETS_VALUE="${DATASETS_OVERRIDE:-${DATASETS_DEFAULT}}"
  MODEL_CFG_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
print(str(cfg.get("model_cfg", "Libs/configs/baselines/rumordiffusion.yaml")))
PY
)
  MODEL_CFG_VALUE="${MODEL_CFG:-${MODEL_CFG_DEFAULT}}"
  DEPENDENCY_VALUE="${SBATCH_DEPENDENCY:-}"
  DATASET_TAG=$(printf '%s' "${DATASETS_VALUE}" | tr ' ' '_' | tr ',' '_')
  if [ -z "${DATASET_TAG}" ]; then
    DATASET_TAG="multi"
  fi
  JOB_NAME="${JOB_PREFIX}_rumordiffusion_${DATASET_TAG}"
  mkdir -p "${LOG_DIR}"
  sinfo
  export SEEDS_OVERRIDE="${SEEDS}"
  export DATASETS_OVERRIDE="${DATASETS_VALUE}"
  export MODEL_CFG="${MODEL_CFG_VALUE}"
  SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --output="${LOG_DIR}/${JOB_NAME}_%j.log"
    --error="${LOG_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",CONDA_ENV_OVERRIDE="${CONDA_ENV}"
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
  sbatch "${SBATCH_ARGS[@]}" Libs/scripts/run_rd_multi.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")}"
LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_rumordiffusion_multi}"

SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3407, 126, 218, 91])
print(",".join(str(seed) for seed in seeds))
PY
)
DATASETS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
datasets = cfg.get("multi_run", {}).get("datasets", ["twitter15", "twitter16", "weibo", "pheme"])
print(" ".join(str(dataset) for dataset in datasets))
PY
)
SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
DATASETS=(${DATASETS_OVERRIDE:-${DATASETS_DEFAULT}})
MODEL_CFG_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")) or {}
print(str(cfg.get("model_cfg", "Libs/configs/baselines/rumordiffusion.yaml")))
PY
)
MODEL_CFG_VALUE="${MODEL_CFG:-${MODEL_CFG_DEFAULT}}"

for dataset_name in "${DATASETS[@]}"; do
  RUN_CMD=(
    python3 -m Libs.exp.run
    --exp "${EXP_CFG_DEFAULT}"
    --model "${MODEL_CFG_VALUE}"
    --seeds "${SEEDS}"
    --dataset_override "${dataset_name}"
    --source_dataset_override "${SOURCE_DATASET_OVERRIDE:-${dataset_name}}"
  )
  log "Dataset: ${dataset_name}"
  log "Source dataset: ${SOURCE_DATASET_OVERRIDE:-${dataset_name}}"
  log "Seeds: ${SEEDS}"
  log_run_header "${RUN_CMD[*]}"
  "${RUN_CMD[@]}"
done
