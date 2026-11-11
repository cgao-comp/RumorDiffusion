#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROFILE="hparam"
EXP_CFG_DEFAULT="${EXP_CFG:-Libs/configs/exp_paper_ready.yaml}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  cd /share/home/hwxu/Researches/D2Plus
  JOB_PREFIX=$(resolve_project_prefix)
  PARTITION=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "partition" "gpu1")
  NODELIST=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "nodelist" "gpu03")
  GRES=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "gres" "gpu:1")
  CPUS_PER_TASK=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "cpus_per_task" "8")
  MEM=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "mem" "32G")
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
  MODEL_CFGS_DEFAULT="Libs/configs/baselines/rumordiffusion.yaml,Libs/configs/baselines/rumordiffusion_proto_prior.yaml,Libs/configs/baselines/rumordiffusion_hparam_depth.yaml,Libs/configs/baselines/rumordiffusion_hparam_guidance.yaml"
  MODEL_CFGS="${MODEL_CFGS_OVERRIDE:-${MODEL_CFGS_DEFAULT}}"
  SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
  DATASET_VALUE="${DATASET_OVERRIDE:-all}"
  DEPENDENCY_VALUE="${SBATCH_DEPENDENCY:-}"
  JOB_NAME="${JOB_PREFIX}_hparam_${DATASET_VALUE}"
  mkdir -p "${LOG_DIR}"
  sinfo
  export MODEL_CFGS_OVERRIDE="${MODEL_CFGS}"
  export SEEDS_OVERRIDE="${SEEDS}"
  export DATASET_OVERRIDE="${DATASET_OVERRIDE:-}"
  export SOURCE_DATASET_OVERRIDE="${SOURCE_DATASET_OVERRIDE:-}"
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
  sbatch "${SBATCH_ARGS[@]}" Libs/scripts/run_rumordiffusion_hparam.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")}"
LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_hparam}"

SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3407, 126, 218, 91])
print(",".join(str(seed) for seed in seeds))
PY
)
MODEL_CFGS_DEFAULT="Libs/configs/baselines/rumordiffusion.yaml,Libs/configs/baselines/rumordiffusion_proto_prior.yaml,Libs/configs/baselines/rumordiffusion_hparam_depth.yaml,Libs/configs/baselines/rumordiffusion_hparam_guidance.yaml"
MODEL_CFGS="${MODEL_CFGS_OVERRIDE:-${MODEL_CFGS_DEFAULT}}"
SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
DATASET_VALUE="${DATASET_OVERRIDE:-}"
SOURCE_DATASET_NAME="${SOURCE_DATASET_OVERRIDE:-${DATASET_VALUE}}"
RUN_CMD=(
  python3 -m Libs.exp.run
  --exp "${EXP_CFG_DEFAULT}"
  --model "${MODEL_CFGS}"
  --seeds "${SEEDS}"
)
if [ -n "${DATASET_VALUE}" ]; then
  RUN_CMD+=(--dataset_override "${DATASET_VALUE}")
  RUN_CMD+=(--source_dataset_override "${SOURCE_DATASET_NAME}")
fi
log "Hyperparameter configs: ${MODEL_CFGS}"
log "Seeds: ${SEEDS}"
if [ -n "${DATASET_VALUE}" ]; then
  log "Dataset override: ${DATASET_VALUE}"
  log "Source dataset: ${SOURCE_DATASET_NAME}"
fi
log_run_header "${RUN_CMD[*]}"
"${RUN_CMD[@]}"
