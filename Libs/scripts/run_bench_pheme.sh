#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROFILE="benchmark"
DATASET_NAME="pheme"
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
  CUDA_DEV=$(profile_get_cuda_device "${EXP_CFG_DEFAULT}" "${PROFILE}" "0" "0")
  SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3407, 126, 218, 91])
print(",".join(str(seed) for seed in seeds))
PY
)
  MODELS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" "${DATASET_NAME}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
dataset_name = sys.argv[2]
bench_models = cfg.get("bench_models", {})
models = bench_models.get(dataset_name, bench_models.get("default", ["textonly", "bigcn", "tgat", "lp_gcn", "dga_fake", "rumordiffusion"]))
print(" ".join(str(model) for model in models))
PY
)
  SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
  MODELS="${MODELS_OVERRIDE:-${MODELS_DEFAULT}}"
  EVAL_ONLY="${EVAL_ONLY_OVERRIDE:-0}"
  CKPT_OVERRIDE_VALUE="${CKPT_OVERRIDE:-}"
  DEPENDENCY_VALUE="${SBATCH_DEPENDENCY:-}"
  JOB_NAME="${JOB_PREFIX}_bench_${DATASET_NAME}"
  mkdir -p "${LOG_DIR}"
  sinfo
  export SEEDS_OVERRIDE="${SEEDS}"
  export MODELS_OVERRIDE="${MODELS}"
  export EVAL_ONLY_OVERRIDE="${EVAL_ONLY}"
  export CKPT_OVERRIDE="${CKPT_OVERRIDE_VALUE}"
  export SOURCE_DATASET_OVERRIDE="${SOURCE_DATASET_OVERRIDE:-}"
  SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --ntasks=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEM}"
    --output="${LOG_DIR}/${JOB_NAME}_%j.log"
    --error="${LOG_DIR}/${JOB_NAME}_%j.err"
    --export=ALL,EXP_CFG="${EXP_CFG_DEFAULT}",CUDA_VISIBLE_DEVICES="${CUDA_DEV}",CONDA_ENV_OVERRIDE="${CONDA_ENV}"
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
  sbatch "${SBATCH_ARGS[@]}" Libs/scripts/run_bench_pheme.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")}"
LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_bench_pheme}"

SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3407, 126, 218, 91])
print(",".join(str(seed) for seed in seeds))
PY
)
MODELS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" "${DATASET_NAME}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
dataset_name = sys.argv[2]
bench_models = cfg.get("bench_models", {})
models = bench_models.get(dataset_name, bench_models.get("default", ["textonly", "bigcn", "tgat", "lp_gcn", "dga_fake", "rumordiffusion"]))
print(" ".join(str(model) for model in models))
PY
)
SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
MODELS_STRING="${MODELS_OVERRIDE:-${MODELS_DEFAULT}}"
EVAL_ONLY="${EVAL_ONLY_OVERRIDE:-0}"
CKPT_OVERRIDE_VALUE="${CKPT_OVERRIDE:-}"
SOURCE_DATASET_NAME="${SOURCE_DATASET_OVERRIDE:-${DATASET_NAME}}"
MODELS=(${MODELS_STRING})

CONFIGS=""
for model in "${MODELS[@]}"; do
  if [ -z "${CONFIGS}" ]; then
    CONFIGS="Libs/configs/baselines/${model}.yaml"
  else
    CONFIGS="${CONFIGS},Libs/configs/baselines/${model}.yaml"
  fi
done

RUN_CMD=(
  python3 -m Libs.exp.run
  --exp "${EXP_CFG_DEFAULT}"
  --model "${CONFIGS}"
  --dataset_override "${DATASET_NAME}"
  --source_dataset_override "${SOURCE_DATASET_NAME}"
  --seeds "${SEEDS}"
)
if [ "${EVAL_ONLY}" = "1" ]; then
  RUN_CMD+=(--eval_only)
fi
if [ -n "${CKPT_OVERRIDE_VALUE}" ]; then
  RUN_CMD+=(--ckpt "${CKPT_OVERRIDE_VALUE}")
fi
log "Dataset: ${DATASET_NAME}"
log "Source dataset: ${SOURCE_DATASET_NAME}"
log "Models: ${MODELS[*]}"
log "Seeds: ${SEEDS}"
log_run_header "${RUN_CMD[*]}"
"${RUN_CMD[@]}"
