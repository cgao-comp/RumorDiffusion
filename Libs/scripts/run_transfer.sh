#!/bin/bash
set -euo pipefail

source /share/home/hwxu/Researches/D2Plus/Libs/scripts/slurm_common.sh

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

PROFILE="transfer"
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
  MODEL_CFG_VALUE="${MODEL_CFG:-Libs/configs/baselines/rumordiffusion.yaml}"
  SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
  TRANSFER_MODE_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(str(cfg.get("transfer", {}).get("mode", "source_only")))
PY
)
  TRANSFER_MODE_VALUE="${TRANSFER_MODE_OVERRIDE:-${TRANSFER_MODE_DEFAULT}}"
  TRANSFER_PAIRS_VALUE="${TRANSFER_PAIRS_OVERRIDE:-}"
  DEPENDENCY_VALUE="${SBATCH_DEPENDENCY:-}"
  JOB_NAME="${JOB_PREFIX}_transfer_eval"
  mkdir -p "${LOG_DIR}"
  sinfo
  export MODEL_CFG="${MODEL_CFG_VALUE}"
  export SEEDS_OVERRIDE="${SEEDS}"
  export TRANSFER_MODE_OVERRIDE="${TRANSFER_MODE_VALUE}"
  export TRANSFER_PAIRS_OVERRIDE="${TRANSFER_PAIRS_VALUE}"
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
  sbatch "${SBATCH_ARGS[@]}" Libs/scripts/run_transfer.sh sbatch
  exit $?
fi

CONDA_ENV="${CONDA_ENV_OVERRIDE:-$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "conda_env" "hw")}"
LOG_DIR=$(profile_get "${EXP_CFG_DEFAULT}" "${PROFILE}" "log_dir" "Output/logs/paper_aligned")
activate_conda_env "${CONDA_ENV}"
cd /share/home/hwxu/Researches/D2Plus
mkdir -p "${LOG_DIR}"
record_env_snapshot "${LOG_DIR}" "${SLURM_JOB_NAME:-DPlus_transfer_eval}"

SEEDS_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
seeds = cfg.get("experiment", {}).get("seeds", [42, 3407, 126, 218, 91])
print(",".join(str(seed) for seed in seeds))
PY
)
SEEDS="${SEEDS_OVERRIDE:-${SEEDS_DEFAULT}}"
MODEL_CFG_VALUE="${MODEL_CFG:-Libs/configs/baselines/rumordiffusion.yaml}"
TRANSFER_MODE_DEFAULT=$(python3 - "${EXP_CFG_DEFAULT}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(str(cfg.get("transfer", {}).get("mode", "source_only")))
PY
)
TRANSFER_MODE_VALUE="${TRANSFER_MODE_OVERRIDE:-${TRANSFER_MODE_DEFAULT}}"
TRANSFER_PAIRS_VALUE="${TRANSFER_PAIRS_OVERRIDE:-}"
RUN_CMD=(
  python3 -m Libs.exp.run
  --exp "${EXP_CFG_DEFAULT}"
  --model "${MODEL_CFG_VALUE}"
  --seeds "${SEEDS}"
  --lp_edges_policy require_existing
)
if [ "${TRANSFER_MODE_VALUE}" = "target_adapt" ]; then
  RUN_CMD+=(--transfer_adapt)
else
  RUN_CMD+=(--transfer_eval)
fi
if [ -n "${TRANSFER_PAIRS_VALUE}" ]; then
  RUN_CMD+=(--transfer_pairs "${TRANSFER_PAIRS_VALUE}")
fi
log "Transfer mode: ${TRANSFER_MODE_VALUE}"
log "Transfer pairs override: ${TRANSFER_PAIRS_VALUE:-config}"
log "Transfer evaluation seeds: ${SEEDS}"
log_run_header "${RUN_CMD[*]}"
"${RUN_CMD[@]}"
