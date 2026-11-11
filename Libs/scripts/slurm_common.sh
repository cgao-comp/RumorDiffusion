#!/bin/bash
set -euo pipefail

# Submission is deliberately opt-in. A shell invocation is not evidence that
# the repository owner explicitly authorized a scheduler state change.
if [ "${RUMORDIFFUSION_SLURM_CONFIRMATION:-}" != "I_HAVE_EXPLICIT_USER_CONFIRMATION" ]; then
  sbatch() {
    echo "Refusing sbatch: set RUMORDIFFUSION_SLURM_CONFIRMATION=I_HAVE_EXPLICIT_USER_CONFIRMATION after explicit user approval." >&2
    return 64
  }
fi

profile_get() {
  local exp_cfg="$1"
  local profile="$2"
  local field="$3"
  local default_value="$4"
  python3 - "$exp_cfg" "$profile" "$field" "$default_value" <<'PY'
import sys
import yaml

exp_cfg, profile, field, default_value = sys.argv[1:]
cfg = yaml.safe_load(open(exp_cfg, "r", encoding="utf-8"))
sbatch_cfg = cfg.get("sbatch", {}) if isinstance(cfg.get("sbatch", {}), dict) else {}
profiles = sbatch_cfg.get("profiles", {}) if isinstance(sbatch_cfg.get("profiles", {}), dict) else {}
profile_cfg = profiles.get(profile, {}) if isinstance(profiles.get(profile, {}), dict) else {}
value = profile_cfg.get(field, sbatch_cfg.get(field, default_value))
if value is None:
    value = ""
print(value)
PY
}

profile_get_cuda_device() {
  local exp_cfg="$1"
  local profile="$2"
  local index="$3"
  local default_value="$4"
  python3 - "$exp_cfg" "$profile" "$index" "$default_value" <<'PY'
import sys
import yaml

exp_cfg, profile, index_text, default_value = sys.argv[1:]
cfg = yaml.safe_load(open(exp_cfg, "r", encoding="utf-8"))
sbatch_cfg = cfg.get("sbatch", {}) if isinstance(cfg.get("sbatch", {}), dict) else {}
profiles = sbatch_cfg.get("profiles", {}) if isinstance(sbatch_cfg.get("profiles", {}), dict) else {}
profile_cfg = profiles.get(profile, {}) if isinstance(profiles.get(profile, {}), dict) else {}
devices = profile_cfg.get("cuda_devices", sbatch_cfg.get("cuda_devices", [default_value]))
if not isinstance(devices, list) or len(devices) == 0:
    print(default_value)
else:
    index = int(index_text)
    safe_index = min(max(index, 0), len(devices) - 1)
    print(devices[safe_index])
PY
}

resolve_project_prefix() {
  local repo_root="/share/home/hwxu/Researches/D2Plus"
  local prefix_file="${repo_root}/PROJECT_PREFIX"
  if [ ! -f "${prefix_file}" ]; then
    echo "Missing PROJECT_PREFIX at ${prefix_file}" >&2
    return 1
  fi
  local prefix
  prefix="$(tr -d '[:space:]' < "${prefix_file}")"
  if [[ ! "${prefix}" =~ ^[A-Z][A-Za-z]{2,7}$ ]]; then
    echo "Invalid PROJECT_PREFIX '${prefix}'. Expected CamelCase letters, length 3-8." >&2
    return 1
  fi
  printf '%s\n' "${prefix}"
}

ensure_repo_pythonpath() {
  local repo_root="/share/home/hwxu/Researches/D2Plus"
  case ":${PYTHONPATH:-}:" in
    *":${repo_root}:"*) ;;
    *) export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

activate_conda_env() {
  local conda_env="$1"
  export USE_TF=0
  export USE_TORCH=1
  export TRANSFORMERS_NO_TF=1
  ensure_repo_pythonpath
  set +u
  source /share/home/hwxu/miniconda3/bin/activate "${conda_env}"
  set -u
  export USE_TF=0
  export USE_TORCH=1
  export TRANSFORMERS_NO_TF=1
  ensure_repo_pythonpath
}

record_env_snapshot() {
  local log_dir="$1"
  local job_name="$2"
  local job_id="${SLURM_JOB_ID:-local}"
  mkdir -p "${log_dir}"
  python3 -m pip freeze > "${log_dir}/${job_name}_${job_id}_pip_freeze.txt"
  env | sort > "${log_dir}/${job_name}_${job_id}_env.txt"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "${log_dir}/${job_name}_${job_id}_gpu.csv"
  fi
}

log_run_header() {
  local cmd_text="$1"
  log "Host: $(hostname)"
  log "SLURM job: ${SLURM_JOB_NAME:-local}/${SLURM_JOB_ID:-local}"
  log "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
  log "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  log "Python: $(python3 --version)"
  log "Git HEAD: $(git rev-parse HEAD)"
  log "Command: ${cmd_text}"
}
