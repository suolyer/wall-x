#!/usr/bin/env bash
set -euo pipefail

# LIBERO evaluation launcher. Configure with environment variables:
#
#   CHECKPOINT_PATH=/path/to/checkpoint bash scripts/run_libero.sh
#   bash scripts/run_libero.sh /path/to/checkpoint
#   CHECKPOINT_PATH=/path/to/checkpoint SMOKE=1 bash scripts/run_libero.sh
#   CONFIG=/path/to/eval.yaml bash scripts/run_libero.sh
#
# Optional knobs:
#   CUDA_ID=0
#   DRIVER_MODE=in_process
#   NUM_WORKERS=1
#   MAX_BATCH_SIZE=1
#   ALL_SUITES=1                    # run all standard LIBERO suites (40 tasks)
#   TASK_SUITES="libero_spatial ..." # custom suite list (space- or comma-separated)
#   TASK_SUITE_NAME=libero_spatial  # single suite when ALL_SUITES=0 and TASK_SUITES unset
#   NUM_TRIALS_PER_TASK=50
#   TASK_INDICES=0,1,2              # omit to run every task in the suite
#   MAX_INFER_TIMES=52
#   NORM_KEY=libero_all
#   ROLLOUT_BASE=/path/to/rollout   # per-suite logs under ${ROLLOUT_BASE}/${suite}/
#   LOG_DIR=/tmp/harrix_libero_eval   # overrides ROLLOUT_BASE when set (single suite)

export PYTHONPATH="/path/to/LIBERO:${PYTHONPATH:-}"

export ALL_SUITES=1
# export TASK_SUITE_NAME=libero_10
# export CUDA_ID=0
export NUM_WORKERS=10
export NUM_TRIALS_PER_TASK=20
export CHECKPOINT_PATH=/path/to/checkpoint
export ROLLOUT_BASE=/path/to/rollout

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_ALL_SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/run_libero.sh [CHECKPOINT_PATH]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    CHECKPOINT_PATH="$1"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_ID:-0}"

# MuJoCo offscreen rendering via NVIDIA EGL (required on headless GPU nodes).
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
EGL_VENDOR_DIR="${HOME}/.config/glvnd/egl_vendor.d"
EGL_VENDOR_FILE="${EGL_VENDOR_DIR}/10_nvidia.json"
if [[ ! -f "${EGL_VENDOR_FILE}" ]]; then
    mkdir -p "${EGL_VENDOR_DIR}"
    cat > "${EGL_VENDOR_FILE}" <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
fi
export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_FILE}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

# After CUDA_VISIBLE_DEVICES remapping, MuJoCo only sees devices from index 0.
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
if [[ -d "${REPO_ROOT}/third_party/harrix/python" ]]; then
    export PYTHONPATH="${REPO_ROOT}/third_party/harrix/python:${PYTHONPATH:-}"
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

resolve_task_suites() {
    local suites=()
    if [[ "${ALL_SUITES:-0}" == "1" ]]; then
        if [[ -n "${TASK_SUITES:-}" ]]; then
            TASK_SUITES="${TASK_SUITES//,/ }"
            read -r -a suites <<< "${TASK_SUITES}"
        else
            suites=("${DEFAULT_ALL_SUITES[@]}")
        fi
    elif [[ -n "${TASK_SUITES:-}" ]]; then
        TASK_SUITES="${TASK_SUITES//,/ }"
        read -r -a suites <<< "${TASK_SUITES}"
    else
        suites=("${TASK_SUITE_NAME:-libero_spatial}")
    fi
    echo "${suites[@]}"
}

resolve_log_dirs() {
    local suite="$1"
    local multi_suite="$2"
    local log_dir rollout_dir

    if [[ -n "${LOG_DIR:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            log_dir="${LOG_DIR}/${suite}"
        else
            log_dir="${LOG_DIR}"
        fi
    elif [[ -n "${ROLLOUT_BASE:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            log_dir="${ROLLOUT_BASE}/${suite}/json"
        else
            log_dir="${ROLLOUT_BASE}/json"
        fi
    else
        log_dir="/tmp/harrix_libero_eval"
    fi

    if [[ -n "${WALLX_ROLLOUT_DIR:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            rollout_dir="${WALLX_ROLLOUT_DIR}/${suite}"
        else
            rollout_dir="${WALLX_ROLLOUT_DIR}"
        fi
    elif [[ -n "${ROLLOUT_BASE:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            rollout_dir="${ROLLOUT_BASE}/${suite}/videos"
        else
            rollout_dir="${ROLLOUT_BASE}/videos"
        fi
    else
        rollout_dir=""
    fi

    mkdir -p "${log_dir}"
    if [[ -n "${rollout_dir}" ]]; then
        mkdir -p "${rollout_dir}"
    fi
    LOG_DIR_RESOLVED="${log_dir}"
    WALLX_ROLLOUT_DIR_RESOLVED="${rollout_dir}"
}

run_suite() {
    local suite="$1"
    local multi_suite="$2"

    resolve_log_dirs "${suite}" "${multi_suite}"
    if [[ -n "${WALLX_ROLLOUT_DIR_RESOLVED}" ]]; then
        export WALLX_ROLLOUT_DIR="${WALLX_ROLLOUT_DIR_RESOLVED}"
    else
        unset WALLX_ROLLOUT_DIR
    fi

    local args=()
    if [[ -n "${CONFIG:-}" ]]; then
        args+=(--config "${CONFIG}")
    else
        if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
            echo "CHECKPOINT_PATH is required when CONFIG is not set." >&2
            exit 2
        fi
        args+=(--checkpoint-path "${CHECKPOINT_PATH}")
    fi

    if [[ -n "${TRAIN_CONFIG_PATH:-}" ]]; then
        args+=(--train-config-path "${TRAIN_CONFIG_PATH}")
    fi
    if [[ "${SMOKE:-0}" == "1" ]]; then
        args+=(--smoke)
    fi
    if [[ "${DET:-0}" == "1" ]]; then
        args+=(--deterministic-model)
    fi
    if [[ -n "${TASK_INDICES:-}" ]]; then
        args+=(--task-indices "${TASK_INDICES}")
    fi
    if [[ -n "${MAX_INFER_TIMES:-}" ]]; then
        args+=(--max-infer-times "${MAX_INFER_TIMES}")
    fi

    args+=(--driver-mode "${DRIVER_MODE:-in_process}")
    args+=(--num-workers "${NUM_WORKERS:-1}")
    args+=(--max-batch-size "${MAX_BATCH_SIZE:-1}")
    args+=(--task-suite-name "${suite}")
    args+=(--num-trials-per-task "${NUM_TRIALS_PER_TASK:-50}")
    args+=(--norm-key "${NORM_KEY:-libero_all}")
    args+=(--log-dir "${LOG_DIR_RESOLVED}")

    echo "=== LIBERO eval: suite=${suite} log_dir=${LOG_DIR_RESOLVED} ==="
    python scripts/infer_libero.py "${args[@]}"
}

suites=($(resolve_task_suites))
multi_suite=0
if [[ "${#suites[@]}" -gt 1 ]]; then
    multi_suite=1
fi

cd "${REPO_ROOT}"
for suite in "${suites[@]}"; do
    run_suite "${suite}" "${multi_suite}"
done
