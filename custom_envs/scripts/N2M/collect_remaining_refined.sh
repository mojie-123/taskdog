#!/usr/bin/env bash

# Run the remaining manipulation-pose searches sequentially.  The script is
# safe to start again: a dataset that already has summaries is resumed, and a
# completed dataset is skipped by find_manipulation_poses.py.

set -Eeuo pipefail

readonly PROJECT_ROOT="/home/mojie/taskdog"
readonly COLLECTOR="${PROJECT_ROOT}/custom_envs/scripts/N2M/find_manipulation_poses.py"
readonly DATA_ROOT="${PROJECT_ROOT}/custom_envs/scripts/N2M/data"
readonly CHECKPOINT="/home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"
readonly RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_DIR="${DATA_ROOT}/overnight_logs/${RUN_STAMP}"

export MPLBACKEND=Agg

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

# Prevent two unattended collectors from writing the same JSONL files.
exec 9>/tmp/taskdog_n2m_refined_collection.lock
if ! flock -n 9; then
    printf '[N2M-BATCH] Another refined collection batch is already running.\n' >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    printf '[N2M-BATCH] AnyGrasp checkpoint not found: %s\n' "${CHECKPOINT}" >&2
    exit 1
fi

# ROS/Conda setup scripts reference optional environment variables and are not
# safe under Bash nounset, so temporarily relax it while sourcing them.
set +u
source /opt/ros/humble/setup.bash
set -u

# A tmux window opened from a fresh shell may not inherit an activated Conda
# environment, so activate the known project environment when necessary.
if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]]; then
    readonly CONDA_PROFILE="/home/mojie/anaconda3/etc/profile.d/conda.sh"
    if [[ ! -f "${CONDA_PROFILE}" ]]; then
        printf '[N2M-BATCH] Conda profile not found: %s\n' "${CONDA_PROFILE}" >&2
        exit 1
    fi
    set +u
    source "${CONDA_PROFILE}"
    conda activate env_isaaclab
    set -u
fi

run_collection() {
    local job_name="$1"
    local scene_id="$2"
    local data_dir="$3"
    local expected_candidates="$4"
    shift 4
    local -a yaw_offsets=("$@")
    local log_path="${LOG_DIR}/${job_name}.log"
    local max_launch_attempts=3
    local launch_attempt

    printf '\n[N2M-BATCH] Starting %s at %s\n' "${job_name}" "$(date --iso-8601=seconds)" | tee -a "${log_path}"
    printf '[N2M-BATCH] Data directory: %s\n' "${data_dir}" | tee -a "${log_path}"

    for ((launch_attempt = 1; launch_attempt <= max_launch_attempts; launch_attempt++)); do
        local -a resume_args=()
        if [[ -s "${data_dir}/candidate_summaries.jsonl" ]]; then
            resume_args=(--resume)
            printf '[N2M-BATCH] Resume enabled (launch attempt %d/%d).\n' \
                "${launch_attempt}" "${max_launch_attempts}" | tee -a "${log_path}"
        else
            printf '[N2M-BATCH] Fresh dataset (launch attempt %d/%d).\n' \
                "${launch_attempt}" "${max_launch_attempts}" | tee -a "${log_path}"
        fi

        if PYTHONUNBUFFERED=1 python "${COLLECTOR}" \
            --mode collect \
            --cube-scene "${scene_id}" \
            --normal-offsets 0 -0.05 -0.10 -0.15 \
            --tangent-offsets 0 -0.125 0.125 -0.25 0.25 \
            --yaw-offsets-deg "${yaw_offsets[@]}" \
            --confirm-repeats 5 \
            --confirm-min-success 2 \
            --grasp-checkpoint "${CHECKPOINT}" \
            --headless \
            --data-dir "${data_dir}" \
            "${resume_args[@]}" 2>&1 | tee -a "${log_path}"; then
            break
        fi

        if ((launch_attempt == max_launch_attempts)); then
            printf '[N2M-BATCH] %s failed after %d launch attempts. Stopping batch.\n' \
                "${job_name}" "${max_launch_attempts}" | tee -a "${log_path}" >&2
            return 1
        fi

        printf '[N2M-BATCH] %s exited with an error; retrying with --resume in 10 seconds.\n' \
            "${job_name}" | tee -a "${log_path}" >&2
        sleep 10
    done

    local summary_path="${data_dir}/candidate_summaries.jsonl"
    local actual_candidates=0
    if [[ -f "${summary_path}" ]]; then
        actual_candidates="$(wc -l < "${summary_path}")"
    fi
    if [[ "${actual_candidates}" -ne "${expected_candidates}" ]]; then
        printf '[N2M-BATCH] %s ended with %d/%d summaries. Stopping batch.\n' \
            "${job_name}" "${actual_candidates}" "${expected_candidates}" \
            | tee -a "${log_path}" >&2
        return 1
    fi

    printf '[N2M-BATCH] Completed %s: %d/%d summaries at %s\n' \
        "${job_name}" "${actual_candidates}" "${expected_candidates}" \
        "$(date --iso-8601=seconds)" | tee -a "${log_path}"
}

run_collection \
    "long_bottom_refined" \
    "long_bottom" \
    "${DATA_ROOT}/collect_long_bottom_refined" \
    140 \
    0 -11.25 11.25 -22.5 22.5 -30 30

run_collection \
    "long_top_refined" \
    "long_top" \
    "${DATA_ROOT}/collect_long_top_refined" \
    140 \
    0 -11.25 11.25 -22.5 22.5 -30 30

run_collection \
    "corner_lb_refined" \
    "corner_lb" \
    "${DATA_ROOT}/collect_corner_lb_refined" \
    280 \
    0 -11.25 11.25 -22.5 22.5 -30 30

run_collection \
    "corner_rt_refined" \
    "corner_rt" \
    "${DATA_ROOT}/collect_corner_rt_refined" \
    280 \
    0 -11.25 11.25 -22.5 22.5 -30 30

printf '\n[N2M-BATCH] All remaining refined collections completed at %s\n' \
    "$(date --iso-8601=seconds)" | tee -a "${LOG_DIR}/batch_complete.log"
printf '[N2M-BATCH] Logs: %s\n' "${LOG_DIR}" | tee -a "${LOG_DIR}/batch_complete.log"
