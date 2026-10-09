#!/usr/bin/env bash

# Collect the final d30 observation dataset sequentially and build all (O, M)
# pairs.  The script is restart-safe: every capture command uses --resume and
# the collector skips observations already accepted or rejected.

set -Eeuo pipefail

readonly PROJECT_ROOT="/home/mojie/taskdog"
readonly PYTHON_BIN="/home/mojie/anaconda3/envs/env_isaaclab/bin/python"
readonly N2M_DIR="${PROJECT_ROOT}/custom_envs/scripts/N2M"
readonly CAPTURE="${N2M_DIR}/data_generation/capture_observations.py"
readonly BUILD_PAIRS="${N2M_DIR}/data_generation/build_pairs.py"
readonly VALIDATE="${N2M_DIR}/data_generation/validate_dataset.py"
readonly SUCCESS_MANIFEST="${N2M_DIR}/data/manipulation_pose_dataset/d30_v1/successful_poses.jsonl"
readonly DATASET_DIR="${N2M_DIR}/data/n2m_dataset/d30_o50_v1"
readonly RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_DIR="${N2M_DIR}/data/overnight_logs/observations_o50_${RUN_STAMP}"
readonly PREVIEW_TARGET=10
readonly FINAL_TARGET=50

readonly -a SCENES=(
    short_left
    short_right
    long_bottom
    long_top
    corner_lb
    corner_rt
)

export MUJOCO_GL=egl
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/n2m-matplotlib
export XDG_CACHE_HOME=/tmp/n2m-cache
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${DATASET_DIR}"
cd "${PROJECT_ROOT}"

# Prevent two unattended observation collectors from writing the same files.
exec 9>/tmp/taskdog_n2m_o50_collection.lock
if ! flock -n 9; then
    printf '[N2M-O50] Another O50 collection batch is already running.\n' >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf '[N2M-O50] Python environment not found: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${SUCCESS_MANIFEST}" ]]; then
    printf '[N2M-O50] Successful-M manifest not found: %s\n' "${SUCCESS_MANIFEST}" >&2
    exit 1
fi

on_error() {
    local status=$?
    printf '\n[N2M-O50] FAILED at %s (line %s, exit=%s).\n' \
        "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" "${status}" >&2
    printf '[N2M-O50] Dataset is resumable. Re-run the same launcher after fixing the error.\n' >&2
    printf '[N2M-O50] Per-stage logs: %s\n' "${LOG_DIR}" >&2
    exit "${status}"
}
trap on_error ERR

run_capture() {
    local scene_id="$1"
    local target="$2"
    local save_accepted_previews="$3"
    local stage_name="${scene_id}_o${target}"
    local log_path="${LOG_DIR}/${stage_name}.log"
    local -a preview_args=()
    local attempt

    if [[ "${save_accepted_previews}" == "yes" ]]; then
        preview_args=(--save-previews)
        stage_name="${stage_name}_with_previews"
        log_path="${LOG_DIR}/${stage_name}.log"
    fi

    printf '\n[N2M-O50] Starting %s at %s\n' \
        "${stage_name}" "$(date --iso-8601=seconds)" | tee -a "${log_path}"

    for attempt in 1 2 3; do
        printf '[N2M-O50] launch attempt %d/3, dataset=%s\n' \
            "${attempt}" "${DATASET_DIR}" | tee -a "${log_path}"
        if "${PYTHON_BIN}" "${CAPTURE}" \
            --cube-scene "${scene_id}" \
            --target-valid-per-edge "${target}" \
            --save-rejected-previews \
            --max-rejected-previews-per-reason 3 \
            --resume \
            --output-dir "${DATASET_DIR}" \
            "${preview_args[@]}" 2>&1 | tee -a "${log_path}"; then
            printf '[N2M-O50] Completed %s at %s\n' \
                "${stage_name}" "$(date --iso-8601=seconds)" | tee -a "${log_path}"
            return 0
        fi
        if ((attempt < 3)); then
            printf '[N2M-O50] %s failed; retrying from manifests in 5 seconds.\n' \
                "${stage_name}" | tee -a "${log_path}" >&2
            sleep 5
        fi
    done

    printf '[N2M-O50] %s failed after 3 attempts.\n' "${stage_name}" \
        | tee -a "${log_path}" >&2
    return 1
}

printf '[N2M-O50] Batch started at %s\n' "$(date --iso-8601=seconds)"
printf '[N2M-O50] Final dataset: %s\n' "${DATASET_DIR}"
printf '[N2M-O50] Strategy: first %d valid O/edge with previews, then %d valid O/edge total.\n' \
    "${PREVIEW_TARGET}" "${FINAL_TARGET}"

for scene_id in "${SCENES[@]}"; do
    run_capture "${scene_id}" "${PREVIEW_TARGET}" yes
    run_capture "${scene_id}" "${FINAL_TARGET}" no
done

# Refresh capture_summary.json so it reports all eight approach edges rather
# than only the scene handled by the last collection command.
run_capture all "${FINAL_TARGET}" no

printf '\n[N2M-O50] Building all observation/manipulation pairs.\n'
"${PYTHON_BIN}" "${BUILD_PAIRS}" \
    --dataset-dir "${DATASET_DIR}" \
    --success-manifest "${SUCCESS_MANIFEST}" \
    2>&1 | tee -a "${LOG_DIR}/build_pairs.log"

printf '\n[N2M-O50] Validating assembled dataset.\n'
"${PYTHON_BIN}" "${VALIDATE}" \
    --dataset-dir "${DATASET_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/validate_dataset.log"

observation_count="$(wc -l < "${DATASET_DIR}/observations.jsonl")"
pair_count="$(wc -l < "${DATASET_DIR}/pairs.jsonl")"
if [[ "${observation_count}" -ne 400 ]]; then
    printf '[N2M-O50] Expected 400 observations, found %s.\n' \
        "${observation_count}" >&2
    exit 1
fi
if [[ "${pair_count}" -ne 14550 ]]; then
    printf '[N2M-O50] Expected 14550 pairs, found %s.\n' "${pair_count}" >&2
    exit 1
fi

{
    printf '[N2M-O50] COMPLETE at %s\n' "$(date --iso-8601=seconds)"
    printf '[N2M-O50] observations=%s pairs=%s\n' "${observation_count}" "${pair_count}"
    printf '[N2M-O50] dataset=%s\n' "${DATASET_DIR}"
    printf '[N2M-O50] logs=%s\n' "${LOG_DIR}"
} | tee "${LOG_DIR}/batch_complete.log"

