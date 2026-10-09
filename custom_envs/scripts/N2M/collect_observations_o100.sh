#!/usr/bin/env bash

# Extend the completed d30/O50 dataset to 100 valid observations per approach
# edge.  The O50 dataset is copied once and kept immutable as an experiment
# baseline; subsequent launches resume the O100 destination in place.

set -Eeuo pipefail

readonly PROJECT_ROOT="/home/mojie/taskdog"
readonly PYTHON_BIN="/home/mojie/anaconda3/envs/env_isaaclab/bin/python"
readonly N2M_DIR="${PROJECT_ROOT}/custom_envs/scripts/N2M"
readonly CAPTURE="${N2M_DIR}/data_generation/capture_observations.py"
readonly BUILD_PAIRS="${N2M_DIR}/data_generation/build_pairs.py"
readonly VALIDATE="${N2M_DIR}/data_generation/validate_dataset.py"
readonly SUCCESS_MANIFEST="${N2M_DIR}/data/manipulation_pose_dataset/d30_v1/successful_poses.jsonl"
readonly SOURCE_DATASET_DIR="${N2M_DIR}/data/n2m_dataset/d30_o50_v1"
readonly DATASET_DIR="${N2M_DIR}/data/n2m_dataset/d30_o100_v1"
readonly RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_DIR="${N2M_DIR}/data/overnight_logs/observations_o100_${RUN_STAMP}"
readonly FINAL_TARGET=100

# The original grid has too few unevaluated candidates for long_top/top to
# reach 100.  These supersets retain every old candidate and insert midpoint
# samples, so all existing IDs remain reusable while the new half is diverse.
readonly RANGES="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0"
readonly TANGENT_OFFSETS="-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6"
readonly YAW_OFFSETS="-20,-15,-10,-5,0,5,10,15,20"

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

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

exec 9>/tmp/taskdog_n2m_o100_collection.lock
if ! flock -n 9; then
    printf '[N2M-O100] Another O100 collection batch is already running.\n' >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf '[N2M-O100] Python environment not found: %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${SUCCESS_MANIFEST}" ]]; then
    printf '[N2M-O100] Successful-M manifest not found: %s\n' "${SUCCESS_MANIFEST}" >&2
    exit 1
fi
if [[ ! -f "${SOURCE_DATASET_DIR}/observations.jsonl" ]]; then
    printf '[N2M-O100] Completed O50 source dataset not found: %s\n' \
        "${SOURCE_DATASET_DIR}" >&2
    exit 1
fi

if [[ ! -e "${DATASET_DIR}" ]]; then
    printf '[N2M-O100] Seeding O100 from the completed O50 dataset.\n'
    cp -a "${SOURCE_DATASET_DIR}" "${DATASET_DIR}"
elif [[ ! -f "${DATASET_DIR}/capture_config.json" || ! -f "${DATASET_DIR}/observations.jsonl" ]]; then
    printf '[N2M-O100] Destination exists but is not a resumable dataset: %s\n' \
        "${DATASET_DIR}" >&2
    exit 1
else
    printf '[N2M-O100] Resuming existing destination: %s\n' "${DATASET_DIR}"
fi

on_error() {
    local status=$?
    printf '\n[N2M-O100] FAILED at %s (line %s, exit=%s).\n' \
        "$(date --iso-8601=seconds)" "${BASH_LINENO[0]:-unknown}" "${status}" >&2
    printf '[N2M-O100] Re-run the same launcher to resume. Logs: %s\n' \
        "${LOG_DIR}" >&2
    exit "${status}"
}
trap on_error ERR

run_capture() {
    local scene_id="$1"
    local stage_name="${scene_id}_o${FINAL_TARGET}"
    local log_path="${LOG_DIR}/${stage_name}.log"
    local attempt

    printf '\n[N2M-O100] Starting %s at %s\n' \
        "${stage_name}" "$(date --iso-8601=seconds)" | tee -a "${log_path}"

    for attempt in 1 2 3; do
        printf '[N2M-O100] launch attempt %d/3, dataset=%s\n' \
            "${attempt}" "${DATASET_DIR}" | tee -a "${log_path}"
        if "${PYTHON_BIN}" "${CAPTURE}" \
            --cube-scene "${scene_id}" \
            --target-valid-per-edge "${FINAL_TARGET}" \
            --ranges "${RANGES}" \
            --tangent-offsets="${TANGENT_OFFSETS}" \
            --yaw-offsets-deg="${YAW_OFFSETS}" \
            --allow-proposal-grid-extension \
            --save-rejected-previews \
            --max-rejected-previews-per-reason 3 \
            --resume \
            --output-dir "${DATASET_DIR}" \
            2>&1 | tee -a "${log_path}"; then
            printf '[N2M-O100] Completed %s at %s\n' \
                "${stage_name}" "$(date --iso-8601=seconds)" | tee -a "${log_path}"
            return 0
        fi
        if ((attempt < 3)); then
            printf '[N2M-O100] %s failed; retrying from manifests in 5 seconds.\n' \
                "${stage_name}" | tee -a "${log_path}" >&2
            sleep 5
        fi
    done

    printf '[N2M-O100] %s failed after 3 attempts.\n' "${stage_name}" \
        | tee -a "${log_path}" >&2
    return 1
}

printf '[N2M-O100] Batch started at %s\n' "$(date --iso-8601=seconds)"
printf '[N2M-O100] Source baseline: %s\n' "${SOURCE_DATASET_DIR}"
printf '[N2M-O100] Final dataset: %s\n' "${DATASET_DIR}"
printf '[N2M-O100] Strategy: reuse 50 O/edge and collect to %d O/edge total.\n' \
    "${FINAL_TARGET}"

for scene_id in "${SCENES[@]}"; do
    run_capture "${scene_id}"
done

# Refresh the summary with all eight edge groups.
run_capture all

printf '\n[N2M-O100] Rebuilding all observation/manipulation pairs.\n'
"${PYTHON_BIN}" "${BUILD_PAIRS}" \
    --dataset-dir "${DATASET_DIR}" \
    --success-manifest "${SUCCESS_MANIFEST}" \
    2>&1 | tee -a "${LOG_DIR}/build_pairs.log"

printf '\n[N2M-O100] Validating assembled dataset.\n'
"${PYTHON_BIN}" "${VALIDATE}" \
    --dataset-dir "${DATASET_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/validate_dataset.log"

observation_count="$(wc -l < "${DATASET_DIR}/observations.jsonl")"
pair_count="$(wc -l < "${DATASET_DIR}/pairs.jsonl")"
if [[ "${observation_count}" -ne 800 ]]; then
    printf '[N2M-O100] Expected 800 observations, found %s.\n' \
        "${observation_count}" >&2
    exit 1
fi
if [[ "${pair_count}" -ne 29100 ]]; then
    printf '[N2M-O100] Expected 29100 pairs, found %s.\n' "${pair_count}" >&2
    exit 1
fi

{
    printf '[N2M-O100] COMPLETE at %s\n' "$(date --iso-8601=seconds)"
    printf '[N2M-O100] observations=%s pairs=%s\n' \
        "${observation_count}" "${pair_count}"
    printf '[N2M-O100] dataset=%s\n' "${DATASET_DIR}"
    printf '[N2M-O100] logs=%s\n' "${LOG_DIR}"
} | tee "${LOG_DIR}/batch_complete.log"
