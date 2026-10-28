#!/usr/bin/env bash

# Collect and process the complete pre-training FloAff dataset in sequence:
#   1. manipulation outcomes M
#   2. front-camera observations O
#   3. placement-level splits
#   4. dense ground-query affordance labels
#   5. fixed-size AnchorAxisNet samples
#   6. final completeness validation
#
# MoveIt is intentionally kept as a separate service.  Start
# custom_envs/launch/moveit_mujoco.launch.py before running this script.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG="${REPO_ROOT}/custom_envs/scripts/FloAff/configs/data_generation.yaml"
GRASP_CHECKPOINT="/home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"
OUTPUT_DIR=""
PYTHON_BIN="${FLOAFF_PYTHON:-python}"
SAVE_PREVIEWS=1
INHIBIT_SLEEP=1

usage() {
    cat <<'EOF'
Usage: collect_dataset.sh [options]

Collect all M samples first, automatically collect all O samples, and then
build splits, dense affordance fields, and the AnchorAxisNet dataset. Both
expensive collection stages use --resume, so the script can be restarted.

Options:
  --config PATH              Dataset-generation YAML.
  --grasp-checkpoint PATH    AnyGrasp checkpoint.
  --output-dir PATH          Override the YAML output directory.
  --python PATH              Python executable (default: current `python`).
  --no-previews              Do not save O mask preview PNG files.
  --allow-sleep              Do not use systemd-inhibit.
  -h, --help                 Show this help message.

Before running, start MoveIt in another terminal:
  source /opt/ros/humble/setup.bash
  cd ~/taskdog
  ros2 launch custom_envs/launch/moveit_mujoco.launch.py
EOF
}

while (($#)); do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --grasp-checkpoint)
            GRASP_CHECKPOINT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --no-previews)
            SAVE_PREVIEWS=0
            shift
            ;;
        --allow-sleep)
            INHIBIT_SLEEP=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[FloAff][ERROR] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# Re-run this same script under a sleep inhibitor.  Screen blanking and locking
# remain allowed; only idle sleep/suspend is blocked while collection runs.
if ((INHIBIT_SLEEP)) \
    && [[ "${FLOAFF_SLEEP_INHIBITED:-0}" != "1" ]] \
    && command -v systemd-inhibit >/dev/null 2>&1; then
    relaunch=(
        "$SCRIPT_DIR/collect_dataset.sh"
        --config "$CONFIG"
        --grasp-checkpoint "$GRASP_CHECKPOINT"
        --python "$PYTHON_BIN"
    )
    if [[ -n "$OUTPUT_DIR" ]]; then
        relaunch+=(--output-dir "$OUTPUT_DIR")
    fi
    if ((!SAVE_PREVIEWS)); then
        relaunch+=(--no-previews)
    fi
    exec env FLOAFF_SLEEP_INHIBITED=1 systemd-inhibit \
        --what=sleep:idle \
        --mode=block \
        --why="FloAff M and O dataset collection" \
        "${relaunch[@]}"
fi

cd "$REPO_ROOT"

if [[ ! -f "$CONFIG" ]]; then
    echo "[FloAff][ERROR] config not found: $CONFIG" >&2
    exit 2
fi
if [[ ! -f "$GRASP_CHECKPOINT" ]]; then
    echo "[FloAff][ERROR] AnyGrasp checkpoint not found: $GRASP_CHECKPOINT" >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c 'import mujoco' >/dev/null 2>&1; then
    echo "[FloAff][ERROR] '$PYTHON_BIN' cannot import mujoco." >&2
    echo "Activate env_isaaclab first, or pass --python /path/to/env_isaaclab/bin/python." >&2
    exit 2
fi
if ! command -v ros2 >/dev/null 2>&1; then
    echo "[FloAff][ERROR] ros2 is unavailable; source /opt/ros/humble/setup.bash first." >&2
    exit 2
fi

# An absent MoveIt server would otherwise turn infrastructure errors into false
# deterministic M failures.  Refuse to start until every required service is up.
moveit_services="$(timeout 10s ros2 service list 2>/dev/null || true)"
required_services=(
    /compute_ik
    /plan_kinematic_path
    /compute_cartesian_path
    /apply_planning_scene
)
missing_services=()
for service in "${required_services[@]}"; do
    if [[ $'\n'"$moveit_services"$'\n' != *$'\n'"$service"$'\n'* ]]; then
        missing_services+=("$service")
    fi
done
if ((${#missing_services[@]})); then
    echo "[FloAff][ERROR] MoveIt is not ready; missing services: ${missing_services[*]}" >&2
    echo "Start it in another terminal with:" >&2
    echo "  ros2 launch custom_envs/launch/moveit_mujoco.launch.py" >&2
    exit 2
fi

common_args=(--config "$CONFIG")
if [[ -n "$OUTPUT_DIR" ]]; then
    common_args+=(--output-dir "$OUTPUT_DIR")
fi

echo "[FloAff] Python: $PYTHON_BIN"
echo "[FloAff] Config: $CONFIG"
echo "[FloAff] ensuring placement and instance manifests exist..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.generate_instances \
    "${common_args[@]}" \
    --resume

echo "[FloAff] stage 1/7: collecting all manipulation labels M..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.collect_manipulation_labels \
    "${common_args[@]}" \
    --grasp-checkpoint "$GRASP_CHECKPOINT" \
    --resume

echo "[FloAff] M collection completed successfully."
echo "[FloAff] stage 2/7: collecting all observations O..."
observation_args=("${common_args[@]}" --resume)
if ((SAVE_PREVIEWS)); then
    observation_args+=(--save-previews)
fi
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.capture_observations \
    "${observation_args[@]}"

echo "[FloAff] stage 3/7: building placement-level dataset splits..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.build_splits \
    "${common_args[@]}"

echo "[FloAff] stage 4/7: validating complete raw M/O data..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.validate_dataset \
    "${common_args[@]}" \
    --require-complete

echo "[FloAff] stage 5/7: building dense affordance fields..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.affordance_labels \
    "${common_args[@]}"

echo "[FloAff] stage 6/7: building fixed-size AnchorAxisNet samples..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.build_axis_dataset \
    "${common_args[@]}" \
    --overwrite

echo "[FloAff] stage 7/7: writing the final validation report..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.validate_dataset \
    "${common_args[@]}" \
    --require-complete

echo "[FloAff] complete: M, O, splits, affordance fields, and AnchorAxisNet samples are ready."
echo "[FloAff] The final FloAff samples are built later from frozen AnchorAxisNet predictions."
