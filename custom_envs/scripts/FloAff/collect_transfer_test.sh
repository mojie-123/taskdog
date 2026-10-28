#!/usr/bin/env bash

# Incrementally append and collect the held-out transfer_test split without
# modifying or recollecting the existing axis_train/FloAff-training examples.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG="${SCRIPT_DIR}/configs/data_generation.yaml"
GRASP_CHECKPOINT="/home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar"
OUTPUT_DIR=""
PYTHON_BIN="${FLOAFF_PYTHON:-python}"
SAVE_PREVIEWS=1
INHIBIT_SLEEP=1

usage() {
    cat <<'EOF'
Usage: collect_transfer_test.sh [options]

Append and collect only the held-out transfer_test regions. The default plan is
4 placements on the other short edge, 3 placements on each of three unseen
corners, and 4 placements on each long edge. Transfer instances use 40 M and
20 O; existing training instances retain 100 M and 40 O.

Options:
  --config PATH              Dataset-generation YAML.
  --grasp-checkpoint PATH    AnyGrasp checkpoint.
  --output-dir PATH          Override the YAML output directory.
  --python PATH              Python executable (default: current `python`).
  --no-previews              Do not save O semantic preview PNG files.
  --allow-sleep              Do not use systemd-inhibit.
  -h, --help                 Show this message.

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

if ((INHIBIT_SLEEP)) \
    && [[ "${FLOAFF_TRANSFER_SLEEP_INHIBITED:-0}" != "1" ]] \
    && command -v systemd-inhibit >/dev/null 2>&1; then
    relaunch=(
        "$SCRIPT_DIR/collect_transfer_test.sh"
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
    exec env FLOAFF_TRANSFER_SLEEP_INHIBITED=1 systemd-inhibit \
        --what=sleep:idle \
        --mode=block \
        --why="FloAff transfer_test M and O collection" \
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
if ! "$PYTHON_BIN" -c 'import mujoco, yaml' >/dev/null 2>&1; then
    echo "[FloAff][ERROR] '$PYTHON_BIN' cannot import the required packages." >&2
    echo "Activate env_isaaclab or pass --python /path/to/env_isaaclab/bin/python." >&2
    exit 2
fi
if ! command -v ros2 >/dev/null 2>&1; then
    echo "[FloAff][ERROR] ros2 is unavailable; source /opt/ros/humble/setup.bash first." >&2
    exit 2
fi

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
    echo "Start ros2 launch custom_envs/launch/moveit_mujoco.launch.py first." >&2
    exit 2
fi

common_args=(--config "$CONFIG")
if [[ -n "$OUTPUT_DIR" ]]; then
    common_args+=(--output-dir "$OUTPUT_DIR")
fi

echo "[FloAff][transfer] Python: $PYTHON_BIN"
echo "[FloAff][transfer] Config: $CONFIG"
echo "[FloAff][transfer] extending manifests without changing existing rows..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.generate_instances \
    "${common_args[@]}" \
    --extend-transfer

if [[ -n "$OUTPUT_DIR" ]]; then
    DATA_ROOT="$(realpath -m "$OUTPUT_DIR")"
else
    DATA_ROOT="$("$PYTHON_BIN" -c \
        'from custom_envs.scripts.FloAff.config import load_data_generation_config; import pathlib,sys; print(pathlib.Path(load_data_generation_config(sys.argv[1]).output_dir).resolve())' \
        "$CONFIG")"
fi

mapfile -t transfer_instances < <(
    "$PYTHON_BIN" -c \
        'import json,sys; [print(r["instance_id"]) for line in open(sys.argv[1]) if (r:=json.loads(line))["placement_id"].startswith("transfer_")]' \
        "$DATA_ROOT/instances.jsonl"
)
if ((${#transfer_instances[@]} == 0)); then
    echo "[FloAff][ERROR] no transfer instances were generated" >&2
    exit 2
fi
instance_args=()
for instance_id in "${transfer_instances[@]}"; do
    instance_args+=(--instance "$instance_id")
done
echo "[FloAff][transfer] selected ${#transfer_instances[@]} transfer instances."

echo "[FloAff][transfer] stage 1/7: collecting 40 M per transfer instance..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.collect_manipulation_labels \
    "${common_args[@]}" \
    "${instance_args[@]}" \
    --grasp-checkpoint "$GRASP_CHECKPOINT" \
    --resume

echo "[FloAff][transfer] stage 2/7: collecting 20 O per transfer instance..."
observation_args=("${common_args[@]}" "${instance_args[@]}" --resume)
if ((SAVE_PREVIEWS)); then
    observation_args+=(--save-previews)
fi
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.capture_observations \
    "${observation_args[@]}"

echo "[FloAff][transfer] stage 3/7: rebuilding placement-level splits..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.build_splits \
    "${common_args[@]}"

echo "[FloAff][transfer] stage 4/7: validating all raw M/O data..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.validate_dataset \
    "${common_args[@]}" \
    --require-complete

echo "[FloAff][transfer] stage 5/7: rebuilding affordance fields..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.affordance_labels \
    "${common_args[@]}"

echo "[FloAff][transfer] stage 6/7: rebuilding all AnchorAxisNet samples..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.build_axis_dataset \
    "${common_args[@]}" \
    --overwrite

echo "[FloAff][transfer] stage 7/7: writing final validation report..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.validate_dataset \
    "${common_args[@]}" \
    --require-complete

echo "[FloAff][transfer] complete. Existing training data were retained."
echo "[FloAff][transfer] Next: predict transfer axes with the frozen AnchorAxisNet,"
echo "[FloAff][transfer] then rebuild FloAff samples for both training and transfer roles."
