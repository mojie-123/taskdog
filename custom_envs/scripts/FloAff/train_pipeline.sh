#!/usr/bin/env bash

# Train AnchorAxisNet, freeze and cache its predictions, build predicted-axis
# FloAff samples, then train FloAff.  Run only after collect_dataset.sh has
# completed all seven stages.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${FLOAFF_PYTHON:-python}"
DATA_CONFIG="${SCRIPT_DIR}/configs/data_generation.yaml"
AXIS_CONFIG="${SCRIPT_DIR}/configs/axis_training.yaml"
FLOAFF_CONFIG="${SCRIPT_DIR}/configs/floaff_training.yaml"

usage() {
    cat <<'EOF'
Usage: train_pipeline.sh [options]

Options:
  --python PATH          Python executable (default: current `python`).
  --data-config PATH     Raw data-generation configuration.
  --axis-config PATH     AnchorAxisNet training configuration.
  --floaff-config PATH   FloAff training configuration.
  -h, --help             Show this message.
EOF
}

while (($#)); do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --data-config)
            DATA_CONFIG="$2"
            shift 2
            ;;
        --axis-config)
            AXIS_CONFIG="$2"
            shift 2
            ;;
        --floaff-config)
            FLOAFF_CONFIG="$2"
            shift 2
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

cd "$REPO_ROOT"

for path in "$DATA_CONFIG" "$AXIS_CONFIG" "$FLOAFF_CONFIG"; do
    if [[ ! -f "$path" ]]; then
        echo "[FloAff][ERROR] configuration not found: $path" >&2
        exit 2
    fi
done
if ! "$PYTHON_BIN" -c 'import torch, yaml, numpy' >/dev/null 2>&1; then
    echo "[FloAff][ERROR] training dependencies are unavailable in $PYTHON_BIN" >&2
    exit 2
fi

config_value() {
    "$PYTHON_BIN" -c \
        'import pathlib,sys,yaml; value=yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]; print(pathlib.Path(value).expanduser().resolve())' \
        "$1" "$2"
}

DATASET_ROOT="$(config_value "$AXIS_CONFIG" dataset_root)"
AXIS_RUN_ROOT="$(config_value "$AXIS_CONFIG" run_root)"
FLOAFF_RUN_ROOT="$(config_value "$FLOAFF_CONFIG" run_root)"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
AXIS_RUN="${AXIS_RUN_ROOT}/${TIMESTAMP}"
FLOAFF_RUN="${FLOAFF_RUN_ROOT}/${TIMESTAMP}"
AXIS_PREDICTIONS="${DATASET_ROOT}/axis_predictions/${TIMESTAMP}.jsonl"

echo "[FloAff] stage 1/6: validating the completed collection..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.validate_dataset \
    --config "$DATA_CONFIG" \
    --require-complete

echo "[FloAff] stage 2/6: training AnchorAxisNet..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.training.train_axis \
    --config "$AXIS_CONFIG" \
    --run-dir "$AXIS_RUN"

echo "[FloAff] stage 3/6: caching frozen AnchorAxisNet predictions..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.training.predict_axis \
    --config "$AXIS_CONFIG" \
    --checkpoint "${AXIS_RUN}/best.pt" \
    --role axis_eval_floaff_train \
    --role transfer_test \
    --output "$AXIS_PREDICTIONS"

echo "[FloAff] stage 4/6: building predicted-axis FloAff samples..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.data_generation.build_floaff_dataset \
    --config "$DATA_CONFIG" \
    --axis-predictions "$AXIS_PREDICTIONS" \
    --role axis_eval_floaff_train \
    --role transfer_test \
    --overwrite

echo "[FloAff] stage 5/6: training FloAff..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.training.train_floaff \
    --config "$FLOAFF_CONFIG" \
    --run-dir "$FLOAFF_RUN"

echo "[FloAff] stage 6/6: evaluating the selected FloAff checkpoint..."
"$PYTHON_BIN" -m custom_envs.scripts.FloAff.training.evaluate_floaff \
    --config "$FLOAFF_CONFIG" \
    --checkpoint "${FLOAFF_RUN}/best_loss.pt" \
    --split validation \
    --save-heatmaps

echo "[FloAff] two-stage training complete."
echo "[FloAff] AnchorAxisNet: ${AXIS_RUN}/best.pt"
echo "[FloAff] Axis predictions: ${AXIS_PREDICTIONS}"
echo "[FloAff] FloAff: ${FLOAFF_RUN}/best_loss.pt"
