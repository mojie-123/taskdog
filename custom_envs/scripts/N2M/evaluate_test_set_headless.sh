#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mojie/taskdog
PYTHON_BIN=${PYTHON_BIN:-/home/mojie/anaconda3/envs/env_isaaclab/bin/python}
OUTPUT=${N2M_EVAL_OUTPUT:-$ROOT/custom_envs/scripts/N2M/data/evaluations/d30_o100_k1_test_m3_v1}

mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/.evaluation.lock"
if ! flock -n 9; then
  echo "[N2M][EVAL] another evaluator already owns $OUTPUT" >&2
  exit 2
fi

cd "$ROOT"
exec "$PYTHON_BIN" custom_envs/scripts/N2M/evaluate_test_set.py \
  --checkpoint custom_envs/scripts/N2M/data/training/d30_o100_k1_v1/best.pt \
  --dataset-dir custom_envs/scripts/N2M/data/n2m_dataset/d30_o100_v1 \
  --output-dir "$OUTPUT" \
  --grasp-checkpoint /home/mojie/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar \
  --split test \
  --num-samples 100 \
  --max-m-candidates 3 \
  --attempts-per-m 3 \
  --device cuda \
  --resume \
  "$@"
