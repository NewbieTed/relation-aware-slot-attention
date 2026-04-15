#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-sd15}"
DEVICE="${DEVICE:-mps}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-1}"
START_INDEX="${START_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$ROOT_DIR/.venv-t2i310sys/bin/python}"
PROMPTS_FILE="${PROMPTS_FILE:-evaluation/benchmarks/t2i_compbench/3d_spatial_val.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/${MODEL}_t2i_compbench_3d_spatial_val}"

"$PYTHON_BIN" -m evaluation.generate \
  --model "$MODEL" \
  --prompts-file "$PROMPTS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --num-images-per-prompt "$NUM_IMAGES_PER_PROMPT" \
  --start-index "$START_INDEX" \
  --device "$DEVICE"

"$PYTHON_BIN" -m evaluation.t2i_compbench \
  --benchmark 3d_spatial \
  --generated-dir "$OUTPUT_DIR" \
  --prompt-file "$PROMPTS_FILE" \
  --python-bin "$EVAL_PYTHON_BIN"
