#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

MODEL="${MODEL:-sd15}"
DEVICE="${DEVICE:-auto}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-1}"
START_INDEX="${START_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$PYTHON_BIN}"
T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"
PROMPTS_FILE="${PROMPTS_FILE:-$T2I_ROOT/examples/dataset/spatial_val.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/${MODEL}_t2i_compbench_spatial_val}"

"$PYTHON_BIN" -m evaluation.generate \
  --model "$MODEL" \
  --prompts-file "$PROMPTS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --num-images-per-prompt "$NUM_IMAGES_PER_PROMPT" \
  --start-index "$START_INDEX" \
  --device "$DEVICE"

"$PYTHON_BIN" -m evaluation.t2i_compbench \
  --benchmark spatial \
  --t2i-compbench-root "$T2I_ROOT" \
  --generated-dir "$OUTPUT_DIR" \
  --prompt-file "$PROMPTS_FILE" \
  --python-bin "$EVAL_PYTHON_BIN"
