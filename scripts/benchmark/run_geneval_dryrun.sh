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
LIMIT_PROMPTS="${LIMIT_PROMPTS:-20}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-4}"
START_INDEX="${START_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
GENEVAL_ROOT="${GENEVAL_ROOT:-$ROOT_DIR/external/geneval}"
PROMPTS_FILE="${PROMPTS_FILE:-$GENEVAL_ROOT/prompts/generation_prompts.txt}"
METADATA_FILE="${METADATA_FILE:-$GENEVAL_ROOT/prompts/evaluation_metadata.jsonl}"
MODEL_PATH="${MODEL_PATH:-$GENEVAL_ROOT/models}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/${MODEL}_geneval_dryrun}"

"$PYTHON_BIN" -m evaluation.generate \
  --model "$MODEL" \
  --prompts-file "$PROMPTS_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --num-images-per-prompt "$NUM_IMAGES_PER_PROMPT" \
  --limit-prompts "$LIMIT_PROMPTS" \
  --start-index "$START_INDEX" \
  --device "$DEVICE"

"$PYTHON_BIN" -m evaluation.geneval \
  --geneval-root "$GENEVAL_ROOT" \
  --generated-dir "$OUTPUT_DIR" \
  --metadata-file "$METADATA_FILE" \
  --model-path "$MODEL_PATH" \
  --python-bin "$PYTHON_BIN"
