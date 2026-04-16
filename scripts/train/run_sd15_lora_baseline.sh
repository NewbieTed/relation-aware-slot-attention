#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/scop_depth_full}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/sd15_scopdepth_lora}"
DEVICE="${DEVICE:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-8}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-10}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
VALIDATION_EVERY="${VALIDATION_EVERY:-0}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-4}"
VALIDATION_PROMPTS_FILE="${VALIDATION_PROMPTS_FILE:-}"
PROMPT_PREFIX="${PROMPT_PREFIX:-a photo of}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

CMD=(
  "$PYTHON_BIN" -m training.train_sd15_lora
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --mixed-precision "$MIXED_PRECISION"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --max-train-steps "$MAX_TRAIN_STEPS"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --lora-rank "$LORA_RANK"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --prompt-prefix "$PROMPT_PREFIX"
  --num-validation-images "$NUM_VALIDATION_IMAGES"
  --validation-every "$VALIDATION_EVERY"
)

if [[ -n "$LIMIT_ROWS" ]]; then
  CMD+=(--limit-rows "$LIMIT_ROWS")
fi

if [[ -n "$VALIDATION_PROMPTS_FILE" ]]; then
  CMD+=(--validation-prompts-file "$VALIDATION_PROMPTS_FILE")
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  CMD+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${CMD[@]}" "$@"
