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
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/sd15_relation_aware}"
DEVICE="${DEVICE:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAPH_LEARNING_RATE="${GRAPH_LEARNING_RATE:-5e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-8}"
SLOT_DIM="${SLOT_DIM:-512}"
GNN_LAYERS="${GNN_LAYERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-10}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
VALIDATION_EVERY="${VALIDATION_EVERY:-0}"
EVAL_EVERY="${EVAL_EVERY:-250}"
NUM_VALIDATION_IMAGES="${NUM_VALIDATION_IMAGES:-4}"
VALIDATION_PROMPTS_FILE="${VALIDATION_PROMPTS_FILE:-}"
PROMPT_PREFIX="${PROMPT_PREFIX:-a photo of}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.1}"
RELATION_LOSS_WEIGHT="${RELATION_LOSS_WEIGHT:-0.1}"
EMBEDDING_LOSS_WEIGHT="${EMBEDDING_LOSS_WEIGHT:-0.05}"
ATTENTION_TOKEN_LOSS_WEIGHT="${ATTENTION_TOKEN_LOSS_WEIGHT:-0.0}"
ATTENTION_PIXEL_LOSS_WEIGHT="${ATTENTION_PIXEL_LOSS_WEIGHT:-0.0}"
SLOT_USAGE_LOSS_WEIGHT="${SLOT_USAGE_LOSS_WEIGHT:-0.0}"
SLOT_USAGE_TARGET="${SLOT_USAGE_TARGET:-0.02}"
OBJECT_TOKEN_DROPOUT_PROB="${OBJECT_TOKEN_DROPOUT_PROB:-0.0}"
INIT_GRAPH_ENCODER="${INIT_GRAPH_ENCODER:-}"
FREEZE_GRAPH_ENCODER="${FREEZE_GRAPH_ENCODER:-0}"
FULL_UNET_FINETUNE="${FULL_UNET_FINETUNE:-0}"

CMD=(
  "$PYTHON_BIN" -m training.train_relation_aware_sd15
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --mixed-precision "$MIXED_PRECISION"
  --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --graph-learning-rate "$GRAPH_LEARNING_RATE"
  --max-train-steps "$MAX_TRAIN_STEPS"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --lora-rank "$LORA_RANK"
  --slot-dim "$SLOT_DIM"
  --gnn-layers "$GNN_LAYERS"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --prompt-prefix "$PROMPT_PREFIX"
  --num-validation-images "$NUM_VALIDATION_IMAGES"
  --validation-every "$VALIDATION_EVERY"
  --eval-every "$EVAL_EVERY"
  --eval-fraction "$EVAL_FRACTION"
  --test-fraction "$TEST_FRACTION"
  --aux-loss-weight "$AUX_LOSS_WEIGHT"
  --relation-loss-weight "$RELATION_LOSS_WEIGHT"
  --embedding-loss-weight "$EMBEDDING_LOSS_WEIGHT"
  --attention-token-loss-weight "$ATTENTION_TOKEN_LOSS_WEIGHT"
  --attention-pixel-loss-weight "$ATTENTION_PIXEL_LOSS_WEIGHT"
  --slot-usage-loss-weight "$SLOT_USAGE_LOSS_WEIGHT"
  --slot-usage-target "$SLOT_USAGE_TARGET"
  --object-token-dropout-prob "$OBJECT_TOKEN_DROPOUT_PROB"
)

if [[ -n "$LIMIT_ROWS" ]]; then
  CMD+=(--limit-rows "$LIMIT_ROWS")
fi

if [[ -n "$VALIDATION_PROMPTS_FILE" ]]; then
  CMD+=(--validation-prompts-file "$VALIDATION_PROMPTS_FILE")
fi

if [[ -n "$INIT_GRAPH_ENCODER" ]]; then
  CMD+=(--init-graph-encoder "$INIT_GRAPH_ENCODER")
fi

if [[ "$FREEZE_GRAPH_ENCODER" == "1" ]]; then
  CMD+=(--freeze-graph-encoder)
fi

if [[ "$FULL_UNET_FINETUNE" == "1" ]]; then
  CMD+=(--full-unet-finetune)
fi

"${CMD[@]}" "$@"
