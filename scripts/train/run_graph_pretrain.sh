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
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/graph_pretrain}"
DEVICE="${DEVICE:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAPH_LEARNING_RATE="${GRAPH_LEARNING_RATE:-5e-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
SLOT_DIM="${SLOT_DIM:-512}"
GNN_LAYERS="${GNN_LAYERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-250}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
PROMPT_PREFIX="${PROMPT_PREFIX:-a photo of}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"
TEST_FRACTION="${TEST_FRACTION:-0.1}"
POSITION_LOSS_WEIGHT="${POSITION_LOSS_WEIGHT:-1.0}"
RELATION_LOSS_WEIGHT="${RELATION_LOSS_WEIGHT:-1.0}"
EMBEDDING_LOSS_WEIGHT="${EMBEDDING_LOSS_WEIGHT:-0.25}"
INVERSE_RELATION_LOSS_WEIGHT="${INVERSE_RELATION_LOSS_WEIGHT:-0.0}"
BOX_LOSS_WEIGHT="${BOX_LOSS_WEIGHT:-0.0}"

CMD=(
  "$PYTHON_BIN" -m training.pretrain_graph_encoder
  --dataset-dir "$DATASET_DIR"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --mixed-precision "$MIXED_PRECISION"
  --batch-size "$BATCH_SIZE"
  --graph-learning-rate "$GRAPH_LEARNING_RATE"
  --max-train-steps "$MAX_TRAIN_STEPS"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --slot-dim "$SLOT_DIM"
  --gnn-layers "$GNN_LAYERS"
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --prompt-prefix "$PROMPT_PREFIX"
  --eval-fraction "$EVAL_FRACTION"
  --test-fraction "$TEST_FRACTION"
  --position-loss-weight "$POSITION_LOSS_WEIGHT"
  --relation-loss-weight "$RELATION_LOSS_WEIGHT"
  --embedding-loss-weight "$EMBEDDING_LOSS_WEIGHT"
  --inverse-relation-loss-weight "$INVERSE_RELATION_LOSS_WEIGHT"
  --box-loss-weight "$BOX_LOSS_WEIGHT"
)

if [[ -n "$LIMIT_ROWS" ]]; then
  CMD+=(--limit-rows "$LIMIT_ROWS")
fi

"${CMD[@]}" "$@"
