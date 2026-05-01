#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/scop_depth_crops_depth}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/flux1dev_oscr_lora128}"
INIT_GRAPH_ENCODER="${INIT_GRAPH_ENCODER:-$ROOT_DIR/outputs/train/graph_pretrain_flux_3dbox/final/graph_encoder.pt}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
DEVICE="${DEVICE:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-24000}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"

"$PYTHON_BIN" -m training.train_relation_flux_lora \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --init-graph-encoder "$INIT_GRAPH_ENCODER" \
  --model-id "$MODEL_ID" \
  --device "$DEVICE" \
  --mixed-precision "$MIXED_PRECISION" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  "$@"
