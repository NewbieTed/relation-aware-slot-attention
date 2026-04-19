#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUNS="${RUNS:-5}"
BASE_SEED="${BASE_SEED:-42}"
SEED_STEP="${SEED_STEP:-100}"

CATEGORY="${CATEGORY:-spatial}"
MODEL="${MODEL:-sd15}"
DEVICE="${DEVICE:-auto}"
LORA_PATH="${LORA_PATH:-}"
RELATION_AWARE_DIR="${RELATION_AWARE_DIR:-}"
GRAPH_ENCODER_PATH="${GRAPH_ENCODER_PATH:-}"
RELATION_ATTENTION_PATH="${RELATION_ATTENTION_PATH:-}"
NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-5}"
START_INDEX="${START_INDEX:-0}"
PRUNE_SAMPLES_KEEP="${PRUNE_SAMPLES_KEEP:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$PYTHON_BIN}"
T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"
PROMPTS_FILE="${PROMPTS_FILE:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/eval/${MODEL}_t2i_compbench_${CATEGORY}_repeats}"

mkdir -p "$OUTPUT_ROOT"

for (( run_idx=0; run_idx<RUNS; run_idx++ )); do
  seed=$((BASE_SEED + run_idx * SEED_STEP))
  run_name=$(printf "run_%03d" "$run_idx")
  run_output_dir="$OUTPUT_ROOT/$run_name"

  echo "=== T2I-CompBench repeat $run_idx / $((RUNS-1)) | seed=$seed | output=$run_output_dir ==="

  env \
    CATEGORY="$CATEGORY" \
    MODEL="$MODEL" \
    DEVICE="$DEVICE" \
    LORA_PATH="$LORA_PATH" \
    RELATION_AWARE_DIR="$RELATION_AWARE_DIR" \
    GRAPH_ENCODER_PATH="$GRAPH_ENCODER_PATH" \
    RELATION_ATTENTION_PATH="$RELATION_ATTENTION_PATH" \
    NUM_IMAGES_PER_PROMPT="$NUM_IMAGES_PER_PROMPT" \
    START_INDEX="$START_INDEX" \
    PRUNE_SAMPLES_KEEP="$PRUNE_SAMPLES_KEEP" \
    PRUNE_SAMPLES_SEED="$seed" \
    SEED="$seed" \
    PYTHON_BIN="$PYTHON_BIN" \
    EVAL_PYTHON_BIN="$EVAL_PYTHON_BIN" \
    T2I_ROOT="$T2I_ROOT" \
    PROMPTS_FILE="$PROMPTS_FILE" \
    OUTPUT_DIR="$run_output_dir" \
    "$ROOT_DIR/scripts/benchmark/run_t2i_category_full.sh"
done

"$PYTHON_BIN" -m evaluation.t2i_compbench_repeats \
  --root-dir "$OUTPUT_ROOT" \
  --benchmark "$CATEGORY"
