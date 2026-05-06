#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/.venv-flux/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/.venv-flux/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
CONFIG_FILE="${CONFIG_FILE:-}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/scop_depth_crops_depth}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/cache/flux_training_cache}"
INIT_GRAPH_ENCODER="${INIT_GRAPH_ENCODER:-$ROOT_DIR/outputs/train/graph_pretrain_flux_3dbox/final/graph_encoder.pt}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
DEVICE="${DEVICE:-auto}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CACHE_DTYPE="${CACHE_DTYPE:-bf16}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
OSCR_SIZE="${OSCR_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-512}"
CONDITION_RENDERER="${CONDITION_RENDERER:-seethrough}"
OSCR_FACE_ALPHA="${OSCR_FACE_ALPHA:-0.10}"
OSCR_AZIMUTH_DEGREES="${OSCR_AZIMUTH_DEGREES:-0}"
PROMPT_PREFIX="${PROMPT_PREFIX:-a photo of}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
BLENDER_CACHE_DIR="${BLENDER_CACHE_DIR:-}"
OVERWRITE="${OVERWRITE:-0}"

CMD=("$PYTHON_BIN" -m training.precompute_flux_training_cache)

if [[ -n "$CONFIG_FILE" ]]; then
  exec "${CMD[@]}" --config "$CONFIG_FILE"
else
  CMD+=(
    --dataset-dir "$DATASET_DIR" \
    --cache-dir "$CACHE_DIR" \
    --init-graph-encoder "$INIT_GRAPH_ENCODER" \
    --model-id "$MODEL_ID" \
    --device "$DEVICE" \
    --mixed-precision "$MIXED_PRECISION" \
    --cache-dtype "$CACHE_DTYPE" \
    --image-size "$IMAGE_SIZE" \
    --oscr-size "$OSCR_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
    --prompt-prefix "$PROMPT_PREFIX" \
    --condition-renderer "$CONDITION_RENDERER" \
    --oscr-face-alpha "$OSCR_FACE_ALPHA" \
    --oscr-azimuth-degrees "$OSCR_AZIMUTH_DEGREES" \
    --blender-bin "$BLENDER_BIN"
  )
fi

if [[ -z "$CONFIG_FILE" && -n "$BLENDER_CACHE_DIR" ]]; then
  CMD+=(--blender-cache-dir "$BLENDER_CACHE_DIR")
fi
if [[ -z "$CONFIG_FILE" && "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

"${CMD[@]}" "$@"
