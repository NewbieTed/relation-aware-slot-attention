#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
DECODER_FILM_SCALE="${DECODER_FILM_SCALE:-1.0}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
SEED="${SEED:-42}"

LOCAL_ROOT="${LOCAL_ROOT:-/local1/fuyifei}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$LOCAL_ROOT/hf-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_ROOT/outputs/eval/smoke_flux_triple_cvae_3dsln_film_v2_clear_spatial_512_5p4i_film${DECODER_FILM_SCALE}}"
BLENDER_CACHE_DIR="${BLENDER_CACHE_DIR:-$LOCAL_ROOT/outputs/cache/blender_smoke_flux_triple_cvae_3dsln_film_v2_clear_spatial_512_5p4i_film${DECODER_FILM_SCALE}}"

PROMPT_FILE="${PROMPT_FILE:-$ROOT_DIR/configs/flux/eval/prompts/clear_spatial_smoke.txt}"
GRAPH_ENCODER_PATH="${GRAPH_ENCODER_PATH:-/local1/cchen025/outputs/train/graph_pretrain_flux_3dbox_triple_cvae_3dsln_film_v2/final/graph_encoder.pt}"
BLENDER_BIN="${BLENDER_BIN:-/local1/cse_481_m_l/blender/blender-3.6.19-linux-x64/blender}"

mkdir -p "$HF_CACHE_DIR/hub" "$HF_CACHE_DIR/xet" "$HF_CACHE_DIR/assets"
mkdir -p "$OUTPUT_DIR" "$BLENDER_CACHE_DIR"

HF_HOME="$HF_CACHE_DIR" \
HF_HUB_CACHE="$HF_CACHE_DIR/hub" \
HF_XET_CACHE="$HF_CACHE_DIR/xet" \
HF_ASSETS_CACHE="$HF_CACHE_DIR/assets" \
CUDA_VISIBLE_DEVICES="$GPU" \
"$PYTHON_BIN" -m evaluation.generate_flux_relation_t2i \
  --prompt-file "$PROMPT_FILE" \
  --graph-encoder-path "$GRAPH_ENCODER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model-id black-forest-labs/FLUX.1-dev \
  --device cuda \
  --mixed-precision bf16 \
  --flux-quantization 8bit \
  --low-vram \
  --image-size 512 \
  --oscr-size 512 \
  --oscr-render-size 512 \
  --num-inference-steps 28 \
  --guidance-scale 3.5 \
  --max-sequence-length 128 \
  --samples-per-prompt "$SAMPLES_PER_PROMPT" \
  --limit-prompts 5 \
  --seed "$SEED" \
  --gnn-layout-sample-mode prior_sample \
  --decoder-film-scale "$DECODER_FILM_SCALE" \
  --use-official-seethrough3d-lora \
  --official-lora-cache-dir "$HF_CACHE_DIR" \
  --condition-renderer blender \
  --oscr-face-alpha 0.0025 \
  --blender-bin "$BLENDER_BIN" \
  --blender-cache-dir "$BLENDER_CACHE_DIR" \
  --prompt-prefix "a photo of"
