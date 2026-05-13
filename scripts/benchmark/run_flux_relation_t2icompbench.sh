#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_FLUX_PYTHON="python3"
if [[ -x "$ROOT_DIR/.venv-flux/bin/python" ]]; then
  DEFAULT_FLUX_PYTHON="$ROOT_DIR/.venv-flux/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_FLUX_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

BENCHMARK="${BENCHMARK:-spatial}"
T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"
FLUX_PYTHON="${FLUX_PYTHON:-$DEFAULT_FLUX_PYTHON}"
T2I_PYTHON="${T2I_PYTHON:-$ROOT_DIR/.venv-t2i/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
GRAPH_ENCODER_PATH="${GRAPH_ENCODER_PATH:-}"
DEVICE="${DEVICE:-cuda}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
FLUX_QUANTIZATION="${FLUX_QUANTIZATION:-4bit}"
LOW_VRAM="${LOW_VRAM:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
OSCR_SIZE="${OSCR_SIZE:-256}"
OSCR_RENDER_SIZE="${OSCR_RENDER_SIZE:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-128}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
EXTERNAL_LORA_SAFETENSORS="${EXTERNAL_LORA_SAFETENSORS:-}"
USE_OFFICIAL_SEETHROUGH3D_LORA="${USE_OFFICIAL_SEETHROUGH3D_LORA:-0}"
OFFICIAL_LORA_CACHE_DIR="${OFFICIAL_LORA_CACHE_DIR:-}"
CONDITION_RENDERER="${CONDITION_RENDERER:-seethrough}"
OSCR_FACE_ALPHA="${OSCR_FACE_ALPHA:-0.10}"
OSCR_AZIMUTH_DEGREES="${OSCR_AZIMUTH_DEGREES:-0}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
BLENDER_CACHE_DIR="${BLENDER_CACHE_DIR:-}"
PROMPT_PREFIX="${PROMPT_PREFIX:-a photo of}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/eval/flux_relation_${BENCHMARK}}"
PROMPT_FILE="${PROMPT_FILE:-}"
COPY_INSTEAD_OF_SYMLINK="${COPY_INSTEAD_OF_SYMLINK:-0}"
PRUNE_SAMPLES_KEEP="${PRUNE_SAMPLES_KEEP:-}"
PRUNE_SAMPLES_SEED="${PRUNE_SAMPLES_SEED:-42}"

if [[ -n "$CONFIG_FILE" ]]; then
  readarray -t CONFIG_EXPORTS < <("$FLUX_PYTHON" - "$CONFIG_FILE" <<'PY'
from pathlib import Path
from training.config import _load_raw_config, _section_config

config = _section_config(_load_raw_config(Path(__import__("sys").argv[1])), "benchmark")
keys = {
    "benchmark": "BENCHMARK",
    "t2i_root": "T2I_ROOT",
    "t2i_python": "T2I_PYTHON",
    "output_dir": "OUTPUT_DIR",
    "prompt_file": "PROMPT_FILE",
    "copy_instead_of_symlink": "COPY_INSTEAD_OF_SYMLINK",
    "prune_samples_keep": "PRUNE_SAMPLES_KEEP",
    "prune_samples_seed": "PRUNE_SAMPLES_SEED",
}
for key, env_key in keys.items():
    value = config.get(key)
    if value is None:
        continue
    print(f"{env_key}={value}")
PY
  )
  for item in "${CONFIG_EXPORTS[@]}"; do
    export "$item"
  done
  BENCHMARK="${BENCHMARK:-spatial}"
  T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"
  T2I_PYTHON="${T2I_PYTHON:-$ROOT_DIR/.venv-t2i/bin/python}"
  OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/eval/flux_relation_${BENCHMARK}}"
  PROMPT_FILE="${PROMPT_FILE:-}"
  COPY_INSTEAD_OF_SYMLINK="${COPY_INSTEAD_OF_SYMLINK:-0}"
  PRUNE_SAMPLES_KEEP="${PRUNE_SAMPLES_KEEP:-}"
  PRUNE_SAMPLES_SEED="${PRUNE_SAMPLES_SEED:-42}"
fi

if [[ -z "$PROMPT_FILE" ]]; then
  case "$BENCHMARK" in
    spatial)
      PROMPT_FILE="$T2I_ROOT/examples/dataset/spatial_val.txt"
      ;;
    3d_spatial)
      PROMPT_FILE="$T2I_ROOT/examples/dataset/3d_spatial_val.txt"
      ;;
    *)
      echo "Unsupported default prompt file for BENCHMARK=$BENCHMARK. Set PROMPT_FILE explicitly." >&2
      exit 2
      ;;
  esac
fi

if [[ -n "$CONFIG_FILE" ]]; then
  GEN_CMD=("$FLUX_PYTHON" -m evaluation.generate_flux_relation_t2i --config "$CONFIG_FILE")
else
  GEN_CMD=(
    "$FLUX_PYTHON" -m evaluation.generate_flux_relation_t2i
    --prompt-file "$PROMPT_FILE"
    --output-dir "$OUTPUT_DIR"
    --model-id "$MODEL_ID"
    --device "$DEVICE"
    --mixed-precision "$MIXED_PRECISION"
    --flux-quantization "$FLUX_QUANTIZATION"
    --image-size "$IMAGE_SIZE"
    --oscr-size "$OSCR_SIZE"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --guidance-scale "$GUIDANCE_SCALE"
    --max-sequence-length "$MAX_SEQUENCE_LENGTH"
    --samples-per-prompt "$SAMPLES_PER_PROMPT"
    --seed "$SEED"
    --lora-rank "$LORA_RANK"
    --lora-alpha "$LORA_ALPHA"
    --condition-renderer "$CONDITION_RENDERER"
    --oscr-face-alpha "$OSCR_FACE_ALPHA"
    --oscr-azimuth-degrees "$OSCR_AZIMUTH_DEGREES"
    --blender-bin "$BLENDER_BIN"
    --prompt-prefix "$PROMPT_PREFIX"
  )

  if [[ -n "$CHECKPOINT_DIR" ]]; then
    GEN_CMD+=(--checkpoint-dir "$CHECKPOINT_DIR")
  fi
  if [[ -n "$GRAPH_ENCODER_PATH" ]]; then
    GEN_CMD+=(--graph-encoder-path "$GRAPH_ENCODER_PATH")
  fi
  if [[ "$LOW_VRAM" == "1" ]]; then
    GEN_CMD+=(--low-vram)
  fi
  if [[ "$USE_OFFICIAL_SEETHROUGH3D_LORA" == "1" ]]; then
    GEN_CMD+=(--use-official-seethrough3d-lora)
  fi
  if [[ -n "$EXTERNAL_LORA_SAFETENSORS" ]]; then
    GEN_CMD+=(--external-lora-safetensors "$EXTERNAL_LORA_SAFETENSORS")
  fi
  if [[ -n "$OFFICIAL_LORA_CACHE_DIR" ]]; then
    GEN_CMD+=(--official-lora-cache-dir "$OFFICIAL_LORA_CACHE_DIR")
  fi
  if [[ -n "$LIMIT_PROMPTS" ]]; then
    GEN_CMD+=(--limit-prompts "$LIMIT_PROMPTS")
  fi
  if [[ -n "$OSCR_RENDER_SIZE" ]]; then
    GEN_CMD+=(--oscr-render-size "$OSCR_RENDER_SIZE")
  fi
  if [[ -n "$BLENDER_CACHE_DIR" ]]; then
    GEN_CMD+=(--blender-cache-dir "$BLENDER_CACHE_DIR")
  fi
fi

"${GEN_CMD[@]}"

EVAL_CMD=(
  "$FLUX_PYTHON" -m evaluation.t2i_compbench
  --t2i-compbench-root "$T2I_ROOT"
  --generated-dir "$OUTPUT_DIR"
  --benchmark "$BENCHMARK"
  --prompt-file "$PROMPT_FILE"
  --python-bin "$T2I_PYTHON"
)

if [[ "$COPY_INSTEAD_OF_SYMLINK" == "1" ]]; then
  EVAL_CMD+=(--copy-instead-of-symlink)
fi
if [[ -n "$PRUNE_SAMPLES_KEEP" ]]; then
  EVAL_CMD+=(--prune-samples-keep "$PRUNE_SAMPLES_KEEP" --prune-samples-seed "$PRUNE_SAMPLES_SEED")
fi

"${EVAL_CMD[@]}"
