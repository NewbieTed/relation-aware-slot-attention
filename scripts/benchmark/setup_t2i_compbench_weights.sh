#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT_DIR/evaluation/vendor/t2i_compbench_spatial/unidet/experts/expert_weights}"

mkdir -p "$WEIGHTS_DIR"

download_if_missing() {
  local url="$1"
  local output_path="$2"

  if [[ -f "$output_path" ]]; then
    echo "Already present: $output_path"
    return 0
  fi

  echo "Downloading $(basename "$output_path")"
  curl -L "$url" -o "$output_path"
}

download_if_missing \
  "https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth" \
  "$WEIGHTS_DIR/Unified_learned_OCIM_RS200_6x+2x.pth"

download_if_missing \
  "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt" \
  "$WEIGHTS_DIR/dpt_hybrid-midas-501f0c75.pt"

echo "T2I-CompBench spatial benchmark weights are ready in:"
echo "  $WEIGHTS_DIR"
