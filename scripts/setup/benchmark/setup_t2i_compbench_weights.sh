#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"

if [[ -d "$T2I_ROOT/UniDet_eval" ]]; then
  UNIDET_ROOT="$T2I_ROOT/UniDet_eval"
else
  UNIDET_ROOT="$T2I_ROOT"
fi

WEIGHTS_DIR="${WEIGHTS_DIR:-$UNIDET_ROOT/experts/expert_weights}"

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
