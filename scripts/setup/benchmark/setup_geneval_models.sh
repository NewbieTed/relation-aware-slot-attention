#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GENEVAL_ROOT="${GENEVAL_ROOT:-$ROOT_DIR/external/geneval}"
MODEL_DIR="${MODEL_DIR:-$GENEVAL_ROOT/models}"

mkdir -p "$MODEL_DIR"
bash "$GENEVAL_ROOT/evaluation/download_models.sh" "$MODEL_DIR"

echo "GenEval detector weights are ready in:"
echo "  $MODEL_DIR"
