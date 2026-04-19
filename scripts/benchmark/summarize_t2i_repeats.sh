#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
CATEGORY="${CATEGORY:-spatial}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/eval/sd15_t2i_compbench_${CATEGORY}_repeats}"

"$PYTHON_BIN" -m evaluation.t2i_compbench_repeats \
  --root-dir "$OUTPUT_ROOT" \
  --benchmark "$CATEGORY"
