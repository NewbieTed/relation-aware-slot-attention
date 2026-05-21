#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
ROOT_OUTPUT_DIR="${ROOT_OUTPUT_DIR:-outputs/eval/t2i_compbench}"

"$PYTHON_BIN" -m evaluation.t2i_compbench_overall --root-dir "$ROOT_OUTPUT_DIR"
