#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Could not find Python interpreter at $VENV_PYTHON"
  echo "Create the repo environment first with ./scripts/setup/bootstrap_env.sh"
  exit 1
fi

"$VENV_PYTHON" -m pip install -r "$T2I_ROOT/requirements.txt"

echo "T2I-CompBench extra Python dependencies are installed."
