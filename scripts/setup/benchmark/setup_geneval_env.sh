#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
MMDET_ROOT="${MMDET_ROOT:-$ROOT_DIR/external/mmdetection}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Could not find Python interpreter at $VENV_PYTHON"
  echo "Create the repo environment first with ./scripts/setup/bootstrap_env.sh"
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install openmim pandas open-clip-torch clip-benchmark
"$VENV_PYTHON" -m mim install "mmcv-full==1.7.2"

if [[ ! -d "$MMDET_ROOT/.git" ]]; then
  git clone https://github.com/open-mmlab/mmdetection.git "$MMDET_ROOT"
  git -C "$MMDET_ROOT" checkout 2.x
fi

"$VENV_PYTHON" -m pip install -v -e "$MMDET_ROOT"

echo "GenEval extra Python dependencies are installed."
