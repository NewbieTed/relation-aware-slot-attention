#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
MMDET_ROOT="${MMDET_ROOT:-$ROOT_DIR/external/mmdetection}"
INSTALL_MMDET_STACK="${INSTALL_MMDET_STACK:-0}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Could not find Python interpreter at $VENV_PYTHON"
  echo "Create the repo environment first with ./scripts/setup/bootstrap_env.sh"
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip "setuptools>=68" wheel
"$VENV_PYTHON" -m pip install \
  pandas \
  open-clip-torch \
  clip-benchmark \
  "requests>=2.31.0" \
  "tqdm>=4.66.0"

if [[ "$INSTALL_MMDET_STACK" != "1" ]]; then
  echo "Installed the lightweight GenEval Python dependencies."
  echo
  echo "Skipped the legacy MMDetection/MMCV stack by default."
  echo "Why: GenEval's original detector stack targets much older PyTorch/CUDA combinations,"
  echo "and on modern environments like torch 2.11 / CUDA 13 it is often brittle."
  echo
  echo "If you explicitly want to try the legacy detector stack in this environment, rerun with:"
  echo "  INSTALL_MMDET_STACK=1 ./scripts/setup/benchmark/setup_geneval_env.sh"
  exit 0
fi

if [[ ! -d "$MMDET_ROOT/.git" ]]; then
  git clone https://github.com/open-mmlab/mmdetection.git "$MMDET_ROOT"
  git -C "$MMDET_ROOT" checkout 2.x
fi

"$VENV_PYTHON" -m pip install openmim
"$VENV_PYTHON" -m pip install --upgrade "setuptools>=68" wheel
"$VENV_PYTHON" -m pip install "Pillow==9.5.0"
"$VENV_PYTHON" -m pip install --no-build-isolation "mmcv-full==1.7.2"
"$VENV_PYTHON" -m pip install -v -e "$MMDET_ROOT"
"$VENV_PYTHON" -m pip install \
  "requests>=2.31.0" \
  "tqdm>=4.66.0" \
  "setuptools>=68"

echo "GenEval extra Python dependencies are installed."
