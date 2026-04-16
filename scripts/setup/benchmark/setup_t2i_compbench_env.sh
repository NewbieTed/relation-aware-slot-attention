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

"$VENV_PYTHON" -m pip install --upgrade pip "setuptools>=68" wheel
"$VENV_PYTHON" -m pip install \
  "transformers==4.30.2" \
  "tokenizers==0.13.3" \
  "timm==0.4.12" \
  "fairscale==0.4.4" \
  "datasets==2.13.0" \
  "opencv-python==4.7.0.72" \
  "pycocotools==2.0.6" \
  "ftfy==6.1.1" \
  "fire==0.5.0" \
  "requests>=2.31.0" \
  "tqdm>=4.66.0"

"$VENV_PYTHON" -m pip install "git+https://github.com/openai/CLIP.git"

echo "T2I-CompBench extra Python dependencies are installed."
