#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

uv venv .venv
uv pip install --python .venv/bin/python -e .

echo
echo "Environment ready."
echo "Activate it with: source .venv/bin/activate"
