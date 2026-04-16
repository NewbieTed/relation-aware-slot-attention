#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

find_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      command -v "$PYTHON_BIN"
      return 0
    fi
    echo "Requested PYTHON_BIN was not found on PATH: $PYTHON_BIN" >&2
  fi

  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find a usable Python interpreter."
  echo "Set PYTHON_BIN explicitly, for example:"
  echo "  PYTHON_BIN=/full/path/to/python ./scripts/setup/bootstrap_all.sh"
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_MAJOR="${PYTHON_VERSION%%.*}"
PYTHON_MINOR="${PYTHON_VERSION#*.}"
if [[ "$PYTHON_MAJOR" -lt 3 || ( "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ) ]]; then
  echo "Python 3.10+ is required, but found $PYTHON_VERSION from $PYTHON_BIN"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip "setuptools<82" wheel
"$VENV_PYTHON" -m pip install -e ".[eval]"
"$VENV_PYTHON" -m pip install "ruamel.yaml==0.17.32" spacy gdown

if ! "$VENV_PYTHON" -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
  "$VENV_PYTHON" -m spacy download en_core_web_sm
fi

if [[ "$(uname -s)" == "Linux" ]]; then
  "$VENV_PYTHON" -m pip install "Pillow==9.5.0"
  "$VENV_PYTHON" -m pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13"
else
  echo "Skipping detectron2 auto-install on non-Linux."
fi

echo
echo "Environment ready."
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"
