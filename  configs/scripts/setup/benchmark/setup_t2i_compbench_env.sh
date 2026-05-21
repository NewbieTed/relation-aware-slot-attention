#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T2I_VENV_DIR="${T2I_VENV_DIR:-$ROOT_DIR/.venv-t2i}"
VENV_PYTHON="${VENV_PYTHON:-$T2I_VENV_DIR/bin/python}"
T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"

find_python() {
  local requested="${PYTHON_BIN:-}"
  if [[ -n "$requested" ]]; then
    if command -v "$requested" >/dev/null 2>&1; then
      command -v "$requested"
      return 0
    fi
    echo "Requested PYTHON_BIN was not found on PATH: $requested" >&2
  fi

  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

ensure_python310_plus() {
  local python_bin="$1"
  local version
  version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  local major="${version%%.*}"
  local minor="${version#*.}"
  if [[ "$major" -lt 3 || ( "$major" -eq 3 && "$minor" -lt 10 ) ]]; then
    echo "Python 3.10+ is required for the T2I-CompBench environment, but found $version from $python_bin"
    exit 1
  fi
}

warn_if_not_python310() {
  local python_bin="$1"
  local version
  version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$version" != "3.10" ]]; then
    echo "Warning: T2I-CompBench is most reliable with Python 3.10; currently using $version from $python_bin"
  fi
}

if [[ "$T2I_VENV_DIR" == "$ROOT_DIR/.venv" || "$VENV_PYTHON" == "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Refusing to install T2I-CompBench dependencies into the repo's main .venv."
  echo "Use the dedicated benchmark environment instead:"
  echo "  T2I_VENV_DIR=$ROOT_DIR/.venv-t2i ./scripts/setup/benchmark/setup_t2i_compbench_env.sh"
  exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$(find_python || true)"
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "Could not find a usable Python interpreter."
    echo "Set PYTHON_BIN explicitly, for example:"
    echo "  PYTHON_BIN=/full/path/to/python3.10 ./scripts/setup/benchmark/setup_t2i_compbench_env.sh"
    exit 1
  fi
  ensure_python310_plus "$PYTHON_BIN"
  warn_if_not_python310 "$PYTHON_BIN"
  echo "Creating dedicated T2I-CompBench environment at $T2I_VENV_DIR"
  "$PYTHON_BIN" -m venv "$T2I_VENV_DIR"
fi

"$VENV_PYTHON" -m pip install --upgrade pip "setuptools<82" wheel
"$VENV_PYTHON" -m pip install \
  "Pillow==9.5.0" \
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

echo "T2I-CompBench environment is ready."
echo "Use this interpreter for evaluation:"
echo "  $VENV_PYTHON"
