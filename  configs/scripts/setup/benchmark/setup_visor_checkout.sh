#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VISOR_ROOT="${VISOR_ROOT:-$ROOT_DIR/external/VISOR}"
VISOR_REPO_URL="${VISOR_REPO_URL:-https://github.com/microsoft/VISOR.git}"

mkdir -p "$(dirname "$VISOR_ROOT")"

if [[ ! -d "$VISOR_ROOT/.git" ]]; then
  echo "Cloning VISOR into $VISOR_ROOT"
  git clone "$VISOR_REPO_URL" "$VISOR_ROOT"
else
  echo "Using existing VISOR checkout at $VISOR_ROOT"
fi
