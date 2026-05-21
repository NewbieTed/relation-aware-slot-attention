#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GENEVAL_ROOT="${GENEVAL_ROOT:-$ROOT_DIR/external/geneval}"
GENEVAL_REPO_URL="${GENEVAL_REPO_URL:-https://github.com/djghosh13/geneval.git}"

mkdir -p "$(dirname "$GENEVAL_ROOT")"

if [[ ! -d "$GENEVAL_ROOT/.git" ]]; then
  echo "Cloning GenEval into $GENEVAL_ROOT"
  git clone "$GENEVAL_REPO_URL" "$GENEVAL_ROOT"
else
  echo "Using existing GenEval checkout at $GENEVAL_ROOT"
fi
