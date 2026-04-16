#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_T2I_COMPBENCH_ENV="${INSTALL_T2I_COMPBENCH_ENV:-0}"

"$ROOT_DIR/scripts/setup/bootstrap_env.sh"
"$ROOT_DIR/scripts/setup/bootstrap_benchmark.sh"

if [[ "$INSTALL_T2I_COMPBENCH_ENV" == "1" ]]; then
  "$ROOT_DIR/scripts/setup/benchmark/setup_t2i_compbench_env.sh"
fi

echo
echo "All onboarding steps completed."
echo "Next:"
echo "  source $ROOT_DIR/.venv/bin/activate"
echo
echo "Optional benchmark compatibility environment:"
echo "  PYTHON_BIN=python3.10 $ROOT_DIR/scripts/setup/benchmark/setup_t2i_compbench_env.sh"
echo "Or create it during bootstrap:"
echo "  INSTALL_T2I_COMPBENCH_ENV=1 PYTHON_BIN=python3.10 $ROOT_DIR/scripts/setup/bootstrap_all.sh"
