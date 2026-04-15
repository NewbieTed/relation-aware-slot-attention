#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/setup/bootstrap_env.sh"
"$ROOT_DIR/setup/bootstrap_benchmark.sh"

echo
echo "All onboarding steps completed."
echo "Next:"
echo "  source $ROOT_DIR/.venv/bin/activate"
