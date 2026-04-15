#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$ROOT_DIR/scripts/setup/benchmark/setup_t2i_compbench_checkout.sh"
"$ROOT_DIR/scripts/setup/benchmark/setup_t2i_compbench_weights.sh"

echo
echo "Benchmark bootstrap complete."
echo "If you have not created the repo environment yet, run:"
echo "  ./scripts/setup/bootstrap_env.sh"
