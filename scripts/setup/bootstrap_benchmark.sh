#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/setup/benchmark/setup_t2i_compbench_checkout.sh"
"$ROOT_DIR/setup/benchmark/setup_t2i_compbench_weights.sh"

echo
echo "Benchmark bootstrap complete."
echo "Next, install Python dependencies if needed:"
echo '  python3 -m pip install -e ".[eval]"'
