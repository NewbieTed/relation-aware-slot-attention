#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

exec env CATEGORY=spatial "$ROOT_DIR/scripts/benchmark/run_t2i_category_full.sh"
