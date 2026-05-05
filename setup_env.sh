#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT_DIR/scripts/setup/bootstrap_env.sh"
"$ROOT_DIR/scripts/setup/setup_seethrough3d_checkout.sh"
