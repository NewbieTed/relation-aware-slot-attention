#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEETHROUGH3D_ROOT="${SEETHROUGH3D_ROOT:-$ROOT_DIR/external/seethrough3d}"
SEETHROUGH3D_REPO_URL="${SEETHROUGH3D_REPO_URL:-https://github.com/va1bhavagrawal/seethrough3d.git}"

if [[ -d "$SEETHROUGH3D_ROOT/.git" ]]; then
  echo "Using existing SeeThrough3D checkout at $SEETHROUGH3D_ROOT"
else
  mkdir -p "$(dirname "$SEETHROUGH3D_ROOT")"
  echo "Cloning SeeThrough3D into $SEETHROUGH3D_ROOT"
  git clone "$SEETHROUGH3D_REPO_URL" "$SEETHROUGH3D_ROOT"
fi

echo "SeeThrough3D checkout ready."
