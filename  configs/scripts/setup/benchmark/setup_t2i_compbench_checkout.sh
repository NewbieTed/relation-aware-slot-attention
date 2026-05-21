#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
T2I_ROOT="${T2I_ROOT:-$ROOT_DIR/external/T2I-CompBench}"
OVERRIDES_DIR="$ROOT_DIR/scripts/benchmark/t2i_compbench_overrides"
T2I_REPO_URL="${T2I_REPO_URL:-https://github.com/Karine-Huang/T2I-CompBench.git}"

mkdir -p "$(dirname "$T2I_ROOT")"

if [[ ! -d "$T2I_ROOT/.git" ]]; then
  echo "Cloning T2I-CompBench into $T2I_ROOT"
  git clone "$T2I_REPO_URL" "$T2I_ROOT"
else
  echo "Using existing T2I-CompBench checkout at $T2I_ROOT"
fi

if [[ ! -d "$T2I_ROOT/UniDet_eval" ]]; then
  echo "Expected UniDet_eval under $T2I_ROOT but did not find it."
  exit 1
fi

echo "Applying local compatibility overrides"
cp -R "$OVERRIDES_DIR"/. "$T2I_ROOT"/

for relative_path in \
  "UniDet_eval/experts/__init__.py" \
  "UniDet_eval/experts/depth/__init__.py" \
  "UniDet_eval/experts/obj_detection/__init__.py" \
  "UniDet_eval/experts/obj_detection/unidet/data/__init__.py" \
  "UniDet_eval/experts/obj_detection/unidet/data/datasets/__init__.py" \
  "UniDet_eval/experts/obj_detection/unidet/evaluation/__init__.py" \
  "UniDet_eval/experts/obj_detection/unidet/modeling/__init__.py" \
  "UniDet_eval/experts/obj_detection/unidet/modeling/backbone/__init__.py" \
  "UniDet_eval/experts/obj_detection/unidet/modeling/meta_arch/__init__.py" \
  "UniDet_eval/experts/obj_detection/unidet/modeling/roi_heads/__init__.py"
do
  mkdir -p "$(dirname "$T2I_ROOT/$relative_path")"
  touch "$T2I_ROOT/$relative_path"
done

echo "T2I-CompBench checkout is ready at:"
echo "  $T2I_ROOT"
