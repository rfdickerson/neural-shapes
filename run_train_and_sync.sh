#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/train/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/train/.venv/bin/python"
else
  PYTHON="python3"
fi

"$PYTHON" "$ROOT_DIR/train/train_torus_residual.py" \
  --output-dir "$ROOT_DIR/train/outputs" \
  --copy-to-viz \
  --viz-mlp-dir "$ROOT_DIR/viz/public/mlp" \
  "$@"
