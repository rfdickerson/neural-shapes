#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$ROOT_DIR/train"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$TRAIN_DIR/.venv/bin/python" ]]; then
  PYTHON="$TRAIN_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi

VDB_PATH="${VDB_PATH:-$TRAIN_DIR/wdas_cloud_quarter.vdb}"
VOLUME_DIM="${VOLUME_DIM:-256}"
VOLUME_BIN="${VOLUME_BIN:-$TRAIN_DIR/outputs/wdas_cloud_quarter_${VOLUME_DIM}.bin}"
VOLUME_META="${VOLUME_META:-$TRAIN_DIR/outputs/wdas_cloud_quarter_${VOLUME_DIM}.json}"
FORCE_REBUILD_VOLUME="${FORCE_REBUILD_VOLUME:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  "$PYTHON" "$TRAIN_DIR/train_torus_residual.py" --help
  exit 0
fi

if [[ ! -f "$VDB_PATH" ]]; then
  echo "error: missing VDB source file: $VDB_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$VOLUME_BIN")"

if [[ "$FORCE_REBUILD_VOLUME" == "1" || ! -f "$VOLUME_BIN" || ! -f "$VOLUME_META" ]]; then
  "$TRAIN_DIR/convert_vdb_to_dense.sh" \
    --input "$VDB_PATH" \
    --output-bin "$VOLUME_BIN" \
    --output-json "$VOLUME_META" \
    --dim "$VOLUME_DIM"
fi

"$PYTHON" "$TRAIN_DIR/train_torus_residual.py" \
  --target-source volume \
  --volume-bin "$VOLUME_BIN" \
  --volume-meta "$VOLUME_META" \
  --output-dir "$TRAIN_DIR/outputs" \
  --copy-to-viz \
  --viz-mlp-dir "$ROOT_DIR/viz/public/mlp" \
  "$@"
