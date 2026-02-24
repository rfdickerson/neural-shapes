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
SAMPLE_COUNT="${SAMPLE_COUNT:-4000000}"
SAMPLES_BIN="${SAMPLES_BIN:-$TRAIN_DIR/outputs/wdas_cloud_quarter_samples_${SAMPLE_COUNT}.bin}"
SAMPLES_META="${SAMPLES_META:-$TRAIN_DIR/outputs/wdas_cloud_quarter_samples_${SAMPLE_COUNT}.json}"
FORCE_REBUILD_SAMPLES="${FORCE_REBUILD_SAMPLES:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  "$PYTHON" "$TRAIN_DIR/train_torus_residual.py" --help
  exit 0
fi

if [[ ! -f "$VDB_PATH" ]]; then
  echo "error: missing VDB source file: $VDB_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$SAMPLES_BIN")"

if [[ "$FORCE_REBUILD_SAMPLES" == "1" || ! -f "$SAMPLES_BIN" || ! -f "$SAMPLES_META" ]]; then
  "$TRAIN_DIR/sample_vdb_points.sh" \
    --input "$VDB_PATH" \
    --output-bin "$SAMPLES_BIN" \
    --output-json "$SAMPLES_META" \
    --samples "$SAMPLE_COUNT"
fi

"$PYTHON" "$TRAIN_DIR/train_torus_residual.py" \
  --target-source samples \
  --samples-bin "$SAMPLES_BIN" \
  --samples-meta "$SAMPLES_META" \
  --output-dir "$TRAIN_DIR/outputs" \
  --copy-to-viz \
  --viz-mlp-dir "$ROOT_DIR/viz/public/mlp" \
  "$@"
