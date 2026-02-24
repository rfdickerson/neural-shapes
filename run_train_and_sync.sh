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
ISO_VALUE="${ISO_VALUE:-0.10}"
ISO_BAND="${ISO_BAND:-0.05}"
ISO_SHELL_RATIO="${ISO_SHELL_RATIO:-0.35}"
SURFACE_BAND_RATIO="${SURFACE_BAND_RATIO:-0.55}"
SURFACE_BAND_MIN="${SURFACE_BAND_MIN:-0.05}"
SURFACE_BAND_MAX="${SURFACE_BAND_MAX:-0.30}"
ISO_LOSS_WEIGHT="${ISO_LOSS_WEIGHT:-3.0}"
EMPTY_DENSITY_THRESHOLD="${EMPTY_DENSITY_THRESHOLD:-0.02}"
EMPTY_SPACE_LOSS_WEIGHT="${EMPTY_SPACE_LOSS_WEIGHT:-1.0}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
VAL_RATIO="${VAL_RATIO:-0.10}"
LR_FINAL="${LR_FINAL:-1e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
LOSS_RAMP_STEPS="${LOSS_RAMP_STEPS:-3000}"
TARGET_FIELD="${TARGET_FIELD:-levelset}"
LEVELSET_DECODE_K="${LEVELSET_DECODE_K:-16.0}"
LEVELSET_DENSITY_EPS="${LEVELSET_DENSITY_EPS:-1e-3}"

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
  --iso-value "$ISO_VALUE" \
  --iso-band "$ISO_BAND" \
  --iso-shell-ratio "$ISO_SHELL_RATIO" \
  --surface-band-ratio "$SURFACE_BAND_RATIO" \
  --surface-band-min "$SURFACE_BAND_MIN" \
  --surface-band-max "$SURFACE_BAND_MAX" \
  --loss-iso-weight "$ISO_LOSS_WEIGHT" \
  --empty-density-threshold "$EMPTY_DENSITY_THRESHOLD" \
  --loss-empty-space-weight "$EMPTY_SPACE_LOSS_WEIGHT" \
  --hidden-layers "$HIDDEN_SIZE" "$HIDDEN_SIZE" \
  --val-ratio "$VAL_RATIO" \
  --lr-final "$LR_FINAL" \
  --lr-warmup-steps "$LR_WARMUP_STEPS" \
  --loss-ramp-steps "$LOSS_RAMP_STEPS" \
  --target-field "$TARGET_FIELD" \
  --levelset-decode-k "$LEVELSET_DECODE_K" \
  --levelset-density-eps "$LEVELSET_DENSITY_EPS" \
  --output-dir "$TRAIN_DIR/outputs" \
  --copy-to-viz \
  --viz-mlp-dir "$ROOT_DIR/viz/public/mlp" \
  "$@"
