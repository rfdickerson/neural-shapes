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
# Smoke-test friendly default; override with SAMPLE_COUNT for full runs.
SAMPLE_COUNT="${SAMPLE_COUNT:-250000}"
SAMPLES_BIN="${SAMPLES_BIN:-$TRAIN_DIR/outputs/wdas_cloud_quarter_samples_${SAMPLE_COUNT}.bin}"
SAMPLES_META="${SAMPLES_META:-$TRAIN_DIR/outputs/wdas_cloud_quarter_samples_${SAMPLE_COUNT}.json}"
FORCE_REBUILD_SAMPLES="${FORCE_REBUILD_SAMPLES:-0}"

TRAINING_MODE="${TRAINING_MODE:-detail_only_density}"
TARGET_SOURCE="${TARGET_SOURCE:-samples}"

BATCH_SIZE="${BATCH_SIZE:-8192}"
EVAL_SAMPLES="${EVAL_SAMPLES:-32768}"
MAX_STEPS="${MAX_STEPS:-12000}"
MIN_STEPS="${MIN_STEPS:-500}"
TARGET_MSE="${TARGET_MSE:-5e-5}"

FOURIER_LEVELS="${FOURIER_LEVELS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"

LR="${LR:-5e-4}"
LR_FINAL="${LR_FINAL:-1e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"

DETAIL_GRID_DIM="${DETAIL_GRID_DIM:-64}"
DETAIL_AMPLITUDE="${DETAIL_AMPLITUDE:-0.15}"
DETAIL_INSIDE_THRESHOLD="${DETAIL_INSIDE_THRESHOLD:-0.01}"
DETAIL_SHELL_BOOST="${DETAIL_SHELL_BOOST:-2.0}"
DETAIL_SMOOTHNESS_COEFF="${DETAIL_SMOOTHNESS_COEFF:-0.0}"
DETAIL_FADE_START="${DETAIL_FADE_START:-0.02}"
DETAIL_FADE_END="${DETAIL_FADE_END:-0.15}"

ISO_VALUE="${ISO_VALUE:-0.10}"
ISO_BAND="${ISO_BAND:-0.05}"

IMPORTANCE_OVERSAMPLE="${IMPORTANCE_OVERSAMPLE:-4}"
IMPORTANCE_MAX_ROUNDS="${IMPORTANCE_MAX_ROUNDS:-8}"

VAL_RATIO="${VAL_RATIO:-0.10}"
COARSE_GRID_SOURCE_BIN="${COARSE_GRID_SOURCE_BIN:-$TRAIN_DIR/outputs/wdas_cloud_quarter_256.bin}"
COARSE_GRID_SOURCE_META="${COARSE_GRID_SOURCE_META:-$TRAIN_DIR/outputs/wdas_cloud_quarter_256.json}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  "$PYTHON" "$TRAIN_DIR/train_torus_residual.py" --help
  exit 0
fi

if [[ "$TARGET_SOURCE" == "samples" ]]; then
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
fi

"$PYTHON" "$TRAIN_DIR/train_torus_residual.py" \
  --training-mode "$TRAINING_MODE" \
  --target-source "$TARGET_SOURCE" \
  --samples-bin "$SAMPLES_BIN" \
  --samples-meta "$SAMPLES_META" \
  --batch-size "$BATCH_SIZE" \
  --eval-samples "$EVAL_SAMPLES" \
  --max-steps "$MAX_STEPS" \
  --min-steps "$MIN_STEPS" \
  --target-mse "$TARGET_MSE" \
  --fourier-levels "$FOURIER_LEVELS" \
  --hidden-layers "$HIDDEN_SIZE" "$HIDDEN_SIZE" \
  --lr "$LR" \
  --lr-final "$LR_FINAL" \
  --lr-warmup-steps "$LR_WARMUP_STEPS" \
  --detail-grid-dim "$DETAIL_GRID_DIM" \
  --detail-amplitude "$DETAIL_AMPLITUDE" \
  --detail-inside-threshold "$DETAIL_INSIDE_THRESHOLD" \
  --detail-shell-boost "$DETAIL_SHELL_BOOST" \
  --detail-smoothness-coeff "$DETAIL_SMOOTHNESS_COEFF" \
  --detail-fade-start "$DETAIL_FADE_START" \
  --detail-fade-end "$DETAIL_FADE_END" \
  --iso-value "$ISO_VALUE" \
  --iso-band "$ISO_BAND" \
  --importance-oversample "$IMPORTANCE_OVERSAMPLE" \
  --importance-max-rounds "$IMPORTANCE_MAX_ROUNDS" \
  --val-ratio "$VAL_RATIO" \
  --coarse-grid-source-bin "$COARSE_GRID_SOURCE_BIN" \
  --coarse-grid-source-meta "$COARSE_GRID_SOURCE_META" \
  --output-dir "$TRAIN_DIR/outputs" \
  --copy-to-viz \
  --viz-mlp-dir "$ROOT_DIR/viz/public/mlp" \
  "$@"
