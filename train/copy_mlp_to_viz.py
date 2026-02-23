#!/usr/bin/env python3
"""Copy exported MLP artifacts from train outputs into viz/public/mlp."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy MLP export files into viz/public/mlp.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("train/outputs"),
        help="Directory containing residual_mlp_metadata.json and residual_mlp_weights.bin",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("viz/public/mlp"),
        help="Destination directory used by the Vite app",
    )
    parser.add_argument("--metadata-name", type=str, default="residual_mlp_metadata.json")
    parser.add_argument("--weights-name", type=str, default="residual_mlp_weights.bin")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_metadata(meta_path: Path) -> None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        fail(f"failed to parse metadata JSON '{meta_path}': {exc}")

    required_top_level = ["layout", "dtype", "encoding", "network", "offsets", "total_floats"]
    missing = [k for k in required_top_level if k not in meta]
    if missing:
        fail(f"metadata is missing required keys: {', '.join(missing)}")

    if meta["layout"] != "[W0][b0][W1][b1][W2][b2]":
        fail(f"unexpected layout '{meta['layout']}'")
    if meta["dtype"] != "float32":
        fail(f"unexpected dtype '{meta['dtype']}', expected 'float32'")


def main() -> None:
    args = parse_args()
    source_dir: Path = args.source_dir
    dest_dir: Path = args.dest_dir

    meta_src = source_dir / args.metadata_name
    weights_src = source_dir / args.weights_name

    if not source_dir.exists():
        fail(f"source directory does not exist: {source_dir}")
    if not meta_src.exists():
        fail(f"missing metadata file: {meta_src}")
    if not weights_src.exists():
        fail(f"missing weights file: {weights_src}")
    if weights_src.stat().st_size == 0:
        fail(f"weights file is empty: {weights_src}")

    validate_metadata(meta_src)

    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_dst = dest_dir / args.metadata_name
    weights_dst = dest_dir / args.weights_name

    shutil.copy2(meta_src, meta_dst)
    shutil.copy2(weights_src, weights_dst)

    print(f"copied metadata: {meta_src} -> {meta_dst}")
    print(f"copied weights:  {weights_src} -> {weights_dst}")
    print(f"weights bytes:   {weights_dst.stat().st_size}")


if __name__ == "__main__":
    main()
