#!/usr/bin/env python3
"""
Train a residual MLP to approximate torus density relative to a box baseline.

This script is self-contained and exports:
  - FP32 flat weight blob in layout [W0][b0][W1][b1][W2][b2]
  - metadata JSON with layer sizes and offsets
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sdf_torus(points: torch.Tensor, major_radius: float = 0.55, minor_radius: float = 0.2) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    qx = torch.sqrt(x * x + z * z + 1e-12) - major_radius
    qy = y
    return torch.sqrt(qx * qx + qy * qy + 1e-12) - minor_radius


def sdf_box(points: torch.Tensor, half_extents: Tuple[float, float, float] = (0.6, 0.25, 0.6)) -> torch.Tensor:
    b = torch.tensor(half_extents, device=points.device, dtype=points.dtype)
    q = torch.abs(points) - b
    outside = torch.clamp(q, min=0.0)
    outside_len = torch.linalg.norm(outside, dim=-1)
    inside = torch.clamp(torch.amax(q, dim=-1), max=0.0)
    return outside_len + inside


def density_from_sdf(sdf: torch.Tensor, sharpness: float) -> torch.Tensor:
    return torch.sigmoid(-sharpness * sdf)


def torus_density(points: torch.Tensor) -> torch.Tensor:
    return density_from_sdf(sdf_torus(points), sharpness=36.0)


def box_density(points: torch.Tensor) -> torch.Tensor:
    # Smooth procedural baseline.
    return density_from_sdf(sdf_box(points), sharpness=14.0)


class FourierEncoding(nn.Module):
    def __init__(self, levels: int = 4, include_input: bool = True) -> None:
        super().__init__()
        self.levels = levels
        self.include_input = include_input

    @property
    def output_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 6 * self.levels

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        out: List[torch.Tensor] = []
        if self.include_input:
            out.append(points)
        for i in range(self.levels):
            freq = float(2**i)
            out.append(torch.sin(freq * points))
            out.append(torch.cos(freq * points))
        return torch.cat(out, dim=-1)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int] = (64, 64), output_dim: int = 1) -> None:
        super().__init__()
        h0, h1 = hidden_sizes
        self.fc0 = nn.Linear(input_dim, h0)
        self.fc1 = nn.Linear(h0, h1)
        self.fc2 = nn.Linear(h1, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def export_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        return [
            ("W0", self.fc0.weight),
            ("b0", self.fc0.bias),
            ("W1", self.fc1.weight),
            ("b1", self.fc1.bias),
            ("W2", self.fc2.weight),
            ("b2", self.fc2.bias),
        ]


@torch.no_grad()
def sample_batch(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    points = torch.rand(batch_size, 3, device=device) * 2.0 - 1.0
    gt = torus_density(points)
    baseline = box_density(points)
    residual = (gt - baseline).unsqueeze(-1)
    return points, residual


def export_model(
    model: ResidualMLP,
    encoding_levels: int,
    output_dir: Path,
    weights_name: str = "residual_mlp_weights.bin",
    meta_name: str = "residual_mlp_metadata.json",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / weights_name
    meta_path = output_dir / meta_name

    tensors = model.export_tensors()
    offsets: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    shapes: Dict[str, List[int]] = {}
    blob = bytearray()

    cursor = 0
    for name, tensor in tensors:
        flat = tensor.detach().cpu().contiguous().view(-1).to(torch.float32)
        arr = flat.tolist()
        offsets[name] = cursor
        counts[name] = int(len(arr))
        shapes[name] = list(tensor.shape)
        blob.extend(struct.pack(f"<{len(arr)}f", *arr))
        cursor += int(len(arr))

    weights_path.write_bytes(blob)

    metadata = {
        "layout": "[W0][b0][W1][b1][W2][b2]",
        "dtype": "float32",
        "encoding": {
            "type": "fourier",
            "levels": encoding_levels,
            "include_input": True,
            "frequency_base": 2.0,
            "input_dims": 3,
            "encoded_dims": 3 + 6 * encoding_levels,
        },
        "network": {
            "layers": [
                int(model.fc0.in_features),
                int(model.fc0.out_features),
                int(model.fc1.out_features),
                int(model.fc2.out_features),
            ]
        },
        "offsets": offsets,
        "counts": counts,
        "shapes": shapes,
        "total_floats": int(cursor),
        "total_bytes": int(len(blob)),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Exported weights: {weights_path}")
    print(f"Exported metadata: {meta_path}")


def copy_exports_to_viz(
    source_dir: Path,
    dest_dir: Path,
    weights_name: str,
    meta_name: str,
) -> None:
    weights_src = source_dir / weights_name
    meta_src = source_dir / meta_name
    if not weights_src.exists():
        raise FileNotFoundError(f"Missing weights file for copy: {weights_src}")
    if not meta_src.exists():
        raise FileNotFoundError(f"Missing metadata file for copy: {meta_src}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    weights_dst = dest_dir / weights_name
    meta_dst = dest_dir / meta_name

    shutil.copy2(weights_src, weights_dst)
    shutil.copy2(meta_src, meta_dst)

    print(f"Copied weights to viz:  {weights_dst}")
    print(f"Copied metadata to viz: {meta_dst}")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if args.fourier_levels < 1:
        raise ValueError("fourier_levels must be >= 1 to enable positional encoding.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    encoder = FourierEncoding(levels=args.fourier_levels).to(device)
    model = ResidualMLP(input_dim=encoder.output_dim, hidden_sizes=(64, 64), output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ema_loss = None
    stop_step = args.max_steps

    for step in range(1, args.max_steps + 1):
        points, target_residual = sample_batch(args.batch_size, device)
        encoded = encoder(points)
        pred_residual = model(encoded)

        loss = F.mse_loss(pred_residual, target_residual)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)

        if step % args.log_every == 0 or step == 1:
            print(f"step={step:6d}  loss={loss_val:.8f}  ema={ema_loss:.8f}")

        if step >= args.min_steps and ema_loss is not None and ema_loss <= args.target_mse:
            stop_step = step
            print(f"Early stop at step {step}: ema_loss={ema_loss:.8f} <= target_mse={args.target_mse:.8f}")
            break

    with torch.no_grad():
        points, target_residual = sample_batch(args.eval_samples, device)
        eval_pred = model(encoder(points))
        eval_mse = F.mse_loss(eval_pred, target_residual).item()
        print(f"final_eval_mse={eval_mse:.8f}  (trained_steps={stop_step})")

    export_model(
        model=model,
        encoding_levels=args.fourier_levels,
        output_dir=Path(args.output_dir),
        weights_name=args.weights_name,
        meta_name=args.meta_name,
    )
    if args.copy_to_viz:
        copy_exports_to_viz(
            source_dir=Path(args.output_dir),
            dest_dir=Path(args.viz_mlp_dir),
            weights_name=args.weights_name,
            meta_name=args.meta_name,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    default_viz_dir = repo_root / "viz" / "public" / "mlp"
    p = argparse.ArgumentParser(description="Train torus-vs-box residual MLP and export flat weights.")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--weights-name", type=str, default="residual_mlp_weights.bin")
    p.add_argument("--meta-name", type=str, default="residual_mlp_metadata.json")
    p.add_argument(
        "--copy-to-viz",
        action="store_true",
        help="After export, copy metadata/weights into viz/public/mlp.",
    )
    p.add_argument(
        "--viz-mlp-dir",
        type=str,
        default=str(default_viz_dir),
        help="Destination folder for renderer artifacts when --copy-to-viz is set.",
    )
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--eval-samples", type=int, default=32768)
    p.add_argument("--max-steps", type=int, default=12000)
    p.add_argument("--min-steps", type=int, default=500)
    p.add_argument("--target-mse", type=float, default=5e-5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fourier-levels", type=int, default=4)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
