#!/usr/bin/env python3
"""
Train a detail-only density MLP for clouds.

Representation:
  density(x) = clamp(coarse_density(x) + detail_amplitude * detail(x), 0, 1)
  detail(x) = tanh(mlp(fourier(x)))

Targets supported:
  - torus (analytic smooth torus)
  - volume (dense preconverted volume sampled in normalized [-1,1]^3)
  - samples (sparse xyz+density samples generated directly from VDB)

This script exports:
  - FP32 flat weight blob in layout [W0][b0][W1][b1][W2][b2]
  - metadata JSON for coarse+detail reconstruction
  - coarse density grid (.bin + .json)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TORUS_MAJOR_RADIUS: float = 0.55
TORUS_MINOR_RADIUS: float = 0.2
TORUS_SHARPNESS: float = 36.0


@dataclass
class SparseSampleBank:
    points: torch.Tensor  # [N,3] in normalized [-1,1]^3
    density: torch.Tensor  # [N] in [0,1]


def sdf_torus(
    points: torch.Tensor, major_radius: float = TORUS_MAJOR_RADIUS, minor_radius: float = TORUS_MINOR_RADIUS
) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    qx = torch.sqrt(x * x + z * z + 1e-12) - major_radius
    qy = y
    return torch.sqrt(qx * qx + qy * qy + 1e-12) - minor_radius


def torus_density(points: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(-TORUS_SHARPNESS * sdf_torus(points))


def load_dense_volume(
    volume_bin_path: Path,
    volume_meta_path: Path | None,
    device: torch.device,
    clamp_to_unit: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    meta_path = volume_meta_path if volume_meta_path is not None else volume_bin_path.with_suffix(".json")
    if not volume_bin_path.exists():
        raise FileNotFoundError(f"missing dense volume binary: {volume_bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing volume metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dims = meta.get("dims")
    if not isinstance(dims, list) or len(dims) != 3:
        raise ValueError(f"volume metadata must contain dims=[D,H,W], got {dims}")

    depth, height, width = int(dims[0]), int(dims[1]), int(dims[2])
    expected_floats = depth * height * width
    raw = volume_bin_path.read_bytes()
    if len(raw) != expected_floats * 4:
        raise ValueError(
            f"volume size mismatch for {volume_bin_path}: expected {expected_floats * 4} bytes, got {len(raw)} bytes"
        )

    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    volume = flat.view(depth, height, width).to(device=device, dtype=torch.float32)
    if clamp_to_unit:
        volume = volume.clamp(0.0, 1.0)
    return volume, meta


def save_dense_grid(
    grid: torch.Tensor,
    grid_bin_path: Path,
    grid_meta_path: Path,
    field_name: str,
    extra_meta: Dict[str, Any] | None = None,
) -> None:
    g = grid.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if g.ndim != 3:
        raise ValueError(f"expected 3D grid, got shape={tuple(g.shape)}")
    depth, height, width = int(g.shape[0]), int(g.shape[1]), int(g.shape[2])
    grid_bin_path.parent.mkdir(parents=True, exist_ok=True)
    grid_meta_path.parent.mkdir(parents=True, exist_ok=True)
    flat = g.view(-1).tolist()
    grid_bin_path.write_bytes(struct.pack(f"<{len(flat)}f", *flat))
    meta: Dict[str, Any] = {
        "dims": [depth, height, width],
        "dtype": "float32",
        "field": field_name,
        "domain": "[-1,1]^3",
    }
    if extra_meta is not None:
        meta.update(extra_meta)
    grid_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_sparse_samples(
    samples_bin_path: Path,
    samples_meta_path: Path | None,
) -> Tuple[SparseSampleBank, Dict[str, Any]]:
    meta_path = samples_meta_path if samples_meta_path is not None else samples_bin_path.with_suffix(".json")
    if not samples_bin_path.exists():
        raise FileNotFoundError(f"missing sparse samples binary: {samples_bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing sparse samples metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = samples_bin_path.read_bytes()
    row_bytes = 4 * 4  # xyz + density, float32
    if len(raw) % row_bytes != 0:
        raise ValueError(f"sparse samples size mismatch for {samples_bin_path}: byte_count={len(raw)} not divisible by 16")

    count = len(raw) // row_bytes
    if count <= 0:
        raise ValueError(f"no samples found in {samples_bin_path}")

    if "count" in meta:
        try:
            meta_count = int(meta["count"])
        except Exception as exc:
            raise ValueError(f"invalid count in sparse samples metadata: {meta.get('count')}") from exc
        if meta_count != count:
            raise ValueError(
                f"sparse samples count mismatch: metadata count={meta_count}, binary count={count}"
            )

    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().view(count, 4)
    points = flat[:, :3].contiguous().clamp(-1.0, 1.0)
    density = flat[:, 3].contiguous().clamp(0.0, 1.0)
    return SparseSampleBank(points=points, density=density), meta


def split_sparse_sample_bank(
    bank: SparseSampleBank,
    val_ratio: float,
    split_seed: int,
) -> Tuple[SparseSampleBank, SparseSampleBank | None]:
    ratio = float(max(0.0, min(0.95, val_ratio)))
    count = int(bank.points.shape[0])
    if ratio <= 0.0 or count < 2:
        return bank, None

    val_count = int(round(count * ratio))
    val_count = max(1, min(val_count, count - 1))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(split_seed))
    perm = torch.randperm(count, generator=gen)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]

    train_bank = SparseSampleBank(
        points=bank.points.index_select(0, train_idx),
        density=bank.density.index_select(0, train_idx),
    )
    val_bank = SparseSampleBank(
        points=bank.points.index_select(0, val_idx),
        density=bank.density.index_select(0, val_idx),
    )
    return train_bank, val_bank


def sample_volume_trilinear(volume: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Trilinear sample of a dense volume defined on normalized [-1,1]^3.
    volume is [D,H,W], points is [N,3] in xyz.
    """
    depth, height, width = volume.shape
    p = points.clamp(-1.0, 1.0)

    fx = (p[:, 0] * 0.5 + 0.5) * (width - 1)
    fy = (p[:, 1] * 0.5 + 0.5) * (height - 1)
    fz = (p[:, 2] * 0.5 + 0.5) * (depth - 1)

    x0 = fx.floor().to(torch.long)
    y0 = fy.floor().to(torch.long)
    z0 = fz.floor().to(torch.long)
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)
    z1 = torch.clamp(z0 + 1, max=depth - 1)

    tx = fx - x0.to(fx.dtype)
    ty = fy - y0.to(fy.dtype)
    tz = fz - z0.to(fz.dtype)

    c000 = volume[z0, y0, x0]
    c100 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c110 = volume[z0, y1, x1]
    c001 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c011 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx

    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty

    return c0 * (1.0 - tz) + c1 * tz


@torch.no_grad()
def build_torus_coarse_density_grid(dim: int, device: torch.device) -> torch.Tensor:
    dim_i = int(dim)
    if dim_i < 8:
        raise ValueError(f"coarse grid dim must be >= 8, got {dim_i}")
    coords = torch.linspace(-1.0, 1.0, dim_i, device=device, dtype=torch.float32)
    z_idx, y_idx, x_idx = torch.meshgrid(coords, coords, coords, indexing="ij")
    points = torch.stack([x_idx.reshape(-1), y_idx.reshape(-1), z_idx.reshape(-1)], dim=-1)
    density = torus_density(points)
    return density.view(dim_i, dim_i, dim_i).contiguous()


@torch.no_grad()
def downsample_density_volume_to_grid(source_density_volume: torch.Tensor, dim: int) -> torch.Tensor:
    dim_i = int(dim)
    if dim_i < 8:
        raise ValueError(f"coarse grid dim must be >= 8, got {dim_i}")
    if source_density_volume.ndim != 3:
        raise ValueError(
            f"source density volume must be 3D [D,H,W], got shape={tuple(source_density_volume.shape)}"
        )
    coords = torch.linspace(-1.0, 1.0, dim_i, device=source_density_volume.device, dtype=torch.float32)
    z_idx, y_idx, x_idx = torch.meshgrid(coords, coords, coords, indexing="ij")
    points = torch.stack([x_idx.reshape(-1), y_idx.reshape(-1), z_idx.reshape(-1)], dim=-1)
    sampled_density = sample_volume_trilinear(source_density_volume, points).clamp(0.0, 1.0)
    return sampled_density.view(dim_i, dim_i, dim_i).contiguous()


@torch.no_grad()
def sample_detail_batch(
    batch_size: int,
    device: torch.device,
    target_source: str,
    target_volume: torch.Tensor | None,
    target_sample_bank: SparseSampleBank | None,
    coarse_density_grid: torch.Tensor,
    coarse_inside_threshold: float,
    importance_oversample: int,
    importance_max_rounds: int,
    sample_bank_inside_indices: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inside_thr = float(max(0.0, min(1.0, coarse_inside_threshold)))
    uniform_count = batch_size // 2
    inside_count = batch_size - uniform_count

    def sample_points_inside_grid(count: int) -> torch.Tensor:
        selected_chunks: List[torch.Tensor] = []
        selected_total = 0
        rounds = 0
        oversample = max(1, int(importance_oversample))
        max_rounds = max(1, int(importance_max_rounds))
        while selected_total < count and rounds < max_rounds:
            remaining = count - selected_total
            candidate_count = max(remaining * oversample, oversample)
            candidates = torch.rand(candidate_count, 3, device=device) * 2.0 - 1.0
            coarse = sample_volume_trilinear(coarse_density_grid, candidates).clamp(0.0, 1.0)
            keep = candidates[coarse > inside_thr]
            if keep.shape[0] > 0:
                take = min(remaining, int(keep.shape[0]))
                selected_chunks.append(keep[:take])
                selected_total += take
            rounds += 1
        if selected_total < count:
            selected_chunks.append(torch.rand(count - selected_total, 3, device=device) * 2.0 - 1.0)
        return torch.cat(selected_chunks, dim=0)

    if target_source in ("torus", "volume"):
        point_chunks: List[torch.Tensor] = []
        if uniform_count > 0:
            point_chunks.append(torch.rand(uniform_count, 3, device=device) * 2.0 - 1.0)
        if inside_count > 0:
            point_chunks.append(sample_points_inside_grid(inside_count))
        points = torch.cat(point_chunks, dim=0)
        perm = torch.randperm(batch_size, device=device)
        points = points[perm]
        if target_source == "torus":
            target_density = torus_density(points)
        else:
            if target_volume is None:
                raise ValueError("target_volume must be provided when target_source='volume'")
            target_density = sample_volume_trilinear(target_volume, points)
        coarse_density = sample_volume_trilinear(coarse_density_grid, points).clamp(0.0, 1.0)
        return points, target_density.unsqueeze(-1), coarse_density.unsqueeze(-1)

    if target_source == "samples":
        if target_sample_bank is None:
            raise ValueError("target_sample_bank must be provided when target_source='samples'")
        bank = target_sample_bank
        sample_count = int(bank.points.shape[0])
        if sample_count <= 0:
            raise ValueError("sparse samples bank is empty")

        point_chunks: List[torch.Tensor] = []
        density_chunks: List[torch.Tensor] = []
        if uniform_count > 0:
            idx_uniform = torch.randint(0, sample_count, (uniform_count,), dtype=torch.long)
            point_chunks.append(bank.points.index_select(0, idx_uniform).to(device=device))
            density_chunks.append(bank.density.index_select(0, idx_uniform).to(device=device))

        if inside_count > 0:
            inside_pool = sample_bank_inside_indices
            if inside_pool is not None and int(inside_pool.numel()) > 0:
                pick = torch.randint(
                    0,
                    int(inside_pool.shape[0]),
                    (inside_count,),
                    dtype=torch.long,
                    device=inside_pool.device,
                )
                idx_inside = inside_pool.index_select(0, pick)
                if idx_inside.device != bank.points.device:
                    idx_inside = idx_inside.to(device=bank.points.device)
            else:
                idx_inside = torch.randint(
                    0,
                    sample_count,
                    (inside_count,),
                    dtype=torch.long,
                    device=bank.points.device,
                )
            point_chunks.append(bank.points.index_select(0, idx_inside).to(device=device))
            density_chunks.append(bank.density.index_select(0, idx_inside).to(device=device))

        points = torch.cat(point_chunks, dim=0)
        target_density = torch.cat(density_chunks, dim=0)
        perm = torch.randperm(batch_size, device=device)
        points = points[perm]
        target_density = target_density[perm]
        coarse_density = sample_volume_trilinear(coarse_density_grid, points).clamp(0.0, 1.0)
        return points, target_density.unsqueeze(-1), coarse_density.unsqueeze(-1)

    raise ValueError(f"unsupported target_source '{target_source}'")


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


class DetailMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int] = (128, 128), output_dim: int = 1) -> None:
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


def gradient_regularization(pred_detail: torch.Tensor, points: torch.Tensor, coeff: float) -> torch.Tensor:
    c = float(max(0.0, coeff))
    if c <= 0.0:
        return pred_detail.new_zeros(())
    grad_outputs = torch.ones_like(pred_detail)
    grads = torch.autograd.grad(
        outputs=pred_detail,
        inputs=points,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return c * torch.mean(torch.sum(grads * grads, dim=-1))


def cosine_lr(step: int, max_steps: int, lr_start: float, lr_final: float, warmup_steps: int) -> float:
    if max_steps <= 1:
        return lr_final
    warm = max(0, int(warmup_steps))
    if warm > 0 and step <= warm:
        return lr_start * (float(step) / float(warm))
    denom = max(1, int(max_steps) - warm)
    t = float(max(0, step - warm)) / float(denom)
    t = max(0.0, min(1.0, t))
    c = 0.5 * (1.0 + math.cos(math.pi * t))
    return lr_final + (lr_start - lr_final) * c


def export_detail_density_model(
    model: DetailMLP,
    encoding_levels: int,
    ground_truth_info: Dict[str, Any],
    detail_amplitude: float,
    fade_start: float,
    fade_end: float,
    coarse_bin_name: str,
    coarse_meta_name: str,
    coarse_dims: Tuple[int, int, int],
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
        "training_mode": "detail_only_density",
        "encoding": {
            "type": "fourier",
            "levels": encoding_levels,
            "include_input": True,
            "frequency_base": 2.0,
            "uses_two_pi": False,
            "ordering": "input_xyz_then_per_level_sin_xyz_cos_xyz",
            "axis_order": "xyz",
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
        "reconstruction": {
            "type": "coarse_plus_detail_tanh",
            "detail_amplitude": float(max(1.0e-6, detail_amplitude)),
            "detail_fade_start": float(max(0.0, fade_start)),
            "detail_fade_end": float(max(fade_start + 1.0e-6, fade_end)),
        },
        "geometry": {
            "type": "coarse_density_volume",
            "bin": str(coarse_bin_name),
            "meta": str(coarse_meta_name),
            "dims": [int(coarse_dims[0]), int(coarse_dims[1]), int(coarse_dims[2])],
            "domain": "[-1,1]^3",
        },
        "ground_truth": ground_truth_info,
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
    extra_filenames: List[str] | None = None,
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

    if extra_filenames:
        for name in extra_filenames:
            rel = Path(name)
            src = source_dir / rel
            if not src.exists():
                raise FileNotFoundError(f"Missing extra export file for copy: {src}")
            dst = dest_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied extra artifact to viz: {dst}")


def train(args: argparse.Namespace) -> None:
    if str(args.training_mode).strip().lower() != "detail_only_density":
        raise ValueError("Legacy training modes were removed. Use --training-mode detail_only_density.")

    torch.manual_seed(args.seed)
    if args.fourier_levels < 1:
        raise ValueError("--fourier-levels must be >= 1.")
    if len(args.hidden_layers) != 2:
        raise ValueError("--hidden-layers must provide exactly 2 integers, e.g. --hidden-layers 128 128")
    hidden0 = int(args.hidden_layers[0])
    hidden1 = int(args.hidden_layers[1])
    if hidden0 <= 0 or hidden1 <= 0:
        raise ValueError(f"--hidden-layers values must be > 0, got {args.hidden_layers}")
    if args.detail_amplitude <= 0:
        raise ValueError(f"--detail-amplitude must be > 0, got {args.detail_amplitude}")
    if args.detail_grid_dim < 8:
        raise ValueError(f"--detail-grid-dim must be >= 8, got {args.detail_grid_dim}")
    if args.detail_inside_threshold < 0 or args.detail_inside_threshold > 1:
        raise ValueError(f"--detail-inside-threshold must be in [0,1], got {args.detail_inside_threshold}")
    if args.detail_shell_boost < 0:
        raise ValueError(f"--detail-shell-boost must be >= 0, got {args.detail_shell_boost}")
    if args.detail_smoothness_coeff < 0:
        raise ValueError(f"--detail-smoothness-coeff must be >= 0, got {args.detail_smoothness_coeff}")
    if args.detail_fade_start < 0 or args.detail_fade_start > 1:
        raise ValueError(f"--detail-fade-start must be in [0,1], got {args.detail_fade_start}")
    if args.detail_fade_end < 0 or args.detail_fade_end > 1:
        raise ValueError(f"--detail-fade-end must be in [0,1], got {args.detail_fade_end}")
    if args.detail_fade_end <= args.detail_fade_start:
        raise ValueError(
            f"--detail-fade-end must be > --detail-fade-start, got {args.detail_fade_end} <= {args.detail_fade_start}"
        )
    if args.iso_value < 0 or args.iso_value > 1:
        raise ValueError(f"--iso-value must be in [0,1], got {args.iso_value}")
    if args.iso_band <= 0:
        raise ValueError(f"--iso-band must be > 0, got {args.iso_band}")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError(f"--val-ratio must be in [0,1), got {args.val_ratio}")
    if args.lr <= 0 or args.lr_final <= 0:
        raise ValueError(f"--lr and --lr-final must be > 0, got lr={args.lr}, lr_final={args.lr_final}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print("Training mode: detail_only_density")
    print(
        "Model: "
        f"fourier_levels={args.fourier_levels}, hidden=[{hidden0}, {hidden1}], output=tanh(detail)"
    )
    print(
        "Detail reconstruction: "
        f"amplitude={args.detail_amplitude:.4f}, "
        f"inside_threshold={args.detail_inside_threshold:.4f}, "
        f"grid_dim={args.detail_grid_dim}, "
        f"fade=[{args.detail_fade_start:.3f}, {args.detail_fade_end:.3f}]"
    )
    print(
        "Sampling: "
        "uniform_ratio=0.50, inside_ratio=0.50, "
        f"importance_oversample={args.importance_oversample}, "
        f"importance_max_rounds={args.importance_max_rounds}"
    )
    print(
        "Loss: "
        f"shell_boost={args.detail_shell_boost:.3f}, "
        f"iso=[{args.iso_value:.3f}, band={args.iso_band:.3f}], "
        f"smoothness_coeff={args.detail_smoothness_coeff:.6f}"
    )

    target_source = str(args.target_source).strip().lower()
    target_volume: torch.Tensor | None = None
    train_sample_bank: SparseSampleBank | None = None
    eval_sample_bank: SparseSampleBank | None = None
    ground_truth_info: Dict[str, Any]

    if target_source == "torus":
        ground_truth_info = {
            "type": "smooth_torus",
            "major_radius": TORUS_MAJOR_RADIUS,
            "minor_radius": TORUS_MINOR_RADIUS,
            "sharpness": TORUS_SHARPNESS,
        }
        print("Target source: torus (analytic)")
    elif target_source == "volume":
        if not args.volume_bin:
            raise ValueError("--volume-bin is required when --target-source=volume")
        volume_bin_path = Path(args.volume_bin)
        volume_meta_path = Path(args.volume_meta) if args.volume_meta else None
        target_volume, volume_meta = load_dense_volume(volume_bin_path, volume_meta_path, device=device)
        print(
            f"Target source: volume ({volume_bin_path}), dims={list(target_volume.shape)}, "
            f"min={float(target_volume.min().item()):.6f}, max={float(target_volume.max().item()):.6f}, "
            f"mean={float(target_volume.mean().item()):.6f}"
        )
        ground_truth_info = {
            "type": "dense_volume",
            "bin_path": str(volume_bin_path),
            "meta_path": str(volume_meta_path if volume_meta_path is not None else volume_bin_path.with_suffix(".json")),
            "dims": list(target_volume.shape),
            "meta": volume_meta,
        }
    elif target_source == "samples":
        if not args.samples_bin:
            raise ValueError("--samples-bin is required when --target-source=samples")
        samples_bin_path = Path(args.samples_bin)
        samples_meta_path = Path(args.samples_meta) if args.samples_meta else None
        loaded_bank, samples_meta = load_sparse_samples(samples_bin_path=samples_bin_path, samples_meta_path=samples_meta_path)
        train_sample_bank, eval_sample_bank = split_sparse_sample_bank(
            bank=loaded_bank,
            val_ratio=args.val_ratio,
            split_seed=args.seed + 1337,
        )
        train_count = int(train_sample_bank.points.shape[0])
        eval_count = int(eval_sample_bank.points.shape[0]) if eval_sample_bank is not None else 0
        print(f"Target source: samples ({samples_bin_path}), train={train_count}, test={eval_count}")
        ground_truth_info = {
            "type": "sparse_samples",
            "bin_path": str(samples_bin_path),
            "meta_path": str(samples_meta_path if samples_meta_path is not None else samples_bin_path.with_suffix(".json")),
            "count": int(loaded_bank.points.shape[0]),
            "train_count": train_count,
            "test_count": eval_count,
            "val_ratio": float(args.val_ratio),
            "meta": samples_meta,
        }
    else:
        raise ValueError(f"--target-source must be one of ['torus', 'volume', 'samples'], got '{args.target_source}'")

    if target_source == "torus":
        coarse_density_grid = build_torus_coarse_density_grid(args.detail_grid_dim, device=device)
        coarse_source_info: Dict[str, Any] = {
            "source": "analytic_torus",
            "dims": [int(v) for v in coarse_density_grid.shape],
        }
    elif target_source == "volume" and target_volume is not None:
        coarse_density_grid = downsample_density_volume_to_grid(target_volume, args.detail_grid_dim)
        coarse_source_info = {
            "source": "target_volume_downsampled",
            "dims": [int(v) for v in coarse_density_grid.shape],
        }
    else:
        source_bin = Path(args.coarse_grid_source_bin)
        source_meta = Path(args.coarse_grid_source_meta) if args.coarse_grid_source_meta else None
        source_volume, source_meta_json = load_dense_volume(source_bin, source_meta, device=device, clamp_to_unit=True)
        coarse_density_grid = downsample_density_volume_to_grid(source_volume, args.detail_grid_dim)
        coarse_source_info = {
            "source": "explicit_source_volume",
            "source_bin": str(source_bin),
            "source_meta": str(source_meta if source_meta is not None else source_bin.with_suffix(".json")),
            "source_dims": [int(v) for v in source_volume.shape],
            "source_meta_json": source_meta_json,
            "dims": [int(v) for v in coarse_density_grid.shape],
        }

    print(
        f"Coarse grid: dims={list(coarse_density_grid.shape)}, "
        f"min={float(coarse_density_grid.min().item()):.6f}, "
        f"max={float(coarse_density_grid.max().item()):.6f}, "
        f"mean={float(coarse_density_grid.mean().item()):.6f}"
    )

    inside_thr = float(max(0.0, min(1.0, args.detail_inside_threshold)))
    train_inside_indices: torch.Tensor | None = None
    eval_inside_indices: torch.Tensor | None = None
    if target_source == "samples" and train_sample_bank is not None:
        train_coarse = sample_volume_trilinear(coarse_density_grid, train_sample_bank.points.to(device=device)).clamp(0.0, 1.0)
        train_inside_indices = torch.nonzero(train_coarse > inside_thr, as_tuple=False).squeeze(1).to(torch.long)
        if eval_sample_bank is not None:
            eval_coarse = sample_volume_trilinear(coarse_density_grid, eval_sample_bank.points.to(device=device)).clamp(0.0, 1.0)
            eval_inside_indices = torch.nonzero(eval_coarse > inside_thr, as_tuple=False).squeeze(1).to(torch.long)
        print(
            "Samples inside-coarse pool: "
            f"train_inside={int(train_inside_indices.shape[0])}, "
            f"test_inside={int(eval_inside_indices.shape[0]) if eval_inside_indices is not None else 0}"
        )

    encoder = FourierEncoding(levels=args.fourier_levels).to(device)
    model = DetailMLP(
        input_dim=encoder.output_dim,
        hidden_sizes=(hidden0, hidden1),
        output_dim=1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    amp = float(args.detail_amplitude)
    iso_center = float(max(0.0, min(1.0, args.iso_value)))
    iso_width = float(max(1.0e-4, args.iso_band))
    ema_density_mse = None
    stop_step = args.max_steps

    for step in range(1, args.max_steps + 1):
        lr_now = cosine_lr(
            step=step,
            max_steps=args.max_steps,
            lr_start=args.lr,
            lr_final=args.lr_final,
            warmup_steps=args.lr_warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        points, target_density, coarse_density = sample_detail_batch(
            batch_size=args.batch_size,
            device=device,
            target_source=target_source,
            target_volume=target_volume,
            target_sample_bank=train_sample_bank,
            coarse_density_grid=coarse_density_grid,
            coarse_inside_threshold=inside_thr,
            importance_oversample=args.importance_oversample,
            importance_max_rounds=args.importance_max_rounds,
            sample_bank_inside_indices=train_inside_indices,
        )
        points.requires_grad_(args.detail_smoothness_coeff > 0.0)
        pred_detail = torch.tanh(model(encoder(points)))

        target_detail = torch.clamp((target_density - coarse_density) / amp, -1.0, 1.0)
        target_detail = torch.where(coarse_density > inside_thr, target_detail, torch.zeros_like(target_detail))

        shell = torch.exp(-((target_density - iso_center) / iso_width).square())
        weight = 1.0 + float(args.detail_shell_boost) * shell
        detail_mse = torch.mean(weight * (pred_detail - target_detail).square())
        smoothness = gradient_regularization(pred_detail=pred_detail, points=points, coeff=args.detail_smoothness_coeff)
        objective = detail_mse + smoothness

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        pred_density = torch.clamp(coarse_density + amp * pred_detail.detach(), 0.0, 1.0)
        density_mse = torch.mean((pred_density - target_density).square())
        density_mse_val = float(density_mse.item())
        ema_density_mse = density_mse_val if ema_density_mse is None else (0.98 * ema_density_mse + 0.02 * density_mse_val)

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:6d}  obj={float(objective.item()):.8f}  detail_mse={float(detail_mse.item()):.8f}  "
                f"smooth={float(smoothness.item()):.8f}  density_mse={density_mse_val:.8f}  "
                f"ema_density_mse={ema_density_mse:.8f}  lr={lr_now:.6f}"
            )

        if step >= args.min_steps and ema_density_mse is not None and ema_density_mse <= args.target_mse:
            stop_step = step
            print(
                f"Early stop at step {step}: ema_density_mse={ema_density_mse:.8f} <= target_mse={args.target_mse:.8f}"
            )
            break

    with torch.no_grad():
        eval_sets: List[Tuple[str, str, SparseSampleBank | None, torch.Tensor | None]] = []
        if target_source == "samples":
            eval_sets.append(("train", "samples", train_sample_bank, train_inside_indices))
            if eval_sample_bank is not None:
                eval_sets.append(("test", "samples", eval_sample_bank, eval_inside_indices))
        else:
            eval_sets.append(("eval", target_source, None, None))

        for label, eval_source, eval_bank, eval_inside in eval_sets:
            eval_points, eval_target_density, eval_coarse_density = sample_detail_batch(
                batch_size=args.eval_samples,
                device=device,
                target_source=eval_source,
                target_volume=target_volume,
                target_sample_bank=eval_bank,
                coarse_density_grid=coarse_density_grid,
                coarse_inside_threshold=inside_thr,
                importance_oversample=args.importance_oversample,
                importance_max_rounds=args.importance_max_rounds,
                sample_bank_inside_indices=eval_inside,
            )
            eval_pred_detail = torch.tanh(model(encoder(eval_points)))
            eval_target_detail = torch.clamp((eval_target_density - eval_coarse_density) / amp, -1.0, 1.0)
            eval_target_detail = torch.where(
                eval_coarse_density > inside_thr, eval_target_detail, torch.zeros_like(eval_target_detail)
            )
            eval_shell = torch.exp(-((eval_target_density - iso_center) / iso_width).square())
            eval_weight = 1.0 + float(args.detail_shell_boost) * eval_shell
            eval_detail_mse = torch.mean(eval_weight * (eval_pred_detail - eval_target_detail).square())
            eval_pred_density = torch.clamp(eval_coarse_density + amp * eval_pred_detail, 0.0, 1.0)
            eval_density_mse = torch.mean((eval_pred_density - eval_target_density).square())
            print(
                f"final_{label}_detail_mse={float(eval_detail_mse.item()):.8f}  "
                f"final_{label}_density_mse={float(eval_density_mse.item()):.8f}  "
                f"(trained_steps={stop_step})"
            )

    output_dir_path = Path(args.output_dir).resolve()
    coarse_rel_bin = "detail_coarse_density.bin"
    coarse_rel_meta = "detail_coarse_density.json"
    save_dense_grid(
        grid=coarse_density_grid,
        grid_bin_path=output_dir_path / coarse_rel_bin,
        grid_meta_path=output_dir_path / coarse_rel_meta,
        field_name="detail_coarse_density",
        extra_meta={
            "representation": "density",
            "domain": "[-1,1]^3",
            **coarse_source_info,
        },
    )

    ground_truth_info["training_focus"] = {
        "type": "detail_only_density",
        "coarse_grid_dim": int(args.detail_grid_dim),
        "detail_amplitude": float(args.detail_amplitude),
        "inside_threshold": float(args.detail_inside_threshold),
        "sampling": {"uniform_ratio": 0.5, "inside_ratio": 0.5},
        "shell_boost": float(args.detail_shell_boost),
        "iso_value": float(args.iso_value),
        "iso_band": float(args.iso_band),
        "detail_fade_start": float(args.detail_fade_start),
        "detail_fade_end": float(args.detail_fade_end),
        "coarse_grid": {
            "bin": coarse_rel_bin,
            "meta": coarse_rel_meta,
            "dims": [int(v) for v in coarse_density_grid.shape],
        },
    }

    export_detail_density_model(
        model=model,
        encoding_levels=args.fourier_levels,
        ground_truth_info=ground_truth_info,
        detail_amplitude=args.detail_amplitude,
        fade_start=args.detail_fade_start,
        fade_end=args.detail_fade_end,
        coarse_bin_name=coarse_rel_bin,
        coarse_meta_name=coarse_rel_meta,
        coarse_dims=(
            int(coarse_density_grid.shape[0]),
            int(coarse_density_grid.shape[1]),
            int(coarse_density_grid.shape[2]),
        ),
        output_dir=output_dir_path,
        weights_name=args.weights_name,
        meta_name=args.meta_name,
    )

    if args.copy_to_viz:
        copy_exports_to_viz(
            source_dir=output_dir_path,
            dest_dir=Path(args.viz_mlp_dir),
            weights_name=args.weights_name,
            meta_name=args.meta_name,
            extra_filenames=[coarse_rel_bin, coarse_rel_meta],
        )


def build_arg_parser() -> argparse.ArgumentParser:
    train_dir = Path(__file__).resolve().parent
    repo_root = train_dir.parent
    default_viz_dir = repo_root / "viz" / "public" / "mlp"
    default_volume_bin = train_dir / "outputs" / "wdas_cloud_quarter_256.bin"
    default_samples_bin = train_dir / "outputs" / "wdas_cloud_quarter_samples_250000.bin"

    p = argparse.ArgumentParser(description="Train detail-only cloud density MLP and export flat weights.")
    p.add_argument(
        "--training-mode",
        type=str,
        default="detail_only_density",
        choices=["detail_only_density"],
        help="Only detail_only_density mode is supported.",
    )
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

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-final", type=float, default=1e-4)
    p.add_argument("--lr-warmup-steps", type=int, default=300)

    p.add_argument("--fourier-levels", type=int, default=8)
    p.add_argument(
        "--hidden-layers",
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=("H0", "H1"),
        help="Two hidden layer sizes for the detail MLP (default: 128 128).",
    )

    p.add_argument(
        "--target-source",
        type=str,
        default="samples",
        choices=["torus", "volume", "samples"],
        help="Ground-truth source. Defaults to sparse samples generated from the quarter-resolution Disney VDB.",
    )
    p.add_argument(
        "--volume-bin",
        type=str,
        default=str(default_volume_bin),
        help="Path to dense float32 volume binary [D,H,W] used when --target-source=volume.",
    )
    p.add_argument(
        "--volume-meta",
        type=str,
        default="",
        help="Optional path to JSON metadata with dims. Defaults to <volume-bin>.json.",
    )
    p.add_argument(
        "--samples-bin",
        type=str,
        default=str(default_samples_bin),
        help="Path to sparse xyz+density samples binary used when --target-source=samples.",
    )
    p.add_argument(
        "--samples-meta",
        type=str,
        default="",
        help="Optional path to sparse samples metadata JSON. Defaults to <samples-bin>.json.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Held-out test split ratio for sparse samples (0 disables split).",
    )

    p.add_argument(
        "--coarse-grid-source-bin",
        type=str,
        default=str(default_volume_bin),
        help="Source dense volume used to build coarse grid when target-source=samples.",
    )
    p.add_argument(
        "--coarse-grid-source-meta",
        type=str,
        default="",
        help="Optional metadata for --coarse-grid-source-bin. Defaults to <bin>.json.",
    )
    p.add_argument("--detail-grid-dim", type=int, default=64)
    p.add_argument("--detail-amplitude", type=float, default=0.15)
    p.add_argument("--detail-inside-threshold", type=float, default=0.01)
    p.add_argument("--detail-shell-boost", type=float, default=2.0)
    p.add_argument("--detail-smoothness-coeff", type=float, default=0.0)
    p.add_argument("--detail-fade-start", type=float, default=0.02)
    p.add_argument("--detail-fade-end", type=float, default=0.15)

    p.add_argument("--iso-value", type=float, default=0.10)
    p.add_argument("--iso-band", type=float, default=0.05)

    p.add_argument(
        "--importance-oversample",
        type=int,
        default=4,
        help="Candidate multiplier used during inside-coarse rejection sampling.",
    )
    p.add_argument(
        "--importance-max-rounds",
        type=int,
        default=8,
        help="Maximum rejection-sampling rounds when filling inside-coarse points.",
    )

    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
