#!/usr/bin/env python3
"""
Train a residual MLP to approximate a density field relative to a box baseline.

Targets supported:
  - torus (analytic smooth torus)
  - volume (dense preconverted volume sampled in normalized [-1,1]^3)
  - samples (sparse xyz+density samples generated directly from VDB)

This script exports:
  - FP32 flat weight blob in layout [W0][b0][W1][b1][W2][b2]
  - metadata JSON with layer sizes and offsets
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

BOX_HALF_EXTENTS: Tuple[float, float, float] = (0.6, 0.25, 0.6)
BOX_SHARPNESS: float = 14.0
TORUS_MAJOR_RADIUS: float = 0.55
TORUS_MINOR_RADIUS: float = 0.2
TORUS_SHARPNESS: float = 36.0


@dataclass
class SparseSampleBank:
    points: torch.Tensor  # [N,3] in normalized [-1,1]^3
    density: torch.Tensor  # [N] in [0,1]
    guided_indices: torch.Tensor  # [M] indices with density > importance threshold
    shell_indices: torch.Tensor  # [K] indices near target iso-density shell


def sdf_torus(
    points: torch.Tensor, major_radius: float = TORUS_MAJOR_RADIUS, minor_radius: float = TORUS_MINOR_RADIUS
) -> torch.Tensor:
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    qx = torch.sqrt(x * x + z * z + 1e-12) - major_radius
    qy = y
    return torch.sqrt(qx * qx + qy * qy + 1e-12) - minor_radius


def sdf_box(points: torch.Tensor, half_extents: Tuple[float, float, float] = BOX_HALF_EXTENTS) -> torch.Tensor:
    b = torch.tensor(half_extents, device=points.device, dtype=points.dtype)
    q = torch.abs(points) - b
    outside = torch.clamp(q, min=0.0)
    outside_len = torch.linalg.norm(outside, dim=-1)
    inside = torch.clamp(torch.amax(q, dim=-1), max=0.0)
    return outside_len + inside


def density_from_sdf(sdf: torch.Tensor, sharpness: float) -> torch.Tensor:
    return torch.sigmoid(-sharpness * sdf)


def torus_density(points: torch.Tensor) -> torch.Tensor:
    return density_from_sdf(sdf_torus(points), sharpness=TORUS_SHARPNESS)


def box_density(points: torch.Tensor) -> torch.Tensor:
    # Smooth procedural baseline.
    return density_from_sdf(sdf_box(points), sharpness=BOX_SHARPNESS)


def load_dense_volume(
    volume_bin_path: Path,
    volume_meta_path: Path | None,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    meta_path = volume_meta_path if volume_meta_path is not None else volume_bin_path.with_suffix(".json")
    if not volume_bin_path.exists():
        raise FileNotFoundError(f"missing volume binary: {volume_bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing volume metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dims = meta.get("dims")
    if not isinstance(dims, list) or len(dims) != 3:
        raise ValueError(f"volume metadata must contain dims=[D,H,W], got {dims}")
    try:
        depth, height, width = int(dims[0]), int(dims[1]), int(dims[2])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid dims values in {meta_path}: {dims}") from exc
    if depth <= 1 or height <= 1 or width <= 1:
        raise ValueError(f"dims must be > 1 in all axes, got {dims}")

    raw = volume_bin_path.read_bytes()
    expected_bytes = depth * height * width * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"volume byte size mismatch: expected {expected_bytes}, got {len(raw)} "
            f"(dims={dims}, dtype=float32)"
        )

    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    volume = flat.view(depth, height, width).to(device=device, dtype=torch.float32)
    volume = volume.clamp(0.0, 1.0)
    return volume, meta


def load_sparse_samples(
    samples_bin_path: Path,
    samples_meta_path: Path | None,
    importance_threshold: float,
    iso_value: float,
    iso_band: float,
) -> Tuple[SparseSampleBank, Dict[str, Any]]:
    meta_path = samples_meta_path if samples_meta_path is not None else samples_bin_path.with_suffix(".json")
    if not samples_bin_path.exists():
        raise FileNotFoundError(f"missing sparse samples binary: {samples_bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"missing sparse samples metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = samples_bin_path.read_bytes()
    if len(raw) % 16 != 0:
        raise ValueError(
            f"sparse samples byte size must be divisible by 16 (xyz+density float32), got {len(raw)}"
        )

    count = len(raw) // 16
    meta_count = meta.get("count")
    if meta_count is not None:
        try:
            meta_count_int = int(meta_count)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid count in sparse samples metadata: {meta_count}") from exc
        if meta_count_int != count:
            raise ValueError(
                f"sparse samples count mismatch: metadata count={meta_count_int}, binary count={count}"
            )

    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().view(count, 4)
    points = flat[:, 0:3].contiguous().clamp(-1.0, 1.0)
    density = flat[:, 3].contiguous().clamp(0.0, 1.0)
    bank = build_sparse_sample_bank(
        points=points,
        density=density,
        importance_threshold=importance_threshold,
        iso_value=iso_value,
        iso_band=iso_band,
    )
    return bank, meta


def build_sparse_sample_bank(
    points: torch.Tensor,
    density: torch.Tensor,
    importance_threshold: float,
    iso_value: float,
    iso_band: float,
) -> SparseSampleBank:
    guided_indices = torch.nonzero(density > max(0.0, float(importance_threshold)), as_tuple=False).squeeze(1)
    iso_center = float(max(0.0, min(1.0, iso_value)))
    iso_width = max(1.0e-6, float(iso_band))
    shell_mask = torch.abs(density - iso_center) <= iso_width
    shell_indices = torch.nonzero(shell_mask, as_tuple=False).squeeze(1)
    return SparseSampleBank(
        points=points,
        density=density,
        guided_indices=guided_indices.to(torch.long),
        shell_indices=shell_indices.to(torch.long),
    )


def split_sparse_sample_bank(
    bank: SparseSampleBank,
    val_ratio: float,
    split_seed: int,
    importance_threshold: float,
    iso_value: float,
    iso_band: float,
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

    train_points = bank.points.index_select(0, train_idx)
    train_density = bank.density.index_select(0, train_idx)
    val_points = bank.points.index_select(0, val_idx)
    val_density = bank.density.index_select(0, val_idx)

    train_bank = build_sparse_sample_bank(
        points=train_points,
        density=train_density,
        importance_threshold=importance_threshold,
        iso_value=iso_value,
        iso_band=iso_band,
    )
    val_bank = build_sparse_sample_bank(
        points=val_points,
        density=val_density,
        importance_threshold=importance_threshold,
        iso_value=iso_value,
        iso_band=iso_band,
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
def sample_batch(
    batch_size: int,
    device: torch.device,
    target_source: str,
    target_volume: torch.Tensor | None,
    target_sample_bank: SparseSampleBank | None,
    baseline_scale: float,
    importance_ratio: float,
    importance_threshold: float,
    importance_oversample: int,
    importance_max_rounds: int,
    iso_shell_ratio: float,
    iso_value: float,
    iso_band: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_source == "torus":
        points = torch.rand(batch_size, 3, device=device) * 2.0 - 1.0
        gt = torus_density(points)
    elif target_source == "volume":
        if target_volume is None:
            raise ValueError("target_volume must be provided when target_source='volume'")

        guided_ratio = float(max(0.0, min(1.0, importance_ratio)))
        shell_ratio = float(max(0.0, min(1.0, iso_shell_ratio)))
        shell_count = int(batch_size * shell_ratio)
        guided_count = int(batch_size * guided_ratio)
        if shell_count + guided_count > batch_size:
            guided_count = max(0, batch_size - shell_count)
        uniform_count = batch_size - guided_count - shell_count

        point_chunks: List[torch.Tensor] = []
        if uniform_count > 0:
            point_chunks.append(torch.rand(uniform_count, 3, device=device) * 2.0 - 1.0)

        if guided_count > 0:
            selected_chunks: List[torch.Tensor] = []
            selected_total = 0
            rounds = 0
            oversample = max(1, int(importance_oversample))
            max_rounds = max(1, int(importance_max_rounds))
            threshold = float(max(0.0, importance_threshold))

            while selected_total < guided_count and rounds < max_rounds:
                remaining = guided_count - selected_total
                candidate_count = max(remaining * oversample, oversample)
                candidates = torch.rand(candidate_count, 3, device=device) * 2.0 - 1.0
                candidate_gt = sample_volume_trilinear(target_volume, candidates)
                keep = candidates[candidate_gt > threshold]
                if keep.shape[0] > 0:
                    take = min(remaining, keep.shape[0])
                    selected_chunks.append(keep[:take])
                    selected_total += take
                rounds += 1

            if selected_total < guided_count:
                fill = torch.rand(guided_count - selected_total, 3, device=device) * 2.0 - 1.0
                selected_chunks.append(fill)

            point_chunks.append(torch.cat(selected_chunks, dim=0))

        if shell_count > 0:
            selected_chunks = []
            selected_total = 0
            rounds = 0
            oversample = max(1, int(importance_oversample))
            max_rounds = max(1, int(importance_max_rounds))
            iso_center = float(max(0.0, min(1.0, iso_value)))
            iso_width = max(1.0e-6, float(iso_band))

            while selected_total < shell_count and rounds < max_rounds:
                remaining = shell_count - selected_total
                candidate_count = max(remaining * oversample, oversample)
                candidates = torch.rand(candidate_count, 3, device=device) * 2.0 - 1.0
                candidate_gt = sample_volume_trilinear(target_volume, candidates)
                keep = candidates[torch.abs(candidate_gt - iso_center) <= iso_width]
                if keep.shape[0] > 0:
                    take = min(remaining, keep.shape[0])
                    selected_chunks.append(keep[:take])
                    selected_total += take
                rounds += 1

            if selected_total < shell_count:
                fill = torch.rand(shell_count - selected_total, 3, device=device) * 2.0 - 1.0
                selected_chunks.append(fill)

            point_chunks.append(torch.cat(selected_chunks, dim=0))

        points = torch.cat(point_chunks, dim=0)
        if points.shape[0] != batch_size:
            raise RuntimeError(f"sampled unexpected point count {points.shape[0]}, expected {batch_size}")

        # Keep stochastic ordering to avoid chunk-order bias between uniform and guided subsets.
        perm = torch.randperm(batch_size, device=device)
        points = points[perm]
        gt = sample_volume_trilinear(target_volume, points)
    elif target_source == "samples":
        if target_sample_bank is None:
            raise ValueError("target_sample_bank must be provided when target_source='samples'")

        bank = target_sample_bank
        sample_count = int(bank.points.shape[0])
        if sample_count <= 0:
            raise ValueError("sparse samples bank is empty")

        guided_ratio = float(max(0.0, min(1.0, importance_ratio)))
        shell_ratio = float(max(0.0, min(1.0, iso_shell_ratio)))
        shell_count = int(batch_size * shell_ratio)
        guided_count = int(batch_size * guided_ratio)
        if shell_count + guided_count > batch_size:
            guided_count = max(0, batch_size - shell_count)
        uniform_count = batch_size - guided_count - shell_count

        point_chunks: List[torch.Tensor] = []
        gt_chunks: List[torch.Tensor] = []

        if uniform_count > 0:
            idx_uniform = torch.randint(0, sample_count, (uniform_count,), dtype=torch.long)
            point_chunks.append(bank.points.index_select(0, idx_uniform).to(device=device))
            gt_chunks.append(bank.density.index_select(0, idx_uniform).to(device=device))

        if guided_count > 0:
            guided_pool = bank.guided_indices
            if guided_pool.numel() > 0:
                pick = torch.randint(0, int(guided_pool.shape[0]), (guided_count,), dtype=torch.long)
                idx_guided = guided_pool.index_select(0, pick)
            else:
                idx_guided = torch.randint(0, sample_count, (guided_count,), dtype=torch.long)
            point_chunks.append(bank.points.index_select(0, idx_guided).to(device=device))
            gt_chunks.append(bank.density.index_select(0, idx_guided).to(device=device))

        if shell_count > 0:
            shell_pool = bank.shell_indices
            if shell_pool.numel() > 0:
                pick = torch.randint(0, int(shell_pool.shape[0]), (shell_count,), dtype=torch.long)
                idx_shell = shell_pool.index_select(0, pick)
            else:
                idx_shell = torch.randint(0, sample_count, (shell_count,), dtype=torch.long)
            point_chunks.append(bank.points.index_select(0, idx_shell).to(device=device))
            gt_chunks.append(bank.density.index_select(0, idx_shell).to(device=device))

        points = torch.cat(point_chunks, dim=0)
        gt = torch.cat(gt_chunks, dim=0)
        if points.shape[0] != batch_size:
            raise RuntimeError(f"sampled unexpected point count {points.shape[0]}, expected {batch_size}")

        # Keep stochastic ordering to avoid chunk-order bias between uniform and guided subsets.
        perm = torch.randperm(batch_size, device=device)
        points = points[perm]
        gt = gt[perm]
    else:
        raise ValueError(f"unsupported target_source '{target_source}'")
    baseline = baseline_scale * box_density(points)
    residual = (gt - baseline).unsqueeze(-1)
    return points, residual, gt.unsqueeze(-1)


def residual_loss_terms(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    target_density: torch.Tensor,
    abs_weight_scale: float,
    l1_coeff: float,
    iso_value: float,
    iso_band: float,
    iso_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred_residual, target_residual)
    weight = 1.0 + max(0.0, float(abs_weight_scale)) * torch.abs(target_residual)
    if iso_loss_weight > 0.0:
        band = max(1.0e-4, float(iso_band))
        iso_center = float(max(0.0, min(1.0, iso_value)))
        shell = torch.exp(-((target_density - iso_center) / band).square())
        weight = weight * (1.0 + float(iso_loss_weight) * shell)
    weighted_mse = (weight * (pred_residual - target_residual).square()).mean()
    l1 = F.l1_loss(pred_residual, target_residual)
    weighted_l1 = (weight * torch.abs(pred_residual - target_residual)).mean()
    objective = weighted_mse + max(0.0, float(l1_coeff)) * weighted_l1
    return objective, mse, l1


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


def empty_space_leak_penalty(
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    empty_density_threshold: float,
    coeff: float,
) -> torch.Tensor:
    if coeff <= 0.0:
        return pred_density.new_zeros(())
    thr = float(max(0.0, min(1.0, empty_density_threshold)))
    empty_mask = (target_density <= thr).to(pred_density.dtype)
    empty_count = torch.sum(empty_mask)
    if float(empty_count.item()) < 1.0:
        return pred_density.new_zeros(())
    leak = torch.relu(pred_density - thr)
    return float(coeff) * torch.sum(empty_mask * leak.square()) / empty_count


def gradient_regularization(
    pred_residual: torch.Tensor,
    points: torch.Tensor,
    coeff: float,
) -> torch.Tensor:
    if coeff <= 0.0:
        return pred_residual.new_zeros(())
    grads = torch.autograd.grad(pred_residual.sum(), points, create_graph=True)[0]
    return float(coeff) * grads.square().mean()


def export_model(
    model: ResidualMLP,
    encoding_levels: int,
    ground_truth_info: Dict[str, Any],
    baseline_scale: float,
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
        "baseline": {
            "type": "smooth_box",
            "half_extents": list(BOX_HALF_EXTENTS),
            "sharpness": BOX_SHARPNESS,
            "scale": baseline_scale,
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
    if len(args.hidden_layers) != 2:
        raise ValueError("--hidden-layers must provide exactly 2 integers, e.g. --hidden-layers 192 192")
    hidden0 = int(args.hidden_layers[0])
    hidden1 = int(args.hidden_layers[1])
    if hidden0 <= 0 or hidden1 <= 0:
        raise ValueError(f"--hidden-layers values must be > 0, got {args.hidden_layers}")
    if args.baseline_scale <= 0:
        raise ValueError(f"--baseline-scale must be > 0, got {args.baseline_scale}")
    if args.grad_reg_coeff < 0:
        raise ValueError(f"--grad-reg-coeff must be >= 0, got {args.grad_reg_coeff}")
    if args.loss_weight_abs_scale < 0:
        raise ValueError(f"--loss-weight-abs-scale must be >= 0, got {args.loss_weight_abs_scale}")
    if args.loss_l1_coeff < 0:
        raise ValueError(f"--loss-l1-coeff must be >= 0, got {args.loss_l1_coeff}")
    if args.lr <= 0:
        raise ValueError(f"--lr must be > 0, got {args.lr}")
    if args.lr_final <= 0:
        raise ValueError(f"--lr-final must be > 0, got {args.lr_final}")
    if args.lr_warmup_steps < 0:
        raise ValueError(f"--lr-warmup-steps must be >= 0, got {args.lr_warmup_steps}")
    if args.loss_ramp_steps < 1:
        raise ValueError(f"--loss-ramp-steps must be >= 1, got {args.loss_ramp_steps}")
    if args.iso_shell_ratio < 0 or args.iso_shell_ratio > 1:
        raise ValueError(f"--iso-shell-ratio must be in [0,1], got {args.iso_shell_ratio}")
    if args.iso_value < 0 or args.iso_value > 1:
        raise ValueError(f"--iso-value must be in [0,1], got {args.iso_value}")
    if args.iso_band <= 0:
        raise ValueError(f"--iso-band must be > 0, got {args.iso_band}")
    if args.loss_iso_weight < 0:
        raise ValueError(f"--loss-iso-weight must be >= 0, got {args.loss_iso_weight}")
    if args.empty_density_threshold < 0 or args.empty_density_threshold > 1:
        raise ValueError(
            f"--empty-density-threshold must be in [0,1], got {args.empty_density_threshold}"
        )
    if args.loss_empty_space_weight < 0:
        raise ValueError(
            f"--loss-empty-space-weight must be >= 0, got {args.loss_empty_space_weight}"
        )
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError(f"--val-ratio must be in [0,1), got {args.val_ratio}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(
        "Encoding: include_input=True, levels="
        f"{args.fourier_levels}, order=input_xyz_then_per_level_sin_xyz_cos_xyz, freq=2**i (no 2pi)"
    )
    print(f"MLP hidden layers: [{hidden0}, {hidden1}]")
    print(
        "Sampling: "
        f"importance_ratio={args.importance_ratio:.2f}, "
        f"importance_threshold={args.importance_threshold:.3f}, "
        f"iso_shell_ratio={args.iso_shell_ratio:.2f}, "
        f"iso_value={args.iso_value:.3f}, "
        f"iso_band={args.iso_band:.3f}, "
        f"importance_oversample={args.importance_oversample}, "
        f"importance_max_rounds={args.importance_max_rounds}"
    )
    print(
        "Loss: "
        f"weighted_mse_abs_scale={args.loss_weight_abs_scale:.3f}, "
        f"l1_coeff={args.loss_l1_coeff:.3f}, "
        f"iso_weight={args.loss_iso_weight:.3f}, "
        f"empty_thr={args.empty_density_threshold:.3f}, "
        f"empty_weight={args.loss_empty_space_weight:.3f}, "
        f"grad_reg_coeff={args.grad_reg_coeff:.6f}"
    )
    print(
        "Optimizer: "
        f"lr_start={args.lr:.6f}, lr_final={args.lr_final:.6f}, "
        f"lr_warmup_steps={args.lr_warmup_steps}, loss_ramp_steps={args.loss_ramp_steps}"
    )
    print(f"Baseline: smooth_box scale={args.baseline_scale:.3f}, sharpness={BOX_SHARPNESS:.3f}")

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
        volume_min = float(target_volume.min().item())
        volume_max = float(target_volume.max().item())
        volume_mean = float(target_volume.mean().item())
        print(
            f"Target source: volume ({volume_bin_path}), "
            f"dims={list(target_volume.shape)}, min={volume_min:.6f}, max={volume_max:.6f}, mean={volume_mean:.6f}"
        )
        ground_truth_info = {
            "type": "dense_volume",
            "bin_path": str(volume_bin_path),
            "meta_path": str(volume_meta_path if volume_meta_path is not None else volume_bin_path.with_suffix('.json')),
            "dims": list(target_volume.shape),
            "meta": volume_meta,
        }
    elif target_source == "samples":
        if not args.samples_bin:
            raise ValueError("--samples-bin is required when --target-source=samples")
        samples_bin_path = Path(args.samples_bin)
        samples_meta_path = Path(args.samples_meta) if args.samples_meta else None
        loaded_bank, samples_meta = load_sparse_samples(
            samples_bin_path=samples_bin_path,
            samples_meta_path=samples_meta_path,
            importance_threshold=args.importance_threshold,
            iso_value=args.iso_value,
            iso_band=args.iso_band,
        )
        sample_min = float(loaded_bank.density.min().item())
        sample_max = float(loaded_bank.density.max().item())
        sample_mean = float(loaded_bank.density.mean().item())
        sample_count = int(loaded_bank.points.shape[0])
        guided_count = int(loaded_bank.guided_indices.shape[0])
        shell_count = int(loaded_bank.shell_indices.shape[0])
        train_sample_bank, eval_sample_bank = split_sparse_sample_bank(
            bank=loaded_bank,
            val_ratio=args.val_ratio,
            split_seed=args.seed + 1337,
            importance_threshold=args.importance_threshold,
            iso_value=args.iso_value,
            iso_band=args.iso_band,
        )
        train_count = int(train_sample_bank.points.shape[0])
        if eval_sample_bank is not None:
            eval_count = int(eval_sample_bank.points.shape[0])
            print(
                f"Sample split: train={train_count}, test={eval_count}, val_ratio={args.val_ratio:.3f}"
            )
        else:
            print(f"Sample split: train={train_count}, test=none")
        print(
            f"Target source: samples ({samples_bin_path}), "
            f"count={sample_count}, guided_count={guided_count}, shell_count={shell_count}, "
            f"min={sample_min:.6f}, max={sample_max:.6f}, mean={sample_mean:.6f}"
        )
        ground_truth_info = {
            "type": "sparse_samples",
            "bin_path": str(samples_bin_path),
            "meta_path": str(samples_meta_path if samples_meta_path is not None else samples_bin_path.with_suffix('.json')),
            "count": sample_count,
            "guided_count": guided_count,
            "shell_count": shell_count,
            "train_count": train_count,
            "test_count": int(eval_sample_bank.points.shape[0]) if eval_sample_bank is not None else 0,
            "val_ratio": float(args.val_ratio),
            "meta": samples_meta,
        }
    else:
        raise ValueError(f"--target-source must be one of ['torus', 'volume', 'samples'], got '{args.target_source}'")

    encoder = FourierEncoding(levels=args.fourier_levels).to(device)
    model = ResidualMLP(input_dim=encoder.output_dim, hidden_sizes=(hidden0, hidden1), output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ema_mse = None
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

        ramp_denom = max(1, int(args.loss_ramp_steps))
        loss_ramp = min(1.0, float(step) / float(ramp_denom))
        abs_weight_scale_eff = args.loss_weight_abs_scale * loss_ramp
        iso_weight_eff = args.loss_iso_weight * loss_ramp
        empty_weight_eff = args.loss_empty_space_weight * loss_ramp

        points, target_residual, target_density = sample_batch(
            batch_size=args.batch_size,
            device=device,
            target_source=target_source,
            target_volume=target_volume,
            target_sample_bank=train_sample_bank,
            baseline_scale=args.baseline_scale,
            importance_ratio=args.importance_ratio,
            importance_threshold=args.importance_threshold,
            importance_oversample=args.importance_oversample,
            importance_max_rounds=args.importance_max_rounds,
            iso_shell_ratio=args.iso_shell_ratio,
            iso_value=args.iso_value,
            iso_band=args.iso_band,
        )
        points.requires_grad_(args.grad_reg_coeff > 0.0)
        encoded = encoder(points)
        pred_residual = model(encoded)
        baseline_from_targets = target_density - target_residual
        pred_density = torch.clamp(baseline_from_targets + pred_residual, 0.0, 1.0)

        data_objective, mse, l1 = residual_loss_terms(
            pred_residual=pred_residual,
            target_residual=target_residual,
            target_density=target_density,
            abs_weight_scale=abs_weight_scale_eff,
            l1_coeff=args.loss_l1_coeff,
            iso_value=args.iso_value,
            iso_band=args.iso_band,
            iso_loss_weight=iso_weight_eff,
        )
        grad_reg = gradient_regularization(
            pred_residual=pred_residual,
            points=points,
            coeff=args.grad_reg_coeff,
        )
        empty_leak = empty_space_leak_penalty(
            pred_density=pred_density,
            target_density=target_density,
            empty_density_threshold=args.empty_density_threshold,
            coeff=empty_weight_eff,
        )
        objective = data_objective + grad_reg + empty_leak

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        objective_val = float(objective.item())
        data_objective_val = float(data_objective.item())
        grad_reg_val = float(grad_reg.item())
        empty_leak_val = float(empty_leak.item())
        mse_val = float(mse.item())
        l1_val = float(l1.item())
        ema_mse = mse_val if ema_mse is None else (0.98 * ema_mse + 0.02 * mse_val)

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:6d}  obj={objective_val:.8f}  data_obj={data_objective_val:.8f}  "
                f"mse={mse_val:.8f}  l1={l1_val:.8f}  grad={grad_reg_val:.8f}  "
                f"empty={empty_leak_val:.8f}  ema_mse={ema_mse:.8f}  "
                f"lr={lr_now:.6f}  ramp={loss_ramp:.3f}"
            )

        if step >= args.min_steps and ema_mse is not None and ema_mse <= args.target_mse:
            stop_step = step
            print(f"Early stop at step {step}: ema_mse={ema_mse:.8f} <= target_mse={args.target_mse:.8f}")
            break

    with torch.no_grad():
        eval_sets: List[Tuple[str, str, SparseSampleBank | None]] = []
        if target_source == "samples":
            eval_sets.append(("train", "samples", train_sample_bank))
            if eval_sample_bank is not None:
                eval_sets.append(("test", "samples", eval_sample_bank))
        else:
            eval_sets.append(("eval", target_source, None))

        for label, eval_source, eval_bank in eval_sets:
            points, target_residual, target_density = sample_batch(
                batch_size=args.eval_samples,
                device=device,
                target_source=eval_source,
                target_volume=target_volume,
                target_sample_bank=eval_bank,
                baseline_scale=args.baseline_scale,
                importance_ratio=args.importance_ratio,
                importance_threshold=args.importance_threshold,
                importance_oversample=args.importance_oversample,
                importance_max_rounds=args.importance_max_rounds,
                iso_shell_ratio=args.iso_shell_ratio,
                iso_value=args.iso_value,
                iso_band=args.iso_band,
            )
            eval_pred = model(encoder(points))
            eval_baseline_from_targets = target_density - target_residual
            eval_pred_density = torch.clamp(eval_baseline_from_targets + eval_pred, 0.0, 1.0)
            eval_objective, eval_mse, eval_l1 = residual_loss_terms(
                pred_residual=eval_pred,
                target_residual=target_residual,
                target_density=target_density,
                abs_weight_scale=args.loss_weight_abs_scale,
                l1_coeff=args.loss_l1_coeff,
                iso_value=args.iso_value,
                iso_band=args.iso_band,
                iso_loss_weight=args.loss_iso_weight,
            )
            eval_empty_leak = empty_space_leak_penalty(
                pred_density=eval_pred_density,
                target_density=target_density,
                empty_density_threshold=args.empty_density_threshold,
                coeff=args.loss_empty_space_weight,
            )
            print(
                f"final_{label}_obj={float(eval_objective.item()):.8f}  "
                f"final_{label}_mse={float(eval_mse.item()):.8f}  "
                f"final_{label}_l1={float(eval_l1.item()):.8f}  "
                f"final_{label}_empty={float(eval_empty_leak.item()):.8f}  "
                f"(trained_steps={stop_step})"
            )

    ground_truth_info["training_focus"] = {
        "type": "iso_surface_shell",
        "iso_value": float(args.iso_value),
        "iso_band": float(args.iso_band),
        "iso_shell_ratio": float(args.iso_shell_ratio),
        "iso_loss_weight": float(args.loss_iso_weight),
        "empty_density_threshold": float(args.empty_density_threshold),
        "empty_space_loss_weight": float(args.loss_empty_space_weight),
    }

    export_model(
        model=model,
        encoding_levels=args.fourier_levels,
        ground_truth_info=ground_truth_info,
        baseline_scale=args.baseline_scale,
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
    train_dir = Path(__file__).resolve().parent
    repo_root = train_dir.parent
    default_viz_dir = repo_root / "viz" / "public" / "mlp"
    default_volume_bin = train_dir / "outputs" / "wdas_cloud_quarter_256.bin"
    default_samples_bin = train_dir / "outputs" / "wdas_cloud_quarter_samples_4000000.bin"
    p = argparse.ArgumentParser(
        description="Train residual MLP vs box baseline (torus, dense volume, or sparse VDB samples) and export flat weights."
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
    p.add_argument(
        "--baseline-scale",
        type=float,
        default=0.8,
        help="Scale factor applied to smooth box baseline before residual target computation.",
    )
    p.add_argument(
        "--grad-reg-coeff",
        type=float,
        default=2e-4,
        help="Coefficient for gradient regularization on d(pred_residual)/d(points). Set 0 to disable.",
    )
    p.add_argument(
        "--loss-weight-abs-scale",
        type=float,
        default=2.0,
        help="Weighted-MSE scale: weight = 1 + scale * abs(target_residual). Set 0 for unweighted MSE.",
    )
    p.add_argument(
        "--loss-l1-coeff",
        type=float,
        default=0.2,
        help="Additional L1 term coefficient in objective. Set 0 to disable.",
    )
    p.add_argument(
        "--loss-iso-weight",
        type=float,
        default=3.0,
        help="Extra weight scale for residual error near iso-density shell.",
    )
    p.add_argument(
        "--loss-empty-space-weight",
        type=float,
        default=1.0,
        help="Penalty weight for predicted density leaking into target-empty regions.",
    )
    p.add_argument(
        "--empty-density-threshold",
        type=float,
        default=0.02,
        help="Target density threshold below which points are considered empty for leak penalty.",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--lr-final",
        type=float,
        default=1e-4,
        help="Final learning rate for cosine decay schedule.",
    )
    p.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=300,
        help="Linear LR warmup steps before cosine decay.",
    )
    p.add_argument(
        "--loss-ramp-steps",
        type=int,
        default=3000,
        help="Ramp-up steps for weighted/iso/empty loss scales.",
    )
    p.add_argument("--fourier-levels", type=int, default=6)
    p.add_argument(
        "--hidden-layers",
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=("H0", "H1"),
        help="Two hidden layer sizes for the residual MLP (default: 128 128).",
    )
    p.add_argument(
        "--target-source",
        type=str,
        default="samples",
        choices=["torus", "volume", "samples"],
        help="Ground-truth source. Defaults to sparse samples generated directly from the quarter-resolution Disney VDB.",
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
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument(
        "--importance-ratio",
        type=float,
        default=0.5,
        help="Fraction of points sampled from non-empty regions for volume targets (0..1).",
    )
    p.add_argument(
        "--importance-threshold",
        type=float,
        default=0.05,
        help="Density threshold used to accept guided samples for volume targets.",
    )
    p.add_argument(
        "--importance-oversample",
        type=int,
        default=4,
        help="Candidate multiplier used during guided sample rejection sampling.",
    )
    p.add_argument(
        "--importance-max-rounds",
        type=int,
        default=8,
        help="Maximum rejection-sampling rounds when filling guided points.",
    )
    p.add_argument(
        "--iso-shell-ratio",
        type=float,
        default=0.35,
        help="Fraction of each batch sampled from iso-density shell candidates.",
    )
    p.add_argument(
        "--iso-value",
        type=float,
        default=0.10,
        help="Iso-density center used for boundary-focused sampling/loss.",
    )
    p.add_argument(
        "--iso-band",
        type=float,
        default=0.05,
        help="Half-width of iso shell for boundary-focused sampling/loss.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Held-out test split ratio for sparse samples (0 disables split).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
