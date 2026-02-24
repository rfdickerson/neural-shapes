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
    guided_indices = torch.nonzero(density > max(0.0, float(importance_threshold)), as_tuple=False).squeeze(1)
    bank = SparseSampleBank(points=points, density=density, guided_indices=guided_indices.to(torch.long))
    return bank, meta


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_source == "torus":
        points = torch.rand(batch_size, 3, device=device) * 2.0 - 1.0
        gt = torus_density(points)
    elif target_source == "volume":
        if target_volume is None:
            raise ValueError("target_volume must be provided when target_source='volume'")

        guided_ratio = float(max(0.0, min(1.0, importance_ratio)))
        guided_count = int(batch_size * guided_ratio)
        uniform_count = batch_size - guided_count

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
        guided_count = int(batch_size * guided_ratio)
        uniform_count = batch_size - guided_count

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
    return points, residual


def residual_loss_terms(
    pred_residual: torch.Tensor,
    target_residual: torch.Tensor,
    abs_weight_scale: float,
    l1_coeff: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(pred_residual, target_residual)
    weight = 1.0 + max(0.0, float(abs_weight_scale)) * torch.abs(target_residual)
    weighted_mse = (weight * (pred_residual - target_residual).square()).mean()
    l1 = F.l1_loss(pred_residual, target_residual)
    objective = weighted_mse + max(0.0, float(l1_coeff)) * l1
    return objective, mse, l1


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
        f"importance_oversample={args.importance_oversample}, "
        f"importance_max_rounds={args.importance_max_rounds}"
    )
    print(
        "Loss: "
        f"weighted_mse_abs_scale={args.loss_weight_abs_scale:.3f}, "
        f"l1_coeff={args.loss_l1_coeff:.3f}, "
        f"grad_reg_coeff={args.grad_reg_coeff:.6f}"
    )
    print(f"Baseline: smooth_box scale={args.baseline_scale:.3f}, sharpness={BOX_SHARPNESS:.3f}")

    target_source = str(args.target_source).strip().lower()
    target_volume: torch.Tensor | None = None
    target_sample_bank: SparseSampleBank | None = None
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
        target_sample_bank, samples_meta = load_sparse_samples(
            samples_bin_path=samples_bin_path,
            samples_meta_path=samples_meta_path,
            importance_threshold=args.importance_threshold,
        )
        sample_min = float(target_sample_bank.density.min().item())
        sample_max = float(target_sample_bank.density.max().item())
        sample_mean = float(target_sample_bank.density.mean().item())
        sample_count = int(target_sample_bank.points.shape[0])
        guided_count = int(target_sample_bank.guided_indices.shape[0])
        print(
            f"Target source: samples ({samples_bin_path}), "
            f"count={sample_count}, guided_count={guided_count}, "
            f"min={sample_min:.6f}, max={sample_max:.6f}, mean={sample_mean:.6f}"
        )
        ground_truth_info = {
            "type": "sparse_samples",
            "bin_path": str(samples_bin_path),
            "meta_path": str(samples_meta_path if samples_meta_path is not None else samples_bin_path.with_suffix('.json')),
            "count": sample_count,
            "guided_count": guided_count,
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
        points, target_residual = sample_batch(
            batch_size=args.batch_size,
            device=device,
            target_source=target_source,
            target_volume=target_volume,
            target_sample_bank=target_sample_bank,
            baseline_scale=args.baseline_scale,
            importance_ratio=args.importance_ratio,
            importance_threshold=args.importance_threshold,
            importance_oversample=args.importance_oversample,
            importance_max_rounds=args.importance_max_rounds,
        )
        points.requires_grad_(args.grad_reg_coeff > 0.0)
        encoded = encoder(points)
        pred_residual = model(encoded)

        data_objective, mse, l1 = residual_loss_terms(
            pred_residual=pred_residual,
            target_residual=target_residual,
            abs_weight_scale=args.loss_weight_abs_scale,
            l1_coeff=args.loss_l1_coeff,
        )
        grad_reg = gradient_regularization(
            pred_residual=pred_residual,
            points=points,
            coeff=args.grad_reg_coeff,
        )
        objective = data_objective + grad_reg

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        objective_val = float(objective.item())
        data_objective_val = float(data_objective.item())
        grad_reg_val = float(grad_reg.item())
        mse_val = float(mse.item())
        l1_val = float(l1.item())
        ema_mse = mse_val if ema_mse is None else (0.98 * ema_mse + 0.02 * mse_val)

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:6d}  obj={objective_val:.8f}  data_obj={data_objective_val:.8f}  "
                f"mse={mse_val:.8f}  l1={l1_val:.8f}  grad={grad_reg_val:.8f}  ema_mse={ema_mse:.8f}"
            )

        if step >= args.min_steps and ema_mse is not None and ema_mse <= args.target_mse:
            stop_step = step
            print(f"Early stop at step {step}: ema_mse={ema_mse:.8f} <= target_mse={args.target_mse:.8f}")
            break

    with torch.no_grad():
        points, target_residual = sample_batch(
            batch_size=args.eval_samples,
            device=device,
            target_source=target_source,
            target_volume=target_volume,
            target_sample_bank=target_sample_bank,
            baseline_scale=args.baseline_scale,
            importance_ratio=args.importance_ratio,
            importance_threshold=args.importance_threshold,
            importance_oversample=args.importance_oversample,
            importance_max_rounds=args.importance_max_rounds,
        )
        eval_pred = model(encoder(points))
        eval_objective, eval_mse, eval_l1 = residual_loss_terms(
            pred_residual=eval_pred,
            target_residual=target_residual,
            abs_weight_scale=args.loss_weight_abs_scale,
            l1_coeff=args.loss_l1_coeff,
        )
        print(
            f"final_eval_obj={float(eval_objective.item()):.8f}  "
            f"final_eval_mse={float(eval_mse.item()):.8f}  "
            f"final_eval_l1={float(eval_l1.item()):.8f}  "
            f"(trained_steps={stop_step})"
        )

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
        default=1e-3,
        help="Coefficient for gradient regularization on d(pred_residual)/d(points). Set 0 to disable.",
    )
    p.add_argument(
        "--loss-weight-abs-scale",
        type=float,
        default=4.0,
        help="Weighted-MSE scale: weight = 1 + scale * abs(target_residual). Set 0 for unweighted MSE.",
    )
    p.add_argument(
        "--loss-l1-coeff",
        type=float,
        default=0.2,
        help="Additional L1 term coefficient in objective. Set 0 to disable.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fourier-levels", type=int, default=6)
    p.add_argument(
        "--hidden-layers",
        type=int,
        nargs=2,
        default=[192, 192],
        metavar=("H0", "H1"),
        help="Two hidden layer sizes for the residual MLP (default: 192 192).",
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
