"""Superheavy GPU augmentation policy for PTA/TTA.

Self-contained: copy this one file to use it. Constants below control the intensity and elastic stages.
Order: spatial resampling -> CLAHE -> blur/brightness/noise -> clamp ->
adaptive bit depth -> existing output conversion. Shared context-channel
mappings preserve channel addressing. See README.md beside the policies.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PTA_GPU_POLICY_API = 2
PTA_GPU_RUNTIME = "torch-cuda-fused-grid-v1"
AUGMENTATION_PROFILE = "superheavy"

BIT_DEPTH_PROBABILITY = 0.5
BIT_DEPTHS = (1, 2, 4, 8)
CLAHE_PROBABILITY = 0.01
CLAHE_CLIP_RANGE = (3.0, 8.0)
CLAHE_TILE_GRID = (8, 8)
ELASTIC_RMS_FRACTION = 0.02



def _subseed(seed: int, salt: int) -> int:
    value = (int(seed) ^ (int(salt) * 0x9E3779B97F4A7C15)) & ((1 << 63) - 1)
    return value or 1


def _gaussian_kernel_1d(sigma: float, *, device: torch.device) -> torch.Tensor:
    sigma_f = max(0.05, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma_f)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma_f * sigma_f))
    return kernel / kernel.sum()


def _separable_gaussian(sample: torch.Tensor, sigma: float) -> torch.Tensor:
    """Filter NCHW tensors independently per channel using two 1D passes."""
    channels = int(sample.shape[1])
    kernel = _gaussian_kernel_1d(float(sigma), device=sample.device)
    radius = int(kernel.shape[0]) // 2
    # Use one boundary mode for both axes, including narrow source ROIs.
    pad_mode = "reflect" if min(int(sample.shape[-2]), int(sample.shape[-1])) > radius else "replicate"
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    intermediate = F.conv2d(
        F.pad(sample, (radius, radius, 0, 0), mode=pad_mode),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(intermediate, (0, 0, radius, radius), mode=pad_mode),
        vertical,
        groups=channels,
    )


def _blur_sample(sample: torch.Tensor, sigma: float) -> torch.Tensor:
    if float(sigma) <= 0.05:
        return sample
    return _separable_gaussian(sample, float(sigma))


def _fused_pointwise(
    images: torch.Tensor,
    brightness: torch.Tensor,
    multiplier: torch.Tensor,
    additive: torch.Tensor,
    shot_selected: torch.Tensor,
    shot_values: torch.Tensor,
    pepper: torch.Tensor,
    salt: torch.Tensor,
) -> torch.Tensor:
    regular = images * brightness * multiplier + additive
    noisy = torch.where(shot_selected, shot_values, regular)
    noisy = torch.where(pepper, torch.zeros_like(noisy), noisy)
    noisy = torch.where(salt, torch.ones_like(noisy), noisy)
    return torch.clamp(noisy, 0.0, 1.0)


def _sample_bit_depth(
    seed: int,
    depths: Sequence[int],
    probability: float = BIT_DEPTH_PROBABILITY,
) -> Optional[int]:
    """Independent seeded stream: one activation draw and one uniform choice.

    Inline this function beside the preset's existing ``_subseed`` helper;
    call it with the explicit sample seed, not a shared RNG's next draw.
    """
    rng = random.Random(_subseed(seed, 307))
    if rng.random() >= probability:
        return None
    return int(rng.choice(depths))


def _adaptive_bit_depth_torch(image, eligible, bits: int):
    """Device-resident Torch equivalent; import torch when inlining in presets.

    Uses only fixed-size tensors for histogram/mapping construction: no
    ``.item()``, ``.cpu()``, data-dependent Python branches, boolean-indexed
    histogram inputs, or threshold-search loops. This works on CPU/CUDA and
    can be captured by ``torch.compile(..., fullgraph=True)``. Gradients are
    not required by the augmentation policy.
    """
    import torch

    if bits not in (1, 2, 4, 8):
        raise ValueError("adaptive bit depth must be one of 1, 2, 4, 8")
    if image.shape != eligible.shape:
        raise ValueError("eligibility must have exactly the sample's shape")
    sample = image.to(dtype=torch.float32).clamp(0.0, 1.0)
    eligibility = eligible.to(dtype=torch.bool)
    base = torch.where(eligibility, sample, 0.0)
    bins = (sample * 255.0).round().to(dtype=torch.int64)
    active = eligibility & (bins > 0)
    hist = torch.zeros(256, dtype=torch.int64, device=image.device)
    hist.scatter_add_(0, bins.reshape(-1), active.reshape(-1).to(dtype=torch.int64))
    grid = torch.arange(256, dtype=torch.int64, device=image.device)
    low = torch.where(hist > 0, grid, 256).amin()
    high = torch.where(hist > 0, grid, -1).amax()
    nonconstant = high > low
    if bits == 8:
        lut = (((grid - low) * 255).float() / (high - low).clamp_min(1).float())
        lut = lut.round().clamp(0.0, 255.0)
    else:
        levels = 1 << bits
        cdf = hist.cumsum(0)
        targets = torch.arange(1, levels, dtype=torch.int64, device=image.device) * cdf[-1]
        thresholds = torch.searchsorted(cdf * levels, targets, right=False)
        thresholds = torch.minimum(thresholds, high - 1)
        lut = torch.searchsorted(thresholds, grid, right=False) * (255 // (levels - 1))
    output = torch.where(active, lut[bins].float() / 255.0, 0.0)
    return torch.where(nonconstant, output, base).to(dtype=image.dtype)


def _sample_clahe(seed, probability, clip_range):
    """Select activation and clip limit from an independent seed stream 401."""
    if seed is None:
        return None
    probability = float(probability)
    low, high = (float(value) for value in clip_range)
    if not 0.0 <= probability <= 1.0 or not 1.0 <= low <= high:
        raise ValueError("CLAHE requires probability in [0,1] and 1 <= low <= high")
    stream_seed = (int(seed) ^ (401 * 0x9E3779B97F4A7C15)) & ((1 << 63) - 1)
    rng = random.Random(stream_seed or 1)
    return rng.uniform(low, high) if rng.random() < probability else None


def _clahe_geometry(height, width, tile_grid_size):
    """Return OpenCV (columns, rows) tile dimensions and reflected canvas size."""
    columns, rows = (int(value) for value in tile_grid_size)
    if height < 1 or width < 1 or columns < 1 or rows < 1:
        raise ValueError("CLAHE needs a nonempty image and positive tile counts")
    if height % rows == 0 and width % columns == 0:
        padded_height, padded_width = height, width
    else:
        # OpenCV extends BOTH axes in this branch, even an already divisible axis.
        padded_height = height + rows - height % rows
        padded_width = width + columns - width % columns
    return columns, rows, padded_height // rows, padded_width // columns


def _clahe_torch(sample, clip_limit, tile_grid_size=(8, 8)):
    """Device-resident CLAHE on normalized CHW/NCHW tensors, returning float32."""
    if sample.ndim not in (3, 4) or not sample.is_floating_point():
        raise ValueError("Torch CLAHE expects a normalized floating CHW or NCHW tensor")
    if float(clip_limit) < 1.0:
        raise ValueError("CLAHE clip_limit must be >= 1")
    squeeze = sample.ndim == 3
    source = sample.unsqueeze(0) if squeeze else sample
    count, channels, height, width = source.shape
    if channels < 1 or count < 1:
        raise ValueError("CLAHE needs at least one sample and one channel")
    columns, rows, tile_height, tile_width = _clahe_geometry(height, width, tile_grid_size)
    area = tile_height * tile_width
    device = source.device
    bins = torch.round(source.float().clamp(0.0, 1.0) * 255.0).to(torch.int64)

    def reflected_indexes(length, target):
        if length == 1:
            return torch.zeros(target, dtype=torch.int64, device=device)
        phase = torch.arange(target, dtype=torch.int64, device=device) % (2 * (length - 1))
        return torch.minimum(phase, 2 * (length - 1) - phase)

    extended = bins.index_select(-2, reflected_indexes(height, rows * tile_height))
    extended = extended.index_select(-1, reflected_indexes(width, columns * tile_width))
    tile_values = extended.reshape(count, channels, rows, tile_height, columns, tile_width)
    tile_values = tile_values.permute(0, 2, 4, 1, 3, 5).reshape(count * rows * columns, -1)
    hist = torch.zeros((count * rows * columns, 256), dtype=torch.int64, device=device)
    hist.scatter_add_(1, tile_values, torch.ones_like(tile_values))
    limit = max(int(float(clip_limit) * area / 256), 1) * channels
    clipped = hist.clamp(max=limit)
    excess = (hist - clipped).sum(dim=1, keepdim=True)
    batch = torch.div(excess, 256 * channels, rounding_mode="floor")
    residue = excess % (256 * channels)
    whole_residue = torch.div(residue, channels, rounding_mode="floor")
    fraction = residue % channels
    step = torch.div(256, whole_residue.clamp(min=1), rounding_mode="floor").clamp(min=1)
    histogram_bins = torch.arange(256, dtype=torch.int64, device=device)[None, :]
    residual_slots = ((histogram_bins % step == 0)
                      & (torch.div(histogram_bins, step, rounding_mode="floor") < whole_residue))
    redistributed = clipped + batch * channels + residual_slots.to(torch.int64) * channels
    redistributed[:, -1:] += fraction
    cumulative = redistributed.cumsum(dim=1).float() / float(channels)
    lut = torch.round(cumulative * (255.0 / area)).clamp(0.0, 255.0)

    x = torch.arange(width, dtype=torch.float32, device=device) * (1.0 / tile_width) - 0.5
    y = torch.arange(height, dtype=torch.float32, device=device) * (1.0 / tile_height) - 0.5
    left, top = x.floor().to(torch.int64), y.floor().to(torch.int64)
    wx = (x - left.float())[None, None, None, :]
    wy = (y - top.float())[None, None, :, None]
    right, bottom = (left + 1).clamp(max=columns - 1), (top + 1).clamp(max=rows - 1)
    left, top = left.clamp(min=0), top.clamp(min=0)
    sample_offsets = torch.arange(count, dtype=torch.int64, device=device)[:, None, None, None] * (rows * columns)

    def mapped(tile_y, tile_x):
        tile = sample_offsets + tile_y[None, None, :, None] * columns + tile_x[None, None, None, :]
        return lut[tile, bins]

    output = ((mapped(top, left) * (1.0 - wx) + mapped(top, right) * wx) * (1.0 - wy)
              + (mapped(bottom, left) * (1.0 - wx) + mapped(bottom, right) * wx) * wy)
    output = output.round().clamp(0.0, 255.0) / 255.0
    return output[0] if squeeze else output


def _normalized_elastic_displacement_torch(seed, height, width, rms_fraction, device):
    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError("Elastic field dimensions must be positive")
    if min(height, width) < 2:
        return torch.zeros((2, height, width), device=device, dtype=torch.float32)
    scale = max(1.0, min(height, width) / 128.0)
    coarse_h = max(2, int(round(height / scale)))
    coarse_w = max(2, int(round(width / scale)))
    generator = torch.Generator(device=device)
    generator.manual_seed(_subseed(seed, 101))
    noise = torch.rand((1, 2, coarse_h, coarse_w), generator=generator,
                       device=device, dtype=torch.float32) * 2.0 - 1.0
    sigma = max(0.5, 0.08 * min(height, width) / scale)
    field = _separable_gaussian(noise, sigma)
    if (coarse_h, coarse_w) != (height, width):
        field = F.interpolate(field, size=(height, width), mode="bilinear", align_corners=False)
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    rms = field.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    field = field * ((float(rms_fraction) * min(height, width)) / rms)
    return field[0].contiguous()


class GPUAugmentation:
    """Torch-CUDA implementation of the baseline probability graph."""

    supports_cuda_sources = True

    def __init__(self, *, device: str, batch_size: int = 32) -> None:
        self.device = torch.device(str(device))
        if self.device.type != "cuda":
            raise ValueError(f"GPUAugmentation requires a CUDA device, got {self.device}")
        self.batch_size = max(1, int(batch_size))
        self._pixel_grid_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._pointwise_kernel = _fused_pointwise
        self._pointwise_compiled = False
        if os.environ.get("PTA_GPU_TORCH_COMPILE", "1").strip().lower() not in {"0", "false", "no"}:
            compile_fn = getattr(torch, "compile", None)
            if callable(compile_fn):
                try:
                    self._pointwise_kernel = compile_fn(
                        _fused_pointwise,
                        fullgraph=True,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                    self._pointwise_compiled = True
                except Exception:
                    self._pointwise_kernel = _fused_pointwise
                    self._pointwise_compiled = False

    def _pixel_grid(self, height: int, width: int) -> torch.Tensor:
        key = (int(height), int(width))
        cached = self._pixel_grid_cache.get(key)
        if cached is not None:
            return cached
        ys, xs = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        grid = torch.stack((xs, ys, ones), dim=-1)
        self._pixel_grid_cache[key] = grid
        return grid

    @staticmethod
    def _sample_parameters(seed: int, height: int, width: int) -> Dict[str, object]:
        rng = random.Random(int(seed))
        d4 = int(rng.randrange(8))
        rotation = float(rng.uniform(-70.0, 70.0))
        if rng.random() < 0.5:
            scale = float(rng.uniform(1.0 / 3.0, 1.0))
        else:
            scale = float(rng.uniform(1.0, 3.0))
        translate_x = float(rng.uniform(-0.175, 0.175) * width)
        translate_y = float(rng.uniform(-0.175, 0.175) * height)
        shear_x = float(rng.uniform(-42.0, 42.0))
        shear_y = float(rng.uniform(-42.0, 42.0))
        elastic = bool(rng.random() < 0.30)
        brightness = float(rng.uniform(0.65, 1.35)) if rng.random() < 0.50 else 1.0
        blur_sigma = float(rng.uniform(0.0, 8.0)) if rng.random() < 0.25 else 0.0
        noise_family = int(rng.randrange(3))
        noise_strength = (
            float(rng.uniform(0.0, 0.85))
            if noise_family == 0
            else float(rng.uniform(0.0, 0.085))
            if noise_family == 1
            else float(rng.uniform(0.15, 1.85))
        )
        salt_pepper_amount = float(rng.uniform(0.0, 0.085)) if rng.random() < 0.25 else 0.0
        return {
            "bit_depth": _sample_bit_depth(seed, BIT_DEPTHS, BIT_DEPTH_PROBABILITY),
            "clahe_clip_limit": _sample_clahe(seed, CLAHE_PROBABILITY, CLAHE_CLIP_RANGE),
            "d4": d4,
            "rotation": rotation,
            "scale": scale,
            "translate_x": translate_x,
            "translate_y": translate_y,
            "shear_x": shear_x,
            "shear_y": shear_y,
            "elastic": elastic,
            "brightness": brightness,
            "blur_sigma": blur_sigma,
            "noise_family": noise_family,
            "noise_strength": noise_strength,
            "salt_pepper_amount": salt_pepper_amount,
        }

    @staticmethod
    def _forward_matrix(params: Dict[str, object], height: int, width: int) -> List[List[float]]:
        d4 = int(params["d4"])
        quarter_angle = math.radians(90.0 * float(d4 % 4))
        qc, qs = math.cos(quarter_angle), math.sin(quarter_angle)
        reflect = -1.0 if d4 >= 4 else 1.0
        d00, d01 = qc * reflect, -qs
        d10, d11 = qs * reflect, qc

        sx = math.tan(math.radians(float(params["shear_x"])))
        sy = math.tan(math.radians(float(params["shear_y"])))
        scale = float(params["scale"])
        # Shear @ (uniform scale * D4).
        a00 = scale * (d00 + sx * d10)
        a01 = scale * (d01 + sx * d11)
        a10 = scale * (sy * d00 + d10)
        a11 = scale * (sy * d01 + d11)

        angle = math.radians(float(params["rotation"]))
        c, s = math.cos(angle), math.sin(angle)
        l00, l01 = c * a00 - s * a10, c * a01 - s * a11
        l10, l11 = s * a00 + c * a10, s * a01 + c * a11

        center_x = (float(width) - 1.0) * 0.5
        center_y = (float(height) - 1.0) * 0.5
        tx = center_x + float(params["translate_x"]) - (l00 * center_x + l01 * center_y)
        ty = center_y + float(params["translate_y"]) - (l10 * center_x + l11 * center_y)
        return [[l00, l01, tx], [l10, l11, ty], [0.0, 0.0, 1.0]]

    def _elastic_displacement(self, seed: int, height: int, width: int) -> torch.Tensor:
        return _normalized_elastic_displacement_torch(seed, height, width, ELASTIC_RMS_FRACTION, self.device)

    def _apply_intensity_noise(
        self,
        images: torch.Tensor,
        seeds: Sequence[Optional[int]],
        params_by_sample: Sequence[Optional[Dict[str, object]]],
    ) -> torch.Tensor:
        # Called by BOTH PTA and the existing TTA adapter, before photometry.
        eligible = (images * 255.0).round() > 0
        output = images.clone()
        count = int(output.shape[0])
        brightness = torch.ones((count, 1, 1, 1), device=self.device, dtype=output.dtype)
        multiplier = torch.ones_like(output)
        additive = torch.zeros_like(output)
        shot_selected = torch.zeros((count, 1, 1, 1), device=self.device, dtype=torch.bool)
        shot_values = torch.zeros_like(output)
        pepper = torch.zeros_like(output, dtype=torch.bool)
        salt = torch.zeros_like(output, dtype=torch.bool)
        for index, (seed, params) in enumerate(zip(seeds, params_by_sample)):
            if seed is None or params is None:
                continue
            clip_limit = params.get("clahe_clip_limit")
            if clip_limit is not None:
                output[index:index + 1] = _clahe_torch(output[index:index + 1], float(clip_limit), CLAHE_TILE_GRID)
            sample = _blur_sample(output[index:index + 1], float(params["blur_sigma"]))
            output[index:index + 1] = sample
            brightness[index] = float(params["brightness"])
            generator = torch.Generator(device=self.device)
            generator.manual_seed(_subseed(int(seed), 211))
            family = int(params["noise_family"])
            strength = float(params["noise_strength"])
            if family == 0 and strength > 0.0:
                additive[index:index + 1] = torch.randn(
                    sample.shape,
                    device=self.device,
                    dtype=sample.dtype,
                    generator=generator,
                ) * strength
            elif family == 1 and strength > 1e-6:
                shot_selected[index] = True
                shot_input = torch.clamp(sample * float(params["brightness"]), 0.0, 1.0)
                shot_values[index:index + 1] = torch.poisson(
                    shot_input / strength,
                    generator=generator,
                ) * strength
            elif family == 2:
                multiplier[index:index + 1] = torch.empty_like(sample).uniform_(
                    0.15,
                    1.85,
                    generator=generator,
                )

            amount = float(params["salt_pepper_amount"])
            if amount > 0.0:
                chooser = torch.rand(
                    sample.shape,
                    device=self.device,
                    dtype=sample.dtype,
                    generator=generator,
                )
                pepper[index:index + 1] = chooser < (amount * 0.5)
                salt[index:index + 1] = chooser > (1.0 - amount * 0.5)
        arguments = (
            output,
            brightness,
            multiplier,
            additive,
            shot_selected,
            shot_values,
            pepper,
            salt,
        )
        try:
            result = self._pointwise_kernel(*arguments)
        except Exception:
            if not self._pointwise_compiled:
                raise
            self._pointwise_kernel = _fused_pointwise
            self._pointwise_compiled = False
            result = _fused_pointwise(*arguments)
        for index, (seed, params) in enumerate(zip(seeds, params_by_sample)):
            if seed is None or params is None:
                continue
            result[index] = torch.where(eligible[index], result[index], 0.0)
            depth = params.get("bit_depth")
            if depth is not None:
                result[index] = _adaptive_bit_depth_torch(result[index], eligible[index], int(depth))
        return result

    @torch.inference_mode()
    def apply_batch_many(
        self,
        *,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        seeds: Sequence[Sequence[Optional[int]]],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upload several source ROIs once and emit one flat CUDA batch.

        Results are source-major and preserve the order of each nested seed
        sequence. A ``None`` seed identifies an unaugmented original.
        """
        if not images or len(images) != len(masks) or len(images) != len(seeds):
            raise ValueError(
                "apply_batch_many requires equally sized nonempty images, masks, and seeds"
            )
        flat_seeds = [seed for source_seeds in seeds for seed in source_seeds]
        if not flat_seeds:
            raise ValueError("apply_batch_many requires at least one output seed/original marker")
        if len(flat_seeds) > self.batch_size:
            raise ValueError(
                f"batch length {len(flat_seeds)} exceeds configured batch_size={self.batch_size}"
            )
        out_h, out_w = int(output_size[0]), int(output_size[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"invalid output_size={output_size}")

        cuda_image_inputs = [
            bool(torch.is_tensor(image) and bool(getattr(image, "is_cuda", False)))
            for image in images
        ]
        if any(cuda_image_inputs) and not all(cuda_image_inputs):
            raise ValueError("apply_batch_many cannot mix NumPy and CUDA source images")
        image_arrays = (
            []
            if all(cuda_image_inputs)
            else [np.ascontiguousarray(np.asarray(image), dtype=np.uint8) for image in images]
        )
        mask_arrays = [
            np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8))
            for mask in masks
        ]
        image_shape = tuple(images[0].shape) if all(cuda_image_inputs) else tuple(image_arrays[0].shape)
        if any(tuple(image.shape) != image_shape for image in images):
            raise ValueError("apply_batch_many source images must share one shape")
        if any(mask.ndim != 2 or tuple(mask.shape) != image_shape[:2] for mask in mask_arrays):
            raise ValueError(
                f"apply_batch_many image/mask shape mismatch for source shape={image_shape}"
            )
        if all(cuda_image_inputs):
            for image in images:
                if image.dtype != torch.uint8 or image.device != self.device:
                    raise ValueError(
                        "CUDA policy source images must be uint8 tensors on "
                        f"{self.device}; got dtype={image.dtype}, device={image.device}"
                    )
            if len(image_shape) == 2:
                image_tensor = torch.stack(
                    [image.unsqueeze(0) for image in images], dim=0
                ).contiguous()
            elif len(image_shape) == 3 and int(image_shape[2]) >= 1:
                image_tensor = torch.stack(
                    [image.permute(2, 0, 1) for image in images], dim=0
                ).contiguous()
            else:
                raise ValueError(
                    f"GPU policy supports HxW gray or HxWxC images with C>=1, got {image_shape}"
                )
        elif len(image_shape) == 2:
            image_nchw = np.stack(image_arrays, axis=0)[:, None, :, :]
            image_host = torch.from_numpy(np.ascontiguousarray(image_nchw)).pin_memory()
            image_tensor = image_host.to(self.device, non_blocking=True)
        elif len(image_shape) == 3 and int(image_shape[2]) >= 1:
            image_nchw = np.transpose(np.stack(image_arrays, axis=0), (0, 3, 1, 2))
            image_host = torch.from_numpy(np.ascontiguousarray(image_nchw)).pin_memory()
            image_tensor = image_host.to(self.device, non_blocking=True)
        else:
            raise ValueError(
                f"GPU policy supports HxW gray or HxWxC images with C>=1, got {image_shape}"
            )
        mask_nhw = np.stack(mask_arrays, axis=0)
        mask_host = torch.from_numpy(np.ascontiguousarray(mask_nhw)).pin_memory()
        mask_tensor = mask_host.to(self.device, non_blocking=True).unsqueeze(1)

        image_float = image_tensor.to(torch.float32) / 255.0
        mask_float = mask_tensor.to(torch.float32)
        if tuple(image_float.shape[-2:]) != (out_h, out_w):
            image_float = F.interpolate(image_float, size=(out_h, out_w), mode="bilinear", align_corners=False)
            mask_float = F.interpolate(mask_float, size=(out_h, out_w), mode="nearest")

        source_indices = torch.as_tensor(
            [
                source_index
                for source_index, source_seeds in enumerate(seeds)
                for _seed in source_seeds
            ],
            device=self.device,
            dtype=torch.int64,
        )
        image_batch = image_float.index_select(0, source_indices)
        mask_batch = mask_float.index_select(0, source_indices)
        if all(seed is None for seed in flat_seeds):
            # ``augmentation_ratio=1`` emits originals only. The rendered
            # sources have already received any required output resize above,
            # so building an identity inverse matrix/grid and sampling both
            # tensors again is pure overhead.
            images_u8 = torch.clamp(
                torch.round(image_batch * 255.0), 0.0, 255.0
            ).to(torch.uint8)
            masks_u8 = (mask_batch[:, 0] >= 0.5).to(torch.uint8)
            return images_u8.contiguous(), masks_u8.contiguous()
        params_by_sample: List[Optional[Dict[str, object]]] = []
        forward_matrices: List[List[List[float]]] = []
        for seed in flat_seeds:
            if seed is None:
                params_by_sample.append(None)
                forward_matrices.append([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            else:
                params = self._sample_parameters(int(seed), out_h, out_w)
                params_by_sample.append(params)
                forward_matrices.append(self._forward_matrix(params, out_h, out_w))

        forward = torch.tensor(forward_matrices, device=self.device, dtype=torch.float32)
        inverse = torch.stack([torch.linalg.inv(matrix) for matrix in forward])
        pixel_grid = self._pixel_grid(out_h, out_w)
        source = torch.stack([torch.einsum("ij,hwj->hwi", matrix, pixel_grid) for matrix in inverse])
        for index, (seed, params) in enumerate(zip(flat_seeds, params_by_sample)):
            if seed is not None and params is not None and bool(params["elastic"]):
                displacement = self._elastic_displacement(int(seed), out_h, out_w)
                source[index, :, :, 0] += displacement[0]
                source[index, :, :, 1] += displacement[1]

        if out_w > 1:
            grid_x = source[:, :, :, 0] * (2.0 / float(out_w - 1)) - 1.0
        else:
            grid_x = torch.zeros_like(source[:, :, :, 0])
        if out_h > 1:
            grid_y = source[:, :, :, 1] * (2.0 / float(out_h - 1)) - 1.0
        else:
            grid_y = torch.zeros_like(source[:, :, :, 1])
        sampling_grid = torch.stack((grid_x, grid_y), dim=-1)

        warped_images = F.grid_sample(
            image_batch,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_masks = F.grid_sample(
            mask_batch,
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_images = self._apply_intensity_noise(warped_images, flat_seeds, params_by_sample)
        images_u8 = torch.clamp(torch.round(warped_images * 255.0), 0.0, 255.0).to(torch.uint8)
        masks_u8 = (warped_masks[:, 0] >= 0.5).to(torch.uint8)
        return images_u8.contiguous(), masks_u8.contiguous()

    @torch.inference_mode()
    def apply_batch(
        self,
        *,
        image: np.ndarray,
        mask: np.ndarray,
        seeds: Sequence[Optional[int]],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the batched policy to one source ROI."""

        return self.apply_batch_many(
            images=(image,),
            masks=(mask,),
            seeds=(tuple(seeds),),
            output_size=output_size,
        )


def build_gpu_augmentation(*, device: str, batch_size: int = 32) -> GPUAugmentation:
    """XTA PTA single-file GPU policy factory."""
    return GPUAugmentation(device=device, batch_size=batch_size)
