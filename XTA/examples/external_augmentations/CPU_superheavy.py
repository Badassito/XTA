"""Superheavy CPU augmentation policy for PTA/TTA.

Self-contained: copy this one file to use it. Constants below control the intensity and elastic stages.
Order: spatial resampling -> CLAHE -> blur/brightness/noise -> clamp ->
adaptive bit depth -> existing output conversion. Shared context-channel
mappings preserve channel addressing. See README.md beside the policies.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np


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


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma_f = max(0.05, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma_f)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2.0 * sigma_f * sigma_f))
    return np.ascontiguousarray(kernel / np.sum(kernel), dtype=np.float32)


def _blur_sample(sample: np.ndarray, sigma: float) -> np.ndarray:
    if float(sigma) <= 0.05:
        return sample
    kernel = _gaussian_kernel_1d(float(sigma))
    radius = int(kernel.size) // 2
    border = (
        cv2.BORDER_REFLECT_101
        if min(int(sample.shape[0]), int(sample.shape[1])) > radius
        else cv2.BORDER_REPLICATE
    )
    blurred = cv2.sepFilter2D(
        sample,
        ddepth=-1,
        kernelX=kernel,
        kernelY=kernel,
        borderType=border,
    )
    if sample.ndim == 3 and blurred.ndim == 2:
        blurred = blurred[:, :, None]
    return np.ascontiguousarray(blurred, dtype=np.float32)


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


def _adaptive_bit_depth_numpy(
    image: np.ndarray,
    eligible: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Quantize one normalized float sample, preserving protected zeros.

    ``eligible`` must be a boolean array of the exact image shape captured
    before brightness/noise/CLAHE. Geometric padding must be false. No label
    or segmentation mask is involved in determining eligibility.
    """
    if bits not in (1, 2, 4, 8):
        raise ValueError("adaptive bit depth must be one of 1, 2, 4, 8")
    if image.shape != eligible.shape:
        raise ValueError("eligibility must have exactly the sample's shape")
    sample = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    eligibility = eligible.astype(bool, copy=False)
    base = np.where(eligibility, sample, 0.0)
    bins = np.rint(sample * np.float32(255.0)).astype(np.int64)
    active = eligibility & (bins > 0)
    hist = np.bincount(bins[active], minlength=256)
    occupied = np.flatnonzero(hist)
    if occupied.size < 2:
        return base.astype(image.dtype, copy=False)
    low, high = occupied[0], occupied[-1]
    grid = np.arange(256, dtype=np.int64)
    if bits == 8:
        lut = np.rint(
            ((grid - low) * 255).astype(np.float32) / np.float32(high - low)
        ).clip(0, 255)
    else:
        levels = 1 << bits
        cdf = np.cumsum(hist, dtype=np.int64)
        # Integer arithmetic defines exactly the same quantiles on CPU/CUDA.
        targets = np.arange(1, levels, dtype=np.int64) * cdf[-1]
        thresholds = np.searchsorted(cdf * levels, targets, side="left")
        thresholds = np.minimum(thresholds, high - 1)
        lut = np.searchsorted(thresholds, grid, side="left") * (255 // (levels - 1))
    output = np.where(active, lut[bins].astype(np.float32) / np.float32(255.0), 0.0)
    return output.astype(image.dtype, copy=False)


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


def _clahe_numpy(sample, clip_limit, tile_grid_size=(8, 8)):
    """CLAHE on an HW/HWC normalized floating NumPy image; return float32."""
    source = np.asarray(sample)
    if source.ndim not in (2, 3) or not np.issubdtype(source.dtype, np.floating):
        raise ValueError("CPU CLAHE expects a normalized floating HW or HWC image")
    if float(clip_limit) < 1.0:
        raise ValueError("CLAHE clip_limit must be >= 1")
    squeeze = source.ndim == 2
    source = source[..., None] if squeeze else source
    height, width, channels = source.shape
    if channels < 1:
        raise ValueError("CLAHE needs at least one channel")
    columns, rows, tile_height, tile_width = _clahe_geometry(height, width, tile_grid_size)
    area = tile_height * tile_width
    bins = np.rint(np.clip(source.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.int64)

    def reflected_indexes(length, target):
        if length == 1:
            return np.zeros(target, dtype=np.int64)
        phase = np.arange(target, dtype=np.int64) % (2 * (length - 1))
        return np.minimum(phase, 2 * (length - 1) - phase)

    extended = bins[reflected_indexes(height, rows * tile_height)[:, None],
                    reflected_indexes(width, columns * tile_width)[None, :], :]
    tile_values = extended.reshape(rows, tile_height, columns, tile_width, channels)
    tile_values = tile_values.transpose(0, 2, 1, 3, 4).reshape(rows * columns, -1)
    histogram_indexes = tile_values + np.arange(rows * columns, dtype=np.int64)[:, None] * 256
    hist = np.bincount(histogram_indexes.ravel(), minlength=rows * columns * 256)
    hist = hist.reshape(rows * columns, 256)
    limit = max(int(float(clip_limit) * area / 256), 1) * channels
    clipped = np.minimum(hist, limit)
    excess = (hist - clipped).sum(axis=1, keepdims=True)
    batch = excess // (256 * channels)
    residue = excess % (256 * channels)
    whole_residue, fraction = residue // channels, residue % channels
    step = np.maximum(256 // np.maximum(whole_residue, 1), 1)
    histogram_bins = np.arange(256, dtype=np.int64)[None, :]
    residual_slots = (histogram_bins % step == 0) & (histogram_bins // step < whole_residue)
    redistributed = clipped + batch * channels + residual_slots * channels
    redistributed[:, -1:] += fraction
    cumulative = redistributed.cumsum(axis=1).astype(np.float32) / np.float32(channels)
    lut = np.rint(cumulative * np.float32(255.0 / area)).clip(0, 255).astype(np.float32)

    x = np.arange(width, dtype=np.float32) * np.float32(1.0 / tile_width) - np.float32(0.5)
    y = np.arange(height, dtype=np.float32) * np.float32(1.0 / tile_height) - np.float32(0.5)
    left, top = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    wx, wy = (x - left.astype(np.float32))[None, :, None], (y - top.astype(np.float32))[:, None, None]
    right, bottom = np.minimum(left + 1, columns - 1), np.minimum(top + 1, rows - 1)
    left, top = np.maximum(left, 0), np.maximum(top, 0)

    def mapped(tile_y, tile_x):
        tile = tile_y[:, None, None] * columns + tile_x[None, :, None]
        return lut[tile, bins]

    output = ((mapped(top, left) * (1.0 - wx) + mapped(top, right) * wx) * (1.0 - wy)
              + (mapped(bottom, left) * (1.0 - wx) + mapped(bottom, right) * wx) * wy)
    output = np.rint(output).clip(0, 255).astype(np.float32) / np.float32(255.0)
    return output[..., 0] if squeeze else output


def _normalized_elastic_displacement_numpy(seed, height, width, rms_fraction):
    height, width = int(height), int(width)
    if height < 1 or width < 1:
        raise ValueError("Elastic field dimensions must be positive")
    if min(height, width) < 2:
        return np.zeros((height, width, 2), dtype=np.float32)
    scale = max(1.0, min(height, width) / 128.0)
    coarse_h = max(2, int(round(height / scale)))
    coarse_w = max(2, int(round(width / scale)))
    generator = np.random.default_rng(_subseed(seed, 101))
    noise = generator.uniform(-1.0, 1.0, size=(coarse_h, coarse_w, 2)).astype(np.float32)
    sigma = max(0.5, 0.08 * min(height, width) / scale)
    kernel = _gaussian_kernel_1d(sigma)
    border = (cv2.BORDER_REFLECT_101 if min(coarse_h, coarse_w) > kernel.size // 2
              else cv2.BORDER_REPLICATE)
    field = cv2.sepFilter2D(noise, -1, kernel, kernel, borderType=border)
    if (coarse_h, coarse_w) != (height, width):
        field = cv2.resize(field, (width, height), interpolation=cv2.INTER_LINEAR)
    field -= np.mean(field, axis=(0, 1), keepdims=True, dtype=np.float64)
    rms = np.sqrt(np.mean(field * field, axis=(0, 1), keepdims=True, dtype=np.float64))
    field *= (float(rms_fraction) * min(height, width)) / np.maximum(rms, 1e-8)
    return np.ascontiguousarray(field, dtype=np.float32)


class CPUAugmentation:
    """Seedable OpenCV/NumPy implementation of the superheavy probability graph."""

    tta_replay_contract = "opencv-affine-elastic-v1"

    def __init__(self) -> None:
        self._seed = 1
        self._mask_interpolation = cv2.INTER_NEAREST

    def set_random_seed(self, seed: int) -> None:
        self._seed = int(seed)

    def set_mask_interpolation(self, interpolation: int) -> None:
        if int(interpolation) != int(cv2.INTER_NEAREST):
            raise ValueError("CPUAugmentation requires nearest-neighbor mask interpolation")
        self._mask_interpolation = int(interpolation)

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
        salt_pepper_amount = (
            float(rng.uniform(0.0, 0.085)) if rng.random() < 0.25 else 0.0
        )
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
    def _forward_matrix(
        params: Mapping[str, object],
        height: int,
        width: int,
    ) -> List[List[float]]:
        d4 = int(params["d4"])
        quarter_angle = math.radians(90.0 * float(d4 % 4))
        qc, qs = math.cos(quarter_angle), math.sin(quarter_angle)
        reflect = -1.0 if d4 >= 4 else 1.0
        d00, d01 = qc * reflect, -qs
        d10, d11 = qs * reflect, qc

        sx = math.tan(math.radians(float(params["shear_x"])))
        sy = math.tan(math.radians(float(params["shear_y"])))
        scale = float(params["scale"])
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
        tx = (
            center_x
            + float(params["translate_x"])
            - (l00 * center_x + l01 * center_y)
        )
        ty = (
            center_y
            + float(params["translate_y"])
            - (l10 * center_x + l11 * center_y)
        )
        return [[l00, l01, tx], [l10, l11, ty], [0.0, 0.0, 1.0]]

    @staticmethod
    def _elastic_displacement(seed: int, height: int, width: int) -> np.ndarray:
        return _normalized_elastic_displacement_numpy(seed, height, width, ELASTIC_RMS_FRACTION)

    @staticmethod
    def _apply_intensity_noise(
        image: np.ndarray,
        *,
        seed: int,
        params: Mapping[str, object],
    ) -> np.ndarray:
        # Capture rendered zeros BEFORE any intensity stage; labels never define padding.
        eligible = np.rint(image * np.float32(255.0)) > 0
        clip_limit = params.get("clahe_clip_limit")
        if clip_limit is not None:
            image = _clahe_numpy(image, float(clip_limit), CLAHE_TILE_GRID)
        sample = _blur_sample(image, float(params["blur_sigma"]))
        generator = np.random.default_rng(_subseed(seed, 211))
        family = int(params["noise_family"])
        strength = float(params["noise_strength"])
        brightness = float(params["brightness"])

        if family == 0 and strength > 0.0:
            additive = generator.standard_normal(sample.shape).astype(np.float32)
            output = sample * brightness + additive * strength
        elif family == 1 and strength > 1e-6:
            shot_input = np.clip(sample * brightness, 0.0, 1.0)
            output = generator.poisson(shot_input / strength).astype(np.float32)
            output *= strength
        elif family == 2:
            multiplier = generator.uniform(0.15, 1.85, size=sample.shape).astype(
                np.float32
            )
            output = sample * brightness * multiplier
        else:
            output = sample * brightness

        amount = float(params["salt_pepper_amount"])
        if amount > 0.0:
            chooser = generator.random(sample.shape).astype(np.float32)
            output = np.where(chooser < amount * 0.5, 0.0, output)
            output = np.where(chooser > 1.0 - amount * 0.5, 1.0, output)
        output = np.where(eligible, np.clip(output, 0.0, 1.0), 0.0).astype(np.float32)
        depth = params.get("bit_depth")
        if depth is not None:
            output = _adaptive_bit_depth_numpy(output, eligible, int(depth))
        return np.ascontiguousarray(output, dtype=np.float32)

    def __call__(self, *, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask)
        if image_array.dtype != np.uint8 or image_array.ndim not in (2, 3):
            raise ValueError(
                "CPUAugmentation expects a uint8 HxW or HxWxC image; "
                f"got shape={image_array.shape}, dtype={image_array.dtype}"
            )
        if (
            mask_array.ndim != 2
            or tuple(mask_array.shape) != tuple(image_array.shape[:2])
        ):
            raise ValueError(
                "CPUAugmentation image/mask shape mismatch: "
                f"image={image_array.shape}, mask={mask_array.shape}"
            )

        height, width = (int(value) for value in image_array.shape[:2])
        params = self._sample_parameters(self._seed, height, width)
        forward = np.asarray(
            self._forward_matrix(params, height, width),
            dtype=np.float64,
        )
        inverse = np.linalg.inv(forward)
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        source_x = (
            inverse[0, 0] * grid_x
            + inverse[0, 1] * grid_y
            + inverse[0, 2]
        ).astype(np.float32)
        source_y = (
            inverse[1, 0] * grid_x
            + inverse[1, 1] * grid_y
            + inverse[1, 2]
        ).astype(np.float32)
        if bool(params["elastic"]):
            displacement = self._elastic_displacement(self._seed, height, width)
            source_x += displacement[:, :, 0]
            source_y += displacement[:, :, 1]

        image_float = np.ascontiguousarray(
            image_array.astype(np.float32) / 255.0
        )
        warped_image = cv2.remap(
            image_float,
            source_x,
            source_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if image_array.ndim == 3 and warped_image.ndim == 2:
            warped_image = warped_image[:, :, None]
        warped_mask = cv2.remap(
            np.ascontiguousarray((mask_array > 0).astype(np.uint8)),
            source_x,
            source_y,
            interpolation=self._mask_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        augmented = self._apply_intensity_noise(
            np.ascontiguousarray(warped_image, dtype=np.float32),
            seed=self._seed,
            params=params,
        )
        image_u8 = np.clip(np.rint(augmented * 255.0), 0.0, 255.0).astype(
            np.uint8
        )
        mask_u8 = np.ascontiguousarray((warped_mask >= 0.5).astype(np.uint8))
        return {
            "image": np.ascontiguousarray(image_u8),
            "mask": mask_u8,
        }


def build_augmentation() -> CPUAugmentation:
    """Build a seedable superheavy policy for XTA's CPU augmentation backend."""
    return CPUAugmentation()
