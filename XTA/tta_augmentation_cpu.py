"""CPU policy replay with conservative inverse support for TTA inference."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .tta_augmentation_config import TtaAugmentationSettings


def _bilinear_displacement(field: np.ndarray, xy: np.ndarray):
    h, w = field.shape[:2]
    x, y = np.clip(xy[..., 0], 0, w - 1), np.clip(xy[..., 1], 0, h - 1)
    x0, y0 = x.astype(np.intp), y.astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    a, b, c, d = field[y0, x0], field[y0, x1], field[y1, x0], field[y1, x1]
    u, v = (x - x0)[..., None], (y - y0)[..., None]
    value = (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v
    dx, dy = (b - a) * (1 - v) + (d - c) * v, (c - a) * (1 - u) + (d - b) * u
    dx *= ((xy[..., 0] >= 0) & (xy[..., 0] <= w - 1))[..., None]
    dy *= ((xy[..., 1] >= 0) & (xy[..., 1] <= h - 1))[..., None]
    return value, dx, dy


def inverse_policy_grid_cpu(forward: np.ndarray, displacement: np.ndarray | None,
                            height: int, width: int, *, iterations: int = 16,
                            tolerance: float = 0.05, strip_rows: int = 128):
    """Solve x = inv(A)y + d(y); reject folds, cropped pixels and failed solves.

    Coordinates are pixel centers in the augmented raster. The Jacobian uses
    exact bilinear displacement derivatives, including reflected affine maps.
    """
    forward = np.asarray(forward, dtype=np.float64)
    if forward.shape != (3, 3) or not np.isfinite(forward).all():
        raise ValueError('CPU policy forward matrix must be finite 3x3')
    if not np.allclose(forward[2], (0., 0., 1.)):
        raise ValueError('CPU policy replay requires an affine forward matrix')
    inverse = np.linalg.inv(forward)
    h, w = int(height), int(width)
    if displacement is not None:
        displacement = np.asarray(displacement, dtype=np.float32)
        if displacement.shape != (h, w, 2) or not np.isfinite(displacement).all():
            raise ValueError('CPU displacement must be finite HxWx2')
    output, validity = np.empty((h, w, 2), np.float32), np.empty((h, w), bool)
    det_base = np.linalg.det(inverse[:2, :2])
    for row in range(0, h, max(1, strip_rows)):
        yy, xx = np.mgrid[row:min(h, row + strip_rows), 0:w]
        target = np.stack((xx, yy), axis=-1).astype(np.float64)
        current = target @ forward[:2, :2].T + forward[:2, 2]
        orientation_ok = np.ones(xx.shape, bool)
        if displacement is not None:
            for _ in range(iterations):
                delta, dx, dy = _bilinear_displacement(displacement, current)
                residual = current @ inverse[:2, :2].T + inverse[:2, 2] + delta - target
                a, b = inverse[0, 0] + dx[..., 0], inverse[0, 1] + dy[..., 0]
                c, d = inverse[1, 0] + dx[..., 1], inverse[1, 1] + dy[..., 1]
                det = a * d - b * c
                denominator = np.where(np.abs(det) > 1e-7, det, 1.)
                step = np.stack(((d * residual[..., 0] - b * residual[..., 1]) / denominator,
                                 (-c * residual[..., 0] + a * residual[..., 1]) / denominator), axis=-1)
                current = np.nan_to_num(current - np.clip(step, -32., 32.),
                                        nan=-1e6, posinf=1e6, neginf=-1e6)
            delta, dx, dy = _bilinear_displacement(displacement, current)
            residual = current @ inverse[:2, :2].T + inverse[:2, 2] + delta - target
            det = ((inverse[0, 0] + dx[..., 0]) * (inverse[1, 1] + dy[..., 1])
                   - (inverse[0, 1] + dy[..., 0]) * (inverse[1, 0] + dx[..., 1]))
            orientation_ok = (det * det_base > 0) & (np.abs(det) > 1e-7)
        else:
            residual = current @ inverse[:2, :2].T + inverse[:2, 2] - target
        valid = (np.isfinite(current).all(axis=-1) & np.isfinite(residual).all(axis=-1)
                 & (np.max(np.abs(residual), axis=-1) <= tolerance) & orientation_ok
                 & (current[..., 0] >= -1e-4) & (current[..., 0] <= w - 1 + 1e-4)
                 & (current[..., 1] >= -1e-4) & (current[..., 1] <= h - 1 + 1e-4))
        current[..., 0] = np.clip(current[..., 0], 0, w - 1)
        current[..., 1] = np.clip(current[..., 1], 0, h - 1)
        output[row:row + xx.shape[0]] = np.where(valid[..., None], current, -1.)
        validity[row:row + xx.shape[0]] = valid
    return output, validity


@dataclass
class CpuSpatialReplay:
    forward_grid: np.ndarray | None
    inverse_grid: np.ndarray
    valid: np.ndarray
    parameters: Any = None

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.forward_grid, self.inverse_grid, self.valid) if x is not None)

    def restore_planes(self, planes: Sequence[np.ndarray]) -> list[np.ndarray]:
        h, w = self.valid.shape
        output = []
        for value in planes:
            plane = np.asarray(value)
            if plane.shape != (h, w):
                plane = cv2.resize(plane, (w, h), interpolation=cv2.INTER_NEAREST)
            restored = cv2.remap(plane, self.inverse_grid[..., 0], self.inverse_grid[..., 1],
                                 cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            output.append(np.ascontiguousarray(np.where(self.valid, restored, 0), dtype=plane.dtype))
        return output

    def packed_validity(self) -> np.ndarray:
        return np.packbits(self.valid, axis=1, bitorder='big')


class CpuPolicyAdapter:
    def __init__(self, policy: Any, *, cache_bytes: int = 512 * 1024**2):
        self.policy, self.cache_bytes = policy, max(0, int(cache_bytes))
        self._cache: OrderedDict[tuple[int, int, int], CpuSpatialReplay] = OrderedDict()
        self._cache_size = 0
        self.custom = callable(getattr(policy, 'apply_tta_batch', None))
        required = ('_sample_parameters', '_forward_matrix', '_elastic_displacement', '_apply_intensity_noise')
        if not self.custom and (getattr(policy, 'tta_replay_contract', '') != 'opencv-affine-elastic-v1'
                                or any(not callable(getattr(policy, name, None)) for name in required)):
            raise TypeError('CPU policy has no supported inverse contract. Use a shipped CPU_*.py profile '
                            'or implement apply_tta_batch(images=..., seeds=...) returning uint8 NHWC images, '
                            'normalized inverse_grid and boolean valid.')

    def _replay(self, seed: int, height: int, width: int) -> CpuSpatialReplay:
        key = (int(seed), height, width)
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache[key] = old
            return old
        params = self.policy._sample_parameters(int(seed), height, width)
        forward = np.asarray(self.policy._forward_matrix(params, height, width), dtype=np.float64)
        displacement = self.policy._elastic_displacement(int(seed), height, width) if params['elastic'] else None
        inverse_grid, valid = inverse_policy_grid_cpu(forward, displacement, height, width)
        inv = np.linalg.inv(forward)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        source = np.stack(((inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]).astype(np.float32),
                           (inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]).astype(np.float32)), axis=-1)
        if displacement is not None:
            source += displacement
        replay = CpuSpatialReplay(source, inverse_grid, valid, params)
        while self._cache and self._cache_size + replay.nbytes > self.cache_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cache_size -= evicted.nbytes
        if replay.nbytes <= self.cache_bytes:
            self._cache[key] = replay
            self._cache_size += replay.nbytes
        return replay

    def apply(self, images: Sequence[np.ndarray], seeds: Sequence[int]):
        arrays = [np.asarray(im) for im in images]
        if not arrays or len(arrays) != len(seeds):
            raise ValueError('CPU policy requires one integer seed per sample')
        if any(im.dtype != np.uint8 or im.ndim not in (2, 3) for im in arrays):
            raise ValueError('CPU policy input must be uint8 HxW or HxWxC')
        h, w = arrays[0].shape[:2]
        if any(im.shape != arrays[0].shape for im in arrays):
            raise ValueError('CPU policy batch image shapes must match')
        if self.custom:
            batch = np.stack([im[..., None] if im.ndim == 2 else im for im in arrays])
            result = self.policy.apply_tta_batch(images=batch.copy(), seeds=tuple(int(s) for s in seeds))
            if not isinstance(result, dict) or not all(k in result for k in ('images', 'inverse_grid', 'valid')):
                raise TypeError('CPU apply_tta_batch must return images, inverse_grid and valid')
            output, inverse, valid = (np.asarray(result[k]) for k in ('images', 'inverse_grid', 'valid'))
            if output.shape != batch.shape or output.dtype != np.uint8:
                raise ValueError('CPU custom policy images must preserve uint8 NHWC shape')
            if (inverse.shape != (len(arrays), h, w, 2) or valid.shape != (len(arrays), h, w)
                    or valid.dtype != np.bool_ or inverse.dtype.kind != 'f'):
                raise ValueError('CPU custom replay must be normalized NxHxWx2 float coordinates and NxHxW bool support')
            valid = valid & np.isfinite(inverse).all(axis=-1) & (np.abs(inverse) <= 1.0001).all(axis=-1)
            pixels = (np.clip(inverse, -1., 1.) + 1.) * np.asarray([(w - 1) / 2., (h - 1) / 2.])
            pixels = np.where(valid[..., None], pixels, -1.).astype(np.float32)
            return ([np.ascontiguousarray(im[..., 0] if arrays[0].ndim == 2 else im) for im in output],
                    [CpuSpatialReplay(None, pixels[i], valid[i]) for i in range(len(arrays))])
        output, replays = [], []
        for im, seed in zip(arrays, seeds):
            replay = self._replay(int(seed), h, w)
            warped = cv2.remap(im.astype(np.float32) / 255., replay.forward_grid[..., 0],
                               replay.forward_grid[..., 1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if im.ndim == 3 and warped.ndim == 2:
                warped = warped[..., None]
            changed = self.policy._apply_intensity_noise(np.ascontiguousarray(warped, dtype=np.float32),
                                                         seed=int(seed), params=replay.parameters)
            if changed.shape != im.shape or not np.isfinite(changed).all():
                raise ValueError('CPU policy must preserve image shape and produce finite intensities')
            output.append(np.ascontiguousarray(np.clip(np.rint(changed * 255.), 0, 255), dtype=np.uint8))
            replays.append(replay)
        return output, replays


_ADAPTERS: dict[tuple[str, int], CpuPolicyAdapter] = {}


def worker_cpu_policy(settings: TtaAugmentationSettings) -> CpuPolicyAdapter:
    from .augmentation_policy import inspect_augmentation_definition
    from .pta_augmentation import _load_external_python_module
    import copy
    settings = settings.for_backend('cpu')
    key = (settings.content_sha256, settings.cache_mib)
    if key not in _ADAPTERS:
        settings.assert_unchanged()
        definition = inspect_augmentation_definition(settings.path)
        if definition.content_sha256 != settings.content_sha256:
            raise RuntimeError('CPU augmentation policy changed while loading')
        module = _load_external_python_module(definition.path, definition.content_sha256)
        exported = getattr(module, definition.export_name)
        if definition.export_name == 'build_augmentation':
            policy = exported()
        elif definition.export_name == 'augmentation':
            policy = copy.deepcopy(exported)
        else:
            raise TypeError('CPU TTA requires a replay-capable build_augmentation or augmentation export')
        adapter = CpuPolicyAdapter(policy, cache_bytes=settings.cache_mib * 1024**2)
        settings.assert_unchanged()
        _ADAPTERS[key] = adapter
    return _ADAPTERS[key]
