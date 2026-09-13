"""GPU policy replay and conservative inverse maps for TTA.

The four shipped GPU_*.py files are deliberately unchanged. Their versioned
fused-grid API supplies deterministic affine/elastic sampling and photometry.
New policies may instead implement apply_tta_batch(images=..., seeds=...).
Torch is imported only while executing numerical work, never for CLI discovery.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .tta_augmentation_config import TtaAugmentationSettings


def _normalized_grid(xy: Any, height: int, width: int) -> Any:
    import torch
    x = xy[..., 0] * (2.0 / (width - 1)) - 1.0 if width > 1 else torch.zeros_like(xy[..., 0])
    y = xy[..., 1] * (2.0 / (height - 1)) - 1.0 if height > 1 else torch.zeros_like(xy[..., 1])
    return torch.stack((x, y), dim=-1)


def _bilinear_displacement(field: Any, xy: Any) -> tuple[Any, Any, Any]:
    """Bilinear displacement plus its exact within-cell x/y derivatives.

    Border extension is used ONLY by the solver. Final out-of-frame samples
    are invalidated, not interpreted as a negative model prediction.
    """
    import torch
    _, h, w = field.shape
    x, y = xy[..., 0].clamp(0, w - 1), xy[..., 1].clamp(0, h - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=w - 1), (y0 + 1).clamp(max=h - 1)
    flat = field.reshape(2, -1)
    def at(xx: Any, yy: Any) -> Any:
        return flat[:, (yy * w + xx).reshape(-1)].reshape(2, *x.shape).movedim(0, -1)
    a, b, c, d = at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1)
    u, v = (x - x0).unsqueeze(-1), (y - y0).unsqueeze(-1)
    value = (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v
    dx = (b - a) * (1 - v) + (d - c) * v
    dy = (c - a) * (1 - u) + (d - b) * u
    dx = torch.where(((xy[..., 0] >= 0) & (xy[..., 0] <= w - 1)).unsqueeze(-1), dx, 0.)
    dy = torch.where(((xy[..., 1] >= 0) & (xy[..., 1] <= h - 1)).unsqueeze(-1), dy, 0.)
    return value, dx, dy


def inverse_policy_grid(forward: Any, displacement: Any | None, height: int, width: int,
                        *, iterations: int = 16, tolerance: float = 0.05,
                        strip_rows: int = 128) -> tuple[Any, Any]:
    """Invert x = inv(A) y + d(y), retaining only trustworthy observations.

    Newton iteration uses the exact bilinear Jacobian of d. A fixed iteration
    budget avoids a device synchronization per iteration. Unsupported/folded or
    unconverged pixels are unknown. This is not a claim of global invertibility
    of an arbitrary elastic field, nor a reconstruction of cropped information.
    """
    import torch
    from .tta_augmentation_cuda import policy_grids_cuda
    accelerated = policy_grids_cuda(forward, displacement, height, width,
                                    iterations=iterations, tolerance=tolerance)
    if accelerated is not None:
        return accelerated[1:]
    h, w = int(height), int(width)
    inverse = torch.linalg.inv(forward.float())
    output = torch.empty((h, w, 2), device=forward.device, dtype=torch.float32)
    validity = torch.empty((h, w), device=forward.device, dtype=torch.bool)
    det_base = inverse[0, 0] * inverse[1, 1] - inverse[0, 1] * inverse[1, 0]
    for row in range(0, h, max(1, strip_rows)):
        yy, xx = torch.meshgrid(torch.arange(row, min(h, row + strip_rows), device=forward.device, dtype=torch.float32),
                                torch.arange(w, device=forward.device, dtype=torch.float32), indexing='ij')
        target = torch.stack((xx, yy), dim=-1)
        current = target @ forward[:2, :2].float().T + forward[:2, 2].float()
        orientation_ok = torch.ones_like(xx, dtype=torch.bool)
        if displacement is not None:
            # Fixed-count iterations remain fully on the producing CUDA stream.
            for _ in range(int(iterations)):
                delta, dx, dy = _bilinear_displacement(displacement, current)
                residual = current @ inverse[:2, :2].T + inverse[:2, 2] + delta - target
                a, b = inverse[0, 0] + dx[..., 0], inverse[0, 1] + dy[..., 0]
                c, d = inverse[1, 0] + dx[..., 1], inverse[1, 1] + dy[..., 1]
                det = a * d - b * c
                denominator = torch.where(det.abs() > 1e-7, det, torch.ones_like(det))
                step = torch.stack(((d * residual[..., 0] - b * residual[..., 1]) / denominator,
                                    (-c * residual[..., 0] + a * residual[..., 1]) / denominator), dim=-1)
                current = current - step.clamp(-32., 32.)
                current = torch.nan_to_num(current, nan=-1e6, posinf=1e6, neginf=-1e6)
            delta, dx, dy = _bilinear_displacement(displacement, current)
            residual = current @ inverse[:2, :2].T + inverse[:2, 2] + delta - target
            det = ((inverse[0, 0] + dx[..., 0]) * (inverse[1, 1] + dy[..., 1])
                   - (inverse[0, 1] + dy[..., 0]) * (inverse[1, 0] + dx[..., 1]))
            orientation_ok = (det * det_base > 0) & (det.abs() > 1e-7)
        else:
            residual = current @ inverse[:2, :2].T + inverse[:2, 2] - target
        # A tiny arithmetic tolerance admits exact rotations whose edge centers
        # differ from zero by float32 roundoff; larger excursions remain unknown.
        edge_epsilon = 1e-4
        valid = (torch.isfinite(current).all(dim=-1) & torch.isfinite(residual).all(dim=-1)
                 & (residual.abs().amax(dim=-1) <= float(tolerance)) & orientation_ok
                 & (current[..., 0] >= -edge_epsilon) & (current[..., 0] <= w - 1 + edge_epsilon)
                 & (current[..., 1] >= -edge_epsilon) & (current[..., 1] <= h - 1 + edge_epsilon))
        current[..., 0].clamp_(0, w - 1)
        current[..., 1].clamp_(0, h - 1)
        grid = _normalized_grid(current, h, w)
        output[row:row + xx.shape[0]] = torch.where(valid.unsqueeze(-1), grid, torch.full_like(grid, 2.))
        validity[row:row + xx.shape[0]] = valid
    return output, validity


@dataclass
class SpatialReplay:
    forward_grid: Any | None
    inverse_grid: Any
    valid: Any
    parameters: Any = None

    @property
    def nbytes(self) -> int:
        return sum(int(x.numel()) * int(x.element_size()) for x in
                   (self.forward_grid, self.inverse_grid, self.valid) if x is not None)

    def restore_planes(self, planes: Sequence[Any]) -> list[Any]:
        import torch
        import torch.nn.functional as F
        h, w = self.valid.shape
        prepared = []
        for value in planes:
            value = value.float().reshape(1, 1, *value.shape[-2:])
            if tuple(value.shape[-2:]) != (h, w):
                value = F.interpolate(value, size=(h, w), mode='nearest')
            prepared.append(value)
        if not prepared:
            return []
        restored = F.grid_sample(torch.cat(prepared, dim=1), self.inverse_grid.unsqueeze(0),
                                 mode='nearest', padding_mode='zeros', align_corners=True)
        restored = restored * self.valid[None, None]
        return [restored[0, i].contiguous() for i in range(len(prepared))]

    def pack_validity_tensor(self) -> Any:
        """Pack one bit per raster pixel without a CUDA synchronization."""
        import torch
        import torch.nn.functional as F
        from .tta_augmentation_cuda import pack_validity_cuda
        packed = pack_validity_cuda(self.valid)
        if packed is not None:
            return packed
        width = int(self.valid.shape[1])
        padded = F.pad(self.valid.to(torch.uint8), (0, (-width) % 8))
        shifts = torch.arange(7, -1, -1, device=padded.device, dtype=torch.int64)
        packed = (padded.reshape(padded.shape[0], -1, 8).long() << shifts).sum(-1).to(torch.uint8)
        return packed

    def packed_validity(self) -> np.ndarray:
        """Synchronous compatibility wrapper for CPU callers and diagnostics."""
        return self.pack_validity_tensor().cpu().numpy()


class GpuPolicyAdapter:
    """An explicit compatibility adapter, not a guessed inverse for arbitrary policies."""
    def __init__(self, policy: Any, runtime_name: str, *, cache_bytes: int = 512 * 1024**2,
                 require_cuda: bool = True) -> None:
        self.policy = policy
        self.require_cuda = bool(require_cuda)
        self.cache_bytes = max(0, int(cache_bytes))
        self._cache: OrderedDict[tuple[int, int, int], SpatialReplay] = OrderedDict()
        self._cache_size = 0
        self.custom = callable(getattr(policy, 'apply_tta_batch', None))
        required = ('_sample_parameters', '_forward_matrix', '_elastic_displacement', '_pixel_grid', '_apply_intensity_noise')
        if not self.custom and (runtime_name != 'torch-cuda-fused-grid-v1' or
                                any(not callable(getattr(policy, name, None)) for name in required)):
            raise TypeError('GPU policy has no supported inverse contract. Use an unchanged shipped GPU_*.py profile '
                            'or implement apply_tta_batch(images=..., seeds=...) returning images, inverse_grid, valid.')

    def _replay(self, seed: int, height: int, width: int) -> SpatialReplay:
        import torch
        key = (int(seed), int(height), int(width))
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache[key] = old
            return old
        params = self.policy._sample_parameters(int(seed), int(height), int(width))
        forward = torch.tensor(self.policy._forward_matrix(params, height, width),
                               device=self.policy.device, dtype=torch.float32)
        displacement = None
        if bool(params['elastic']):
            displacement = self.policy._elastic_displacement(int(seed), height, width)
        from .tta_augmentation_cuda import policy_grids_cuda
        accelerated = policy_grids_cuda(forward, displacement, height, width, source=True)
        if accelerated is None:
            source = torch.einsum('ij,hwj->hwi', torch.linalg.inv(forward), self.policy._pixel_grid(height, width))[..., :2]
            if displacement is not None:
                source = source + displacement.movedim(0, -1)
            inverse, valid = inverse_policy_grid(forward, displacement, height, width)
            forward_grid = _normalized_grid(source, height, width)
        else:
            forward_grid, inverse, valid = accelerated
        replay = SpatialReplay(forward_grid, inverse, valid, params)
        while self._cache and self._cache_size + replay.nbytes > self.cache_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cache_size -= evicted.nbytes
        if replay.nbytes <= self.cache_bytes:
            self._cache[key] = replay
            self._cache_size += replay.nbytes
        return replay

    def apply(self, images: Any, seeds: Sequence[int]) -> tuple[Any, list[SpatialReplay]]:
        import torch
        import torch.nn.functional as F
        if images.ndim != 4 or int(images.shape[0]) != len(seeds):
            raise ValueError('TTA policy input must be NCHW with one integer seed per sample')
        if self.require_cuda and not images.is_cuda:
            raise ValueError('TTA augmentation cannot run on a CPU tensor')
        with torch.inference_mode():
            if self.custom:
                result = self.policy.apply_tta_batch(images=images, seeds=tuple(int(s) for s in seeds))
                if not isinstance(result, dict) or not all(k in result for k in ('images', 'inverse_grid', 'valid')):
                    raise TypeError('apply_tta_batch must return a dict with images, inverse_grid, valid')
                output, inverse, valid = result['images'], result['inverse_grid'], result['valid']
                n, _, h, w = images.shape
                if (not all(torch.is_tensor(t) for t in (output, inverse, valid)) or
                    tuple(output.shape) != tuple(images.shape) or tuple(inverse.shape) != (n, h, w, 2) or
                    tuple(valid.shape) != (n, h, w) or any(t.device != images.device for t in (output, inverse, valid))):
                    raise ValueError('apply_tta_batch returned an invalid shape or device')
                if not output.is_floating_point() or inverse.dtype != torch.float32 or valid.dtype != torch.bool:
                    raise TypeError('images must be floating point, inverse_grid float32, and valid bool')
                if not bool((torch.isfinite(output).all() & (output >= 0).all() & (output <= 1).all()).item()):
                    raise ValueError('apply_tta_batch images must be finite normalized values in [0, 1]')
                supported = torch.isfinite(inverse).all(-1) & (inverse.abs() <= 1.0001).all(-1)
                if bool((valid & ~supported).any().item()):
                    raise ValueError('apply_tta_batch marks an out-of-frame or non-finite inverse as valid')
                inverse = torch.where(valid[..., None], inverse.clamp(-1, 1), torch.full_like(inverse, 2.))
                # Custom hooks own photometry/channel policy, never the caller.
                return output.to(images.dtype), [SpatialReplay(None, inverse[i], valid[i]) for i in range(n)]
            height, width = (int(v) for v in images.shape[-2:])
            # A shared view/slab seed needs one replay even when the persistent
            # cache is disabled or too small to retain a full-resolution map.
            current_replays = {}
            replay = []
            for seed in seeds:
                key = int(seed)
                if key not in current_replays:
                    current_replays[key] = self._replay(key, height, width)
                replay.append(current_replays[key])
            grid = (replay[0].forward_grid.unsqueeze(0).expand(len(replay),-1,-1,-1)
                    if len(current_replays)==1 else torch.stack([value.forward_grid for value in replay]))
            # Match apply_batch_many's uint8 input/output boundary, without a host mask or image round-trip.
            from .tta_augmentation_cuda import quantize_boundary_cuda
            image_float = quantize_boundary_cuda(images, dtype=torch.float32, clamp_input=True)
            if image_float is None:
                image_float = (images.float().clamp(0, 1) * 255.).round() / 255.
            warped = F.grid_sample(image_float, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            warped = self.policy._apply_intensity_noise(warped, seeds, [value.parameters for value in replay])
            output = quantize_boundary_cuda(warped, dtype=images.dtype, clamp_input=False)
            if output is None:
                output = ((warped * 255.).round().clamp(0, 255.) / 255.).to(images.dtype).contiguous()
            return output, replay

    def clear(self) -> None:
        self._cache.clear()
        self._cache_size = 0
        cache = getattr(self.policy, '_pixel_grid_cache', None)
        if isinstance(cache, dict):
            cache.clear()


_ADAPTERS: dict[tuple[str, str, int], GpuPolicyAdapter] = {}


def worker_policy(settings: TtaAugmentationSettings, *, device: str, batch_size: int) -> GpuPolicyAdapter:
    from .pta_augmentation import load_gpu_augmentation_definition
    key = (settings.content_sha256, str(device), int(batch_size))
    if key not in _ADAPTERS:
        settings.assert_unchanged()
        loaded = load_gpu_augmentation_definition(settings.path)
        if loaded.content_sha256 != settings.content_sha256:
            raise RuntimeError('Augmentation policy changed while loading')
        # TTA-only hooks need not implement PTA's image/mask apply_batch method.
        policy = loaded.policy_builder(device=str(device), batch_size=int(batch_size))
        settings.assert_unchanged()
        adapter = GpuPolicyAdapter(policy, loaded.runtime_name,
                                   cache_bytes=settings.cache_mib * 1024**2)
        if not adapter.custom:
            from .tta_augmentation_cuda import require_policy_cuda
            require_policy_cuda(str(device))
        _ADAPTERS[key] = adapter
    return _ADAPTERS[key]


def clear_worker_policies() -> None:
    for adapter in _ADAPTERS.values():
        adapter.clear()
    _ADAPTERS.clear()
