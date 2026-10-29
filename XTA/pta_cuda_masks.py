"""Resident categorical PTA projection for GPU augmentation workers.

The image renderer already owns the processing cube on each GPU.  This module
keeps the corresponding foreground and (when present) annotation-coverage
volumes on that GPU for the lifetime of one shared-memory volume generation.
Family-specific projectors produce a categorical item directly in its final
full-frame or tile raster, so neither mask traverses a CPU projection loop.

Importing this module does not import PyTorch or initialize CUDA.  Each worker
creates its owner lazily after selecting its assigned CUDA device.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from . import geometry as shared_geometry


_GIB = 1024 ** 3
_OWNER_CREATION_LOCK = threading.Lock()


def _reserve_bytes() -> int:
    raw = os.environ.get("PTA_GPU_CATEGORICAL_RESERVE_MIB", "2048")
    try:
        mib = int(raw)
    except ValueError as exc:
        raise ValueError("PTA_GPU_CATEGORICAL_RESERVE_MIB must be an integer") from exc
    if mib < 0:
        raise ValueError("PTA_GPU_CATEGORICAL_RESERVE_MIB must be nonnegative")
    return mib * 1024 ** 2


def _array_identity(array: np.ndarray) -> Tuple[int, Tuple[int, int, int]]:
    return (
        int(array.__array_interface__["data"][0]),
        tuple(int(v) for v in array.shape),
    )


def _validate_source(array: np.ndarray, role: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"PTA {role} must be a NumPy array")
    if array.dtype != np.uint8 or array.ndim != 3 or not array.flags.c_contiguous:
        raise ValueError(
            f"PTA {role} must be contiguous uint8 (t,Y,X), got "
            f"shape={array.shape}, dtype={array.dtype}, contiguous={array.flags.c_contiguous}"
        )
    if min(array.shape) < 1:
        raise ValueError(f"PTA {role} has invalid shape {array.shape}")


@dataclass(frozen=True)
class _ItemGrid:
    forward: np.ndarray
    inverse: np.ndarray
    height: int
    width: int


def _item_grid(plan: object, item_key: str) -> Optional[_ItemGrid]:
    if str(item_key) == "full":
        aff = plan.aff
        forward = aff.M_src_to_out
        inverse = aff.M_out_to_src
        height, width = int(aff.out_h), int(aff.out_w)
    else:
        tile = next(
            (t for t in plan.tile_layout if str(t.tile_tag) == str(item_key)),
            None,
        )
        if tile is None:
            return None
        if tile.shared_job is None:
            if getattr(plan.view, "family", None) != "cpr":
                return None
            # CPR uses the actual native parent -> canvas -> crop -> resize
            # sequence below; this grid supplies the final admission/shape size.
            forward = plan.aff.M_src_to_canvas
            inverse = plan.aff.M_canvas_to_src
        else:
            forward = tile.shared_job.M_src_to_out
            inverse = tile.shared_job.M_out_to_src
        height, width = int(tile.out_h), int(tile.out_w)
    if height < 1 or width < 1:
        raise ValueError(f"PTA categorical output has invalid size {height}x{width}")
    return _ItemGrid(
        np.asarray(forward, dtype=np.float32).reshape(2, 3),
        np.asarray(inverse, dtype=np.float32).reshape(2, 3),
        height,
        width,
    )


def _family(view: object) -> Optional[str]:
    if view is None:
        return None
    if getattr(view, "family", None) == "cpr" and getattr(view, "cpr_geometry", None) is not None:
        return "cpr"
    if shared_geometry.is_spherical_view(view):
        return "shell"
    if shared_geometry.is_radial_view(view):
        return "shell"
    if shared_geometry.is_azimuthal_view(view):
        return "azimuthal"
    if shared_geometry.is_tilted_view(view):
        return "cartesian"
    if shared_geometry.physical_view_name(view) in {"transverse", "sagittal", "coronal"}:
        return "cartesian"
    return None


def _cpr_native_cuda(source, geometry, frame_idx, *, torch, stream, categorical=False, resize_to_t=None):
    """Borrow an existing Torch source allocation; only the output plane is new."""
    import cupy as cp
    from .geometry import _cupy_external_stream
    from .pta_cpr_cuda import render_cpr_frame_cuda

    with torch.cuda.device(source.device), torch.cuda.stream(stream), cp.cuda.Device(int(source.device.index)):
        result = torch.empty((geometry.height, geometry.width), dtype=torch.uint8, device=source.device)
        with _cupy_external_stream(cp, stream):
            source_cp = cp.asarray(source)
            output_cp = cp.asarray(result)
            render_cpr_frame_cuda(
                source_cp, geometry, int(frame_idx), categorical=bool(categorical),
                resize_to_t=resize_to_t, out=output_cp,
            )
        # Return the original allocator-owned Torch tensor. A foreign CuPy
        # allocation rewrapped through DLPack cannot honor Torch record_stream
        # when the producer LRU evicts it before asynchronous consumers finish.
        result.record_stream(stream)
        return result


def _cpr_categorical_warp(native, forward, height, width, *, torch, cache=None):
    """Apply the legacy OpenCV nearest affine to an already sampled CUDA plane."""
    height, width = int(height), int(width)
    matrix = np.asarray(forward, np.float32).reshape(2, 3)
    if (tuple(native.shape) == (height, width)
            and np.array_equal(matrix, np.asarray([[1, 0, 0], [0, 1, 0]], np.float32))):
        return native
    key = (tuple(int(v) for v in native.shape), height, width, matrix.tobytes())
    addresses = None if cache is None else cache.get(key)
    if addresses is None:
        import cv2
        # OpenCV can choose IPP or its fixed-point nearest path for different
        # transforms. Warping synthetic index bytes with the identical uint8
        # single-channel contract captures that choice exactly. This bounded
        # geometry setup reads no source pixels; native labels stay on device.
        count = int(native.shape[0]) * int(native.shape[1])
        dtype = np.uint32 if count < 2**32 else np.uint64
        source_indices = np.arange(1, count + 1, dtype=dtype).reshape(tuple(native.shape))
        selected = np.zeros((height, width), dtype=dtype)
        for shift in range(0, max(8, ((count.bit_length() + 7) // 8) * 8), 8):
            byte = np.ascontiguousarray((source_indices >> shift).astype(np.uint8))
            warped = cv2.warpAffine(byte, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            selected |= warped.astype(dtype) << shift
        addresses = torch.as_tensor(selected.astype(np.int64) - 1, device=native.device)
        if cache is not None:
            cache[key] = addresses
            while len(cache) > 4:
                cache.popitem(last=False)
    if cache is not None:
        cache.move_to_end(key)
    sampled = native.reshape(-1)[addresses.clamp(0, int(native.numel()) - 1)]
    return torch.where(addresses >= 0, sampled, 0).to(torch.uint8).contiguous()


def _cpr_intensity_warp(native, forward, height, width, *, torch):
    """Match the existing post-quantization OpenCV bilinear affine grid."""
    from .pta_cuda_cartesian import _cv_inverse_for_nearest

    height, width = int(height), int(width)
    matrix = np.asarray(forward, np.float32).reshape(2, 3)
    if (tuple(native.shape) == (height, width)
            and np.array_equal(matrix, np.asarray([[1, 0, 0], [0, 1, 0]], np.float32))):
        return native
    inverse = _cv_inverse_for_nearest(matrix).astype(np.float64)
    xs = torch.arange(width, device=native.device, dtype=torch.float64)[None, :]
    ys = torch.arange(height, device=native.device, dtype=torch.float64)[:, None]
    sx = xs * float(inverse[0, 0]) + ys * float(inverse[0, 1]) + float(inverse[0, 2])
    sy = xs * float(inverse[1, 0]) + ys * float(inverse[1, 1]) + float(inverse[1, 2])
    x0, y0 = torch.floor(sx).to(torch.int64), torch.floor(sy).to(torch.int64)
    ax, ay = sx - x0, sy - y0
    total = torch.zeros((height, width), dtype=torch.float64, device=native.device)
    for ox, oy, weight in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)),
                           (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        ix, iy = x0 + ox, y0 + oy
        valid = (ix >= 0) & (ix < int(native.shape[1])) & (iy >= 0) & (iy < int(native.shape[0]))
        value = native[iy.clamp(0, int(native.shape[0]) - 1), ix.clamp(0, int(native.shape[1]) - 1)]
        total += torch.where(valid, value.to(torch.float64), 0) * weight
    return total.round().clamp_(0, 255).to(torch.uint8).contiguous()


def _cpr_item_from_native(native, plan, item_key, *, torch, categorical, affine_cache=None):
    """Preserve native quantization -> affine -> padded crop -> output resize."""
    tile = None if str(item_key) == "full" else next(
        (tile for tile in plan.tile_layout if str(tile.tile_tag) == str(item_key)), None)
    if str(item_key) != "full" and tile is None:
        raise ValueError(f"Unknown CPR item {item_key!r} for {plan.tag}")
    aff = plan.aff
    matrix = aff.M_src_to_out if tile is None else aff.M_src_to_canvas
    height, width = ((int(aff.out_h), int(aff.out_w)) if tile is None
                     else (int(aff.canvas_h), int(aff.canvas_w)))
    if categorical:
        output = _cpr_categorical_warp(native, matrix, height, width, torch=torch, cache=affine_cache)
    else:
        output = _cpr_intensity_warp(native, matrix, height, width, torch=torch)
    if tile is None:
        return output
    side = int(tile.cfg.tile_size)
    x, y = int(tile.x), int(tile.y)
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(width, x + side), min(height, y + side)
    if sx0 == x and sy0 == y and sx1 == x + side and sy1 == y + side:
        cropped = output[sy0:sy1, sx0:sx1]
    else:
        cropped = torch.zeros((side, side), dtype=torch.uint8, device=output.device)
        if sx0 < sx1 and sy0 < sy1:
            cropped[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = output[sy0:sy1, sx0:sx1]
    target_h, target_w = int(tile.out_h), int(tile.out_w)
    if tuple(cropped.shape) == (target_h, target_w):
        return cropped.contiguous()
    if categorical:
        ys = torch.arange(target_h, device=cropped.device, dtype=torch.int64) * side // target_h
        xs = torch.arange(target_w, device=cropped.device, dtype=torch.int64) * side // target_w
        return cropped[ys[:, None], xs[None, :]].contiguous()
    from torch.nn import functional as F
    return F.interpolate(cropped.to(torch.float32)[None, None], size=(target_h, target_w),
                         mode="bilinear", align_corners=False)[0, 0].round().clamp_(0, 255).to(torch.uint8).contiguous()


def _require_canonical_cuda_plan(plan: object, item_key: str) -> None:
    """Bind this GPU categorical render to the registered sampling policy."""
    if getattr(plan.view, "family", None) == "cpr":
        if getattr(plan.view, "cpr_geometry", None) is None:
            raise RuntimeError("CPR categorical item has no sampling geometry")
        return
    from .unification.contracts import DataRole
    from .unification.sampling import forward_sampling_policy, require_forward_sampling

    if str(item_key) == "full":
        canonical = getattr(plan, "canonical_plan", None)
    else:
        tile = next(
            (t for t in plan.tile_layout if str(t.tile_tag) == str(item_key)),
            None,
        )
        canonical = getattr(tile, "canonical_plan", None)
    if canonical is None:
        raise RuntimeError(
            f"Shared PTA categorical item {getattr(plan, 'tag', '')}/{item_key} "
            "has no canonical RasterPlan"
        )
    policy = forward_sampling_policy()
    if canonical.sampling_policy.digest != policy.digest:
        raise RuntimeError(
            f"PTA categorical RasterPlan policy drift for "
            f"{getattr(plan, 'tag', '')}/{item_key}: "
            f"plan={canonical.sampling_policy.digest}, current={policy.digest}"
        )
    require_forward_sampling("cuda", DataRole.CATEGORICAL_GROUND_TRUTH)


class CategoricalVolumeOwner:
    """One worker's bounded foreground/coverage residency and producer stream."""

    def __init__(self, torch: object, device_id: int, stream: object):
        self.torch = torch
        self.device_id = int(device_id)
        self.device = f"cuda:{self.device_id}"
        self.stream = stream
        self.lock = threading.RLock()
        self.key: Optional[tuple] = None
        self.mask_gpu: Optional[object] = None
        self.coverage_gpu: Optional[object] = None
        self.rejected_key: Optional[tuple] = None
        self.rejected_reason = ""
        self._capacity_rejected_at: Optional[float] = None
        self._capacity_retry_eligible = False
        from .pta_cuda_azimuthal import AzimuthalTableCache
        from .runtime import runtime_telemetry
        self.azimuthal_tables = AzimuthalTableCache()
        self.cpr_native_planes = OrderedDict()
        self.cpr_affine_maps = OrderedDict()
        runtime_telemetry().register_sample_provider('pta.cuda.azimuthal_tables', self.azimuthal_tables.snapshot)

    def retire(self) -> None:
        """Fence this producer before dropping a former volume generation."""
        with self.lock:
            had_sources = self.mask_gpu is not None or self.coverage_gpu is not None
            if had_sources:
                self.stream.synchronize()
            self.azimuthal_tables.clear()
            self.cpr_native_planes.clear()
            self.cpr_affine_maps.clear()
            self.mask_gpu = None
            self.coverage_gpu = None
            self.key = None
            self.rejected_key = None
            self.rejected_reason = ""
            self._capacity_rejected_at = None
            self._capacity_retry_eligible = False
            if not had_sources:
                return
            shell_module = sys.modules.get(f"{__package__}.pta_cuda_shells")
            clear_shell_cache = getattr(shell_module, "clear_shell_direction_cache", None)
            if callable(clear_shell_cache):
                clear_shell_cache()
            empty_cache = getattr(self.torch.cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()

    def arm_residency_retry(self) -> bool:
        """Reconsider headroom only at a joined task boundary after five seconds.

        This method never probes or flushes CUDA, changes live source owners,
        or retries an allocator/transfer failure. The next ensure owns admission.
        """
        with self.lock:
            if (not self._capacity_retry_eligible or self.rejected_key is None
                    or self._capacity_rejected_at is None
                    or time.monotonic() - self._capacity_rejected_at < 5.0):
                return False
            self.rejected_key = None
            self.rejected_reason = ""
            self._capacity_rejected_at = None
            self._capacity_retry_eligible = False
            return True

    def _key(self, mask: np.ndarray, coverage: Optional[np.ndarray], identity: str) -> tuple:
        return (
            str(identity),
            _array_identity(mask),
            _array_identity(coverage) if coverage is not None else None,
        )

    def ensure(
        self,
        mask: np.ndarray,
        coverage: Optional[np.ndarray],
        *,
        identity: str,
        largest_output_bytes: int,
    ) -> bool:
        """Admit both volumes only if the complete batch fits with headroom.

        An insufficient-memory decision happens before any item is launched.
        Once transfer starts, upload failures propagate to the caller so a
        partially initialized CUDA source cannot be mistaken for a CPU retry.
        """
        _validate_source(mask, "foreground volume")
        if coverage is not None:
            _validate_source(coverage, "annotation coverage volume")
            if coverage.shape != mask.shape:
                raise ValueError("PTA categorical coverage shape differs from foreground")
        key = self._key(mask, coverage, identity)
        with self.lock:
            if self.key == key and self.mask_gpu is not None:
                return True
            if self.rejected_key == key:
                return False
            if self.key is not None:
                self.retire()
            self._capacity_rejected_at = None
            self._capacity_retry_eligible = False

            same_source = (
                coverage is not None and _array_identity(coverage) == _array_identity(mask)
            )
            new_bytes = int(mask.nbytes) + (
                0 if coverage is None or same_source else int(coverage.nbytes)
            )
            # Leave room for policy batches, encoded publication and several
            # simultaneous projected rasters.  Full 2048px foreground and
            # coverage cubes are ~15 GiB together; H100 residency is viable.
            temporary_bytes = max(256 * 1024 ** 2, 4 * int(largest_output_bytes))
            free_bytes, _total_bytes = self.torch.cuda.mem_get_info(self.device_id)
            need_bytes = new_bytes + temporary_bytes + _reserve_bytes()
            empty_cache = getattr(self.torch.cuda, "empty_cache", None)
            if int(free_bytes) < need_bytes and callable(empty_cache):
                # Returned source/cache blocks may still be reserved by the
                # allocator. One recovery probe precedes this cached decision.
                with self.torch.cuda.device(self.device_id):
                    empty_cache()
                    free_bytes, _total_bytes = self.torch.cuda.mem_get_info(self.device_id)
            if int(free_bytes) < need_bytes:
                self.rejected_key = key
                self.rejected_reason = (
                    f"need {need_bytes / _GIB:.1f} GiB including reserve, "
                    f"free {int(free_bytes) / _GIB:.1f} GiB"
                )
                self._capacity_rejected_at = time.monotonic()
                self._capacity_retry_eligible = True
                print(
                    f"PTA CUDA categorical projection cuda:{self.device_id}: "
                    f"source not resident ({self.rejected_reason}); CPU categorical fallback.",
                    flush=True,
                )
                return False

            # Allocate everything before beginning transfer.  An allocation
            # failure is still a safe prelaunch fallback; a transfer failure
            # raises and never silently crosses back to CPU.
            torch = self.torch
            plane_bytes = int(mask.shape[1]) * int(mask.shape[2])
            chunk = max(1, min(256, (512 * 1024 ** 2) // max(1, plane_bytes)))
            with torch.cuda.device(self.device_id), torch.cuda.stream(self.stream):
                try:
                    mask_gpu = torch.empty(mask.shape, dtype=torch.uint8, device=self.device)
                    coverage_gpu = (
                        mask_gpu if same_source else
                        torch.empty(coverage.shape, dtype=torch.uint8, device=self.device)
                        if coverage is not None else None
                    )
                except torch.cuda.OutOfMemoryError:
                    self.rejected_key = key
                    self.rejected_reason = "CUDA allocator rejected source allocation"
                    self._capacity_rejected_at = None
                    self._capacity_retry_eligible = False
                    print(
                        f"PTA CUDA categorical projection cuda:{self.device_id}: "
                        f"{self.rejected_reason}; CPU categorical fallback.",
                        flush=True,
                    )
                    return False
                sources = [(mask, mask_gpu)]
                if coverage is not None and not same_source:
                    sources.append((coverage, coverage_gpu))
                for source, dest in sources:
                    for start in range(0, int(source.shape[0]), chunk):
                        end = min(int(source.shape[0]), start + chunk)
                        dest[start:end].copy_(
                            torch.from_numpy(source[start:end]), non_blocking=False,
                        )
            self.mask_gpu = mask_gpu
            self.coverage_gpu = coverage_gpu
            self.key = key
            self.rejected_key = None
            self.rejected_reason = ""
            self._capacity_rejected_at = None
            self._capacity_retry_eligible = False
            print(
                f"PTA CUDA categorical projection cuda:{self.device_id}: "
                f"foreground{'+coverage' if coverage is not None else ''} resident "
                f"({new_bytes / _GIB:.1f} GiB); full/tile masks stay on GPU.",
                flush=True,
            )
            return True


def _owner_for_runtime(runtime: Mapping[str, object]) -> CategoricalVolumeOwner:
    if not isinstance(runtime, dict):
        raise TypeError("PTA CUDA categorical runtime must be mutable")
    with _OWNER_CREATION_LOCK:
        owner = runtime.get("categorical_volume_owner")
        if isinstance(owner, CategoricalVolumeOwner):
            return owner
        torch = runtime["torch"]
        device_id = int(runtime["device_id"])
        renderer = runtime.get("azimuthal_renderer")
        stream = getattr(renderer, "_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=device_id)
        owner = CategoricalVolumeOwner(torch, device_id, stream)
        from .pta_cuda_shells import shell_direction_cache_stats
        from .runtime import runtime_telemetry
        runtime_telemetry().register_sample_provider(
            'pta.cuda.spherical_directions', lambda: shell_direction_cache_stats(renderer),
        )
        runtime["categorical_volume_owner"] = owner
        # multiprocessing fork workers terminate through multiprocessing's
        # finalizer registry.  Run after the publication drain (priority 15),
        # while its CUDA context is still alive.
        from multiprocessing.util import Finalize

        runtime["categorical_volume_finalizer"] = Finalize(
            None, owner.retire, exitpriority=14,
        )
        return owner


def retire_gpu_categorical_volume(runtime: Mapping[str, object]) -> None:
    """Release resident mask/coverage on explicit worker-generation retirement."""
    if isinstance(runtime, dict):
        owner = runtime.get("categorical_volume_owner")
        if isinstance(owner, CategoricalVolumeOwner):
            owner.retire()


def render_gpu_categorical_item(
    runtime: Mapping[str, object],
    mask: np.ndarray,
    coverage: Optional[np.ndarray],
    plan: object,
    frame_idx: int,
    item_key: str,
    *,
    identity: Optional[str] = None,
) -> Optional[Tuple[object, Optional[object], object]]:
    """Render one canonical categorical item and return CUDA tensors + event.

    The event marks when both tensors are ready for a consumer stream.  The
    return is None only for an unsupported plan or a prelaunch memory rejection.
    Any error once a family kernel starts propagates to the render task.
    """
    if os.environ.get("YOLO_TTA_PTA_GPU_CATEGORICAL", "1").strip().lower() in {
        "0", "false", "no", "off", "disabled",
    }:
        return None
    physical = getattr(plan, "view", None)
    view = physical if getattr(physical, "family", None) == "cpr" else getattr(physical, "shared_view", None)
    family = _family(view)
    grid = _item_grid(plan, item_key)
    if family is None or grid is None:
        return None
    _require_canonical_cuda_plan(plan, item_key)
    owner = _owner_for_runtime(runtime)
    source_identity = identity or f"array:{_array_identity(mask)[0]}"
    with owner.lock:
        native_pixels = (int(view.cpr_geometry.height) * int(view.cpr_geometry.width)
                         if family == "cpr" else 0)
        if not owner.ensure(
            mask, coverage, identity=source_identity,
            largest_output_bytes=max(native_pixels, grid.height * grid.width) * (1 + (coverage is not None)),
        ):
            return None
        torch = owner.torch
        with torch.cuda.device(owner.device_id), torch.cuda.stream(owner.stream):
            if family == "cpr":
                geometry = view.cpr_geometry
                key = (id(geometry), int(frame_idx))
                cached = owner.cpr_native_planes.get(key)
                if cached is None or cached[0] is not geometry:
                    native_mask = _cpr_native_cuda(owner.mask_gpu, geometry, frame_idx,
                                                   torch=torch, stream=owner.stream, categorical=True)
                    native_coverage = (_cpr_native_cuda(owner.coverage_gpu, geometry, frame_idx,
                                                       torch=torch, stream=owner.stream, categorical=True)
                                       if coverage is not None else None)
                    cached = (geometry, native_mask, native_coverage)
                    owner.cpr_native_planes[key] = cached
                    while len(owner.cpr_native_planes) > 8:
                        owner.cpr_native_planes.popitem(last=False)
                owner.cpr_native_planes.move_to_end(key)
                pair = (
                    _cpr_item_from_native(cached[1], plan, item_key, torch=torch, categorical=True,
                                          affine_cache=getattr(owner, "cpr_affine_maps", None)),
                    (_cpr_item_from_native(cached[2], plan, item_key, torch=torch, categorical=True,
                                           affine_cache=getattr(owner, "cpr_affine_maps", None))
                     if cached[2] is not None else None),
                )
            elif family == "cartesian":
                from .pta_cuda_cartesian import (
                    render_categorical_item,
                    render_categorical_pair,
                )

                if coverage is None:
                    pair = (
                        render_categorical_item(
                            owner.mask_gpu, view, grid.inverse, int(frame_idx),
                            grid.height, grid.width,
                            M_src_to_out=grid.forward, stream=owner.stream,
                        ),
                        None,
                    )
                else:
                    pair = render_categorical_pair(
                        owner.mask_gpu, owner.coverage_gpu, view, grid.inverse,
                        int(frame_idx), grid.height, grid.width,
                        M_src_to_out=grid.forward, stream=owner.stream,
                    )
            elif family == "azimuthal":
                from .pta_cuda_azimuthal import render_azimuthal_categorical_pair

                pair = render_azimuthal_categorical_pair(
                    owner.mask_gpu, owner.coverage_gpu, view, int(frame_idx),
                    M_grid_to_src=grid.inverse, M_src_to_out=grid.forward,
                    out_h=grid.height, out_w=grid.width, stream=owner.stream,
                    table_cache=owner.azimuthal_tables,
                )
            else:
                from .pta_cuda_shells import render_shell_categorical_pair

                pair = render_shell_categorical_pair(
                    owner.mask_gpu, owner.coverage_gpu, view, int(frame_idx),
                    M_grid_to_src=grid.inverse, M_src_to_out=grid.forward,
                    out_h=grid.height, out_w=grid.width, stream=owner.stream,
                    direction_renderer=runtime.get("azimuthal_renderer"),
                )
            if pair is None or pair[0] is None:
                raise RuntimeError(
                    f"PTA CUDA categorical {family} projection declined after source admission"
                )
            mask_out, coverage_out = pair
            expected = (grid.height, grid.width)
            for role, output in (("foreground", mask_out), ("coverage", coverage_out)):
                if output is None:
                    continue
                if (
                    not bool(getattr(output, "is_cuda", False))
                    or output.dtype != torch.uint8
                    or tuple(int(v) for v in output.shape) != expected
                    or int(output.device.index) != owner.device_id
                ):
                    raise RuntimeError(
                        f"PTA CUDA categorical {role} output violates uint8 CUDA "
                        f"{expected} contract: shape={getattr(output, 'shape', None)}, "
                        f"dtype={getattr(output, 'dtype', None)}, "
                        f"device={getattr(output, 'device', None)}"
                    )
            if coverage is not None and coverage_out is None:
                raise RuntimeError("PTA CUDA categorical projector dropped coverage")
            ready = torch.cuda.Event()
            ready.record(owner.stream)
            return mask_out, coverage_out, ready


__all__ = [
    "CategoricalVolumeOwner",
    "render_gpu_categorical_item",
    "retire_gpu_categorical_volume",
]
