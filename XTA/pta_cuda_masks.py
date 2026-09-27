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
        if tile is None or tile.shared_job is None:
            return None
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


def _require_canonical_cuda_plan(plan: object, item_key: str) -> None:
    """Bind this GPU categorical render to the registered sampling policy."""
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

    def retire(self) -> None:
        """Fence this producer before dropping a former volume generation."""
        with self.lock:
            had_sources = self.mask_gpu is not None or self.coverage_gpu is not None
            if had_sources:
                self.stream.synchronize()
            self.mask_gpu = None
            self.coverage_gpu = None
            self.key = None
            self.rejected_key = None
            self.rejected_reason = ""
            if not had_sources:
                return
            shell_module = sys.modules.get(f"{__package__}.pta_cuda_shells")
            clear_shell_cache = getattr(shell_module, "clear_shell_direction_cache", None)
            if callable(clear_shell_cache):
                clear_shell_cache()
            empty_cache = getattr(self.torch.cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()

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
            if int(free_bytes) < need_bytes:
                self.rejected_key = key
                self.rejected_reason = (
                    f"need {need_bytes / _GIB:.1f} GiB including reserve, "
                    f"free {int(free_bytes) / _GIB:.1f} GiB"
                )
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
    view = getattr(getattr(plan, "view", None), "shared_view", None)
    family = _family(view)
    grid = _item_grid(plan, item_key)
    if family is None or grid is None:
        return None
    _require_canonical_cuda_plan(plan, item_key)
    owner = _owner_for_runtime(runtime)
    source_identity = identity or f"array:{_array_identity(mask)[0]}"
    with owner.lock:
        if not owner.ensure(
            mask, coverage, identity=source_identity,
            largest_output_bytes=grid.height * grid.width * (1 + (coverage is not None)),
        ):
            return None
        torch = owner.torch
        with torch.cuda.device(owner.device_id), torch.cuda.stream(owner.stream):
            if family == "cartesian":
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
                )
            else:
                from .pta_cuda_shells import render_shell_categorical_pair

                pair = render_shell_categorical_pair(
                    owner.mask_gpu, owner.coverage_gpu, view, int(frame_idx),
                    M_grid_to_src=grid.inverse, M_src_to_out=grid.forward,
                    out_h=grid.height, out_w=grid.width, stream=owner.stream,
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
