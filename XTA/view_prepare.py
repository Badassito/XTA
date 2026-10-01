"""Explicit admission and ownership state for parent view preparation.

The orchestrator still decides when a view is ready.  This module owns the
work submitted to the postprocess executor and the dense backing handoff so
both can be exercised without extracting a closure from ``pipeline``.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .config import GIB
from .geometry import ViewInfo, physical_view_name
from .interpolation import NrrdLayerRef, PreparedViewResult, _DirectUnionBackingLease
from .runtime import close_memmap_array_without_flush


def scratch_unlink_path_for_memmap(arr: object, path: Path | None) -> Path | None:
    """Return an exact pathname backing eligible for deferred scratch deletion."""
    if path is None or not isinstance(arr, np.memmap):
        return None
    path = Path(path)
    try:
        if (not str(path).startswith('/proc/')
                and path.samefile(Path(str(arr.filename)))):
            return path
    except OSError:
        pass
    return None


@dataclass
class ComponentProjectionSubmitter:
    """Estimate a component's reservation and submit its immutable store."""

    queue: object
    source_shape: tuple[int, int, int]
    materialize: Callable[..., NrrdLayerRef]

    def __call__(self, component_path: Path, **kwargs):
        component_path = Path(component_path)
        metadata = json.loads((component_path / 'meta.json').read_text(encoding='utf-8'))
        shape = tuple(int(v) for v in metadata['shape'])
        source_bytes = sum(path.stat().st_size for path in component_path.iterdir() if path.is_file())
        view = kwargs['view']
        source_shape = tuple(int(v) for v in self.source_shape)
        source_volume_bytes = math.prod(source_shape)
        if int(kwargs.get('added_voxels', 0)) <= 0 and view.family not in ('radial', 'spherical'):
            working_bytes = 64 * 1024 * 1024
        elif str(view.family) == 'azimuthal' and str(kwargs.get('source')) == 'fullframe':
            packed_source_bytes = source_shape[0] * source_shape[1] * ((source_shape[2] + 7) // 8)
            map_bound = 16 * max(
                int(view.full_t) * int(view.full_h), int(view.full_t) * int(view.full_w),
                int(view.full_h) * int(view.full_w), int(view.num_slices) * shape[2],
                source_shape[0] * source_shape[1], source_shape[0] * source_shape[2],
                source_shape[1] * source_shape[2],
            )
            working_bytes = packed_source_bytes + map_bound + GIB
        elif physical_view_name(view) == 'transverse' and float(view.tta_angle_deg) == 0.0:
            working_bytes = 256 * 1024 * 1024
        else:
            working_bytes = 2 * math.prod(shape) + 2 * source_volume_bytes + 4 * GIB
        return self.queue.submit(
            self.materialize, component_path,
            source_bytes=int(source_bytes), working_bytes=int(working_bytes), **kwargs,
        )


@dataclass
class AdmittedViewPrepare:
    """A submitted parent prepare with all former closure inputs made explicit."""

    admission: object
    transient_bytes: int
    model_name: str
    view: ViewInfo
    union_mm: np.ndarray | None
    confmap_mm: np.ndarray | None
    d1_shadow_path: Path | None
    union_path: Path
    confmap_path: Path | None
    temp_dir: Path
    dense_tiling_active: bool
    min_conf: float
    min_radius: float
    interpolation_distance: int
    interpolation_walk_back: int
    interpolation_candidates: int
    interpolation_passes: int
    interpolation_min_radius: float
    interpolation_search_angle: float
    keep_temp_artifacts: bool
    slice_workers: int
    interpolation_task_workers: int
    component_layers_needed: bool
    precleaned_slice_cleanup: bool
    hole_fill_done_on_device: bool
    slice_meta: Mapping[str, object] | None
    fuse_azimuthal_component_layers: Callable[[], bool]
    component_ref_dense_retirement_active: bool
    preinterpolation_layer_already_published: bool
    parent_mask_ready_callback: Callable[..., object] | None
    submit_component_projection: Callable[..., object]
    materialize_workspace: Callable[..., np.ndarray]
    prepare: Callable[..., PreparedViewResult]
    close_dense: Callable[..., object] = close_memmap_array_without_flush
    interpolation_backend: str = 'sdf'
    sam_context: object | None = None

    def __call__(self) -> PreparedViewResult:
        with self.admission.reserve(
            int(self.transient_bytes), f'{self.model_name}/{self.view.name}',
        ):
            local_union_mm = self.union_mm
            try:
                if local_union_mm is None:
                    if self.d1_shadow_path is None:
                        raise RuntimeError(
                            f'{self.model_name}/{self.view.name}: missing D1 view shadow'
                        )
                    local_union_mm = self.materialize_workspace(
                        self.d1_shadow_path,
                        self.union_path,
                        desc=(f'D1 view-native shadow materialization '
                              f'{self.model_name}/{self.view.name}'),
                        workers=int(self.slice_workers),
                    )
                    if not self.keep_temp_artifacts:
                        try:
                            shutil.rmtree(self.d1_shadow_path, ignore_errors=True)
                        except Exception:
                            pass
                return self.prepare(
                    model_name=str(self.model_name),
                    view=self.view,
                    union_mm=local_union_mm,
                    confmap_mm=self.confmap_mm,
                    union_path=self.union_path,
                    confmap_path=self.confmap_path,
                    temp_dir=self.temp_dir,
                    dense_tiling_active=bool(self.dense_tiling_active),
                    min_conf=float(self.min_conf),
                    min_radius=float(self.min_radius),
                    interpolate=int(self.interpolation_distance),
                    interpolation_walk_back=int(self.interpolation_walk_back),
                    interpolation_candidates=int(self.interpolation_candidates),
                    interpolate_passes=int(self.interpolation_passes),
                    interpolate_min_radius=float(self.interpolation_min_radius),
                    interpolation_search_angle=float(self.interpolation_search_angle),
                    interpolation_backend=str(self.interpolation_backend),
                    sam_context=self.sam_context,
                    keep_temp=bool(self.keep_temp_artifacts),
                    slice_workers=int(self.slice_workers),
                    interpolation_task_workers=int(self.interpolation_task_workers),
                    nrrd_layers_enabled=bool(self.component_layers_needed),
                    precleaned_slice_cleanup=bool(self.precleaned_slice_cleanup),
                    hole_fill_done_on_device=bool(self.hole_fill_done_on_device),
                    slice_meta=self.slice_meta,
                    fuse_azimuthal_component_layers=bool(
                        self.fuse_azimuthal_component_layers()
                    ),
                    parent_mask_ready_callback=(
                        self.parent_mask_ready_callback if self.dense_tiling_active else None
                    ),
                    internal_final_layer_enabled=bool(
                        self.component_ref_dense_retirement_active
                        and not self.component_layers_needed
                    ),
                    retire_dense_after_prepare=bool(
                        self.component_ref_dense_retirement_active
                        and not self.dense_tiling_active
                        and not self.keep_temp_artifacts
                    ),
                    preinterpolation_layer_already_published=bool(
                        self.preinterpolation_layer_already_published
                    ),
                    submit_component_projection=self.submit_component_projection,
                )
            except BaseException:
                # The original/local dense mapping belongs to this reservation.
                # Immutable component and tile support stores have other owners.
                unlink_path = (
                    scratch_unlink_path_for_memmap(local_union_mm, self.union_path)
                    if not self.keep_temp_artifacts else None
                )
                self.close_dense(local_union_mm, unlink_path=unlink_path)
                local_union_mm = None
                raise


@dataclass
class ViewPrepareLeaseState:
    """Shared inference/postprocess lease registries used at handoff and drain."""

    leases: dict[tuple[str, str], _DirectUnionBackingLease]
    inference_views: set[tuple[str, str]]
    inference_bytes: dict[tuple[str, str], int]
    postprocess_views: set[tuple[str, str]]
    postprocess_bytes: dict[tuple[str, str], int]

    def handoff(self, key: tuple[str, str]) -> bool:
        lease = self.leases.get(key)
        if lease is None:
            return False
        if key not in self.inference_views or key in self.postprocess_views:
            raise RuntimeError(
                f'direct-union backing {key} is not exclusively inference-owned at handoff'
            )
        lease.transition('inference', 'postprocess')
        self.inference_views.remove(key)
        self.inference_bytes.pop(key, None)
        self.postprocess_views.add(key)
        self.postprocess_bytes[key] = int(lease.nbytes)
        return True

    def rollback_handoff(self, key: tuple[str, str]) -> None:
        lease = self.leases[key]
        lease.transition('postprocess', 'inference')
        self.postprocess_views.discard(key)
        self.postprocess_bytes.pop(key, None)
        self.inference_views.add(key)
        self.inference_bytes[key] = int(lease.nbytes)

    def complete(self, key: tuple[str, str], *, retain_for_dense_retirement: bool) -> bool:
        lease = self.leases.get(key)
        if lease is None:
            return False
        if key not in self.postprocess_views:
            raise RuntimeError(
                f'direct-union backing {key} completed without a postprocess lease'
            )
        if retain_for_dense_retirement:
            return False
        self.leases.pop(key, None)
        lease.release('postprocess')
        self.postprocess_views.remove(key)
        self.postprocess_bytes.pop(key, None)
        return True
