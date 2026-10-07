"""Explicit admission and ownership state for parent view preparation.

The orchestrator still decides when a view is ready.  This module owns the
work submitted to the postprocess executor and the dense backing handoff so
both can be exercised without extracting a closure from ``pipeline``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import threading
import weakref
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .config import GIB
from .geometry import ViewInfo, is_tilted_view, physical_view_name
from .interpolation import NrrdLayerRef, PreparedViewResult, _DirectUnionBackingLease
from .runtime import (_interpolation_array_backing_path, close_memmap_array_without_flush,
                      _memfd_owner_key_from_array, _mount_fstype_for_path,
                      _MEMORY_BACKED_FSTYPES)


def _array_lifetime_owner(array):
    owner, seen = array, set()
    while id(owner) not in seen:
        base = getattr(owner, 'base', None)
        if base is None and isinstance(owner, memoryview):
            base = owner.obj
            try:
                weakref.ref(base)
            except TypeError:
                break  # Keep the weakrefable carrier for bytes/bytearray storage.
        if base is None:
            break
        seen.add(id(owner))
        owner = base
    return owner


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


def _windows_volume_ram_backing(path) -> bool | None:
    """Use the native volume/drive classification; failed probes grant nothing."""
    try:
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetVolumePathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        kernel.GetVolumePathNameW.restype = ctypes.c_int
        kernel.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
        kernel.GetDriveTypeW.restype = ctypes.c_uint
        root = ctypes.create_unicode_buffer(32768)
        if not kernel.GetVolumePathNameW(str(path), root, len(root)) or not root.value:
            return None
        kind = int(kernel.GetDriveTypeW(root.value))
        return True if kind == 6 else False if kind in (2,3,4,5) else None
    except (OSError, AttributeError, TypeError, ValueError):
        return None


def _dense_mapping_ram_backing(path) -> bool | None:
    filesystem = _mount_fstype_for_path(path)
    if filesystem is not None:
        return str(filesystem).lower() in _MEMORY_BACKED_FSTYPES
    return _windows_volume_ram_backing(path) if os.name == 'nt' else None


def classify_dense_ram_backing(arrays, *, future_ram_bytes=0) -> bool | None:
    """Exclude only proven disk-only mappings with no future RAM allocation."""
    if int(future_ram_bytes) < 0:
        raise ValueError('Future dense RAM promise cannot be negative')
    if future_ram_bytes:
        return None
    kinds = []
    for array in arrays:
        if array is None:
            continue
        if not isinstance(array, np.memmap):
            kinds.append(True)
            continue
        if array._mmap is None or array._mmap.closed:
            return None
        if array.mode == 'c' or _memfd_owner_key_from_array(array) is not None:
            kinds.append(True)
            continue
        backing = _interpolation_array_backing_path(array)
        if backing is None:
            return None
        path = Path(backing)
        if str(path).startswith('/proc/'):
            kinds.append(True)
            continue
        try:
            if path.is_symlink() or path.resolve() != path.absolute():
                return None
            if not path.is_file() or path.stat().st_size < int(array.offset) + int(array.nbytes):
                return None
        except OSError:
            return None
        kind = _dense_mapping_ram_backing(path)
        if kind is None:
            return None
        kinds.append(kind)
    if not kinds or any(kinds) and not all(kinds):
        return None
    return any(kinds)


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


@dataclass(frozen=True)
class _SamPublicationSource:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _SamPublicationIdentity:
    """Only publication metadata crosses threads; no live tracker or profile."""

    detector_identity: str
    bundle_identity: str
    source_volume: _SamPublicationSource | None


@dataclass
class SamLayerProjectionSubmitter:
    """Detach immutable SAM stores under the existing projection byte limits."""

    queue: object
    source_shape: tuple[int, int, int]
    materialize_directional: Callable[..., NrrdLayerRef]
    materialize_extrapolation: Callable[..., NrrdLayerRef]

    def __call__(self, entry, *, layer_kind, **kwargs):
        if layer_kind not in ('interpolation', 'extrapolation'):
            raise ValueError(f'Unknown SAM publication kind: {layer_kind}')
        entry = copy.deepcopy(dict(entry))
        source = Path(str(entry['path']))
        metadata = json.loads((source / 'meta.json').read_text(encoding='utf-8'))
        native_shape = tuple(int(value) for value in metadata['shape'])
        if len(native_shape) != 3 or min(native_shape) < 1:
            raise ValueError('SAM publication requires a positive TYX store')
        source_bytes = sum(path.stat().st_size for path in source.iterdir() if path.is_file())
        view = kwargs['view']
        target_shape = tuple(int(value) for value in self.source_shape)
        target_bytes = math.prod(target_shape)
        if int(entry.get('voxel_count', 0)) <= 0:
            working_bytes = 64 * 1024**2
        elif view.family == 'azimuthal':
            packed_target_bytes = target_shape[0] * target_shape[1] * ((target_shape[2] + 7) // 8)
            map_bound = 16 * max(
                int(view.full_t) * int(view.full_h), int(view.full_t) * int(view.full_w),
                int(view.full_h) * int(view.full_w), int(view.num_slices) * native_shape[2],
                target_shape[0] * target_shape[1], target_shape[0] * target_shape[2],
                target_shape[1] * target_shape[2],
            )
            working_bytes = packed_target_bytes + map_bound + GIB
        elif (view.family == 'orthogonal' and not is_tilted_view(view)
                and physical_view_name(view) == 'transverse'):
            working_bytes = 256 * 1024**2
        else:
            # Sparse transpose or nonlinear source projection may own native
            # decode and source outputs. Encoded input size cannot bound them.
            working_bytes = 2 * math.prod(native_shape) + 2 * target_bytes + 4 * GIB
        context = kwargs['sam_context']
        processing_shape = getattr(getattr(context, 'source_volume', None), 'shape', None)
        kwargs['sam_context'] = _SamPublicationIdentity(
            detector_identity=str(context.detector_identity or ''),
            bundle_identity=str(context.bundle_identity),
            source_volume=(None if processing_shape is None else
                           _SamPublicationSource(tuple(int(value) for value in processing_shape))),
        )
        materialize = (self.materialize_directional if layer_kind == 'interpolation'
                       else self.materialize_extrapolation)
        future = self.queue.submit(materialize, entry, source_bytes=int(source_bytes),
                                   working_bytes=int(working_bytes), **kwargs)
        # The scheduler may retire a completed parent's dense canvas before
        # publication finishes, but only for these independent immutable inputs.
        future._xta_dense_independent_publication_paths = (source,)
        return future


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
    extrapolation_distance: int = 0
    extrapolation_walk_back: int = 1
    extrapolation_min_radius: float = 3.
    sam_base_allowance_bytes: int = 0
    confidence_retired_callback: Callable[[str, str, int], object] | None = None
    confidence_retired_callback_factory: Callable[[object], Callable] | None = None
    submit_sam_layer_projection: Callable[..., object] | None = None
    backing_lease: _DirectUnionBackingLease | None = None
    _cancelled_owner_refs: tuple = field(default=(), init=False, repr=False)
    _cancelled_before_prepare: bool = field(default=False, init=False, repr=False)
    def rebind_confidence_retirement(self, lease):
        if self.confidence_retired_callback_factory is not None:
            self.confidence_retired_callback = self.confidence_retired_callback_factory(lease)

    def _take_confidence(self):
        owner, self.confmap_mm = self.confmap_mm, None
        return owner

    def _retire_unstarted_inputs(self):
        self._cancelled_before_prepare = True
        self._cancelled_owner_refs = tuple(weakref.ref(_array_lifetime_owner(value))
            for value in (self.union_mm, self.confmap_mm) if value is not None)
        for name, path_name in (('union_mm', 'union_path'), ('confmap_mm', 'confmap_path')):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, None)
                path = getattr(self, path_name)
                try:
                    self.close_dense(value, unlink_path=(None if self.keep_temp_artifacts else
                        scratch_unlink_path_for_memmap(value, path)))
                except BaseException as error:
                    self._cancelled_cleanup_error = error
                    raise

    def retire_cancelled_future(self, future):
        """A successful Future.cancel proves this prepare never acquired inputs."""
        if not (future.cancelled() or self._cancelled_before_prepare):
            return
        try:
            if not self._cancelled_before_prepare:
                self._retire_unstarted_inputs()
        except BaseException as error:
            future._xta_cancelled_prepare_cleanup_error = error
        finally:
            if getattr(self, '_cancelled_cleanup_error', None) is not None:
                future._xta_cancelled_prepare_cleanup_error = self._cancelled_cleanup_error
            future._xta_cancelled_prepare_owner_refs = self._cancelled_owner_refs

    def track_cancellation(self, future):
        # A successful future must not pin the task's original dense input.
        reference = weakref.ref(self)
        def retired(done):
            task = reference()
            if task is not None:
                task.retire_cancelled_future(done)
        future.add_done_callback(retired)

    def check_cancelled(self):
        cancelled = getattr(self.sam_context, '_cancel', None)
        if cancelled is not None and cancelled.is_set():
            self._retire_unstarted_inputs()
            raise CancelledError(getattr(self.sam_context, '_failure', 'SAM preparation cancelled'))

    @contextmanager
    def _reservation(self):
        self.check_cancelled()
        if (self.sam_context is not None and (int(self.extrapolation_distance) > 0
                or self.interpolation_backend == 'sam' and int(self.interpolation_distance) > 0)):
            from .sam_resources import admit_sam_parent_resources, sam_worker_count
            workers = len(getattr(self.sam_context, 'device_ids', ())) or 1
            with ExitStack() as reservation:
                try:
                    profile = reservation.enter_context(admit_sam_parent_resources(
                        self.admission, self.transient_bytes,
                        f'{self.model_name}/{self.view.name}/fullframe', worker_count=workers,
                        execution_slots=sam_worker_count(self.sam_context),
                        base_allowance_bytes=self.sam_base_allowance_bytes,
                        cancel_event=getattr(self.sam_context, '_cancel', None)))
                except CancelledError:
                    self.check_cancelled()
                    raise
                with self.sam_context.resource_scope(profile):
                    yield
        else:
            with self.admission.reserve(int(self.transient_bytes), f'{self.model_name}/{self.view.name}'):
                yield

    def __call__(self) -> PreparedViewResult:
        with self._reservation():
            self.check_cancelled()
            if self.backing_lease is not None:
                # Preparation can replace a disk input with a newly allocated
                # canvas. Keep its full dense promise until actual retirement.
                self.backing_lease.ram_backed = None
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
                # The prepare function owns confidence through capture only.
                # Keeping a second task alias pinned its mmap until the much
                # longer mask projection finished, even after capture retired it.
                input_owner_ref = weakref.ref(_array_lifetime_owner(local_union_mm))
                input_nbytes = int(np.asarray(local_union_mm).nbytes)
                input_unlink_path = None
                input_path_identity = None
                if not self.keep_temp_artifacts:
                    input_unlink_path = scratch_unlink_path_for_memmap(local_union_mm, self.union_path)
                    if input_unlink_path is not None:
                        try:
                            identity = input_unlink_path.lstat()
                            input_path_identity = (identity.st_dev, identity.st_ino)
                        except OSError:
                            input_unlink_path = None
                result = self.prepare(
                    model_name=str(self.model_name),
                    view=self.view,
                    union_mm=local_union_mm,
                    confmap_mm=(self._take_confidence()
                                if self.confidence_retired_callback is None else None),
                    confidence_owner=([self._take_confidence()]
                                      if self.confidence_retired_callback is not None else None),
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
                    extrapolation_distance=int(self.extrapolation_distance),
                    extrapolation_walk_back=int(self.extrapolation_walk_back),
                    extrapolation_min_radius=float(self.extrapolation_min_radius),
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
                    submit_sam_layer_projection=self.submit_sam_layer_projection,
                    confidence_retired_callback=self.confidence_retired_callback,
                )
                # The scheduler needs no strong input alias. Its retirement
                # fence still observes any producer/caller view of the backing.
                result._dense_input_owner_ref = input_owner_ref
                result._dense_input_nbytes = input_nbytes
                if not self.keep_temp_artifacts and all(
                    volume is None or _array_lifetime_owner(volume) is not input_owner_ref()
                    for volume in (result.native_support_mm, result.final_view_volume_mm)
                ):
                    # All native readers/retries finished, and publications own
                    # independent CVOLs. Preserve outside aliases via deferred
                    # retirement of the original backing, not the rebound mask.
                    if input_unlink_path is not None:
                        try:
                            identity = input_unlink_path.lstat()
                            if (identity.st_dev, identity.st_ino) != input_path_identity:
                                input_unlink_path = None
                        except OSError:
                            input_unlink_path = None
                    self.close_dense(local_union_mm, unlink_path=input_unlink_path)
                return result
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
    retired_inputs: set[tuple[tuple[str, str], int, str]] = field(default_factory=set)
    retired_for_publication: set[tuple[str, str]] = field(default_factory=set)

    @property
    def ram_commitment_bytes(self) -> int:
        return sum(lease.ram_commitment_bytes for lease in tuple(self.leases.values()))

    @property
    def disk_backed_logical_bytes(self) -> int:
        return sum(int(lease.nbytes) for lease in tuple(self.leases.values()) if lease.ram_backed is False)

    def retire_dense_for_publication(self, prepared, *, enabled: bool, tiled: bool,
                                    keep_temp: bool, retired_callback,
                                    close_dense=close_memmap_array_without_flush) -> bool:
        """Detach dense inputs; return admission only after their last owner dies."""
        key = (str(prepared.model_name), str(prepared.view_name))
        input_owner_ref = getattr(prepared, '_dense_input_owner_ref', None)
        if (not enabled or tiled or keep_temp or key in self.retired_for_publication
                or input_owner_ref is None
                or not prepared.pending_component_layers
                or prepared.parent_mask_support_mm is not None
                or prepared.parent_bridge_support_mm is not None
                or getattr(prepared, 'confidence', None) is not None
                or any(getattr(ref, 'live_array', None) is not None for ref in prepared.nrrd_layers)):
            return False
        lease = self.leases.get(key)
        if lease is not None:
            if (lease.phase != 'postprocess' or key not in self.postprocess_views
                    or self.postprocess_bytes.get(key) != lease.nbytes):
                raise RuntimeError(f'dense publication {key} has no consistent postprocess owner')
            if int(lease.nbytes) > int(prepared._dense_input_nbytes):
                # A transferred confidence owner has not yet returned its
                # separate credit; its still-live bytes cannot be released here.
                return False
        sources = []
        for future in prepared.pending_component_layers:
            paths = getattr(future, '_xta_dense_independent_publication_paths', ())
            if not paths:
                return False
            sources.extend(Path(path).resolve() for path in paths)
        arrays = []
        for volume in (prepared.native_support_mm, prepared.final_view_volume_mm):
            if volume is not None and all(volume is not prior for prior in arrays):
                arrays.append(volume)
        backings = [_interpolation_array_backing_path(volume) for volume in arrays]
        if any(backing is not None and any(
                Path(backing).resolve().is_relative_to(source) for source in sources)
               for backing in backings):
            # Never retire a dense file belonging to a queued input store.
            return False
        roots = []
        original_owner = input_owner_ref()
        if original_owner is not None:
            roots.append(original_owner)
        for volume in arrays:
            owner = _array_lifetime_owner(volume)
            if all(owner is not prior for prior in roots):
                roots.append(owner)
        for volume, backing in zip(arrays, backings):
            close_dense(volume, unlink_path=(None if backing is None
                or str(backing).startswith('/proc/') else Path(backing)))
        prepared.native_support_mm = None
        prepared.final_view_volume_mm = None
        return self._retire_after_owners(key, lease, roots, retired_callback)

    def retire_cancelled(self, key, future, *, retired_callback) -> bool:
        """Queued cancellation returns dense credit only after its last array owner dies."""
        refs = getattr(future, '_xta_cancelled_prepare_owner_refs', None)
        lease = self.leases.get(key)
        if refs is None or lease is None or key in self.retired_for_publication:
            return False
        if getattr(future, '_xta_cancelled_prepare_cleanup_error', None) is not None:
            return False
        roots = {id(owner): owner for reference in refs if (owner := reference()) is not None}
        return self._retire_after_owners(key, lease, list(roots.values()), retired_callback)

    def _retire_after_owners(self, key, lease, roots, retired_callback):
        self.retired_for_publication.add(key)
        if lease is None:
            return False
        if not roots:
            return self.complete(key, retain_for_dense_retirement=False)
        remaining, lock = [len(roots)], threading.Lock()

        def retired():
            with lock:
                remaining[0] -= 1
                completed = remaining[0] == 0
            if completed:
                retired_callback(key, lease)

        for owner in roots:
            finalizer = weakref.finalize(owner, retired)
            finalizer.atexit = False
        return False

    def settle_publication_retirement(self, key, expected_lease) -> bool:
        """Apply an owner-death receipt only on the scheduler state thread."""
        if key not in self.retired_for_publication or self.leases.get(key) is not expected_lease:
            return False
        return self.complete(key, retain_for_dense_retirement=False)

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

    def retire_input_bytes(self, key: tuple[str, str], expected_lease: _DirectUnionBackingLease,
                           nbytes: int, *, token: str) -> bool:
        """Return an input's credit after its final owner dies, on the scheduler thread.

        A finalizer can arrive after whole-parent retirement. Lease identity and
        the input token make stale/duplicate notifications harmless and prevent
        them from crediting a replacement parent's allocation.
        """
        identity = (key, id(expected_lease), str(token))
        lease = self.leases.get(key)
        if lease is not expected_lease or identity in self.retired_inputs:
            return False
        if (lease.phase != 'postprocess' or key not in self.postprocess_views
                or int(self.postprocess_bytes.get(key, -1)) != int(lease.nbytes)):
            raise RuntimeError(f'direct-union input {key} retired without its postprocess owner')
        released = int(nbytes)
        if not 0 < released < int(lease.nbytes):
            raise RuntimeError(f'direct-union input {key} returned invalid dense credit {released}')
        self.retired_inputs.add(identity)
        lease.nbytes -= released
        self.postprocess_bytes[key] = int(lease.nbytes)
        return True
