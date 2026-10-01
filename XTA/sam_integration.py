"""Image lifetime and GPU admission for integrated TTA SAM interpolation.

The tracker remains resident only after detector inference has completed and its
GPU assets have been retired.  Device-stage leases then protect the predictor
and bounded session scratch from concurrent projection allocation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Mapping

import numpy as np


_UNSETTLED_SAM_CONTEXTS = {}
_UNSETTLED_SAM_LOCK = threading.Lock()


def sam_workers_unsettled() -> bool:
    """Whether a failed embedded run still owns unproven predictor residency."""
    with _UNSETTLED_SAM_LOCK:
        return bool(_UNSETTLED_SAM_CONTEXTS)


def retry_unsettled_sam_workers() -> None:
    """Refuse a new embedded GPU run until retained workers have settled."""
    with _UNSETTLED_SAM_LOCK:
        pending = tuple(_UNSETTLED_SAM_CONTEXTS.values())
    for context in pending:
        try:
            context.close()
        except BaseException:
            pass
    if sam_workers_unsettled():
        raise RuntimeError('Prior SAM workers remain resident; new GPU admission is refused until cleanup is proven')


def _retain_unsettled_context(context) -> None:
    with _UNSETTLED_SAM_LOCK:
        _UNSETTLED_SAM_CONTEXTS[id(context)] = context


def _forget_settled_context(context) -> None:
    with _UNSETTLED_SAM_LOCK:
        _UNSETTLED_SAM_CONTEXTS.pop(id(context), None)


def validate_sam_interpolation_geometry(views) -> None:
    """Reject geometry not verified by the native Transverse rollout."""
    from .geometry import physical_view_name
    unsupported = [str(view.name) for view in views if (
        str(view.family) != 'orthogonal'
        or physical_view_name(view) != 'transverse'
        or float(view.tta_angle_deg) != 0.0
    )]
    if unsupported:
        raise ValueError(
            'SAM interpolation currently supports native Transverse at angle zero; '
            'unsupported view(s): ' + ', '.join(unsupported)
        )


def _store_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    metadata = json.loads((Path(path) / 'meta.json').read_text(encoding='utf-8'))
    digest.update(json.dumps({'shape': metadata['shape'], 'format': metadata['format']},
                             sort_keys=True).encode('utf-8'))
    for item in sorted(Path(path).iterdir(), key=lambda item: item.name):
        if item.is_file() and item.name != 'meta.json':
            digest.update(item.name.encode('utf-8'))
            with item.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def publish_sam_gate_identity(store, *, policy_identity: str, evidence_path: str) -> str:
    """Attach selected-support identity before the immutable gate handoff."""
    path = Path(store.root)
    identity = _store_fingerprint(path)
    metadata_path = path / 'meta.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata.update(interpolation_backend='sam', gate_support_identity=identity,
                    interpolation_policy_identity=str(policy_identity),
                    proposal_evidence_path=str(evidence_path),
                    proposal_selection_status='policy_selected')
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding='utf-8')
    # The gate and retained confidence receipts read store metadata directly.
    store.meta.update(metadata)
    return identity


class SamInterpolationContext:
    """One run's persistent predictor, immutable image caches, and device leases."""

    def __init__(self, *, model_path, device_ids, temp_dir, evidence_root,
                 source_volume, source_identity, policy=None,
                 bundle_identity='', detector_identity='', source_grid_shape=None,
                 detector_device_ids=None, source_resize_semantics='caller_owned_exact_raster',
                 feature_cache_mib=512, crop_mode='whole', delayed_native_expansion=None):
        self.model_path = str(model_path)
        self.crop_mode = str(crop_mode).strip().lower()
        if self.crop_mode not in {'whole', 'tiled'}:
            raise ValueError('SAM crop mode must be whole or tiled')
        self.delayed_native_expansion_at_launch = delayed_native_expansion
        self.feature_cache_mib = int(feature_cache_mib)
        if self.feature_cache_mib < 0:
            raise ValueError('SAM frame-feature cache MiB must be nonnegative')
        self.device_ids = tuple(f'cuda:{int(str(value).split(":")[-1])}' for value in device_ids)
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids) or any(
                int(value.split(':')[-1]) < 0 for value in self.device_ids):
            raise ValueError('SAM context requires unique nonnegative CUDA devices')
        self.temp_dir = Path(temp_dir)
        self.evidence_root = Path(evidence_root)
        self.source_volume = source_volume
        self.source_identity = str(source_identity)
        self.source_resize_semantics = str(source_resize_semantics)
        self.bundle_identity = str(bundle_identity)
        self.detector_identity = str(detector_identity)
        self.source_grid_shape = tuple(int(value) for value in (source_grid_shape or source_volume.shape))
        self.policy = policy
        self._ready = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._runtime_lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._active_passes = 0
        self._runtime = None
        self._starting_runtime = None
        self._leases = []
        self._caches = {}
        self._cache_transforms = {}
        self._failure = ''
        self._closed = False
        self.wait_seconds = 0.0
        self.start_seconds = 0.0
        self.render_seconds = 0.0
        self.rendered_frames = 0
        self.rendered_pixels = 0
        self.cache_logical_bytes = 0
        self.exact_backing_reuses = 0
        self.planning_seconds = 0.0
        self.no_job_passes = 0
        self.dispatch_summary = {}
        self.shared_detector_devices = tuple(self.device_ids if detector_device_ids is None else
            sorted(set(self.device_ids) & {f'cuda:{int(str(value).split(":")[-1])}' for value in detector_device_ids}))
        if detector_device_ids is not None and not self.shared_detector_devices:
            self._ready.set()

    def detector_assets_retired(self) -> None:
        self._ready.set()

    def cancel(self, reason='TTA scheduler cancelled') -> None:
        self._failure = str(reason)
        self._cancel.set()
        self._ready.set()
        for runtime in (self._runtime, self._starting_runtime):
            cancel = getattr(runtime, 'cancel', None)
            if callable(cancel):
                cancel(str(reason))

    def _canvas_transform(self, view, shape):
        from .geometry import build_affine
        if tuple(shape[-2:]) == (int(view.src_h), int(view.src_w)):
            affine = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
            inverse = affine
        else:
            plan = build_affine(view=str(view.name), src_w=int(view.src_w),
                src_h=int(view.src_h), out_size=int(shape[1]), angle_deg=0.0,
                pad_mode=str(view.pad_mode))
            affine, inverse = plan.M_src_to_out, plan.M_out_to_src
        transform = {
            'M_native_to_canvas': np.asarray(affine).tolist(),
            'M_canvas_to_native': np.asarray(inverse).tolist(),
            'native_view_shape_tyx': [int(view.num_slices), int(view.src_h), int(view.src_w)],
            'canvas_shape_tyx': list(shape), 'source_grid_shape_tyx': list(self.source_grid_shape),
            'source_processing_shape_tyx': list(self.source_volume.shape),
            'source_resize_semantics': self.source_resize_semantics,
            'projection_contract': 'tta_native_transverse_categorical_source_restore',
        }
        return affine, inverse, transform

    def image_provider(self, view, shape, prepared_plan=None):
        """Reuse exact backing or render only the immutable planned frame crops."""
        from .geometry import build_affine, render_intensity_frame_on_grid
        from .lta_rendering import LtaPhysicalViewCacheRef, reference_existing_physical_view_cache
        from .media import wait_for_volume_ready
        from .runtime import _interpolation_array_backing_path
        shape = tuple(int(value) for value in shape)
        if len(shape) != 3 or any(value <= 0 for value in shape) or shape[0] != int(view.num_slices):
            raise ValueError('SAM image canvas must match the positive view-native frame count')
        if shape[-2:] != (int(view.src_h), int(view.src_w)) and shape[1] != shape[2]:
            raise ValueError('SAM detector processing canvas must be native or square')
        required = dict(getattr(prepared_plan, 'frame_crop_bounds', {}) or {}) if prepared_plan is not None else {
            index: (0, 0, shape[1], shape[2]) for index in range(shape[0])}
        if not required:
            raise ValueError('SAM image provider requires at least one planned tracking frame')
        required = {int(frame): tuple(int(value) for value in bbox) for frame, bbox in required.items()}
        for frame, (y0, x0, y1, x1) in required.items():
            if not 0 <= frame < shape[0] or not (0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
                raise ValueError('SAM image demand is outside the detector canvas')
        demand_identity = hashlib.sha256(json.dumps(required, sort_keys=True).encode()).hexdigest()
        key = (str(view.name), shape, demand_identity)
        with self._lock:
            if self._closed:
                raise RuntimeError('SAM interpolation source lifetime has ended')
            if key in self._caches:
                self._caches[key].revalidate()
                return self._caches[key]
            validate_sam_interpolation_geometry([view])
            render_started = time.perf_counter()
            lazy_source = bool(getattr(self.source_volume, '_is_lazy_processing_cube', False))
            if not lazy_source or self.source_volume.materialized:
                wait_for_volume_ready(self.source_volume)
            affine, inverse, transform = self._canvas_transform(view, shape)
            image_identity = json.dumps(dict(source=self.source_identity, view=str(view.name),
                affine=np.asarray(affine).tolist(), shape=shape,
                source_resize_semantics=self.source_resize_semantics), sort_keys=True)
            # Native angle-zero input already is the canonical uint8 frame stack.
            # Adopt the immutable existing map instead of copying every frame.
            source_array = (getattr(self.source_volume, '_array', None)
                if bool(getattr(self.source_volume, '_is_lazy_processing_cube', False)) else self.source_volume)
            backing = _interpolation_array_backing_path(source_array)
            if (backing is not None and tuple(source_array.shape) == shape
                    and np.dtype(source_array.dtype) == np.uint8
                    and bool(source_array.flags['C_CONTIGUOUS'])
                    and np.array_equal(affine, np.array([[1., 0., 0.], [0., 1., 0.]], np.float32))
                    and Path(backing).is_file() and Path(backing).stat().st_size == int(np.prod(shape))):
                stat = Path(backing).stat()
                reference = LtaPhysicalViewCacheRef(path=Path(backing), shape=shape, dtype='uint8',
                    physical_view_id=str(view.name), identity_sha256=hashlib.sha256(image_identity.encode()).hexdigest(),
                    size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
                self._caches[key] = reference
                self._cache_transforms[key] = transform
                self.exact_backing_reuses += 1
                return reference
            render_identity = hashlib.sha256(json.dumps(
                dict(source=self.source_identity, view=str(view.name), shape=shape,
                     angle=float(view.tta_angle_deg), pad=str(view.pad_mode), demand=demand_identity), sort_keys=True).encode()).hexdigest()[:16]
            path = self.temp_dir / 'sam_image_cache' / (f'{view.name}.{render_identity}.' + 'x'.join(map(str, shape)) + '.gray8.dat')
            path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            payload_bytes = 0
            for index, (y0, x0, y1, x1) in sorted(required.items()):
                records.append((index, y0, x0, y1, x1, payload_bytes))
                payload_bytes += (y1-y0)*(x1-x0)
            from .workspace import _env_int
            budget = max(1, _env_int('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', 1024**3))
            if payload_bytes > budget:
                raise RuntimeError(f'SAM planned image demand {payload_bytes} bytes exceeds cache budget {budget}')
            cache = np.memmap(path, dtype=np.uint8, mode='w+', shape=(payload_bytes,))
            try:
                for index, y0, x0, y1, x1, offset in records:
                    if self._cancel.is_set():
                        raise RuntimeError(self._failure)
                    crop_affine = np.array(affine, copy=True)
                    crop_affine[:, 2] -= [x0, y0]
                    crop_inverse = np.array(inverse, copy=True)
                    crop_inverse[:, 2] += np.asarray(inverse)[:, :2] @ np.array([x0, y0])
                    image = self._render_demand_crop(view, index, crop_affine, crop_inverse,
                        output_height=y1-y0, output_width=x1-x0)
                    if image.dtype != np.uint8 or image.shape != (y1-y0, x1-x0):
                        raise ValueError('SAM image provider returned a mismatched detector canvas')
                    cache[offset:offset+image.size] = image.reshape(-1)
                    self.rendered_frames += 1
                    self.rendered_pixels += int(image.size)
                cache.flush()
            finally:
                del cache
            # Identity describes canonical image bytes, independently of how
            # much of that immutable canvas this compact descriptor stores.
            # Exact model/crop/frame feature keys can therefore survive a
            # different scope's demand inventory without admitting new pixels.
            stat = path.stat()
            reference = LtaPhysicalViewCacheRef(path=path, shape=shape, dtype='uint8',
                physical_view_id=str(view.name), identity_sha256=hashlib.sha256(image_identity.encode()).hexdigest(),
                size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                frame_crops=tuple(records) if prepared_plan is not None else ())
            self._caches[key] = reference
            self._cache_transforms[key] = transform
            self.render_seconds += time.perf_counter() - render_started
            self.cache_logical_bytes += payload_bytes
            return reference

    def _render_demand_crop(self, view, index, affine, inverse, *, output_height, output_width):
        """Render needed native frames without materializing an unused lazy cube."""
        from ._deps import cv2
        from .geometry import render_intensity_frame_on_grid
        from .media import (_linear_source_index, _resize_gray_slice_nearest_or_linear,
                            wait_for_volume_ready, wait_for_volume_slice_ready)
        source = self.source_volume
        frames = None
        if bool(getattr(source, '_is_lazy_processing_cube', False)) and not source.materialized:
            decoded = source.source
            in_t, in_h, in_w = (int(value) for value in decoded.shape)
            out_t, out_h, out_w = (int(value) for value in source.shape)
            if (in_h, in_w) == (out_h, out_w) and not source.streaming_backend:
                # Match the production OpenCV slab resize exactly. Restrict the
                # spatial slab to the native pixels sampled by this tracker crop.
                corners = np.array([[0., 0.], [output_width-1., 0.],
                    [0., output_height-1.], [output_width-1., output_height-1.]])
                mapped = corners @ np.asarray(inverse)[:, :2].T + np.asarray(inverse)[:, 2]
                x0 = max(0, int(np.floor(mapped[:, 0].min()))-2)
                x1 = min(out_w, int(np.ceil(mapped[:, 0].max()))+3)
                y0 = max(0, int(np.floor(mapped[:, 1].min()))-2)
                y1 = min(out_h, int(np.ceil(mapped[:, 1].max()))+3)
                if x1 <= x0 or y1 <= y0:
                    return np.zeros((output_height, output_width), np.uint8)
                wait_for_volume_ready(decoded)
                from .workspace import _env_int
                render_budget = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
                native_bytes = (y1-y0)*(x1-x0)
                if native_bytes >= render_budget:
                    raise RuntimeError('SAM native crop exceeds the bounded rendering memory budget')
                native = np.empty((y1-y0, x1-x0), np.uint8)
                columns = max(1, min(32760, (render_budget-native_bytes)//max(1, in_t+out_t)))
                for column in range(x0, x1, columns):
                    column_stop = min(x1, column+columns)
                    rows = max(1, columns//(column_stop-column))
                    for row in range(y0, y1, rows):
                        stop = min(y1, row+rows)
                        slab = np.ascontiguousarray(decoded[:, row:stop, column:column_stop], dtype=np.uint8)
                        resized = cv2.resize(slab.reshape(in_t, -1), (slab.shape[1]*slab.shape[2], out_t),
                            interpolation=cv2.INTER_LINEAR)
                        native[row-y0:stop-y0, column-x0:column_stop-x0] = resized[index].reshape(stop-row, column_stop-column)
                origin = np.array([x0, y0])
                affine = np.array(affine, copy=True)
                affine[:, 2] += np.asarray(affine)[:, :2] @ origin
                inverse = np.array(inverse, copy=True)
                inverse[:, 2] -= origin
            else:
                from .workspace import _env_int
                render_budget = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
                if 3*out_h*out_w > render_budget:
                    raise RuntimeError('SAM exact XY resize exceeds the bounded rendering memory budget')
                z = _linear_source_index(index, out_t, in_t)
                z0, z1 = int(np.floor(z)), min(in_t-1, int(np.floor(z))+1)
                alpha = z-z0
                wait_for_volume_slice_ready(decoded, z0)
                native = _resize_gray_slice_nearest_or_linear(decoded[z0], out_w, out_h, cv2.INTER_LINEAR)
                if z1 != z0 and alpha > 1e-7:
                    wait_for_volume_slice_ready(decoded, z1)
                    other = _resize_gray_slice_nearest_or_linear(decoded[z1], out_w, out_h, cv2.INTER_LINEAR)
                    native = cv2.addWeighted(native, 1.-alpha, other, alpha, 0.)
            class OneNativeFrame:
                def __getitem__(self, frame_index):
                    if int(frame_index) != int(index):
                        raise RuntimeError('SAM demand renderer accessed an unplanned frame')
                    return native
            frames = OneNativeFrame()
        return render_intensity_frame_on_grid(source, view, index, M_src_to_out=affine,
            M_out_to_src=inverse, output_height=output_height, output_width=output_width,
            view_frames=frames)

    def _start(self):
        with self._runtime_lock:
            self._start_admitted()

    def _start_admitted(self):
        if self._cancel.is_set():
            raise RuntimeError(self._failure)
        if self._runtime is not None:
            return
        started = time.perf_counter()
        while not self._ready.wait(timeout=0.25):
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
        self.wait_seconds += time.perf_counter() - started
        if self._cancel.is_set():
            raise RuntimeError(self._failure)
        import torch
        from .backprojection import _try_acquire_specific_main_process_gpu_stage
        from .sam_tracker_runtime import SamInterpolationTracker
        visible_devices = int(torch.cuda.device_count())
        unavailable = [device for device in self.device_ids
                       if int(device.split(':')[-1]) >= visible_devices]
        if unavailable:
            raise RuntimeError(f'SAM device(s) unavailable in the visible CUDA pool: {unavailable}')
        started = time.perf_counter()
        runtime = None
        try:
            for device in self.device_ids:
                index = int(device.split(':')[-1])
                deadline = time.monotonic() + 300.0
                lease = None
                while lease is None:
                    if self._cancel.is_set():
                        raise RuntimeError(self._failure)
                    lease = _try_acquire_specific_main_process_gpu_stage(
                        torch, index, 'TTA persistent SAM interpolation predictor')
                    if lease is None:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f'SAM GPU admission timed out on {device}')
                        self._cancel.wait(timeout=0.05)
                self._leases.append(lease)
            runtime = SamInterpolationTracker(
                model_path=self.model_path, device_ids=tuple(int(value.split(':')[-1]) for value in self.device_ids),
                artifact_root=self.temp_dir / 'sam_runtime',
                feature_cache_bytes=self.feature_cache_mib*1024**2)
            self._starting_runtime = runtime
            if self._cancel.is_set():
                cancel = getattr(runtime, 'cancel', None)
                if callable(cancel):
                    cancel(self._failure)
                raise RuntimeError(self._failure)
            runtime.start()
            self._runtime = runtime
            self._starting_runtime = None
            self.start_seconds += time.perf_counter() - started
        except BaseException as startup_error:
            self._starting_runtime = None
            settled = True
            if runtime is not None:
                self._runtime = runtime
                try:
                    runtime.close()
                except BaseException:
                    settled = getattr(runtime, 'residency_released', False) is True
            if settled:
                self._runtime = None
                for lease in reversed(self._leases):
                    lease.release()
                self._leases.clear()
            else:
                self._runtime = runtime
                _retain_unsettled_context(self)
                self.cancel('SAM startup failed with unsettled worker residency')
                raise RuntimeError('SAM startup failed; GPU ownership retained until worker cleanup is proven') from startup_error
            raise

    def interpolate(self, observation_volume, *, view, scope, **kwargs):
        from .sam_interpolation import (interpolate_sam_view_volume_pass,
                                        prepare_sam_interpolation_pass)
        with self._lock:
            if self._closed:
                raise RuntimeError('SAM interpolation runtime is closed')
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
            self._active_passes += 1
        prepared = None
        execution_started = False
        try:
            from .geometry import physical_view_name
            # The baseline owner retains mutation rights after this bounded
            # pass. Planning and tracking share a pinned read-only view.
            observed = np.asarray(observation_volume).view()
            observed.flags.writeable = False
            shape = tuple(int(value) for value in observation_volume.shape)
            if len(shape) != 3 or any(value < 1 for value in shape):
                raise ValueError('SAM observation canvas must have positive TYX dimensions')
            _, _, transform = self._canvas_transform(view, shape)
            scope_metadata = dict(scope_id=str(scope), detector_identity=self.detector_identity,
                sam_bundle_identity=self.bundle_identity, view_name=str(view.name),
                physical_view=physical_view_name(view), angle_deg=float(view.tta_angle_deg),
                augmentation_pass=int(getattr(view, 'augmentation_pass', 0)),
                canvas_transform=transform, sam_crop_mode=self.crop_mode,
                sam_crop_tile_side=1008, sam_crop_halo=128,
                sam_crop_canvas_contract='current_interpolation_working_canvas',
                delayed_native_expansion_at_launch=self.delayed_native_expansion_at_launch,
                sam_working_canvas_kind=('native_view' if shape[-2:] ==
                    (int(view.src_h), int(view.src_w)) else 'detector_processing'),
                sam_working_canvas_shape_tyx=list(shape),
                sam_native_view_shape_tyx=[int(view.num_slices), int(view.src_h), int(view.src_w)])
            planning_keys = {'pass_index', 'gap_distance', 'min_radius', 'search_angle_deg',
                'interpolation_walk_back', 'interpolation_candidates', 'interpolation_passes',
                'wrap_axis', 'upstream_lineage', 'spacing_zyx', 'planner_limits', 'canonical_labels'}
            prepared = prepare_sam_interpolation_pass(observed, view=view,
                scope=scope_metadata, policy=self.policy,
                **{key: value for key, value in kwargs.items() if key in planning_keys})
            tracked_group_ids = {str(run.group_id) for run in prepared.runs}
            oversized_groups = [group for group in prepared.groups if str(group.group_id) in tracked_group_ids
                and max(group.context_bbox_yx[2]-group.context_bbox_yx[0],
                        group.context_bbox_yx[3]-group.context_bbox_yx[1]) > 1008]
            group_tile_crops = {}
            for job in getattr(prepared, 'tracker_jobs', ()):
                group_tile_crops.setdefault(str(job.original_run.group_id), set()).add(tuple(job.tile.crop_bbox_yx))
            multi_tile_groups = sum(len(crops)>1 for crops in group_tile_crops.values())
            with self._lock:
                self.planning_seconds += prepared.planner_wall_seconds + prepared.snapshot_wall_seconds
            provider = None
            runtime = None
            if prepared.needs_tracking:
                provider = self.image_provider(view, shape, prepared_plan=prepared)
                self._start()
                runtime = self._runtime
            else:
                with self._lock:
                    self.no_job_passes += 1
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
            # Each runtime iterator carries an immutable source descriptor.
            # Planning, rendering and reconciliation for different scopes can
            # overlap; the tracker owns its bounded GPU/result-consumer lock.
            execution_started = True
            merged, stats, components = interpolate_sam_view_volume_pass(
                observed, image_provider=provider, view=view,
                runtime=runtime, scope=scope_metadata, policy=self.policy, cancel_event=self._cancel,
                prepared_plan=prepared,
                runtime_work_dir=self.temp_dir / 'sam_merged' / hashlib.sha256(str(scope).encode()).hexdigest()[:20],
                **kwargs)
            if merged is observed:
                merged = observation_volume
            stats = dict(stats)
            stats.setdefault('sam_crop_mode', self.crop_mode)
            stats.setdefault('sam_oversized_group_count', len(oversized_groups))
            stats.setdefault('sam_multi_tile_group_count', multi_tile_groups)
            stats.setdefault('sam_working_canvas_kind', scope_metadata['sam_working_canvas_kind'])
            stats.setdefault('sam_working_canvas_shape_tyx', list(shape))
            stats.setdefault('sam_native_view_shape_tyx', scope_metadata['sam_native_view_shape_tyx'])
            stats.setdefault('delayed_native_expansion_at_launch', self.delayed_native_expansion_at_launch)
            return merged, stats, components
        except BaseException as error:
            if prepared is not None and prepared.needs_tracking and not execution_started:
                destination = Path(kwargs.get('work_dir', self.evidence_root/hashlib.sha256(str(scope).encode()).hexdigest()[:20]))
                destination.mkdir(parents=True, exist_ok=True)
                receipt = {
                    'schema': 'xta.sam-context-preparation-failure/1', 'complete': False,
                    'status': 'infrastructure_invalid', 'phase': 'image_or_gpu_admission',
                    'error': str(error), 'scope_id': str(scope),
                    'pass_index': int(kwargs.get('pass_index', 1)),
                    'observation_snapshot_sha256': prepared.observation_snapshot_sha256,
                    'planning_settings_sha256': prepared.settings_sha256,
                    'native_shape_tyx': list(prepared.native_shape),
                    'runs': [{'run_id': str(run.run_id), 'group_id': str(run.group_id),
                              'status': 'not_attempted'} for run in prepared.runs],
                }
                target = destination/'context_preparation_failure.json'
                temporary = target.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(receipt, sort_keys=True), encoding='utf-8')
                temporary.replace(target)
            raise
        finally:
            with self._idle:
                self._active_passes -= 1
                self._idle.notify_all()

    def close(self):
        self.cancel('SAM interpolation runtime is closing')
        with self._idle:
            if self._closed:
                return
            while self._active_passes:
                self._idle.wait(timeout=0.25)
            close_error = None
            try:
                if self._runtime is not None:
                    dispatch = getattr(self._runtime, 'dispatch_stats', {})
                    self.dispatch_summary = dict(dispatch) if isinstance(dispatch, Mapping) else {}
                    try:
                        self._runtime.close()
                    except BaseException as error:
                        if getattr(self._runtime, 'residency_released', False) is not True:
                            # A retry still owns both model and device lease.
                            # Never advertise memory as available while a
                            # worker process may retain the predictor.
                            _retain_unsettled_context(self)
                            raise
                        close_error = error
                    self._runtime = None
                self._closed = True
                _forget_settled_context(self)
                for lease in reversed(self._leases):
                    lease.release()
                self._leases.clear()
                self._caches.clear()
                self._cache_transforms.clear()
                self.source_volume = None
            finally:
                self._idle.notify_all()
            if close_error is not None:
                raise close_error

