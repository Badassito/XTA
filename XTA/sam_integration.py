"""Image lifetime and GPU admission for integrated TTA SAM interpolation.

The tracker remains resident only after detector inference has completed and its
GPU assets have been retired.  Device-stage leases then protect the predictor
and bounded session scratch from concurrent projection allocation.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
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
    """Validate the canonical TTA samplers, independently of LTA restrictions."""
    from .sam_view_geometry import validate_sam_view_geometry
    for view in views:
        validate_sam_view_geometry(view)


def _intersect_bbox(left, right):
    y0, x0 = max(left[0], right[0]), max(left[1], right[1])
    y1, x1 = min(left[2], right[2]), min(left[3], right[3])
    return (y0, x0, y1, x1) if y0 < y1 and x0 < x1 else None


def _subtract_bbox(bbox, covered):
    """Disjoint rectangles left after one covered intersection."""
    y0, x0, y1, x1 = bbox
    cy0, cx0, cy1, cx1 = covered
    return tuple(item for item in ((y0, x0, cy0, x1), (cy1, x0, y1, x1),
                                  (cy0, x0, cy1, cx0), (cy0, cx1, cy1, x1))
                 if item[0] < item[2] and item[1] < item[3])


def _canonical_image_transform(transform):
    # These fields describe the detector pass, which has already been undone
    # before accumulation. Every sampler/source/basis field remains in the key.
    return {key: value for key, value in transform.items() if key not in
            {'runtime_view_name', 'detector_augmentation_angle_deg'}}


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
        from .sam_policy import resolve_sam_bridge_policy
        # Resolve launch-scoped controls before dispatching threaded scopes.
        # Pin only the guard fields: copying every inherited default here would
        # falsely turn workspace defaults into explicit resource caps.
        self.policy = dict(policy or {})
        if 'kind' in self.policy and 'mode' not in self.policy:
            self.policy = {'sam_bridge_policy': self.policy}
        resolved_policy = resolve_sam_bridge_policy(self.policy, generation_mode=self.crop_mode)
        declared_policy = self.policy.get('sam_bridge_policy')
        bridge_policy = (dict(declared_policy) if isinstance(declared_policy, Mapping)
                         else {'kind': resolved_policy['kind']})
        self.tight_crop_guard = bool(resolved_policy['strict_containment'])
        bridge_policy['strict_containment'] = self.tight_crop_guard
        if not self.tight_crop_guard:
            bridge_policy.update(guarded_rescue=False, name=resolved_policy['name'])
        self.policy['sam_bridge_policy'] = bridge_policy
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
        self._cache_entries = []
        self._resource_local = threading.local()
        self.resource_assignments = {}
        self._failure = ''
        self._closed = False
        self.wait_seconds = 0.0
        self.start_seconds = 0.0
        self.render_seconds = 0.0
        self.rendered_frames = 0
        self.rendered_pixels = 0
        self.cache_logical_bytes = 0
        self.exact_backing_reuses = 0
        self.image_cache_hits = 0
        self.image_cache_superset_hits = 0
        self.image_cache_reused_pixels = 0
        self.source_materializations = 0
        self.source_materialization_seconds = 0.0
        self.native_sampling_calls = 0
        self.native_sampling_pixels = 0
        self.canonical_sampling_pixels = 0
        self.canonical_phase_self_check_receipt = {'status': 'not_required'}
        self.planning_seconds = 0.0
        self.no_job_passes = 0
        self.dispatch_summary = {}
        self.shared_detector_devices = tuple(self.device_ids if detector_device_ids is None else
            sorted(set(self.device_ids) & {f'cuda:{int(str(value).split(":")[-1])}' for value in detector_device_ids}))
        if detector_device_ids is not None and not self.shared_detector_devices:
            self._ready.set()

    def detector_assets_retired(self) -> None:
        self._ready.set()

    @property
    def detector_retirement_ready(self) -> bool:
        """Read the admission signal without starting or rendering SAM work."""
        # Cancellation wakes existing waiters, but does not grant new work
        # detector-retirement or GPU-residency permission.
        return self._ready.is_set() and not self._cancel.is_set() and not self._closed

    @contextmanager
    def resource_scope(self, profile):
        """Bind this preparation thread's live lease through all SAM phases."""
        from .sam_resources import SamResourceProfile
        if not isinstance(profile, SamResourceProfile):
            raise TypeError('SAM resource scope requires an admitted profile')
        profile._validate_owner()
        previous = getattr(self._resource_local, 'profile', None)
        self._resource_local.profile = profile
        try:
            yield
        finally:
            self._resource_local.profile = previous

    def cancel(self, reason='TTA scheduler cancelled') -> None:
        self._failure = str(reason)
        self._cancel.set()
        self._ready.set()
        for runtime in (self._runtime, self._starting_runtime):
            cancel = getattr(runtime, 'cancel', None)
            if callable(cancel):
                cancel(str(reason))

    def _canvas_transform(self, view, shape):
        from .sam_view_geometry import sam_native_transform_record
        from .sam_canvas_rendering import (CANONICAL_CROP_RENDER_CONTRACT, IMPLEMENTATION_SHA256,
            CANONICAL_PHASE_SELF_CHECK_CONTRACT, canonical_sampling_backend)
        transform = sam_native_transform_record(view, shape, self.source_grid_shape,
            source_processing_shape_tyx=self.source_volume.shape)
        transform.update(canvas_shape_tyx=list(shape), source_grid_shape_tyx=list(self.source_grid_shape),
            source_resize_semantics=self.source_resize_semantics,
            interpolation_metric_units='view_native_frame_index_and_working_canvas_pixels',
            canonical_crop_render_contract=CANONICAL_CROP_RENDER_CONTRACT,
            canonical_crop_render_implementation_sha256=IMPLEMENTATION_SHA256,
            canonical_crop_sampling_backend=canonical_sampling_backend(),
            canonical_phase_self_check_contract=CANONICAL_PHASE_SELF_CHECK_CONTRACT)
        affine = np.asarray(transform['M_native_to_canvas'], dtype=np.float32)
        inverse = np.asarray(transform['M_canvas_to_native'], dtype=np.float32)
        return affine, inverse, transform

    def image_provider(self, view, shape, prepared_plan=None):
        """Materialize only missing canonical pixels; retain immutable cache files.

        The public canvas has native frame addresses. A cyclic plan may append
        aliases in its logical tracker cache; their u reflection is applied in
        working coordinates after the ordinary TTA sampler/affine.
        """
        from .geometry import physical_view_name
        from .lta_rendering import LtaPhysicalViewCacheRef
        from .sam_cyclic import mirror_bbox_yx, validate_cyclic_frame_addressing
        from .runtime import _interpolation_array_backing_path
        shape = tuple(int(value) for value in shape)
        if len(shape) != 3 or any(value <= 0 for value in shape) or shape[0] != int(view.num_slices):
            raise ValueError('SAM image canvas must match the positive view-native frame count')
        if shape[-2:] != (int(view.src_h), int(view.src_w)) and shape[1] != shape[2]:
            raise ValueError('SAM detector processing canvas must be native or square')
        plan = getattr(prepared_plan, 'plan', prepared_plan)
        logical_shape = tuple(getattr(plan, 'virtual_shape_tyx', ()) or shape)
        addressing = dict(getattr(plan, 'frame_addressing', {}) or {})
        if addressing:
            address_lookup = validate_cyclic_frame_addressing(addressing)
            if tuple(addressing['native_shape_tyx']) != shape or tuple(addressing['evidence_shape_tyx']) != logical_shape:
                raise ValueError('SAM cyclic image addresses differ from the prepared canvas')
        else:
            if logical_shape != shape:
                raise ValueError('SAM extended image frames require explicit cyclic addresses')
            address_lookup = None
        required = dict(getattr(prepared_plan, 'frame_crop_bounds', {}) or {}) if prepared_plan is not None else {
            index: (0, 0, shape[1], shape[2]) for index in range(shape[0])}
        if not required:
            raise ValueError('SAM image provider requires at least one planned tracking frame')
        required = {int(frame): tuple(int(value) for value in bbox) for frame, bbox in required.items()}
        for frame, (y0, x0, y1, x1) in required.items():
            if not 0 <= frame < logical_shape[0] or not (0 <= y0 < y1 <= shape[1] and 0 <= x0 < x1 <= shape[2]):
                raise ValueError('SAM image demand is outside the detector canvas')
        addresses = {frame: dict(address_lookup[frame]) if address_lookup is not None else
                     dict(unfolded_index=frame, native_index=frame, cycle_index=0, mirror_u=False)
                     for frame in required}
        demand_identity = hashlib.sha256(json.dumps(required, sort_keys=True).encode()).hexdigest()
        with self._lock:
            if self._closed:
                raise RuntimeError('SAM interpolation source lifetime has ended')
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
            validate_sam_interpolation_geometry([view])
            if addressing:
                from .sam_view_geometry import validate_sam_view_geometry
                validate_sam_view_geometry(view, wrap_axis=True)
            affine, inverse, transform = self._canvas_transform(view, shape)
            canonical_transform = _canonical_image_transform(transform)
            geometry_identity = hashlib.sha256(json.dumps(dict(source=self.source_identity,
                transform=canonical_transform), sort_keys=True, allow_nan=False).encode()).hexdigest()
            from .sam_cyclic import IMPLEMENTATION_SHA256 as cyclic_sha256
            image_identity = hashlib.sha256(json.dumps(dict(geometry=geometry_identity,
                shape=logical_shape, frame_addressing=addressing,
                cyclic_implementation_sha256=cyclic_sha256 if addressing else None),
                sort_keys=True, allow_nan=False).encode()).hexdigest()
            key = (image_identity, demand_identity)
            if key in self._caches:
                self._caches[key].revalidate()
                self.image_cache_hits += 1
                return self._caches[key]
            # A prior immutable descriptor can satisfy a later subset without
            # rendering or changing bytes still owned by an active worker.
            for entry in self._cache_entries:
                reference = entry['reference']
                if reference.identity_sha256 != image_identity or reference.shape != logical_shape:
                    continue
                coverage = {record[0]: record[1:5] for record in reference.frame_crops}
                if not coverage or all(frame in coverage and _intersect_bbox(bbox, coverage[frame]) == bbox
                                       for frame, bbox in required.items()):
                    reference.revalidate()
                    self._caches[key] = reference
                    self._cache_transforms[key] = transform
                    self.image_cache_superset_hits += 1
                    return reference
            render_started = time.perf_counter()
            # Shape equality is insufficient: a cubic sagittal/coronal stack
            # still has another source-axis recipe.
            source_array = (getattr(self.source_volume, '_array', None)
                if bool(getattr(self.source_volume, '_is_lazy_processing_cube', False)) else self.source_volume)
            backing = _interpolation_array_backing_path(source_array)
            if (str(view.family) == 'orthogonal' and physical_view_name(view) == 'transverse'
                    and logical_shape == shape and backing is not None and tuple(source_array.shape) == shape
                    and np.dtype(source_array.dtype) == np.uint8
                    and bool(source_array.flags['C_CONTIGUOUS'])
                    and np.array_equal(affine, np.array([[1., 0., 0.], [0., 1., 0.]], np.float32))
                    and Path(backing).is_file() and Path(backing).stat().st_size == int(np.prod(shape))):
                from .media import wait_for_volume_ready
                wait_for_volume_ready(self.source_volume)
                stat = Path(backing).stat()
                reference = LtaPhysicalViewCacheRef(path=Path(backing), shape=shape, dtype='uint8',
                    physical_view_id=physical_view_name(view), identity_sha256=image_identity,
                    size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
                self._caches[key] = reference
                self._cache_transforms[key] = transform
                self._cache_entries.append(dict(reference=reference, geometry_identity=geometry_identity,
                                                addresses={}, native_shape=shape))
                self.exact_backing_reuses += 1
                return reference
            identity_affine = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
            if not (np.array_equal(affine, identity_affine) and np.array_equal(inverse, identity_affine)):
                from .sam_canvas_rendering import ensure_canonical_phase_supported
                self.canonical_phase_self_check_receipt = ensure_canonical_phase_supported(
                    transform['canonical_crop_sampling_backend'])
            render_identity = hashlib.sha256((image_identity+demand_identity).encode()).hexdigest()[:24]
            path = self.temp_dir / 'sam_image_cache' / (f'{physical_view_name(view)}.{render_identity}.gray8.dat')
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
            completed = []
            try:
                for index, y0, x0, y1, x1, offset in records:
                    if self._cancel.is_set():
                        raise RuntimeError(self._failure)
                    address = addresses[index]
                    physical_bbox = mirror_bbox_yx((y0, x0, y1, x1), shape[2]) if address['mirror_u'] else (y0, x0, y1, x1)
                    target = cache[offset:offset+(y1-y0)*(x1-x0)].reshape(y1-y0, x1-x0)
                    physical_target = target[:, ::-1] if address['mirror_u'] else target
                    native_frame_cache = {}
                    missing = self._copy_cached_pixels(physical_target, physical_bbox,
                        int(address['native_index']), geometry_identity)
                    # Earlier native/alias records in this private transaction
                    # are already complete. Reuse their physical pixels before
                    # publishing an immutable descriptor to any worker.
                    for done_frame, done_bbox, done_mirror, done_offset in completed:
                        if done_frame != int(address['native_index']) or not missing:
                            continue
                        done_size = (done_bbox[2]-done_bbox[0])*(done_bbox[3]-done_bbox[1])
                        done = cache[done_offset:done_offset+done_size].reshape(
                            done_bbox[2]-done_bbox[0], done_bbox[3]-done_bbox[1])
                        if done_mirror:
                            done = done[:, ::-1]
                        remainder = []
                        for remaining in missing:
                            intersection = _intersect_bbox(remaining, done_bbox)
                            if intersection is None:
                                remainder.append(remaining)
                                continue
                            cy0, cx0, cy1, cx1 = intersection
                            physical_target[cy0-physical_bbox[0]:cy1-physical_bbox[0],
                                cx0-physical_bbox[1]:cx1-physical_bbox[1]] = done[
                                    cy0-done_bbox[0]:cy1-done_bbox[0], cx0-done_bbox[1]:cx1-done_bbox[1]]
                            self.image_cache_reused_pixels += (cy1-cy0)*(cx1-cx0)
                            remainder.extend(_subtract_bbox(remaining, intersection))
                        missing = tuple(remainder)
                        del done
                    for cy0, cx0, cy1, cx1 in missing:
                        if self._cancel.is_set():
                            raise RuntimeError(self._failure)
                        image = self._render_demand_crop(view, int(address['native_index']), affine, inverse,
                            output_height=cy1-cy0, output_width=cx1-cx0,
                            output_origin_yx=(cy0, cx0), output_canvas_width=shape[2],
                            native_frame_cache=native_frame_cache, native_preparation_bbox=physical_bbox)
                        if image.dtype != np.uint8 or image.shape != (cy1-cy0, cx1-cx0):
                            raise ValueError('SAM image provider returned a mismatched detector canvas')
                        physical_target[cy0-physical_bbox[0]:cy1-physical_bbox[0],
                                        cx0-physical_bbox[1]:cx1-physical_bbox[1]] = image
                        self.rendered_frames += 1
                        self.rendered_pixels += int(image.size)
                    completed.append((int(address['native_index']), physical_bbox,
                                      bool(address['mirror_u']), offset))
                    del target, physical_target, native_frame_cache
                cache.flush()
            except BaseException:
                cache._mmap.close()
                path.unlink(missing_ok=True)
                raise
            finally:
                del cache
            # Identity describes canonical image bytes, independently of how
            # much of that immutable canvas this compact descriptor stores.
            # Exact model/crop/frame feature keys can therefore survive a
            # different scope's demand inventory without admitting new pixels.
            stat = path.stat()
            reference = LtaPhysicalViewCacheRef(path=path, shape=logical_shape, dtype='uint8',
                physical_view_id=physical_view_name(view), identity_sha256=image_identity,
                size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                frame_crops=tuple(records) if prepared_plan is not None else ())
            self._caches[key] = reference
            self._cache_transforms[key] = transform
            self._cache_entries.append(dict(reference=reference, geometry_identity=geometry_identity,
                                            addresses=addresses, native_shape=shape))
            self.render_seconds += time.perf_counter() - render_started
            self.cache_logical_bytes += payload_bytes
            return reference

    def _copy_cached_pixels(self, target, bbox, native_frame, geometry_identity):
        """Copy intersections in physical working coordinates; return holes."""
        from .sam_cyclic import mirror_bbox_yx
        missing = [bbox]
        for entry in self._cache_entries:
            if entry['geometry_identity'] != geometry_identity:
                continue
            reference = entry['reference']
            if reference.frame_crops:
                records = [record for record in reference.frame_crops
                    if int(entry['addresses'][record[0]]['native_index']) == native_frame]
            else:
                records = [(native_frame, 0, 0, reference.shape[1], reference.shape[2], 0)]
            for frame, y0, x0, y1, x1, offset in records:
                mirrored = bool(entry['addresses'].get(frame, {}).get('mirror_u', False))
                coverage = mirror_bbox_yx((y0, x0, y1, x1), reference.shape[2]) if mirrored else (y0, x0, y1, x1)
                intersections = [(remaining, _intersect_bbox(remaining, coverage)) for remaining in missing]
                if not any(intersection is not None for _, intersection in intersections):
                    continue
                cached = reference.open()
                try:
                    plane = (cached[offset:offset+(y1-y0)*(x1-x0)].reshape(y1-y0, x1-x0)
                             if reference.frame_crops else cached[frame])
                    if mirrored:
                        plane = plane[:, ::-1]
                    remainder = []
                    for remaining, intersection in intersections:
                        if intersection is None:
                            remainder.append(remaining)
                            continue
                        cy0, cx0, cy1, cx1 = intersection
                        target[cy0-bbox[0]:cy1-bbox[0], cx0-bbox[1]:cx1-bbox[1]] = plane[
                            cy0-coverage[0]:cy1-coverage[0], cx0-coverage[1]:cx1-coverage[1]]
                        self.image_cache_reused_pixels += (cy1-cy0)*(cx1-cx0)
                        remainder.extend(_subtract_bbox(remaining, intersection))
                    missing = remainder
                    del plane
                finally:
                    cached._mmap.close()
                if not missing:
                    return ()
        return tuple(missing)

    def _render_demand_crop(self, view, index, affine, inverse, *, output_height, output_width,
                            output_origin_yx=(0, 0), output_canvas_width=None,
                            native_frame_cache=None, native_preparation_bbox=None):
        """Use the established TTA grayscale sampler on the canonical grid.

        Only a Transverse plane can be reconstructed from decoded Z slices.
        Other orientations materialize the existing shared processing memmap
        once, then reuse it for every needed frame/crop and endpoint session.
        """
        from ._deps import cv2
        from .geometry import physical_view_name
        from .sam_canvas_rendering import render_canonical_crop
        from .media import (_linear_source_index, _resize_gray_slice_nearest_or_linear,
                            wait_for_volume_ready, wait_for_volume_slice_ready)
        source = self.source_volume
        frames = None
        native_origin_xy = (0, 0)
        if native_frame_cache is not None and 'plane' in native_frame_cache:
            previous_pixels = int(native_frame_cache.get('sampled_output_pixels', 0))
            rendered = render_canonical_crop(source, view, index, affine=affine, inverse=inverse,
                output_origin_yx=output_origin_yx, output_height=output_height, output_width=output_width,
                output_canvas_width=output_canvas_width, native_frame_cache=native_frame_cache)
            self.canonical_sampling_pixels += int(native_frame_cache['sampled_output_pixels'])-previous_pixels
            return rendered
        lazy_unmaterialized = bool(getattr(source, '_is_lazy_processing_cube', False)) and not source.materialized
        transverse = str(view.family) == 'orthogonal' and physical_view_name(view) == 'transverse'
        if lazy_unmaterialized and not transverse:
            started = time.perf_counter()
            wait_for_volume_ready(source)
            self.source_materializations += 1
            self.source_materialization_seconds += time.perf_counter() - started
        elif not lazy_unmaterialized:
            wait_for_volume_ready(source)
        if bool(getattr(source, '_is_lazy_processing_cube', False)) and source.materialized:
            # Standard coronal/block and shell renderers require ndarray
            # strides/flags. Keep the owning lazy proxy alive in the context,
            # and lend its existing map rather than rematerializing a proxy.
            source = source._array
        if lazy_unmaterialized and transverse:
            decoded = source.source
            in_t, in_h, in_w = (int(value) for value in decoded.shape)
            out_t, out_h, out_w = (int(value) for value in source.shape)
            if (in_h, in_w) == (out_h, out_w) and not source.streaming_backend:
                # Match the production OpenCV slab resize exactly. Restrict the
                # spatial slab to the native pixels sampled by this tracker crop.
                cy0, cx0, cy1, cx1 = native_preparation_bbox or (output_origin_yx[0],
                    output_origin_yx[1], output_origin_yx[0]+output_height, output_origin_yx[1]+output_width)
                corners = np.array([[cx0, cy0], [cx1-1., cy0], [cx0, cy1-1.], [cx1-1., cy1-1.]])
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
                del slab, resized
                native_origin_xy = (x0, y0)
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
                    del other
            class OneNativeFrame:
                def __getitem__(self, frame_index):
                    if int(frame_index) != int(index):
                        raise RuntimeError('SAM demand renderer accessed an unplanned frame')
                    return native
            frames = OneNativeFrame()
        local_cache = native_frame_cache if native_frame_cache is not None else {}
        rendered = render_canonical_crop(source, view, index, affine=affine, inverse=inverse,
            output_origin_yx=output_origin_yx, output_height=output_height, output_width=output_width,
            view_frames=frames, native_origin_xy=native_origin_xy, output_canvas_width=output_canvas_width,
            native_frame_cache=local_cache)
        self.native_sampling_calls += 1
        self.native_sampling_pixels += int(local_cache['plane'].size)
        self.canonical_sampling_pixels += int(local_cache['sampled_output_pixels'])
        return rendered

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
                sam_tight_crop_guard=self.tight_crop_guard,
                sam_crop_tile_side=1008, sam_crop_halo=128,
                sam_crop_canvas_contract='current_interpolation_working_canvas',
                delayed_native_expansion_at_launch=self.delayed_native_expansion_at_launch,
                sam_working_canvas_kind=('native_view' if shape[-2:] ==
                    (int(view.src_h), int(view.src_w)) else 'detector_processing'),
                sam_working_canvas_shape_tyx=list(shape),
                sam_native_view_shape_tyx=[int(view.num_slices), int(view.src_h), int(view.src_w)])
            profile = getattr(self._resource_local, 'profile', None)
            live_profile = None
            if profile is not None:
                scope_metadata['sam_resource_profile'] = profile.metadata()
                live_profile = profile
                if profile.has_extra_credit:
                    from .sam_bridge_planning import SamPlanningLimits
                    from .sam_resources import validate_live_sam_resource_profile
                    validate_live_sam_resource_profile(profile)
                    declared_limits = kwargs.get('planner_limits')
                    if declared_limits is None:
                        kwargs['planner_limits'] = SamPlanningLimits(
                            max_group_bytes=profile.assigned_contract_bytes,
                            max_total_contract_bytes=profile.assigned_live_contract_bytes)
                    else:
                        kwargs['planner_limits'] = replace(declared_limits,
                            max_group_bytes=min(declared_limits.max_group_bytes, profile.assigned_contract_bytes),
                            max_total_contract_bytes=min(declared_limits.max_total_contract_bytes,
                                profile.assigned_live_contract_bytes))
            else:
                scope_metadata['sam_resource_profile'] = {
                    'schema': 'xta.sam_live_resources/1', 'status': 'direct_declared_bounds',
                    'reserved_extra_bytes': 0}
            with self._lock:
                self.resource_assignments[str(scope)] = dict(scope_metadata['sam_resource_profile'])
            profile_kwargs = {} if live_profile is None else {'resource_profile': live_profile}
            planning_keys = {'pass_index', 'gap_distance', 'min_radius', 'search_angle_deg',
                'interpolation_walk_back', 'interpolation_candidates', 'interpolation_passes',
                'wrap_axis', 'upstream_lineage', 'spacing_zyx', 'planner_limits', 'canonical_labels'}
            prepared = prepare_sam_interpolation_pass(observed, view=view,
                scope=scope_metadata, policy=self.policy,
                **profile_kwargs,
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
                **profile_kwargs,
                prepared_plan=prepared,
                runtime_work_dir=self.temp_dir / 'sam_merged' / hashlib.sha256(str(scope).encode()).hexdigest()[:20],
                **kwargs)
            if merged is observed:
                merged = observation_volume
            stats = dict(stats)
            stats.setdefault('sam_crop_mode', self.crop_mode)
            stats.setdefault('sam_tight_crop_guard', self.tight_crop_guard)
            stats.setdefault('sam_oversized_group_count', len(oversized_groups))
            stats.setdefault('sam_multi_tile_group_count', multi_tile_groups)
            stats.setdefault('sam_working_canvas_kind', scope_metadata['sam_working_canvas_kind'])
            stats.setdefault('sam_working_canvas_shape_tyx', list(shape))
            stats.setdefault('sam_native_view_shape_tyx', scope_metadata['sam_native_view_shape_tyx'])
            stats.setdefault('delayed_native_expansion_at_launch', self.delayed_native_expansion_at_launch)
            stats.setdefault('sam_canonical_phase_self_check', dict(self.canonical_phase_self_check_receipt))
            stats.setdefault('sam_resource_profile', dict(scope_metadata['sam_resource_profile']))
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
                self._cache_entries.clear()
                self.source_volume = None
            finally:
                self._idle.notify_all()
            if close_error is not None:
                raise close_error

