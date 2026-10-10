"""Image lifetime and GPU admission for integrated TTA SAM interpolation.

The tracker remains resident only after detector inference has completed and its
GPU assets have been retired. Resident ownership fences detector and auxiliary
work; exclusive compute leases protect startup, sessions and shutdown. Idle
devices may serve admitted projections after a worker CUDA-completion proof.
"""
from __future__ import annotations

import hashlib
from collections import deque
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
import threading
import time
import uuid
from typing import Mapping

import numpy as np
from .runtime import runtime_telemetry, sam_sessions_per_gpu


_UNSETTLED_SAM_CONTEXTS = {}
_UNSETTLED_SAM_LOCK = threading.Lock()


class SamConcurrentStartupResourceError(RuntimeError):
    """Measured resources cannot safely start two isolated predictor contexts."""


def _concurrent_startup_resource_failure(error):
    from .lta_workers import LtaWorkerStartupError
    if isinstance(error, SamConcurrentStartupResourceError):
        return True
    if not isinstance(error, LtaWorkerStartupError):
        return False
    event = error.event
    return (event.error_type in {'OutOfMemoryError', 'MemoryError'} or
            event.error_type == 'RuntimeError' and 'cuda out of memory' in event.message.lower())


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


def _write_context_preparation_failure(destination, receipt, error):
    """Diagnostic publication cannot replace the actual preparation failure."""
    try:
        from .json_publication import write_json_atomic
        write_json_atomic(Path(destination) / 'context_preparation_failure.json', receipt, sort_keys=True)
    except Exception as receipt_error:
        if callable(getattr(error, 'add_note', None)):
            error.add_note(f'SAM context failure receipt could not be written: {receipt_error}')


class SamInterpolationContext:
    """One run's persistent predictor, immutable image caches, and device leases."""

    def __init__(self, *, model_path, device_ids, temp_dir, evidence_root,
                 source_volume, source_identity, policy=None,
                 bundle_identity='', detector_identity='', source_grid_shape=None,
                 detector_device_ids=None, source_resize_semantics='caller_owned_exact_raster',
                 feature_cache_mib=1024, crop_mode='whole', delayed_native_expansion=None,
                 interpolation_policy_enabled=True, extrapolation_evidence_root=None,
                 adaptive_crop=False, progressive_startup=False, startup_pending_host_bytes=None):
        self.model_path = str(model_path)
        self.crop_mode = str(crop_mode).strip().lower()
        if self.crop_mode not in {'whole', 'tiled'}:
            raise ValueError('SAM crop mode must be whole or tiled')
        self.delayed_native_expansion_at_launch = delayed_native_expansion
        if adaptive_crop is not None and not isinstance(adaptive_crop, bool):
            raise ValueError('SAM adaptive crop must be a boolean or None')
        self.adaptive_crop = bool(adaptive_crop)
        self.crop_retry_policy = None
        if self.adaptive_crop:
            from .sam_crop_retry import SamCropRetryPolicy
            self.crop_retry_policy = SamCropRetryPolicy(enabled=True)
        self.feature_cache_mib = int(feature_cache_mib)
        if self.feature_cache_mib < 0:
            raise ValueError('SAM frame-feature cache MiB must be nonnegative')
        self.device_ids = tuple(f'cuda:{int(str(value).split(":")[-1])}' for value in device_ids)
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids) or any(
                int(value.split(':')[-1]) < 0 for value in self.device_ids):
            raise ValueError('SAM context requires unique nonnegative CUDA devices')
        self.sessions_per_gpu = sam_sessions_per_gpu()
        if not isinstance(progressive_startup, bool):
            raise TypeError('SAM progressive startup must be boolean')
        if startup_pending_host_bytes is not None and not callable(startup_pending_host_bytes):
            raise TypeError('SAM pending host commitment probe must be callable')
        shared = (self.device_ids if detector_device_ids is None else set(self.device_ids) &
            {f'cuda:{int(str(value).split(":")[-1])}' for value in detector_device_ids})
        # A shared retirement ACK proves YOLO has stopped creating RAM debt.
        # Mixed fleets retain legacy startup rather than fund before that proof.
        self.progressive_startup = bool(progressive_startup and len(self.device_ids) > 1
            and set(shared) == set(self.device_ids))
        self._startup_pending_host_bytes = startup_pending_host_bytes
        self.startup_admission = dict(requested_sessions_per_gpu=self.sessions_per_gpu,
            effective_sessions_per_gpu=self.sessions_per_gpu, attempts=[])
        if self.progressive_startup:
            self.startup_admission.update(effective_sessions_per_gpu=None,
                effective_sessions_per_device={}, admitted_device_ids=[], complete=False)
        self.temp_dir = Path(temp_dir)
        self.evidence_root = Path(evidence_root)
        self.extrapolation_evidence_root = (Path(extrapolation_evidence_root)
            if extrapolation_evidence_root is not None else self.evidence_root / 'extrapolation')
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
        self.tight_crop_guard = None
        if interpolation_policy_enabled:
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
        self._active_image_calls = 0
        self._active_image_cohorts = 0
        self._image_retirements = 0
        self._image_prefetches = set()
        self._retained_image_prefetch_credits = []
        self._image_builds = {}
        self._image_build_bytes = 0
        self._image_build_credit_bytes = 0
        self._image_build_peak_bytes = 0
        self._image_build_peak_count = 0
        self._source_materialization_lock = threading.Lock()
        self._runtime = None
        self._starting_runtime = None
        self._startup_pool = None
        self._progressive_pool = None
        self._progressive_threads = {}
        self._progressive_error = None
        self._startup_host_reserved = 0
        self._startup_host_grants = {}
        self._startup_fleet_funded = False
        self._startup_fleet_credit_bytes = 0
        self._startup_parent_envelope_bytes = 0
        self._startup_funding_lock = threading.Lock()
        self._startup_diagnostics_lock = threading.Lock()
        self._startup_progress = {}
        self._startup_wait_timeout = 300.
        self._startup_future_peaks = {}
        self._progressive_bundle = None
        self._startup_host_plan = None
        self._startup_host_condition = threading.Condition(threading.RLock())
        self._runtime_admitted = threading.Event()
        self._leases = []
        self._resident_leases = {}
        self._active_compute = {}
        self._shutdown_compute = {}
        self._gpu_lease_lock = threading.RLock()
        self._gpu_image_waiters = deque()
        self._gpu_image_target = None
        self._gpu_image_cursor = 0
        self._gpu_image_owners = {}
        self._gpu_image_sdk_owed = set()
        self._caches = {}
        self._cache_transforms = {}
        self._cache_entries = []
        self._cache_owners = {}
        self._image_sampling_proofs = {}
        self._unsettled_image_renderers = []
        self.image_cohort_retirement_receipts = []
        self.image_cache_retired_bytes = 0
        self.image_cohort_peak_owned_bytes = 0
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
        self._retired_devices = {int(device.split(':')[-1]) for device in self.device_ids
            if device not in self.shared_detector_devices}
        if detector_device_ids is not None and not self.shared_detector_devices:
            self._ready.set()
        if self.progressive_startup:
            from .lta_sam import resolve_local_sam_bundle
            from .sam_resources import GIB
            self._progressive_bundle = resolve_local_sam_bundle(self.model_path)
            peak = 2*int(Path(self._progressive_bundle.checkpoint_path).stat().st_size)*self.sessions_per_gpu+2*GIB
            self._startup_future_peaks = {int(device.split(':')[-1]):peak for device in self.device_ids}

    def detector_assets_retired(self) -> None:
        if self.progressive_startup:
            for device in self.device_ids:
                self.detector_device_assets_retired(int(device.split(':')[-1]))
        self._ready.set()

    def detector_device_assets_retired(self, device_index):
        """Pipeline calls only after this exact detector release ACK was verified."""
        if type(device_index) is not int or device_index < 0:
            raise ValueError('SAM detector retirement requires an exact nonnegative device ID')
        device = device_index
        configured = {int(value.split(':')[-1]) for value in self.device_ids}
        if device not in configured:
            return False
        if not self.progressive_startup:
            return False
        with self._runtime_lock:
            self._retired_devices.add(device)
            if configured <= self._retired_devices:
                self._ready.set()
            if self.progressive_startup and self._progressive_pool is not None:
                self._launch_progressive_devices()
        return True

    @property
    def runtime_start_ready(self):
        return (bool(self._retired_devices) if self.progressive_startup else self._ready.is_set()) and not self._cancel.is_set() and not self._closed

    @property
    def runtime_ready(self):
        return (self._runtime_admitted.is_set() if self.progressive_startup else self._runtime is not None) and not self._cancel.is_set() and not self._closed

    @property
    def gpu_image_ready(self):
        return self.runtime_ready if self.progressive_startup else self.detector_retirement_ready

    def check_startup(self):
        if self._progressive_error is not None:
            raise self._progressive_error
        if self._cancel.is_set() and not self._closed:
            raise RuntimeError(self._failure or 'SAM startup cancelled')

    def ensure_all_devices(self, timeout=300.):
        if not self.progressive_startup:
            self._start()
            return
        self._start_progressive()
        deadline = time.monotonic()+float(timeout)
        configured = {int(device.split(':')[-1]) for device in self.device_ids}
        while (set(self._progressive_pool.device_ids) != configured
                or any(thread.is_alive() for thread in self._progressive_threads.values())):
            self.check_startup()
            if time.monotonic() >= deadline:
                raise RuntimeError('SAM configured device startup did not complete')
            self._cancel.wait(.05)

    @property
    def worker_count(self):
        return len(self.device_ids)*self.sessions_per_gpu

    @property
    def worker_slots(self):
        return tuple((int(device.split(':')[-1]), index)
            for device in self.device_ids for index in range(self.sessions_per_gpu))

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
        with self._idle:
            self._failure = str(reason)
            self._cancel.set()
            self._ready.set()
            prefetches = tuple(self._image_prefetches)
            self._idle.notify_all()
        for prefetch in prefetches:
            prefetch.cancel_builder()
        self._quarantine_sam_residency(str(reason))
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

    def _register_image_cache_owner(self, reference, *, owned=False):
        key = str(reference.path)
        state = self._cache_owners.setdefault(key, dict(reference=reference, owned=False,
            persistent=False, leases=0, pins=0, retiring=False))
        state['owned'] |= bool(owned)
        state['persistent'] |= not bool(getattr(self._resource_local, 'ephemeral_images', False))
        current = sum(owner['reference'].size_bytes for owner in self._cache_owners.values()
            if owner['owned'] and not owner['persistent'])
        self.image_cohort_peak_owned_bytes = max(self.image_cohort_peak_owned_bytes, current)

    def image_cache_lifetime_snapshot(self):
        with self._lock:
            snapshot = dict(owned_cohort_current_bytes=sum(state['reference'].size_bytes
                    for state in self._cache_owners.values() if state['owned'] and not state['persistent']),
                owned_cohort_peak_bytes=self.image_cohort_peak_owned_bytes,
                owned_cache_retired_bytes=self.image_cache_retired_bytes,
                protected_owned_cache_bytes=sum(state['reference'].size_bytes
                    for state in self._cache_owners.values() if state['owned'] and state['persistent']),
                borrowed_source_logical_bytes=sum(state['reference'].size_bytes
                    for state in self._cache_owners.values() if not state['owned']),
                borrowed_source_bytes_are_not_staged_cache_bytes=True,
                retirement_unproven_count=sum(bool(state.get('retirement_unproven'))
                    for state in self._cache_owners.values()),
                active_image_builders=sum(bool(item.get('admitted')) for item in self._image_builds.values()),
                pending_image_builds=len(self._image_builds),
                active_image_build_bytes=self._image_build_bytes,
                active_image_build_credit_bytes=self._image_build_credit_bytes,
                image_build_peak_bytes=self._image_build_peak_bytes,
                image_build_peak_count=self._image_build_peak_count,
                image_cache_borrower_pins=sum(state.get('pins', 0) for state in self._cache_owners.values()),
                pinned_image_cache_logical_bytes=sum(state['reference'].size_bytes for state in self._cache_owners.values()
                    if state.get('pins', 0)),
                active_image_calls=self._active_image_calls,
                live_image_prefetches=len(self._image_prefetches),
                unconsumed_image_prefetches=sum(not prefetch._entered for prefetch in self._image_prefetches),
                image_prefetch_charged_bytes=sum(prefetch._admitted_bytes for prefetch in self._image_prefetches),
                retained_image_prefetch_credit_bytes=sum(getattr(release, 'image_phase_bytes', 0)
                    for release in self._retained_image_prefetch_credits))
            profile = getattr(self._resource_local, 'profile', None)
            pool = self._startup_pool or (profile._lease.pool if profile is not None else None)
        if pool is not None:
            from .sam_resources import sam_image_staging_snapshot
            with pool.condition:
                snapshot.update(sam_image_staging_snapshot(pool))
        return snapshot

    def prefetch_image_cohort(self, view, shape, prepared_plan, *, max_cache_bytes=None):
        """Try current/next cohorts per parent, each on fresh image-only credit."""
        from .sam_image_prefetch import SamImageCohortPrefetch
        from .workspace import _env_int
        profile = getattr(self._resource_local, 'profile', None)
        if profile is None or not callable(getattr(self._runtime, 'release_source_cache', None)):
            return None
        profile._validate_owner()
        required = dict(getattr(prepared_plan, 'frame_crop_bounds', {}) or {})
        payload = sum((int(box[2])-int(box[0]))*(int(box[3])-int(box[1])) for box in required.values())
        cap = (max(1, _env_int('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', 1024**3))
               if max_cache_bytes is None else int(max_cache_bytes))
        if not required or payload <= 0 or payload > cap:
            return None  # The ordinary provider remains the admission authority.
        scratch = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
        with self._idle:
            self._check_image_lifetime()
            if (sum(prefetch.parent is profile for prefetch in self._image_prefetches) >= 2
                    or any(owner.get('retirement_unproven') for owner in self._cache_owners.values())):
                return None
            prefetch = SamImageCohortPrefetch(self, view, tuple(shape), prepared_plan,
                cap, profile, payload+scratch)
            self._image_prefetches.add(prefetch)
        try:
            prefetch.start()
        except BaseException:
            with self._idle:
                self._image_prefetches.discard(prefetch)
                self._idle.notify_all()
            raise
        return prefetch

    def _prepare_image_cohort(self, view, shape, prepared_plan, *, max_cache_bytes=None):
        if self._can_prefetch_cpu_images(view, prepared_plan):
            pending = self.prefetch_image_cohort(view, shape, prepared_plan,
                max_cache_bytes=max_cache_bytes)
            if pending is not None:
                return pending
        return self.image_cohort_provider(view, shape, prepared_plan,
            max_cache_bytes=max_cache_bytes)

    @contextmanager
    def image_cohort_provider(self, view, shape, prepared_plan, *, max_cache_bytes=None):
        """Hold an immutable cohort file through its final worker handoff.

        Provider overrides must use _claim_image_reference under the registry
        lock; this invocation's claim ticket records its atomic lease. Returning
        a bare reference, or another consumer's lease, is a lifecycle error.
        """
        reference = None
        claim = dict(reference=None)
        with self._idle:
            self._check_image_lifetime()
            if any(state.get('retirement_unproven') for state in self._cache_owners.values()):
                raise RuntimeError('SAM cannot stage another image cohort while an owned cache retirement is unproven')
            self._active_image_cohorts += 1
        previous = getattr(self._resource_local, 'ephemeral_images', False)
        previous_cap = getattr(self._resource_local, 'image_cache_byte_cap', None)
        previous_lease = getattr(self._resource_local, 'image_cohort_lease', False)
        previous_claim = getattr(self._resource_local, 'image_cohort_claim', None)
        try:
            self._resource_local.ephemeral_images = True
            self._resource_local.image_cache_byte_cap = max_cache_bytes
            self._resource_local.image_cohort_lease = True
            self._resource_local.image_cohort_claim = claim
            try:
                # Lookup/publication claims this invocation's lease atomically.
                reference = self.image_provider(view, shape, prepared_plan=prepared_plan)
            finally:
                self._resource_local.ephemeral_images = previous
                self._resource_local.image_cache_byte_cap = previous_cap
                self._resource_local.image_cohort_lease = previous_lease
                self._resource_local.image_cohort_claim = previous_claim
            with self._idle:
                owner = self._cache_owners.get(str(reference.path))
                if claim['reference'] != reference or owner is None or owner['leases'] <= 0:
                    raise RuntimeError('SAM image provider returned a missing atomic cohort lease')
            yield reference
        finally:
            primary_error = sys.exc_info()[1]
            held_reference = claim['reference']
            try:
                if held_reference is not None:
                    retire_owner = False
                    with self._idle:
                        state = self._cache_owners.get(str(held_reference.path))
                        valid_lease = state is not None and state['leases'] > 0
                        if not valid_lease:
                            error = RuntimeError('SAM image cohort lease underflow or missing owner')
                            if state is not None and state['owned']:
                                state['persistent'] = True
                                state['retirement_unproven'] = True
                            if primary_error is None:
                                raise error
                            if callable(getattr(primary_error, 'add_note', None)):
                                primary_error.add_note(str(error))
                        else:
                            state['leases'] -= 1
                            if state['leases'] or state['persistent'] or not state['owned']:
                                self.image_cohort_retirement_receipts.append(dict(status='retained_shared_or_borrowed',
                                    path=str(held_reference.path), leases=int(state['leases']), owned=state['owned']))
                            elif state.get('retiring'):
                                self.image_cohort_retirement_receipts.append(dict(status='retirement_pending_shared_owner',
                                    path=str(held_reference.path), leases=0, owned=state['owned']))
                            elif not callable(getattr(self._runtime, 'release_source_cache', None)):
                                state['persistent'] = True
                                state['retirement_unproven'] = True
                                self.image_cohort_retirement_receipts.append(dict(
                                    status='retained_unproven_runtime_mapping_lifetime', path=str(held_reference.path)))
                            else:
                                state['retiring'] = True
                                self._image_retirements += 1
                                retire_owner = True
                        self._idle.notify_all()
                    if retire_owner:
                        try:
                            self._retire_image_cohort_owner(held_reference, state, primary_error)
                        finally:
                            with self._idle:
                                self._image_retirements -= 1
                                self._idle.notify_all()
            finally:
                with self._idle:
                    self._active_image_cohorts -= 1
                    self._idle.notify_all()

    def _retire_image_cohort_owner(self, reference, state, primary_error):
        claimed = False
        try:
            with self._idle:
                # Hide the descriptor before waiting so fresh donor/lookups do
                # not extend this retirement indefinitely. Already promised
                # same-demand consumers may still convert their pin to a lease.
                claimed = True
                while state.get('pins', 0):
                    self._idle.wait(timeout=0.05)
                if state['leases'] or state['persistent']:
                    self.image_cohort_retirement_receipts.append(dict(status='retained_shared_or_borrowed',
                        path=str(reference.path), leases=int(state['leases']), owned=state['owned']))
                    state['retiring'] = False
                    claimed = False
                    self._idle.notify_all()
                    return
            # Check this descriptor's worker-mapping proof outside the image registry lock.
            proof = self._runtime.release_source_cache(reference)
            if (not isinstance(proof, Mapping) or proof.get('status') != 'retired'
                    or proof.get('workers_finished') is not True
                    or proof.get('gray_mappings_retired') is not True):
                raise RuntimeError('SAM cohort worker cache-mapping retirement was not proven')
            reference.revalidate()
            with self._idle:
                if state['leases'] or state.get('pins', 0) or state['persistent']:
                    raise RuntimeError('SAM cohort acquired an unexpected owner during retirement')
                reference.path.unlink()
                for key, cached in tuple(self._caches.items()):
                    if cached.path == reference.path:
                        del self._caches[key]
                        self._cache_transforms.pop(key, None)
                self._cache_entries[:] = [entry for entry in self._cache_entries
                    if entry['reference'].path != reference.path]
                self._cache_owners.pop(str(reference.path), None)
                self.cache_logical_bytes -= reference.size_bytes
                self.image_cache_retired_bytes += reference.size_bytes
                self.image_cohort_retirement_receipts.append(dict(proof,
                    path=str(reference.path), retired_bytes=reference.size_bytes))
        except BaseException as error:
            with self._idle:
                state['persistent'] = True
                state['retirement_unproven'] = True
                self.image_cohort_retirement_receipts.append(dict(
                    status='retained_unproven_runtime_mapping_lifetime', path=str(reference.path),
                    error=str(error)))
            if primary_error is None:
                raise
            if callable(getattr(primary_error, 'add_note', None)):
                primary_error.add_note(f'SAM image cohort retirement remains unproven: {error}')
        finally:
            if claimed:
                with self._idle:
                    state['retiring'] = False
                    self._idle.notify_all()

    def image_provider(self, view, shape, prepared_plan=None):
        """Return an immutable reference with an atomic cohort lease when asked.

        Implementations overriding this method must claim their result through
        _claim_image_reference while holding _idle before returning a reference.
        """
        from .sam_interpolation import _trace_sam_phase
        operation, scope_id = getattr(self._resource_local, 'sam_phase_scope', ('', ''))
        with self._idle:
            self._check_image_lifetime()
            self._active_image_calls += 1
        try:
            with _trace_sam_phase('image_render', scope_id, operation=operation):
                from .sam_gpu_rendering import (try_gpu_crop_renderer, SamGpuRenderingUnavailable,
                    clear_image_error_frames)
                if (self._cache_entries and self.detector_retirement_ready
                        and getattr(self._resource_local, 'profile', None) is not None):
                    # An immutable complete CPU cache needs no upload/rerender.
                    cached = self._image_provider(view, shape, prepared_plan=prepared_plan, _cache_only=True)
                    if cached is not None:
                        return cached
                cpu_prefetch = (getattr(self._resource_local, 'image_prefetch_cancel', None) is not None
                    and self._can_prefetch_cpu_images(view, prepared_plan))
                cpu_started = time.perf_counter() if cpu_prefetch else None
                if cpu_prefetch:
                    renderer = None
                    runtime_telemetry().add('sam.gpu_images.cpu_admissions', 1)
                    runtime_telemetry().add('sam.cpu_images.prefetch_admissions', 1)
                    runtime_telemetry().gauge('sam.gpu_images.last_cpu_reason', 'bounded_cpu_prefetch')
                else:
                    renderer = try_gpu_crop_renderer(self, view, shape, prepared_plan)
                previous = getattr(self._resource_local, 'gpu_image_renderer', None)
                try:
                    self._resource_local.gpu_image_renderer = renderer
                    try:
                        return self._image_provider(view, shape, prepared_plan=prepared_plan)
                    except RuntimeError as error:
                        cause = error
                        while cause is not None and not isinstance(cause, SamGpuRenderingUnavailable):
                            cause = cause.__cause__
                        if renderer is None or cause is None:
                            raise
                        self._check_image_lifetime()
                        clear_image_error_frames(error)
                        renderer.close()
                        renderer = None
                        self._resource_local.gpu_image_renderer = None
                        runtime_telemetry().add('sam.gpu_images.cpu_fallbacks', 1)
                        runtime_telemetry().gauge('sam.gpu_images.last_fallback_reason', str(cause))
                        return self._image_provider(view, shape, prepared_plan=prepared_plan)
                finally:
                    self._resource_local.gpu_image_renderer = previous
                    if cpu_started is not None:
                        runtime_telemetry().add('sam.cpu_images.prefetch_host_seconds',
                            time.perf_counter()-cpu_started)
                    if renderer is not None:
                        clear_image_error_frames(sys.exc_info()[1])
                        renderer.close()
        finally:
            with self._idle:
                self._active_image_calls -= 1
                self._idle.notify_all()

    def _can_prefetch_cpu_images(self, view, prepared_plan):
        if str(view.family) not in {'radial', 'spherical'}:
            return False
        source = self.source_volume
        if bool(getattr(source, '_is_lazy_processing_cube', False)):
            source = source._array if source.materialized else None
        if (not isinstance(source, np.ndarray) or source.ndim != 3 or source.dtype != np.uint8
                or tuple(source.shape) != (int(view.full_t), int(view.full_h), int(view.full_w))):
            return False
        from .media import volume_readiness
        ready = volume_readiness(source)
        if ready is not None and not ready._all_event.is_set():
            return False
        required = dict(getattr(prepared_plan, 'frame_crop_bounds', {}) or {})
        if not required:
            return False
        from .workspace import _env_int
        from .sam_canvas_rendering import native_shell_workspace_bytes
        crop_bytes = max((int(box[2])-int(box[0]))*(int(box[3])-int(box[1]))
                         +64*(int(box[3])-int(box[1])+32) for box in required.values())
        return native_shell_workspace_bytes(view)+crop_bytes <= max(
            1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))

    def _check_image_lifetime(self):
        if self._closed:
            raise RuntimeError('SAM interpolation source lifetime has ended')
        if self._cancel.is_set():
            raise RuntimeError(self._failure)
        prefetch_cancel = getattr(self._resource_local, 'image_prefetch_cancel', None)
        if prefetch_cancel is not None and prefetch_cancel.is_set():
            raise RuntimeError('SAM image prefetch was abandoned')
        if getattr(self._resource_local, 'ephemeral_images', False) and any(
                state.get('retirement_unproven') for state in self._cache_owners.values()):
            raise RuntimeError('SAM cannot stage another image cohort while an owned cache retirement is unproven')

    def _claim_image_reference(self, reference, *, owned=False):
        """Publish/lookup and cohort lease acquisition share the registry lock."""
        claim = None
        if getattr(self._resource_local, 'image_cohort_lease', False):
            claim = getattr(self._resource_local, 'image_cohort_claim', None)
            if claim is None or claim['reference'] is not None:
                raise RuntimeError('SAM image provider lacks a fresh atomic cohort claim ticket')
        self._register_image_cache_owner(reference, owned=owned)
        if claim is not None:
            self._cache_owners[str(reference.path)]['leases'] += 1
            claim['reference'] = reference
        return reference

    def _add_image_metrics(self, **values):
        with self._lock:
            for key, value in values.items():
                setattr(self, key, getattr(self, key)+value)

    def _admit_image_build(self, payload_bytes, scratch_bytes, cache_budget):
        """Use distinct live parent credits, or one aggregate legacy allowance."""
        from .sam_resources import validate_live_sam_resource_profile
        profile = getattr(self._resource_local, 'profile', None)
        assigned = validate_live_sam_resource_profile(profile) if profile is not None else None
        credit_bytes = int(payload_bytes)+int(scratch_bytes)
        credited = (assigned is not None
            and credit_bytes <= int(assigned['base_non_cpu_allowance_bytes']))
        lease_id = assigned['lease_id'] if credited else None
        fallback_limit = int(cache_budget)+int(scratch_bytes)
        while True:
            self._check_image_lifetime()
            active = tuple(item for item in self._image_builds.values() if item.get('admitted'))
            # A single producer's allowance cannot be spent twice. Uncredited
            # callers share the previous target+render peak instead of minting
            # a new parent reservation merely by starting another thread.
            uncredited = tuple(item for item in active if not item['credited'])
            # Parent executor lanes already bound CPU producers. Independently
            # owned image credits may render together; a global count limit
            # leaves every GPU waiting behind two large view-cache builders.
            fits = (credited and all(item['lease_id'] != lease_id for item in active)
                or not credited and sum(item['credit_bytes'] for item in uncredited)+credit_bytes
                    <= min([fallback_limit]+[item['fallback_limit'] for item in uncredited]))
            if fits:
                return dict(credited=credited, lease_id=lease_id, credit_bytes=credit_bytes,
                    payload_bytes=int(payload_bytes), fallback_limit=fallback_limit)
            self._idle.wait(timeout=0.05)

    def _image_provider(self, view, shape, prepared_plan=None, *, _cache_only=False):
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
        from .workspace import _env_int
        pinned_cap = getattr(self._resource_local, 'image_cache_byte_cap', None)
        cache_budget = (int(pinned_cap) if pinned_cap is not None else
            max(1, _env_int('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', 1024**3)))
        scratch_bytes = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
        records, payload_bytes = [], 0
        for index, (y0, x0, y1, x1) in sorted(required.items()):
            records.append((index, y0, x0, y1, x1, payload_bytes))
            payload_bytes += (y1-y0)*(x1-x0)
        with self._idle:
            self._check_image_lifetime()
            validate_sam_interpolation_geometry([view])
            if addressing:
                from .sam_view_geometry import validate_sam_view_geometry
                validate_sam_view_geometry(view, wrap_axis=True)
            affine, inverse, transform = self._canvas_transform(view, shape)
            from .sam_cyclic import IMPLEMENTATION_SHA256 as cyclic_sha256
            # Geometry/source proof excludes only numerical sampler provenance.
            # Cache/feature identity below still includes the chosen byte contract.
            image_geometry_identity = hashlib.sha256(json.dumps(dict(source=self.source_identity,
                transform={field: value for field, value in _canonical_image_transform(transform).items()
                    if not field.startswith(('canonical_crop_', 'canonical_phase_'))},
                shape=logical_shape, frame_addressing=addressing,
                cyclic_implementation_sha256=cyclic_sha256 if addressing else None), sort_keys=True,
                allow_nan=False).encode()).hexdigest()
            gpu_renderer = getattr(self._resource_local, 'gpu_image_renderer', None)
            from .sam_canvas_rendering import CANONICAL_CROP_RENDER_CONTRACT
            sampling = dict(contract=CANONICAL_CROP_RENDER_CONTRACT, backend='cpu',
                numerical_backend=transform['canonical_crop_sampling_backend'])
            if gpu_renderer is None and str(view.family) in {'radial', 'spherical'}:
                from .sam_canvas_rendering import NATIVE_SHELL_CROP_CONTRACT
                # Different ROI shapes may round differently; feature reuse
                # must not promise identical bytes across those demands.
                sampling.update(native_sampler=dict(contract=NATIVE_SHELL_CROP_CONTRACT,
                    absolute_tolerance=1.0), demand_sha256=demand_identity)
                transform.update(canonical_crop_native_sampler=sampling['native_sampler'],
                    canonical_crop_native_demand_sha256=demand_identity)
            if gpu_renderer is not None:
                # Translating a float32 CUDA affine can change a crop's
                # numerical phase. Different demand origins must not donate
                # pixels or share frame features under one image identity.
                sampling = dict(gpu_renderer.sampling_identity(),
                    demand_identity_sha256=demand_identity)
                transform.update(canonical_crop_render_contract=sampling['contract'],
                    canonical_crop_render_implementation_sha256=sampling['implementation_sha256'],
                    canonical_crop_sampling_backend=sampling,
                    canonical_phase_self_check_contract='registered_TTA_CUDA_intensity_sampler')
            canonical_transform = _canonical_image_transform(transform)
            geometry_identity = hashlib.sha256(json.dumps(dict(source=self.source_identity,
                transform=canonical_transform), sort_keys=True, allow_nan=False).encode()).hexdigest()
            from .sam_cyclic import IMPLEMENTATION_SHA256 as cyclic_sha256
            image_identity = hashlib.sha256(json.dumps(dict(geometry=geometry_identity,
                shape=logical_shape, frame_addressing=addressing,
                cyclic_implementation_sha256=cyclic_sha256 if addressing else None),
                sort_keys=True, allow_nan=False).encode()).hexdigest()
            key = (image_identity, demand_identity)
            while True:
                self._check_image_lifetime()
                exact = self._caches.get(key)
                exact_state = self._cache_owners.get(str(exact.path), {}) if exact is not None else {}
                if exact is not None and exact_state.get('retiring'):
                    if _cache_only:
                        return None
                    self._idle.wait(timeout=0.05)
                    continue
                if exact is not None and not (pinned_cap is not None and exact_state.get('owned')
                        and exact.size_bytes > cache_budget):
                    exact.revalidate()
                    self.image_cache_hits += 1
                    return self._claim_image_reference(exact)
                for entry in self._cache_entries:
                    reference = entry['reference']
                    owner = self._cache_owners.get(str(reference.path), {})
                    if (owner.get('retiring') or reference.identity_sha256 != image_identity
                            or reference.shape != logical_shape
                            or pinned_cap is not None and owner.get('owned') and reference.size_bytes > cache_budget):
                        continue
                    coverage = {record[0]: record[1:5] for record in reference.frame_crops}
                    if not coverage or all(frame in coverage and _intersect_bbox(bbox, coverage[frame]) == bbox
                                           for frame, bbox in required.items()):
                        reference.revalidate()
                        self._caches[key] = reference
                        self._cache_transforms[key] = transform
                        self.image_cache_superset_hits += 1
                        return self._claim_image_reference(reference)
                if _cache_only:
                    return None
                ticket = self._image_builds.get(key)
                if ticket is not None:
                    ticket['waiters'] += 1
                    try:
                        while not ticket['done']:
                            self._check_image_lifetime()
                            self._idle.wait(timeout=0.05)
                        self._check_image_lifetime()
                        if ticket['error'] is not None:
                            raise RuntimeError('SAM shared image transaction failed') from ticket['error']
                        reference = ticket['reference']
                        if (pinned_cap is not None and self._cache_owners[str(reference.path)]['owned']
                                and reference.size_bytes > cache_budget):
                            raise RuntimeError(f'SAM planned image demand {reference.size_bytes} bytes exceeds cache budget {cache_budget}')
                        reference.revalidate()
                        self.image_cache_hits += 1
                        return self._claim_image_reference(reference)
                    finally:
                        ticket['waiters'] -= 1
                        if ticket['reference'] is not None:
                            self._cache_owners[str(ticket['reference'].path)]['pins'] -= 1
                        self._idle.notify_all()
                # Admission is a private target bound. A borrowed native backing
                # can bypass this later without generating an owned cache.
                source_array = (getattr(self.source_volume, '_array', None)
                    if bool(getattr(self.source_volume, '_is_lazy_processing_cube', False)) else self.source_volume)
                backing = _interpolation_array_backing_path(source_array)
                borrow_backing = (gpu_renderer is None and str(view.family) == 'orthogonal' and physical_view_name(view) == 'transverse'
                    and logical_shape == shape and backing is not None and tuple(source_array.shape) == shape
                    and np.dtype(source_array.dtype) == np.uint8 and bool(source_array.flags['C_CONTIGUOUS'])
                    and np.array_equal(affine, np.array([[1., 0., 0.], [0., 1., 0.]], np.float32))
                    and Path(backing).is_file() and Path(backing).stat().st_size == int(np.prod(shape)))
                if payload_bytes > cache_budget and not borrow_backing:
                    raise RuntimeError(f'SAM planned image demand {payload_bytes} bytes exceeds cache budget {cache_budget}')
                ticket = dict(done=False, error=None, reference=None, waiters=0, admitted=False)
                self._image_builds[key] = ticket
                try:
                    reservation = self._admit_image_build(0 if borrow_backing else payload_bytes,
                        0 if borrow_backing else scratch_bytes, cache_budget)
                    ticket.update(reservation, admitted=True)
                    donor_entries = tuple(entry for entry in self._cache_entries
                        if entry['geometry_identity'] == geometry_identity
                        and not self._cache_owners[str(entry['reference'].path)].get('retiring'))
                    donors = tuple({str(entry['reference'].path): self._cache_owners[str(entry['reference'].path)]
                        for entry in donor_entries}.values())
                    for owner in donors:
                        owner['reference'].revalidate()
                    for owner in donors:
                        owner['pins'] += 1
                    ticket['donors'] = donors
                    self._image_build_bytes += ticket['payload_bytes']
                    self._image_build_credit_bytes += ticket['credit_bytes']
                    self._image_build_peak_bytes = max(self._image_build_peak_bytes, self._image_build_bytes)
                    self._image_build_peak_count = max(self._image_build_peak_count,
                        sum(bool(value.get('admitted')) for value in self._image_builds.values()))
                except BaseException as error:
                    ticket.update(done=True, error=error)
                    self._image_builds.pop(key, None)
                    self._idle.notify_all()
                    raise
                break

        def build_reference():
            render_started = time.perf_counter()
            # Shape equality is insufficient: a cubic sagittal/coronal stack
            # still has another source-axis recipe.
            source_array = (getattr(self.source_volume, '_array', None)
                if bool(getattr(self.source_volume, '_is_lazy_processing_cube', False)) else self.source_volume)
            backing = _interpolation_array_backing_path(source_array)
            if (gpu_renderer is None and str(view.family) == 'orthogonal' and physical_view_name(view) == 'transverse'
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
                self._add_image_metrics(exact_backing_reuses=1)
                return reference, False, dict(reference=reference, geometry_identity=geometry_identity,
                    addresses={}, native_shape=shape)
            identity_affine = np.array([[1., 0., 0.], [0., 1., 0.]], np.float32)
            if gpu_renderer is None and not (np.array_equal(affine, identity_affine) and np.array_equal(inverse, identity_affine)):
                from .sam_canvas_rendering import ensure_canonical_phase_supported
                receipt = ensure_canonical_phase_supported(transform['canonical_crop_sampling_backend'])
                with self._lock:
                    self.canonical_phase_self_check_receipt = receipt
            render_identity = hashlib.sha256((image_identity+demand_identity).encode()).hexdigest()[:24]
            path = self.temp_dir / 'sam_image_cache' / (
                f'{physical_view_name(view)}.{render_identity}.{uuid.uuid4().hex[:12]}.gray8.dat')
            with self._lock:
                ticket['private_path'] = path
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cache = np.memmap(path, dtype=np.uint8, mode='w+', shape=(payload_bytes,))
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            completed = []
            target = physical_target = done = native_frame_cache = native_plane = None
            batch_iterator = None
            native_iterator = None
            native_rois = {}
            batch_images = {}
            try:
                if gpu_renderer is None:
                    batch_iterator = self._transverse_batch_iterator(view, shape, records, addresses,
                        affine, inverse, geometry_identity, payload_bytes, cache_entries=donor_entries)
                    prefetch_cancel = getattr(self._resource_local, 'image_prefetch_cancel', None)
                    if prefetch_cancel is not None and self._can_prefetch_cpu_images(view, prepared_plan):
                        from .sam_canvas_rendering import (iter_prefetched_native_planes,
                                                          native_shell_workspace_bytes,
                                                          native_shell_crop_bbox)
                        for index, y0, x0, y1, x1, _offset in records:
                            address = addresses[index]
                            box = ((y0, x0, y1, x1) if not address['mirror_u'] else
                                mirror_bbox_yx((y0, x0, y1, x1), shape[2]))
                            roi = native_shell_crop_bbox(view, affine, box)
                            frame = int(address['native_index'])
                            previous_roi = native_rois.get(frame)
                            if previous_roi is not None:
                                roi = previous_roi if roi is None else (
                                    min(previous_roi[0], roi[0]), min(previous_roi[1], roi[1]),
                                    max(previous_roi[2], roi[2]), max(previous_roi[3], roi[3]))
                            native_rois[frame] = roi
                        native_bytes = max(((box[2]-box[0])*(box[3]-box[1])
                            if box is not None else 1 for box in native_rois.values()), default=1)
                        native_work_bytes = max((native_shell_workspace_bytes(view, box)
                            if box is not None else 1 for box in native_rois.values()), default=1)
                        remap_minimum = native_bytes+max((y1-y0)*(x1-x0)+64*(x1-x0+32)
                            for _index, y0, x0, y1, x1, _offset in records)
                        if (not donor_entries and len(records) > 1 and scratch_bytes >= native_work_bytes
                                +remap_minimum+native_bytes):
                            def check_native_preparation():
                                with self._idle:
                                    self._check_image_lifetime()
                                    if prefetch_cancel.is_set():
                                        raise RuntimeError('SAM image prefetch was abandoned')
                                    if any(owner.get('retirement_unproven') for owner in self._cache_owners.values()):
                                        raise RuntimeError('SAM image cache retirement is unproven')
                            native_source = (self.source_volume._array
                                if bool(getattr(self.source_volume, '_is_lazy_processing_cube', False))
                                else self.source_volume)
                            native_iterator = iter_prefetched_native_planes(native_source, view,
                                (addresses[index]['native_index'] for index, *_ in records),
                                max_workspace_bytes=scratch_bytes,
                                min_remap_workspace_bytes=remap_minimum,
                                check_cancel=check_native_preparation,
                                native_crop_bounds=native_rois)
                            runtime_telemetry().add('sam.cpu_images.native_prefetch_cohorts', 1)
                for index, y0, x0, y1, x1, offset in records:
                    self._check_image_lifetime()
                    address = addresses[index]
                    physical_bbox = mirror_bbox_yx((y0, x0, y1, x1), shape[2]) if address['mirror_u'] else (y0, x0, y1, x1)
                    target = cache[offset:offset+(y1-y0)*(x1-x0)].reshape(y1-y0, x1-x0)
                    physical_target = target[:, ::-1] if address['mirror_u'] else target
                    pre_rendered = None
                    if batch_iterator is not None:
                        if not batch_images:
                            batch, counters = next(batch_iterator)
                            batch_images = {frame: image for frame, _bbox, image in batch}
                            if counters.get('batches'):
                                self._add_image_metrics(native_sampling_calls=counters['frames'],
                                    native_sampling_pixels=counters['native_prepared_pixels'],
                                    canonical_sampling_pixels=counters['canonical_sampled_pixels'])
                            del batch
                        pre_rendered = batch_images.pop(index)
                    native_frame_cache = {}
                    if native_iterator is not None:
                        waiting_started = time.perf_counter()
                        native_index, native_plane, remap_budget = next(native_iterator)
                        runtime_telemetry().add('sam.cpu_images.native_prefetch_wait_seconds',
                            time.perf_counter()-waiting_started)
                        if native_index != int(address['native_index']):
                            raise RuntimeError('SAM native prefetch returned an unplanned frame')
                        roi = native_rois[native_index]
                        native_frame_cache.update(plane=native_plane,
                            origin_xy=(roi[1], roi[0]) if roi is not None else (0, 0),
                            max_workspace_bytes=remap_budget)
                        self._add_image_metrics(native_sampling_calls=1,
                            native_sampling_pixels=int(native_plane.size))
                        native_plane = None
                        runtime_telemetry().add('sam.cpu_images.native_prefetch_frames', 1)
                    missing = self._copy_cached_pixels(physical_target, physical_bbox,
                        int(address['native_index']), geometry_identity, cache_entries=donor_entries)
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
                            self._add_image_metrics(image_cache_reused_pixels=(cy1-cy0)*(cx1-cx0))
                            remainder.extend(_subtract_bbox(remaining, intersection))
                        missing = tuple(remainder)
                        del done
                    for cy0, cx0, cy1, cx1 in missing:
                        self._check_image_lifetime()
                        image = (pre_rendered[cy0-physical_bbox[0]:cy1-physical_bbox[0],
                                              cx0-physical_bbox[1]:cx1-physical_bbox[1]]
                            if pre_rendered is not None else
                            self._render_demand_crop(view, int(address['native_index']), affine, inverse,
                                output_height=cy1-cy0, output_width=cx1-cx0,
                                output_origin_yx=(cy0, cx0), output_canvas_width=shape[2],
                                native_frame_cache=native_frame_cache, native_preparation_bbox=physical_bbox))
                        if image.dtype != np.uint8 or image.shape != (cy1-cy0, cx1-cx0):
                            raise ValueError('SAM image provider returned a mismatched detector canvas')
                        copy_started = time.perf_counter()
                        physical_target[cy0-physical_bbox[0]:cy1-physical_bbox[0],
                                        cx0-physical_bbox[1]:cx1-physical_bbox[1]] = image
                        if gpu_renderer is None:
                            runtime_telemetry().add('sam.cpu_images.cache_write_host_seconds',
                                time.perf_counter()-copy_started)
                        self._add_image_metrics(rendered_frames=1, rendered_pixels=int(image.size))
                        del image
                    completed.append((int(address['native_index']), physical_bbox,
                                      bool(address['mirror_u']), offset))
                    del target, physical_target, native_frame_cache, pre_rendered
                flush_started = time.perf_counter()
                cache.flush()
                if gpu_renderer is None:
                    runtime_telemetry().add('sam.cpu_images.cache_flush_host_seconds',
                        time.perf_counter()-flush_started)
            except BaseException:
                from .runtime import close_memmap_array_without_flush
                target = physical_target = done = native_frame_cache = native_plane = None
                close_memmap_array_without_flush(cache, unlink_path=path)
                cache = None
                raise
            finally:
                try:
                    try:
                        if native_iterator is not None:
                            native_iterator.close()
                    finally:
                        if batch_iterator is not None:
                            batch_iterator.close()
                finally:
                    from .runtime import close_memmap_array_without_flush
                    batch_images.clear()
                    if cache is not None:
                        close_memmap_array_without_flush(cache)
                        cache = None
            # Identity describes canonical image bytes, independently of how
            # much of that immutable canvas this compact descriptor stores.
            # Exact model/crop/frame feature keys can therefore survive a
            # different scope's demand inventory without admitting new pixels.
            stat = path.stat()
            reference = LtaPhysicalViewCacheRef(path=path, shape=logical_shape, dtype='uint8',
                physical_view_id=physical_view_name(view), identity_sha256=image_identity,
                size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                frame_crops=tuple(records) if prepared_plan is not None else ())
            self._add_image_metrics(render_seconds=time.perf_counter()-render_started)
            return reference, True, dict(reference=reference, geometry_identity=geometry_identity,
                addresses=addresses, native_shape=shape)


        reference, owned, committed = None, False, False
        try:
            from .sam_interpolation import _trace_sam_phase
            operation, scope_id = getattr(self._resource_local, 'sam_phase_scope', ('', ''))
            with _trace_sam_phase('image_cache_build', scope_id, operation=operation):
                reference, owned, entry = build_reference()
                reference.revalidate()
            with self._idle:
                self._check_image_lifetime()
                from .sam_gpu_rendering import register_live_image
                register_live_image(self, reference, image_geometry_identity, sampling)
                self._claim_image_reference(reference, owned=owned)
                self._caches[key] = reference
                self._cache_transforms[key] = transform
                self._cache_entries.append(entry)
                if owned:
                    self.cache_logical_bytes += payload_bytes
                # Promised same-demand consumers pin this committed descriptor
                # until their own lookup/lease claim, including cancellation.
                self._cache_owners[str(reference.path)]['pins'] += ticket['waiters']
                ticket.update(reference=reference, done=True)
                committed = True
                self._idle.notify_all()
            return reference
        except BaseException as error:
            # Completed renderer frames can retain private mmap views through
            # their traceback. Drop those aliases before refunding the ticket
            # and exposing the failure to every same-demand waiter.
            import traceback
            traceback.clear_frames(error.__traceback__)
            private_path = (reference.path if reference is not None and owned else ticket.get('private_path'))
            if private_path is not None and not committed:
                try:
                    private_path.unlink(missing_ok=True)
                except BaseException as cleanup_error:
                    if callable(getattr(error, 'add_note', None)):
                        error.add_note(f'SAM private cache cleanup failed: {cleanup_error}')
            with self._idle:
                ticket.update(done=True, error=error)
                self._idle.notify_all()
            raise
        finally:
            with self._idle:
                for owner in ticket['donors']:
                    owner['pins'] -= 1
                self._image_build_bytes -= ticket['payload_bytes']
                self._image_build_credit_bytes -= ticket['credit_bytes']
                self._image_builds.pop(key, None)
                self._idle.notify_all()

    def _transverse_batch_iterator(self, view, shape, records, addresses, affine, inverse,
                                    geometry_identity, payload_bytes, *, cache_entries=None):
        """Batch only the existing fresh lazy-Transverse temporal-resize route."""
        from .geometry import physical_view_name
        source = self.source_volume
        if (not bool(getattr(source, '_is_lazy_processing_cube', False))
                or str(view.family) != 'orthogonal' or physical_view_name(view) != 'transverse'):
            return None
        if source.materialized:
            runtime_telemetry().add('sam.transverse_cache.skipped_materialized', 1)
            return None
        if (source.streaming_backend or tuple(source.source.shape[1:]) != tuple(source.shape[1:])
                or any(address['mirror_u'] or int(address['native_index']) != frame
                       for frame,address in addresses.items())
                or any(entry['geometry_identity'] == geometry_identity
                       for entry in (self._cache_entries if cache_entries is None else cache_entries))):
            return None  # Keep partial cached-pixel reuse and other resize paths unchanged.
        from .workspace import _env_int
        cap = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
        profile = getattr(self._resource_local, 'profile', None)
        if profile is not None:
            from .sam_resources import validate_live_sam_resource_profile
            validate_live_sam_resource_profile(profile)
            # The fixed non-CPU allowance already covers this render/cache work.
            # Reserve the complete output cache before spending remaining scratch.
            cap = min(cap, max(0, profile.non_cpu_base_allowance_bytes-int(payload_bytes)),
                max(0,profile.physical_headroom_bytes-profile.other_promised_bytes-int(payload_bytes)))
        if cap <= 0:
            return None
        from .media import wait_for_volume_ready
        wait_for_volume_ready(source.source)
        from .sam_transverse_cache_rendering import iter_transverse_crop_batches
        runtime_telemetry().add('sam.transverse_cache.eligible_scopes', 1)
        return iter_transverse_crop_batches(source.source, source.shape,
            [(frame,(y0,x0,y1,x1)) for frame,y0,x0,y1,x1,_offset in records],
            affine=affine,inverse=inverse,canvas_width=shape[2],max_workspace_bytes=cap,
            cancel_event=self._cancel)

    def _copy_cached_pixels(self, target, bbox, native_frame, geometry_identity, *, cache_entries=None):
        """Copy intersections in physical working coordinates; return holes."""
        from .sam_cyclic import mirror_bbox_yx
        missing = [bbox]
        for entry in (self._cache_entries if cache_entries is None else cache_entries):
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
                        self._add_image_metrics(image_cache_reused_pixels=(cy1-cy0)*(cx1-cx0))
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
        gpu_renderer = getattr(self._resource_local, 'gpu_image_renderer', None)
        if gpu_renderer is not None:
            image = gpu_renderer.render(view, index, inverse, output_origin_yx=output_origin_yx,
                output_height=output_height, output_width=output_width)
            runtime_telemetry().add('sam.gpu_images.frames', 1)
            runtime_telemetry().add('sam.gpu_images.pixels', int(image.size))
            return image
        from ._deps import cv2
        from .geometry import physical_view_name
        from .sam_canvas_rendering import (render_canonical_crop, cartesian_source_view,
                                           prepare_cartesian_native_crop)
        from .media import (_linear_source_index, _resize_gray_slice_nearest_or_linear,
                            wait_for_volume_ready, wait_for_volume_slice_ready)
        source = self.source_volume
        frames = None
        native_origin_xy = (0, 0)
        if native_frame_cache is not None and 'plane' in native_frame_cache:
            previous_pixels = int(native_frame_cache.get('sampled_output_pixels', 0))
            rendered = render_canonical_crop(source, view, index, affine=affine, inverse=inverse,
                output_origin_yx=output_origin_yx, output_height=output_height, output_width=output_width,
                output_canvas_width=output_canvas_width, native_frame_cache=native_frame_cache,
                max_workspace_bytes=native_frame_cache.get('max_workspace_bytes'))
            self._add_image_metrics(canonical_sampling_pixels=int(native_frame_cache['sampled_output_pixels'])-previous_pixels)
            return rendered
        lazy_unmaterialized = bool(getattr(source, '_is_lazy_processing_cube', False)) and not source.materialized
        transverse = str(view.family) == 'orthogonal' and physical_view_name(view) == 'transverse'
        if lazy_unmaterialized and not transverse:
            with self._source_materialization_lock:
                if not source.materialized:
                    started = time.perf_counter()
                    wait_for_volume_ready(source)
                    self._add_image_metrics(source_materializations=1,
                        source_materialization_seconds=time.perf_counter()-started)
        elif not lazy_unmaterialized:
            wait_for_volume_ready(source)
        if (bool(getattr(source, '_is_lazy_processing_cube', False)) and source.materialized
                and not (lazy_unmaterialized and transverse)):
            # Standard coronal/block and shell renderers require ndarray
            # strides/flags. Keep the owning lazy proxy alive in the context,
            # and lend its existing map rather than rematerializing a proxy.
            source = source._array
        if cartesian_source_view(source, view) is not None:
            native, native_origin_xy = prepare_cartesian_native_crop(source, view, index, affine=affine,
                output_bbox_yx=native_preparation_bbox or (output_origin_yx[0], output_origin_yx[1],
                    output_origin_yx[0]+output_height, output_origin_yx[1]+output_width))
        elif lazy_unmaterialized and transverse:
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
        if cartesian_source_view(source, view) is not None or lazy_unmaterialized and transverse:
            class OneNativeFrame:
                def __getitem__(self, frame_index):
                    if int(frame_index) != int(index):
                        raise RuntimeError('SAM demand renderer accessed an unplanned frame')
                    return native
            frames = OneNativeFrame()
        local_cache = native_frame_cache if native_frame_cache is not None else {}
        if str(view.family) in {'radial', 'spherical'}:
            from .sam_canvas_rendering import prepare_native_shell_crop
            from .workspace import _env_int
            native, native_origin_xy = prepare_native_shell_crop(source, view, index,
                affine=affine, output_bbox_yx=native_preparation_bbox or (
                    output_origin_yx[0], output_origin_yx[1],
                    output_origin_yx[0]+output_height, output_origin_yx[1]+output_width),
                max_workspace_bytes=max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2)))
            local_cache.update(plane=native, origin_xy=native_origin_xy)
        rendered = render_canonical_crop(source, view, index, affine=affine, inverse=inverse,
            output_origin_yx=output_origin_yx, output_height=output_height, output_width=output_width,
            view_frames=frames, native_origin_xy=native_origin_xy, output_canvas_width=output_canvas_width,
            native_frame_cache=local_cache)
        self._add_image_metrics(native_sampling_calls=1,
            native_sampling_pixels=int(local_cache['plane'].size),
            canonical_sampling_pixels=int(local_cache['sampled_output_pixels']))
        return rendered

    def _next_gpu_image_target(self):
        if not self._gpu_image_waiters:
            return None
        owner = self._gpu_image_waiters[0]
        if owner._gpu_wait_deadline is None:
            owner._gpu_wait_deadline = time.monotonic()+30.
        rejected = owner._gpu_image_rejected_devices
        devices = self._image_device_ids()
        for _ in devices:
            device = devices[self._gpu_image_cursor % len(devices)]
            self._gpu_image_cursor = (self._gpu_image_cursor+1) % len(devices)
            if device not in rejected:
                return device
        return None

    def _image_device_ids(self):
        if self.progressive_startup:
            return tuple(device for device in (self._progressive_pool.device_ids
                if self._progressive_pool is not None else ()) if device in self._resident_leases)
        return tuple(int(token.split(':')[-1]) for token in self.device_ids)

    def _queue_gpu_image(self, renderer):
        self._check_image_lifetime()
        with self._gpu_lease_lock:
            renderer._gpu_image_rejected_devices = set()
            renderer._gpu_wait_deadline = None
            self._gpu_image_waiters.append(renderer)
            if self._gpu_image_target is None:
                self._gpu_image_target = self._next_gpu_image_target()

    def _sam_sdk_ready(self):
        # Never take the scheduler condition while holding the GPU lock.
        ready = getattr(self._runtime, 'has_ready_work', None)
        return bool(ready()) if callable(ready) else False

    def _can_extend_gpu_image_burst(self):
        return (callable(getattr(self._runtime, 'idle_image_handoff', None))
                and not self._sam_sdk_ready())

    def _try_gpu_image_lease(self, renderer, torch):
        sdk_ready = self._sam_sdk_ready()
        with self._gpu_lease_lock:
            self._check_image_lifetime()
            if renderer.lease is not None:
                return renderer.lease
            if not self._gpu_image_waiters or self._gpu_image_waiters[0] is not renderer:
                return None
            target = self._gpu_image_target
            if target is None:
                return None
            # Drain only the head's target, but use another already-idle GPU
            # without waiting for that drain. Memory refusals rotate the head.
            devices = [target]+[device for device in self._image_device_ids() if device != target]
            for device in devices:
                if (device in renderer._gpu_image_rejected_devices
                        or sdk_ready and device in self._gpu_image_sdk_owed):
                    continue
                runtime_telemetry().add('sam.gpu_images.admission_attempts', 1)
                resident = self._resident_leases.get(device)
                if resident is not None:
                    lease = resident.try_acquire_compute(torch, 'SAM image preparation')
                else:
                    from .backprojection import _try_acquire_specific_main_process_gpu_stage
                    lease = _try_acquire_specific_main_process_gpu_stage(torch, device, 'SAM image preparation')
                if lease is not None:
                    renderer.lease, renderer.device_index = lease, device
                    self._gpu_image_target = device
                    return lease
            return None

    def _reject_gpu_image_device(self, renderer, released_lease):
        """A released, proven-insufficient GPU cannot block the same FIFO head."""
        with self._gpu_lease_lock:
            if (renderer.lease is not released_lease or renderer.engine is not None
                    or not self._gpu_image_waiters
                    or self._gpu_image_waiters[0] is not renderer):
                raise RuntimeError('SAM image memory rejection requires its exact released FIFO-head lease')
            device = int(released_lease.device_index)
            # Keep the released token attached until this atomic commit, so a
            # source handoff cannot replace it while its rejection is pending.
            renderer.lease = None
            if self._gpu_image_owners.get(device) is renderer:
                self._gpu_image_owners.pop(device)
            renderer._gpu_image_rejected_devices.add(device)
            self._gpu_image_target = self._next_gpu_image_target()
            return self._gpu_image_target is not None

    def _finish_gpu_image_wait(self, renderer, *, granted=False):
        with self._gpu_lease_lock:
            if renderer in self._gpu_image_waiters:
                head = self._gpu_image_waiters[0] is renderer
                self._gpu_image_waiters.remove(renderer)
                if head:
                    self._gpu_image_target = self._next_gpu_image_target()
            if granted:
                device = int(renderer.lease.device_index)
                self._gpu_image_owners[device] = renderer
                self._gpu_image_sdk_owed.add(device)

    def _finish_gpu_image(self, renderer):
        with self._gpu_lease_lock:
            for device, owner in tuple(self._gpu_image_owners.items()):
                if owner is renderer:
                    self._gpu_image_owners.pop(device)

    def _try_gpu_image_handoff(self, old_renderer, transfer_callback):
        guard = getattr(self._runtime, 'idle_image_handoff', None)
        # Scheduler state precedes the GPU lock; enqueue cannot race this grant.
        with (guard() if callable(guard) else nullcontext(False)) as extend:
            with self._gpu_lease_lock:
                if self._cancel.is_set() or self._closed or not self._gpu_image_waiters:
                    return False
                device = int(old_renderer.lease.device_index)
                if (self._gpu_image_owners.get(device) is not old_renderer
                        or old_renderer._image_burst_count >= 2 and not extend):
                    return False
                owner = self._gpu_image_waiters[0]
                if not transfer_callback(owner):
                    return False
                self._gpu_image_waiters.popleft()
                self._gpu_image_owners[device] = owner
                # Ready SDK work keeps its owed turn after the bounded image pair.
                self._gpu_image_sdk_owed.add(device)
                self._gpu_image_target = self._next_gpu_image_target()
                return True

    def _sam_compute_should_yield(self, device):
        with self._gpu_lease_lock:
            return (not self._cancel.is_set() and bool(self._gpu_image_waiters)
                    and self._gpu_image_target == int(device)
                    and int(device) not in self._gpu_image_sdk_owed)

    def _try_sam_compute_lease(self, device_index, purpose):
        import torch
        with self._gpu_lease_lock:
            resident = self._resident_leases.get(int(device_index))
            if resident is None or int(device_index) in self._active_compute:
                return None
            lease = resident.try_acquire_compute(torch, purpose)
            if lease is not None:
                self._active_compute[int(device_index)] = lease
                self._gpu_image_sdk_owed.discard(int(device_index))
            return lease

    def _release_sam_compute_lease(self, lease):
        with self._gpu_lease_lock:
            device = int(lease.device_index)
            if self._active_compute.get(device) is lease:
                lease.release()
                self._active_compute.pop(device)

    def _quarantine_sam_residency(self, reason):
        with self._gpu_lease_lock:
            for resident in self._resident_leases.values():
                resident.quarantine(reason)

    def _before_sam_worker_shutdown(self):
        if self.progressive_startup:
            self._cancel.set()
            self._join_progressive_startup()
        import torch
        self._quarantine_sam_residency('SAM predictor shutdown or failed worker settlement')
        deadline = time.monotonic() + 30.
        for device, resident in tuple(self._resident_leases.items()):
            while True:
                with self._gpu_lease_lock:
                    if device in self._active_compute or device in self._shutdown_compute:
                        break
                    lease = resident.try_acquire_compute(torch, 'SAM predictor shutdown')
                    if lease is not None:
                        self._shutdown_compute[device] = lease
                        break
                if time.monotonic() >= deadline:
                    raise RuntimeError('SAM predictor shutdown could not fence a borrowed GPU stage; residency retained')
                threading.Event().wait(0.05)

    def _after_sam_worker_shutdown(self):
        with self._gpu_lease_lock:
            for lease in tuple(self._active_compute.values()):
                lease.release()
            self._active_compute.clear()
            for lease in tuple(self._shutdown_compute.values()):
                lease.release()
            self._shutdown_compute.clear()

    def _start(self):
        if self.progressive_startup:
            self._start_progressive()
            return
        with self._runtime_lock:
            self._start_admitted()

    def prepare_runtime(self, parent_pool):
        """Warm the existing predictor pool before admitting deferred parents."""
        if parent_pool is None:
            raise TypeError('SAM runtime preparation requires the parent admission pool')
        with self._idle:
            if self._closed or self._cancel.is_set():
                raise RuntimeError(self._failure or 'SAM interpolation runtime is closed')
            self._active_passes += 1
        try:
            with self._runtime_lock:
                if self._startup_pool is not None and self._startup_pool is not parent_pool:
                    raise RuntimeError('SAM runtime preparation cannot change its parent admission pool')
                profile = getattr(self._resource_local, 'profile', None)
                if profile is not None:
                    profile._validate_owner()
                    if profile._lease.pool is not parent_pool:
                        raise RuntimeError('SAM startup pool differs from the live parent profile')
                self._startup_pool = parent_pool
                with parent_pool.condition:
                    self._update_startup_debt_locked()
                if not self.progressive_startup:
                    self._start_admitted()
            if self.progressive_startup:
                self._start_progressive()
        finally:
            with self._idle:
                self._active_passes -= 1
                self._idle.notify_all()

    def _startup_host_headroom(self):
        from .sam_resources import physical_sam_headroom, sam_parent_promised_bytes
        profile = getattr(self._resource_local, 'profile', None)
        pool = self._startup_pool
        if profile is not None:
            profile._validate_owner()
            if pool is not None and profile._lease.pool is not pool:
                raise RuntimeError('SAM startup pool differs from the live parent profile')
            pool = profile._lease.pool
        if pool is None:
            return max(0, int(physical_sam_headroom())), 0
        # Parents can have been admitted before these duplicated models load.
        # Physical free RAM already subtracts resident bytes; all outstanding
        # pool promises remain unavailable for model startup and its reserve.
        with pool.condition:
            if profile is not None:
                profile._validate_owner()
            return (max(0, int(physical_sam_headroom())),
                    sam_parent_promised_bytes(pool))

    def configure_startup_parent_pool(self, parent_pool):
        """Bind the shared admission condition before checkpoints open RAM births."""
        if parent_pool is None:
            raise TypeError('SAM startup requires its parent admission pool')
        with self._runtime_lock:
            if self._startup_pool is not None and self._startup_pool is not parent_pool:
                raise RuntimeError('SAM startup cannot change parent admission ownership')
            self._startup_pool = parent_pool
            with parent_pool.condition:
                self._update_startup_debt_locked()

    def _update_startup_debt_locked(self):
        if self._startup_pool is not None:
            self._startup_pool._sam_startup_future_bytes = (
                sum(self._startup_future_peaks.values()) if self._startup_fleet_funded else 0)
            self._startup_pool._sam_startup_active_bytes = (
                self._startup_fleet_credit_bytes if self._startup_fleet_funded else self._startup_host_reserved)

    def _sync_startup_fleet_credit_locked(self):
        if self._startup_fleet_funded:
            remaining = sum(self._startup_future_peaks.values())
            if remaining > self._startup_fleet_credit_bytes:
                raise RuntimeError('SAM startup cannot create an unfunded model promise')
            refund = self._startup_fleet_credit_bytes-remaining
            if self._startup_pool is not None:
                self._startup_pool.in_use -= refund
            self._startup_fleet_credit_bytes = remaining
        self._update_startup_debt_locked()

    def _publish_startup_progress(self, device, stage, snapshot=None, error=None):
        # No context/lifecycle lock is acquired while holding the admission lock.
        key = 'fleet' if device is None else str(device)
        with self._startup_diagnostics_lock:
            previous = self._startup_progress.get(key, {})
            record = dict(previous, stage=stage, device_index=device, monotonic=time.monotonic())
            if snapshot is not None:
                record.update(snapshot)
            if error is not None:
                record['error'] = str(error)
            elif stage != 'failed':
                record.pop('error', None)
            self._startup_progress[key] = record
            publish = (previous.get('stage') != stage
                or record['monotonic']-previous.get('published_monotonic', 0) >= 1.)
            record['published_monotonic'] = (record['monotonic'] if publish
                else previous.get('published_monotonic', 0))
            progress = {name:dict(value) for name,value in self._startup_progress.items()}
        if publish:
            try:
                runtime_telemetry().gauge('sam.startup_progress', progress)
            except Exception:
                pass

    def _join_progressive_startup(self):
        current = threading.current_thread()
        with self._runtime_lock:
            threads = tuple(self._progressive_threads.values())
        for thread in threads:
            if thread is current:
                continue
            thread.join(timeout=2.)
            if thread.is_alive():
                _retain_unsettled_context(self)
                raise RuntimeError('SAM cohort startup is unsettled; model and host grants retained')

    def _launch_progressive_devices(self):
        if self._cancel.is_set() or self._closed or not self._startup_fleet_funded:
            return
        for device in sorted(self._retired_devices):
            if device in self._progressive_threads:
                continue
            thread = threading.Thread(target=self._boot_progressive_device, args=(device,),
                name=f'sam-startup-cuda-{device}', daemon=True)
            self._progressive_threads[device] = thread
            self._progressive_pool.begin_boot(device)
            try:
                thread.start()
            except BaseException:
                self._progressive_pool.finish_boot(device)
                del self._progressive_threads[device]
                raise

    def _start_progressive(self):
        self.check_startup()
        with self._runtime_lock:
            if self._progressive_pool is None:
                from .sam_device_pool import SamDeviceWorkerPools
                from .sam_tracker_runtime import SamInterpolationTracker
                profile = getattr(self._resource_local, 'profile', None)
                if self._startup_pool is None and profile is not None:
                    profile._validate_owner()
                    self._startup_pool = profile._lease.pool
                    with self._startup_pool.condition:
                        self._update_startup_debt_locked()
                self._progressive_pool = SamDeviceWorkerPools(
                    tuple(int(device.split(':')[-1]) for device in self.device_ids), cancel_event=self._cancel)
                self._starting_runtime = SamInterpolationTracker(model_path=self.model_path,
                    device_ids=self._progressive_pool.configured_device_ids,
                    artifact_root=self.temp_dir / 'sam_runtime', feature_cache_bytes=self.feature_cache_mib*1024**2,
                    workers_per_device=self.sessions_per_gpu, progressive_pool=self._progressive_pool,
                    compute_lease_factory=self._try_sam_compute_lease,
                    compute_lease_release=self._release_sam_compute_lease,
                    compute_yield_requested=self._sam_compute_should_yield,
                    residency_quarantine=self._quarantine_sam_residency,
                    before_worker_shutdown=self._before_sam_worker_shutdown,
                    after_worker_shutdown=self._after_sam_worker_shutdown)
        self._fund_progressive_fleet()
        with self._runtime_lock:
            self._launch_progressive_devices()
        started = time.perf_counter()
        while not self._runtime_admitted.wait(.05):
            self.check_startup()
        self.wait_seconds += time.perf_counter()-started
        self.check_startup()

    def _progressive_future_host_bytes(self):
        from .sam_resources import sam_image_staging_snapshot
        pool = self._startup_pool
        if pool is None:
            parent, staging = 0, 0
        else:
            charged = self._startup_fleet_credit_bytes if self._startup_fleet_funded else self._startup_host_reserved
            parent = max(int(pool.capacity),
                max(0, int(pool.in_use)-charged), int(getattr(pool, 'oversize_requested_bytes', 0)))
            staging = int(sam_image_staging_snapshot(pool)['image_staging_capacity_bytes'])
        pending = (0 if self._startup_pending_host_bytes is None else
                   max(0, int(self._startup_pending_host_bytes())))
        source = self.source_volume
        if bool(getattr(source, '_is_lazy_processing_cube', False)) and not source.materialized:
            pending += int(np.prod(source.shape, dtype=np.int64))
        return parent, staging, pending

    def startup_budget_snapshot_locked(self, *, additional_pending_bytes=0):
        """Caller holds the shared condition; resident proof precedes fresh RAM."""
        from .sam_resources import GIB, physical_sam_headroom
        parent, staging, pending = self._progressive_future_host_bytes()
        remaining = sum(self._startup_future_peaks.values())
        pending += max(0, int(additional_pending_bytes))
        required = parent+staging+pending+remaining+2*GIB
        return dict(physical_headroom_bytes=max(0,int(physical_sam_headroom())),
            required_host_bytes=required, protected_parent_bytes=parent,
            protected_image_bytes=staging, pending_host_bytes=pending,
            remaining_startup_bytes=remaining, active_startup_grants_bytes=self._startup_host_reserved,
            owned_startup_credit_bytes=self._startup_fleet_credit_bytes,
            startup_fleet_funded=self._startup_fleet_funded,
            mandatory_reserve_bytes=2*GIB)

    def _progressive_cohort_budget_snapshot_locked(self):
        from .sam_resources import GIB, physical_sam_headroom
        parent, staging, pending = self._progressive_future_host_bytes()
        # The initial grant owns C even after its parents allocate. New dense
        # debt still pays the full envelope at birth; only expanded rights are new.
        parent = max(0,parent-self._startup_parent_envelope_bytes)
        remaining = sum(self._startup_future_peaks.values())
        return dict(physical_headroom_bytes=max(0,int(physical_sam_headroom())),
            required_host_bytes=parent+staging+pending+remaining+2*GIB,
            protected_parent_bytes=parent, protected_image_bytes=staging,
            pending_host_bytes=pending, remaining_startup_bytes=remaining,
            active_startup_grants_bytes=self._startup_host_reserved,
            owned_startup_credit_bytes=self._startup_fleet_credit_bytes,
            owned_parent_envelope_bytes=self._startup_parent_envelope_bytes,
            startup_fleet_funded=self._startup_fleet_funded,
            mandatory_reserve_bytes=2*GIB)

    def _fund_progressive_fleet(self):
        from .sam_resources import GIB, sam_image_staging_snapshot
        pool = self._startup_pool
        condition = self._startup_host_condition if pool is None else pool.condition
        with self._startup_funding_lock:
            if self._startup_fleet_funded:
                return
            while not self._retired_devices:
                self.check_startup()
                self._publish_startup_progress(None, 'waiting_detector_retirement')
                self._cancel.wait(.05)
            deadline = time.monotonic()+self._startup_wait_timeout
            with condition:
                while True:
                    self.check_startup()
                    snapshot = self.startup_budget_snapshot_locked()
                    host, required = snapshot['physical_headroom_bytes'], snapshot['required_host_bytes']
                    if host >= required:
                        self._startup_fleet_credit_bytes = sum(self._startup_future_peaks.values())
                        self._startup_parent_envelope_bytes = snapshot['protected_parent_bytes']
                        if pool is not None:
                            pool.in_use += self._startup_fleet_credit_bytes
                        self._startup_fleet_funded = True
                        self._update_startup_debt_locked()
                        self._publish_startup_progress(None, 'funded', dict(snapshot,
                            owned_startup_credit_bytes=self._startup_fleet_credit_bytes,
                            owned_parent_envelope_bytes=self._startup_parent_envelope_bytes,
                            startup_fleet_funded=True))
                        for device in self._startup_future_peaks:
                            if device not in self._retired_devices:
                                self._publish_startup_progress(device, 'waiting_detector_retirement')
                        return
                    parents = 0 if pool is None else int(pool.in_use)
                    staging = 0 if pool is None else sam_image_staging_snapshot(pool)['image_staging_in_use_bytes']
                    cold = not parents and not staging and not snapshot['pending_host_bytes']
                    if cold:
                        checkpoint = int(Path(self._progressive_bundle.checkpoint_path).stat().st_size)
                        minimum = 2*checkpoint+2*GIB
                        minimum_required = required-snapshot['remaining_startup_bytes']+minimum*len(self._startup_future_peaks)
                        if self.sessions_per_gpu == 2 and host >= minimum_required:
                            self._startup_future_peaks = {index:minimum for index in self._startup_future_peaks}
                            self._startup_host_plan = dict(status='minimum_single_fleet',
                                physical_headroom_bytes=host, minimum_required_host_bytes=minimum_required,
                                planned_device_ids=list(self._startup_future_peaks), sessions_per_device=1)
                            self._update_startup_debt_locked()
                            continue
                        error = SamConcurrentStartupResourceError('SAM cold host budget cannot admit the configured minimum fleet')
                        self._publish_startup_progress(None, 'failed', snapshot, error)
                        raise error
                    self._publish_startup_progress(None, 'waiting_host_funding', snapshot)
                    if time.monotonic() >= deadline:
                        error = SamConcurrentStartupResourceError('SAM whole-fleet host funding timed out')
                        self._publish_startup_progress(None, 'failed', snapshot, error)
                        raise error
                    condition.wait(.05)

    @contextmanager
    def dense_restore_allocation_guard(self):
        if not self.progressive_startup:
            yield True
            return
        if self._startup_pool is None:
            raise RuntimeError('SAM restore admission has no configured parent condition')
        with self._startup_pool.condition:
            self.check_startup()
            snapshot = self.startup_budget_snapshot_locked()
            yield snapshot['physical_headroom_bytes'] >= snapshot['required_host_bytes']

    def _reserve_progressive_host(self, device, peak, attempt):
        pool = self._startup_pool
        condition = self._startup_host_condition if pool is None else pool.condition
        deadline = time.monotonic()+self._startup_wait_timeout
        with condition:
            while True:
                self.check_startup()
                if not self._startup_fleet_funded:
                    raise RuntimeError('SAM cohort has no owned fleet host grant')
                snapshot = self._progressive_cohort_budget_snapshot_locked()
                host, required = snapshot['physical_headroom_bytes'], snapshot['required_host_bytes']
                attempt.update(host_headroom_before_bytes=host, minimum_host_startup_bytes=required,
                    future_parent_capacity_bytes=snapshot['protected_parent_bytes'],
                    future_image_staging_bytes=snapshot['protected_image_bytes'],
                    future_external_host_bytes=snapshot['pending_host_bytes'],
                    remaining_startup_bytes=snapshot['remaining_startup_bytes'],
                    prior_startup_grants_bytes=self._startup_host_reserved,
                    owned_parent_envelope_bytes=snapshot['owned_parent_envelope_bytes'],
                    owned_startup_credit_bytes=snapshot['owned_startup_credit_bytes'],
                    host_startup_basis='preowned_fleet_and_parent_envelope_plus_new_debt_and_reserve',
                    startup_host_grant_bytes=peak)
                if peak > self._startup_future_peaks.get(device, peak):
                    raise SamConcurrentStartupResourceError('SAM host budget planned a single-session cohort')
                if host >= required:
                    self._startup_host_reserved += peak
                    self._startup_host_grants[device] = peak
                    self._update_startup_debt_locked()
                    self._publish_startup_progress(device, 'host_grant_claimed', snapshot)
                    return
                self._publish_startup_progress(device, 'waiting_host_headroom', snapshot)
                if time.monotonic() >= deadline:
                    error = SamConcurrentStartupResourceError(f'SAM cohort host startup admission timed out on cuda:{device}')
                    self._publish_startup_progress(device, 'failed', snapshot, error)
                    raise error
                condition.wait(.05)

    def _release_progressive_host(self, device):
        pool = self._startup_pool
        condition = self._startup_host_condition if pool is None else pool.condition
        with condition:
            amount = self._startup_host_grants.pop(device, 0)
            self._startup_host_reserved -= amount
            if pool is not None and not self._startup_fleet_funded:
                pool.in_use -= amount
            self._update_startup_debt_locked()
            condition.notify_all()

    def _check_progressive_host(self, device, attempt):
        pool = self._startup_pool
        condition = self._startup_host_condition if pool is None else pool.condition
        with condition:
            snapshot = self._progressive_cohort_budget_snapshot_locked()
            host, required = snapshot['physical_headroom_bytes'], snapshot['required_host_bytes']
            required -= self._startup_future_peaks.get(device, 0)
            attempt.update(host_headroom_after_bytes=host, minimum_host_after_bytes=required)
            if host < required:
                raise SamConcurrentStartupResourceError('SAM cohort leaves insufficient future parent/startup headroom')
            self._publish_startup_progress(device, 'host_after_load_proven', dict(snapshot,
                required_host_bytes=required,
                completed_model_peak_bytes=self._startup_future_peaks.get(device, 0)))

    def _boot_progressive_device(self, device):
        from .sam_resources import GIB
        from .lta_sam import resolve_local_sam_bundle
        from .lta_workers import LtaWorkerPool
        from .backprojection import _try_acquire_specific_main_process_gpu_stage
        import torch
        started, lease, pool, admitted = time.perf_counter(), None, None, False
        runtime = self._runtime or self._starting_runtime
        attempt = {}
        try:
            if device >= int(torch.cuda.device_count()):
                raise RuntimeError(f'SAM device cuda:{device} is unavailable')
            bundle = self._progressive_bundle
            checkpoint = int(Path(bundle.checkpoint_path).stat().st_size)
            sessions = (1 if self._startup_host_plan is not None else self.sessions_per_gpu)
            while True:
                attempt = dict(device_index=device, sessions_per_gpu=sessions,
                    checkpoint_identity_sha256=bundle.checkpoint_identity_sha256,
                    checkpoint_bytes=checkpoint, started_monotonic=time.monotonic())
                try:
                    deadline = time.monotonic()+self._startup_wait_timeout
                    while lease is None:
                        self._reserve_progressive_host(device, 2*checkpoint*sessions+2*GIB, attempt)
                        lease = _try_acquire_specific_main_process_gpu_stage(torch, device,
                            'TTA persistent SAM interpolation predictor')
                        if lease is None:
                            self._release_progressive_host(device)
                            self._publish_startup_progress(device, 'waiting_gpu_ownership')
                            if time.monotonic() >= deadline:
                                raise RuntimeError(f'SAM cohort GPU admission timed out on cuda:{device}')
                            self._cancel.wait(.05)
                    with self._gpu_lease_lock:
                        self._leases.append(lease)
                    free, total = map(int, torch.cuda.mem_get_info(device))
                    headroom = max(2*GIB, int(total*.15))
                    fraction = (max(0, free-headroom)//2)/total if sessions == 2 and total > 0 else None
                    if sessions == 2 and (fraction is None or fraction <= 0):
                        raise SamConcurrentStartupResourceError('SAM cohort cannot preserve mandatory GPU headroom')
                    attempt['devices_before'] = [dict(device_index=device, free_bytes=free,
                        total_bytes=total, headroom_bytes=headroom)]
                    fractions = None if fraction is None else {str(device):fraction}
                    self._publish_startup_progress(device, 'loading_model')
                    try:
                        pool = LtaWorkerPool((device,), runtime.worker_init(cuda_allocator_fractions=fractions),
                            workers_per_device=sessions, startup_timeout=runtime.startup_timeout, cancel_event=self._cancel)
                    except BaseException as error:
                        pool = getattr(error, 'unsettled_worker_pool', None)
                        if pool is not None:
                            self._progressive_pool.retain(pool)
                        raise
                    self._progressive_pool.retain(pool)
                    self.check_startup()
                    free, total = map(int, torch.cuda.mem_get_info(device))
                    attempt['devices_after'] = [dict(device_index=device, free_bytes=free, total_bytes=total)]
                    if free < max(2*GIB, int(total*.15)):
                        raise SamConcurrentStartupResourceError('SAM cohort model load violated mandatory GPU headroom')
                    self._check_progressive_host(device, attempt)

                    def activate():
                        self.check_startup()
                        resident = lease.promote_residency()
                        with self._gpu_lease_lock:
                            self._resident_leases[device] = resident
                            self._leases.remove(lease)

                    runtime.register_device_pool(device, pool, cuda_allocator_fraction=fraction,
                        admission_callback=activate)
                    admitted = True
                    with self._runtime_lock:
                        runtime.start()
                        self._runtime = runtime
                    attempt.update(status='admitted', workers=tuple(receipt for receipt in runtime.startup_receipts
                        if receipt['execution_device_id'] == device),
                        elapsed_seconds=time.perf_counter()-started)
                    with self._idle:
                        self.startup_admission['attempts'].append(attempt)
                        self.startup_admission.setdefault('effective_sessions_per_device', {})[str(device)] = sessions
                        counts = self.startup_admission['effective_sessions_per_device']
                        complete = len(counts) == len(self.device_ids)
                        self.startup_admission.update(complete=complete,
                            admitted_device_ids=list(self._progressive_pool.device_ids),
                            effective_sessions_per_gpu=(next(iter(counts.values())) if complete
                                and len(set(counts.values())) == 1 else None))
                    condition = self._startup_host_condition if self._startup_pool is None else self._startup_pool.condition
                    with condition:
                        self._startup_future_peaks.pop(device, None)
                        self._sync_startup_fleet_credit_locked()
                    self._release_progressive_host(device)
                    with self._runtime_lock:
                        if not self._runtime_admitted.is_set():
                            self.start_seconds += time.perf_counter()-started
                        self._runtime_admitted.set()
                    self._publish_startup_progress(device, 'admitted')
                    break
                except BaseException as error:
                    if admitted:
                        raise
                    if pool is not None:
                        try:
                            pool.shutdown(timeout=1., force=True)
                        except BaseException as cleanup_error:
                            if callable(getattr(error, 'add_note', None)):
                                error.add_note(f'SAM cohort startup cleanup failed: {cleanup_error}')
                        if not pool.workers_settled:
                            raise RuntimeError('SAM failed cohort retains unproven child ownership') from error
                    self._release_progressive_host(device)
                    attempt.update(status='failed', error=str(error))
                    with self._idle:
                        self.startup_admission['attempts'].append(attempt)
                    if sessions != 2 or admitted or self._cancel.is_set() or not _concurrent_startup_resource_failure(error):
                        raise
                    with self._gpu_lease_lock:
                        if lease in self._leases:
                            lease.release()
                            self._leases.remove(lease)
                    lease, pool, sessions = None, None, 1
                    condition = self._startup_host_condition if self._startup_pool is None else self._startup_pool.condition
                    with condition:
                        self._startup_future_peaks[device] = 2*checkpoint+2*GIB
                        self._sync_startup_fleet_credit_locked()
        except BaseException as error:
            self._publish_startup_progress(device, 'failed', error=error)
            with self._idle:
                self._progressive_error = error
            self.cancel(f'SAM progressive startup failed on cuda:{device}: {error}')
            settled = pool is None or bool(pool.workers_settled)
            if settled and not admitted:
                with self._gpu_lease_lock:
                    resident = self._resident_leases.pop(device, None)
                    if resident is not None:
                        resident.release(residency_settled=True)
                    if lease in self._leases:
                        lease.release()
                        self._leases.remove(lease)
                self._release_progressive_host(device)
            else:
                _retain_unsettled_context(self)
        finally:
            self._progressive_pool.finish_boot(device)
            try:
                with self._idle:
                    snapshot = json.loads(json.dumps(self.startup_admission))
                if self._startup_host_plan is not None:
                    snapshot['host_plan'] = self._startup_host_plan
                runtime_telemetry().gauge('sam.startup_admission', snapshot)
            except Exception:
                pass

    def _concurrent_startup_budget(self, torch, attempt):
        from .sam_resources import GIB
        from .lta_sam import resolve_local_sam_bundle
        bundle = resolve_local_sam_bundle(self.model_path)
        checkpoint_bytes = int(Path(bundle.checkpoint_path).stat().st_size)
        host, promised = self._startup_host_headroom()
        # Conservative startup eligibility, not a measured peak: mmap pages
        # can be shared/reclaimed, and loader/library overhead is variable.
        # Existing full histories retain their separate parent-wave admission.
        required_host = 2*checkpoint_bytes*self.worker_count+2*GIB+promised
        attempt.update(host_headroom_before_bytes=host,
            minimum_host_startup_bytes=required_host, devices_before=[],
            promised_host_before_bytes=promised,
            host_startup_basis='conservative_two_checkpoint_copies_per_worker_plus_reserve_and_parent_promises',
            checkpoint_bytes=checkpoint_bytes,
            checkpoint_identity_sha256=bundle.checkpoint_identity_sha256)
        if host < required_host:
            raise SamConcurrentStartupResourceError(
                f'Two SAM sessions per GPU require {required_host} bytes of physical host startup '
                f'headroom; measured {host}')
        fractions = {}
        for device in self.device_ids:
            index = int(device.split(':')[-1])
            free, total = map(int, torch.cuda.mem_get_info(index))
            headroom = max(2*GIB, int(total*.15))
            quota = max(0, free-headroom)//2
            attempt['devices_before'].append(dict(device_index=index, free_bytes=free,
                total_bytes=total, headroom_bytes=headroom, allocator_bytes_per_worker=quota))
            if total <= 0 or quota <= 0:
                raise SamConcurrentStartupResourceError(
                    f'Two SAM sessions cannot retain mandatory CUDA headroom on {device}')
            fractions[str(index)] = quota/total
        return fractions

    def _check_concurrent_startup(self, torch, attempt):
        from .sam_resources import GIB
        # Probe after the complete worker readiness inventory, when every
        # duplicated model is resident. Early worker snapshots can overclaim.
        attempt['devices_after'] = []
        for device in self.device_ids:
            index = int(device.split(':')[-1])
            free, total = map(int, torch.cuda.mem_get_info(index))
            headroom = max(2*GIB, int(total*.15))
            attempt['devices_after'].append(dict(device_index=index, free_bytes=free,
                total_bytes=total, headroom_bytes=headroom))
            if free < headroom:
                raise SamConcurrentStartupResourceError(
                    f'Two SAM predictor contexts on {device} leave {free} bytes free; '
                    f'mandatory CUDA headroom is {headroom}')
        host, promised = self._startup_host_headroom()
        attempt['host_headroom_after_bytes'] = host
        attempt['promised_host_after_bytes'] = promised
        if host < 2*GIB+promised:
            raise SamConcurrentStartupResourceError(
                f'Two SAM predictor contexts leave {host} bytes of physical host headroom; '
                f'minimum reserve including admitted parent work is {2*GIB+promised}')

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
            while True:
                attempt = dict(sessions_per_gpu=self.sessions_per_gpu)
                self.startup_admission['attempts'].append(attempt)
                try:
                    fractions = (self._concurrent_startup_budget(torch, attempt)
                        if self.sessions_per_gpu == 2 else None)
                    runtime = SamInterpolationTracker(
                        model_path=self.model_path, device_ids=tuple(int(value.split(':')[-1]) for value in self.device_ids),
                        artifact_root=self.temp_dir / 'sam_runtime',
                        feature_cache_bytes=self.feature_cache_mib*1024**2,
                        workers_per_device=self.sessions_per_gpu,
                        cuda_allocator_fractions=fractions,
                        compute_lease_factory=self._try_sam_compute_lease,
                        compute_lease_release=self._release_sam_compute_lease,
                        compute_yield_requested=self._sam_compute_should_yield,
                        residency_quarantine=self._quarantine_sam_residency,
                        before_worker_shutdown=self._before_sam_worker_shutdown,
                        after_worker_shutdown=self._after_sam_worker_shutdown)
                    self._starting_runtime = runtime
                    if self._cancel.is_set():
                        cancel = getattr(runtime, 'cancel', None)
                        if callable(cancel):
                            cancel(self._failure)
                        raise RuntimeError(self._failure)
                    runtime.start()
                    receipts = getattr(runtime, 'startup_receipts', ())
                    if isinstance(receipts, (tuple, list)):
                        attempt['workers'] = receipts
                    if self._cancel.is_set():
                        raise RuntimeError(self._failure)
                    if self.sessions_per_gpu == 2:
                        self._check_concurrent_startup(torch, attempt)
                    attempt['status'] = 'admitted'
                    break
                except BaseException as error:
                    attempt.update(status='failed', error=str(error))
                    if (self.sessions_per_gpu != 2 or self._cancel.is_set()
                            or not _concurrent_startup_resource_failure(error)):
                        raise
                    if runtime is not None:
                        try:
                            runtime.close()
                        except BaseException as cleanup_error:
                            error.add_note(f'SAM two-session startup cleanup: {cleanup_error}')
                        if getattr(runtime, 'residency_released', False) is not True:
                            raise
                    if self._cancel.is_set():
                        raise RuntimeError(self._failure)
                    # The physical startup fence remains held throughout the
                    # fully settled retry. No work/history/crop was submitted.
                    runtime = self._starting_runtime = None
                    self.sessions_per_gpu = 1
                    self.startup_admission.update(effective_sessions_per_gpu=1,
                        fallback_reason=str(error))
            if getattr(runtime, 'startup_cuda_quiescent', False) is True:
                with self._gpu_lease_lock:
                    for lease in self._leases:
                        if self._cancel.is_set():
                            raise RuntimeError(self._failure)
                        resident = lease.promote_residency()
                        self._resident_leases[int(resident.device_index)] = resident
                        if self._cancel.is_set():
                            resident.quarantine(self._failure)
                            raise RuntimeError(self._failure)
                    self._leases.clear()
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
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
                with self._gpu_lease_lock:
                    for resident in self._resident_leases.values():
                        resident.release(residency_settled=True)
                    self._resident_leases.clear()
            else:
                self._runtime = runtime
                _retain_unsettled_context(self)
                self.cancel('SAM startup failed with unsettled worker residency')
                raise RuntimeError('SAM startup failed; GPU ownership retained until worker cleanup is proven') from startup_error
            raise
        finally:
            try:
                runtime_telemetry().gauge('sam.startup_admission', self.startup_admission)
            except Exception:
                pass  # Diagnostics must not replace startup or cleanup results.

    def interpolate(self, observation_volume, *, view, scope, **kwargs):
        from .sam_interpolation import (interpolate_sam_view_volume_pass,
                                        prepare_sam_interpolation_pass, _trace_sam_phase)
        with self._lock:
            if self._closed:
                raise RuntimeError('SAM interpolation runtime is closed')
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
            self._active_passes += 1
        previous_phase_scope = getattr(self._resource_local, 'sam_phase_scope', ('', ''))
        self._resource_local.sam_phase_scope = ('interpolation', str(scope))
        prepared = None
        execution_started = False
        preparation_phase = 'geometry_or_resource_admission'
        shape, scope_metadata = None, {}
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
            preparation_phase = 'planning'
            with _trace_sam_phase('planning', scope_metadata['scope_id'], operation='interpolation'):
                prepared = prepare_sam_interpolation_pass(observed, view=view,
                    scope=scope_metadata, policy=self.policy,
                    **profile_kwargs,
                    **{key: value for key, value in kwargs.items() if key in planning_keys})
            preparation_phase = 'image_or_gpu_admission'
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
            if self.crop_retry_policy is not None:
                kwargs.setdefault('crop_retry_policy', self.crop_retry_policy)
                kwargs.setdefault('retry_image_provider',
                    lambda retry: self._prepare_image_cohort(view, shape, retry))
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
            if not execution_started:
                destination = kwargs.get('work_dir', self.evidence_root/hashlib.sha256(str(scope).encode()).hexdigest()[:20])
                receipt = {
                    'schema': 'xta.sam-context-preparation-failure/1', 'complete': False,
                    'status': 'infrastructure_invalid', 'phase': preparation_phase,
                    'error_type': type(error).__name__, 'error': str(error), 'scope_id': str(scope),
                    'sam_operation': 'interpolation',
                    'view_name': scope_metadata.get('view_name', str(getattr(view, 'name', ''))),
                    'physical_view': scope_metadata.get('physical_view'),
                    'sam_crop_mode': self.crop_mode,
                    'sam_resource_profile': scope_metadata.get('sam_resource_profile'),
                    'pass_index': kwargs.get('pass_index', 1),
                    'prepared_plan_available': prepared is not None,
                    'observation_snapshot_sha256': None if prepared is None else prepared.observation_snapshot_sha256,
                    'planning_settings_sha256': None if prepared is None else prepared.settings_sha256,
                    'native_shape_tyx': shape if prepared is None else list(prepared.native_shape),
                    'runs': [{'run_id': str(run.run_id), 'group_id': str(run.group_id),
                              'status': 'not_attempted'} for run in (() if prepared is None else prepared.runs)],
                }
                if isinstance(getattr(error, 'receipt', None), Mapping):
                    receipt['resource_admission'] = dict(error.receipt)
                _write_context_preparation_failure(destination, receipt, error)
            raise
        finally:
            self._resource_local.sam_phase_scope = previous_phase_scope
            with self._idle:
                self._active_passes -= 1
                self._idle.notify_all()

    def extrapolate(self, observation_volume, *, view, scope, **kwargs):
        """Track original terminals with the same admitted image/model owners."""
        from .sam_extrapolation import (prepare_sam_extrapolation_pass,
                                        extrapolate_sam_view_volume_pass,
                                        plan_sam_extrapolation_image_cohorts)
        from .sam_interpolation import _trace_sam_phase
        from .geometry import physical_view_name
        with self._lock:
            if self._closed:
                raise RuntimeError('SAM runtime is closed')
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
            self._active_passes += 1
        previous_phase_scope = getattr(self._resource_local, 'sam_phase_scope', ('', ''))
        self._resource_local.sam_phase_scope = ('extrapolation', str(scope))
        prepared = None
        execution_started = False
        preparation_phase = 'geometry_or_resource_admission'
        shape, metadata = None, {}
        try:
            observed = np.asarray(observation_volume).view()
            observed.flags.writeable = False
            shape = tuple(int(value) for value in observed.shape)
            if len(shape) != 3 or min(shape) < 1:
                raise ValueError('SAM observation canvas must have positive TYX dimensions')
            _, _, transform = self._canvas_transform(view, shape)
            metadata = dict(scope_id=str(scope), evidence_purpose='sam_extrapolation',
                detector_identity=self.detector_identity, sam_bundle_identity=self.bundle_identity,
                view_name=str(view.name), physical_view=physical_view_name(view),
                angle_deg=float(view.tta_angle_deg), canvas_transform=transform,
                sam_crop_mode=self.crop_mode, sam_working_canvas_shape_tyx=list(shape),
                sam_native_view_shape_tyx=[int(view.num_slices), int(view.src_h), int(view.src_w)],
                source_resize_semantics=self.source_resize_semantics)
            profile = getattr(self._resource_local, 'profile', None)
            if profile is not None:
                profile._validate_owner()
                metadata['sam_resource_profile'] = profile.metadata()
                if profile.has_extra_credit:
                    from .sam_bridge_planning import SamPlanningLimits
                    limits = kwargs.get('planner_limits')
                    kwargs['planner_limits'] = (SamPlanningLimits(
                        max_group_bytes=profile.assigned_contract_bytes,
                        max_total_contract_bytes=profile.assigned_live_contract_bytes)
                        if limits is None else replace(limits,
                            max_group_bytes=min(limits.max_group_bytes, profile.assigned_contract_bytes),
                            max_total_contract_bytes=min(limits.max_total_contract_bytes,
                                profile.assigned_live_contract_bytes)))
            else:
                metadata['sam_resource_profile'] = {
                    'schema': 'xta.sam_live_resources/1', 'status': 'direct_declared_bounds',
                    'reserved_extra_bytes': 0}
            profile_kwargs = {} if profile is None else {'resource_profile': profile}
            planning_keys = {'distance', 'walk_back', 'min_radius', 'wrap_axis',
                'upstream_lineage', 'spacing_zyx', 'planner_limits', 'canonical_labels',
                'eligible_terminals'}
            preparation_phase = 'planning'
            with _trace_sam_phase('planning', metadata['scope_id'], operation='extrapolation'):
                prepared = prepare_sam_extrapolation_pass(observed, view=view, scope=metadata,
                    crop_mode=self.crop_mode, **profile_kwargs,
                    **{key: value for key, value in kwargs.items() if key in planning_keys})
            preparation_phase = 'image_or_gpu_admission'
            with self._lock:
                self.planning_seconds += (getattr(prepared, 'planner_wall_seconds', 0.)
                                          + getattr(prepared, 'snapshot_wall_seconds', 0.))
                self.resource_assignments[str(scope)] = dict(metadata['sam_resource_profile'])
            provider = runtime = None
            cohort_record = None
            if prepared.needs_tracking:
                from .workspace import _env_int
                cap = max(1, _env_int('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', 1024**3))
                cohorts = plan_sam_extrapolation_image_cohorts(prepared, cap)
                cohort_record = dict(configured_cache_bytes=cap,
                    complete_original_groups=True, cohorts=[dict(cohort_id=item.cohort_id,
                        group_ids=list(item.group_ids), payload_bytes=item.payload_bytes) for item in cohorts])
                if len(cohorts) == 1:
                    # Preserve the ordinary retained cache path when the full
                    # scope already fits. The multi-cohort path needs explicit
                    # worker barriers before staging its next descriptor.
                    provider = self.image_provider(view, shape, prepared_plan=prepared)
                    kwargs['image_cohorts'] = cohorts
                    kwargs['image_cohort_provider'] = lambda subset: nullcontext(provider)
                self._start()
                runtime = self._runtime
                if len(cohorts) > 1:
                    if not callable(getattr(runtime, 'release_source_cache', None)):
                        raise RuntimeError('Multiple SAM image cohorts require a worker source-cache retirement barrier')
                    kwargs['image_cohorts'] = cohorts
                    kwargs['image_cohort_provider'] = lambda subset: self._prepare_image_cohort(
                        view, shape, subset, max_cache_bytes=cap)
                    kwargs['image_cohort_prefetch'] = lambda subset: self.prefetch_image_cohort(
                        view, shape, subset, max_cache_bytes=cap)
            else:
                with self._lock:
                    self.no_job_passes += 1
            if self._cancel.is_set():
                raise RuntimeError(self._failure)
            execution_started = True
            if self.crop_retry_policy is not None:
                kwargs.setdefault('crop_retry_policy', self.crop_retry_policy)
                kwargs.setdefault('retry_image_provider',
                    lambda retry: self._prepare_image_cohort(view, shape, retry))
            kwargs.setdefault('runtime_work_dir', self.temp_dir / 'sam_extrap_runtime')
            _, stats, components = extrapolate_sam_view_volume_pass(observed,
                image_provider=provider, runtime=runtime, view=view, scope=metadata,
                prepared_plan=prepared, crop_mode=self.crop_mode, cancel_event=self._cancel,
                **profile_kwargs, **kwargs)
            stats = dict(stats)
            if cohort_record is not None:
                stats['sam_image_cache_cohorts'] = cohort_record
            stats['sam_image_cache_lifetime'] = self.image_cache_lifetime_snapshot()
            return observation_volume, stats, components
        except BaseException as error:
            if not execution_started:
                destination = kwargs.get('work_dir', self.extrapolation_evidence_root /
                    hashlib.sha256(str(scope).encode()).hexdigest()[:20])
                receipt = dict(schema='xta.sam-context-preparation-failure/1', complete=False,
                    evidence_purpose='sam_extrapolation', source_stage='post_interpolation',
                    status='infrastructure_invalid', phase=preparation_phase,
                    error_type=type(error).__name__, error=str(error), scope_id=str(scope),
                    sam_operation='extrapolation',
                    view_name=metadata.get('view_name', str(getattr(view, 'name', ''))),
                    physical_view=metadata.get('physical_view'), sam_crop_mode=self.crop_mode,
                    sam_resource_profile=metadata.get('sam_resource_profile'),
                    prepared_plan_available=prepared is not None,
                    observation_snapshot_sha256=None if prepared is None else prepared.observation_snapshot_sha256,
                    planning_settings_sha256=None if prepared is None else prepared.settings_sha256,
                    native_shape_tyx=shape if prepared is None else list(prepared.native_shape))
                if isinstance(getattr(error, 'receipt', None), Mapping):
                    receipt['resource_admission'] = dict(error.receipt)
                _write_context_preparation_failure(destination, receipt, error)
            raise
        finally:
            self._resource_local.sam_phase_scope = previous_phase_scope
            with self._idle:
                self._active_passes -= 1
                self._idle.notify_all()

    def close(self):
        self.cancel('SAM interpolation runtime is closing')
        if self.progressive_startup:
            self._join_progressive_startup()
            if self._runtime is None:
                self._runtime = self._starting_runtime
            self._starting_runtime = None
        with self._idle:
            prefetches = tuple(self._image_prefetches)
        for prefetch in prefetches:
            try:
                prefetch.close_if_unused()
            except BaseException:
                _retain_unsettled_context(self)
                raise
        with self._idle:
            if self._closed:
                return
            while (self._active_passes or self._active_image_calls or self._active_image_cohorts
                    or self._image_retirements or self._image_prefetches):
                self._idle.wait(timeout=0.25)
            close_error = None
            try:
                for renderer in tuple(self._unsettled_image_renderers):
                    renderer.close()
                if self._runtime is not None:
                    dispatch = getattr(self._runtime, 'dispatch_stats', {})
                    self.dispatch_summary = dict(dispatch) if isinstance(dispatch, Mapping) else {}
                    try:
                        self._runtime.close()
                    except BaseException as error:
                        if (getattr(self._runtime, 'residency_released', False) is not True
                                or getattr(self._runtime, 'cleanup_settled', True) is not True):
                            # A retry still owns both model and device lease.
                            # Never advertise memory as available while a
                            # worker process may retain the predictor.
                            _retain_unsettled_context(self)
                            raise
                        close_error = error
                    self._runtime = None
                if self._progressive_pool is not None and not self._progressive_pool.workers_settled:
                    self._progressive_pool.force_close(timeout=1.)
                    if not self._progressive_pool.workers_settled:
                        _retain_unsettled_context(self)
                        raise RuntimeError('SAM progressive child ownership remains unsettled')
                for device in tuple(self._startup_host_grants):
                    self._release_progressive_host(device)
                if self._startup_pool is not None:
                    with self._startup_pool.condition:
                        self._startup_future_peaks.clear()
                        self._sync_startup_fleet_credit_locked()
                for release_credit in self._retained_image_prefetch_credits:
                    release_credit()
                self._retained_image_prefetch_credits.clear()
                self._closed = True
                _forget_settled_context(self)
                for lease in reversed(self._leases):
                    lease.release()
                self._leases.clear()
                with self._gpu_lease_lock:
                    for resident in self._resident_leases.values():
                        resident.release(residency_settled=True)
                    self._resident_leases.clear()
                self._caches.clear()
                self._cache_transforms.clear()
                self._cache_entries.clear()
                self._cache_owners.clear()
                self._image_sampling_proofs.clear()
                self.cache_logical_bytes = 0
                self.source_volume = None
            finally:
                self._idle.notify_all()
            if close_error is not None:
                raise close_error

