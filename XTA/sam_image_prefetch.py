"""Bounded per-parent cohort preparation, with credit owned by its producer."""
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
import threading
import time
import sys
import uuid

from . import sam_resources as resources

PREFETCH_CANCEL_JOIN_SECONDS = 30.


class _ImageProfile(resources.SamResourceProfile):
    @property
    def assigned_observation_bytes(self):
        return 0

    @property
    def assigned_contract_metadata_bytes(self):
        return 0

    @property
    def nominal_cpu_wave_base_bytes(self):
        return 0

    @property
    def assigned_session_cpu_bytes(self):
        return 0

    def metadata(self):
        result = super().metadata()
        result['status'] = 'image_only_admission'
        result['persistent_metadata_allowance_status'] = 'image_only_no_persistent_metadata'
        return result


@contextmanager
def _try_image_profile(pool, needed, scope_id, probe, *, parent_profile=None, cancel_event=None):
    """Keep proven standalone grants; fund optional envelope refusals from parents."""
    profile = None
    optional_request = parent_profile is not None
    parent_funded = False
    if optional_request:
        needed = resources._scope_integer(needed, 'SAM prefetched image bytes', positive=True)
    with pool.condition:
        reason = None
        parent_lease = getattr(parent_profile, '_lease', None)
        if optional_request:
            with resources._LIVE_LOCK:
                live_parent = (parent_lease is not None and
                    resources._LIVE_PROFILES.get(parent_lease.lease_id) is parent_profile)
            if (not live_parent or not parent_lease.active or parent_lease.owner_closed
                    or parent_lease.pool is not pool or hasattr(pool, '_sam_parent_pool')):
                reason = 'parent_inactive_or_not_canonical'
        staging = getattr(pool, '_sam_image_staging_pool', None)
        if staging is None and reason is None:
            staging = pool._sam_image_staging_pool = SimpleNamespace(
                capacity=resources.MAX_IMAGE_STAGING_BYTES, in_use=0, condition=pool.condition,
                _sam_parent_pool=pool)
        physical = 0
        if reason is None:
            try:
                physical = max(0, int(probe()))
            except Exception:
                pass
        other = resources.sam_parent_promised_bytes(pool)
        # Parent-funded images already occupy parent rights. Only external
        # image grants extend the full future-parent physical envelope.
        external_images = (0 if staging is None else int(staging.in_use)) \
            -resources._parent_funded_image_bytes(pool, staging)
        standalone_envelope = max(int(pool.capacity), int(pool.in_use))+external_images
        protected = standalone_envelope
        pending = startup = 0
        if reason is None:
            reason = ('parent_emergency' if getattr(pool, 'oversize_requested_bytes', 0) else
                'staging_capacity' if needed > max(0, int(staging.capacity)-int(staging.in_use)) else
                'physical_headroom' if needed > max(0, physical-protected) else None)
        if optional_request and reason == 'physical_headroom':
            # Existing successful standalone overlap must not lose concurrency
            # merely because parents currently use all of their own capacity.
            # Only an envelope refusal may spend uncommitted parent rights.
            parent_funded = True
            protected = other
            pending = sum(resources._bounded_parent_protections(pool).values())
            startup = max(0, int(getattr(pool, '_sam_startup_future_bytes', 0))
                -int(getattr(pool, '_sam_startup_active_bytes', 0)))
            reason = ('parent_capacity_or_protection' if needed > max(0,
                int(pool.capacity)-int(pool.in_use)-pending-startup) else
                'physical_headroom' if needed > max(0, physical-protected) else None)
        # Cancellation can arrive while the physical probe is reading procfs.
        # Recheck at the transaction boundary, without creating a waiting grant.
        if optional_request and reason is None:
            with resources._LIVE_LOCK:
                live_parent = resources._LIVE_PROFILES.get(parent_lease.lease_id) is parent_profile
            if not live_parent or not parent_lease.active or parent_lease.owner_closed:
                reason = 'parent_inactive_or_not_canonical'
            elif (cancel_event is not None and cancel_event.is_set()
                    or parent_lease.cancel_event is not None and parent_lease.cancel_event.is_set()):
                reason = 'cancelled'
        if reason is None:
            lease = resources._LiveLease(uuid.uuid4().hex, threading.get_ident(), pool=staging,
                charged_bytes=needed, headroom_probe=probe, scope_id=str(scope_id),
                admitted_monotonic=time.monotonic())
            profile = _ImageProfile(scope_id, needed, needed, 0, int(staging.capacity), physical,
                1, needed, other, lease)
            if parent_funded:
                resources._charge_parent_funded_image(lease)
            else:
                # Generic parents may fill their whole capacity without a new
                # physical probe; preserve this standalone admission contract.
                staging.in_use += needed
            try:
                with resources._LIVE_LOCK:
                    resources._LIVE_PROFILES[lease.lease_id] = profile
            except BaseException:
                with resources._LIVE_LOCK:
                    if resources._LIVE_PROFILES.get(lease.lease_id) is profile:
                        resources._LIVE_PROFILES.pop(lease.lease_id)
                lease.active = False
                lease.owner_closed = True
                resources._return_parent_credit_if_settled(lease)
                raise
        resources._trace_parent_memory('sam_image_memory_admitted' if profile is not None else
            'sam_image_memory_declined', scope_id=str(scope_id),
            lease_id=None if profile is None else profile._lease.lease_id,
            requested_bytes=needed, parent_pool_in_use_bytes=int(pool.in_use),
            parent_pool_capacity_bytes=int(pool.capacity), physical_headroom_bytes=physical,
            protected_parent_and_image_bytes=protected, reason=reason,
            standalone_envelope_bytes=standalone_envelope,
            image_funding_mode='parent_pool' if parent_funded else 'standalone',
            parent_protection_bytes=pending, future_startup_bytes=startup,
            **resources.sam_image_staging_snapshot(pool))
    if profile is None:
        yield None
        return
    try:
        yield profile
    finally:
        with pool.condition:
            lease.active = False
            lease.owner_closed = True
            with resources._LIVE_LOCK:
                resources._LIVE_PROFILES.pop(lease.lease_id, None)
            resources._return_parent_credit_if_settled(lease)


class SamImageCohortPrefetch:
    """Transfer a reference; original producer alone exits image/profile contexts."""
    def __init__(self, context, view, shape, prepared, cap, parent_profile, phase_bytes):
        self.context, self.view, self.shape, self.prepared = context, view, shape, prepared
        self.cap, self.parent = cap, parent_profile
        self.phase_bytes = phase_bytes
        self.creator = threading.get_ident()
        self.phase_scope = getattr(context._resource_local, 'sam_phase_scope', ('', ''))
        self.future, self.finish, self.cancelled = Future(), threading.Event(), threading.Event()
        self._lock = threading.RLock()
        self._entered = self._closed = self._consumer_failed = False
        self._fallback = None
        self._worker_error = None
        self._admitted_bytes = 0
        self._thread = threading.Thread(target=self._produce, name='sam-image-prefetch', daemon=True)

    def start(self):
        self._thread.start()

    def _produce(self):
        try:
            lease = self.parent._lease
            with _try_image_profile(lease.pool, self.phase_bytes,
                    self.parent.scope_id+'/next-images', lease.headroom_probe,
                    parent_profile=self.parent, cancel_event=self.cancelled) as profile:
                if profile is None:
                    self.future.set_result(None)
                    return
                self._admitted_bytes = self.phase_bytes
                self.context._resource_local.image_prefetch_cancel = self.cancelled
                self.context._resource_local.sam_phase_scope = self.phase_scope
                try:
                    with self.context.resource_scope(profile):
                        with self.context.image_cohort_provider(self.view, self.shape, self.prepared,
                                max_cache_bytes=self.cap) as reference:
                            self.future.set_result(reference)
                            while not self.finish.wait(.05):
                                with self._lock:
                                    entered = self._entered
                                # A consumed lease still belongs to its caller,
                                # even during cancellation before SDK admission.
                                # Its lexical exit retires readers and signals us.
                                if not entered:
                                    self.context._check_image_lifetime()
                            if self._consumer_failed:
                                raise RuntimeError('SAM prefetched cohort consumer failed')
                finally:
                    self.context._resource_local.image_prefetch_cancel = None
        except BaseException as error:
            self._worker_error = error
            if not self.future.done():
                self.future.set_exception(error)
        finally:
            with self.context._idle:
                self.context._image_prefetches.discard(self)
                self.context._idle.notify_all()

    def __enter__(self):
        accepted = False
        try:
            with self._lock:
                if self._entered or self._closed or threading.get_ident() != self.creator:
                    raise RuntimeError('SAM image prefetch must be consumed once on its original preparation thread')
                accepted = True
                self.parent._validate_owner()
                self._entered = True
            with self.context._idle:
                self.context._check_image_lifetime()
            reference = self.future.result()
            with self._lock:
                if self._closed:
                    raise RuntimeError('SAM image prefetch was abandoned before consumption')
                self.context._check_image_lifetime()
                if reference is None:
                    self._fallback = self.context.image_cohort_provider(self.view, self.shape,
                        self.prepared, max_cache_bytes=self.cap)
                    return self._fallback.__enter__()
                reference.revalidate()
                return reference
        except BaseException as error:
            if accepted:
                # No reference was handed to this caller. There will be no
                # lexical __exit__ to signal the producer after failed entry.
                with self._lock:
                    self._consumer_failed = True
                    self.cancelled.set()
                    self.finish.set()
                try:
                    self.close()
                except BaseException as cleanup:
                    if callable(getattr(error, 'add_note', None)):
                        error.add_note(f'SAM image prefetch cleanup also failed: {cleanup}')
            raise

    def __exit__(self, exc_type, exc, tb):
        if self._fallback is not None:
            try:
                return self._fallback.__exit__(exc_type, exc, tb)
            finally:
                with self._lock:
                    self._closed = True
        with self._lock:
            self._consumer_failed = exc is not None
            self.finish.set()
        try:
            if exc is None:
                self._join(normal_exit=True)
            else:
                self._join()
        except BaseException as cleanup:
            if exc is None:
                raise
            if callable(getattr(exc, 'add_note', None)):
                exc.add_note(f'SAM image prefetch cleanup also failed: {cleanup}')
            return False
        with self._lock:
            self._closed = True
        if self._worker_error is not None:
            if exc is None:
                raise self._worker_error
            if callable(getattr(exc, 'add_note', None)):
                exc.add_note(f'SAM prefetched image retirement: {self._worker_error}')

    def cancel_builder(self):
        with self._lock:
            self.cancelled.set()
            if not self._entered:
                self.finish.set()

    def _join(self, *, normal_exit=False):
        # Normal retirement can wait for live donors; failed or cancelled work has a deadline.
        deadline = None if normal_exit else time.monotonic()+PREFETCH_CANCEL_JOIN_SECONDS
        while True:
            self._thread.join(timeout=.1)
            if not self._thread.is_alive():
                return
            now = time.monotonic()
            if deadline is None and (self.cancelled.is_set() or self.context._cancel.is_set()
                    or self.context._closed):
                deadline = now+PREFETCH_CANCEL_JOIN_SECONDS
            if deadline is not None and now >= deadline:
                raise RuntimeError('SAM image prefetch producer remains active; image/profile ownership retained')

    def close(self):
        """Abandon an unused wrapper; its errors cannot replace a current failure."""
        primary = sys.exc_info()[1]
        with self._lock:
            if self._closed and not self._thread.is_alive():
                return
            if self._fallback is not None:
                raise RuntimeError('Consumed synchronous image cohort must exit on its consumer thread')
            self._closed = True
            self.cancel_builder()
        try:
            self._join()
        except BaseException as cleanup:
            if primary is None:
                raise
            if callable(getattr(primary, 'add_note', None)):
                primary.add_note(f'SAM image prefetch cleanup also failed: {cleanup}')

    def close_if_unused(self):
        """Context shutdown cannot steal a concurrently entered caller's lease."""
        with self._lock:
            if self._entered:
                return False
            self._closed = True
            self.cancelled.set()
            self.finish.set()
        self._join()
        return True
