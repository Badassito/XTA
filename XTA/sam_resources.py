"""Live, atomic parent admission for production SAM CPU workspaces.

Pool capacity is an accounting limit, not evidence of physical RAM. Larger
family contracts are enabled only by additional credit reserved together with
the parent's existing work; serialized profiles never authorize replay memory.
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
import operator
from pathlib import Path
import re
import threading
import time
from types import MappingProxyType
import uuid

GIB = 1024**3
SCHEMA = 'xta.sam_live_resources/1'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAX_PRODUCTION_EXTRA_BYTES = 16*GIB
MIN_PRODUCTION_EXTRA_BYTES = 512*1024**2
_LIVE_PROFILES = {}
_LIVE_LOCK = threading.RLock()
_LIVE_TRACKER_ADMISSIONS = {}
_TRACKER_ADMISSION_SEAL = object()
_MASK_PACKING_SEAL = object()
MAX_TRACKER_LOOKAHEAD_BYTES = GIB
MAX_IMAGE_STAGING_BYTES = 16*GIB


def sam_parent_promised_bytes(pool):
    """Read both ledgers while holding their shared parent condition."""
    pool = getattr(pool, '_sam_parent_pool', pool)
    staging = getattr(pool, '_sam_image_staging_pool', None)
    future_startup = max(0, int(getattr(pool, '_sam_startup_future_bytes', 0))
        -int(getattr(pool, '_sam_startup_active_bytes', 0)))
    return int(pool.in_use)+(0 if staging is None else int(staging.in_use))+future_startup


def sam_image_staging_snapshot(pool):
    """Include retained image debt even after its producer wrapper has exited."""
    pool = getattr(pool, '_sam_parent_pool', pool)
    staging = getattr(pool, '_sam_image_staging_pool', None)
    return dict(image_staging_in_use_bytes=0 if staging is None else int(staging.in_use),
        image_staging_capacity_bytes=MAX_IMAGE_STAGING_BYTES if staging is None else int(staging.capacity))


def physical_sam_headroom():
    """Physical/cgroup/SLURM headroom; node-wide swap never grants credit."""
    from .publication_memory import publication_ram_headroom
    from .workspace import available_anon_work_bytes
    physical = max(0, int(publication_ram_headroom()))
    anon = max(0, int(available_anon_work_bytes()))
    # /proc accounting is unavailable on Windows; publication's psutil RAM
    # branch is authoritative there, rather than treating absent data as zero.
    if os.name != 'nt' or anon:
        physical = min(physical, anon)
    limits = []
    for name in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU'):
        raw = os.environ.get(name)
        if raw is None:
            continue
        if not str(raw).isdigit():
            return 0  # A declared but unrecognized allocation grants no extra.
        # Slurm --mem=0 grants all node/job memory, not a zero-byte allocation.
        # Physical/cgroup headroom and atomic pool credit still bound this run.
        if int(raw) == 0:
            continue
        limit = int(raw)*1024**2
        if name.endswith('CPU'):
            cpus = os.environ.get('SLURM_CPUS_ON_NODE') or os.environ.get('SLURM_CPUS_PER_TASK')
            match = re.match(r'^([1-9][0-9]*)(?:\(x[1-9][0-9]*\))?$', str(cpus or ''))
            if match is None:
                return 0
            limit *= int(match.group(1))
        limits.append(limit)
    if limits:
        try:
            import psutil
            process = psutil.Process()
            # RSS can double count shared mappings; that only reduces credit.
            resident = int(process.memory_info().rss)
            for child in process.children(recursive=True):
                try:
                    resident += int(child.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return 0
        except (ImportError, OSError, ValueError, AttributeError):
            return 0
        physical = min(physical, max(0, min(limits)-resident))
    return physical


@dataclass
class _LiveLease:
    lease_id: str
    owner_thread: int
    active: bool = True
    pool: object = field(default=None, repr=False)
    charged_bytes: int = 0
    headroom_probe: object = field(default=None, repr=False)
    scope_holds: int = 0
    owner_closed: bool = False
    credit_returned: bool = False
    tracker_scope_identity: str | None = None
    execution_slots: int = 0
    scope_id: str = ''
    admitted_monotonic: float = 0.
    oversize_requested_bytes: int = 0


def _trace_parent_memory(event, **fields):
    try:
        from .runtime import runtime_trace_event
        runtime_trace_event(event, **fields)
    except Exception:
        pass  # Optional diagnostics must not alter admission or settlement.


def _return_parent_credit_if_settled(lease):
    """Called with the existing pool condition held; permission and credit differ."""
    if lease.owner_closed and not lease.scope_holds and not lease.credit_returned:
        lease.pool.in_use = max(0, int(lease.pool.in_use)-int(lease.charged_bytes))
        if lease.oversize_requested_bytes:
            lease.pool.oversize_requested_bytes = 0
        lease.credit_returned = True
        lease.pool.condition.notify_all()
        event = ('sam_image_memory_credit_returned' if hasattr(lease.pool, '_sam_parent_pool')
                 else 'sam_parent_memory_credit_returned')
        _trace_parent_memory(event, scope_id=lease.scope_id,
            lease_id=lease.lease_id, charged_bytes=lease.charged_bytes,
            pool_in_use_bytes=lease.pool.in_use, pool_capacity_bytes=lease.pool.capacity,
            scope_holds=lease.scope_holds,
            lifetime_seconds=max(0., time.monotonic()-lease.admitted_monotonic),
            **sam_image_staging_snapshot(lease.pool))


def _scope_integer(value, name, *, positive=False):
    if isinstance(value, bool) or getattr(getattr(value, 'dtype', None), 'kind', None) == 'b':
        raise ValueError(f'{name} must be an integer')
    try:
        parsed = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f'{name} must be an integer') from error
    if parsed < int(bool(positive)):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return parsed


class SamTrackerScopeAdmission:
    """An authenticated scope bank; background settlement retains parent credit.

    Only the original producer can authorize a request. The scheduler receives
    immutable limits and one retained scope hold, rather than an owner-thread
    profile or serialized allocation permission. A creator exiting expires
    preparation permission immediately; already accepted work keeps its bytes.
    """

    def __init__(self, profile, limits, *, _seal=None, _identity=None):
        if _seal is not _TRACKER_ADMISSION_SEAL:
            raise TypeError('SAM tracker admission must be minted from a live parent profile')
        self._profile = profile
        self._limits = MappingProxyType(dict(limits))
        self._identity = _identity or uuid.uuid4().hex
        self._lock = threading.RLock()
        self._creator_active = True
        self._scope_active = False
        self._scope_used = False
        self._returned = False
        self._packing_admissions = set()

    @property
    def lookahead_jobs(self):
        return self._limits['lookahead_jobs']

    @property
    def max_in_flight(self):
        return self._limits['max_in_flight']

    def _validate_producer(self):
        self._require_minted()
        self._profile._validate_owner()
        if not self._creator_active or self._returned:
            raise RuntimeError('SAM tracker preparation admission has expired')

    def _require_minted(self):
        with _LIVE_LOCK:
            minted = _LIVE_TRACKER_ADMISSIONS.get(self._identity) is self
        if not minted or self._returned:
            raise RuntimeError('SAM tracker scope admission has expired or was not minted')

    def validate_request(self, frames, pixels):
        with self._lock:
            self._validate_producer()
            frames = _scope_integer(frames, 'SAM tracker frame count', positive=True)
            pixels = _scope_integer(pixels, 'SAM tracker seed pixels', positive=True)
            if (frames > self._limits['maximum_frame_count'] or pixels > self._limits['maximum_seed_pixels']
                    or frames*pixels > self._limits['maximum_raw_mask_bytes']
                    or cpu_session_bytes(frames, pixels)['estimated_peak_bytes'] > self._limits['maximum_session_cpu_bytes']):
                raise RuntimeError('SAM prepared request exceeds its original per-attempt admission')
            return self._limits

    def validate_cache(self, cache_ref):
        with self._lock:
            self._validate_producer()
            if sam_cache_descriptor_bytes(cache_ref) > self._limits['maximum_cache_payload_bytes']:
                raise RuntimeError('SAM cache descriptor exceeds its admitted prepared-bank metadata bound')
            return self._limits

    def acquire_scope(self):
        with self._lock:
            self._validate_producer()
            if self._scope_used:
                raise RuntimeError('SAM tracker admission belongs to one scope')
            with self._profile._lease.pool.condition:
                self._profile._lease.scope_holds += 1
            self._scope_used = self._scope_active = True
            return self._limits

    def release_scope(self):
        """Scheduler calls only after producer, SDK and transfer owners settle."""
        for packing in tuple(self._packing_admissions):
            packing.close()
        with self._lock:
            if not self._scope_active:
                return
            self._require_minted()
            self._scope_active = False
            with self._profile._lease.pool.condition:
                self._profile._lease.scope_holds -= 1
                _return_parent_credit_if_settled(self._profile._lease)
            self._return_bank_if_settled()

    def _return_bank_if_settled(self):
        if self._creator_active or self._scope_active or self._returned:
            return
        self._returned = True
        pool = self._profile._lease.pool
        with pool.condition:
            pool.in_use = max(0, int(pool.in_use)-int(self._limits['prepared_bank_bytes']))
            if self._profile._lease.tracker_scope_identity == self._identity:
                self._profile._lease.tracker_scope_identity = None
            pool.condition.notify_all()
        with _LIVE_LOCK:
            _LIVE_TRACKER_ADMISSIONS.pop(self._identity, None)

    def close(self):
        with self._lock:
            if not self._creator_active:
                return
            self._require_minted()
            self._creator_active = False
            with self._profile._lease.pool.condition:
                self._profile._lease.scope_holds -= 1
                _return_parent_credit_if_settled(self._profile._lease)
            self._return_bank_if_settled()

    def admit_mask_packing(self):
        """Lend only unused, already funded attempt-wave bytes after raw decode."""
        with self._lock:
            self._validate_producer()
            if not self._scope_active:
                raise RuntimeError('SAM mask packing requires an active tracker scope')
            packing = SamMaskPackingAdmission(self, _seal=_MASK_PACKING_SEAL)
            self._packing_admissions.add(packing)
            return packing

    def __enter__(self):
        with self._lock:
            self._validate_producer()
        return self

    def __exit__(self, *exc):
        self.close()


class SamMaskPackingAdmission:
    """One live result's packing scratch, borrowed from unused attempt credit.

    The full admitted SDK wave and all three raw-transfer histories remain
    reserved. Neither saved receipts nor a worker thread authorize this lane.
    Futures are retained through publication and joined before scope refund.
    """

    def __init__(self, scope, *, _seal=None):
        if _seal is not _MASK_PACKING_SEAL:
            raise TypeError('SAM mask packing must be minted from a live tracker scope')
        self._scope = scope
        self._closed = False
        self._futures = set()
        self._submission_unproven = False
        self._lock = threading.RLock()

    def validate(self):
        with self._scope._lock:
            self._scope._validate_producer()
            if self._closed or not self._scope._scope_active:
                raise RuntimeError('SAM mask packing admission has expired')
        return self

    def scratch_bytes(self, plane_pixels):
        self.validate()
        plane = _scope_integer(plane_pixels, 'SAM evidence plane pixels', positive=True)
        limits = self._scope._limits
        contract = limits['evidence_contract_bytes']
        if contract is None:
            return 0  # An unproven live contract cannot authorize helper scratch.
        # Protect the serial writer's full plane allowance and persistent
        # assembly, in addition to the complete original SDK/raw peak.
        return max(0, int(limits['unused_attempt_wave_bytes'])
            -int(limits['evidence_fixed_bytes'])-max(16*plane, int(contract)))

    def cpu_worker_limit(self):
        self.validate()
        from .lta_cpu import resolve_worker_cpu_budget
        budget = resolve_worker_cpu_budget(self._scope._profile.execution_slots)
        return min(32, max(1, int(budget['coordinator_cpu_reserve'])-1))

    def track(self, future):
        self.validate()
        with self._lock:
            self._futures.add(future)

    def submit(self, executor, function, *args):
        self.validate()
        with self._lock:
            if self._closed:
                raise RuntimeError('SAM mask packing admission has expired')
            # Keep the grant if interruption prevents the executor from
            # returning its submitted future. Unknown work cannot be refunded.
            self._submission_unproven = True
            future = executor.submit(function, *args)
            self._futures.add(future)
            self._submission_unproven = False
            return future

    def release(self, future):
        with self._lock:
            if not future.done():
                raise RuntimeError('SAM mask packing future is still active')
            self._futures.discard(future)

    def close(self):
        with self._lock:
            futures = tuple(self._futures)
            if self._submission_unproven:
                raise RuntimeError('SAM mask packing submission quiescence is unproven; credit retained')
        for future in futures:
            future.cancel()
        error = None
        for future in futures:
            try:
                future.result()
            except CancelledError:
                pass
            except BaseException as exc:
                if not future.done():
                    # An interrupted wait is not evidence that the worker
                    # finished. Keep all ownership for a later settlement.
                    raise
                import traceback
                traceback.clear_frames(exc.__traceback__)
                error = error or exc
            if not future.done():
                raise RuntimeError('SAM mask packing future completion is unproven; credit retained')
        with self._lock:
            self._futures.clear()
            self._closed = True
        with self._scope._lock:
            self._scope._packing_admissions.discard(self)
        if error is not None:
            raise error


def validate_sam_tracker_scope_admission(admission):
    """Authenticate detached scheduler limits without claiming producer authority."""
    if not isinstance(admission, SamTrackerScopeAdmission):
        raise TypeError('SAM scheduler requires a live SamTrackerScopeAdmission')
    with admission._lock:
        with _LIVE_LOCK:
            minted = _LIVE_TRACKER_ADMISSIONS.get(admission._identity) is admission
        if not minted or admission._returned or not (admission._creator_active or admission._scope_active):
            raise RuntimeError('SAM tracker scope admission has expired')
        return admission._limits


def sam_cache_descriptor_bytes(cache_ref):
    """Conservative retained primitive inventory, not serialized allocation permission."""
    payload = cache_ref.payload() if callable(getattr(cache_ref, 'payload', None)) else {}
    return 8*len(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode())+64*1024


def admit_sam_tracker_scope(profile, cpu_wave_admission, *, max_seed_pixels, max_frame_count,
                            max_in_flight=None, cache_payload_bytes=None,
                            evidence_persistent_bytes=0, evidence_contract_bytes=None):
    """Try an additional bounded prepared bank; never wait behind the parent.

    A funded base wave remains concurrent when the extra bank cannot be funded.
    The supplied attempt wave is authoritative, including retry clamps and the
    exclusive transfer barrier. Bank bytes pay staging/packed backlog separately
    from that wave's SDK inputs and original decoded-consumer transfer margin.
    """
    record = validate_live_sam_resource_profile(profile)
    wave = dict(cpu_wave_admission)
    capacity = _scope_integer(wave['max_in_flight'], 'SAM admitted SDK slots', positive=True)
    wave_slots = _scope_integer(wave.get('execution_slots', capacity), 'SAM wave execution slots', positive=True)
    if max_in_flight is not None:
        capacity = min(capacity, _scope_integer(max_in_flight, 'SAM requested SDK slots', positive=True))
    pixels = _scope_integer(max_seed_pixels, 'SAM maximum seed pixels', positive=True)
    frames = _scope_integer(max_frame_count, 'SAM maximum frame count', positive=True)
    descriptor_bytes = (1024**2 if cache_payload_bytes is None else
                        _scope_integer(cache_payload_bytes, 'SAM cache descriptor bytes', positive=True))
    session = _scope_integer(wave['maximum_session_cpu_estimate_bytes'], 'SAM session byte bound', positive=True)
    raw = _scope_integer(wave['maximum_raw_mask_bytes'], 'SAM raw byte bound', positive=True)
    transfer = _scope_integer(wave['transfer_margin_bytes'], 'SAM consumer transfer bytes')
    attempt_peak = _scope_integer(wave['peak_cpu_wave_estimate_bytes'], 'SAM approved attempt peak', positive=True)
    owned = _scope_integer(wave['assigned_cpu_wave_bytes'], 'SAM attempt phase credit', positive=True)
    evidence_persistent = _scope_integer(evidence_persistent_bytes, 'SAM persistent evidence bytes')
    evidence_contract = (None if evidence_contract_bytes is None else
        _scope_integer(evidence_contract_bytes, 'SAM live evidence contract bytes'))
    deferred = wave['defer_refill_until_consumed']
    if not isinstance(deferred, bool):
        raise ValueError('SAM transfer deferral must be boolean')
    required_peak = max(session, transfer) if deferred else capacity*session+transfer
    if (attempt_peak > owned or owned > int(record['assigned_cpu_wave_bytes'])
            or required_peak > attempt_peak or capacity > int(record.get('execution_slots', record['worker_count']))
            or wave_slots > int(record['execution_slots']) or capacity > wave_slots
            or transfer < 3*raw or deferred and capacity != 1):
        raise RuntimeError('SAM scheduler wave exceeds its original live per-attempt credit')
    # Deflate/ZIP overhead is conservatively bounded above uncompressed bytes.
    # Frame vectors contain int64 indices/float64 scores/bool status: 17 bytes.
    seed_npz = 2*pixels+64*1024
    packed_npz = 2*(frames*((pixels+7)//8)+17*frames+16)+64*1024
    manifest = 256*1024+1024*frames
    # One foreground seed creator/compressor can own temporary crop buffers.
    # Charging that scratch for every extra slot is conservative and explicit.
    per_job = 2*pixels+32*1024**2+seed_npz+packed_npz+manifest+descriptor_bytes
    lease, pool = profile._lease, profile._lease.pool
    identity = uuid.uuid4().hex
    with pool.condition:
        profile._validate_owner()
        if lease.tracker_scope_identity is not None:
            raise RuntimeError('SAM parent base-wave credit already belongs to an unsettled tracker scope')
        try:
            physical = max(0, int(lease.headroom_probe()))
        except Exception:
            physical = 0  # Unproven extra headroom keeps the formerly admitted path.
        profile._validate_owner()
        free_pool = max(0, int(pool.capacity)-int(pool.in_use))
        free_ram = max(0, physical-sam_parent_promised_bytes(pool)
                       -max(0, profile.base_requested_bytes-profile.base_charged_bytes))
        available = min(MAX_TRACKER_LOOKAHEAD_BYTES, free_pool, free_ram)
        lookahead = min(capacity, available//per_job) if not deferred else 0
        charged = lookahead*per_job
        pool.in_use += charged
        lease.scope_holds += 1  # Creator lifetime; a scheduler hold is separate.
        lease.tracker_scope_identity = identity
    try:
        limits = dict(schema='xta.sam_tracker_scope_admission/1', lease_id=lease.lease_id,
            max_in_flight=capacity, lookahead_jobs=lookahead, prepared_bank_bytes=charged,
            per_prepared_job_bytes=per_job, maximum_seed_npz_bytes=seed_npz,
            maximum_packed_packet_bytes=packed_npz, maximum_manifest_bytes=manifest,
            maximum_cache_payload_bytes=descriptor_bytes, maximum_seed_pixels=pixels,
            maximum_frame_count=frames, maximum_session_cpu_bytes=session,
            maximum_raw_mask_bytes=raw, consumer_transfer_bytes=transfer,
            active_cpu_bytes_limit=capacity*session, attempt_peak_bytes=attempt_peak,
            unused_attempt_wave_bytes=owned-attempt_peak,
            evidence_fixed_bytes=32*1024**2+evidence_persistent,
            evidence_contract_bytes=evidence_contract,
            defer_refill_until_consumed=deferred,
            raw_consumer_margin_is_not_prepared_bank_credit=True)
        admission = SamTrackerScopeAdmission(profile, limits, _seal=_TRACKER_ADMISSION_SEAL,
                                             _identity=identity)
        with _LIVE_LOCK:
            _LIVE_TRACKER_ADMISSIONS[admission._identity] = admission
        return admission
    except BaseException:
        with pool.condition:
            pool.in_use = max(0, int(pool.in_use)-charged)
            lease.scope_holds -= 1
            if lease.tracker_scope_identity == identity:
                lease.tracker_scope_identity = None
            _return_parent_credit_if_settled(lease)
            pool.condition.notify_all()
        raise


@contextmanager
def admit_sam_prepared_scope(profile, prepared, cache_ref, *, max_in_flight=None):
    """Bind a prepared plan's actual wave and image inventory to one tracker scope."""
    wave = getattr(prepared, 'cpu_wave_admission', None)
    if profile is None or not wave:
        yield None
        return
    groups = {str(group.group_id): group for group in prepared.groups}
    tiled = str(getattr(prepared, 'crop_mode', 'whole')) == 'tiled'
    work = tuple(prepared.tracker_jobs if tiled else prepared.runs)
    if not work:
        yield None
        return
    max_pixels = max_frames = 0
    for item in work:
        run = item.original_run if tiled else item
        bbox = item.tile.crop_bbox_yx if tiled else groups[str(run.group_id)].context_bbox_yx
        max_pixels = max(max_pixels, (int(bbox[2])-int(bbox[0]))*(int(bbox[3])-int(bbox[1])))
        max_frames = max(max_frames, len(run.expected_frames))
    descriptor_bytes = sam_cache_descriptor_bytes(cache_ref)
    from .sam_extrapolation_planning import SamExtrapolationPlan
    if isinstance(getattr(prepared, 'plan', None), SamExtrapolationPlan):
        # Tail contracts already have the established streamed 16-plane peak.
        contract_bytes = 0
    else:
        charges = [getattr(group, 'crop_contract', None) for group in prepared.groups
            if str(getattr(group, 'status', 'planned')) == 'planned']
        try:
            contract_bytes = max((_scope_integer(contract.get('charged_contract_bytes'),
                'SAM live evidence contract bytes', positive=True)
                for contract in charges if isinstance(contract, Mapping)), default=0)
            if any(not isinstance(contract, Mapping) for contract in charges):
                contract_bytes = None
            if not charges:
                contract_bytes = None
        except ValueError:
            contract_bytes = None
    admission = admit_sam_tracker_scope(profile, wave, max_seed_pixels=max_pixels,
        max_frame_count=max_frames, max_in_flight=max_in_flight, cache_payload_bytes=descriptor_bytes,
        evidence_persistent_bytes=int(getattr(prepared, 'tiled_assembly_bytes', 0)),
        evidence_contract_bytes=contract_bytes)
    with admission:
        yield admission


@dataclass(frozen=True)
class SamResourceProfile:
    scope_id: str
    base_requested_bytes: int
    base_charged_bytes: int
    reserved_extra_bytes: int
    pool_capacity_bytes: int
    physical_headroom_bytes: int
    worker_count: int
    base_allowance_bytes: int
    other_promised_bytes: int
    _lease: _LiveLease = field(compare=False, repr=False)

    @property
    def has_extra_credit(self):
        return self.reserved_extra_bytes > 0

    @property
    def execution_slots(self):
        return self._lease.execution_slots or self.worker_count

    def _validate_owner(self):
        with _LIVE_LOCK:
            minted = _LIVE_PROFILES.get(self._lease.lease_id) is self
        if not minted or not self._lease.active or self._lease.owner_thread != threading.get_ident():
            raise RuntimeError('SAM resource credit expired or belongs to another preparation thread')

    @property
    def assigned_contract_bytes(self):
        return self.reserved_extra_bytes if self.has_extra_credit else 256*1024**2

    @property
    def assigned_live_contract_bytes(self):
        return self.reserved_extra_bytes if self.has_extra_credit else 512*1024**2

    @property
    def assigned_topology_bytes(self):
        return self.reserved_extra_bytes if self.has_extra_credit else 256*1024**2

    @property
    def assigned_plane_bytes(self):
        return self.reserved_extra_bytes if self.has_extra_credit else 128*1024**2

    @property
    def assigned_session_cpu_bytes(self):
        return self.assigned_cpu_wave_bytes if self.has_extra_credit else 2*GIB

    @property
    def assigned_cpu_wave_base_bytes(self):
        # Pipeline explicitly identifies its fixed 4GiB allowance. Retain at
        # least 2GiB for bounded observation inventory, disk-cache pages and
        # legacy contract/reader work; known dense charges are never spent twice.
        nominal = self.nominal_cpu_wave_base_bytes
        residual = max(0, self.cpu_wave_physical_residual_bytes-self.reserved_extra_bytes)
        return min(nominal, residual)

    @property
    def nominal_cpu_wave_base_bytes(self):
        allowance = min(self.base_charged_bytes, self.base_allowance_bytes, 4*GIB)
        return max(0, allowance-2*GIB)

    @property
    def non_cpu_base_allowance_bytes(self):
        return min(self.base_charged_bytes, self.base_allowance_bytes)-self.nominal_cpu_wave_base_bytes

    @property
    def cpu_wave_physical_residual_bytes(self):
        # The legacy pool's floor/emergency lane is not physical RAM. Even its
        # identified base allowance grants CPU input bytes only after deducting
        # other promised credits and the fixed non-CPU phase allowance.
        return max(0, self.physical_headroom_bytes-self.other_promised_bytes
                   -self.non_cpu_base_allowance_bytes)

    @property
    def assigned_cpu_wave_bytes(self):
        return self.assigned_cpu_wave_base_bytes+self.reserved_extra_bytes

    def metadata(self):
        self._validate_owner()
        value = dict(schema=SCHEMA, scope_id=self.scope_id,
            status='admitted_extra_credit' if self.has_extra_credit else 'legacy_bounds_uncredited',
            base_requested_bytes=self.base_requested_bytes, base_charged_bytes=self.base_charged_bytes,
            reserved_extra_bytes=self.reserved_extra_bytes, pool_capacity_bytes=self.pool_capacity_bytes,
            physical_headroom_bytes=self.physical_headroom_bytes, worker_count=self.worker_count,
            execution_slots=self.execution_slots,
            assigned_contract_bytes=self.assigned_contract_bytes,
            assigned_live_contract_bytes=self.assigned_live_contract_bytes,
            assigned_topology_bytes=self.assigned_topology_bytes, assigned_plane_bytes=self.assigned_plane_bytes,
            assigned_session_cpu_bytes=self.assigned_session_cpu_bytes,
            assigned_cpu_wave_bytes=self.assigned_cpu_wave_bytes,
            assigned_cpu_wave_base_bytes=self.assigned_cpu_wave_base_bytes,
            assigned_cpu_wave_extra_bytes=self.reserved_extra_bytes,
            base_fixed_allowance_bytes=self.base_allowance_bytes,
            base_non_cpu_allowance_bytes=self.non_cpu_base_allowance_bytes,
            base_cpu_wave_nominal_bytes=self.nominal_cpu_wave_base_bytes,
            base_cpu_wave_physical_clamp_bytes=self.nominal_cpu_wave_base_bytes
                -self.assigned_cpu_wave_base_bytes,
            other_promised_bytes_at_admission=self.other_promised_bytes,
            cpu_wave_physical_residual_bytes=self.cpu_wave_physical_residual_bytes,
            base_known_dense_and_other_work_bytes=self.base_charged_bytes
                -min(self.base_charged_bytes, self.base_allowance_bytes),
            lease_id=self._lease.lease_id, resource_implementation_sha256=IMPLEMENTATION_SHA256,
            contract_and_quality_phases_share_credit=True,
            cuda_history_bound='not_claimed; original_full_history_retained; allocation_failure_is_infrastructure')
        # Live image debt belongs in telemetry, not this fixed assignment receipt.
        # Lease nonce and measured headroom prove runtime ownership, but do not
        # alter deterministic planning when effective budgets are identical.
        effective = {key: value[key] for key in ('schema', 'resource_implementation_sha256',
            'assigned_contract_bytes', 'assigned_live_contract_bytes', 'assigned_topology_bytes',
            'assigned_plane_bytes', 'assigned_session_cpu_bytes', 'assigned_cpu_wave_bytes')}
        value['effective_budgets'] = effective
        value['profile_id'] = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
        return value


def validate_live_sam_resource_profile(profile, *, require_extra=False):
    """Accept only live in-process extra credit, never a saved receipt mapping."""
    if not isinstance(profile, SamResourceProfile):
        raise TypeError('SAM enlarged resources require a live SamResourceProfile, not saved metadata')
    profile._validate_owner()
    if require_extra and not profile.has_extra_credit:
        raise RuntimeError('SAM profile has no additional leased memory; retain declared legacy bounds')
    return profile.metadata()


def sam_worker_count(owner, *, legacy=False):
    """Actual session slots; uncredited callers retain the physical-device wave."""
    physical = max(1, len(getattr(owner, 'device_ids', ())))
    count = _scope_integer(getattr(owner, 'worker_count', physical), 'SAM execution slots', positive=True)
    return min(count, physical) if legacy else count


def estimate_sam_session_cpu_bytes(frame_count, crop_pixels, *, image_side=1008):
    """Conservative known-input/raw-buffer estimate, never a CUDA guarantee.

    SDK loading overlaps its list and stacked float16 RGB tensor. The native
    PIL inputs and raw/stack/transfer buffers grow with crop pixels. The fixed
    allowance covers the last float64 normalization raster and small copying
    workspaces; allocator/model failures remain explicit infrastructure errors.
    """
    return cpu_session_bytes(frame_count, crop_pixels, image_side=image_side)['estimated_peak_bytes']


def cpu_session_bytes(frame_count, crop_pixels, *, image_side=1008):
    """Explain the admitted host-buffer estimate independently of GPU history."""
    frames, pixels, side = int(frame_count), int(crop_pixels), int(image_side)
    if min(frames, pixels, side) < 1:
        raise ValueError('SAM session CPU estimate requires positive geometry')
    normalized = 12*side*side*frames
    native = 10*pixels*frames
    copying = 64*side*side+16*pixels
    return dict(estimated_peak_bytes=normalized+native+copying,
        normalized_rgb_list_and_stack_bytes=normalized,
        native_pil_raw_packing_transfer_bytes=native,
        normalization_seed_and_copy_workspace_bytes=copying,
        frame_count=frames, crop_pixels=pixels, image_side=side,
        basis='conservative_known_CPU_buffers; allocator_failure_remains_infrastructure',
        cuda_history_bound='not_claimed; original_full_history_retained')


def cpu_wave_admission(max_session_bytes, max_raw_mask_bytes, owned_wave_bytes, worker_count):
    """Bound active input waves and one verified/yielded host transfer."""
    session, raw, owned, workers = map(int, (max_session_bytes, max_raw_mask_bytes,
                                           owned_wave_bytes, worker_count))
    if session <= 0:
        return dict(assigned_cpu_wave_bytes=max(0, owned), max_in_flight=0,
            execution_slots=max(1, workers),
            defer_refill_until_consumed=False, maximum_session_cpu_estimate_bytes=0,
            transfer_margin_bytes=0, peak_cpu_wave_estimate_bytes=0, status='no_tracking_jobs')
    transfer = 3*max(0, raw)  # previous yielded owner + unpackbits + bool destination
    capacity = min(max(1, workers), max(0, (owned-transfer)//session))
    deferred = capacity == 0
    if deferred:
        # End the sole worker's task before decode, then finish consuming and
        # dropping the decoded result before starting the next SDK input.
        if max(session, transfer) > owned:
            raise RuntimeError(f'SAM known CPU wave requires at least {max(session, transfer)} bytes; '
                               f'owned phase credit is {owned}; no model was admitted or interval truncated')
        capacity = 1
        peak = max(session, transfer)
    else:
        peak = capacity*session+transfer
    return dict(assigned_cpu_wave_bytes=owned, max_in_flight=capacity,
        execution_slots=max(1, workers),
        defer_refill_until_consumed=deferred, maximum_session_cpu_estimate_bytes=session,
        maximum_raw_mask_bytes=raw, transfer_margin_bytes=transfer,
        peak_cpu_wave_estimate_bytes=peak,
        status='serial_transfer_barrier' if deferred else 'bounded_overlap')


@contextmanager
def admit_sam_parent_resources(pool, base_bytes, scope_id, *, worker_count=1,
                               base_allowance_bytes=0, headroom_probe=None, execution_slots=None,
                               cancel_event=None):
    """Resolve late and reserve base+SAM credit in one atomic acquisition.

    The emergency base lane retains its legacy behavior and never grants extra
    credit. Waiting owns no part of this reservation, so no nested lease can
    wait behind itself. Other promised pool credits are conservatively deducted
    from physical headroom even if some are already reflected in RSS.
    """
    base = max(1, int(base_bytes))
    workers = max(1, int(worker_count))
    slots = workers if execution_slots is None else _scope_integer(
        execution_slots, 'SAM execution slots', positive=True)
    if not workers <= slots <= 2*workers:
        raise ValueError('SAM execution slots must represent one or two sessions per physical GPU')
    probe = headroom_probe or physical_sam_headroom
    minimum_extra = max(MIN_PRODUCTION_EXTRA_BYTES, workers*2*GIB)
    requested_at, wait_seconds, wait_count = time.monotonic(), 0., 0
    _trace_parent_memory('sam_parent_memory_requested', scope_id=str(scope_id),
        base_requested_bytes=base, minimum_extra_bytes=minimum_extra,
        pool_in_use_bytes=pool.in_use, pool_capacity_bytes=pool.capacity,
        **sam_image_staging_snapshot(pool))
    with pool.condition:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError('SAM parent resource admission cancelled')
            capacity = max(1, int(pool.capacity))
            base_charge = min(base, capacity)
            if (int(pool.in_use) > 0 and int(pool.in_use)+base_charge > capacity
                    or base > capacity and sam_parent_promised_bytes(pool) > 0):
                waiting_at = time.monotonic()
                pool.condition.wait(timeout=.1 if cancel_event is not None else None)
                wait_seconds += time.monotonic()-waiting_at
                wait_count += 1
                continue
            physical = max(0, int(probe()))
            other_promised = sam_parent_promised_bytes(pool)
            free_pool = max(0, capacity-int(pool.in_use)-base_charge)
            unpromised_ram = max(0, physical-base-other_promised)
            fixed_allowance = min(base_charge, max(0, int(base_allowance_bytes)))
            nominal_cpu_wave = max(0, min(fixed_allowance, 4*GIB)-2*GIB)
            # Concurrent incumbents must not force legacy planning bounds when
            # the isolated parent can afford them. This includes physical RAM
            # promised to incumbents, and the identified base CPU wave. Wait
            # without partial credit; genuine isolated shortages still fall back.
            base_blocked = (nominal_cpu_wave > 0 and physical >= fixed_allowance
                            and physical-other_promised < fixed_allowance)
            extra_blocked = (minimum_extra <= MAX_PRODUCTION_EXTRA_BYTES
                             and base+minimum_extra <= capacity
                             and max(0, physical-base)//2 >= minimum_extra
                             and (free_pool < minimum_extra or unpromised_ram//2 < minimum_extra))
            if other_promised > 0 and (base_blocked or extra_blocked):
                waiting_at = time.monotonic()
                pool.condition.wait(timeout=.1 if cancel_event is not None else None)
                wait_seconds += time.monotonic()-waiting_at
                wait_count += 1
                continue
            extra = min(MAX_PRODUCTION_EXTRA_BYTES, free_pool, unpromised_ram//2)
            break
        # Enlarging contracts must never tighten a previously admissible SDK
        # CPU session. Decline a modest grant rather than claim 2GiB on an
        # unowned smaller per-worker share or introduce a new frame cutoff.
        if base > capacity or extra < minimum_extra:
            extra = 0
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError('SAM parent resource admission cancelled')
        charged = base_charge+extra
        other_promised = sam_parent_promised_bytes(pool)
        pool.in_use += charged
        oversize_requested = base if base > capacity else 0
        if oversize_requested:
            pool.oversize_requested_bytes = oversize_requested
        lease = _LiveLease(uuid.uuid4().hex, threading.get_ident(), pool=pool,
                           charged_bytes=charged, headroom_probe=probe, execution_slots=slots,
                           scope_id=str(scope_id), admitted_monotonic=time.monotonic(),
                           oversize_requested_bytes=oversize_requested)
        profile = SamResourceProfile(str(scope_id), base, base_charge, extra, capacity, physical,
                                     workers, max(0, int(base_allowance_bytes)), other_promised, lease)
        with _LIVE_LOCK:
            _LIVE_PROFILES[lease.lease_id] = profile
    try:
        _trace_parent_memory('sam_parent_memory_admitted', scope_id=lease.scope_id,
            lease_id=lease.lease_id, base_charged_bytes=base_charge, extra_charged_bytes=extra,
            charged_bytes=charged, pool_in_use_bytes=pool.in_use, pool_capacity_bytes=pool.capacity,
            admission_seconds=time.monotonic()-requested_at,
            condition_wait_seconds=wait_seconds, condition_wait_count=wait_count,
            **sam_image_staging_snapshot(pool))
        yield profile
    finally:
        with pool.condition:
            lease.active = False
            lease.owner_closed = True
            with _LIVE_LOCK:
                _LIVE_PROFILES.pop(lease.lease_id, None)
            try:
                _trace_parent_memory('sam_parent_memory_owner_expired', scope_id=lease.scope_id,
                    lease_id=lease.lease_id, scope_holds=lease.scope_holds,
                    lifetime_seconds=max(0., time.monotonic()-lease.admitted_monotonic),
                    **sam_image_staging_snapshot(pool))
            finally:
                _return_parent_credit_if_settled(lease)


def run_admitted_sam_call(pool, base_bytes, scope_id, sam_context, function, kwargs,
                          base_allowance_bytes=4*GIB):
    """Tile/full-frame callback wrapper without changing assembly signatures."""
    sam_context.prepare_runtime(pool)
    workers = len(getattr(sam_context, 'device_ids', ())) or 1
    with admit_sam_parent_resources(pool, base_bytes, scope_id, worker_count=workers,
            execution_slots=sam_worker_count(sam_context), base_allowance_bytes=base_allowance_bytes,
            cancel_event=getattr(sam_context, '_cancel', None)) as profile:
        with sam_context.resource_scope(profile):
            return function(**kwargs)


__all__ = ['SCHEMA', 'IMPLEMENTATION_SHA256', 'SamResourceProfile',
           'sam_parent_promised_bytes', 'sam_image_staging_snapshot',
           'physical_sam_headroom', 'validate_live_sam_resource_profile',
           'estimate_sam_session_cpu_bytes', 'cpu_session_bytes', 'cpu_wave_admission',
           'admit_sam_parent_resources', 'run_admitted_sam_call',
           'SamTrackerScopeAdmission', 'admit_sam_tracker_scope',
           'validate_sam_tracker_scope_admission', 'admit_sam_prepared_scope',
           'sam_cache_descriptor_bytes', 'sam_worker_count']
