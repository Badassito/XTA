"""Live, atomic parent admission for production SAM CPU workspaces.

Pool capacity is an accounting limit, not evidence of physical RAM. Larger
family contracts are enabled only by additional credit reserved together with
the parent's existing work; serialized profiles never authorize replay memory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

GIB = 1024**3
SCHEMA = 'xta.sam_live_resources/1'
IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAX_PRODUCTION_EXTRA_BYTES = 16*GIB
MIN_PRODUCTION_EXTRA_BYTES = 512*1024**2
_LIVE_PROFILES = {}
_LIVE_LOCK = threading.RLock()


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
        defer_refill_until_consumed=deferred, maximum_session_cpu_estimate_bytes=session,
        maximum_raw_mask_bytes=raw, transfer_margin_bytes=transfer,
        peak_cpu_wave_estimate_bytes=peak,
        status='serial_transfer_barrier' if deferred else 'bounded_overlap')


@contextmanager
def admit_sam_parent_resources(pool, base_bytes, scope_id, *, worker_count=1,
                               base_allowance_bytes=0, headroom_probe=None):
    """Resolve late and reserve base+SAM credit in one atomic acquisition.

    The emergency base lane retains its legacy behavior and never grants extra
    credit. Waiting owns no part of this reservation, so no nested lease can
    wait behind itself. Other promised pool credits are conservatively deducted
    from physical headroom even if some are already reflected in RSS.
    """
    base = max(1, int(base_bytes))
    workers = max(1, int(worker_count))
    probe = headroom_probe or physical_sam_headroom
    with pool.condition:
        while True:
            capacity = max(1, int(pool.capacity))
            base_charge = min(base, capacity)
            if int(pool.in_use) == 0 or int(pool.in_use)+base_charge <= capacity:
                break
            pool.condition.wait()
        physical = max(0, int(probe()))
        free_pool = max(0, capacity-int(pool.in_use)-base_charge)
        unpromised_ram = max(0, physical-base-int(pool.in_use))
        extra = min(MAX_PRODUCTION_EXTRA_BYTES, free_pool, unpromised_ram//2)
        # Enlarging contracts must never tighten a previously admissible SDK
        # CPU session. Decline a modest grant rather than claim 2GiB on an
        # unowned smaller per-worker share or introduce a new frame cutoff.
        minimum_extra = max(MIN_PRODUCTION_EXTRA_BYTES, workers*2*GIB)
        if base > capacity or extra < minimum_extra:
            extra = 0
        charged = base_charge+extra
        other_promised = int(pool.in_use)
        pool.in_use += charged
        lease = _LiveLease(uuid.uuid4().hex, threading.get_ident())
        profile = SamResourceProfile(str(scope_id), base, base_charge, extra, capacity, physical,
                                     workers, max(0, int(base_allowance_bytes)), other_promised, lease)
        with _LIVE_LOCK:
            _LIVE_PROFILES[lease.lease_id] = profile
    try:
        yield profile
    finally:
        with pool.condition:
            lease.active = False
            with _LIVE_LOCK:
                _LIVE_PROFILES.pop(lease.lease_id, None)
            pool.in_use = max(0, int(pool.in_use)-charged)
            pool.condition.notify_all()


def run_admitted_sam_call(pool, base_bytes, scope_id, sam_context, function, kwargs,
                          base_allowance_bytes=4*GIB):
    """Tile/full-frame callback wrapper without changing assembly signatures."""
    workers = len(getattr(sam_context, 'device_ids', ())) or 1
    with admit_sam_parent_resources(pool, base_bytes, scope_id, worker_count=workers,
            base_allowance_bytes=base_allowance_bytes) as profile:
        with sam_context.resource_scope(profile):
            return function(**kwargs)


__all__ = ['SCHEMA', 'IMPLEMENTATION_SHA256', 'SamResourceProfile',
           'physical_sam_headroom', 'validate_live_sam_resource_profile',
           'estimate_sam_session_cpu_bytes', 'cpu_session_bytes', 'cpu_wave_admission',
           'admit_sam_parent_resources', 'run_admitted_sam_call']
