"""Release completed parent canvases while shared SAM awaits detector retirement.

Checkpoints live on ordinary disk, retain exact numeric bytes, and own no dense
mapping while deferred. Resumed preparation receives a fresh dense reservation.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from concurrent.futures import CancelledError
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np

from .interpolation import _DirectUnionBackingLease, _ByteAdmissionPool
from .runtime import (path_is_memory_backed, allocate_workspace_array,
                      _memfd_owner_key_from_array, _memfd_backing_path_from_array,
                      close_memmap_array_without_flush, workspace_anon_cap_bytes, runtime_telemetry,
                      memfd_ram_headroom, capture_memfd_owner_proofs)
from .publication_memory import publication_ram_headroom
from .view_prepare import scratch_unlink_path_for_memmap, classify_dense_ram_backing

STREAM_BYTES = 8 * 1024**2
CODEC_CONTROL_BYTES = 64 * 1024**2
RAM_RESERVE_BYTES = 16 * 1024**3


def _ram_fits(required_bytes, active_bytes=0, dense_limit=None, receipt=None, promises=()):
    """Use actual physical/cgroup headroom, never swap, for another live owner."""
    required, active = max(0, int(required_bytes)), max(0, int(active_bytes))
    total = required + active
    cap = int(workspace_anon_cap_bytes())
    headroom, resident, proof = memfd_ram_headroom(promises, publication_ram_headroom)
    resident = min(active, resident)
    future = active-resident
    physical_required = required+future+RAM_RESERVE_BYTES
    reason = ('empty_request' if not required else
              'dense_limit' if dense_limit is not None and total > int(dense_limit) else
              'anonymous_workspace_cap' if cap > 0 and total > cap else
              'physical_headroom' if headroom < physical_required else None)
    if receipt is not None:
        receipt.update(total_checked_bytes=total, anonymous_cap_bytes=cap,
            physical_headroom_bytes=headroom, physical_reserve_bytes=RAM_RESERVE_BYTES,
            proven_resident_ram_bytes=resident, future_ram_commitments_bytes=future,
            physical_required_bytes=physical_required, reason=reason, **proof)
    return reason is None


def _codec_workspace_bytes(shape):
    # The mask decoder owns at most one bbox plane; the incremental encoder
    # owns <=1MiB row slabs plus its per-frame index. Confidence cells are 128².
    # An unusually wide single row exceeds the writer's 1MiB slab target;
    # charge overlapping row normalization/packing and confidence vectors too.
    return (CODEC_CONTROL_BYTES + int(shape[1]) * int(shape[2])
            + 4 * int(shape[2]) + int(shape[0]) * 64)


def _parent_condition(task):
    return getattr(getattr(task, 'admission', None), 'condition', nullcontext())


def _dense_startup_budget_locked(task, additional_bytes=0, *, reset_lease=None):
    context = getattr(task, 'sam_context', None)
    if context is None or not getattr(context, 'progressive_startup', False):
        return None
    # Forecasts are not owned grants. Mixed/empty parents must still retire
    # before startup is needed; the later initial grant sees their new debt.
    if (not getattr(context, '_startup_fleet_funded', False)
            or not int(getattr(context, '_startup_fleet_credit_bytes', 0))):
        return None
    admission = getattr(task, 'admission', None)
    snapshot = getattr(context, 'startup_budget_snapshot_locked', None)
    if (snapshot is None or getattr(admission, 'condition', None) is None
            or getattr(context, '_startup_pool', admission) is not admission):
        raise RuntimeError('SAM dense admission requires its configured startup condition')
    check = getattr(context, 'check_startup', None)
    if check is not None:
        check()
    if reset_lease is not None:
        additional_bytes = dense_proof_reset_bytes(reset_lease)
    return snapshot(additional_pending_bytes=max(0, int(additional_bytes)))


def _dense_startup_fits(budget):
    return (budget is None or not int(budget.get('remaining_startup_bytes', 0))
            or int(budget['physical_headroom_bytes']) >= int(budget['required_host_bytes']))


def dense_proof_reset_bytes(lease):
    """Bound credit removed by a proof reset without a page-fault sampling race."""
    promised = max(0, int(lease.ram_commitment_bytes))
    disk_to_ram = max(0, int(lease.nbytes)-promised)
    # Occupancy can grow between a first hint and startup's resident probe.
    # Charge every old proof's possible credit, including stale proofs (which
    # may overcharge). No array/RSS assumption reduces this upper bound.
    try:
        sizes = sum(max(0, int(proof[4])) for proof in lease.memfd_owner_proofs)
    except (IndexError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError('SAM dense residency proof is malformed') from error
    return disk_to_ram+min(promised, sizes)


def fence_dense_proof_reset(task):
    """Fund new dense debt before taking base/GPU credit or dropping proofs."""
    lease = task.backing_lease
    with _parent_condition(task):
        while True:
            task.check_cancelled()
            budget = _dense_startup_budget_locked(task, reset_lease=lease)
            if _dense_startup_fits(budget):
                lease.memfd_owner_proofs = ()
                lease.ram_backed = None
                return
            task.admission.condition.wait(.05)


@contextmanager
def _restore_allocation_guard(task, receipt):
    """Serialize backing birth with startup grants, never with decoding."""
    context = getattr(task, 'sam_context', None)
    if context is None or not getattr(context, 'progressive_startup', False):
        yield True
        return
    snapshot = getattr(context, 'startup_budget_snapshot_locked', None)
    admission = getattr(task, 'admission', None)
    condition = getattr(admission, 'condition', None)
    if (snapshot is None or condition is None
            or getattr(context, '_startup_pool', admission) is not admission):
        receipt['startup_ram_admitted'] = False
        yield False  # Missing accounting permission selects ordinary disk.
        return
    with condition:
        check = getattr(context, 'check_startup', None)
        if check is not None:
            check()
        budget = snapshot()
        headroom, required = (max(0, int(budget[name])) for name in
                              ('physical_headroom_bytes', 'required_host_bytes'))
        admitted = headroom >= required
        receipt.update(startup_ram_admitted=admitted,
            startup_headroom_bytes=headroom, startup_required_host_bytes=required)
        yield admitted


@contextmanager
def _codec_reservation(task, amount, *, try_only=False):
    admission = getattr(task, 'admission', None)
    if try_only and isinstance(admission, _ByteAdmissionPool):
        # The existing pool has no try API. Use its identical condition/credit
        # ledger once, without waiting or entering its emergency oversize lane.
        requested = int(amount)
        with admission.condition:
            admitted = requested <= admission.capacity and admission.in_use + requested <= admission.capacity
            if admitted:
                admission.in_use += requested
        try:
            yield admitted
        finally:
            if admitted:
                with admission.condition:
                    admission.in_use -= requested
                    admission.condition.notify_all()
    else:
        with (admission.reserve(int(amount), f'{task.model_name}/{task.view.name}/checkpoint-codec')
              if admission is not None else nullcontext()):
            yield True


def select_checkpoint_root(temp_dir, output_dir):
    """Select one run-owned disk directory, including tmpfs scratch fallback."""
    for parent in (Path(temp_dir), Path(output_dir)):
        parent.mkdir(parents=True, exist_ok=True)
        if not path_is_memory_backed(parent):
            root = parent / 'sam_parent_checkpoints' / uuid.uuid4().hex
            root.mkdir(parents=True)
            return root
    raise RuntimeError('Shared-GPU SAM requires disk-backed --temp or --output for bounded parent checkpoints; both destinations are memory-backed')


def _stat_identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


@dataclass(frozen=True)
class ArrayCheckpoint:
    path: Path
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    identity: tuple[int, ...]
    reused: bool = False
    encoding: str = 'raw'
    stored_bytes: int | None = None
    files: tuple = ()
    foreground_value: int = 1

    def open(self, *, prefer_ram=False, stop=None, metrics=None,
             allocation_guard=None, allocation_callback=None):
        if metrics is not None:
            metrics.update(allocation_seconds=0., allocation_bytes=0, decoded_bytes=0)
        if path_is_memory_backed(self.path) or _stat_identity(self.path) != self.identity:
            raise RuntimeError('SAM parent checkpoint backing changed before resume')
        if self.encoding == 'raw':
            started = time.perf_counter()
            try:
                return np.memmap(self.path, mode='r+', dtype=np.dtype(self.dtype), shape=self.shape)
            finally:
                if metrics is not None:
                    metrics['allocation_seconds'] = time.perf_counter()-started
        for name, identity in self.files:
            if _stat_identity(self.path / name) != identity:
                raise RuntimeError('SAM compact checkpoint backing changed before resume')
        destination = self.path.with_name(self.path.name + '.restored.dat')
        started = time.perf_counter()
        output = None
        try:
            with (allocation_guard() if allocation_guard is not None else nullcontext(True)) as admitted:
                output = allocate_workspace_array(self.shape, np.dtype(self.dtype), destination,
                    'SAM compact parent restore', prefer_memory=False,
                    prefer_memfd=bool(prefer_ram and admitted),
                    reserve_bytes=RAM_RESERVE_BYTES, initialize_zero=True)
                if allocation_callback is not None:
                    allocation_callback(output)
            if metrics is not None:
                metrics['allocation_bytes'] = self.nbytes
        except BaseException:
            if output is not None:
                close_memmap_array_without_flush(output, unlink_path=destination)
                if isinstance(output, np.memmap) and not output._mmap.closed:
                    output._mmap.close()
            raise
        finally:
            if metrics is not None:
                metrics['allocation_seconds'] = time.perf_counter()-started
        decoded_bytes = 0
        try:
            # The allocator creates fresh zeroed backing. Only decoded crops
            # touch pages; absent score cells and empty frames remain zero.
            if self.encoding == 'packed_mask':
                from .interpolation import RawBBoxMaskStore
                store = RawBBoxMaskStore.open(self.path, mmap_payload=True)
                try:
                    if tuple(store.shape) != tuple(self.shape):
                        raise RuntimeError('SAM mask checkpoint grid differs from its native owner')
                    for z in range(self.shape[0]):
                        if stop is not None and stop.is_set():
                            raise RuntimeError('SAM compact parent restore cancelled')
                        crop = store.decode_slice_crop(z)
                        if crop is not None:
                            y0,x0,y1,x1,values = crop
                            if self.foreground_value != 1:
                                np.multiply(values, self.foreground_value, out=values)
                            output[z,y0:y1,x0:x1] = values
                            decoded_bytes += int(values.nbytes)
                            del values
                        del crop  # Keep only one decoded bbox plane live.
                finally:
                    store.close()
            elif self.encoding == 'score_blocks':
                from .confidence_evidence import ConfidenceEvidenceRef
                reference = ConfidenceEvidenceRef.open(self.path)
                if tuple(reference.storage_shape) != tuple(self.shape):
                    raise RuntimeError('SAM score checkpoint grid differs from its native owner')
                with reference.native_reader() as reader:
                    for z in range(self.shape[0]):
                        if stop is not None and stop.is_set():
                            raise RuntimeError('SAM compact parent restore cancelled')
                        for y0,y1,x0,x1,crop in reader.iter_crops(z):
                            output[z,y0:y1,x0:x1] = crop
                            decoded_bytes += int(crop.nbytes)
            else:
                raise ValueError('Unknown SAM checkpoint encoding')
            return output
        except BaseException:
            close_memmap_array_without_flush(output, unlink_path=destination)
            if isinstance(output, np.memmap) and not output._mmap.closed:
                output._mmap.close()
            raise
        finally:
            if metrics is not None:
                metrics['decoded_bytes'] = decoded_bytes

    @property
    def physical_bytes(self):
        return self.nbytes if self.stored_bytes is None else self.stored_bytes


@dataclass(frozen=True)
class ParentCheckpoint:
    mask: ArrayCheckpoint | None
    confidence: ArrayCheckpoint | None
    required_bytes: int
    written_bytes: int
    reused_bytes: int
    seconds: float
    shadow_path: Path | None = None
    owned_dir: Path | None = None
    codec_fallback_bytes: int = 0


def _validate_owner(array):
    value = np.asanyarray(array)
    if value.ndim != 3 or value.dtype.hasobject or value.dtype.kind not in 'buifc' or not value.flags.c_contiguous:
        raise ValueError('SAM parent checkpoint requires a contiguous numeric TYX owner')
    if isinstance(value, np.memmap):
        if value.base is not value._mmap or int(value.offset) != 0:
            raise ValueError('SAM parent checkpoint cannot release a derived/shared array alias')
    elif not value.flags.owndata:
        raise ValueError('SAM parent checkpoint cannot release a derived/shared array alias')
    return value


def _reusable_disk_owner(array, declared_path, source_root):
    if not isinstance(array, np.memmap) or declared_path is None:
        return None
    path = scratch_unlink_path_for_memmap(array, Path(declared_path))
    if path is None or str(path).startswith('/proc/'):
        return None
    if (not path.resolve().is_relative_to(Path(source_root).resolve()) or path_is_memory_backed(path)
            or path.stat().st_size != array.nbytes):
        return None
    return path


def _write_all(stream, data):
    """Keep an exact checkpoint when unbuffered disk writes are short."""
    chunk = memoryview(data)
    while chunk:
        written = stream.write(chunk)
        if not written:
            raise OSError('SAM parent checkpoint write made no progress')
        chunk = chunk[written:]


def _uniform_foreground_value(array, stop):
    flat = array.reshape(-1)
    value = 0
    for offset in range(0, flat.size, STREAM_BYTES):
        if stop.is_set():
            raise RuntimeError('SAM parent checkpoint cancelled')
        part = flat[offset:offset + STREAM_BYTES]
        maximum = int(part.max(initial=0))
        if not maximum:
            continue
        if value and maximum != value or np.any((part != 0) & (part != maximum)):
            return None
        value = maximum
    return value or 1


def _snapshot_array(array, declared_path, destination, source_root, stop, *, mask=False,
                    allow_compact=True, capture_workers=1, capture_workspace_bytes=None):
    array = _validate_owner(array)
    reused = _reusable_disk_owner(array, declared_path, source_root)
    if reused is not None:
        array.flush()
        with reused.open('r+b') as stream:
            os.fsync(stream.fileno())
        path = reused
    else:
        ram_owner = (not isinstance(array, np.memmap)
                     or _memfd_owner_key_from_array(array) is not None
                     or path_is_memory_backed(Path(declared_path or array.filename)))
        # Small arrays cost less as raw bytes than the existing format headers.
        compact = allow_compact and ram_owner and array.dtype == np.uint8 and array.nbytes > 4096 + array.shape[0]*64
        foreground = _uniform_foreground_value(array, stop) if compact and mask else 1
        if compact and foreground is not None:
            path = destination.with_suffix('.cvol' if mask else '.scores')
            if shutil.disk_usage(destination.parent).free < array.nbytes + STREAM_BYTES:
                raise RuntimeError('Insufficient disk space for bounded SAM parent checkpoint')
            if mask:
                from .interpolation import IncrementalRawBBoxMaskStoreWriter, INTERNAL_PACKED_CVOL_FORMAT
                writer = IncrementalRawBBoxMaskStoreWriter(shape=tuple(array.shape), store_dir=path,
                    format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='SAM compact parent checkpoint')
                try:
                    for z in range(array.shape[0]):
                        if stop.is_set():
                            raise RuntimeError('SAM parent checkpoint cancelled')
                        writer.consume(z, array[z:z+1])
                    writer.finalize()
                except BaseException as error:
                    writer.abort(error)
                    writer.discard()
                    raise
            else:
                from .confidence_storage import write_blocks
                from .confidence_capture import plan_confidence_capture, confidence_capture_resources
                from .confidence_evidence import _MaskedNativeScoreReader
                def read(z):
                    if stop.is_set():
                        raise RuntimeError('SAM parent checkpoint cancelled')
                    return array[z]
                capture_plan = None
                if capture_workspace_bytes is not None:
                    try:
                        capture_plan = plan_confidence_capture(array.shape, capture_workers,
                            workspace_bytes=capture_workspace_bytes)
                    except MemoryError:
                        pass  # The already-funded serial codec needs no frame window.
                metrics = {}
                with confidence_capture_resources(capture_plan):
                    reader = read if capture_plan is None else _MaskedNativeScoreReader(
                        array, array, np.ones(array.shape[0], dtype=bool),
                        np.broadcast_to(np.array([0, array.shape[1], 0, array.shape[2]],
                                                 dtype=np.int64), (array.shape[0], 4)), stop=stop)
                    # Borrow scores as their own mask: all nonzero uint8 values,
                    # including scores outside detector support, remain exact.
                    write_blocks(path, tuple(array.shape), reader, layer_key='sam-parent-checkpoint',
                        model_name='checkpoint', provenance={'sam_parent_checkpoint': True},
                        coordinate_space='native_view_processing', source_shape=tuple(array.shape),
                        metrics=metrics)
                runtime_telemetry().add_scheduler_counter('sam_parent_checkpoint.confidence.' +
                    ('compiled' if capture_plan is not None else 'serial'))
                runtime_telemetry().trace_event('sam_parent_checkpoint_confidence',
                    **metrics)
            files = []
            for file in sorted(path.iterdir()):
                with file.open('r+b') as stream:
                    os.fsync(stream.fileno())
                files.append((file.name, _stat_identity(file)))
            stored = sum(identity[2] for _, identity in files)
            return ArrayCheckpoint(path, tuple(array.shape), array.dtype.str, int(array.nbytes),
                _stat_identity(path), encoding='packed_mask' if mask else 'score_blocks',
                stored_bytes=stored, files=tuple(files), foreground_value=foreground)
        if shutil.disk_usage(destination.parent).free < array.nbytes + STREAM_BYTES:
            raise RuntimeError('Insufficient disk space for bounded SAM parent checkpoint')
        flat = array.reshape(-1).view(np.uint8)
        with destination.open('xb', buffering=0) as stream:
            for offset in range(0, flat.size, STREAM_BYTES):
                if stop.is_set():
                    raise RuntimeError('SAM parent checkpoint cancelled')
                _write_all(stream, flat[offset:offset + STREAM_BYTES])
            os.fsync(stream.fileno())
        del flat
        path = destination
    return ArrayCheckpoint(path, tuple(array.shape), array.dtype.str, int(array.nbytes), _stat_identity(path), reused is not None)


def _close_sources(task, reused_paths=(), *, preserve=False):
    """Clear all task references and close root maps before returning credit."""
    for name, path_name in (('union_mm', 'union_path'), ('confmap_mm', 'confmap_path')):
        array = getattr(task, name)
        if array is None:
            continue
        # An invalid derived owner belongs to its external/root caller. Do not
        # close or schedule unlink of that caller's mapping on validation error.
        if ((isinstance(array, np.memmap) and (array.base is not array._mmap or int(array.offset) != 0))
                or (not isinstance(array, np.memmap) and not np.asanyarray(array).flags.owndata)):
            continue
        setattr(task, name, None)
        path = getattr(task, path_name)
        unlink = None if preserve or path is None or Path(path) in reused_paths else scratch_unlink_path_for_memmap(array, Path(path))
        task.close_dense(array, unlink_path=unlink)
        # This seam accepts root owners only, and no worker may still write a
        # completed parent. There are no legitimate NumPy aliases to protect.
        if isinstance(array, np.memmap) and array.base is array._mmap and not array._mmap.closed:
            array._mmap.close()
        del array


def _close_prepared_sources(task, *, preserve=False):
    """CPU/SAM preparation may retain derived views; use normal retirement."""
    for name,path_name in (('union_mm','union_path'),('confmap_mm','confmap_path')):
        array = getattr(task,name)
        if array is not None:
            setattr(task,name,None)
            path = getattr(task,path_name)
            task.close_dense(array, unlink_path=(None if preserve or path is None
                else scratch_unlink_path_for_memmap(array,path)))


def checkpoint_parent(task, root, source_root, required_bytes, stop):
    if task.union_mm is not None:
        _validate_owner(task.union_mm)
        if task.confmap_mm is not None:
            _validate_owner(task.confmap_mm)
    started = time.perf_counter()
    owned = Path(root) / uuid.uuid4().hex
    owned.mkdir()
    snapshots = []
    try:
        if task.union_mm is None:
            declared = Path(task.d1_shadow_path)
            source = declared.resolve()
            base = Path(source_root).resolve()
            if source == base or not source.is_relative_to(base) or declared.is_symlink():
                raise ValueError('SAM cannot relocate an unowned memory-backed D1 shadow')
            destination = owned / 'shadow'
            files = tuple(source.rglob('*'))
            if any(path.is_symlink() for path in files):
                raise ValueError('SAM D1 shadow checkpoint cannot follow symlinks')
            needed = sum(path.stat().st_size for path in files if path.is_file())
            if shutil.disk_usage(owned).free < needed + STREAM_BYTES:
                raise RuntimeError('Insufficient disk space for bounded SAM D1 shadow checkpoint')
            destination.mkdir()
            for path in files:
                target = destination / path.relative_to(source)
                if path.is_dir():
                    target.mkdir(exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as reader, target.open('xb', buffering=0) as writer:
                    while chunk := reader.read(STREAM_BYTES):
                        if stop.is_set():
                            raise RuntimeError('SAM parent checkpoint cancelled')
                        _write_all(writer, chunk)
                    os.fsync(writer.fileno())
            if stop.is_set():
                raise RuntimeError('SAM parent checkpoint cancelled')
            # The shadow is a private completed native store, not the already
            # published source-grid component. Debug ownership moves to disk.
            shutil.rmtree(source)
            return ParentCheckpoint(None, None, int(required_bytes), needed, 0,
                time.perf_counter()-started, destination, owned)
        workspace = max(_codec_workspace_bytes(task.union_mm.shape),
                        0 if task.confmap_mm is None else _codec_workspace_bytes(task.confmap_mm.shape))
        with _codec_reservation(task, workspace, try_only=True) as compact_credit:
            for name, path_name, filename in (('union_mm', 'union_path', 'mask.dat'),
                                              ('confmap_mm', 'confmap_path', 'confidence.dat')):
                if stop.is_set():
                    raise RuntimeError('SAM parent checkpoint cancelled')
                array = getattr(task, name)
                snapshots.append(None if array is None else _snapshot_array(array,
                    getattr(task, path_name), owned / filename, source_root, stop,
                    mask=name=='union_mm', allow_compact=compact_credit,
                    capture_workers=int(getattr(task, 'slice_workers', 1)),
                    capture_workspace_bytes=workspace if compact_credit else None))
                del array
        if stop.is_set():
            raise RuntimeError('SAM parent checkpoint cancelled')
        reused = tuple(value.path for value in snapshots if value is not None and value.reused)
        _close_sources(task, reused)
        logical = sum(value.nbytes for value in snapshots if value is not None)
        written = sum(value.physical_bytes for value in snapshots if value is not None and not value.reused)
        reused_bytes = sum(value.nbytes for value in snapshots if value is not None and value.reused)
        return ParentCheckpoint(*snapshots, max(int(required_bytes), logical), written,
            reused_bytes, time.perf_counter()-started, owned_dir=owned,
            codec_fallback_bytes=0 if compact_credit else
                sum(value.nbytes for value in snapshots if value is not None and not value.reused))
    except BaseException:
        _close_sources(task, preserve=bool(task.keep_temp_artifacts))
        shutil.rmtree(owned)
        raise


class DeferredSamParentQueue:
    """Scheduler-thread queue with a bounded RAM backlog and no SAM waits."""

    def __init__(self, *, temp_dir, output_dir, checkpoint_executor, prepare_executor,
                 leases, dense_limit, ready, keep_temp=False, bounded_parent_keys=()):
        self.root = select_checkpoint_root(temp_dir, output_dir)
        self.source_root = Path(temp_dir)
        self.checkpoint_executor = checkpoint_executor
        self.prepare_executor = prepare_executor
        self.leases = leases
        self._dense_limit = dense_limit
        self.ready = ready
        self.keep_temp = bool(keep_temp)
        self.bounded_parent_keys = bounded_parent_keys
        self.stop = threading.Event()
        self.checkpoint_futures = {}
        self.restore_futures = {}
        self.deferred = {}
        self.transferred_roots = set()
        self.count = self.written_bytes = self.reused_bytes = self.resumed_count = 0
        self.io_seconds = 0.
        self.compact_bytes = self.compact_logical_bytes = self.restore_raw_bytes = 0
        self.restore_startup_fallback_bytes = 0
        self.codec_fallback_bytes = 0
        self.ram_first_owners = {}
        self.ram_first_grants = self.ram_first_denials = self.ram_first_releases = 0
        self.last_ram_denial = None
        self.raw_reuse_submissions = 0

    @property
    def pending(self):
        return bool(self.checkpoint_futures or self.deferred)

    @property
    def dense_limit(self):
        return int(self._dense_limit() if callable(self._dense_limit) else self._dense_limit)

    @property
    def ram_first_bytes(self):
        return sum(self.ram_first_owners.values())

    @property
    def ram_first_key(self):
        return next(iter(self.ram_first_owners)) if len(self.ram_first_owners) == 1 else None

    def _publish_staging(self):
        record = self.snapshot()
        if record != getattr(self, '_published_staging', None):
            self._published_staging = record
            runtime_telemetry().gauge('sam_interpolation.parent_staging', record)

    def claim_ram_first(self, key, required_bytes, active_bytes, detector_room_bytes=0):
        """Reserve a RAM owner at birth, leaving room for detector progress."""
        required_bytes, active_bytes, detector_room_bytes = map(int,
            (required_bytes, active_bytes, detector_room_bytes))
        if required_bytes <= 0 or active_bytes < 0 or detector_room_bytes < 0:
            raise ValueError('RAM-first claim requires positive owner bytes and nonnegative admission room')
        key = tuple(key)
        if key in self.ram_first_owners:
            raise RuntimeError('SAM RAM parent was claimed twice')
        receipt = dict(parent=key, requested_owner_bytes=required_bytes,
            detector_room_bytes=detector_room_bytes, active_ram_commitments_bytes=active_bytes,
            excluded_disk_logical_bytes=self.leases.disk_backed_logical_bytes,
            ram_first_reserved_bytes=self.ram_first_bytes, ram_limit_bytes=self.dense_limit,
            total_checked_bytes=required_bytes+detector_room_bytes+active_bytes,
            anonymous_cap_bytes=None, physical_headroom_bytes=None,
            physical_reserve_bytes=RAM_RESERVE_BYTES)
        reason = ('cancelled' if self.stop.is_set() else
                  'ram_owner_limit' if self.ram_first_bytes + required_bytes > self.dense_limit else None)
        if reason is not None or not _ram_fits(required_bytes+detector_room_bytes,
                active_bytes, self.dense_limit, receipt, self.leases.ram_residency_promises()):
            receipt['reason'] = reason or receipt.get('reason') or 'ram_admission_refused'
            self.last_ram_denial = receipt
            self.ram_first_denials += 1
            self._publish_staging()
            return False
        self.ram_first_owners[key] = int(required_bytes)
        self.ram_first_grants += 1
        self._publish_staging()
        return True

    def owns_ram_first_parent(self, key):
        return tuple(key) in self.ram_first_owners

    def release_ram_first(self, key):
        """Caller has settled actors and retired this owner's actual arrays."""
        if not self.owns_ram_first_parent(key):
            return False
        del self.ram_first_owners[tuple(key)]
        self.ram_first_releases += 1
        self._publish_staging()
        return True

    def _ram_checkpoint_bytes(self, key, task, amount):
        if key not in self.ram_first_owners:
            return 0
        lease = self.leases.leases.get(key)
        if (lease is None or lease is not getattr(task, '_sam_checkpoint_lease', None)
                or lease.phase != 'postprocess'
                or key not in self.leases.postprocess_views
                or self.leases.postprocess_bytes.get(key) != lease.nbytes
                or int(amount) != int(lease.nbytes)
                or int(amount) != self.ram_first_owners[key]):
            raise RuntimeError('SAM RAM backlog lost its original postprocess lease')
        return int(amount)

    def detector_backlog_bytes(self):
        """Exclude immutable RAM checkpoints, retaining birth charges and leases."""
        return sum(self._ram_checkpoint_bytes(*saved) for saved in self.checkpoint_futures.values())

    def checkpoint_copy_promises(self):
        """Actual future RAM debt; call under the shared parent condition.

        Startup probes physical headroom after this resident proof. Disk-only
        deferred checkpoints and unused dense capacity are not allocations.
        """
        promises = self.leases.ram_residency_promises()
        promised = sum(max(0, int(lease.ram_commitment_bytes)) for lease in promises)
        _headroom, resident, _proof = memfd_ram_headroom(promises, publication_ram_headroom)
        return max(0, promised-min(promised, resident))

    def _submit_checkpoint(self, key, task, required_bytes):
        # Completed owned disk arrays need only the established flush/fsync
        # and descriptor capture. Never park their retirement behind RAM codec.
        regular = (task.union_mm is not None and all(
            _reusable_disk_owner(value, getattr(task,path_name), self.source_root) is not None
            for name,path_name in (('union_mm','union_path'),('confmap_mm','confmap_path'))
            if (value := getattr(task,name)) is not None))
        executor = self.prepare_executor if regular else self.checkpoint_executor
        future = executor.submit(checkpoint_parent, task, self.root,
            self.source_root, required_bytes, self.stop)
        if regular:
            self.raw_reuse_submissions += 1
        self.checkpoint_futures[future] = (key, task, int(required_bytes))
        self._publish_staging()

    def _restore_and_prepare(self, task, snapshot, lease):
        """Decode after dense admission, on the prepare executor, never the scheduler."""
        key = (str(task.model_name), str(task.view.name))
        started = time.perf_counter()
        record = dict(model_name=key[0], view_name=key[1], restore_started_monotonic=started,
            codec_admission_started_monotonic=None, codec_admitted=False,
            codec_admission_wait_seconds=0., allocation_seconds=0., allocation_bytes=0,
            mask_restore_seconds=0., confidence_restore_seconds=0.,
            mask_decoded_bytes=0, confidence_decoded_bytes=0, active_stage='preflight')
        for name, saved in (('mask', snapshot.mask), ('confidence', snapshot.confidence)):
            record[name+'_logical_bytes'] = 0 if saved is None else saved.nbytes
            record[name+'_stored_bytes'] = 0 if saved is None else saved.physical_bytes
        failed = cancelled = False
        error_class = None
        try:
            check_cancelled = getattr(task, 'check_cancelled', None)
            if check_cancelled is not None:
                check_cancelled()
            if self.leases.leases.get(key) is not lease:
                raise RuntimeError('SAM compact restore lost its dense reservation')
            active = self.leases.ram_commitment_bytes
            promises = tuple(other for other in tuple(self.leases.leases.values()) if other is not lease)
            prefer_ram = _ram_fits(snapshot.required_bytes,
                max(0, active-lease.ram_commitment_bytes), self.dense_limit, None, promises)
            record['prefer_ram'] = prefer_ram
            workspace = max(_codec_workspace_bytes(saved.shape)
                            for saved in (snapshot.mask, snapshot.confidence) if saved is not None)
            record.update(active_stage='codec_admission',
                codec_admission_started_monotonic=time.perf_counter())
            try:
                with _codec_reservation(task, workspace):
                    record['codec_admission_wait_seconds'] = (time.perf_counter()
                        - record['codec_admission_started_monotonic'])
                    record['codec_admitted'] = True
                    for saved, name, path_name, kind in (
                            (snapshot.mask, 'union_mm', 'union_path', 'mask'),
                            (snapshot.confidence, 'confmap_mm', 'confmap_path', 'confidence')):
                        if saved is None:
                            continue
                        record['active_stage'] = kind
                        if self.stop.is_set():
                            raise RuntimeError('SAM compact parent restore cancelled')
                        metrics = {}
                        restore_started = time.perf_counter()
                        def allocated(array):
                            if getattr(getattr(task, 'sam_context', None), 'progressive_startup', False):
                                # Facts do not retain arrays/FDs. Decode can add
                                # resident pages while startup sees the rest owed.
                                proofs = capture_memfd_owner_proofs({kind: array})
                                lease.memfd_owner_proofs = tuple(proof for proof in
                                    lease.memfd_owner_proofs if proof[0] != kind) + proofs
                        try:
                            array = saved.open(prefer_ram=prefer_ram, stop=self.stop, metrics=metrics,
                                allocation_guard=lambda: _restore_allocation_guard(task, metrics),
                                allocation_callback=allocated)
                        finally:
                            allocation = metrics.get('allocation_seconds', 0.)
                            record['allocation_seconds'] += allocation
                            record['allocation_bytes'] += metrics.get('allocation_bytes', 0)
                            record[kind+'_restore_seconds'] = max(0., time.perf_counter()-restore_started-allocation)
                            record[kind+'_decoded_bytes'] = metrics.get('decoded_bytes', 0)
                            for field in ('startup_ram_admitted', 'startup_headroom_bytes',
                                          'startup_required_host_bytes'):
                                if field in metrics:
                                    record[kind+'_'+field] = metrics[field]
                        setattr(task, name, array)
                        backing = _memfd_backing_path_from_array(array) or Path(str(array.filename))
                        setattr(task, path_name, backing or saved.path)
                        if saved.encoding != 'raw' and _memfd_owner_key_from_array(array) is None:
                            self.restore_raw_bytes += saved.nbytes
                            if prefer_ram and metrics.get('startup_ram_admitted') is False:
                                self.restore_startup_fallback_bytes += saved.nbytes
                        del array  # Confidence retirement must not retain a wrapper alias.
            except BaseException:
                _close_sources(task, preserve=False)
                raise
            lease.ram_backed = self._restored_ram_backing(task, lease)
            task._sam_checkpoint_prepare_started = True
            record['active_stage'] = 'complete'
        except BaseException as exc:
            failed = True
            cancelled = self.stop.is_set() or isinstance(exc, CancelledError)
            error_class = type(exc).__name__
            raise
        finally:
            if not record['codec_admitted'] and record['codec_admission_started_monotonic'] is not None:
                record['codec_admission_wait_seconds'] = (time.perf_counter()
                    - record['codec_admission_started_monotonic'])
            record.update(elapsed_seconds=time.perf_counter()-started, failed=failed,
                cancelled=cancelled, status='cancelled' if cancelled else 'failed' if failed else 'complete',
                error_class=error_class)
            try:
                telemetry = runtime_telemetry()
                telemetry.trace_event('sam_parent_restore', **record)
                telemetry.add_scheduler_counter('sam_parent_restore.'+record['status'])
                for name in ('elapsed_seconds', 'codec_admission_wait_seconds', 'allocation_seconds',
                        'allocation_bytes', 'mask_restore_seconds', 'confidence_restore_seconds',
                        'mask_decoded_bytes', 'confidence_decoded_bytes'):
                    telemetry.add_scheduler_counter('sam_parent_restore.'+name, record[name])
            except Exception:
                pass  # Diagnostics cannot replace the restore error or cancel preparation.
        return task()

    @staticmethod
    def _restored_ram_backing(task, lease):
        if hasattr(task, 'backing_lease'):
            return None  # Real preparation may create/rebind an output canvas.
        values = (task.union_mm, task.confmap_mm)
        return classify_dense_ram_backing(values, future_ram_bytes=max(
            0, lease.nbytes-sum(value.nbytes for value in values if value is not None)))

    def _retire_failed_restore(self, key, task, lease):
        if getattr(task, '_sam_checkpoint_prepare_started', False):
            # Preparation may publish/retain legitimate derived views. Preserve
            # the normal alias-aware close and its dense credit until retirement.
            _close_prepared_sources(task, preserve=self.keep_temp)
            return False
        _close_sources(task, preserve=self.keep_temp)
        return bool(self.leases.leases.get(key) is lease
                    and self.leases.complete(key, retain_for_dense_retirement=False))

    def reusable_workspace_paths(self, *paths):
        """Allow disk-from-birth only on the existing owned regular-disk root.

        A tmpfs input keeps the ordinary checkpoint-copy route to the verified
        output backing. This does not introduce another input ownership root.
        """
        if self.stop.is_set() or not paths:
            return False
        root = self.source_root.resolve()
        return all(Path(path).resolve().is_relative_to(root)
                   and not Path(path).is_symlink()
                   and not path_is_memory_backed(Path(path).parent) for path in paths)

    def defer(self, task, required_bytes):
        key = (str(task.model_name), str(task.view.name))
        if self.stop.is_set():
            raise RuntimeError('SAM parent staging is cancelled')
        if key in self.deferred or any(saved[0] == key for saved in self.checkpoint_futures.values()):
            raise RuntimeError('SAM parent was deferred twice')
        if task.union_mm is None:
            # D1 already has a compact immutable native shadow; defer without
            # expanding it or retiring its previously published components.
            if not path_is_memory_backed(Path(task.d1_shadow_path)):
                self.deferred[key] = (task, ParentCheckpoint(None, None, int(required_bytes), 0, 0, 0.))
                self.count += 1
                self.leases.complete(key, retain_for_dense_retirement=False)
                return
        else:
            # Reject aliases before a writer receives ownership. Submission
            # rollback can then restore the untouched original owner safely.
            _validate_owner(task.union_mm)
            if task.confmap_mm is not None:
                _validate_owner(task.confmap_mm)
        task._sam_checkpoint_lease = self.leases.leases.get(key)
        self._ram_checkpoint_bytes(key, task, required_bytes)
        self._submit_checkpoint(key, task, required_bytes)

    def pump(self, *, budget=None, resumed_futures=None):
        """Resume parents within the caller's shared background-completion budget.

        One ownership transition/probe is atomic and can exceed the time budget.
        Denied parents rotate so an interrupted pass still reaches later views.
        A caller-owned future registry receives each handoff before a budget
        checkpoint can fail, so cancellation can still find submitted work.
        """
        def items(mapping):
            return list(mapping) if budget is None else budget.items(0, mapping)

        def completed():
            if budget is not None:
                budget.completed(0)

        released = False
        # A previously failed checkpoint/restore must surface even when this
        # pass has yielded. Failed restores still retire their exact ownership.
        for future, (key, task, lease) in list(self.restore_futures.items()):
            if future.done() and not future.cancelled() and future.exception() is not None:
                del self.restore_futures[future]
                self._retire_failed_restore(key, task, lease)
                future.result()
        for future in self.checkpoint_futures:
            if future.done() and not future.cancelled() and future.exception() is not None:
                future.result()
        for future in items(self.restore_futures):
            key, task, lease = self.restore_futures[future]
            if not future.done():
                continue
            del self.restore_futures[future]
            if future.cancelled() or future.exception() is not None:
                released |= self._retire_failed_restore(key, task, lease)
                if not future.cancelled():
                    future.result()
            completed()
        for future in items(self.checkpoint_futures):
            key, task, required_bytes = self.checkpoint_futures[future]
            if not future.done():
                continue
            snapshot = future.result()
            del self.checkpoint_futures[future]
            if snapshot.shadow_path is not None:
                task.d1_shadow_path = snapshot.shadow_path
            released |= self.leases.complete(key, retain_for_dense_retirement=False)
            self.release_ram_first(key)
            self.deferred[key] = (task, snapshot)
            self.count += 1
            self.written_bytes += snapshot.written_bytes
            self.reused_bytes += snapshot.reused_bytes
            self.io_seconds += snapshot.seconds
            self.codec_fallback_bytes += snapshot.codec_fallback_bytes
            for saved in (snapshot.mask, snapshot.confidence):
                if saved is not None and saved.encoding != 'raw':
                    self.compact_bytes += saved.physical_bytes
                    self.compact_logical_bytes += saved.nbytes
            completed()
        resumed = {}
        if self.stop.is_set() or not self.ready():
            self._publish_staging()
            return resumed, released
        for key in items(self.deferred):
            task, snapshot = self.deferred[key]
            if key in self.bounded_parent_keys and snapshot.required_bytes > self.dense_limit:
                raise RuntimeError('SAM deferred policy parent exceeds bounded dense limit')
            denied = False
            with _parent_condition(task):
                active = sum(self.leases.inference_bytes.values()) + sum(self.leases.postprocess_bytes.values())
                emergency = not self.leases.leases and key not in self.bounded_parent_keys
                if not emergency and active + snapshot.required_bytes > self.dense_limit:
                    denied = True
                elif not _dense_startup_fits(_dense_startup_budget_locked(task, snapshot.required_bytes)):
                    denied = True  # No dense/base/GPU credit is parked behind startup.
                else:
                    self.leases.leases[key] = _DirectUnionBackingLease(key, snapshot.required_bytes, phase='postprocess')
                    self.leases.postprocess_views.add(key)
                    self.leases.postprocess_bytes[key] = snapshot.required_bytes
                    lease = self.leases.leases[key]
            if denied:
                self.deferred[key] = self.deferred.pop(key)
                completed()
                continue
            try:
                if hasattr(task, 'backing_lease'):
                    task.backing_lease = lease
                rebind_retirement = getattr(task, 'rebind_confidence_retirement', None)
                if callable(rebind_retirement):
                    rebind_retirement(lease)
                compact = any(saved is not None and saved.encoding != 'raw'
                              for saved in (snapshot.mask, snapshot.confidence))
                if compact:
                    task._sam_checkpoint_prepare_started = False
                    future = self.prepare_executor.submit(self._restore_and_prepare, task, snapshot,
                        lease)
                    self.restore_futures[future] = (key, task, lease)
                else:
                    if snapshot.mask is not None:
                        task.union_mm = snapshot.mask.open()
                        task.union_path = snapshot.mask.path
                    if snapshot.confidence is not None:
                        task.confmap_mm = snapshot.confidence.open()
                        task.confmap_path = snapshot.confidence.path
                    lease.ram_backed = self._restored_ram_backing(task, lease)
                    future = self.prepare_executor.submit(task)
                track_cancellation = getattr(task, 'track_cancellation', None)
                if track_cancellation is not None:
                    track_cancellation(future)
            except BaseException:
                _close_sources(task, preserve=True)
                self.leases.complete(key, retain_for_dense_retirement=False)
                raise
            del self.deferred[key]
            if snapshot.owned_dir is not None:
                # The normal prepared-view/run lifecycle now owns these exact
                # backings. Queue teardown must not unlink a live result.
                self.transferred_roots.add(snapshot.owned_dir)
            resumed[future] = key
            if resumed_futures is not None:
                resumed_futures[future] = key
            self.resumed_count += 1
            completed()
        self._publish_staging()
        return resumed, released

    def abort(self):
        self.stop.set()
        for future, (_key, task, _required) in self.checkpoint_futures.items():
            if future.cancel():
                _close_sources(task, preserve=self.keep_temp)
        for future, (_key, task, _lease) in self.restore_futures.items():
            if future.cancel():
                _close_sources(task, preserve=self.keep_temp)

    def cancel(self, reason='TTA parent staging cancelled'):
        self.abort()

    def snapshot(self):
        return dict(checkpoint_parents=self.count, resumed_parents=self.resumed_count,
            pending_checkpoints=len(self.checkpoint_futures), deferred_parents=len(self.deferred),
            pending_checkpoint_dense_bytes=sum(int(saved[2]) for saved in self.checkpoint_futures.values()),
            ram_first_owner_key=self.ram_first_key,
            ram_first_owner_keys=tuple(self.ram_first_owners),
            ram_first_owner_count=len(self.ram_first_owners),
            ram_first_reserved_bytes=self.ram_first_bytes,
            ram_backlog_limit_bytes=self.dense_limit,
            detector_excluded_ram_backlog_bytes=self.detector_backlog_bytes(),
            ram_first_grants=self.ram_first_grants, ram_first_denials=self.ram_first_denials,
            last_ram_denial=self.last_ram_denial,
            ram_first_releases=self.ram_first_releases,
            raw_reuse_submissions=self.raw_reuse_submissions,
            written_bytes=self.written_bytes, reused_regular_disk_bytes=self.reused_bytes,
            io_seconds=self.io_seconds, stream_buffer_limit=STREAM_BYTES,
            checkpoint_wall_seconds=self.io_seconds,
            compact_checkpoint_bytes=self.compact_bytes,
            compact_logical_input_bytes=self.compact_logical_bytes,
            restore_raw_fallback_bytes=self.restore_raw_bytes,
            restore_startup_fallback_bytes=self.restore_startup_fallback_bytes,
            codec_raw_fallback_bytes=self.codec_fallback_bytes,
            codec_raw_fallback_reason='transient_credit_unavailable' if self.codec_fallback_bytes else None,
            restore_raw_fallback_reason='physical_headroom_or_memfd_unavailable' if self.restore_raw_bytes else None,
            checkpoint_root=str(self.root), checkpoint_backing='disk',
            keep_temp_debug_ownership='durable checkpoint replaces retired memory-backed input')

    def close(self):
        self.abort()
        # Called after the checkpoint executor has settled, so no writer owns
        # these maps or files and failures cannot expose new admission credit.
        for future, (key, task, _required) in list(self.checkpoint_futures.items()):
            if not future.done():
                raise RuntimeError('SAM checkpoint writer must settle before staging cleanup')
            _close_sources(task, preserve=self.keep_temp)
            self.leases.complete(key, retain_for_dense_retirement=False)
            self.release_ram_first(key)
        self.checkpoint_futures.clear()
        for future, (key, task, lease) in list(self.restore_futures.items()):
            if not future.done():
                raise RuntimeError('SAM compact restore must settle before staging cleanup')
            if future.cancelled() or future.exception() is not None:
                self._retire_failed_restore(key, task, lease)
        self.restore_futures.clear()
        self.deferred.clear()
        if not self.keep_temp and self.root.exists():
            for child in self.root.iterdir():
                if child not in self.transferred_roots:
                    shutil.rmtree(child)

    def finalize_cleanup(self):
        """Retire transferred files only after final fusion and output settle."""
        if not self.keep_temp and self.root.exists():
            shutil.rmtree(self.root)


__all__ = ['DeferredSamParentQueue', 'checkpoint_parent', 'select_checkpoint_root']
