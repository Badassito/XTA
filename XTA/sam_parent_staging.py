"""Release completed parent canvases while shared SAM awaits detector retirement.

Checkpoints live on ordinary disk, retain exact numeric bytes, and own no dense
mapping while deferred. Resumed preparation receives a fresh dense reservation.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
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
                      close_memmap_array_without_flush, workspace_anon_cap_bytes, runtime_telemetry)
from .publication_memory import publication_ram_headroom
from .view_prepare import scratch_unlink_path_for_memmap, classify_dense_ram_backing

STREAM_BYTES = 8 * 1024**2
CODEC_CONTROL_BYTES = 64 * 1024**2
RAM_RESERVE_BYTES = 16 * 1024**3


def _ram_fits(required_bytes, active_bytes=0, dense_limit=None, receipt=None):
    """Use actual physical/cgroup headroom, never swap, for another live owner."""
    required, active = max(0, int(required_bytes)), max(0, int(active_bytes))
    total = required + active
    cap = int(workspace_anon_cap_bytes())
    headroom = int(publication_ram_headroom())
    reason = ('empty_request' if not required else
              'dense_limit' if dense_limit is not None and total > int(dense_limit) else
              'anonymous_workspace_cap' if cap > 0 and total > cap else
              'physical_headroom' if headroom < total + RAM_RESERVE_BYTES else None)
    if receipt is not None:
        receipt.update(total_checked_bytes=total, anonymous_cap_bytes=cap,
            physical_headroom_bytes=headroom, physical_reserve_bytes=RAM_RESERVE_BYTES,
            reason=reason)
    return reason is None


def _codec_workspace_bytes(shape):
    # The mask decoder owns at most one bbox plane; the incremental encoder
    # owns <=1MiB row slabs plus its per-frame index. Confidence cells are 128².
    # An unusually wide single row exceeds the writer's 1MiB slab target;
    # charge overlapping row normalization/packing and confidence vectors too.
    return (CODEC_CONTROL_BYTES + int(shape[1]) * int(shape[2])
            + 4 * int(shape[2]) + int(shape[0]) * 64)


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

    def open(self, *, prefer_ram=False, stop=None):
        if path_is_memory_backed(self.path) or _stat_identity(self.path) != self.identity:
            raise RuntimeError('SAM parent checkpoint backing changed before resume')
        if self.encoding == 'raw':
            return np.memmap(self.path, mode='r+', dtype=np.dtype(self.dtype), shape=self.shape)
        for name, identity in self.files:
            if _stat_identity(self.path / name) != identity:
                raise RuntimeError('SAM compact checkpoint backing changed before resume')
        destination = self.path.with_name(self.path.name + '.restored.dat')
        output = allocate_workspace_array(self.shape, np.dtype(self.dtype), destination,
            'SAM compact parent restore', prefer_memory=False, prefer_memfd=bool(prefer_ram),
            reserve_bytes=RAM_RESERVE_BYTES, initialize_zero=True)
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
            else:
                raise ValueError('Unknown SAM checkpoint encoding')
            return output
        except BaseException:
            close_memmap_array_without_flush(output, unlink_path=destination)
            if isinstance(output, np.memmap) and not output._mmap.closed:
                output._mmap.close()
            raise

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


def _snapshot_array(array, declared_path, destination, source_root, stop, *, mask=False, allow_compact=True):
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
                def read(z):
                    if stop.is_set():
                        raise RuntimeError('SAM parent checkpoint cancelled')
                    return array[z]
                write_blocks(path, tuple(array.shape), read, layer_key='sam-parent-checkpoint',
                    model_name='checkpoint', provenance={'sam_parent_checkpoint': True},
                    coordinate_space='native_view_processing', source_shape=tuple(array.shape))
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
                    mask=name=='union_mm', allow_compact=compact_credit))
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
                                             active_bytes, self.dense_limit, receipt):
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
        check_cancelled = getattr(task, 'check_cancelled', None)
        if check_cancelled is not None:
            check_cancelled()
        key = (str(task.model_name), str(task.view.name))
        if self.leases.leases.get(key) is not lease:
            raise RuntimeError('SAM compact restore lost its dense reservation')
        active = self.leases.ram_commitment_bytes
        prefer_ram = _ram_fits(snapshot.required_bytes,
            max(0, active-lease.ram_commitment_bytes), self.dense_limit)
        workspace = max(_codec_workspace_bytes(saved.shape)
                        for saved in (snapshot.mask, snapshot.confidence) if saved is not None)
        try:
            with _codec_reservation(task, workspace):
                for saved, name, path_name in ((snapshot.mask, 'union_mm', 'union_path'),
                                               (snapshot.confidence, 'confmap_mm', 'confmap_path')):
                    if saved is None:
                        continue
                    if self.stop.is_set():
                        raise RuntimeError('SAM compact parent restore cancelled')
                    array = saved.open(prefer_ram=prefer_ram, stop=self.stop)
                    setattr(task, name, array)
                    backing = _memfd_backing_path_from_array(array) or Path(str(array.filename))
                    setattr(task, path_name, backing or saved.path)
                    if saved.encoding != 'raw' and _memfd_owner_key_from_array(array) is None:
                        self.restore_raw_bytes += saved.nbytes
                    del array  # Confidence retirement must not retain a wrapper alias.
        except BaseException:
            _close_sources(task, preserve=False)
            raise
        lease.ram_backed = self._restored_ram_backing(task, lease)
        task._sam_checkpoint_prepare_started = True
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

    def pump(self):
        """Return newly resumed prepare futures and whether dense credit returned."""
        released = False
        for future, (key, task, lease) in list(self.restore_futures.items()):
            if not future.done():
                continue
            del self.restore_futures[future]
            if future.cancelled() or future.exception() is not None:
                released |= self._retire_failed_restore(key, task, lease)
                if not future.cancelled():
                    future.result()
        for future, (key, task, required_bytes) in list(self.checkpoint_futures.items()):
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
        resumed = {}
        if self.stop.is_set() or not self.ready():
            self._publish_staging()
            return resumed, released
        for key, (task, snapshot) in list(self.deferred.items()):
            if key in self.bounded_parent_keys and snapshot.required_bytes > self.dense_limit:
                raise RuntimeError('SAM deferred policy parent exceeds bounded dense limit')
            active = sum(self.leases.inference_bytes.values()) + sum(self.leases.postprocess_bytes.values())
            emergency = not self.leases.leases and key not in self.bounded_parent_keys
            if not emergency and active + snapshot.required_bytes > self.dense_limit:
                continue
            self.leases.leases[key] = _DirectUnionBackingLease(key, snapshot.required_bytes, phase='postprocess')
            self.leases.postprocess_views.add(key)
            self.leases.postprocess_bytes[key] = snapshot.required_bytes
            lease = self.leases.leases[key]
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
            self.resumed_count += 1
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
