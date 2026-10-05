"""Release completed parent canvases while shared SAM awaits detector retirement.

Checkpoints live on ordinary disk, retain exact numeric bytes, and own no dense
mapping while deferred. Resumed preparation receives a fresh dense reservation.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np

from .interpolation import _DirectUnionBackingLease
from .runtime import path_is_memory_backed
from .view_prepare import scratch_unlink_path_for_memmap

STREAM_BYTES = 8 * 1024**2


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

    def open(self):
        if path_is_memory_backed(self.path) or _stat_identity(self.path) != self.identity:
            raise RuntimeError('SAM parent checkpoint backing changed before resume')
        return np.memmap(self.path, mode='r+', dtype=np.dtype(self.dtype), shape=self.shape)


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


def _snapshot_array(array, declared_path, destination, source_root, stop):
    array = _validate_owner(array)
    reused = _reusable_disk_owner(array, declared_path, source_root)
    if reused is not None:
        array.flush()
        with reused.open('r+b') as stream:
            os.fsync(stream.fileno())
        path = reused
    else:
        if shutil.disk_usage(destination.parent).free < array.nbytes + STREAM_BYTES:
            raise RuntimeError('Insufficient disk space for bounded SAM parent checkpoint')
        flat = array.reshape(-1).view(np.uint8)
        with destination.open('xb', buffering=0) as stream:
            for offset in range(0, flat.size, STREAM_BYTES):
                if stop.is_set():
                    raise RuntimeError('SAM parent checkpoint cancelled')
                chunk = memoryview(flat[offset:offset + STREAM_BYTES])
                while chunk:
                    written = stream.write(chunk)
                    if not written:
                        raise OSError('SAM parent checkpoint write made no progress')
                    chunk = chunk[written:]
                del chunk
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
                        writer.write(chunk)
                    os.fsync(writer.fileno())
            if stop.is_set():
                raise RuntimeError('SAM parent checkpoint cancelled')
            # The shadow is a private completed native store, not the already
            # published source-grid component. Debug ownership moves to disk.
            shutil.rmtree(source)
            return ParentCheckpoint(None, None, int(required_bytes), needed, 0,
                time.perf_counter()-started, destination, owned)
        for name, path_name, filename in (('union_mm', 'union_path', 'mask.dat'),
                                          ('confmap_mm', 'confmap_path', 'confidence.dat')):
            if stop.is_set():
                raise RuntimeError('SAM parent checkpoint cancelled')
            array = getattr(task, name)
            snapshots.append(None if array is None else _snapshot_array(array,
                getattr(task, path_name), owned / filename, source_root, stop))
            del array
        if stop.is_set():
            raise RuntimeError('SAM parent checkpoint cancelled')
        reused = tuple(value.path for value in snapshots if value is not None and value.reused)
        _close_sources(task, reused)
        logical = sum(value.nbytes for value in snapshots if value is not None)
        written = sum(value.nbytes for value in snapshots if value is not None and not value.reused)
        return ParentCheckpoint(*snapshots, max(int(required_bytes), logical), written,
            logical-written, time.perf_counter()-started, owned_dir=owned)
    except BaseException:
        _close_sources(task, preserve=bool(task.keep_temp_artifacts))
        shutil.rmtree(owned)
        raise


class DeferredSamParentQueue:
    """Scheduler-thread queue with one bounded disk writer and no SAM waits."""

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
        self.deferred = {}
        self.transferred_roots = set()
        self.count = self.written_bytes = self.reused_bytes = self.resumed_count = 0
        self.io_seconds = 0.

    @property
    def pending(self):
        return bool(self.checkpoint_futures or self.deferred)

    @property
    def dense_limit(self):
        return int(self._dense_limit() if callable(self._dense_limit) else self._dense_limit)

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
        future = self.checkpoint_executor.submit(checkpoint_parent, task, self.root,
            self.source_root, required_bytes, self.stop)
        self.checkpoint_futures[future] = (key, task)

    def pump(self):
        """Return newly resumed prepare futures and whether dense credit returned."""
        released = False
        for future, (key, task) in list(self.checkpoint_futures.items()):
            if not future.done():
                continue
            snapshot = future.result()
            del self.checkpoint_futures[future]
            if snapshot.shadow_path is not None:
                task.d1_shadow_path = snapshot.shadow_path
            released |= self.leases.complete(key, retain_for_dense_retirement=False)
            self.deferred[key] = (task, snapshot)
            self.count += 1
            self.written_bytes += snapshot.written_bytes
            self.reused_bytes += snapshot.reused_bytes
            self.io_seconds += snapshot.seconds
        resumed = {}
        if self.stop.is_set() or not self.ready():
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
            try:
                rebind_retirement = getattr(task, 'rebind_confidence_retirement', None)
                if callable(rebind_retirement):
                    rebind_retirement(self.leases.leases[key])
                if snapshot.mask is not None:
                    task.union_mm = snapshot.mask.open()
                    task.union_path = snapshot.mask.path
                if snapshot.confidence is not None:
                    task.confmap_mm = snapshot.confidence.open()
                    task.confmap_path = snapshot.confidence.path
                future = self.prepare_executor.submit(task)
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
        return resumed, released

    def abort(self):
        self.stop.set()
        for future, (_key, task) in self.checkpoint_futures.items():
            if future.cancel():
                _close_sources(task, preserve=self.keep_temp)

    def cancel(self, reason='TTA parent staging cancelled'):
        self.abort()

    def snapshot(self):
        return dict(checkpoint_parents=self.count, resumed_parents=self.resumed_count,
            pending_checkpoints=len(self.checkpoint_futures), deferred_parents=len(self.deferred),
            written_bytes=self.written_bytes, reused_regular_disk_bytes=self.reused_bytes,
            io_seconds=self.io_seconds, stream_buffer_limit=STREAM_BYTES,
            checkpoint_root=str(self.root), checkpoint_backing='disk',
            keep_temp_debug_ownership='durable checkpoint replaces retired memory-backed input')

    def close(self):
        self.abort()
        # Called after the checkpoint executor has settled, so no writer owns
        # these maps or files and failures cannot expose new admission credit.
        for future, (key, task) in list(self.checkpoint_futures.items()):
            if not future.done():
                raise RuntimeError('SAM checkpoint writer must settle before staging cleanup')
            _close_sources(task, preserve=self.keep_temp)
            self.leases.complete(key, retain_for_dense_retirement=False)
        self.checkpoint_futures.clear()
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
