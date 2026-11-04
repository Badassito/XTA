"""Parent-side SAM checkpoint digest prefetch for predictor startup.

Every SAM predictor process records ``sam_model.checkpoint_sha256``. Without a
prefetch, each worker hashes the multi-GiB checkpoint after loading its model,
on the critical path between detector retirement and the first tracker job.
With ``YOLO_TTA_SAM_CHECKPOINT_DIGEST_PREFETCH=1`` the coordinator hashes the
same file once on a host thread while detector inference still owns the GPUs.
The sequential read also leaves the file in reclaimable page cache for the
workers' memory-mapped load. No CUDA, GPU lease, model, or held allocation is
involved: the thread owns one bounded read buffer and an open read-only handle.

A worker reuses the digest only when its own post-load stat identity equals
the identity recorded before and after the coordinator's hash. Otherwise it
hashes the file itself exactly as before. The recorded digest value, and all
evidence identities derived from it, are unchanged.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Mapping, Optional

ENV_FLAG = 'YOLO_TTA_SAM_CHECKPOINT_DIGEST_PREFETCH'
RECEIPT_SCHEMA = 'xta.sam_checkpoint_digest/1'
_BLOCK_BYTES = 8 * 1024 * 1024
_SHA256_HEX = re.compile(r'^[0-9a-f]{64}$')
_IDENTITY_KEYS = ('path', 'device', 'inode', 'size_bytes', 'mtime_ns', 'ctime_ns')


def prefetch_enabled() -> bool:
    raw = os.environ.get(ENV_FLAG)
    return raw is not None and raw.strip().lower() not in ('', '0', 'false', 'no', 'off')


def checkpoint_stat_identity(path) -> dict:
    """Content-change witness: resolved path plus device/inode/size/mtime/ctime."""
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return dict(path=str(resolved), device=int(stat.st_dev), inode=int(stat.st_ino),
                size_bytes=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns),
                ctime_ns=int(stat.st_ctime_ns))


def hash_checkpoint(path, *, cancel_event=None, block_bytes=_BLOCK_BYTES):
    """Return a stable digest receipt, or None after cancellation/identity change."""
    started = time.perf_counter()
    before = checkpoint_stat_identity(path)
    digest = hashlib.sha256()
    buffer = bytearray(int(block_bytes))
    view = memoryview(buffer)
    total = 0
    with open(before['path'], 'rb', buffering=0) as stream:
        advise = getattr(os, 'posix_fadvise', None)
        if callable(advise):
            try:
                advise(stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except OSError:
                pass  # Advice is optional; the read below is authoritative.
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return None
            count = stream.readinto(view)
            if not count:
                break
            # hashlib releases the GIL for large updates.
            digest.update(view[:count])
            total += count
    after = checkpoint_stat_identity(path)
    if before != after or total != before['size_bytes']:
        return None
    return dict(schema=RECEIPT_SCHEMA, sha256=digest.hexdigest(), identity=before,
                hashed_bytes=total, seconds=time.perf_counter()-started)


def validated_digest(receipt: Optional[Mapping[str, object]], path) -> Optional[str]:
    """Return the receipt digest only if ``path`` still has the hashed identity."""
    if not isinstance(receipt, Mapping) or receipt.get('schema') != RECEIPT_SCHEMA:
        return None
    digest, identity = receipt.get('sha256'), receipt.get('identity')
    if (not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None
            or not isinstance(identity, Mapping) or set(identity) != set(_IDENTITY_KEYS)):
        return None
    try:
        current = checkpoint_stat_identity(path)
    except OSError:
        return None
    if any(type(identity[key]) is not type(current[key]) or identity[key] != current[key]
           for key in _IDENTITY_KEYS):
        return None
    return digest


def _lower_current_thread_priority():
    """Best effort: yield CPU to detector feeding/scheduling threads."""
    try:
        if os.name == 'nt':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -2)  # THREAD_PRIORITY_LOWEST
        elif sys.platform.startswith('linux') and hasattr(os, 'setpriority'):
            # Linux applies PRIO_PROCESS with a TID to that thread only.
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except Exception:
        pass


class CheckpointDigestPrefetch:
    """One background hash of one checkpoint; queried without blocking."""

    def __init__(self, path):
        self.path = str(path)
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._receipt = None
        self.error = None
        self._thread = threading.Thread(target=self._run, name='sam-checkpoint-digest', daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        _lower_current_thread_priority()
        try:
            self._receipt = hash_checkpoint(self.path, cancel_event=self._cancel)
        except Exception as error:  # Workers fall back to hashing locally.
            self.error = f'{type(error).__name__}: {error}'
        finally:
            self._done.set()

    @property
    def done(self):
        return self._done.is_set()

    def receipt(self, timeout=0.):
        """Return a copy of a completed stable receipt, else None (never raises)."""
        if not self._done.wait(max(0., float(timeout))) or self._receipt is None:
            return None
        receipt = dict(self._receipt)
        receipt['identity'] = dict(receipt['identity'])
        return receipt

    def cancel(self, timeout=2.):
        self._cancel.set()
        self._thread.join(timeout=max(0., float(timeout)))
        return not self._thread.is_alive()


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict[str, CheckpointDigestPrefetch] = {}


def _key(model_path) -> str:
    return str(Path(model_path).expanduser().resolve(strict=False))


def start_checkpoint_digest_prefetch(model_path, checkpoint_path):
    """Start (once per model path) when enabled; returns the prefetch or None."""
    if not prefetch_enabled():
        return None
    key = _key(model_path)
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None and existing.path == str(checkpoint_path):
            return existing
        prefetch = CheckpointDigestPrefetch(checkpoint_path)
        _REGISTRY[key] = prefetch
    return prefetch.start()


def checkpoint_digest_receipt(model_path):
    """Non-blocking lookup used when building worker configuration."""
    with _REGISTRY_LOCK:
        prefetch = _REGISTRY.get(_key(model_path))
    return None if prefetch is None else prefetch.receipt()


def cancel_checkpoint_digest_prefetch(model_path, timeout=2.):
    with _REGISTRY_LOCK:
        prefetch = _REGISTRY.pop(_key(model_path), None)
    return True if prefetch is None else prefetch.cancel(timeout)
