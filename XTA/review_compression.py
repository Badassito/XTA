"""CPU gzip admission and a shared executor for review publication."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import re
from threading import BoundedSemaphore

from .env_flags import env_int
from .gzip_spans import PARALLEL_GZIP_MEMBER_BYTES


_MAX_REVIEW_GZIP_WORKERS = 32


class _OldNrrdLibdeflateBindingError(RuntimeError):
    """A known python-deflate release holds the GIL during compression."""


def _known_nrrd_libdeflate_version(module):
    """Use the imported binding's version, never unrelated site metadata."""
    raw = getattr(module, '__version__', None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    module_file = getattr(module, '__file__', None)
    if not module_file:
        return None
    try:
        selected_path = Path(str(module_file)).resolve()
    except (OSError, ValueError):
        return None
    for distribution_name in ('deflate', 'python-deflate'):
        try:
            distribution = importlib_metadata.distribution(distribution_name)
            files = distribution.files
            if files is None:
                continue
            if any(Path(distribution.locate_file(file)).resolve() == selected_path
                   for file in files):
                return str(distribution.version)
        except Exception:
            continue
    return None


def _libdeflate_version_family(version):
    match = re.match(r'^(\d+)\.(\d+)(?:\.\d+)?(?:$|[.+])', version or '')
    return (int(match[1]), int(match[2])) if match else None


def _old_nrrd_libdeflate_binding_version(module):
    """Recognize pre-0.9 releases; unknown/custom bindings need separate GIL proof."""
    version = _known_nrrd_libdeflate_version(module)
    family = _libdeflate_version_family(version)
    return version if family is not None and family < (0, 9) else None


def load_libdeflate_gzip(*, require_parallel_version=False):
    """Resolve python-deflate's gzip member encoder and check its GIL compatibility."""
    import deflate

    version = _known_nrrd_libdeflate_version(deflate)
    family = _libdeflate_version_family(version)
    if family is not None and family < (0, 9):
        raise _OldNrrdLibdeflateBindingError(
            f'python-deflate {version} holds the Python GIL during compression; '
            'install deflate>=0.9.0 for parallel NRRD compression or select '
            'YOLO_TTA_NRRD_MEMBER_CODEC=isal/zlib'
        )
    if family is None and require_parallel_version:
        raise RuntimeError('Review compression requires a known deflate>=0.9.0 binding')
    return deflate.gzip_compress


def _affinity_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL('kernel32')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessAffinityMask.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if kernel.GetProcessAffinityMask(
            kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system),
        ):
            return process.value.bit_count()
    return None


def _review_gzip_workers():
    visible = _affinity_count() or getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    allocated = env_int('SLURM_CPUS_PER_TASK', visible)
    configured = env_int('YOLO_TTA_NRRD_GZIP_WORKERS', _MAX_REVIEW_GZIP_WORKERS)
    return max(1, min(visible, allocated, configured, _MAX_REVIEW_GZIP_WORKERS))


def _stdlib_gzip(payload, level):
    return gzip.compress(payload, compresslevel=level, mtime=0)


class ReviewGzipPool:
    """Share available compression threads while writers own their own buffers."""

    def __init__(self, executor, encoder, backend, max_workers):
        self._executor = executor
        self._encoder = encoder
        self._slots = BoundedSemaphore(max_workers)
        self.backend = backend
        self.max_workers = max_workers

    def compress(self, data, level):
        # A writer without a queue budget compresses on its caller's thread,
        # sharing the same concurrency limit as requests submitted to the pool.
        with self._slots:
            return bytes(self._encoder(data, level))

    def submit(self, data, level):
        return self._executor.submit(self.compress, data, level)

    def plan(self, available_bytes, payload_bytes):
        """Reserve writer coalescing space and a bounded number of in-flight members."""
        # A short channel already fits in the writer's synchronous buffer.
        if payload_bytes <= 64 * 1024:
            return 0, 0
        member_bytes = min(payload_bytes, PARALLEL_GZIP_MEMBER_BYTES)
        payload_members = (payload_bytes + PARALLEL_GZIP_MEMBER_BYTES - 1) // PARALLEL_GZIP_MEMBER_BYTES
        coalescing = 2 * member_bytes
        # Detached input, encoded result and headroom for the level-3 codec.
        per_job = 2 * member_bytes + PARALLEL_GZIP_MEMBER_BYTES
        budgeted_members = (available_bytes - coalescing) // per_job
        pending = min(self.max_workers, payload_members, budgeted_members)
        return (pending, coalescing + pending * per_job) if pending > 0 else (0, 0)


@contextmanager
def review_gzip_pool():
    """Admit one codec, then own one executor for all cutoff writers in a review."""
    try:
        encoder = load_libdeflate_gzip(require_parallel_version=True)
        known = bytes(range(256)) * 16 + b'\0' * 4097
        if gzip.decompress(encoder(known, 3)) != known:
            raise RuntimeError('gzip round trip returned different bytes')
    except Exception as exc:
        print(f'Review libdeflate unavailable ({exc}); using stdlib gzip.', flush=True)
        encoder, backend = _stdlib_gzip, 'stdlib'
    else:
        backend = 'libdeflate'
    workers = _review_gzip_workers()
    print(f'Review gzip compression uses {backend} with up to {workers} workers.', flush=True)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='xta-review-gzip') as executor:
        yield ReviewGzipPool(executor, encoder, backend, workers)
