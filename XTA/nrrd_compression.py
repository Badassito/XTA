"""Complete-member NRRD compression: codec admission, worker lifetime and ordered streams.

This owner does not open or publish output paths. Raster producers choose a proven
codec before publication and submit bytes to its bounded streaming writer. Native
session retirement, fork reset and telemetry providers share this process state.
"""
from __future__ import annotations

import atexit
import importlib
from importlib import metadata as importlib_metadata
import io
import json
import os
import re
import struct
import sys
import threading
import time
import weakref
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from .nrrd_spans import canonical_zero_member
from .runtime import runtime_telemetry
from .workspace import _cpu_count, _env_flag, _env_int

def nrrd_gzip_compresslevel() -> int:
    """Return the NRRD member-gzip compression level.

    QAT maps this intent by hardware generation; ISA-L maps it to 0..3.
    """
    return int(np.clip(_env_int('YOLO_TTA_NRRD_GZIP_LEVEL', 3), 0, 9))

def nrrd_gzip_workers() -> int:
    """Size of the shared complete-member compression pool."""
    cores = max(1, _cpu_count())
    return max(2, _env_int('YOLO_TTA_NRRD_GZIP_WORKERS', max(4, cores // 2)))

def nrrd_gzip_chunk_bytes() -> int:
    return max(1, _env_int('YOLO_TTA_NRRD_GZIP_CHUNK_MIB', 16)) * 1024 * 1024

_NRRD_GZIP_EXECUTORS: Dict[Tuple[str, int], ThreadPoolExecutor] = {}

_NRRD_GZIP_EXECUTOR_LOCK = threading.Lock()

_NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED = False

def _nrrd_codec_worker_count(
    codec_spec: Tuple[str, int, Callable[[object], bytes]],
) -> int:
    """Bound hardware sessions by the binding's advertised usable capacity."""
    backend, _level, compressor = codec_spec
    requested = int(nrrd_gzip_workers())
    if not bool(getattr(compressor, 'hardware_backend', False)):
        return max(1, requested)
    limit = max(1, int(getattr(compressor, 'max_concurrency', 1)))
    env_name = (
        'YOLO_TTA_NRRD_QAT_WORKERS'
        if str(backend) == 'qat' else
        'YOLO_TTA_NRRD_IAA_WORKERS'
    )
    configured = max(1, _env_int(str(env_name), min(requested, limit)))
    return max(1, min(int(configured), int(limit)))

def _nrrd_gzip_executor(
    codec_spec: Tuple[str, int, Callable[[object], bytes]],
) -> ThreadPoolExecutor:
    """Return a shared pool sized for the selected CPU or hardware backend."""
    global _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED
    backend = str(codec_spec[0])
    workers = int(_nrrd_codec_worker_count(codec_spec))
    key = (backend, workers)
    with _NRRD_GZIP_EXECUTOR_LOCK:
        executor = _NRRD_GZIP_EXECUTORS.get(key)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=int(workers), thread_name_prefix=f'nrrd-gzip-{backend}',
            )
            _NRRD_GZIP_EXECUTORS[key] = executor
        if not _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED:
            atexit.register(shutdown_nrrd_gzip_executors)
            _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED = True
        return executor

def _run_on_every_executor_thread(
    executor: ThreadPoolExecutor,
    workers: int,
    callback: Callable[[], None],
) -> None:
    """Use a barrier so one task is resident on every pool thread before callback."""
    count = max(1, int(workers))
    barrier = threading.Barrier(count)

    def _call() -> None:
        barrier.wait(timeout=60.0)
        callback()

    futures = [executor.submit(_call) for _ in range(count)]
    first_error: Optional[BaseException] = None
    for fut in futures:
        try:
            fut.result()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error

def _report_nrrd_gzip_cleanup_failure(
    operation: str,
    exc: BaseException,
) -> None:
    """Best-effort cleanup diagnostics that cannot replace a pipeline exception."""
    try:
        runtime_telemetry().fallback(str(operation), exc)
    except BaseException:
        pass
    try:
        print(
            'Warning: Intel compression cleanup failed during '
            f'{operation} ({type(exc).__name__}: {exc}).'
        )
    except BaseException:
        pass

def _shutdown_nrrd_gzip_executor_safely(executor: ThreadPoolExecutor) -> None:
    """Drain an executor without allowing teardown failures to escape cleanup."""
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except TypeError:
        # Python/runtime-compatible fallback for executors without cancel_futures.
        try:
            executor.shutdown(wait=True)
        except BaseException as exc:
            _report_nrrd_gzip_cleanup_failure(
                'nrrd.compression.executor_shutdown', exc,
            )
    except BaseException as exc:
        _report_nrrd_gzip_cleanup_failure(
            'nrrd.compression.executor_shutdown', exc,
        )

def _record_nrrd_native_codec_stats(backend: str) -> None:
    """Publish and reset native hardware counters after a codec pool drains."""
    name = str(backend)
    if name not in {'qat', 'iaa'}:
        return
    try:
        from .intel_compression import native_stats

        stats = dict(native_stats(name, reset=True))
        if not stats:
            return
        telemetry = runtime_telemetry()
        telemetry.gauge(f'nrrd.compression.{name}.native_stats', stats)
        cumulative = {
            'input_bytes',
            'output_bytes',
            'logical_requests',
            'physical_members',
            'hardware_requests',
            'software_fallback_requests',
            'queue_busy_events',
            'failures',
            'sessions_created',
            'sessions_closed',
            'elapsed_ns',
        }
        for key in cumulative:
            value = stats.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            telemetry.add(f'nrrd.compression.{name}.native.{key}', value)
    except BaseException as exc:
        _report_nrrd_gzip_cleanup_failure(
            f'nrrd.compression.{name}.native_stats', exc,
        )

def _close_nrrd_gzip_executor_entry(
    backend: str,
    workers: int,
    executor: ThreadPoolExecutor,
) -> None:
    """Close one idle pool, including native state on hardware worker threads."""
    try:
        # CPython's private ThreadPoolExecutor exit hook may already have joined the
        # pool before ordinary atexit callbacks run. Native TLS destructors have then
        # run too, so there is no worker left on which to schedule explicit cleanup.
        if (
            str(backend) in {'qat', 'iaa'}
            and not bool(getattr(executor, '_shutdown', False))
        ):
            from .intel_compression import close_current_thread_state

            _run_on_every_executor_thread(
                executor, int(workers), close_current_thread_state,
            )
    except BaseException as exc:
        _report_nrrd_gzip_cleanup_failure(
            'nrrd.compression.thread_cleanup', exc,
        )
    finally:
        _shutdown_nrrd_gzip_executor_safely(executor)
        _record_nrrd_native_codec_stats(str(backend))

def _retire_nrrd_codec_executor(codec_spec: NrrdMemberCodecSpec) -> None:
    """Release a failed hardware codec's sessions immediately after preflight/KAT."""
    backend = str(codec_spec[0])
    workers = int(_nrrd_codec_worker_count(codec_spec))
    with _NRRD_GZIP_EXECUTOR_LOCK:
        executor = _NRRD_GZIP_EXECUTORS.pop((backend, workers), None)
    if executor is not None:
        _close_nrrd_gzip_executor_entry(backend, workers, executor)

def shutdown_nrrd_gzip_executors() -> None:
    """Drain pools and close native TLS sessions on their owning worker threads."""
    global _NRRD_MEMBER_GZIP_ANNOUNCED, _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED
    with _NRRD_GZIP_EXECUTOR_LOCK:
        entries = list(_NRRD_GZIP_EXECUTORS.items())
        _NRRD_GZIP_EXECUTORS.clear()
    try:
        for (backend, workers), executor in entries:
            _close_nrrd_gzip_executor_entry(str(backend), int(workers), executor)
    finally:
        # A subsequent embedded run gets a fresh hardware probe, KAT, thread preflight,
        # and selected-backend announcement even when its environment policy changed.
        with _NRRD_MEMBER_GZIP_TEST_LOCK:
            _NRRD_MEMBER_GZIP_OK.clear()
            _NRRD_MEMBER_GZIP_FAILURE_REASONS.clear()
        _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED.clear()
        _NRRD_MEMBER_GZIP_ANNOUNCED = False
        _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = False

_GZIP_MEMBER_HEADER = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03'

_NRRD_ZERO_MEMBER_CACHE: Dict[int, bytes] = {}

_NRRD_ZERO_MEMBER_LOCK = threading.Lock()

def _gzip_member_from_raw_stream(raw_stream: bytes, crc: int, length: int) -> bytes:
    """Frame one raw RFC-1951 stream as a complete RFC-1952 gzip member."""
    return (
        _GZIP_MEMBER_HEADER
        + raw_stream
        + struct.pack('<II', int(crc) & 0xFFFFFFFF, int(length) & 0xFFFFFFFF)
    )

def _zero_gzip_member(length: int) -> bytes:
    """Return a cached complete gzip member representing ``length`` zero bytes."""
    ln = max(0, int(length))
    with _NRRD_ZERO_MEMBER_LOCK:
        cached = _NRRD_ZERO_MEMBER_CACHE.get(ln)
    if cached is not None:
        return cached
    zeros = bytes(ln)
    cobj = zlib.compressobj(1, zlib.DEFLATED, -15)
    raw = cobj.compress(zeros) + cobj.flush(zlib.Z_FINISH)
    member = _gzip_member_from_raw_stream(raw, zlib.crc32(zeros), ln)
    with _NRRD_ZERO_MEMBER_LOCK:
        _NRRD_ZERO_MEMBER_CACHE[ln] = member
    return member

_NRRD_MEMBER_CODEC_SETTING_WARNED = False

NrrdMemberCodecSpec = Tuple[str, int, Callable[[object], bytes]]

def nrrd_member_codec_requested() -> str:
    """NRRD gzip policy; ``cpu`` is the opt-out from preferred automatic QAT."""
    global _NRRD_MEMBER_CODEC_SETTING_WARNED
    raw = os.environ.get('YOLO_TTA_NRRD_MEMBER_CODEC', 'auto').strip().lower()
    if raw in {'auto', 'cpu', 'qat', 'iaa', 'libdeflate', 'isal', 'zlib'}:
        return str(raw)
    if not _NRRD_MEMBER_CODEC_SETTING_WARNED:
        _NRRD_MEMBER_CODEC_SETTING_WARNED = True
        print(
            f'Warning: invalid YOLO_TTA_NRRD_MEMBER_CODEC={raw!r}; using auto '
            '(QAT -> libdeflate -> ISA-L -> zlib).'
        )
    return 'auto'

def nrrd_libdeflate_level() -> int:
    """Resolve the libdeflate complete-member compression level.

    Level zero remains valid input to codec self-test so unsupported store mode can fall through safely."""
    return int(np.clip(
        _env_int('YOLO_TTA_NRRD_LIBDEFLATE_LEVEL', int(nrrd_gzip_compresslevel())),
        0,
        12,
    ))

def _nrrd_member_codec_candidates() -> Tuple[str, ...]:
    requested = str(nrrd_member_codec_requested())
    if requested == 'auto':
        return ('qat', 'libdeflate', 'isal', 'zlib')
    if requested == 'cpu':
        return ('libdeflate', 'isal', 'zlib')
    return (requested,)

def _nrrd_optional_numa_id(environment_name: str) -> Optional[int]:
    raw = os.environ.get(str(environment_name), '').strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'{environment_name} must be an integer, got {raw!r}') from exc
    return None if value < 0 else int(value)

def _nrrd_qat_level(capabilities: Dict[str, object]) -> int:
    """Map zlib intent to Intel's generation-dependent QAT compression levels."""
    raw_supported = capabilities.get('supported_levels', tuple(range(1, 10)))
    try:
        if isinstance(raw_supported, (str, bytes)):
            supported = tuple(
                sorted({int(token.strip()) for token in str(raw_supported).split(',') if token.strip()})
            )
        else:
            supported = tuple(sorted({int(value) for value in raw_supported}))  # type: ignore[union-attr]
    except Exception:
        supported = tuple(range(1, 10))

    def _validated(level: int) -> int:
        if int(level) not in supported:
            raise ValueError(
                f'QAT level {int(level)} is unsupported; binding reports {supported}'
            )
        return int(level)

    explicit = os.environ.get('YOLO_TTA_NRRD_QAT_LEVEL', '').strip()
    if explicit:
        try:
            level = int(explicit)
        except ValueError as exc:
            raise ValueError(
                f'YOLO_TTA_NRRD_QAT_LEVEL must be an integer, got {explicit!r}'
            ) from exc
        if not 1 <= int(level) <= 9:
            raise ValueError('YOLO_TTA_NRRD_QAT_LEVEL must be in [1, 9]')
        return _validated(int(level))
    configured = int(nrrd_gzip_compresslevel())
    if configured <= 0:
        raise RuntimeError(
            'QAT auto selection is ineligible for gzip level 0; choose '
            'YOLO_TTA_NRRD_MEMBER_CODEC=cpu or set an explicit QAT level'
        )
    generation = str(
        capabilities.get('hardware_generation', capabilities.get('qat_generation', ''))
    ).strip().lower()
    qat_1x = bool(
        generation.startswith('1')
        or generation.startswith('qat1')
        or 'qat 1.' in generation
    )
    qat_2plus = bool(
        generation.startswith(('2', '3', '4'))
        or generation.startswith(('qat2', 'qat3', 'qat4'))
        or any(f'qat {major}.' in generation for major in (2, 3, 4))
    )
    if configured <= 4:
        return _validated(1)
    if not qat_1x and not qat_2plus:
        raise RuntimeError(
            'QAT binding must report hardware_generation to map gzip levels above 4; '
            'set YOLO_TTA_NRRD_QAT_LEVEL explicitly to override'
        )
    if configured == 5:
        return _validated(5 if qat_1x else 1)
    if configured <= 8:
        return _validated(5 if qat_1x else 6)
    return _validated(9)

def _nrrd_iaa_level(capabilities: Dict[str, object]) -> int:
    configured = int(nrrd_gzip_compresslevel())
    default = 1 if configured <= 5 else 3
    level = int(_env_int('YOLO_TTA_NRRD_IAA_LEVEL', int(default)))
    raw_supported = capabilities.get('supported_levels', (1, 3))
    try:
        supported = tuple(sorted({int(value) for value in raw_supported}))  # type: ignore[union-attr]
    except Exception:
        supported = (1, 3)
    if int(level) not in supported:
        raise ValueError(
            f'YOLO_TTA_NRRD_IAA_LEVEL={level} is unsupported; binding reports {supported}'
        )
    return int(level)

class _OldNrrdLibdeflateBindingError(RuntimeError):
    """A known python-deflate release holds the GIL during compression."""


def _known_nrrd_libdeflate_version(module: object) -> Optional[str]:
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


def _old_nrrd_libdeflate_binding_version(module: object) -> Optional[str]:
    """Recognize pre-0.9 releases; unknown/custom bindings need separate GIL proof.

    This is a version compatibility check, not an automatic throughput benchmark.
    Unparseable or unassociated package metadata never proves an old binding.
    """
    version = _known_nrrd_libdeflate_version(module)
    if version is None:
        return None
    match = re.match(r'^(\d+)\.(\d+)(?:\.\d+)?(?:$|[.+])', version)
    if match is None:
        return None
    return version if (int(match.group(1)), int(match.group(2))) < (0, 9) else None


def _nrrd_member_codec_spec(
    codec_name: str,
) -> NrrdMemberCodecSpec:
    """Resolve one gzip-member-sequence codec without silently substituting."""
    name = str(codec_name).strip().lower()
    configured_level = int(nrrd_gzip_compresslevel())
    if name in {'qat', 'iaa'}:
        from .intel_compression import create_gzip_compressor, probe_capabilities

        capabilities = probe_capabilities(name)
        if name == 'qat':
            level = int(_nrrd_qat_level(capabilities))
            numa_id = None
        else:
            level = int(_nrrd_iaa_level(capabilities))
            numa_id = _nrrd_optional_numa_id('YOLO_TTA_NRRD_IAA_NUMA_ID')
        compressor = create_gzip_compressor(
            name, int(level), numa_id=numa_id, capabilities=capabilities,
        )
        return str(name), int(level), compressor
    if name == 'libdeflate':
        import deflate  # type: ignore

        old_version = _old_nrrd_libdeflate_binding_version(deflate)
        if old_version is not None:
            raise _OldNrrdLibdeflateBindingError(
                f'python-deflate {old_version} holds the Python GIL during compression; '
                'install deflate>=0.9.0 for parallel NRRD compression or select '
                'YOLO_TTA_NRRD_MEMBER_CODEC=isal/zlib'
            )
        level = int(nrrd_libdeflate_level())
        gzip_compress = getattr(deflate, 'gzip_compress')

        def _compress_libdeflate(payload: bytes) -> bytes:
            # Positional level is intentional: this is the python-deflate public API and
            # ensures the exact one-shot call shape is covered by the known-answer test.
            return bytes(gzip_compress(payload, int(level)))

        return 'libdeflate', int(level), _compress_libdeflate
    if name == 'isal':
        from isal import isal_zlib  # type: ignore

        level = 0 if configured_level <= 0 else (
            1 if configured_level <= 4 else (2 if configured_level <= 7 else 3)
        )
        level = int(np.clip(_env_int('YOLO_TTA_NRRD_ISAL_LEVEL', int(level)), 0, 3))

        def _compress_isal(payload: bytes) -> bytes:
            cobj = isal_zlib.compressobj(
                int(level),
                isal_zlib.DEFLATED,
                16 + int(getattr(isal_zlib, 'MAX_WBITS', 15)),
            )
            return bytes(cobj.compress(payload) + cobj.flush())

        return 'isal', int(level), _compress_isal
    if name == 'zlib':
        level = int(configured_level)

        def _compress_zlib(payload: bytes) -> bytes:
            cobj = zlib.compressobj(int(level), zlib.DEFLATED, 16 + int(zlib.MAX_WBITS))
            return bytes(cobj.compress(payload) + cobj.flush())

        return 'zlib', int(level), _compress_zlib
    raise ValueError(f'Unsupported NRRD member codec {codec_name!r}')

def nrrd_member_gzip_window_bytes() -> int:
    """Uncompressed bytes one layer writer may hold in flight (copied chunks awaiting deflate)."""
    return max(64, _env_int('YOLO_TTA_NRRD_MEMBER_GZIP_WINDOW_MIB', 512)) * 1024 * 1024


@dataclass(frozen=True)
class _RepeatedGzipMembers:
    """An encoded member repeated without retaining its logical zero raster."""

    member: bytes
    count: int


_NRRD_LIVE_WRITERS_LOCK = threading.Lock()
_NRRD_LIVE_WRITERS: weakref.WeakSet['_MemberParallelGzipPayloadWriter'] = weakref.WeakSet()
_NRRD_LIVE_RETIRED_WAIT_SECONDS = 0.0


def _nrrd_live_writer_sample() -> Dict[str, object]:
    """Bounded snapshot of waits still in progress, sampled by RuntimeSystemSampler."""
    now_ns = time.monotonic_ns()
    with _NRRD_LIVE_WRITERS_LOCK:
        writers = tuple(_NRRD_LIVE_WRITERS)
        completed_wait_seconds = float(_NRRD_LIVE_RETIRED_WAIT_SECONDS)
        waits = tuple((writer, writer._diag_wait) for writer in writers)
        completed_wait_seconds += sum(writer._diag_completed_wait_seconds for writer in writers)
    causes = {'zero_descriptor': 0, 'close': 0, 'window': 0}
    head_stages = {'queued': 0, 'running': 0, 'done': 0, 'unknown': 0}
    oldest_head: Optional[Dict[str, object]] = None
    oldest_head_age = 0.0
    oldest_wait_age = 0.0
    waiting = 0
    for writer, state in waits:
        if state is None:
            continue
        waiting += 1
        (started_ns, cause, head_sequence, submitted_ns, head_future,
         pending_count, ready_count, inflight_bytes) = state
        causes[str(cause)] = causes.get(str(cause), 0) + 1
        wait_age = max(0.0, (now_ns - int(started_ns)) / 1e9)
        head_age = (
            max(0.0, (now_ns - int(submitted_ns)) / 1e9)
            if submitted_ns is not None else None
        )
        oldest_wait_age = max(oldest_wait_age, wait_age)
        stage = 'unknown'
        if head_future is not None:
            stage = ('done' if head_future.done() else
                     'running' if head_future.running() else 'queued')
        head_stages[stage] += 1
        if head_age is not None and head_age >= oldest_head_age:
            oldest_head_age = head_age
            oldest_head = {
                'layer': writer._diag_layer_name,
                'sequence': int(head_sequence),
                'future_stage': stage,
                'wait_cause': str(cause),
                'head_age_seconds': head_age,
                'wait_age_seconds': wait_age,
                'pending_members': int(pending_count),
                'ready_descriptors': int(ready_count),
                'inflight_bytes': int(inflight_bytes),
            }
    return {
        'sample_monotonic_ns': now_ns,
        'active_writers': len(writers),
        'waiting_writers': waiting,
        'waiting_by_cause': causes,
        'head_future_stages': head_stages,
        'oldest_wait_seconds': oldest_wait_age,
        'oldest_head_age_seconds': oldest_head_age if oldest_head is not None else None,
        'oldest_head': oldest_head,
        'completed_wait_seconds_process_lifetime': completed_wait_seconds,
    }


def _register_nrrd_live_writer(writer: '_MemberParallelGzipPayloadWriter') -> None:
    try:
        telemetry = runtime_telemetry()
        if not (bool(getattr(telemetry, 'enabled', False))
                and _env_flag('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', True)):
            return
        with _NRRD_LIVE_WRITERS_LOCK:
            _NRRD_LIVE_WRITERS.add(writer)
            writer._diag_enabled = True
        # A module function keeps no writer alive. Re-registration replaces the
        # same named provider if observability is reinitialized in this process.
        telemetry.register_sample_provider('nrrd.member_stream.live', _nrrd_live_writer_sample)
    except BaseException:
        # Telemetry must never change output success or failure.
        with _NRRD_LIVE_WRITERS_LOCK:
            _NRRD_LIVE_WRITERS.discard(writer)
            writer._diag_enabled = False


def _unregister_nrrd_live_writer(writer: '_MemberParallelGzipPayloadWriter') -> None:
    global _NRRD_LIVE_RETIRED_WAIT_SECONDS
    if not writer._diag_enabled:
        return
    with _NRRD_LIVE_WRITERS_LOCK:
        if not writer._diag_enabled:
            return
        _NRRD_LIVE_RETIRED_WAIT_SECONDS += writer._diag_completed_wait_seconds
        _NRRD_LIVE_WRITERS.discard(writer)
        writer._diag_enabled = False


class _MemberParallelGzipPayloadWriter:
    """Pipelined encoder emitting one complete gzip member sequence per chunk.

 ``write(data)`` detaches chunks before deferring them, so callers may immediately reuse
 their buffers. Completion is unordered, but a sequence-indexed ready map drains the
 longest available ordered prefix. This prevents a slow early member from hiding later
 completed work while preserving byte-exact payload order."""

    def __init__(
        self,
        fh: object,
        *,
        chunk_bytes: Optional[int] = None,
        codec_spec: Optional[NrrdMemberCodecSpec] = None,
    ) -> None:
        resolved = codec_spec if codec_spec is not None else _select_nrrd_member_codec()
        if resolved is None:
            raise RuntimeError('No validated complete-member NRRD gzip codec is available')
        self.codec_spec = resolved
        self.fh = fh
        self.backend = str(resolved[0])
        self.eff_level = int(resolved[1])
        self.member_compress = resolved[2]
        self.minimum_input_bytes = max(
            1, int(getattr(self.member_compress, 'minimum_input_bytes', 1))
        )
        self.chunk_bytes = int(chunk_bytes) if chunk_bytes else int(nrrd_gzip_chunk_bytes())
        self.window_bytes = int(nrrd_member_gzip_window_bytes())
        self.closed = False
        self._next_sequence = 0
        self._next_write_sequence = 0
        self._pending: Dict[Future, Tuple[int, int]] = {}
        self._completed: Dict[int, Tuple[bytes | _RepeatedGzipMembers, int]] = {}
        self._inflight_bytes = 0
        self._small_pending = bytearray()
        self._hardware_lookbehind: Optional[object] = None
        self._hardware_lookbehind_is_zero = False
        self._encoded_batch_bytes = 1024 * 1024
        self._writer_stats: Dict[str, int | float] = {
            'writers': 1, 'failed_writers': 0,
            'canonical_zero_logical_bytes': 0, 'canonical_zero_members': 0,
            'canonical_zero_descriptors': 0, 'output_write_calls': 0,
            'canonical_zero_write_calls': 0, 'output_write_bytes': 0,
            'raw_write_seconds': 0.0, 'pending_wait_seconds': 0.0,
            'ordered_prefix_wait_seconds': 0.0,
            'later_ready_prefix_wait_seconds': 0.0,
            'window_wait_seconds': 0.0,
            'zero_descriptor_wait_seconds': 0.0,
            'close_wait_seconds': 0.0,
        }
        self._writer_stats_published = False
        self._diag_enabled = False
        self._diag_wait: Optional[Tuple[object, ...]] = None
        self._diag_pending: Dict[int, Tuple[Future, int]] = {}
        self._diag_completed_wait_seconds = 0.0
        name = getattr(fh, 'name', '<memory>')
        self._diag_layer_name = Path(str(name)).name[:160]
        _register_nrrd_live_writer(self)

    def _compress_member(self, payload: object) -> Tuple[bytes, int]:
        """Encode in one native gzip pass; CRC is produced by ISA-L/zlib itself.

 Known zeros enter through ``write_zeros`` from extent/cvol metadata. Unknown data is
 deliberately not pre-scanned: native DEFLATE handles an unexpected zero member more
 cheaply than adding a full memory pass to every nonzero member."""
        mv = payload if isinstance(payload, memoryview) else memoryview(payload)  # type: ignore[arg-type]
        mv = mv.cast('B')
        ln = int(len(mv))
        telemetry = runtime_telemetry()
        telemetry.add(f'nrrd.compression.{self.backend}.input_bytes', int(ln))
        try:
            with telemetry.span(f'nrrd.compression.{self.backend}.compress'):
                try:
                    member_sequence = bytes(self.member_compress(mv))
                except TypeError:
                    if bool(getattr(self.member_compress, 'hardware_backend', False)):
                        # A hardware call may already have completed before a binding
                        # raises while validating status/output. Never resubmit it.
                        raise
                    # A codec wheel without generic buffer-protocol support is still
                    # valid; make its one required copy inside the worker, never on the
                    # sparse assembly thread.
                    member_sequence = bytes(self.member_compress(bytes(mv)))
        except Exception:
            telemetry.add(f'nrrd.compression.{self.backend}.failures', 1)
            raise
        telemetry.add(f'nrrd.compression.{self.backend}.output_bytes', len(member_sequence))
        telemetry.add(f'nrrd.compression.{self.backend}.logical_requests', 1)
        return member_sequence, int(ln)

    def _enqueue_completed(self, member: bytes) -> None:
        seq = int(self._next_sequence)
        self._next_sequence += 1
        self._completed[seq] = (member, 0)

    def _enqueue_repeated_zeros(self, member: bytes, count: int, *, members_per_repeat: int = 1) -> None:
        if not member or int(count) <= 0:
            raise ValueError('Repeated gzip members require nonempty bytes and a positive count')
        seq = int(self._next_sequence)
        self._next_sequence += 1
        self._completed[seq] = (_RepeatedGzipMembers(member, int(count)), 0)
        self._writer_stats['canonical_zero_members'] += int(count) * int(members_per_repeat)
        self._writer_stats['canonical_zero_descriptors'] += 1

    def _enqueue_chunk(self, chunk_mv: memoryview) -> None:
        seq = int(self._next_sequence)
        self._next_sequence += 1
        payload = bytes(chunk_mv)  # detach once; classification no longer adds a NumPy scan
        ln = int(len(payload))
        fut = _nrrd_gzip_executor(self.codec_spec).submit(
            self._compress_member, payload,
        )
        self._pending[fut] = (int(seq), int(ln))
        self._inflight_bytes += int(ln)
        if self._diag_enabled:
            self._diag_pending[int(seq)] = (fut, time.monotonic_ns())

    def _enqueue_owned_chunk(self, owner: object) -> None:
        """Transfer one caller-owned contiguous allocation to a compressor worker.

 The Future retains ``mv`` (and therefore its NumPy exporter) until native DEFLATE
 finishes. This removes the former 8--16 MiB ``bytes`` detachment copy for every
 sparse cvol member while preserving the ordinary write buffer-reuse contract."""
        mv = owner if isinstance(owner, memoryview) else memoryview(owner)  # type: ignore[arg-type]
        mv = mv.cast('B')
        seq = int(self._next_sequence)
        self._next_sequence += 1
        ln = int(len(mv))
        fut = _nrrd_gzip_executor(self.codec_spec).submit(
            self._compress_member, mv,
        )
        self._pending[fut] = (int(seq), int(ln))
        self._inflight_bytes += int(ln)
        if self._diag_enabled:
            self._diag_pending[int(seq)] = (fut, time.monotonic_ns())

    def _hardware_lookbehind_enabled(self) -> bool:
        return bool(
            getattr(self.member_compress, 'hardware_backend', False)
            and int(self.minimum_input_bytes) > 1
        )

    def _submit_hardware_lookbehind(self) -> None:
        owner = self._hardware_lookbehind
        if owner is None:
            return
        is_zero = bool(self._hardware_lookbehind_is_zero)
        self._hardware_lookbehind = None
        self._hardware_lookbehind_is_zero = False
        if is_zero:
            self._enqueue_completed(_zero_gzip_member(len(memoryview(owner).cast('B'))))
        else:
            self._enqueue_owned_chunk(owner)

    def _queue_data_chunk(self, chunk: object, *, owned: bool = False) -> None:
        """Queue data, retaining one final hardware request for cross-write tails."""
        mv = chunk if isinstance(chunk, memoryview) else memoryview(chunk)  # type: ignore[arg-type]
        mv = mv.cast('B')
        if not self._hardware_lookbehind_enabled():
            if owned:
                self._enqueue_owned_chunk(mv)
            else:
                self._enqueue_chunk(mv)
            return
        self._submit_hardware_lookbehind()
        # Ordinary write() promises that its caller may immediately reuse the source;
        # the owned path deliberately retains its exporter instead.
        self._hardware_lookbehind = mv if owned else bytes(mv)
        self._hardware_lookbehind_is_zero = False

    def _collect_completions(self, *, block: bool,
                             wait_cause: Optional[str] = None) -> int:
        if not self._pending:
            return 0
        if bool(block):
            # Preserve the historical counter for comparisons with older runs.
            # It is usually equal to pending_wait_seconds after a ready-prefix
            # drain and must not be interpreted as proof of head-of-line blocking.
            prefix_missing = self._next_write_sequence not in self._completed
            # This separate counter requires a later member already ready when
            # the wait starts; it is not comparable with the historical key.
            later_ready = any(seq > self._next_write_sequence
                              for seq in self._completed)
            started = time.perf_counter()
            if self._diag_enabled:
                head_sequence = int(self._next_write_sequence)
                head = self._diag_pending.get(head_sequence)
                head_future: Optional[Future]
                submitted_ns: Optional[int]
                if head is None:
                    # Tests and unusual callers can insert a Future directly.
                    head_future = next(
                        (fut for fut, (seq, _charged) in self._pending.items()
                         if int(seq) == head_sequence), None,
                    )
                    submitted_ns = None
                else:
                    head_future, submitted_ns = head
                self._diag_wait = (
                    time.monotonic_ns(), str(wait_cause or 'other'), head_sequence,
                    submitted_ns, head_future, len(self._pending),
                    len(self._completed), int(self._inflight_bytes),
                )
            try:
                done, _not_done = wait(set(self._pending), return_when=FIRST_COMPLETED)
            finally:
                elapsed = time.perf_counter() - started
                self._writer_stats['pending_wait_seconds'] += elapsed
                if self._diag_enabled:
                    with _NRRD_LIVE_WRITERS_LOCK:
                        self._diag_completed_wait_seconds += elapsed
                        self._diag_wait = None
                if prefix_missing:
                    self._writer_stats['ordered_prefix_wait_seconds'] += elapsed
                if later_ready:
                    self._writer_stats['later_ready_prefix_wait_seconds'] += elapsed
                if wait_cause is not None:
                    self._writer_stats[f'{wait_cause}_wait_seconds'] += elapsed
        else:
            done = {fut for fut in self._pending if fut.done()}
        for fut in done:
            seq, charged = self._pending.pop(fut)
            if self._diag_enabled:
                self._diag_pending.pop(int(seq), None)
            member, ln = fut.result()
            if int(ln) != int(charged):
                raise RuntimeError(f'NRRD member length mismatch: {ln} != {charged}')
            self._completed[int(seq)] = (member, int(charged))
        return int(len(done))

    def _write_ready_prefix(self) -> int:
        if self._next_write_sequence not in self._completed:
            return 0
        written = 0
        buffer = bytearray()
        contains_zero = False
        limit = self._encoded_batch_bytes

        def flush_buffer() -> None:
            nonlocal contains_zero
            if buffer:
                self._write_encoded_bytes(buffer, contains_zero=contains_zero)
                buffer.clear()
                contains_zero = False

        def append(data: bytes, *, zeros: bool) -> None:
            nonlocal contains_zero
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                count = min(limit - len(buffer), len(view) - offset)
                if not buffer and count == limit:
                    self._write_encoded_bytes(view[offset:offset + count], contains_zero=zeros)
                else:
                    buffer.extend(view[offset:offset + count])
                    contains_zero = contains_zero or zeros
                    if len(buffer) == limit:
                        flush_buffer()
                offset += count

        while int(self._next_write_sequence) in self._completed:
            member, charged = self._completed.pop(int(self._next_write_sequence))
            if isinstance(member, _RepeatedGzipMembers):
                per_batch = max(1, limit // len(member.member))
                remaining = member.count
                bundle_count = min(remaining, per_batch)
                bundle = member.member * bundle_count
                while remaining >= bundle_count:
                    append(bundle, zeros=True)
                    remaining -= bundle_count
                if remaining:
                    append(member.member * remaining, zeros=True)
            else:
                append(member, zeros=False)
            self._inflight_bytes -= int(charged)
            self._next_write_sequence += 1
            written += 1
        flush_buffer()
        return int(written)

    def _write_encoded_bytes(self, data: object, *, contains_zero: bool) -> None:
        """Finish one bounded raw write, including sinks which accept only a prefix."""
        view = memoryview(data).cast('B')
        offset = 0
        while offset < len(view):
            started = time.perf_counter()
            self._writer_stats['output_write_calls'] += 1
            if contains_zero:
                self._writer_stats['canonical_zero_write_calls'] += 1
            try:
                count = self.fh.write(view[offset:])
            finally:
                self._writer_stats['raw_write_seconds'] += time.perf_counter() - started
            count = len(view) - offset if count is None else int(count)
            if count <= 0 or count > len(view) - offset:
                raise OSError('NRRD encoded write made invalid forward progress')
            offset += count
            self._writer_stats['output_write_bytes'] += count

    def _publish_writer_stats(self) -> None:
        if self._writer_stats_published:
            return
        self._writer_stats_published = True
        try:
            telemetry = runtime_telemetry()
            for name, value in self._writer_stats.items():
                telemetry.add(f'nrrd.member_stream.{name}', value)
        except Exception:
            pass

    def _drain(self, *, block: bool) -> None:
        self._collect_completions(block=False)
        self._write_ready_prefix()
        while self._pending and (bool(block) or self._inflight_bytes > self.window_bytes):
            self._collect_completions(block=True,
                wait_cause='close' if block else 'window')
            self._write_ready_prefix()
        if bool(block):
            self._write_ready_prefix()

    def _drain_zero_descriptor_pressure(self) -> None:
        """Bound ready zero descriptors without waiting for unrelated later jobs."""
        self._drain(block=False)
        while self._pending and len(self._completed) >= 128:
            self._collect_completions(block=True, wait_cause='zero_descriptor')
            self._write_ready_prefix()

    def _write_impl(self, data: bytes | bytearray | memoryview) -> int:
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        mv = data if isinstance(data, memoryview) else memoryview(data)
        mv = mv.cast('B')
        total = len(mv)
        off = 0
        if self._small_pending:
            take = min(total, int(self.minimum_input_bytes) - len(self._small_pending))
            if take > 0:
                self._small_pending.extend(mv[:take])
                off += int(take)
            if len(self._small_pending) >= int(self.minimum_input_bytes):
                self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                self._small_pending.clear()
                self._drain(block=False)
        remaining = int(total - off)
        if 0 < remaining < int(self.minimum_input_bytes):
            self._small_pending.extend(mv[off:])
            return int(total)
        while off < total:
            remaining = int(total - off)
            ln = min(int(self.chunk_bytes), remaining)
            tail = int(remaining - ln)
            # QATzip cannot prove hardware execution for sub-threshold tails. Merge a
            # tiny tail into the preceding logical request instead of allowing SW work.
            if 0 < tail < int(self.minimum_input_bytes):
                ln = int(remaining)
            self._queue_data_chunk(mv[off:off + ln])
            off += int(ln)
            self._drain(block=False)
        return int(total)

    def write(self, data: bytes | bytearray | memoryview) -> int:
        try:
            return self._write_impl(data)
        except BaseException:
            self.closed = True
            self._abandon_and_settle()
            raise

    def write_known_nonzero(self, data: bytes | bytearray | memoryview) -> int:
        """Encode cvol-index-classified bytes without a redundant all-zero scan."""
        return self.write(data)

    def write_owned_known_nonzero(self, data: object) -> int:
        """Ownership-transfer fast path for one already slice-aligned member."""
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        mv = data if isinstance(data, memoryview) else memoryview(data)  # type: ignore[arg-type]
        mv = mv.cast('B')
        if len(mv) <= 0:
            return 0
        if self._small_pending or len(mv) < int(self.minimum_input_bytes):
            return self.write(mv)
        try:
            self._queue_data_chunk(mv, owned=True)
            self._drain(block=False)
        except BaseException:
            self.closed = True
            self._abandon_and_settle()
            raise
        return int(len(mv))

    def write_canonical_zeros(self, nbytes: int) -> int:
        """Use at most 21 reusable zero-member sizes for arbitrary sparse gaps.

        Software codecs accept these small complete gzip members. Hardware codecs
        retain their existing minimum-input/lookbehind policy through write_zeros.
        """
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        remaining = int(nbytes)
        if remaining < 0:
            raise ValueError('A zero run cannot have negative length')
        if remaining == 0:
            return 0
        if self.minimum_input_bytes > 1:
            return self.write_zeros(remaining)
        try:
            self._writer_stats['canonical_zero_logical_bytes'] += remaining
            full_members, remaining = divmod(remaining, 1024 * 1024)
            if full_members:
                self._enqueue_repeated_zeros(canonical_zero_member(1024 * 1024), full_members)
            tail_members = []
            while remaining:
                size = 1 << (remaining.bit_length() - 1)
                tail_members.append(canonical_zero_member(size))
                remaining -= size
            if tail_members:
                self._enqueue_repeated_zeros(b''.join(tail_members), 1,
                                             members_per_repeat=len(tail_members))
            # A whole run owns at most two tiny descriptors, independent of its
            # logical size. Wait only until the ready map is bounded; close()
            # remains the only unconditional full drain.
            if len(self._completed) >= 128:
                self._drain_zero_descriptor_pressure()
            else:
                self._drain(block=False)
        except BaseException:
            self.closed = True
            self._abandon_and_settle()
            raise
        return int(nbytes)

    def write_aligned_zeros(self, nbytes: int) -> int:
        """Emit one complete cached zero member matching a whole-slice sparse member."""
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        requested = int(nbytes)
        ln = int(requested)
        if ln <= 0:
            return 0
        if self._small_pending:
            take = min(ln, int(self.minimum_input_bytes) - len(self._small_pending))
            self._small_pending.extend(bytes(int(take)))
            ln -= int(take)
            if len(self._small_pending) >= int(self.minimum_input_bytes):
                self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                self._small_pending.clear()
                self._drain(block=False)
        if ln <= 0:
            return int(nbytes)
        if self._hardware_lookbehind_enabled() and ln < int(self.minimum_input_bytes):
            if self._hardware_lookbehind is None:
                self._small_pending.extend(bytes(int(ln)))
            else:
                self._hardware_lookbehind = (
                    bytes(memoryview(self._hardware_lookbehind).cast('B'))
                    + bytes(int(ln))
                )
            return int(requested)
        self._submit_hardware_lookbehind()
        if self._hardware_lookbehind_enabled():
            # Retain the final hardware-sized zero span so a later sub-threshold
            # nonzero write can be appended without reordering bytes or falling back
            # to software. Most of a large zero run still uses the cached member.
            cached = int(ln) - int(self.minimum_input_bytes)
            if cached > 0:
                self._enqueue_completed(_zero_gzip_member(int(cached)))
            self._hardware_lookbehind = bytes(int(self.minimum_input_bytes))
            self._hardware_lookbehind_is_zero = True
        else:
            self._enqueue_completed(_zero_gzip_member(int(ln)))
        self._drain(block=False)
        return int(requested)

    def write_zeros(self, nbytes: int) -> int:
        """Emit cached all-zero members without materializing the zeros."""
        if self.closed:
            raise RuntimeError('Cannot write to a closed gzip payload stream')
        remaining = int(nbytes)
        if self._small_pending and remaining > 0:
            take = min(remaining, int(self.minimum_input_bytes) - len(self._small_pending))
            self._small_pending.extend(bytes(int(take)))
            remaining -= int(take)
            if len(self._small_pending) >= int(self.minimum_input_bytes):
                self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                self._small_pending.clear()
                self._drain(block=False)
        if (
            self._hardware_lookbehind_enabled()
            and 0 < remaining < int(self.minimum_input_bytes)
        ):
            if self._hardware_lookbehind is None:
                self._small_pending.extend(bytes(int(remaining)))
            else:
                self._hardware_lookbehind = (
                    bytes(memoryview(self._hardware_lookbehind).cast('B'))
                    + bytes(int(remaining))
                )
            remaining = 0
        if remaining > 0:
            self._submit_hardware_lookbehind()
        if self._hardware_lookbehind_enabled() and remaining > 0:
            # Keep the logical tail of the zero run unresolved. If the next call is
            # tiny, close() merges it into this hardware-eligible span; if there is no
            # next call, _submit_hardware_lookbehind() emits it from the zero cache.
            retained = int(self.minimum_input_bytes)
            cached_remaining = int(remaining) - retained
            while cached_remaining > 0:
                ln = min(self.chunk_bytes, cached_remaining)
                self._enqueue_completed(_zero_gzip_member(int(ln)))
                cached_remaining -= int(ln)
                self._drain(block=False)
            self._hardware_lookbehind = bytes(retained)
            self._hardware_lookbehind_is_zero = True
            remaining = 0
        while remaining > 0:
            ln = min(self.chunk_bytes, remaining)
            self._enqueue_completed(_zero_gzip_member(int(ln)))
            remaining -= int(ln)
            self._drain(block=False)
        return int(nbytes)

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self._small_pending:
                if self._hardware_lookbehind is not None:
                    combined = bytes(memoryview(self._hardware_lookbehind).cast('B')) + bytes(
                        self._small_pending
                    )
                    self._hardware_lookbehind = None
                    self._hardware_lookbehind_is_zero = False
                    self._small_pending.clear()
                    self._queue_data_chunk(memoryview(combined), owned=True)
                elif len(self._small_pending) < int(self.minimum_input_bytes):
                    pending_bytes = int(len(self._small_pending))
                    raise RuntimeError(
                        f'{self.backend} cannot encode the final {pending_bytes}-byte '
                        f'payload entirely in hardware (minimum {self.minimum_input_bytes})'
                    )
                else:
                    self._queue_data_chunk(memoryview(bytes(self._small_pending)))
                    self._small_pending.clear()
            self._submit_hardware_lookbehind()
            self.closed = True
            self._drain(block=True)
            if self._pending or self._completed or int(self._next_write_sequence) != int(self._next_sequence):
                raise RuntimeError('NRRD member completion map did not drain completely')
        except BaseException:
            self.closed = True
            self._abandon_and_settle()
            raise
        finally:
            self._publish_writer_stats()
            _unregister_nrrd_live_writer(self)

    def _abandon_and_settle(self) -> None:
        """Retain inputs and wait until no failed native/DMA request can still run."""
        self._writer_stats['failed_writers'] = 1
        pending = list(self._pending)
        for fut in pending:
            fut.cancel()
        for fut in pending:
            try:
                fut.result()
            except BaseException:
                pass
        self._pending.clear()
        self._completed.clear()
        self._small_pending.clear()
        self._hardware_lookbehind = None
        self._hardware_lookbehind_is_zero = False
        self._inflight_bytes = 0
        self._diag_pending.clear()
        self._publish_writer_stats()
        _unregister_nrrd_live_writer(self)

    def __enter__(self) -> '_MemberParallelGzipPayloadWriter':
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
            return
        # Broken stream: the file is already failed — abandon pending members.
        self.closed = True
        self._abandon_and_settle()

_NRRD_MEMBER_GZIP_OK: Dict[Tuple[str, int], bool] = {}

_NRRD_MEMBER_GZIP_FAILURE_REASONS: Dict[Tuple[str, int], str] = {}

_NRRD_MEMBER_GZIP_TEST_LOCK = threading.Lock()

_NRRD_MEMBER_GZIP_ANNOUNCED = False

_NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED: set = set()

_NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = False

_NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK = threading.Lock()

def _after_nrrd_compression_fork_child() -> None:
    """Discard parent compressor pools, locks, and policy caches in a fork child."""
    global _NRRD_GZIP_EXECUTORS
    global _NRRD_GZIP_EXECUTOR_LOCK, _NRRD_ZERO_MEMBER_LOCK
    global _NRRD_ZERO_MEMBER_CACHE, _NRRD_MEMBER_CODEC_SETTING_WARNED
    global _NRRD_MEMBER_GZIP_OK, _NRRD_MEMBER_GZIP_FAILURE_REASONS
    global _NRRD_MEMBER_GZIP_TEST_LOCK, _NRRD_MEMBER_GZIP_ANNOUNCED
    global _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED
    global _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED
    global _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK
    global _NRRD_LIVE_WRITERS_LOCK, _NRRD_LIVE_WRITERS
    global _NRRD_LIVE_RETIRED_WAIT_SECONDS

    # No ThreadPoolExecutor worker survives fork(), and any inherited lock may have
    # been held by a vanished parent thread. Never call shutdown() or acquire one of
    # those locks in the child; replace the complete process-local state directly.
    _NRRD_GZIP_EXECUTORS = {}
    _NRRD_GZIP_EXECUTOR_LOCK = threading.Lock()
    _NRRD_ZERO_MEMBER_CACHE = {}
    _NRRD_ZERO_MEMBER_LOCK = threading.Lock()
    _NRRD_MEMBER_CODEC_SETTING_WARNED = False
    _NRRD_MEMBER_GZIP_OK = {}
    _NRRD_MEMBER_GZIP_FAILURE_REASONS = {}
    _NRRD_MEMBER_GZIP_TEST_LOCK = threading.Lock()
    _NRRD_MEMBER_GZIP_ANNOUNCED = False
    _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED = set()
    _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = False
    _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK = threading.Lock()
    _NRRD_LIVE_WRITERS_LOCK = threading.Lock()
    _NRRD_LIVE_WRITERS = weakref.WeakSet()
    _NRRD_LIVE_RETIRED_WAIT_SECONDS = 0.0
    # _NRRD_GZIP_EXECUTOR_ATEXIT_REGISTERED intentionally remains unchanged:
    # Python's atexit registry is inherited, so the child already owns that callback.

if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_nrrd_compression_fork_child)

def _nrrd_codec_runtime_provenance(backend: str) -> Dict[str, object]:
    """Best-effort interpreter and selected CPU codec identity for diagnostics."""
    name = str(backend)
    module_name = {'libdeflate': 'deflate', 'isal': 'isal', 'zlib': 'zlib'}.get(name)
    distribution_names = {
        'libdeflate': ('deflate', 'python-deflate'),
        'isal': ('isal', 'python-isal'),
    }.get(name, ())
    provenance: Dict[str, object] = {
        'python_version': sys.version.split()[0],
        'python_executable': sys.executable,
        'codec_backend': name,
        'codec_module': module_name,
        'codec_module_path': None,
        'codec_distribution': None,
        'codec_distribution_version': None,
        'codec_runtime_version': None,
    }
    if module_name is None:
        return provenance
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = None
    if module is not None:
        module_path = getattr(module, '__file__', None)
        provenance['codec_module_path'] = str(module_path) if module_path else None
        runtime_version = (
            getattr(module, 'ZLIB_RUNTIME_VERSION', None) if name == 'zlib'
            else getattr(module, '__version__', None)
        )
        provenance['codec_runtime_version'] = (
            str(runtime_version) if runtime_version is not None else None
        )
    if name == 'libdeflate':
        known_version = _known_nrrd_libdeflate_version(module) if module is not None else None
        provenance['gil_release_qualification'] = (
            'binding version unavailable; qualify GIL release independently'
            if known_version is None else
            'version compatibility checked; custom builds require independent qualification'
        )
    for distribution_name in distribution_names:
        try:
            version = importlib_metadata.version(distribution_name)
        except Exception:
            continue
        provenance['codec_distribution'] = distribution_name
        provenance['codec_distribution_version'] = str(version)
        break
    return provenance


def _announce_nrrd_cpu_deflate_backend(
    tier: str,
    *,
    codec_spec: NrrdMemberCodecSpec,
) -> None:
    """Report the actual encoder selected after hardware proof/KAT, once per process."""
    global _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED
    if _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED:
        return
    with _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCE_LOCK:
        if _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED:
            return
        backend, level, _compress = codec_spec
        display = {
            'qat': 'Intel QAT/QATzip hardware',
            'iaa': 'Intel IAA/QPL hardware',
            'libdeflate': 'libdeflate (python-deflate)',
            'isal': 'ISA-L (python-isal)',
            'zlib': 'zlib',
        }.get(str(backend), str(backend))
        requested = str(nrrd_member_codec_requested())
        print(
            f'NRRD DEFLATE backend selected: {display}, level={int(level)}, '
            f'policy={requested}, tier={tier}.'
        )
        telemetry = runtime_telemetry()
        telemetry.gauge('nrrd.compression.requested_backend', requested)
        telemetry.gauge('nrrd.compression.selected_backend', str(backend))
        telemetry.gauge('nrrd.compression.effective_level', int(level))
        provenance = _nrrd_codec_runtime_provenance(str(backend))
        print('NRRD DEFLATE runtime provenance: ' + json.dumps(provenance, sort_keys=True))
        telemetry.gauge('nrrd.compression.runtime_provenance', provenance)
        capabilities = getattr(_compress, 'capabilities', None)
        if isinstance(capabilities, dict):
            telemetry.gauge(f'nrrd.compression.{backend}.capabilities', capabilities)
        _NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED = True

def _preflight_nrrd_codec_threads(codec_spec: NrrdMemberCodecSpec) -> None:
    """Initialize every hardware pool worker before any logical NRRD is opened."""
    compressor = codec_spec[2]
    if not bool(getattr(compressor, 'hardware_backend', False)):
        return
    preflight = getattr(compressor, 'preflight_thread_state', None)
    if not callable(preflight):
        raise RuntimeError(
            f'{codec_spec[0]} hardware codec does not expose thread-session preflight'
        )
    workers = int(_nrrd_codec_worker_count(codec_spec))
    executor = _nrrd_gzip_executor(codec_spec)
    _run_on_every_executor_thread(executor, workers, preflight)

def _nrrd_member_codec_test_key(codec_spec: NrrdMemberCodecSpec) -> Tuple[str, int]:
    compressor_identity = getattr(codec_spec[2], 'cache_key', str(codec_spec[0]))
    return (repr(compressor_identity), int(codec_spec[1]))

def _nrrd_member_codec_self_test(
    codec_spec: NrrdMemberCodecSpec,
) -> bool:
    """KAT one codec's complete framing plus the member writer's ordered/zero paths."""
    key = _nrrd_member_codec_test_key(codec_spec)
    cached = _NRRD_MEMBER_GZIP_OK.get(key)
    if cached is not None:
        return bool(cached)
    with _NRRD_MEMBER_GZIP_TEST_LOCK:
        cached = _NRRD_MEMBER_GZIP_OK.get(key)
        if cached is not None:
            return bool(cached)
        failure_reason = ''
        try:
            import gzip as _gzip
            _preflight_nrrd_codec_threads(codec_spec)
            minimum_input = max(
                1, int(getattr(codec_spec[2], 'minimum_input_bytes', 1))
            )
            kat_chunk = max(4096, int(minimum_input))
            # Scale both nonzero phases from the binding's hardware threshold. Fixed
            # kilobyte-sized KAT data would incorrectly reject a valid accelerator whose
            # minimum request is larger. The short tail makes the writer merge it into
            # the preceding chunk instead of accidentally exercising software fallback.
            short_tail = 1 if int(minimum_input) <= 1 else min(3000, minimum_input - 1)
            part_a_size = int(kat_chunk) * 2 + int(short_tail)
            part_a_seed = bytes(range(256)) + b'\x00' * 31
            part_a = (
                part_a_seed * ((int(part_a_size) + len(part_a_seed) - 1) // len(part_a_seed))
            )[:int(part_a_size)]
            part_b = b'\x00' * 8192  # write_zeros: cached members only
            part_c_size = int(kat_chunk) + int(short_tail)
            part_c_seed = b'nrrd-member-gzip-self-test\x00'
            part_c = (
                part_c_seed * ((int(part_c_size) + len(part_c_seed) - 1) // len(part_c_seed))
            )[:int(part_c_size)]
            sink = io.BytesIO()
            writer = _MemberParallelGzipPayloadWriter(
                sink,
                chunk_bytes=int(kat_chunk),
                codec_spec=codec_spec,
            )
            writer.write(part_a)
            writer.write_zeros(len(part_b))
            writer.write(part_c)
            writer.close()
            ok = bool(_gzip.decompress(sink.getvalue()) == part_a + part_b + part_c)
            if not ok:
                failure_reason = 'known-answer gzip round trip returned different bytes'
        except Exception as exc:
            ok = False
            failure_reason = f'{type(exc).__name__}: {exc}'
        if not bool(ok) and bool(getattr(codec_spec[2], 'hardware_backend', False)):
            _retire_nrrd_codec_executor(codec_spec)
        _NRRD_MEMBER_GZIP_OK[key] = bool(ok)
        if ok:
            _NRRD_MEMBER_GZIP_FAILURE_REASONS.pop(key, None)
        else:
            _NRRD_MEMBER_GZIP_FAILURE_REASONS[key] = (
                str(failure_reason) or 'known-answer/round-trip self-test failed'
            )
        return bool(ok)

def _select_nrrd_member_codec(
    *,
    expected_input_bytes: Optional[int] = None,
) -> Optional[NrrdMemberCodecSpec]:
    """First available and KAT-validated configured complete-member codec wins."""
    requested = str(nrrd_member_codec_requested())
    for name in _nrrd_member_codec_candidates():
        codec_loaded = False
        module_missing = False
        old_libdeflate_binding = False
        try:
            spec = _nrrd_member_codec_spec(str(name))
            codec_loaded = True
            minimum_input = max(1, int(getattr(spec[2], 'minimum_input_bytes', 1)))
            if (
                expected_input_bytes is not None
                and int(expected_input_bytes) < int(minimum_input)
            ):
                reason = (
                    f'logical payload is {int(expected_input_bytes)} bytes, below the '
                    f'{minimum_input}-byte hardware-only minimum'
                )
                raise RuntimeError(reason)
            if _nrrd_member_codec_self_test(spec):
                runtime_telemetry().gauge('nrrd.compression.requested_backend', requested)
                runtime_telemetry().gauge('nrrd.compression.selected_backend', str(spec[0]))
                return spec
            reason = _NRRD_MEMBER_GZIP_FAILURE_REASONS.get(
                _nrrd_member_codec_test_key(spec),
                'known-answer/round-trip self-test failed',
            )
        except Exception as exc:
            old_libdeflate_binding = isinstance(exc, _OldNrrdLibdeflateBindingError)
            if isinstance(exc, _OldNrrdLibdeflateBindingError) and requested == 'libdeflate':
                runtime_telemetry().fallback('nrrd.compression.libdeflate', exc)
                raise RuntimeError(
                    'Explicit YOLO_TTA_NRRD_MEMBER_CODEC=libdeflate is incompatible: '
                    f'{exc}'
                ) from exc
            reason = str(exc) or type(exc).__name__
            module_missing = bool(getattr(exc, 'module_missing', False))
        # Missing optional codecs are expected in auto mode. Forced selection and actual
        # codec corruption are announced once, without preventing the lower gzip tiers.
        policy_chain = requested in {'auto', 'cpu'}
        announce = bool(
            not policy_chain
            or codec_loaded
            or old_libdeflate_binding
            or str(name) == 'zlib'
            or (str(name) in {'qat', 'iaa'} and not module_missing)
        )
        runtime_telemetry().fallback(f'nrrd.compression.{name}', RuntimeError(str(reason)))
        failure_key = (str(name), str(reason))
        if announce and failure_key not in _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED:
            _NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED.add(failure_key)
            if policy_chain:
                print(
                    f'Warning: NRRD member codec {name!r} unavailable ({reason}); '
                    'using the next validated NRRD gzip tier.'
                )
            else:
                print(f'Warning: requested NRRD member codec {name!r} failed ({reason}).')
    return None

def _require_nrrd_member_codec(
    *,
    expected_input_bytes: Optional[int] = None,
) -> NrrdMemberCodecSpec:
    member_codec = _select_nrrd_member_codec(
        expected_input_bytes=expected_input_bytes,
    )
    if member_codec is None:
        requested = nrrd_member_codec_requested()
        raise RuntimeError(
            'No validated complete-member NRRD gzip codec is available '
            f'(YOLO_TTA_NRRD_MEMBER_CODEC={requested!r}; tried '
            f'{list(_nrrd_member_codec_candidates())}).'
        )
    return member_codec

def _open_nrrd_payload_writer(
    fh: object,
    *,
    codec_spec: Optional[NrrdMemberCodecSpec] = None,
) -> _MemberParallelGzipPayloadWriter:
    """Open one already-proven or newly selected complete gzip codec.

 Automatic selection tries hardware-only QAT, libdeflate, ISA-L, and zlib in order.
 If every configured codec fails validation, output stops with a specific
 error rather than switching to an unvalidated framing implementation."""
    member_codec = codec_spec if codec_spec is not None else _require_nrrd_member_codec()
    global _NRRD_MEMBER_GZIP_ANNOUNCED
    if not _NRRD_MEMBER_GZIP_ANNOUNCED:
        _NRRD_MEMBER_GZIP_ANNOUNCED = True
        print(
            f'Member-parallel NRRD gzip active: codec={member_codec[0]}, '
            f'level={int(member_codec[1])}, '
            f'{max(1, nrrd_gzip_chunk_bytes() // (1024 * 1024))} MiB members, '
            'whole-layer pipelining.'
        )
    _announce_nrrd_cpu_deflate_backend(
        'member-parallel gzip', codec_spec=member_codec,
    )
    return _MemberParallelGzipPayloadWriter(fh, codec_spec=member_codec)
