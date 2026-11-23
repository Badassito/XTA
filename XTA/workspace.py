"""Memory accounting and process-local pipeline settings."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from time import perf_counter as _perf_counter
from typing import (
    Dict,
    Optional,
)
import numpy as np

from .env_flags import env_flag as _env_flag, env_float as _env_float, env_int as _env_int

# RAM headroom probe reads (/proc/meminfo, /proc/self/{cgroup,mountinfo} and
# each cgroup level's limit/usage/memory.stat). Path.read_text releases the GIL
# around open/fstat/read/close; in the busy main process (hundreds of threads,
# few cores) each release costs about one switch interval, so one probe took
# 12-131 ms (seconds with 6 busy Python threads) and ~1,600 probes per run add
# up to minutes. On Linux the probe reads these kernel-generated files through
# libc with the GIL held (_gil_holding_text_reader): same text and errors, no
# convoy. Every probe stays fresh: it starts after its caller's call, which
# memfd_ram_headroom's "block sample before headroom" rule and every admission
# check rely on, so limits, usage and memory.stat are read on every probe. Only
# the process cgroup membership and the cgroupfs mount verification are resolved
# once per process (and again after fork); unreadable files are not cached.
#
# Kernel-side waits. /proc/meminfo and the cgroup page-counter files only load
# counters. Three files can wait on kernel locks: /proc/self/cgroup
# (cgroup_mutex before Linux 6.8, held across cgroup create/destroy/attach),
# /proc/self/mountinfo (namespace_sem, O(mounts)) and memory.stat (rstat
# flush). With the GIL held such a wait stalls every Python thread, so each read
# class is timed: once one GIL-held read of a class takes longer than the hold
# limit (default: the interpreter switch interval, the time any thread may
# already hold the GIL between forced switches), that class is demoted to the
# GIL-releasing Path.read_text for the rest of the process. A process therefore
# sees at most one over-limit stall per class. YOLO_TTA_RAM_PROBE_GIL_READS=
# counters never holds the GIL for the three lock-prone classes; =0 never holds
# it at all. Releasing every lock-prone read costs ~3 s per two probes with 6
# busy threads (measured), so it is not the default.
_GIL_READER_UNRESOLVED = object()
# The reader in use, None for Path.read_text, or _GIL_READER_UNRESOLVED.
_GIL_READER = _GIL_READER_UNRESOLVED
_PATH_READ_TEXT_ORIGINAL = Path.read_text
_CGROUP_FS_PREFIX = '/sys/fs/cgroup/'
# Counter files whose read handlers only load page_counter/flag fields.
_CGROUP_COUNTER_NAMES = frozenset((
    'memory.max', 'memory.current', 'memory.limit_in_bytes', 'memory.usage_in_bytes',
    'memory.use_hierarchy'))
_LOCK_FREE_PROBE_CLASSES = frozenset(('meminfo', 'cgroup_counter'))
_LOCK_PRONE_PROBE_CLASSES = frozenset(('cgroup_membership', 'mountinfo', 'memory_stat'))
# Classes still read with the GIL held (demotion removes one; set when resolved).
_GIL_HELD_CLASSES = set(_LOCK_FREE_PROBE_CLASSES | _LOCK_PRONE_PROBE_CLASSES)
_GIL_HOLD_LIMIT_SECONDS = 0.005
# Per class: GIL-held read count and longest duration (updated under the GIL;
# a racy max is benign), and the classes demoted in this process.
_GIL_HELD_READ_CALLS: Dict[str, int] = {}
_GIL_HELD_READ_MAX_SECONDS: Dict[str, float] = {}
_GIL_HELD_DEMOTED: Dict[str, float] = {}
# (pid, entries, unresolved_scope); entries are (is_v2, base, limit_name,
# usage_name, leaf) tuples parsed from /proc/self/cgroup.
_CGROUP_LAYOUT: Optional[tuple] = None
# (pid, base_posix, is_v2) -> canonical cgroupfs mount verified.
_CGROUP_MOUNT_VERIFIED: Dict[tuple, bool] = {}


def _probe_read_class(posix: str) -> Optional[str]:
    """Classify a probe pseudo-file; None means never read with the GIL held."""
    if posix == '/proc/meminfo':
        return 'meminfo'
    if posix == '/proc/self/cgroup':
        return 'cgroup_membership'
    if posix == '/proc/self/mountinfo':
        return 'mountinfo'
    if posix.startswith(_CGROUP_FS_PREFIX):
        name = posix.rpartition('/')[2]
        if name == 'memory.stat':
            return 'memory_stat'
        if name in _CGROUP_COUNTER_NAMES:
            return 'cgroup_counter'
    return None


def _gil_releasing_read_text(path: Path) -> str:
    """Plain Path.read_text (releases the GIL); a module hook for tests."""
    return path.read_text()


def ram_probe_gil_read_stats() -> Dict[str, object]:
    """Process-wide GIL-held probe read counts, longest reads and demotions."""
    calls, longest = dict(_GIL_HELD_READ_CALLS), dict(_GIL_HELD_READ_MAX_SECONDS)
    return {'ram_probe_gil_held_reads': int(sum(calls.values())),
            'ram_probe_gil_held_max_seconds': float(max(longest.values(), default=0.0)),
            'ram_probe_gil_held_reads_by_class': calls,
            'ram_probe_gil_held_max_seconds_by_class': longest,
            'ram_probe_gil_held_demoted': dict(_GIL_HELD_DEMOTED),
            'ram_probe_gil_hold_limit_seconds': float(_GIL_HOLD_LIMIT_SECONDS)}


def _probe_read_text(path: Path) -> str:
    # A replaced Path.read_text (tests) always wins over the reader.
    if Path.read_text is _PATH_READ_TEXT_ORIGINAL:
        reader = _GIL_READER
        if reader is _GIL_READER_UNRESOLVED:
            reader = _resolve_gil_holding_reader()
        kind = _probe_read_class(path.as_posix()) if reader is not None else None
        if kind is not None and kind in _GIL_HELD_CLASSES:
            started = _perf_counter()
            try:
                return reader(path)
            finally:
                elapsed = _perf_counter() - started
                _GIL_HELD_READ_CALLS[kind] = _GIL_HELD_READ_CALLS.get(kind, 0) + 1
                if elapsed > _GIL_HELD_READ_MAX_SECONDS.get(kind, 0.0):
                    _GIL_HELD_READ_MAX_SECONDS[kind] = elapsed
                if elapsed > _GIL_HOLD_LIMIT_SECONDS:
                    _GIL_HELD_CLASSES.discard(kind)
                    _GIL_HELD_DEMOTED.setdefault(kind, elapsed)
        return _gil_releasing_read_text(path)
    return path.read_text()


def _read_meminfo_bytes() -> Dict[str, int]:
    info: Dict[str, int] = {}
    try:
        for line in _probe_read_text(Path('/proc/meminfo')).splitlines():
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            raw_val = parts[1].strip().split()[0]
            info[key] = int(raw_val) * 1024
    except Exception:
        pass
    return info

def _cgroup_read_int(path: Path) -> Optional[int]:
    """Read a nonnegative counter; zero differs from an unreadable counter."""
    try:
        raw = _probe_read_text(path).strip()
    except Exception:
        return None
    if not raw.isascii() or not raw.isdigit():
        return None
    try:
        return int(raw)
    except ValueError:
        return None

def _cgroup_reclaimable_file_bytes(node: Path, *, v2: bool) -> int:
    """Return clean inactive file-cache bytes reclaimable by this cgroup."""
    if v2:
        keys = ('inactive_file', 'file_dirty', 'file_writeback')
    else:
        keys = ('total_inactive_file', 'total_dirty', 'total_writeback')
    found: Dict[str, int] = {}
    try:
        for line in _probe_read_text(node / 'memory.stat').splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in keys:
                if (parts[0] in found or not parts[1].isascii()
                        or not parts[1].isdigit()):
                    return 0
                found[parts[0]] = int(parts[1])
    except Exception:
        return 0
    # Missing dirty/writeback counters do not prove that file pages are clean.
    if any(key not in found for key in keys):
        return 0
    return max(0, found[keys[0]] - found[keys[1]] - found[keys[2]])

def _cgroup_canonical_mount_verified(mountinfo: str, base: Path, *, v2: bool) -> bool:
    """Authenticate the simple membership-to-path mapping used below.

    Non-root bind mounts and other layouts retain the conservative RSS fallback.
    """
    matches = []
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(' - ')
        fields, filesystem = before.split(), after.split()
        if separator and len(fields) >= 6:
            mountpoint = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[4])
            # A nested mount can replace a cgroup directory or an individual
            # counter file. Do not infer enforcement through such an overlay.
            if mountpoint.startswith(base.as_posix() + '/'):
                return False
            if mountpoint == base.as_posix():
                matches.append((fields, filesystem))
    if len(matches) != 1:
        return False
    fields, filesystem = matches[0]
    if fields[3] != '/' or len(filesystem) < 3:
        return False
    return (filesystem[0] == 'cgroup2' if v2 else
            filesystem[0] == 'cgroup' and 'memory' in filesystem[2].split(','))


def _cgroup_layout() -> tuple:
    """Return (memory cgroup entries, unresolved scope) for this process.

    Parsed once per process and again after fork (PID change). A read failure
    propagates and is not cached.
    """
    global _CGROUP_LAYOUT
    pid = os.getpid()
    layout = _CGROUP_LAYOUT
    if layout is not None and layout[0] == pid:
        return layout[1], layout[2]
    cgroup_text = _probe_read_text(Path('/proc/self/cgroup'))
    entries = []
    unresolved_scope = False
    for line in cgroup_text.splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        hierarchy_id, controllers, cg_path = parts
        if hierarchy_id == '0' and not controllers:
            # cgroup unified hierarchy.
            is_v2 = True
            base = Path('/sys/fs/cgroup')
            limit_name, usage_name = 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            # cgroup memory controller.
            is_v2 = False
            base = Path('/sys/fs/cgroup/memory')
            limit_name, usage_name = 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        if (not cg_path.startswith('/') or '\\' in cg_path or '//' in cg_path
                or any(part in ('.', '..') for part in cg_path.split('/'))):
            unresolved_scope = True
            continue
        entries.append((is_v2, base, limit_name, usage_name, base / cg_path.lstrip('/')))
    _CGROUP_LAYOUT = (pid, tuple(entries), unresolved_scope)
    return _CGROUP_LAYOUT[1], unresolved_scope


def _cgroup_mount_verified_cached(base: Path, is_v2: bool) -> bool:
    """Verify the canonical cgroupfs mount once per process; failures are not cached."""
    key = (os.getpid(), base.as_posix(), bool(is_v2))
    verified = _CGROUP_MOUNT_VERIFIED.get(key)
    if verified is None:
        try:
            mountinfo = _probe_read_text(Path('/proc/self/mountinfo'))
        except Exception:
            return False
        verified = _cgroup_canonical_mount_verified(mountinfo, base, v2=is_v2)
        _CGROUP_MOUNT_VERIFIED[key] = verified
    return verified


def _reset_cgroup_layout_cache() -> None:
    """Forget the cached cgroup membership and mount verification (test hook)."""
    global _CGROUP_LAYOUT
    _CGROUP_LAYOUT = None
    _CGROUP_MOUNT_VERIFIED.clear()


def _cgroup_memory_budget(*, verify_enforcement: bool = True) -> tuple[Optional[int], Optional[int]]:
    """Return (tightest headroom, verified enforcing cap) for this process.

    All finite ancestors bound headroom. The second value additionally requires
    a verified canonical mount and hierarchical accounting, so callers can use
    it to prove that a separate allocation cap is already enforced. Unknown
    scope never supplies that proof. Unreadable usage under a known finite cap
    grants zero headroom, rather than treating missing usage as zero bytes.
    Headroom-only callers skip mount discovery and never receive a cap proof.
    Limits, usage and memory.stat are read on every call; only the membership
    and mount verification are resolved once per process.
    """
    try:
        entries, unresolved_scope = _cgroup_layout()
    except Exception:
        return None, None
    headroom: Optional[int] = None
    enforcing_cap: Optional[int] = None
    for is_v2, base, limit_name, usage_name, node in entries:
        verified = verify_enforcement and _cgroup_mount_verified_cached(base, is_v2)
        while node == base or base in node.parents:
            try:
                raw_limit = _probe_read_text(node / limit_name).strip()
            except FileNotFoundError:
                # A controller need not expose a limit at every ancestry level.
                raw_limit = None
            except Exception:
                return 0, None
            limit = None
            if raw_limit is not None and not (is_v2 and raw_limit == 'max'):
                if not raw_limit.isascii() or not raw_limit.isdigit():
                    return 0, None
                try:
                    limit = int(raw_limit)
                except ValueError:
                    return 0, None
                # v1 reports unlimited as a huge page-rounded sentinel (~2^63).
                if not is_v2 and limit >= (1 << 60):
                    limit = None
            if limit is not None:
                usage = _cgroup_read_int(node / usage_name)
                if usage is None:
                    return 0, None
                hierarchical = is_v2 or _cgroup_read_int(node / 'memory.use_hierarchy') == 1
                # v1 total_* stats cover descendants; do not subtract those
                # from a non-hierarchical local usage counter.
                reclaimable = (min(usage, _cgroup_reclaimable_file_bytes(node, v2=is_v2))
                               if hierarchical else 0)
                level_headroom = max(0, limit - (usage - reclaimable))
                headroom = level_headroom if headroom is None else min(headroom, level_headroom)
                if verified and hierarchical:
                    enforcing_cap = limit if enforcing_cap is None else min(enforcing_cap, limit)
            if node == base:
                break
            node = node.parent
    return headroom, None if unresolved_scope else enforcing_cap


def _cgroup_memory_headroom_bytes() -> Optional[int]:
    """Return tightest cgroup headroom, excluding reclaimable clean file cache."""
    return _cgroup_memory_budget(verify_enforcement=False)[0]

def available_anon_work_bytes(*, meminfo: Optional[Dict[str, int]] = None) -> int:
    """Read current allocation headroom, optionally sharing this call's meminfo.

    A caller already reading physical availability can supply that same snapshot
    to avoid a duplicate proc read. Cgroup limits and usage are always read anew.
    """
    info = _read_meminfo_bytes() if meminfo is None else meminfo
    mem_avail = int(info.get('MemAvailable', 0))
    swap_free = int(info.get('SwapFree', 0))
    node_avail = max(0, mem_avail + swap_free)
    cgroup_headroom = _cgroup_memory_headroom_bytes()
    if cgroup_headroom is None:
        return node_avail
    # Under a cgroup limit do not count node-wide swap: the cgroup charge is what the OOM killer
    # enforces (cluster cgroups commonly run with memory.swap.max=0 / memsw==mem), and blowing it
    # is an uncatchable SIGKILL that bypasses the MemoryError -> disk-memmap fallback, so err on
    # the tight side.
    return min(node_avail, int(cgroup_headroom))


# In-kernel pseudo filesystems whose reads never wait on storage or a daemon.
_KERNEL_PSEUDO_FSTYPES = frozenset((
    'proc', 'sysfs', 'cgroup', 'cgroup2', 'tmpfs', 'devtmpfs', 'devpts', 'binfmt_misc',
    'securityfs', 'debugfs', 'tracefs', 'bpf', 'pstore', 'configfs', 'fusectl', 'efivarfs',
    'hugetlbfs', 'mqueue', 'autofs', 'rpc_pipefs', 'nsfs', 'selinuxfs'))


def _probe_mounts_kernel_backed(mountinfo: str) -> bool:
    """True when /proc and /sys carry only in-kernel filesystems.

    lxcfs (FUSE) commonly overlays /proc/meminfo in containers; a read there
    waits on a userspace daemon, so it must not run with the GIL held.
    """
    found_proc = False
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(' - ')
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 5 or not filesystem:
            continue
        mountpoint = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[4])
        if mountpoint in ('/proc', '/sys') or mountpoint.startswith(('/proc/', '/sys/')):
            if filesystem[0] not in _KERNEL_PSEUDO_FSTYPES:
                return False
            found_proc = found_proc or (mountpoint == '/proc' and filesystem[0] == 'proc')
    return found_proc


def _gil_holding_text_reader(libc=None):
    """Return a Linux reader for small /proc and cgroupfs files that keeps the GIL.

    _probe_read_text uses it only for files whose _probe_read_class is still in
    _GIL_HELD_CLASSES. These pseudo-files are generated in memory and never
    wait on storage, so the reader calls libc through ctypes.PyDLL (which does
    not release the GIL). The text and errors match Path.read_text: locale
    encoding with strict decoding, universal newlines, and the errno-specific
    OSError subclass. It is
    reentrant (one buffer per call), so any thread may use it. Returns None (use
    Path.read_text) when unavailable or disabled by YOLO_TTA_RAM_PROBE_GIL_READS=0
    or when /proc or /sys carries a non-kernel (e.g. lxcfs/FUSE) mount. ``libc``
    is a test hook for a GIL-holding C runtime on other platforms.
    """
    if libc is None and (not sys.platform.startswith('linux')
                         or not _env_flag('YOLO_TTA_RAM_PROBE_GIL_READS', True)):
        return None
    if libc is None:
        try:
            mountinfo = Path('/proc/self/mountinfo').read_text()
        except Exception:
            return None
        if not _probe_mounts_kernel_backed(mountinfo):
            return None
    try:
        import ctypes
        import errno
        import io
        import locale
        if libc is None:
            libc = ctypes.PyDLL(None, use_errno=True)
        c_open, c_read, c_close = libc.open, libc.read, libc.close
        c_open.argtypes, c_open.restype = (ctypes.c_char_p, ctypes.c_int), ctypes.c_int
        c_read.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
        c_read.restype = ctypes.c_ssize_t
        c_close.argtypes, c_close.restype = (ctypes.c_int,), ctypes.c_int
        encoding = io.text_encoding(None)
        if encoding == 'locale':
            encoding = (locale.getencoding() if hasattr(locale, 'getencoding')
                        else locale.getpreferredencoding(False))
        ''.encode(encoding)  # validate the codec once
    except Exception:
        return None
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_BINARY', 0)
    size = 65536
    create_buffer, string_at, get_errno = ctypes.create_string_buffer, ctypes.string_at, ctypes.get_errno

    def read(path: Path) -> str:
        fd = c_open(os.fsencode(path), flags)
        if fd < 0:
            err = get_errno()
            raise OSError(err, os.strerror(err), str(path))
        try:
            buffer = create_buffer(size)
            chunks = []
            while True:
                count = c_read(fd, buffer, size)
                if count < 0:
                    err = get_errno()
                    if err == errno.EINTR:
                        continue
                    raise OSError(err, os.strerror(err), str(path))
                if count == 0:
                    break
                chunks.append(string_at(buffer, count))
        finally:
            c_close(fd)
        text = b''.join(chunks).decode(encoding)
        return text.replace('\r\n', '\n').replace('\r', '\n') if '\r' in text else text
    return read


def _resolve_gil_holding_reader():
    """Pick the probe reader once per process (a benign race may build it twice)."""
    global _GIL_READER, _GIL_HOLD_LIMIT_SECONDS
    reader = _GIL_READER
    if reader is _GIL_READER_UNRESOLVED:
        reader = _gil_holding_text_reader()
        if reader is not None:
            limit_ms = _env_float('YOLO_TTA_RAM_PROBE_GIL_HOLD_LIMIT_MS', 0.0)
            _GIL_HOLD_LIMIT_SECONDS = (limit_ms / 1e3 if limit_ms > 0
                                       else float(sys.getswitchinterval()))
            if os.environ.get('YOLO_TTA_RAM_PROBE_GIL_READS', '').strip().lower() == 'counters':
                _GIL_HELD_CLASSES.difference_update(_LOCK_PRONE_PROBE_CLASSES)
        _GIL_READER = reader
    return reader


_V1613_FAST_BUNDLE_ACTIVE = False

_V1613_D1_PIPELINE_ACTIVE = False

_YOLO_HOLE_FILL_ACTIVE = False

def configure_pipeline_modes(*, fast_bundle_active: bool, d1_pipeline_active: bool,
                             enable_yolo_hole_fill: bool = False) -> None:
    """Publish resolved run modes within this process.

    Spawned workers receive the resolved values in their initialization payload instead of
    inheriting user-overridable environment state.
    """
    global _V1613_FAST_BUNDLE_ACTIVE, _V1613_D1_PIPELINE_ACTIVE, _YOLO_HOLE_FILL_ACTIVE
    _V1613_FAST_BUNDLE_ACTIVE = bool(fast_bundle_active)
    _V1613_D1_PIPELINE_ACTIVE = bool(
        _V1613_FAST_BUNDLE_ACTIVE and d1_pipeline_active
    )
    _YOLO_HOLE_FILL_ACTIVE = bool(enable_yolo_hole_fill)


def yolo_hole_fill_enabled() -> bool:
    """The resolved CLI opt-in; independent of every inference fast-path proof."""
    return bool(_YOLO_HOLE_FILL_ACTIVE)

def v1613_fast_bundle_requested() -> bool:
    """Enable the command-specialized v16.1.3 pipeline unless explicitly disabled."""
    return _env_flag('YOLO_TTA_V1613_FAST_BUNDLE', True)

def v1613_fast_bundle_active() -> bool:
    """True after ``main`` proves the current command satisfies the fast-path contract."""
    return bool(_V1613_FAST_BUNDLE_ACTIVE)

def v1613_d1_pipeline_active() -> bool:
    """Project, infer, treat 2D topology, and backproject inside each eligible GPU lease."""
    return bool(_V1613_FAST_BUNDLE_ACTIVE and _V1613_D1_PIPELINE_ACTIVE)

def proto_hole_treatment_mode() -> str:
    """Resident-proto topology treatment: ``off`` or bounded binary ``closing``."""
    if not yolo_hole_fill_enabled():
        return 'off'
    default = 'close' if v1613_fast_bundle_active() else 'off'
    raw = os.environ.get('YOLO_TTA_PROTO_HOLE_TREATMENT', default).strip().lower()
    aliases = {
        '': default, '0': 'off', 'false': 'off', 'none': 'off', 'disabled': 'off',
        '1': 'close', 'true': 'close', 'close': 'close', 'closing': 'close',
    }
    return aliases.get(raw, default)

def proto_hole_treatment_radius() -> int:
    """Square-neighborhood radius used by resident proto closing."""
    if not yolo_hole_fill_enabled():
        return 0
    default = 2 if v1613_fast_bundle_active() else 0
    return max(0, min(8, _env_int('YOLO_TTA_PROTO_HOLE_RADIUS', default)))

_TILTED_IDENTITY_M = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

def _tilted_grid_is_identity(M_grid_to_src: np.ndarray, grid_h: int, grid_w: int, view: object) -> bool:
    """True when the output grid IS the native tilted raster (no in-plane resample)."""
    if int(grid_h) != int(view.src_h) or int(grid_w) != int(view.src_w):
        return False
    matrix = np.asarray(M_grid_to_src, dtype=np.float32).reshape(2, 3)
    return bool(np.allclose(matrix, _TILTED_IDENTITY_M, atol=1e-6))

def _slurm_allocated_cpu_count() -> Optional[int]:
    for env_name in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_CPUS_PER_NODE'):
        raw = os.environ.get(env_name, '').strip()
        if not raw:
            continue
        m = re.search(r'(\d+)', raw)
        if m is not None:
            try:
                return max(1, int(m.group(1)))
            except Exception:
                continue
    return None

def _cpu_count() -> int:
    if os.name == 'nt':
        from .runtime import _current_process_cpu_affinity

        affinity = _current_process_cpu_affinity()
    else:
        try:
            affinity = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            affinity = None
    if affinity:
        return len(affinity)

    slurm_cpus = _slurm_allocated_cpu_count()
    if slurm_cpus is not None:
        return max(1, int(slurm_cpus))

    return max(1, int(os.cpu_count() or 1))
