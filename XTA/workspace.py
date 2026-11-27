"""Memory accounting and process-local pipeline settings."""

from __future__ import annotations

import io
import locale
import os
import re
import sys
from pathlib import Path
from typing import (
    Dict,
    Optional,
)
import numpy as np

from .env_flags import env_flag as _env_flag, env_float as _env_float, env_int as _env_int

# Membership and canonical mount proofs are stable for this process. Every
# pressure observation below reads fresh counters. A fork discovers its own scope.
_MEMINFO_PATH = Path('/proc/meminfo')
_CGROUP_MEMBERSHIP_PATH = Path('/proc/self/cgroup')
_MOUNTINFO_PATH = Path('/proc/self/mountinfo')
_CGROUP_V1_BASE = Path('/sys/fs/cgroup/memory')
_CGROUP_V2_BASE = Path('/sys/fs/cgroup')
_CGROUP_LAYOUT: Optional[tuple] = None
_CGROUP_MOUNT_VERIFIED: Dict[tuple, bool] = {}


def _default_encoding():
    encoding = io.text_encoding(None)
    if encoding == 'locale':
        return locale.getencoding() if hasattr(locale, 'getencoding') else locale.getpreferredencoding(False)
    return encoding


def _text(raw, encoding):
    if isinstance(raw, bytes):
        return raw.decode(encoding).replace('\r\n', '\n').replace('\r', '\n')
    return raw


def _meminfo(raw, encoding):
    info = {}
    try:
        for line in _text(raw, encoding).splitlines():
            parts = line.split(':', 1)
            if len(parts) == 2:
                info[parts[0].strip()] = int(parts[1].strip().split()[0]) * 1024
    except (AttributeError, UnicodeError, ValueError, IndexError):
        pass
    return info


def _reclaimable(raw, v2, encoding):
    keys = ('inactive_file', 'file_dirty', 'file_writeback') if v2 else (
        'total_inactive_file', 'total_dirty', 'total_writeback')
    found = {}
    try:
        for line in _text(raw, encoding).splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in keys:
                if parts[0] in found or not parts[1].isascii() or not parts[1].isdigit():
                    return 0
                found[parts[0]] = int(parts[1])
        return max(0, found[keys[0]] - found[keys[1]] - found[keys[2]])
    except (AttributeError, UnicodeError, KeyError, ValueError):
        return 0


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
    cgroup_text = _CGROUP_MEMBERSHIP_PATH.read_text()
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
            base = _CGROUP_V2_BASE
        elif 'memory' in controllers.split(','):
            # cgroup memory controller.
            is_v2 = False
            base = _CGROUP_V1_BASE
        else:
            continue
        if (not cg_path.startswith('/') or '\\' in cg_path or '//' in cg_path
                or any(part in ('.', '..') for part in cg_path.split('/'))):
            unresolved_scope = True
            continue
        node = base / cg_path.lstrip('/')
        ancestors = []
        while node == base or base in node.parents:
            ancestors.append(str(node))
            if node == base:
                break
            node = node.parent
        entries.append((is_v2, base, tuple(ancestors)))
    _CGROUP_LAYOUT = (pid, tuple(entries), unresolved_scope)
    return _CGROUP_LAYOUT[1], unresolved_scope


def _reset_cgroup_layout_cache() -> None:
    global _CGROUP_LAYOUT
    _CGROUP_LAYOUT = None
    _CGROUP_MOUNT_VERIFIED.clear()


def _read_optional(path):
    try:
        return path.read_text()
    except Exception:
        return None


def _portable_capture(meminfo, plan, mountinfo, encoding, digit_limit):
    """Portable acquisition with the same ordered file and failure semantics."""
    info = _read_optional(meminfo) if meminfo is not None else None
    groups = []
    mounts = {}
    if callable(plan):
        plan = plan()
    for v2, need_mount, nodes in plan:
        levels = []
        mount = None
        if need_mount:
            key = (v2, nodes[-1] if nodes else None)
            mount = mounts.get(key)
            if mount is None:
                mount = _read_optional(mountinfo)
                if mount is not None:
                    mounts[key] = mount
        groups.append((mount, levels))
        for path in nodes:
            node = Path(path)
            try:
                raw = (node / ('memory.max' if v2 else 'memory.limit_in_bytes')).read_text().strip()
            except FileNotFoundError:
                continue
            except Exception:
                return info, groups, True
            if v2 and raw == 'max':
                continue
            if not raw.isascii() or not raw.isdigit():
                return info, groups, True
            try:
                cap = int(raw)
            except ValueError:
                return info, groups, True
            if not v2 and cap >= 1 << 60:
                continue
            usage = (_read_optional(node / ('memory.current' if v2 else 'memory.usage_in_bytes')) or '').strip()
            if not usage.isascii() or not usage.isdigit():
                return info, groups, True
            try:
                int(usage)
            except ValueError:
                return info, groups, True
            hierarchical = v2
            if not v2:
                flag = (_read_optional(node / 'memory.use_hierarchy') or '').strip()
                try:
                    hierarchical = flag.isascii() and flag.isdigit() and int(flag) == 1
                except ValueError:
                    hierarchical = False
            levels.append((raw, usage, hierarchical,
                           _read_optional(node / 'memory.stat') if hierarchical else None))
    return info, groups, False


def _sample_with(capture, *, include_meminfo, include_cgroup, verify_enforcement):
    pid = os.getpid()
    entries = ()
    proofs = ()
    unresolved = False

    def prepare():
        nonlocal entries, proofs, unresolved
        try:
            entries, unresolved = _cgroup_layout()
        except Exception:
            return ()
        proofs = tuple(_CGROUP_MOUNT_VERIFIED.get((pid, base, v2)) if verify_enforcement else False
                       for v2, base, _ in entries)
        return tuple((v2, proof is None, nodes) for (v2, _, nodes), proof in zip(entries, proofs))

    # On first use, discover membership after reading meminfo, exactly where
    # the cgroup-only portion of the original probe started. Ordinary samples
    # use the cached paths and make one uninterrupted native I/O transaction.
    plan = ()
    if include_cgroup:
        plan = prepare() if _CGROUP_LAYOUT is not None and _CGROUP_LAYOUT[0] == pid else prepare
    encoding = _default_encoding()
    digit_limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    meminfo, observations, failed = capture(
        _MEMINFO_PATH if include_meminfo else None, plan, _MOUNTINFO_PATH, encoding, digit_limit)
    info = _meminfo(meminfo, encoding) if include_meminfo else {}
    headroom = enforcing = None
    for (v2, base, _), previous, (mount, levels) in zip(entries, proofs, observations):
        key = (pid, base, v2)
        verified = bool(previous)
        if mount is not None:
            try:
                verified = _cgroup_canonical_mount_verified(_text(mount, encoding), base, v2=v2)
                _CGROUP_MOUNT_VERIFIED[key] = verified
            except UnicodeError:
                pass  # A failed read/decoding never becomes a cached proof.
        if failed:
            continue  # Preserve completed immutable discovery even on a failed pressure read.
        for cap_raw, usage_raw, hierarchical, stat in levels:
            try:
                cap = int(_text(cap_raw, encoding).strip())
                usage = int(_text(usage_raw, encoding).strip())
            except (UnicodeError, ValueError):
                return info, 0, None
            reclaimable = min(usage, _reclaimable(stat, v2, encoding)) if hierarchical else 0
            remaining = max(0, cap - (usage - reclaimable))
            headroom = remaining if headroom is None else min(headroom, remaining)
            if verified and hierarchical:
                enforcing = cap if enforcing is None else min(enforcing, cap)
    return (info, 0, None) if failed else (info, headroom, None if unresolved else enforcing)


def _portable_memory_sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
    return _sample_with(_portable_capture, include_meminfo=include_meminfo, include_cgroup=include_cgroup,
                        verify_enforcement=verify_enforcement)


def _native_memory_sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
    from .native_cpu import kernels
    return _sample_with(kernels.sample_memory_files, include_meminfo=include_meminfo,
                        include_cgroup=include_cgroup, verify_enforcement=verify_enforcement)


def _memory_sample(*, include_meminfo, include_cgroup, verify_enforcement=False):
    capture = _native_memory_sample if sys.platform.startswith('linux') else _portable_memory_sample
    return capture(include_meminfo=include_meminfo, include_cgroup=include_cgroup,
                   verify_enforcement=verify_enforcement)


def _read_meminfo_bytes() -> Dict[str, int]:
    return _memory_sample(include_meminfo=True, include_cgroup=False)[0]


def _cgroup_memory_budget(*, verify_enforcement: bool = True) -> tuple[Optional[int], Optional[int]]:
    """Fresh minimum across all finite cgroups, with an independently verified enforcing cap."""
    return _memory_sample(include_meminfo=False, include_cgroup=True, verify_enforcement=verify_enforcement)[1:]


def _cgroup_memory_headroom_bytes() -> Optional[int]:
    return _cgroup_memory_budget(verify_enforcement=False)[0]


def _anon_headroom(info, cgroup_headroom):
    node = max(0, int(info.get('MemAvailable', 0)) + int(info.get('SwapFree', 0)))
    return node if cgroup_headroom is None else min(node, cgroup_headroom)


def available_anon_work_bytes(*, meminfo: Optional[Dict[str, int]] = None) -> int:
    """Sample meminfo and cgroups together unless this caller already owns the meminfo observation."""
    if meminfo is not None:
        return _anon_headroom(meminfo, _cgroup_memory_headroom_bytes())
    info, headroom, _ = _memory_sample(include_meminfo=True, include_cgroup=True)
    return _anon_headroom(info, headroom)


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
