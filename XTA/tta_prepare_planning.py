"""TTA parent preparation admission and CPU input lookahead policy.

Planning is shared by policy preflight and actual scheduler dispatch. It must
resolve bounded scratch before admission and enter capture resources only while
the caller holds that reservation.
"""
from __future__ import annotations

import math
import os
from typing import Dict

from .config import GIB
from .interpolation import _view_uses_interpolation

PARENT_CONFIDENCE_CAPTURE_WORKSPACE_BYTES = 256 * 1024**2


def _parent_confidence_capture_plan(shape, workers, *, enabled):
    """Resolve bounded capture scratch before its parent admission is requested."""
    if not enabled:
        return None
    from .confidence_evidence import plan_confidence_capture
    from .native_pull_scheduling import get_native_pull_scheduler
    scheduler = get_native_pull_scheduler()
    if scheduler is not None:
        # This is a memory window, charged before parent admission, rather than
        # a private CPU allocation. Actual capture jobs share the run scheduler
        # with native projection. A queued parent must not freeze this to its
        # executor's one-slice cleanup budget, nor upgrade it after admission.
        workers = scheduler.max_workers
    return plan_confidence_capture(tuple(int(v) for v in shape), int(workers),
        workspace_bytes=PARENT_CONFIDENCE_CAPTURE_WORKSPACE_BYTES)


def _prepare_parent_with_confidence_capture(prepare, *, capture_plan, **kwargs):
    """Enter capture's CPU budget inside AdmittedViewPrepare's live reservation."""
    if capture_plan is None:
        return prepare(**kwargs)
    from .confidence_evidence import confidence_capture_resources
    with confidence_capture_resources(capture_plan):
        return prepare(**kwargs)


def _parent_prepare_transient_bytes(view, processing_bytes, source_bytes, *,
                                   interpolation_enabled, confidence_workspace_bytes=0,
                                   packed_native_source=False, native_shape=None,
                                   source_shape=None, cleanup_workers=1,
                                   cleanup_is_simple=False):
    """Use one estimate for actual reservations and policy pre-dispatch planning."""
    if packed_native_source and not interpolation_enabled and cleanup_is_simple:
        from .interpolation import CTILE_INDEX_DTYPE
        from .packed_dense_source import DENSE_PACKED_PAYLOAD_BYTES, DENSE_PACKED_CONTROL_BYTES
        from .workspace import _env_int
        native = tuple(int(value) for value in native_shape or ())
        output = tuple(int(value) for value in source_shape or ())
        if (len(native) != 3 or len(output) != 3 or min((*native, *output)) <= 0
                or math.prod(native) != int(processing_bytes)
                or math.prod(output) != int(source_bytes)):
            raise ValueError('Packed parent transient admission requires its actual native/source shapes')
        mib = 1024**2
        # The dense input remains charged to its existing lease while packing.
        # Charge the complete worst-case cropped row-bit payload during that
        # overlap, including padding at the end of EVERY row, not just the file.
        packed = native[0] * native[1] * ((native[2] + 7) // 8)
        records = 4 * (native[0] + output[0]) * int(CTILE_INDEX_DTYPE.itemsize)
        states = native[0] + output[0]
        # These are the same configured caps enforced by the encoded projector;
        # large user overrides cannot silently retain the default admission.
        plan = max(1, _env_int('YOLO_TTA_NATIVE_PULL_PLAN_MIB', 64)) * mib
        workers = max(1, _env_int('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', 256)) * mib
        # Only precleaned/zero-threshold masks qualify for this estimate. Their
        # remaining per-slice work is OpenCV u8/int32/boolean hole filling or
        # SciPy boolean hole filling, with no EDT/radius/confidence filtering.
        # Keep a conservative 32 B/pixel per active cleanup reader, and include
        # the unpacked device row metadata that can coexist with publication.
        cleanup = 32 * native[1] * native[2] * max(1, min(int(cleanup_workers), native[0]))
        metadata = native[0] * native[1]
        # Row codecs normalize bounded slabs; an exceptionally wide single row
        # can exceed their usual 1 MiB target and must still be charged.
        row_codec = 6 * (max(mib, native[2]) + max(mib, output[2]))
        # The terminal encoder borrows the dense mask but owns one bounded
        # packed batch (at most 64 slices), including metadata and offsets.
        source_pack = DENSE_PACKED_PAYLOAD_BYTES + DENSE_PACKED_CONTROL_BYTES
        return (max(2 * GIB, cleanup) + packed + records + states + plan + workers
                + metadata + row_codec + source_pack + int(confidence_workspace_bytes))
    name = str(view.name).lower()
    transient = 2 * GIB
    if 'tilted' in name:
        transient = int(source_bytes) + 4 * GIB
    elif interpolation_enabled:
        transient = 2 * int(processing_bytes) + 4 * GIB
    return transient + int(confidence_workspace_bytes)


def _policy_parent_prepare_transient_bytes(tasks, source_bytes, workers, *,
                                          interpolation_distance, retain_confidence,
                                          extrapolation_distance=0, sam_image_phase_bytes=0):
    """Charge the same capture scratch before policy parent dispatch begins."""
    largest = 2 * GIB
    plans = {}
    for task in tasks:
        if str(task['kind']) != 'fullframe':
            continue
        shape = tuple(int(v) for v in task['processing_shape'])
        enabled = bool(retain_confidence) and 'd1_output_shape' not in task
        key = (shape, enabled)
        if key not in plans:
            plans[key] = _parent_confidence_capture_plan(shape, workers, enabled=enabled)
        plan = plans[key]
        workspace = 0 if plan is None else int(plan.workspace_bytes)
        interpolates = (int(extrapolation_distance) > 0 or
            _view_uses_interpolation(task['view'], int(interpolation_distance)))
        transient = _parent_prepare_transient_bytes(task['view'], math.prod(shape),
            source_bytes, interpolation_enabled=interpolates, confidence_workspace_bytes=workspace)
        if sam_image_phase_bytes and (int(extrapolation_distance) > 0 or int(interpolation_distance) > 0):
            from .sam_resources import sam_parent_image_reservation_bytes
            transient += sam_parent_image_reservation_bytes(transient,
                4*GIB if interpolates else 0, sam_image_phase_bytes)
        largest = max(largest, transient)
    return largest


def _cpu_input_prefetch_policy(gpu_worker_count: int, worker_budget: int) -> Dict[str, int]:
    """Bound optional CPU lookahead independently of its RAM grant/executor."""
    devices, budget = max(1, int(gpu_worker_count)), max(1, int(worker_budget))
    def positive_override(name, default):
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be a positive integer') from None
        if parsed < 1:
            raise ValueError(f'{name} must be a positive integer')
        return parsed
    return dict(max_builders=min(budget, positive_override('YOLO_TTA_CPU_INPUT_BUILDERS', devices)),
                max_packets=positive_override('YOLO_TTA_CPU_INPUT_PACKETS', 2 * devices))
