"""Dependency-light admission and task-local budgets for native confidence capture."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math


_CONTROL_BYTES = 256 * 1024
_CODEC_BYTES = 1024 * 1024
_CELL_RECORD_BYTES = 768
_CAPTURE_PLAN = ContextVar('native_confidence_capture_plan', default=None)


@dataclass(frozen=True)
class NativeConfidenceCapturePlan:
    shape: tuple[int, int, int]
    block_size: int
    requested_workers: int
    workers: int
    workspace_bytes: int
    workspace_limit_bytes: int
    fixed_bytes: int
    worker_bytes: int
    consumer_bytes: int
    max_pending_frames: int
    initialization_peak_bytes: int


def plan_confidence_capture(shape, workers, *, workspace_bytes=64 * 1024**2,
                            block_size=128):
    """Charge a complete ordered frame window without allocating numeric data.

    Source mask/scores remain borrowed. Each worker retains at most one frame's
    compressed cells, one cell copy/bytes pair, metadata and bounded zlib state.
    The consumer's just-published frame is charged separately from pending work.
    """
    shape = tuple(shape)
    if (len(shape) != 3 or any(isinstance(value, bool) or int(value) != value
                               or not 0 < int(value) < 2**31 for value in shape)):
        raise ValueError('Confidence capture needs three positive int32 dimensions')
    shape = tuple(map(int, shape))
    requested = int(workers)
    block = int(block_size)
    budget = int(workspace_bytes)
    if requested < 1 or not 1 <= block <= 65535 or budget <= 0:
        raise ValueError('Confidence capture workers/block/workspace must be positive and bounded')
    cells = math.ceil(shape[1] / block) * math.ceil(shape[2] / block)
    cell_bytes = min(block, shape[1]) * min(block, shape[2])
    zlib_extra = (cell_bytes >> 12) + (cell_bytes >> 14) + (cell_bytes >> 25) + 13
    if cell_bytes + zlib_extra >= 2**32:
        raise ValueError('Confidence cell compression bound exceeds its uint32 payload length')
    encoded_frame = shape[1] * shape[2] + cells * (max(64, zlib_extra) + _CELL_RECORD_BYTES)
    worker_bytes = encoded_frame + _CODEC_BYTES + 2 * cell_bytes
    consumer_bytes = encoded_frame
    fixed_bytes = _CONTROL_BYTES + 33 * shape[0]
    # Integer-list conversions, snapshots and full validation scratch need
    # at most 137/frame; retain a conservative 160/frame envelope.
    initialization_peak = _CONTROL_BYTES + 160 * shape[0]
    if initialization_peak > budget:
        raise MemoryError('Confidence capture metadata initialization exceeds its workspace budget')
    available = budget - fixed_bytes - consumer_bytes
    admitted = min(requested, shape[0], max(0, available // worker_bytes))
    if admitted < 1:
        raise MemoryError('Confidence capture cannot admit one worker plus its ordered consumer frame')
    charged = max(initialization_peak, fixed_bytes + consumer_bytes + admitted * worker_bytes)
    return NativeConfidenceCapturePlan(shape, block, requested, admitted, charged, budget,
                                        fixed_bytes, worker_bytes, consumer_bytes, admitted, initialization_peak)


@contextmanager
def confidence_capture_resources(plan):
    """Bind one admitted parent budget to its synchronous retiring-score capture."""
    if plan is not None and not isinstance(plan, NativeConfidenceCapturePlan):
        raise TypeError('Confidence capture resources require an admitted plan')
    if plan is not None and plan != plan_confidence_capture(plan.shape, plan.requested_workers,
            workspace_bytes=plan.workspace_limit_bytes, block_size=plan.block_size):
        raise ValueError('Confidence capture plan differs from its admission proof')
    token = _CAPTURE_PLAN.set(plan)
    try:
        yield plan
    finally:
        _CAPTURE_PLAN.reset(token)


def current_confidence_capture_plan():
    return _CAPTURE_PLAN.get()
