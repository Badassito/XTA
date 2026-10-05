"""Amortize full collection in tiny in-process CPU scheduling fixtures only."""
from types import SimpleNamespace
from unittest import mock

from XTA import lta_worker_adapter


def batch_fake_worker_collections(testcase):
    """Record worker collection requests, restore its module, collect at teardown.

    These fixtures own no CUDA predictor or allocator. Artifact writes, fsync,
    checksums, SQLite and resource/trace cleanup still execute normally. The
    separate worker-adapter tests retain real per-window full collection.
    No global ``gc`` module function or process-wide fixture is replaced.
    """
    requested = SimpleNamespace(count=0)
    original_gc = lta_worker_adapter.gc

    def collect():
        requested.count += 1
        return 0

    patcher = mock.patch.object(lta_worker_adapter, 'gc', SimpleNamespace(collect=collect))
    patcher.start()
    testcase.addCleanup(original_gc.collect)
    testcase.addCleanup(patcher.stop)
    return requested
