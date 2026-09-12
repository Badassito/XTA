"""CPU fault injection at the asynchronous CUDA coverage ownership boundary."""
from __future__ import annotations

from contextlib import nullcontext
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import tta_augmentation_retirement as retirement


def _cuda_fixture(monkeypatch, *, fail_at=None, fail_fence=False):
    trace = []
    device = SimpleNamespace(type='cuda')
    host_refs = []

    class Stream:
        def wait_event(self, event):
            trace.append('wait_ready')

        def synchronize(self):
            # A live local/closure must own the pinned destination until fencing.
            assert all(reference() is not None for reference in host_refs)
            trace.append('stream_fenced')
            if fail_fence:
                raise RuntimeError('injected fence failure')

    stream = Stream()
    event_count = 0

    class Event:
        def __init__(self):
            nonlocal event_count
            self.index = event_count
            event_count += 1
            if self.index == 1 and fail_at == 'event_creation':
                raise RuntimeError('injected event_creation')

        def record(self, selected_stream):
            if self.index == 1 and fail_at == 'event_record':
                raise RuntimeError('injected event_record')

        def synchronize(self):
            trace.append('event_fenced')

    class Tensor:
        shape = (1, 8, 1)

        def __init__(self):
            self.device = device

        def record_stream(self, selected_stream):
            if fail_at == 'record_stream':
                raise RuntimeError('injected record_stream')

    class Host:
        def copy_(self, packed, *, non_blocking):
            assert non_blocking
            trace.append('copy_enqueued')

        def numpy(self):
            return np.ones((1, 8, 1), dtype=np.uint8)

    def host_buffer(*args, **kwargs):
        assert kwargs['pin_memory'] is True
        host = Host()
        host_refs.append(weakref.ref(host))
        return host

    fake_torch = SimpleNamespace(
        stack=lambda values: Tensor(), empty=host_buffer, uint8=object(),
        cuda=SimpleNamespace(Event=Event, Stream=lambda **kwargs: stream,
                             current_stream=lambda selected_device: stream,
                             stream=lambda selected_stream: nullcontext()))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    if fail_at in ('executor_creation', 'executor_submit'):
        class FailingExecutor:
            def __init__(self, **kwargs):
                if fail_at == 'executor_creation':
                    raise RuntimeError('injected executor_creation')

            def submit(self, operation):
                raise RuntimeError('injected executor_submit')

            def shutdown(self, *, wait):
                trace.append('executor_closed')

        monkeypatch.setattr(retirement, 'ThreadPoolExecutor', FailingExecutor)

    replay = SimpleNamespace(valid=SimpleNamespace(device=device), pack_validity_tensor=lambda: Tensor())
    return trace, replay


@pytest.mark.parametrize('fail_at', [
    'record_stream', 'event_creation', 'event_record', 'executor_creation', 'executor_submit',
])
def test_post_copy_setup_failure_fences_pinned_destination(monkeypatch, fail_at):
    trace, replay = _cuda_fixture(monkeypatch, fail_at=fail_at)
    transfers = retirement.CoverageTransfers()
    writer = SimpleNamespace(put_array=lambda *args: pytest.fail('failed setup must not publish support'))
    try:
        with pytest.raises(RuntimeError, match=f'injected {fail_at}'):
            transfers.submit(writer, [(0, object(), 101, replay)])
        assert trace.index('copy_enqueued') < trace.index('stream_fenced')
        assert not transfers.pending
    finally:
        transfers.close()


def test_failed_fence_reports_original_setup_error_and_fence_failure(monkeypatch):
    trace, replay = _cuda_fixture(monkeypatch, fail_at='event_creation', fail_fence=True)
    transfers = retirement.CoverageTransfers()
    try:
        with pytest.raises(RuntimeError, match='asynchronous copy/publication could not be drained') as failure:
            transfers.submit(SimpleNamespace(), [(0, object(), 101, replay)])
        assert 'injected event_creation' in str(failure.value)
        assert 'injected fence failure' in str(failure.value.__cause__)
        assert trace[-1] == 'stream_fenced'
    finally:
        transfers.close()


def test_close_drains_remaining_commit_after_first_commit_failure(monkeypatch):
    trace, replay = _cuda_fixture(monkeypatch)
    transfers = retirement.CoverageTransfers()
    committed = []

    def commit(slot, spec, seed, packed):
        assert 'event_fenced' in trace
        committed.append(slot)
        if slot == 0:
            raise RuntimeError('injected first commit failure')

    writer = SimpleNamespace(put_array=commit)
    transfers.submit(writer, [(0, object(), 100, replay)])
    transfers.submit(writer, [(1, object(), 101, replay)])
    with pytest.raises(RuntimeError, match='injected first commit failure'):
        transfers.close()
    assert sorted(committed) == [0, 1]
    assert not transfers.pending
    assert transfers.executor is None
    assert transfers.stream is None
