"""Optional CPU rendering never fills its RAM grant with unbounded executor work."""
from pathlib import Path
import threading

import numpy as np
import pytest

from XTA.yolo_cpu_input import CpuInputPrefetch
from tests.test_yolo_cpu_input import ManualExecutor, task


def descriptors(count):
    # Independent parents preserve the production round-robin order without
    # changing the small scientific renderer used by these ownership tests.
    return [dict(task(task_id=i, count=1), job_id=f'a{i}') for i in range(count)]


def idle(owner):
    with owner.condition:
        assert owner.condition.wait_for(
            lambda: not owner.pump_scheduled and not owner.pump_requested, timeout=5)


def prefetch(request, tasks, *, close=lambda array: None, allocate=None, **limits):
    executor = ManualExecutor()
    owner = CpuInputPrefetch(tasks, np.zeros((5, 9, 10), np.uint8), executor,
        max_bytes=64 * 1024**3, headroom=lambda: 64 * 1024**3,
        allocate=allocate or (lambda shape, *a, **kw: np.empty(shape, np.uint8)),
        backing=lambda array: Path('/proc/fake/input'), close_array=close, **limits)
    executor.owner = owner
    request.addfinalizer(owner.shutdown)
    idle(owner)
    return owner, executor


def test_default_limits_bound_submissions_independently_of_large_ram_grant(request):
    owner, executor = prefetch(request, descriptors(160))
    state = owner.snapshot()
    assert state['max_builders'] == 4 and state['max_packets'] == 8
    assert len(executor.jobs) == state['building_packets'] == state['charged_packets'] == 4
    assert state['pending_packets'] == 156
    assert [args[0].task['task_id'] for _, _, args in executor.jobs] == list(range(4))
    fallback = dict(descriptors(160)[100])
    owner.attach_ready(fallback)
    assert 'cpu_input_path' not in fallback  # Optional lookahead never becomes a wait.


def test_completed_ready_and_borrowed_packets_bound_refill_until_actual_retirement(request):
    tasks = descriptors(12)
    owner, executor = prefetch(request, tasks, max_builders=2, max_packets=4)
    assert len(executor.jobs) == 2
    for index in range(4):
        executor.run(index)
    state = owner.snapshot()
    assert len(executor.jobs) == 4
    assert state['building_packets'] == 0 and state['ready_packets'] == 4
    assert state['charged_packets'] == 4 and state['pending_packets'] == 8
    owner.attach_ready(tasks[0])
    assert 'cpu_input_path' in tasks[0]
    owner.pump()
    idle(owner)
    assert len(executor.jobs) == 4
    owner.retire(tasks[0])
    idle(owner)
    assert len(executor.jobs) == 5
    assert executor.jobs[4][2][0].task['task_id'] == 4
    assert owner.snapshot()['charged_packets'] == 4


def test_retiring_owner_keeps_packet_slot_while_close_is_in_progress(request):
    entered, release = threading.Event(), threading.Event()
    def close(array):
        entered.set()
        assert release.wait(5)
    tasks = descriptors(3)
    owner, executor = prefetch(request, tasks, close=close, max_builders=2, max_packets=1)
    executor.run()
    try:
        owner.retire(tasks[0])
        assert entered.wait(5)
        assert owner.snapshot()['charged_packets'] == 1
        assert len(executor.jobs) == 1
        assert owner.bytes > 0
    finally:
        release.set()
    idle(owner)
    assert len(executor.jobs) == 2
    assert owner.snapshot()['charged_packets'] == 1


def test_failed_builder_settles_both_credits_before_next_submission(request):
    calls = []
    def allocate(shape, *args, **kwargs):
        calls.append(shape)
        if len(calls) == 1:
            raise MemoryError('injected optional allocation refusal')
        return np.empty(shape, np.uint8)
    owner, executor = prefetch(request, descriptors(3), allocate=allocate,
                               max_builders=1, max_packets=1)
    executor.run()
    assert len(executor.jobs) == 2
    assert owner.packets[0].failed and owner.packets[0].charged_bytes == 0
    executor.run(1)
    assert owner.snapshot()['ready_packets'] == 1
    assert len(executor.jobs) == 2


def test_explicit_packet_limit_is_tighter_than_builder_override_and_shutdown_does_not_refill(request):
    owner, executor = prefetch(request, descriptors(20), max_builders=7, max_packets=3)
    assert len(executor.jobs) == 3
    owner.shutdown()
    assert len(executor.jobs) == 3
    assert all(future.cancelled() for future, _, _ in executor.jobs)
    assert owner.bytes == 0 and owner.snapshot()['charged_packets'] == 0


@pytest.mark.parametrize('limits', [dict(max_builders=0), dict(max_packets=0),
                                   dict(max_builders=-1), dict(max_packets=-3)])
def test_invalid_limits_fail_before_starting_controller(limits):
    with pytest.raises(ValueError, match='limits must be positive'):
        CpuInputPrefetch([], np.zeros((1, 1, 1), np.uint8), ManualExecutor(),
                         max_bytes=100, headroom=lambda: 100, **limits)
