"""Optional detector inputs yield only after their actual owners settle."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import queue
import threading
import os
import weakref

import numpy as np

from XTA.yolo_cpu_input import CpuInputPrefetch
from XTA import runtime
from tests.test_yolo_cpu_input import task, ManualExecutor
from tests.test_tta_scheduler_boundary import _scheduler, _state


def idle(owner):
    with owner.condition:
        assert owner.condition.wait_for(
            lambda: not owner.pump_scheduled and not owner.pump_requested, timeout=5)


@contextmanager
def prefetch(tasks, *, executor=None, close=None, allocate=None, wake=None):
    executor = executor or ManualExecutor()
    closed = []
    owner = CpuInputPrefetch(tasks, np.zeros((5, 9, 10), np.uint8), executor,
        max_bytes=100000, headroom=lambda:1000000,
        allocate=allocate or (lambda shape,*a,**kw:np.empty(shape,np.uint8)),
        backing=lambda array:Path('/proc/fake/prepared'), close_array=close or closed.append,
        on_capacity_change=wake)
    if isinstance(executor, ManualExecutor):
        executor.owner = owner
    idle(owner)
    try:
        yield owner, executor, closed
    finally:
        owner.shutdown()


def test_preview_is_pure_and_enough_room_discards_nothing():
    descriptor = task()
    with prefetch([descriptor]) as (owner, executor, closed):
        executor.run()
        before = owner.snapshot()
        assert not owner.can_yield_capacity(100)
        assert owner.can_reclaim_capacity(100)
        assert owner.snapshot() == before
        assert owner.try_yield_capacity(1000) == 1000
        assert owner.ready(descriptor)
        assert not owner.packets[0].revoked
        assert closed == []


def test_slow_retirement_keeps_credit_and_stops_refill_until_commit():
    entered, release = threading.Event(), threading.Event()
    def close(_array):
        entered.set()
        assert release.wait(5)
    with prefetch([task()], close=close) as (owner, executor, _):
        executor.run()
        charged = owner.bytes
        try:
            assert owner.request_reclaim_capacity(0)
            assert entered.wait(5)
            assert owner.bytes == owner.reserved_bytes == charged
            assert owner.try_yield_capacity(0) is None
            assert owner.set_capacity(100000) == charged
            assert not owner.ready(task())
            release.set()
            idle(owner)
            assert owner.bytes == owner.reserved_bytes == 0
            assert owner.set_capacity(100000) == 0  # Claim not yet committed.
            assert owner.try_yield_capacity(0) == 0
            assert owner.set_capacity(100000) == 100000
            idle(owner)
            assert len(executor.jobs) == 1  # No revoke/rebuild thrash.
        finally:
            release.set()


def test_attached_split_inputs_survive_pressure_and_duplicate_releases():
    original = task()
    with prefetch([original]) as (owner, executor, closed):
        executor.run()
        first = dict(original, slice_count=1)
        second = dict(original, task_id=7, slice_start=1, slice_count=2)
        owner.attach_ready(first)
        owner.attach_ready(first)
        owner.attach_ready(second)
        assert owner.packets[0].borrowers == {0, 7}
        assert not owner.can_reclaim_capacity(0)
        assert not owner.request_reclaim_capacity(0)
        owner.retire(first)
        owner.retire(first)
        assert owner.packets[0].borrowers == {7}
        assert not owner.request_reclaim_capacity(0)
        assert not closed
        owner.retire(second)
        idle(owner)
        assert owner.bytes == 0
        assert len(closed) == 1
        replay = dict(first)
        replay.pop('cpu_input_path')
        owner.attach_ready(replay)
        assert 'cpu_input_path' not in replay


def test_overlapping_borrowers_wait_for_each_actual_release():
    descriptor = task()
    with prefetch([descriptor]) as (owner, executor, closed):
        executor.run()
        duplicate_range = dict(descriptor, task_id=2)
        owner.attach_ready(descriptor)
        owner.attach_ready(duplicate_range)
        owner.retire(descriptor)
        idle(owner)
        assert owner.bytes and not closed
        owner.retire(duplicate_range)
        idle(owner)
        assert owner.bytes == 0 and len(closed) == 1


def test_partial_packet_can_yield_after_its_last_current_borrower_releases():
    descriptor = task()
    with prefetch([descriptor]) as (owner, executor, closed):
        executor.run()
        first = dict(descriptor, slice_count=1)
        owner.attach_ready(first)
        owner.retire(first)
        assert owner.packets[0].remaining == {1, 2}
        assert owner.request_reclaim_capacity(0)
        idle(owner)
        sibling = dict(descriptor, task_id=9, slice_start=1, slice_count=2)
        owner.attach_ready(sibling)
        assert 'cpu_input_path' not in sibling
        owner.retire(sibling)
        idle(owner)
        assert owner.bytes == 0 and len(closed) == 1


def test_selected_parent_inputs_are_preserved_when_other_packets_suffice():
    preferred, other = task(), task('sagittal', task_id=1)
    with prefetch([preferred, other]) as (owner, executor, closed):
        executor.run(0)
        executor.run(1)
        assert owner.request_reclaim_capacity(432,
            preferred_parent=('model', preferred['view'].name))
        idle(owner)
        assert owner.ready(preferred)
        assert not owner.ready(other)
        assert owner.bytes == 432 and len(closed) == 1


def test_queued_builder_is_cancelled_on_controller_without_allocation():
    allocate = mock.Mock(side_effect=lambda shape,*a,**kw:np.empty(shape,np.uint8))
    with prefetch([task()], allocate=allocate) as (owner, executor, closed):
        future = executor.jobs[0][0]
        assert owner.bytes > 0
        assert owner.request_reclaim_capacity(0)
        idle(owner)
        assert future.cancelled()
        assert owner.bytes == 0 and not closed
        allocate.assert_not_called()


def test_running_builder_retires_only_after_render_returns():
    entered, release = threading.Event(), threading.Event()
    def allocate(shape, *a, **kw):
        entered.set()
        assert release.wait(5)
        return np.empty(shape, np.uint8)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with prefetch([task()], executor=executor, allocate=allocate) as (owner, _, closed):
            try:
                assert entered.wait(5)
                charged = owner.bytes
                assert owner.request_reclaim_capacity(0)
                idle(owner)
                assert owner.bytes == charged
                assert owner.try_yield_capacity(0) is None
                release.set()
                with owner.condition:
                    assert owner.condition.wait_for(lambda:owner.bytes == 0, timeout=5)
                assert len(closed) == 1
            finally:
                release.set()


def test_finished_future_requires_adoption_before_credit_returns():
    with prefetch([task()]) as (owner, executor, closed):
        future = executor.jobs[0][0]
        callbacks = future._done_callbacks[:]
        future._done_callbacks.clear()
        executor.run()
        assert future.done()
        assert owner.request_reclaim_capacity(0)
        idle(owner)
        assert owner.bytes > 0 and not closed
        assert owner.try_yield_capacity(0) is None
        for callback in callbacks:
            callback(future)
        idle(owner)
        assert owner.bytes == 0 and len(closed) == 1


def test_submit_adoption_race_retains_credit_and_cancels_on_control_thread():
    entered, release = threading.Event(), threading.Event()
    class BlockedSubmit(ManualExecutor):
        def submit(self, function, *args):
            entered.set()
            assert release.wait(5)
            return super().submit(function, *args)
    executor = BlockedSubmit()
    owner = CpuInputPrefetch([task()],np.zeros((5,9,10),np.uint8),executor,
        max_bytes=100000,headroom=lambda:1000000,
        allocate=lambda shape,*a,**kw:np.empty(shape,np.uint8),
        backing=lambda array:Path('/proc/fake/prepared'),close_array=lambda array:None)
    try:
        assert entered.wait(5)
        charged = owner.bytes
        assert owner.request_reclaim_capacity(0)
        assert owner.bytes == charged
        assert owner.try_yield_capacity(0) is None
        release.set()
        idle(owner)
        assert executor.jobs[0][0].cancelled()
        assert owner.bytes == 0
    finally:
        release.set()
        owner.shutdown()


def test_reclaim_during_submit_failure_and_shutdown_settles_exactly_once():
    entered, release = threading.Event(), threading.Event()
    class FailingSubmit:
        def submit(self, *args):
            entered.set()
            assert release.wait(5)
            raise RuntimeError('submit failed before ownership transfer')
    owner = CpuInputPrefetch([task()],np.zeros((5,9,10),np.uint8),FailingSubmit(),
        max_bytes=100000,headroom=lambda:1000000)
    try:
        assert entered.wait(5)
        assert owner.request_reclaim_capacity(0)
        assert owner.bytes > 0
        release.set()
        idle(owner)
        assert owner.bytes == 0
        owner.shutdown()
        owner.shutdown()
        assert owner.bytes == 0
    finally:
        release.set()
        owner.shutdown()


def test_real_scheduler_waits_for_retirement_then_births_parent_and_falls_back():
    entered, release = threading.Event(), threading.Event()
    def close(_array):
        entered.set()
        assert release.wait(5)
    descriptor = dict(task(), result_mode='direct_union', processing_shape=(1,1,19900))
    state = _state()
    state.gpu_task_queues[0] = queue.Queue()
    state.gpu_worker_tasks_by_id[0] = descriptor
    state.gpu_worker_pending_task_ids.append(0)
    state.direct_union_postprocess_bytes[('model','old')] = 10000
    state.direct_union_backing_leases[('model','old')] = SimpleNamespace(phase='postprocess')
    births = []
    def birth(model, view):
        assert owner.bytes + 10000 + 19900 <= 30000
        births.append((model, view.name))
        state.baseline_union_paths[(model,view.name)] = Path('mask.dat')
        state.direct_union_inference_views.add((model,view.name))
        state.direct_union_inference_bytes[(model,view.name)] = 19900
        state.direct_union_backing_leases[(model,view.name)] = SimpleNamespace(phase='inference')
    scheduler = _scheduler(Path('.'), state=state, input_overrides=dict(
        direct_union_total_dense_byte_limit=30000,direct_union_inference_byte_limit=30000,
        ensure_baseline_workspaces=birth))
    with prefetch([descriptor], close=close,
            wake=scheduler.notify_gpu_stage_admission_change) as (owner, executor, _):
        executor.run()
        scheduler.cpu_input_prefetch = owner
        try:
            assert scheduler.direct_union_task_admissible(descriptor)
            assert not owner.packets[0].revoked  # The feasibility scan is pure.
            scheduler.dispatch_gpu_worker_inference_window()
            assert entered.wait(5)
            assert not births and state.gpu_task_queues[0].empty()
            assert owner.bytes == 432
            scheduler.dispatch_gpu_worker_inference_window()
            assert not births and owner.bytes == 432
            release.set()
            idle(owner)
            assert state.gpu_stage_admission_dirty.is_set()
            assert scheduler.service_gpu_stage_admission_retry()
            dispatched = state.gpu_task_queues[0].get_nowait()
            assert births == [('model',descriptor['view'].name)]
            assert 'cpu_input_path' not in dispatched
            assert dispatched['slice_start'] == descriptor['slice_start']
            assert dispatched['slice_count'] == descriptor['slice_count']
            assert owner.bytes == 0
        finally:
            release.set()


def test_mapping_alias_and_completed_future_hold_no_unfunded_backing(tmp_path):
    path = tmp_path / 'input.dat'
    def allocate(shape, *args, **kwargs):
        return np.memmap(path, mode='w+', shape=shape, dtype=np.uint8)
    with prefetch([task()], allocate=allocate,
            close=runtime.close_memmap_array_without_flush) as (owner, executor, _):
        executor.run()
        assert executor.jobs[0][0].result() is None
        alias = owner.packets[0].array[:, :, :]
        mapping = weakref.ref(alias._mmap)
        assert owner.request_reclaim_capacity(0)
        idle(owner)
        assert owner.packets[0].array is None
        assert mapping() is not None
        assert owner.bytes == 432
        assert owner.try_yield_capacity(0) is None
        alias = None
        with owner.condition:
            assert owner.condition.wait_for(lambda:owner.bytes == 0, timeout=5)
        assert mapping() is None
        assert owner.try_yield_capacity(0) == 0


def test_mapping_death_does_not_refund_before_keepalive_fd_close(tmp_path):
    path = tmp_path / 'input.dat'
    key = str(path)
    descriptor = None
    def allocate(shape, *args, **kwargs):
        nonlocal descriptor
        array = np.memmap(path, mode='w+', shape=shape, dtype=np.uint8)
        descriptor = os.open(path, os.O_RDONLY)
        runtime._register_memfd_owner(key, descriptor, 'test retained input FD')
        array._workspace_memfd_owner_key = key
        return array
    entered, release = threading.Event(), threading.Event()
    actual_release = runtime._release_memfd_owner_key
    def delayed_release(owner_key):
        entered.set()
        assert release.wait(5)
        return actual_release(owner_key)
    with prefetch([task()], allocate=allocate,
            close=runtime.close_memmap_array_without_flush) as (owner, executor, _):
        executor.run()
        aliases = [owner.packets[0].array[:]]
        assert owner.request_reclaim_capacity(0)
        idle(owner)
        with mock.patch.object(runtime, '_release_memfd_owner_key', side_effect=delayed_release):
            drop = threading.Thread(target=aliases.clear)
            drop.start()
            try:
                assert entered.wait(5)
                assert owner.packets[0].retirement_mapping() is None
                assert owner.bytes == 432
                assert owner.try_yield_capacity(0) is None
                os.fstat(descriptor)
                release.set()
                drop.join(5)
                assert not drop.is_alive()
                with owner.condition:
                    assert owner.condition.wait_for(lambda:owner.bytes == 0, timeout=5)
            finally:
                release.set()
                drop.join(5)


def test_failed_renderer_payload_retires_without_future_traceback_alias(tmp_path):
    path = tmp_path / 'input.dat'
    mappings = []
    def allocate(shape, *args, **kwargs):
        array = np.memmap(path, mode='w+', shape=shape, dtype=np.uint8)
        mappings.append(weakref.ref(array._mmap))
        return array
    with prefetch([task()], allocate=allocate,
            close=runtime.close_memmap_array_without_flush) as (owner, executor, _):
        with mock.patch('XTA.yolo_cpu_input.NativeCartesianRenderer', side_effect=RuntimeError('render failed')):
            executor.run()
        with owner.condition:
            assert owner.condition.wait_for(lambda:owner.bytes == 0, timeout=5)
        assert executor.jobs[0][0].result() is None
        assert mappings[0]() is None
        assert not owner.ready(task())


def test_reclaim_counters_separate_request_from_settlement_and_do_not_double_count():
    entered, release = threading.Event(), threading.Event()
    def close(_array):
        entered.set()
        assert release.wait(5)
    telemetry = SimpleNamespace(add=mock.Mock(), add_scheduler_counter=mock.Mock(), gauge=mock.Mock())
    with mock.patch('XTA.yolo_cpu_input.runtime_telemetry', return_value=telemetry):
        with prefetch([task()], close=close) as (owner, executor, _):
            executor.run()
            try:
                owner.request_reclaim_capacity(0)
                assert entered.wait(5)
                owner.request_reclaim_capacity(0)
                calls = telemetry.add_scheduler_counter.call_args_list
                assert calls.count(mock.call('yolo.cpu_input.reclaim_requested_packets',1)) == 1
                assert mock.call('yolo.cpu_input.reclaim_requested_charged_bytes',432) in calls
                assert not any('reclaim_settled' in call.args[0] for call in calls)
                release.set()
                idle(owner)
                calls = telemetry.add_scheduler_counter.call_args_list
                assert calls.count(mock.call('yolo.cpu_input.reclaim_settled_packets',1)) == 1
                assert mock.call('yolo.cpu_input.reclaim_settled_charged_bytes',432) in calls
                assert owner.reserved_bytes == 0
            finally:
                release.set()


def test_close_failure_keeps_credit_and_retirement_retry_owner():
    entered, release = threading.Event(), threading.Event()
    calls = 0
    def close(_array):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('temporary close failure')
        entered.set()
        assert release.wait(5)
    with prefetch([task()], close=close) as (owner, executor, _):
        executor.run()
        try:
            assert owner.request_reclaim_capacity(0)
            assert entered.wait(5)
            assert owner.bytes == 432
            assert owner.try_yield_capacity(0) is None
            release.set()
            idle(owner)
            assert owner.bytes == 0
            assert calls == 2
        finally:
            release.set()
