from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading
import time

import pytest

from XTA.native_pull_scheduling import (
    NativePullScheduler, configure_native_pull_scheduler,
    get_native_pull_scheduler, shutdown_native_pull_scheduler,
)


def wait_for(predicate):
    until = time.monotonic() + 5
    while not predicate():
        if time.monotonic() >= until:
            pytest.fail('Timed out waiting for shared projection state')
        time.sleep(.005)


def test_existing_projection_expands_at_inference_drain_and_keeps_order():
    scheduler = NativePullScheduler(workers=1, max_workers=4)
    release = threading.Event()
    entered = set()
    lock = threading.Lock()

    def operation(index):
        with lock:
            entered.add(index)
        assert release.wait(5)
        return index * 10

    stream = scheduler.ordered_planes(6, operation, max_pending=4)
    try:
        wait_for(lambda: len(entered) == 1)
        assert scheduler.snapshot()['active_operations'] == 1
        scheduler.resize(4)
        wait_for(lambda: len(entered) == 4)
        assert scheduler.snapshot()['active_operations'] == 4
        release.set()
        assert list(stream) == [(index, index * 10) for index in range(6)]
        assert stream.snapshot()['projection_peak_active_operations'] == 4
        assert scheduler.snapshot()['peak_retained_operations'] <= 4
    finally:
        release.set()
        stream.close()
        scheduler.close()


def test_completed_planes_keep_global_credit_and_new_stream_gets_fair_turn():
    scheduler = NativePullScheduler(workers=2, max_workers=4)
    second_entered = threading.Event()
    first = scheduler.ordered_planes(8, lambda index: index, max_pending=4)
    second = None
    try:
        wait_for(lambda: scheduler.snapshot()['active_operations'] == 0)
        assert scheduler.snapshot()['retained_operations'] == 2
        second = scheduler.ordered_planes(2, lambda index: second_entered.set() or index,
                                           max_pending=4)
        assert not second_entered.wait(.025)
        assert next(first) == (0, 0)
        assert not second_entered.is_set()  # Consumer still holds that plane.
        assert next(first) == (1, 1)
        assert second_entered.wait(5)  # Released credit went to the new stream.
        assert scheduler.snapshot()['retained_operations'] <= 2
        first.close()
        assert list(second) == [(0, 0), (1, 1)]
        assert scheduler.snapshot()['peak_retained_operations'] == 2
    finally:
        first.close()
        if second is not None:
            second.close()
        scheduler.close()


def test_failure_joins_own_borrowed_readers_and_other_projection_survives():
    scheduler = NativePullScheduler(workers=2, max_workers=2)
    reader_entered, release_reader = threading.Event(), threading.Event()
    consumer_done = threading.Event()
    observed = []

    def operation(index):
        if index == 0:
            assert reader_entered.wait(5)
            raise ValueError('bad projection')
        reader_entered.set()
        assert release_reader.wait(5)
        return index

    failed = scheduler.ordered_planes(20, operation, max_pending=2)

    def consume_failed():
        try:
            next(failed)
        except ValueError as error:
            observed.append(str(error))
        finally:
            consumer_done.set()

    consumer = threading.Thread(target=consume_failed)
    consumer.start()
    other = scheduler.ordered_planes(2, lambda index: index + 20, max_pending=2)
    try:
        assert reader_entered.wait(5)
        assert not consumer_done.wait(.025)
        release_reader.set()
        assert consumer_done.wait(5)
        assert observed == ['bad projection']
        assert list(other) == [(0, 20), (1, 21)]
        assert failed.snapshot()['projection_active_operations'] == 0
        assert scheduler.snapshot()['retained_operations'] == 0
    finally:
        release_reader.set()
        consumer.join(5)
        failed.close()
        other.close()
        scheduler.close()


def test_per_projection_workspace_window_and_global_cap_are_both_enforced():
    scheduler = NativePullScheduler(workers=4, max_workers=4)
    release = threading.Event()

    def operation(index):
        assert release.wait(5)
        return bytearray(1024)

    first = scheduler.ordered_planes(8, operation, max_pending=1)
    second = scheduler.ordered_planes(8, operation, max_pending=2)
    try:
        wait_for(lambda: scheduler.snapshot()['active_operations'] == 3)
        assert first.snapshot()['projection_retained_operations'] <= 1
        assert second.snapshot()['projection_retained_operations'] <= 2
        release.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(list, first), pool.submit(list, second)
            assert len(a.result(timeout=5)) == len(b.result(timeout=5)) == 8
        assert scheduler.snapshot()['peak_retained_operations'] <= 4
    finally:
        release.set()
        first.close()
        second.close()
        scheduler.close()


def test_shrink_only_limits_new_admission_and_does_not_join_live_operations():
    scheduler = NativePullScheduler(workers=3, max_workers=3)
    release = threading.Event()
    stream = scheduler.ordered_planes(6, lambda index: release.wait(5) and index, max_pending=3)
    try:
        wait_for(lambda: scheduler.snapshot()['active_operations'] == 3)
        scheduler.resize(1)
        assert scheduler.snapshot()['cpu_budget'] == 1
        assert scheduler.snapshot()['active_operations'] == 3
        release.set()
        assert list(stream) == [(index, index) for index in range(6)]
        assert scheduler.snapshot()['retained_operations'] == 0
    finally:
        release.set()
        stream.close()
        scheduler.close()


def test_memory_limited_stream_lends_unused_cpu_share_to_another_stream():
    scheduler = NativePullScheduler(workers=4, max_workers=4)
    release = threading.Event()

    def operation(index):
        assert release.wait(5)
        return index

    small = scheduler.ordered_planes(4, operation, max_pending=1)
    large = scheduler.ordered_planes(4, operation, max_pending=4)
    try:
        wait_for(lambda: scheduler.snapshot()['active_operations'] == 4)
        assert small.snapshot()['projection_retained_operations'] == 1
        assert large.snapshot()['projection_retained_operations'] == 3
        release.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(list, small), pool.submit(list, large)
            assert a.result(timeout=5) == b.result(timeout=5) == [(index, index) for index in range(4)]
    finally:
        release.set()
        small.close()
        large.close()
        scheduler.close()


def test_native_projection_and_confidence_capture_share_cpu_credits_with_truthful_roles():
    scheduler = NativePullScheduler(workers=2, max_workers=4)
    release = threading.Event()

    def operation(index):
        assert release.wait(5)
        return index

    native = scheduler.ordered_planes(4, operation, max_pending=2)
    capture = scheduler.ordered_planes(4, operation, max_pending=2, role='confidence_capture')
    try:
        assert scheduler.snapshot()['active_jobs'] == 2
        assert scheduler.snapshot()['active_projections'] == 1
        assert scheduler.snapshot()['active_confidence_captures'] == 1
        scheduler.resize(4)
        wait_for(lambda: scheduler.snapshot()['active_operations'] == 4)
        assert capture.snapshot()['job_role'] == 'confidence_capture'
        assert capture.snapshot()['job_active_operations'] == 2
        release.set()
        assert list(native) == [(index, index) for index in range(4)]
        assert scheduler.snapshot()['active_projections'] == 0
        assert scheduler.snapshot()['active_confidence_captures'] == 1
        assert list(capture) == [(index, index) for index in range(4)]
        assert scheduler.snapshot()['active_jobs'] == 0
        assert scheduler.snapshot()['retained_operations'] == 0
    finally:
        release.set()
        native.close()
        capture.close()
        scheduler.close()


def test_pipeline_scheduler_cleanup_is_owned_and_overlap_is_rejected():
    first = configure_native_pull_scheduler(workers=1, max_workers=2)
    try:
        assert get_native_pull_scheduler() is first
        with pytest.raises(RuntimeError, match='already configured'):
            configure_native_pull_scheduler(workers=2, max_workers=2)
    finally:
        shutdown_native_pull_scheduler(first)
    second = configure_native_pull_scheduler(workers=1, max_workers=2)
    try:
        shutdown_native_pull_scheduler(first)
        assert get_native_pull_scheduler() is second
    finally:
        shutdown_native_pull_scheduler(second)
    assert get_native_pull_scheduler() is None


def test_run_cancellation_wakes_queued_consumers_and_joins_abandoned_readers():
    scheduler = NativePullScheduler(workers=1, max_workers=2)
    entered, release = threading.Event(), threading.Event()

    def borrowed_reader(index):
        entered.set()
        assert release.wait(5)
        return index

    borrowed = scheduler.ordered_planes(4, borrowed_reader, max_pending=2)
    queued = scheduler.ordered_planes(4, lambda index: index, max_pending=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        consumer = pool.submit(list, queued)
        try:
            assert entered.wait(5)
            scheduler.cancel('failed pipeline')
            with pytest.raises(CancelledError):
                consumer.result(timeout=5)
            closing = pool.submit(scheduler.close)
            assert not closing.done()
            release.set()
            closing.result(timeout=5)
            assert scheduler.snapshot()['active_operations'] == 0
            assert scheduler.snapshot()['retained_operations'] == 0
            assert borrowed._lane.operation is None
            assert not borrowed._lane.futures
        finally:
            release.set()
            borrowed.close()
            queued.close()
            scheduler.close()


def test_production_projection_uses_shared_budget_with_caller_one_and_keeps_workspace_cap(
        tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    import numpy as np
    from XTA import backprojection as bp
    from XTA.geometry import _build_tilted_view_infos

    view = _build_tilted_view_infos(9, 7, 9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]
    source = np.ones((view.num_slices, 7, 9), np.uint8)
    target = (9, 512, 512)
    records, published = {}, []
    monkeypatch.setattr(bp, 'runtime_telemetry', lambda: SimpleNamespace(
        gauge=lambda name, value: records.__setitem__(name, value)))
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_CUDA', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', '1')
    plan = SimpleNamespace(persistent_bytes=0, workspace_bytes=0, temporary_strip_bytes=0,
                           max_strip_voxels=512*512, backend='test_compiled')
    first_pair = threading.Barrier(2)
    counter_lock = threading.Lock()
    calls = 0

    def pull(array, actual_plan, output, **kwargs):
        nonlocal calls
        assert array is source and actual_plan is plan
        with counter_lock:
            calls += 1
            ordinal = calls
        if ordinal <= 2:
            first_pair.wait(timeout=5)
        output[:] = 1

    monkeypatch.setitem(sys.modules, 'XTA.projection_coverage_cpu', SimpleNamespace(
        prepare_native_pull_plan=lambda *args, **kwargs: plan, pull_native_flat_into=pull))
    scheduler = configure_native_pull_scheduler(workers=2, max_workers=8)
    try:
        result = bp._backproject_native_destination_pull(source, view, tmp_path/'unused.dat',
            'shared bounded publication', output_shape=target, sink_only=True,
            callback=lambda z, block: published.append(z), workers=1)
        assert result.shape == target
        assert published == list(range(9))
        final = records['projection.native_destination_pull']
        assert final['workers'] == 2
        assert final['worker_window'] == 3
        assert final['scheduling'] == 'shared_cpu_budget'
        assert final['shared_scheduler']['peak_retained_operations'] <= 2
        assert final['worker_window']*final['worker_workspace_bytes'] + final['consumer_plane_bytes'] <= 1024**2
        assert not (tmp_path/'unused.dat').exists()
    finally:
        shutdown_native_pull_scheduler(scheduler)


@pytest.mark.parametrize('parent_budget,tail_budget,ceiling', [(32, 128, 128), (32, 32, 32), (32, 8, 32)])
def test_pipeline_configures_parent_budget_tail_ceiling_and_owns_cleanup(parent_budget, tail_budget, ceiling):
    import ast
    from pathlib import Path
    from XTA.tta_lifecycle import PipelineRunResources

    tree = ast.parse((Path(__file__).parents[1]/'XTA'/'pipeline.py').read_text(encoding='utf-8'))
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'native_pull_scheduler'
                for target in node.targets))
    resources = PipelineRunResources()
    namespace = dict(worker_budget=parent_budget, tail_worker_budget=tail_budget,
        _run_resources=lambda: resources, configure_native_pull_scheduler=configure_native_pull_scheduler)
    try:
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<production projection setup>', 'exec'), namespace)
        owner = namespace['native_pull_scheduler']
        assert owner is get_native_pull_scheduler()
        assert resources.closeables == [owner]
        assert owner.snapshot()['cpu_budget'] == parent_budget
        assert owner.snapshot()['max_workers'] == ceiling
    finally:
        resources.close(failed=False)
    assert get_native_pull_scheduler() is None


@pytest.mark.parametrize('received,announced,expected', [(9, False, []), (10, True, ['affinity', ('resize', 128)])])
def test_pipeline_expands_existing_pool_before_already_announced_drain_guard(received, announced, expected):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    tree = ast.parse((Path(__file__).parents[1]/'XTA'/'pipeline.py').read_text(encoding='utf-8'))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == '_announce_process_inference_drain_if_complete')
    events = []
    namespace = dict(scheduler_state=SimpleNamespace(gpu_worker_results_collected=received,
        gpu_worker_total_tasks=10, gpu_inference_drain_announced=announced),
        gpu_worker_process_active=False, sam_context=None, tail_worker_budget=128,
        _restore_parent_post_inference_affinity=lambda: events.append('affinity'),
        native_pull_scheduler=SimpleNamespace(resize=lambda workers: events.append(('resize', workers))))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<production inference drain>', 'exec'), namespace)
    namespace[function.name]()
    assert events == expected


def test_failed_pipeline_resource_cleanup_cancels_queued_parent_before_joining_readers():
    from XTA.tta_lifecycle import PipelineRunResources

    resources = PipelineRunResources()
    scheduler = resources.track_closeable(configure_native_pull_scheduler(workers=1, max_workers=2))
    entered, release = threading.Event(), threading.Event()

    def reader(index):
        entered.set()
        assert release.wait(5)
        return index

    first = scheduler.ordered_planes(4, reader, max_pending=2)
    queued = scheduler.ordered_planes(4, lambda index: index, max_pending=2)
    parents = resources.track_executor(ThreadPoolExecutor(max_workers=1))
    pending_parent = parents.submit(list, queued)
    with ThreadPoolExecutor(max_workers=1) as controller:
        try:
            assert entered.wait(5)
            closing = controller.submit(resources.close, failed=True)
            with pytest.raises(CancelledError):
                pending_parent.result(timeout=5)
            assert not closing.done()  # The source reader is still borrowing.
            release.set()
            closing.result(timeout=5)
            assert get_native_pull_scheduler() is None
            assert scheduler.snapshot()['active_operations'] == 0
            assert first._lane.operation is None
        finally:
            release.set()
            first.close()
            queued.close()
            resources.close(failed=True)
