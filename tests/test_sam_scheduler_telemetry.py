"""Existing telemetry samples bounded SAM ownership without owning a tracker."""
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import time
import weakref

import pytest

from XTA import runtime, sam_resources, sam_tracker_runtime as sam
from tests.test_sam_multiscope_scheduler import (
    FourSlotPool, LiveAdmissions, cache_for, consume, request, tracker_for)


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, 'scheduler did not reach the observed ownership gate'
        time.sleep(.005)


def telemetry(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path))
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '1')
    sink = runtime.RuntimeTelemetry()
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', sink)
    monkeypatch.setattr(sam, '_LIVE_SAM_TRACKERS', weakref.WeakSet())
    monkeypatch.setattr(runtime, 'initialize_runtime_observability',
        lambda: (_ for _ in ()).throw(AssertionError('SAM diagnostics initialized observability')))
    monkeypatch.setattr(runtime.RuntimeSystemSampler, 'start',
        lambda self: (_ for _ in ()).throw(AssertionError('SAM diagnostics started a sampler')))
    return sink


def test_uninitialized_observability_is_not_started_by_tracker(monkeypatch, tmp_path):
    sink = telemetry(monkeypatch, tmp_path)
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', None)
    old_sampler = runtime._RUNTIME_SYSTEM_SAMPLER
    tracker = sam.SamInterpolationTracker(model_path='unused', device_ids=(0,), artifact_root=tmp_path/'runs')
    assert not sink._sample_providers and not sam._LIVE_SAM_TRACKERS
    assert runtime._RUNTIME_TELEMETRY is None and runtime._RUNTIME_SYSTEM_SAMPLER is old_sampler
    assert not tracker.artifact_root.exists()
    tracker.close()


def test_one_aggregate_provider_keeps_no_tracker_or_new_sampler(monkeypatch, tmp_path):
    sink = telemetry(monkeypatch, tmp_path)
    old_sampler = runtime._RUNTIME_SYSTEM_SAMPLER
    first = sam.SamInterpolationTracker(model_path='unused', device_ids=(0,), artifact_root=tmp_path/'a')
    second = sam.SamInterpolationTracker(model_path='unused', device_ids=(0,), artifact_root=tmp_path/'b')
    first.close()
    assert tuple(sink._sample_providers) == ('sam.scheduler.live',)
    sink.sample_registered_providers()
    sample = sink.snapshot()['gauges']['sam.scheduler.live']
    assert sample['tracker_instances'] == 2 and sample['closed_instances'] == 1
    assert all(sample[key] == 0 for key in sam._SAM_SNAPSHOT_COUNTS)
    assert runtime._RUNTIME_SYSTEM_SAMPLER is old_sampler and sink._writer_thread is None
    assert not sink.path.exists()
    references = weakref.ref(first), weakref.ref(second)
    second.close()
    del first, second
    gc.collect()
    assert all(reference() is None for reference in references)
    sink.sample_registered_providers()
    assert sink.snapshot()['gauges']['sam.scheduler.live']['tracker_instances'] == 0


def test_snapshot_records_preparation_ready_ack_and_held_consumer(monkeypatch, tmp_path):
    sink = telemetry(monkeypatch, tmp_path/'telemetry')
    pool = FourSlotPool(gated=('live-0', 'live-1', 'live-2'))
    tracker = tracker_for(tmp_path, pool, monkeypatch)
    cache = cache_for(tmp_path, 'live', frames=3)
    admitted = LiveAdmissions(1, bank=True)
    preparing, prepared_resume = threading.Event(), threading.Event()
    consuming, consumer_resume = threading.Event(), threading.Event()
    bank_bytes = []

    def producer():
        requests = [request(index, label='live', frames=3) for index in range(3)]
        for item in requests:
            item['metadata']['producer_thread'] = threading.get_ident()
        with admitted.scope(requests, capacity=1) as admission:
            bank_bytes.append(sam_resources.validate_sam_tracker_scope_admission(admission)['prepared_bank_bytes'])
            def original_requests():
                for index, item in enumerate(requests):
                    if index == 1:
                        preparing.set()
                        assert prepared_resume.wait(5)
                    yield item
            stream = tracker.iter_results(original_requests(), source_cache_ref=cache,
                max_in_flight=1, scope_admission=admission)
            try:
                index, result = next(stream)
                assert index == 0
                consuming.set()
                assert consumer_resume.wait(5)
                tracker.release_result(result)
                assert len(list(stream)) == 2
            finally:
                stream.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(producer)
        try:
            assert preparing.wait(5)
            wait_for(lambda: tracker.snapshot()['submitted_jobs'] == 1)
            sample = tracker.snapshot()
            assert sample['active_scopes'] == sample['credited_scopes'] == 1
            assert sample['preparing_jobs'] == sample['running_jobs'] == 1
            assert sample['ready_jobs'] == sample['completion_acks'] == sample['consumer_held_jobs'] == 0
            assert sample['scope_dispatchable_jobs'] == 0 and sample['free_worker_slots'] == 3
            assert sum(sample['active_workers_by_device'].values()) == 1
            assert sample['live_prepared_bank_bytes'] == bank_bytes[0] > 0
            prepared_resume.set()
            wait_for(lambda: tracker.snapshot()['ready_jobs'] == 1)
            pool.gates['live-0'].set()
            assert consuming.wait(5)
            wait_for(lambda: tracker.snapshot()['submitted_jobs'] == 2)
            sink.sample_registered_providers()
            sample = sink.snapshot()['gauges']['sam.scheduler.live']
            assert sample['ready_jobs'] == sample['running_jobs'] == sample['consumer_held_jobs'] == 1
            assert sample['scope_dispatchable_jobs'] == 0 and sample['free_worker_slots'] == 3
            assert sum(sample['active_workers_by_device'].values()) == 1
            assert sample['completion_acks'] == sample['completed_jobs'] == 1
            assert sample['preparing_jobs'] == sample['acked_awaiting_consumer_jobs'] == 0
            assert sample['raw_transfer_decode_seconds'] > 0
            with tracker._state_condition:
                next(iter(tracker._scopes.values())).transfer_received_perf_counter -= 2.
            held_before=tracker.snapshot()['consumer_hold_seconds']
            assert held_before >= 2.
            pool.gates['live-1'].set()
            wait_for(lambda: tracker.snapshot()['completion_acks'] == 2
                and tracker.snapshot()['submitted_jobs'] == 3)
            sample = tracker.snapshot()
            assert sample['acked_awaiting_consumer_jobs'] == sample['consumer_held_jobs'] == sample['running_jobs'] == 1
            assert sample['ready_jobs'] == sample['preparing_jobs'] == 0
            assert sample['completed_jobs'] == 1 and sample['completion_acks'] == 2
        finally:
            prepared_resume.set()
            consumer_resume.set()
            pool.unblock()
            task.result(timeout=8)
            tracker.close()
    sample = tracker.snapshot()
    assert all(sample[key] == 0 for key in sam._SAM_SNAPSHOT_COUNTS[:8])
    assert sample['submitted_jobs'] == sample['completed_jobs'] == sample['completion_acks'] == 3
    assert sample['closed'] and sample['cancelled'] and admitted.pool.in_use == 0
    assert sample['consumer_hold_seconds'] >= held_before
    assert sample['raw_transfer_decode_seconds'] > 0
    counters = sink.snapshot()['counters']
    assert counters['sam.worker.family.unknown.completed_jobs'] == counters['sam.worker.completed_jobs'] == 3
    assert counters['sam.worker.family.unknown.completed_output_frames'] == 9


def test_worker_timers_count_once_after_ack_despite_consumer_manifest_reread(monkeypatch, tmp_path):
    sink = telemetry(monkeypatch, tmp_path/'telemetry')
    values = dict(render_seconds=1.25, input_preparation_seconds=2.5, tracker_seconds=4.,
        sdk_session_init_seconds=.5, pack_seconds=.75, worker_seconds=8.5, queue_seconds=.1)
    malformed = dict(zip(values, (True, float('nan'), float('inf'), 'bad', -1., 10**400, None)))
    host_receipts = {}
    def mutate(task, manifest):
        index = int(task.payload['run_id'].rsplit('-', 1)[1])
        host_receipts[task.payload['run_id']] = manifest['session_host_timings']
        if index == 0:
            manifest['timings'] = dict(values, worker_started_monotonic=123., unknown_seconds=999.)
        elif index == 1:
            manifest['timings'] = malformed
        elif index == 2:
            manifest['timings'] = ['not a mapping']
        else:
            manifest.pop('timings', None)
    pool = FourSlotPool(mutate=mutate)
    tracker = tracker_for(tmp_path, pool, monkeypatch)
    cache = cache_for(tmp_path, 'timed', frames=3)
    reads = []
    original = sam._read_tracker_manifest
    def read(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(sam, '_read_tracker_manifest', read)
    try:
        assert len(consume(tracker, cache, [request(index, label='timed', frames=3)
            for index in range(4)], view_family='spherical')) == 4
    finally:
        tracker.close()
    assert len(reads) == 8  # Each ACK is authenticated again when its packet is consumed.
    counters = sink.snapshot()['counters']
    expected = {
        'sam.worker.completed_jobs': 4., 'sam.worker.completed_output_frames': 12.,
        **{'sam.worker.'+key: value for key,value in values.items()}}
    expected_all = {
        **expected, **{key.replace('sam.worker.', 'sam.worker.family.spherical.'): value
            for key, value in expected.items()}}
    for device, task, _submitted in pool.completions:
        device_prefix = f'sam.worker.device.{device}.'
        for key, value in {'completed_jobs': 1, 'completed_output_frames': 3,
                **(values if task.payload['run_id'].endswith('-0') else {})}.items():
            name = device_prefix+key
            expected_all[name] = expected_all.get(name, 0)+value
        for prefix in ('sam.worker.', 'sam.worker.family.spherical.', device_prefix):
            name = prefix+'session_timed_jobs'
            expected_all[name] = expected_all.get(name, 0)+1
            for stage, record in host_receipts[task.payload['run_id']]['stages'].items():
                for field, value in record.items():
                    name = prefix+f'session.{stage}.{field}'
                    expected_all[name] = expected_all.get(name, 0)+value
    assert {key: value for key,value in counters.items() if key.startswith('sam.worker.')} == expected_all


def test_completion_families_are_captured_on_concurrent_producers(monkeypatch, tmp_path):
    sink = telemetry(monkeypatch, tmp_path/'telemetry')
    def mutate(task, manifest):
        manifest['timings'] = {'tracker_seconds': 3. if task.payload['run_id'].startswith('first-') else 7.}
    pool = FourSlotPool(gated=('first-0', 'first-1', 'second-0', 'second-1'), mutate=mutate)
    tracker = tracker_for(tmp_path, pool, monkeypatch)
    admitted = LiveAdmissions(2)
    first, second = cache_for(tmp_path, 'first', frames=3), cache_for(tmp_path, 'second', frames=5)
    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(consume, tracker, cache,
            [request(index, label=label, frames=frames) for index in range(2)],
            capacity=2, admissions=admitted, view_family=family)
            for cache, label, frames, family in (
                (first, 'first', 3, 'radial'), (second, 'second', 5, 'spherical'))]
        try:
            pool.until(lambda: len(pool.submissions) == 4)
            with tracker._state_condition:
                assert {scope.view_family for scope in tracker._scopes.values()} == {'radial', 'spherical'}
            assert getattr(sam._SAM_VIEW_PRODUCER, 'view_family', 'unknown') == 'unknown'
        finally:
            pool.unblock()
            for task in tasks:
                assert len(task.result(timeout=8)) == 2
            tracker.close()
    counters = sink.snapshot()['counters']
    assert counters['sam.worker.completed_jobs'] == 4
    assert counters['sam.worker.completed_output_frames'] == 16
    assert counters['sam.worker.tracker_seconds'] == 20
    for family, frames, seconds in (('radial', 6, 6), ('spherical', 10, 14)):
        prefix = f'sam.worker.family.{family}.'
        assert counters[prefix+'completed_jobs'] == 2
        assert counters[prefix+'completed_output_frames'] == frames
        assert counters[prefix+'tracker_seconds'] == seconds
    assert not any(key.startswith('sam.worker.family.unknown.') for key in counters)


def test_family_context_bounds_and_restores_nested_labels():
    assert getattr(sam._SAM_VIEW_PRODUCER, 'view_family', 'unknown') == 'unknown'
    with sam.sam_view_producer_family('radial'):
        for family in (*sorted(sam._SAM_VIEW_FAMILIES), 'future-family', None):
            with sam.sam_view_producer_family(family):
                assert sam._SAM_VIEW_PRODUCER.view_family == (family if family in sam._SAM_VIEW_FAMILIES else 'unknown')
            assert sam._SAM_VIEW_PRODUCER.view_family == 'radial'
        with pytest.raises(RuntimeError, match='producer stopped'):
            with sam.sam_view_producer_family('spherical'):
                raise RuntimeError('producer stopped')
        assert sam._SAM_VIEW_PRODUCER.view_family == 'radial'
    assert sam._SAM_VIEW_PRODUCER.view_family == 'unknown'


@pytest.mark.parametrize('failure', ['contract', 'retirement'])
def test_rejected_completion_does_not_publish_worker_counters(monkeypatch, tmp_path, failure):
    sink = telemetry(monkeypatch, tmp_path/'telemetry')
    def mutate(_task, manifest):
        if failure == 'contract':
            manifest['seed_frame'] = -1
    if failure == 'retirement':
        def reject(*args):
            raise RuntimeError('controlled retirement rejection')
        monkeypatch.setattr(sam, '_cpu_completion_retired', reject)
    tracker = tracker_for(tmp_path, FourSlotPool(mutate=mutate), monkeypatch)
    cache = cache_for(tmp_path, 'rejected', frames=3)
    try:
        with pytest.raises(RuntimeError, match='contract|retirement rejection'):
            consume(tracker, cache, [request(0, label='rejected', frames=3)], view_family='radial')
    finally:
        tracker.close()
    assert not any(key.startswith('sam.worker.') for key in sink.snapshot()['counters'])
