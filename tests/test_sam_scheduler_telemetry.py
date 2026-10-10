"""Existing telemetry samples bounded SAM ownership without owning a tracker."""
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import time
import weakref

from XTA import runtime, sam_resources, sam_tracker_runtime as sam
from tests.test_sam_multiscope_scheduler import (
    FourSlotPool, LiveAdmissions, cache_for, request, tracker_for)


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
            assert sample['live_prepared_bank_bytes'] == bank_bytes[0] > 0
            prepared_resume.set()
            wait_for(lambda: tracker.snapshot()['ready_jobs'] == 1)
            pool.gates['live-0'].set()
            assert consuming.wait(5)
            wait_for(lambda: tracker.snapshot()['submitted_jobs'] == 2)
            sink.sample_registered_providers()
            sample = sink.snapshot()['gauges']['sam.scheduler.live']
            assert sample['ready_jobs'] == sample['running_jobs'] == sample['consumer_held_jobs'] == 1
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
