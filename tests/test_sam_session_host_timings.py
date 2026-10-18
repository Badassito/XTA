"""Host timing diagnostics preserve tracker results, failures, and cleanup."""
import gc
import json
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from XTA import lta_experimental as adapter
from XTA import sam_tracker_runtime as runtime
from XTA.lta_sam import SamSessionPlan
from tests.test_lta_experimental import _Measured, _Predictor, _Tracker


class Clock:
    wall = 0.
    cpu = 0.

    def advance(self, wall, cpu):
        self.wall += wall
        self.cpu += cpu

    def timers(self):
        return adapter._SessionHostTimings(wall_clock=lambda: self.wall, cpu_clock=lambda: self.cpu)


def test_fixed_size_nested_clocks_measure_host_wait_and_cpu_independently():
    clock = Clock()
    timers = clock.timers()
    for _ in range(100):
        with timers.measure('frame_observation'):
            clock.advance(5, .5)
            with timers.measure('raw_callback'):
                clock.advance(2, .25)
    result = timers.receipt()
    assert tuple(result['stages']) == adapter.SAM_SESSION_HOST_TIMING_STAGES
    assert result['stages']['frame_observation'] == dict(calls=100, host_seconds=700., thread_cpu_seconds=75.)
    assert result['stages']['raw_callback'] == dict(calls=100, host_seconds=200., thread_cpu_seconds=25.)
    assert result['cuda_synchronized'] is False
    assert result['nested_stages']['frame_observation'] == ('prediction_callback', 'raw_callback')
    result['stages']['raw_callback']['calls'] = 0
    assert timers.receipt()['stages']['raw_callback']['calls'] == 100


def test_close_gc_observer_is_thread_local_nested_and_removed_on_failure():
    clock = Clock()
    timers = clock.timers()
    callbacks = tuple(gc.callbacks)
    failure = RuntimeError('close failed')
    with pytest.raises(RuntimeError) as caught:
        with timers.measure('close_session'), timers.close_gc():
            observer, = [callback for callback in gc.callbacks if callback not in callbacks]
            other = threading.Thread(target=lambda: (observer('start', {}), observer('stop', {})))
            other.start()
            other.join()
            observer('start', {'generation': 2})
            clock.advance(3, .25)
            observer('stop', {'generation': 2})
            clock.advance(2, .5)
            raise failure
    assert caught.value is failure and tuple(gc.callbacks) == callbacks
    stages = timers.receipt()['stages']
    assert stages['close_session'] == dict(calls=1, host_seconds=5., thread_cpu_seconds=.75)
    assert stages['close_session_gc'] == dict(calls=1, host_seconds=3., thread_cpu_seconds=.25)


def session_fixture(monkeypatch, clock, *, raw_callback=None, prediction_callback=None, tracking_failure=None):
    tracker = _Tracker()
    predictor = _Predictor(tracker)
    measured = _Measured()
    finished = []
    original = tracker.propagate_in_video

    def propagate(*args, **kwargs):
        try:
            clock.advance(4, .5)
            if tracking_failure is not None:
                raise tracking_failure
            yield from original(*args, **kwargs)
        finally:
            finished.append(kwargs['start_frame_idx'])

    tracker.propagate_in_video = propagate
    timers = clock.timers()
    monkeypatch.setattr(adapter, '_SessionHostTimings', lambda: timers)
    original_scores = adapter._sigmoid_tracker_score_logits
    original_masks = adapter._binary_mask_batch_to_host

    def scores(*args, **kwargs):
        clock.advance(1, .125)
        return original_scores(*args, **kwargs)

    def masks(*args, **kwargs):
        clock.advance(2, .25)
        return original_masks(*args, **kwargs)

    monkeypatch.setattr(adapter, '_sigmoid_tracker_score_logits', scores)
    monkeypatch.setattr(adapter, '_binary_mask_batch_to_host', masks)
    seed = np.zeros((9, 13), dtype=bool)
    seed[2:7, 3:9] = True

    def run():
        return adapter.run_mask_seed_session(measured, predictor,
            resource=[Image.fromarray(np.zeros((9, 13, 3), dtype=np.uint8)) for _ in range(3)],
            session=SamSessionPlan('timing-test', 0, 0, 3), prompt_frame=0,
            ground_truth=seed, seed=None, object_masks=(seed,), conf=0.,
            propagation_mode='tracker-only', retain_predictions=True,
            raw_observation_callback=raw_callback, prediction_callback=prediction_callback,
            retain_raw_observations=True)

    return run, timers, measured, tracker, finished, seed


def test_real_adapter_clock_boundaries_preserve_masks_and_generator_exhaustion(monkeypatch):
    clock = Clock()
    observations = []
    def raw(item):
        observations.append(item.frame_index)
        clock.advance(3, .375)
    def prediction(_item):
        clock.advance(2, .25)
    run, timers, measured, tracker, finished, seed = session_fixture(
        monkeypatch, clock, raw_callback=raw, prediction_callback=prediction)
    result = run()
    assert measured.closed and all(not state for state in tracker.states)
    assert finished == observations == [0, 1, 2]
    assert all(np.array_equal(item.binary_mask, seed) for item in result['raw_observations'])
    stages = result['session_host_timings']['stages']
    assert stages['frame_propagation'] == dict(calls=3, host_seconds=12., thread_cpu_seconds=1.5)
    assert stages['frame_score_d2h'] == dict(calls=3, host_seconds=3., thread_cpu_seconds=.375)
    assert stages['frame_mask_d2h'] == dict(calls=3, host_seconds=6., thread_cpu_seconds=.75)
    assert stages['seed_mask_d2h'] == dict(calls=1, host_seconds=2., thread_cpu_seconds=.25)
    assert stages['frame_observation'] == dict(calls=3, host_seconds=15., thread_cpu_seconds=1.875)
    assert stages['raw_callback']['host_seconds'] == 9.
    assert stages['prediction_callback']['host_seconds'] == 6.
    assert stages['seed_prepare']['host_seconds'] == 2.
    assert stages['close_session']['calls'] == 1
    assert result['session_host_timings'] == timers.receipt()


@pytest.mark.parametrize('failure_boundary', ['raw_callback', 'prediction_callback', 'propagation'])
def test_failures_keep_original_exception_and_retire_generators_and_session(monkeypatch, failure_boundary):
    clock = Clock()
    failure = RuntimeError(failure_boundary+' failed')
    def callback(_item):
        clock.advance(5, .5)
        raise failure
    callbacks = tuple(gc.callbacks)
    run, timers, measured, tracker, finished, _seed = session_fixture(monkeypatch, clock,
        raw_callback=callback if failure_boundary == 'raw_callback' else None,
        prediction_callback=callback if failure_boundary == 'prediction_callback' else None,
        tracking_failure=failure if failure_boundary == 'propagation' else None)
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is failure
    assert measured.closed and all(not state for state in tracker.states)
    assert finished == [0] and tuple(gc.callbacks) == callbacks
    stages = timers.receipt()['stages']
    assert stages['close_session']['calls'] == stages['frame_propagation']['calls'] == 1
    stage = 'frame_propagation' if failure_boundary == 'propagation' else failure_boundary
    assert stages[stage]['calls'] == 1
    assert stages[stage]['host_seconds'] == (4 if failure_boundary == 'propagation' else 5)


def test_parent_aggregates_known_stages_once_with_authenticated_device_and_old_receipts(monkeypatch):
    from XTA import runtime as diagnostics
    counters = {}
    def add(key, value):
        counters[key] = counters.get(key, 0)+value
    monkeypatch.setattr(diagnostics, '_RUNTIME_TELEMETRY', SimpleNamespace(enabled=True, add=add))
    prepared = SimpleNamespace(task=SimpleNamespace(payload=dict(seed_frame=0, frame_stop=3, direction='forward')))
    manifest = dict(timings=dict(tracker_seconds=9.), coverage_complete=True, expected_frames=[0, 1, 2],
        cpu_task_retirement=dict(execution_device_id='untrusted.unbounded.label'),
        session_host_timings=dict(schema='xta.sam_session_host_timings/1', stages={
            'frame_propagation': dict(calls=3, host_seconds=6., thread_cpu_seconds=.5),
            'frame_mask_d2h': dict(calls=True, host_seconds=float('nan'), thread_cpu_seconds=-1.),
            'close_session': dict(calls=1.5, host_seconds=10**400, thread_cpu_seconds='invalid'),
            'arbitrary-session-id': dict(calls=9, host_seconds=999., thread_cpu_seconds=999.)}))
    runtime._record_worker_completion(manifest, prepared, 'radial', execution_device_id=1)
    for prefix in ('sam.worker.', 'sam.worker.family.radial.', 'sam.worker.device.1.'):
        assert counters[prefix+'session.frame_propagation.host_seconds'] == 6.
        assert counters[prefix+'session.frame_propagation.thread_cpu_seconds'] == .5
        assert counters[prefix+'session.frame_propagation.calls'] == 3
        assert counters[prefix+'completed_output_frames'] == 3
        assert counters[prefix+'tracker_seconds'] == 9.
    assert not any('untrusted' in key or 'arbitrary-session' in key or 'session.close_session' in key
                   or 'session.frame_mask_d2h' in key for key in counters)
    old = {key: value for key, value in manifest.items() if key != 'session_host_timings'}
    runtime._record_worker_completion(old, prepared, 'radial', execution_device_id=1)
    assert counters['sam.worker.device.1.completed_jobs'] == 2
    assert counters['sam.worker.device.1.session.frame_propagation.calls'] == 3


def test_merged_generator_is_closed_when_prediction_callback_fails(monkeypatch):
    clock = Clock()
    timers = clock.timers()
    monkeypatch.setattr(adapter, '_SessionHostTimings', lambda: timers)
    predictor = _Predictor(_Tracker())
    private = {}
    predictor.model._get_sam2_inference_states_by_obj_ids = lambda *_args: [private]
    predictor.model._cache_frame_outputs = lambda *_args: None
    seed = np.zeros((9, 13), dtype=bool)
    seed[2:7, 3:9] = True
    closed = []
    def stream(_request):
        try:
            yield dict(frame_index=0, outputs=dict(out_obj_ids=[0], out_probs=[1.],
                out_binary_masks=[seed.copy()]))
        finally:
            clock.advance(2, .25)
            closed.append('iterator')
    def handle(request):
        if request['type'] == 'start_session':
            return dict(session_id='mask-test')
        if request['type'] == 'add_prompt':
            return dict(outputs={})
        assert request['type'] == 'close_session'
        closed.append('session')
        return dict(success=True)
    failure = RuntimeError('merged callback failed')
    def callback(_item):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        adapter.run_mask_seed_session(SimpleNamespace(handle_request=handle, handle_stream_request=stream),
            predictor, resource=[object()], session=SamSessionPlan('merged', 0, 0, 1),
            prompt_frame=0, ground_truth=seed, seed=SimpleNamespace(x=.5, y=.5), object_masks=None,
            conf=0., propagation_mode='merged', prediction_callback=callback)
    assert caught.value is failure and closed == ['iterator', 'session'] and private == {}
    stages = timers.receipt()['stages']
    assert stages['iterator_close'] == dict(calls=1, host_seconds=2., thread_cpu_seconds=.25)
    assert stages['close_session']['calls'] == stages['prediction_callback']['calls'] == 1


def test_worker_manifest_retains_stage_receipt_and_accepts_old_adapter_results(tmp_path, monkeypatch):
    from tests.test_sam_worker_retirement import worker_fixture
    context, payload, _ = worker_fixture(tmp_path)
    output = runtime.execute_interpolation_tracker_task(context, 'sam_interpolation_run', payload)
    manifest = json.loads(open(output['artifact_path'], encoding='utf-8').read())
    assert manifest['session_host_timings']['schema'] == 'xta.sam_session_host_timings/1'
    assert manifest['session_host_timings']['stages']['frame_propagation']['calls'] == 3
    # Older/custom adapters may omit diagnostics without invalidating raw masks.
    original = adapter.run_mask_seed_session
    def without_timing(*args, **kwargs):
        result = original(*args, **kwargs)
        result.pop('session_host_timings')
        return result
    monkeypatch.setattr(adapter, 'run_mask_seed_session', without_timing)
    context, payload, _ = worker_fixture(tmp_path/'old')
    output = runtime.execute_interpolation_tracker_task(context, 'sam_interpolation_run', payload)
    manifest = json.loads(open(output['artifact_path'], encoding='utf-8').read())
    assert 'session_host_timings' not in manifest and manifest['coverage_complete'] is True
