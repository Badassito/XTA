"""Two CPU-only SDK sessions in real isolated workers exercise one GPU's routes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from XTA import sam_resources, sam_tracker_runtime as sam
from XTA.lta_workers import LtaWorkerInit, LtaWorkerPool
from tests.test_sam_multiscope_scheduler import cache_for, consume, request
from tests.test_sam_selection_resources import Pool, GIB


ADAPTER = r'''
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from tests.test_sam_tracker_runtime import _CombinedPredictor
from tests.test_lta_experimental import _Tracker
from XTA.sam_tracker_runtime import execute_interpolation_tracker_task

DEVICE = int(os.environ['LTA_EXECUTION_DEVICE_ID'])
SLOT = int(os.environ['LTA_WORKER_INDEX'])
VISIBLE = os.environ['CUDA_VISIBLE_DEVICES']


def record(root, kind, value):
    path = Path(root)/f'{kind}-{SLOT}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class Session(_CombinedPredictor):
    def __init__(self, config):
        super().__init__(_Tracker())
        self.config = config
        self.payload = None

    def handle_request(self, request):
        result = super().handle_request(request)
        if request['type'] == 'start_session':
            payload = self.payload
            assert result['session_id'] in self._all_inference_states
            record(self.config['root'], 'active', dict(pid=os.getpid(), device=DEVICE,
                slot=SLOT, visible=VISIBLE, session_id=result['session_id'],
                state_exists=True, frames=len(request['resource_path']),
                run_id=payload['run_id'], crop=payload['crop_xyxy'],
                source=payload['image_cache']['identity_sha256'], seed_sha256=payload['seed_sha256']))
            deadline = time.monotonic()+40
            while not (Path(self.config['root'])/f'release-{SLOT}').exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError('CPU protocol SDK gate was not released')
                time.sleep(.005)
            if self.config.get('fail_slot') == SLOT:
                raise RuntimeError('controlled independent SDK child failure')
        return result


def build(config):
    predictor = Session(dict(config))
    record(config['root'], 'factory', dict(pid=os.getpid(), slot=SLOT, visible=VISIBLE))
    return SimpleNamespace(predictor=predictor, config=dict(config), profile={},
        sam_runtime={'package_tree_sha256':('b' if config.get('runtime_drift_slot') == SLOT else 'a')*64,
            'startup_cuda_quiescence':dict(synchronized=True,
            worker_local_device=0, worker_index=SLOT)},
        torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda local:None)))


def execute(context, kind, payload):
    context.predictor.payload = payload
    context.predictor.model.tracker.score_logits = (payload['request_metadata'].get('fixture_logit', 1.),)
    output = execute_interpolation_tracker_task(context, kind, payload)
    path = Path(output['artifact_path'])
    packet = json.loads(path.read_text())
    packet['cuda_quiescence'].update(execution_device_id=DEVICE, worker_index=SLOT)
    if context.config.get('bad_proof_slot') == SLOT:
        packet['cuda_quiescence']['worker_index'] = 1-SLOT
    path.write_text(json.dumps(packet))
    record(context.config['root'], 'finished', dict(pid=os.getpid(), slot=SLOT, run_id=payload['run_id']))
    return output


def close(context):
    record(context.config['root'], 'closed', dict(pid=os.getpid(), slot=SLOT))
'''


def wait_for(predicate, timeout=20.):
    deadline = time.monotonic()+timeout
    while not predicate():
        assert time.monotonic() < deadline, 'independent SDK sessions did not reach their execution gate'
        time.sleep(.01)


class Admissions:
    def __init__(self, parties, *, physical_devices=1):
        self.parties = parties
        self.physical_devices = physical_devices
        self.pool = Pool((4+4*physical_devices)*GIB if physical_devices > 1 else (12*GIB if parties == 1 else 24*GIB))
        self.barrier = threading.Barrier(parties)

    @contextmanager
    def scope(self, requests, capacity):
        with sam_resources.admit_sam_parent_resources(self.pool, 4*GIB, 'two-session-proof',
                worker_count=self.physical_devices, execution_slots=2*self.physical_devices,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:64*GIB if self.physical_devices > 1 else 32*GIB) as profile:
            self.barrier.wait(10.)
            pixels = max(item['seed_mask'].size for item in requests)
            frames = max(item['frame_stop']-item['frame_start'] for item in requests)
            wave = sam_resources.cpu_wave_admission(
                sam_resources.cpu_session_bytes(frames, pixels)['estimated_peak_bytes'],
                frames*pixels, profile.assigned_cpu_wave_bytes, profile.execution_slots)
            assert wave['max_in_flight'] >= capacity
            for item in requests:
                item['resource_profile'] = profile
            with sam_resources.admit_sam_tracker_scope(profile, wave,
                    max_seed_pixels=pixels, max_frame_count=frames, max_in_flight=capacity) as permit:
                yield permit


def protocol(tmp_path, monkeypatch, *, fault=None):
    # These workers deliberately expose a non-existent token, and use only CPU Torch tensors.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-cpu-protocol-only')
    module_name = 'sam_two_session_adapter_'+uuid.uuid4().hex
    (tmp_path/(module_name+'.py')).write_text(ADAPTER)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    config = dict(root=str(tmp_path))
    if fault == 'sdk_failure':
        config['fail_slot'] = 0
    if fault == 'wrong_cuda_slot':
        config['bad_proof_slot'] = 0
    if fault == 'runtime_identity':
        config['runtime_drift_slot'] = 1
    pool = LtaWorkerPool((0,), LtaWorkerInit(adapter_module=module_name,
        adapter_factory='build', adapter_execute='execute', adapter_shutdown='close',
        adapter_config=config), workers_per_device=2, startup_timeout=20)
    held, acquisitions, releases, quarantines = {}, [], [], []
    def acquire(device, purpose):
        assert device == 0 and not held, 'a second SDK context tried to claim a separate physical GPU fence'
        lease = SimpleNamespace(device_index=device)
        held[device] = lease
        acquisitions.append(device)
        return lease
    def release(lease):
        assert pool.workers_settled or all((tmp_path/f'finished-{slot}.json').exists() for slot in (0, 1))
        assert held.pop(lease.device_index) is lease
        releases.append(lease.device_index)
    tracker = sam.SamInterpolationTracker(model_path='CPU protocol only', device_ids=(0,),
        workers_per_device=2, artifact_root=tmp_path/'runs', compute_lease_factory=acquire,
        compute_lease_release=release, residency_quarantine=quarantines.append)
    tracker._pool = pool
    tracker._residency_released = False
    if fault in ('wrong_event_slot', 'wrong_pid', 'stale_attempt'):
        receive = pool.wait_result
        def altered(*args, **kwargs):
            event = receive(*args, **kwargs)
            if event.worker_index == 0:
                if fault == 'wrong_event_slot':
                    return replace(event, worker_index=1, worker_pid=pool.pids_by_slot[0, 1])
                if fault == 'wrong_pid':
                    return replace(event, worker_pid=pool.pids_by_slot[0, 1])
                return replace(event, attempt_token='old-attempt')
            return event
        monkeypatch.setattr(pool, 'wait_result', altered)
    return tracker, pool, held, acquisitions, releases, quarantines


def release_slot(root, slot):
    (root/f'release-{slot}').touch()


@pytest.fixture
def protocol_factory(tmp_path, monkeypatch):
    trackers = []
    def create(**kwargs):
        result = protocol(tmp_path, monkeypatch, **kwargs)
        trackers.append(result[0])
        return result
    yield create
    for tracker in trackers:
        release_slot(tmp_path, 0)
        release_slot(tmp_path, 1)
        tracker._before_worker_shutdown = None
        tracker.close()


def test_two_independent_sdk_states_are_active_on_same_physical_gpu_and_share_fence(tmp_path, monkeypatch, protocol_factory):
    tracker, pool, held, acquisitions, releases, _ = protocol_factory()
    cache = cache_for(tmp_path, 'A', 13)
    admissions = Admissions(1)
    completed = threading.Event()
    seen = []
    iterator = tracker.iter_results
    def observed(*args, **kwargs):
        for index, result in iterator(*args, **kwargs):
            assert result.receipt['worker_startup_receipt']['startup_cuda_quiescence']['worker_index'] == result.receipt['dispatch']['worker_index']
            assert 'startup_cuda_quiescence' not in result.receipt['sam_runtime']
            seen.append(result.receipt['dispatch'])
            completed.set()
            yield index, result
    monkeypatch.setattr(tracker, 'iter_results', observed)
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(consume, tracker, cache,
            [request(index, label='A', frames=13) for index in range(2)], capacity=2, admissions=admissions)
        try:
            wait_for(lambda:all((tmp_path/f'active-{slot}.json').exists() for slot in (0, 1)))
            active = [json.loads((tmp_path/f'active-{slot}.json').read_text()) for slot in (0, 1)]
            assert all(row['state_exists'] and row['frames'] == 13 and row['device'] == 0 for row in active)
            assert {row['pid'] for row in active} == set(pool.pids_by_slot.values())
            assert {row['worker_pid'] for row in tracker.startup_receipts} == set(pool.pids_by_slot.values())
            assert len({row['pid'] for row in active}) == 2 and acquisitions == [0] and held and not releases
            assert not future.done() and tracker.worker_slots == ((0, 0), (0, 1)) and tracker.worker_count == 2
            release_slot(tmp_path, 0)
            assert completed.wait(10.)
            assert seen[0]['worker_index'] == 0 and held and not releases
            assert pool.is_alive(0, 1) and not (tmp_path/'finished-1.json').exists()
            release_slot(tmp_path, 1)
            assert len(future.result(15.)) == 2
            assert releases == [0] and not held
            assert {(row['execution_device_id'], row['worker_index']) for row in seen} == {(0, 0), (0, 1)}
        finally:
            release_slot(tmp_path, 0)
            release_slot(tmp_path, 1)
    tracker.close()
    assert pool.workers_settled and admissions.pool.in_use == 0 and not tracker._scopes


def test_same_original_ids_in_two_scopes_remain_exact_through_actual_child_slot_routes(tmp_path, protocol_factory):
    tracker, pool, held, _, releases, _ = protocol_factory()
    caches = [cache_for(tmp_path, label, frames) for label, frames in (('A', 13), ('B', 5))]
    admissions = Admissions(2)
    with ThreadPoolExecutor(max_workers=2) as threads:
        futures = [threads.submit(consume, tracker, cache,
            [request(0, label=label, frames=frames, run_id='same-original')], capacity=1, admissions=admissions)
            for cache, (label, frames) in zip(caches, (('A', 13), ('B', 5)))]
        try:
            def both_active():
                for future in futures:
                    if future.done():
                        future.result()
                return all((tmp_path/f'active-{slot}.json').exists() for slot in (0, 1))
            wait_for(both_active)
            active = [json.loads((tmp_path/f'active-{slot}.json').read_text()) for slot in (0, 1)]
            assert {row['run_id'] for row in active} == {'same-original'}
            assert {row['frames'] for row in active} == {5, 13}
            assert len({row['source'] for row in active}) == len({row['seed_sha256'] for row in active}) == 2
            assert len(pool._expected_attempts) == 2 and len({key for key in pool._expected_attempts}) == 2
            release_slot(tmp_path, 1)
            release_slot(tmp_path, 0)
            assert all(future.result(15.) == [(0, 'same-original')] for future in futures)
            assert releases == [0] and not held
        finally:
            release_slot(tmp_path, 0)
            release_slot(tmp_path, 1)
    tracker.close()
    assert pool.workers_settled and admissions.pool.in_use == 0


@pytest.mark.parametrize('fault', ('sdk_failure', 'wrong_cuda_slot', 'wrong_event_slot', 'wrong_pid', 'stale_attempt'))
def test_one_child_failure_keeps_peer_source_and_physical_fence_until_all_processes_settle(tmp_path, protocol_factory, fault):
    tracker, pool, held, _, releases, quarantines = protocol_factory(fault=fault)
    cache = cache_for(tmp_path, 'A', 13)
    admissions = Admissions(1)
    shutdown = []
    def before_shutdown():
        assert quarantines and held and not releases and pool.is_alive(0, 1)
        assert cache.path.exists() and admissions.pool.in_use >= 4*GIB
        assert list(tracker.artifact_root.glob('run-*'))
        shutdown.append(True)
    tracker._before_worker_shutdown = before_shutdown
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(consume, tracker, cache,
            [request(index, label='A', frames=13) for index in range(2)], capacity=2, admissions=admissions)
        try:
            wait_for(lambda:all((tmp_path/f'active-{slot}.json').exists() for slot in (0, 1)))
            release_slot(tmp_path, 0)
            with pytest.raises(RuntimeError):
                future.result(15.)
            assert shutdown and pool.workers_settled and releases == [0] and not held
            assert admissions.pool.in_use == 0 and not tracker._scopes
            assert not list(tracker.artifact_root.glob('run-*')) and cache.path.exists()
        finally:
            release_slot(tmp_path, 0)
            release_slot(tmp_path, 1)
            tracker._before_worker_shutdown = None
            tracker.close()


def test_cancellation_after_first_ack_keeps_other_session_and_cpu_source_owners(tmp_path, protocol_factory):
    tracker, pool, held, _, releases, quarantines = protocol_factory()
    cache = cache_for(tmp_path, 'A', 13)
    admissions = Admissions(1)
    paused, resume = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=1) as threads:
        future = threads.submit(consume, tracker, cache,
            [request(index, label='A', frames=13) for index in range(2)], capacity=2,
            admissions=admissions, paused=paused, resume=resume)
        try:
            wait_for(lambda:all((tmp_path/f'active-{slot}.json').exists() for slot in (0, 1)))
            release_slot(tmp_path, 0)
            assert paused.wait(10.)
            tracker.cancel('two-session consumer cancelled')
            assert quarantines and held and not releases and pool.is_alive(0, 1)
            assert cache.path.exists() and admissions.pool.in_use >= 4*GIB
        finally:
            resume.set()
        with pytest.raises(RuntimeError, match='cancel'):
            future.result(15.)
    assert pool.workers_settled and not held and releases == [0]
    assert admissions.pool.in_use == 0 and not tracker._scopes
    assert cache.path.exists() and not list(tracker.artifact_root.glob('run-*'))


@pytest.mark.parametrize('runtime_drift', (False, True))
def test_real_context_normal_interpolation_uses_both_funded_logical_slots(tmp_path, monkeypatch, protocol_factory, runtime_drift):
    from XTA import geometry
    from tests.test_sam_interpolation import _observations, _close
    from tests.test_sam_view_image_cache import context_for
    tracker, pool, held, _, releases, _ = protocol_factory(fault='runtime_identity' if runtime_drift else None)
    observed = _observations()
    original = observed.copy()
    source = np.arange(observed.size, dtype=np.uint16).reshape(observed.shape).astype(np.uint8)
    context = context_for(tmp_path/'context', source)
    context._runtime = tracker
    monkeypatch.setattr(context, '_start', lambda:None)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    account = Pool(12*GIB)
    def generate():
        with sam_resources.admit_sam_parent_resources(account, 4*GIB, 'real-context',
                worker_count=1, execution_slots=2, base_allowance_bytes=4*GIB,
                headroom_probe=lambda:32*GIB) as profile:
            with context.resource_scope(profile):
                return context.interpolate(observed, view=view, scope='real-two-session-bridge',
                    work_dir=tmp_path/'output', gap_distance=5, min_radius=0, interpolation_walk_back=0)
    merged = None
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(generate)
            try:
                wait_for(lambda:all((tmp_path/f'active-{slot}.json').exists() for slot in (0, 1)))
                assert held and not releases and not future.done()
                release_slot(tmp_path, 0)
                release_slot(tmp_path, 1)
                if runtime_drift:
                    with pytest.raises(RuntimeError, match='model/runtime identity changed'):
                        future.result(15.)
                else:
                    merged, stats, _ = future.result(15.)
            finally:
                release_slot(tmp_path, 0)
                release_slot(tmp_path, 1)
        if not runtime_drift:
            assert stats['sam_generated_runs'] == stats['sam_selected_runs'] == 2
            expected = observed.copy()
            expected[:, 9:15, 10:16] = 1
            np.testing.assert_array_equal(merged, expected)
        else:
            failures = list((tmp_path/'output').rglob('failure.json'))
            assert len(failures) == 1 and not json.loads(failures[0].read_text())['complete']
        np.testing.assert_array_equal(observed, original)
        assert account.in_use == 0 and releases == [0] and not held
    finally:
        _close(merged)
        context.close()


def test_first_four_dispatches_spread_physical_devices_before_eight_slots_fill(tmp_path):
    # Placement-only check; actual independent SDK execution is proved by the process tests above.
    class GatedPool:
        def __init__(self):
            self.closed = False
            self.submissions = []
            self.ready_events = tuple(SimpleNamespace(execution_device_id=device, worker_index=index,
                worker_pid=8000+2*device+index, metadata={}) for device in range(4) for index in range(2))
        def submit(self, task, *, execution_device_id, worker_index):
            self.submissions.append((execution_device_id, worker_index))
        def wait_result(self, *, timeout):
            time.sleep(timeout)
            raise TimeoutError
        def shutdown(self, *, timeout, force):
            self.closed = True
        def force_close(self, *, timeout):
            self.closed = True
        @property
        def workers_settled(self):
            return self.closed
    pool = GatedPool()
    cache = cache_for(tmp_path, 'A', 13)
    tracker = sam.SamInterpolationTracker(model_path='placement only', device_ids=(0, 1, 2, 3),
        workers_per_device=2, artifact_root=tmp_path/'runs')
    tracker._pool = pool
    tracker._residency_released = False
    tracker._crop_affinity[(cache.identity_sha256, (4, 2, 17, 11))] = (0, 0)
    admissions = Admissions(1, physical_devices=4)
    try:
        with ThreadPoolExecutor(max_workers=1) as threads:
            future = threads.submit(consume, tracker, cache,
                [request(index, label='A', frames=13) for index in range(8)], capacity=8, admissions=admissions)
            try:
                wait_for(lambda:len(pool.submissions) == 8)
                assert {device for device, index in pool.submissions[:4]} == set(range(4))
                assert set(pool.submissions) == set(tracker.worker_slots)
            finally:
                tracker.cancel('placement gate cancelled after proof')
            with pytest.raises(RuntimeError, match='cancel'):
                future.result(10.)
        assert admissions.pool.in_use == 0 and pool.workers_settled
    finally:
        tracker.close()
