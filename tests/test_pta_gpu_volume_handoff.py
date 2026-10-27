"""CPU-only checks for the serial GPU process worker's source handoff."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from XTA import pta_cuda_masks, pta_gpu_classification, pta_workers
from XTA.pta_gpu_publication import GpuPublicationTask, _GpuBatch


A = ('volume-a', 'mask-a', None, None)
B = ('volume-b', 'mask-b', 'coverage-b', 'native-b')


class FakeCuda:
    def __init__(self, events):
        self.events = events
        self.fail_fence = False
        self.queries = 0

    def device(self, _device):
        return nullcontext()

    def stream(self, _stream):
        return nullcontext()

    def synchronize(self, _device):
        self.events.append('fence')
        if self.fail_fence:
            raise RuntimeError('simulated device fence failure')

    def mem_get_info(self, _device):
        self.queries += 1
        return 64 * 1024**3, 80 * 1024**3

    def memory_allocated(self, _device):
        return 1024

    def memory_reserved(self, _device):
        return 2048


class FakeRenderer:
    def __init__(self, events):
        self.events = events
        self._volume_gpu = SimpleNamespace(nbytes=8)
        self._volume_key = A
        self.fail_retire = False

    def retire_volume_array(self):
        self.events.append('renderer_retire')
        if self.fail_retire:
            raise RuntimeError('simulated renderer retirement failure')
        self._volume_gpu = self._volume_key = None

    def arm_residency_retry(self):
        self.events.append('arm_retry')
        return False


@pytest.fixture
def worker(monkeypatch):
    events = []
    cuda = FakeCuda(events)
    renderer = FakeRenderer(events)
    owner = SimpleNamespace(mask_gpu=SimpleNamespace(nbytes=8), coverage_gpu=None, fail_retire=False)
    owner.arm_residency_retry = lambda: events.append('category_arm_retry') or False
    resources = SimpleNamespace(poisoned=False, closed=False)
    policy, encoder = object(), object()
    runtime = dict(torch=SimpleNamespace(cuda=cuda), device_id=0,
        azimuthal_renderer=renderer, categorical_volume_owner=owner,
        publication_resources=resources, policy=policy, encoder=encoder,
        gpu_volume_task_key=(0, A), source_residency_telemetry_key='old-source',
        cuda_projection_disabled_families={'unsupported'}, codec_error='retained-codec-state')
    telemetry = SimpleNamespace(span=lambda *_args, **_kwargs: nullcontext(),
                                gauge=lambda *_args, **_kwargs: None)

    def retire_category(actual):
        assert actual is runtime
        events.append('category_retire')
        if owner.fail_retire:
            raise RuntimeError('simulated categorical retirement failure')
        owner.mask_gpu = owner.coverage_gpu = None

    monkeypatch.setattr(pta_cuda_masks, 'retire_gpu_categorical_volume', retire_category)
    monkeypatch.setattr(pta_workers, 'runtime_telemetry', lambda: telemetry)
    monkeypatch.setattr(pta_workers, 'runtime_trace_event', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_DEVICE_ID', 0)
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_RUNTIME', runtime)
    return SimpleNamespace(runtime=runtime, events=events, cuda=cuda,
        renderer=renderer, owner=owner, resources=resources, policy=policy, encoder=encoder)


def prepare(worker, identity, generation):
    pta_workers._prepare_gpu_volume_task(worker.runtime, identity=identity, generation=generation)


def assert_persistent_objects(worker):
    assert worker.runtime['policy'] is worker.policy
    assert worker.runtime['encoder'] is worker.encoder
    assert worker.runtime['publication_resources'] is worker.resources
    assert not worker.resources.closed
    assert worker.runtime['cuda_projection_disabled_families'] == {'unsupported'}
    assert worker.runtime['codec_error'] == 'retained-codec-state'


def test_changed_source_fences_and_retires_before_committing_marker(worker):
    prepare(worker, B, 1)
    assert worker.events == ['fence', 'category_retire', 'renderer_retire']
    assert worker.runtime['gpu_volume_task_key'] == (1, B)
    assert worker.owner.mask_gpu is None and worker.renderer._volume_gpu is None
    assert 'source_residency_telemetry_key' not in worker.runtime
    assert_persistent_objects(worker)


def test_unmarked_runtime_with_existing_owners_must_retire(worker):
    worker.runtime.pop('gpu_volume_task_key')
    prepare(worker, A, 0)
    assert worker.events == ['fence', 'category_retire', 'renderer_retire']
    assert worker.runtime['gpu_volume_task_key'] == (0, A)


def test_same_source_only_arms_retry_without_query_or_retirement(worker):
    prepare(worker, A, 0)
    prepare(worker, A, 0)
    assert worker.events == ['arm_retry', 'category_arm_retry', 'arm_retry', 'category_arm_retry']
    assert worker.cuda.queries == 0
    assert worker.owner.mask_gpu is not None and worker.renderer._volume_gpu is not None
    assert_persistent_objects(worker)


def test_generation_a_b_a_retires_once_per_transition(worker):
    for identity, generation in [(A, 0), (B, 1), (B, 1), (A, 0), (A, 0)]:
        prepare(worker, identity, generation)
    assert worker.events == ['arm_retry', 'category_arm_retry', 'fence', 'category_retire', 'renderer_retire',
        'arm_retry', 'category_arm_retry', 'fence', 'category_retire', 'renderer_retire',
        'arm_retry', 'category_arm_retry']
    assert worker.runtime['gpu_volume_task_key'] == (0, A)
    assert_persistent_objects(worker)


@pytest.mark.parametrize('component', range(4))
def test_direct_identity_includes_all_source_blocks(worker, component):
    changed = list(A)
    changed[component] = f'changed-source-{component}'
    prepare(worker, tuple(changed), 0)
    assert worker.events == ['fence', 'category_retire', 'renderer_retire']
    assert worker.runtime['gpu_volume_task_key'] == (0, tuple(changed))


@pytest.mark.parametrize('failure', ['fence', 'category', 'renderer'])
def test_failed_handoff_does_not_commit_or_retry_unsafe_release(worker, failure):
    worker.cuda.fail_fence = failure == 'fence'
    worker.owner.fail_retire = failure == 'category'
    worker.renderer.fail_retire = failure == 'renderer'
    with pytest.raises(RuntimeError, match='safely retired'):
        prepare(worker, B, 1)
    assert worker.runtime['gpu_volume_task_key'] == (0, A)
    assert worker.runtime['gpu_volume_handoff_error']
    first_events = list(worker.events)
    if failure == 'fence':
        assert first_events == ['fence']
        assert worker.owner.mask_gpu is not None
    elif failure == 'category':
        assert first_events == ['fence', 'category_retire']
        assert worker.renderer._volume_gpu is not None
    else:
        assert first_events == ['fence', 'category_retire', 'renderer_retire']
    with pytest.raises(RuntimeError, match='previously failed'):
        prepare(worker, A, 0)
    assert worker.events == first_events
    assert_persistent_objects(worker)


@pytest.mark.parametrize('state', ['poisoned', 'closed'])
def test_publication_health_is_checked_even_for_same_source(worker, state):
    setattr(worker.resources, state, True)
    with pytest.raises(RuntimeError, match='healthy publication'):
        prepare(worker, A, 0)
    assert worker.events == []
    assert worker.runtime['gpu_volume_task_key'] == (0, A)


def install_entries(monkeypatch, worker, payloads, body=None):
    volume, mask = np.zeros((2, 2, 2), np.uint8), np.ones((2, 2, 2), np.uint8)
    monkeypatch.setattr(pta_workers, '_WORKER_VOLUME_IDENTITY_BY_POINTER',
                        {int(volume.__array_interface__['data'][0]): 'fake-volume'})
    monkeypatch.setattr(pta_workers, '_worker_load_payload', lambda name, size: payloads[name])

    def attach(payload):
        identity = tuple(payload.get(name) for name in ('volume_shm', 'mask_shm',
            'semantic_coverage_shm', 'native_intensity_shm'))
        if pta_workers._WORKER_GPU_DEVICE_ID is not None:
            assert worker.runtime['gpu_volume_task_key'] == (payload['gen'], identity)
        worker.events.append(('attach', payload['gen']))
        return volume, mask, None

    def execute(payload, task, *_arrays):
        worker.events.append(('execute', payload['gen']))
        if body is not None:
            body(payload)
        return 'rendered'

    def classify(runtime, *_args, **_kwargs):
        assert runtime is worker.runtime
        worker.events.append('classify')
        return 'classified'

    monkeypatch.setattr(pta_workers, '_worker_volume_arrays', attach)
    monkeypatch.setattr(pta_workers, '_execute_traced_render_task', execute)
    monkeypatch.setattr(pta_gpu_classification, 'classify_task', classify)


def payload(identity, generation):
    return dict(zip(('volume_shm', 'mask_shm', 'semantic_coverage_shm',
                     'native_intensity_shm'), identity), gen=generation, plans=[], stem=str(generation))


def test_actual_process_entries_share_boundary_a_b_a(monkeypatch, worker):
    install_entries(monkeypatch, worker, {'a': payload(A, 0), 'b': payload(B, 1)})
    assert pta_workers._render_task_entry(('a', 1, object())) == 'rendered'
    assert pta_workers._classification_task_entry(('b', 1, object())) == 'classified'
    assert pta_workers._render_task_entry(('b', 1, object())) == 'rendered'
    assert pta_workers._render_task_entry(('a', 1, object())) == 'rendered'
    assert worker.events == ['arm_retry', 'category_arm_retry', ('attach', 0), ('execute', 0),
        'fence', 'category_retire', 'renderer_retire', ('attach', 1), 'classify',
        'arm_retry', 'category_arm_retry', ('attach', 1), ('execute', 1),
        'fence', 'category_retire', 'renderer_retire', ('attach', 0), ('execute', 0)]
    assert_persistent_objects(worker)


@pytest.mark.parametrize('entry', ['_render_task_entry', '_classification_task_entry'])
def test_actual_entry_handoff_failure_prevents_host_attach(monkeypatch, worker, entry):
    install_entries(monkeypatch, worker, {'b': payload(B, 1)})
    worker.cuda.fail_fence = True
    with pytest.raises(RuntimeError, match='safely retired'):
        getattr(pta_workers, entry)(('b', 1, object()))
    assert worker.events == ['fence']


def test_phase_payload_name_does_not_change_source_identity(monkeypatch, worker):
    install_entries(monkeypatch, worker, {'phase-a': payload(A, 0), 'phase-b': payload(A, 0)})
    pta_workers._render_task_entry(('phase-a', 1, object()))
    pta_workers._render_task_entry(('phase-b', 1, object()))
    assert worker.events.count('arm_retry') == 2
    assert 'fence' not in worker.events


def test_cpu_process_entry_does_not_initialize_gpu_runtime(monkeypatch, worker):
    install_entries(monkeypatch, worker, {'a': payload(A, 0)})
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_DEVICE_ID', None)
    monkeypatch.setattr(pta_workers, '_gpu_runtime_for_worker',
                        lambda: pytest.fail('CPU process attempted CUDA runtime creation'))
    assert pta_workers._render_task_entry(('a', 1, object())) == 'rendered'
    assert worker.events == [('attach', 0), ('execute', 0)]


def test_thread_entry_does_not_use_process_gpu_handoff(monkeypatch, worker):
    install_entries(monkeypatch, worker, {})
    monkeypatch.setattr(pta_workers, '_gpu_runtime_for_worker',
                        lambda: pytest.fail('thread backend attempted CUDA runtime creation'))
    monkeypatch.setattr(pta_workers, '_prepare_gpu_payload_task',
                        lambda _payload: pytest.fail('thread backend attempted GPU source handoff'))
    data = dict(gen=0, plans=[], tasks=[object()], volume=object(), mask=object())
    assert pta_workers._render_task_entry_thread(data, 0) == 'rendered'
    assert worker.events == [('execute', 0)]


def test_next_serial_entry_waits_for_previous_host_publication_drain(monkeypatch, worker):
    """Model one process rank, not concurrent calls through the CPU backend."""
    host_started, release_host = threading.Event(), threading.Event()
    resources = worker.resources
    resources.runtime = worker.runtime
    resources.torch = worker.runtime['torch']
    resources.device = 0
    resources.gpu_bytes = resources.host_bytes = 1024
    resources.label_executor = resources.file_executor = None
    resources.lock = threading.Lock()
    resources.counts = {}
    resources.measure = lambda _name: nullcontext()
    resources.report_if_due = lambda: None
    resources.quarantine = lambda _payload: pytest.fail('fake publication should fence safely')
    resources.stream = SimpleNamespace(wait_event=lambda _event: None, synchronize=lambda: None)
    tensor = SimpleNamespace(shape=(1, 1, 2, 2), record_stream=lambda _stream: None)

    def body(data):
        if data['gen'] != 0:
            return

        def consume(*_args, publication, **_kwargs):
            def write_host():
                host_started.set()
                assert release_host.wait(10), 'test host publication was not released'
                worker.events.append('host_finished')
            publication.submit_host(1, write_host)
            return 'published'

        with GpuPublicationTask(resources, consume, lambda _value: None) as publication:
            with publication.reserve(1) as reservation:
                reservation.submit(_GpuBatch(dict(batch_images=tensor, batch_masks=tensor), object(), ()))

    install_entries(monkeypatch, worker, {'a': payload(A, 0), 'b': payload(B, 1)}, body)
    with ThreadPoolExecutor(max_workers=1) as serial_rank, \
            ThreadPoolExecutor(max_workers=1) as gpu_executor, \
            ThreadPoolExecutor(max_workers=1) as host_executor:
        resources.gpu_executor, resources.host_executor = gpu_executor, host_executor
        first = serial_rank.submit(pta_workers._render_task_entry, ('a', 1, object()))
        try:
            assert host_started.wait(10), 'actual publication consumer did not start'
            second = serial_rank.submit(pta_workers._render_task_entry, ('b', 1, object()))
            assert not first.done() and not second.running()
            assert 'fence' not in worker.events and ('attach', 1) not in worker.events
        finally:
            release_host.set()
        assert first.result(timeout=10) == second.result(timeout=10) == 'rendered'
    assert worker.events.index('host_finished') < worker.events.index('fence')
    assert worker.events.index('fence') < worker.events.index(('attach', 1))
