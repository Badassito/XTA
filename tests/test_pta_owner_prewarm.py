"""GPU owners build their runtime and stage native intensity before first use."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_backend, pta, pta_workers
from XTA.pta_cpr import build_cpr_geometry

GIB = 1024 ** 3


@pytest.fixture
def telemetry(monkeypatch):
    record = SimpleNamespace(spans=[], gauges=[], fallbacks=[])

    def span(name):
        record.spans.append(name)
        return nullcontext()

    fake = SimpleNamespace(span=span, gauge=lambda name, fields: record.gauges.append((name, dict(fields))),
                           fallback=lambda name, exc=None: record.fallbacks.append((name, exc)))
    monkeypatch.setattr(pta_workers, 'runtime_telemetry', lambda: fake)
    monkeypatch.setattr(pta_workers, 'runtime_trace_event', lambda *_args, **_kwargs: None)
    return record


@pytest.fixture
def clean_worker(monkeypatch):
    for name, value in (('_WORKER_GPU_DEVICE_ID', None), ('_WORKER_GPU_RUNTIME', None),
                        ('_WORKER_GPU_RUNTIME_ERROR', None), ('_WORKER_GPU_CODEC_WARNING_EMITTED', False),
                        ('_WORKER_GPU_BATCH_CAP_WARNING_EMITTED', False)):
        monkeypatch.setattr(pta_workers, name, value)
    monkeypatch.setattr(pta_workers, '_WORKER_STATIC', {'gpu_render_threads': 8})
    monkeypatch.setattr(pta_workers, 'bind_current_thread_to_cpus', lambda cpus: None)
    monkeypatch.setattr(multiprocessing.current_process(), '_identity', (2,), raising=False)
    monkeypatch.delenv('YOLO_TTA_PTA_GPU_OWNER_PREWARM', raising=False)
    yield
    # The initializer resets module caches; leave none for later tests.
    pta_workers._WORKER_GPU_RUNTIME = None
    pta_workers._WORKER_GPU_RUNTIME_ERROR = None


# Initializer warm-up -------------------------------------------------------

def test_initializer_builds_runtime_once_after_rank_binding(monkeypatch, telemetry, clean_worker):
    bound, built = [], []
    monkeypatch.setattr(pta_workers, 'bind_current_thread_to_cpus', bound.append)
    runtime = {'device_id': 5}

    def build():
        built.append((pta_workers._WORKER_GPU_DEVICE_ID, list(bound),
                      pta_workers._WORKER_STATIC['gpu_render_threads']))
        pta_workers._WORKER_GPU_RUNTIME = runtime
        return runtime

    monkeypatch.setattr(pta_workers, '_gpu_runtime_for_worker', build)
    pta_workers._render_worker_initializer((9,), (3, 5), ((0, 1, 2), (3, 4)), None)
    # Worker identity 2 is rank 1: device 5 on its two local CPUs.
    assert built == [(5, [(3, 4)], 2)]
    assert telemetry.spans == ['pta.worker.gpu_runtime_prewarm']
    assert pta_workers._WORKER_GPU_RUNTIME is runtime
    assert pta_workers._WORKER_GPU_RUNTIME_ERROR is None


@pytest.mark.parametrize('case', ['disabled', 'cpu_pool'])
def test_lazy_runtime_is_kept_when_disabled_or_cpu_only(monkeypatch, telemetry, clean_worker, case):
    monkeypatch.setattr(pta_workers, '_gpu_runtime_for_worker', lambda: pytest.fail('unexpected warm-up'))
    if case == 'disabled':
        monkeypatch.setenv('YOLO_TTA_PTA_GPU_OWNER_PREWARM', '0')
        pta_workers._render_worker_initializer((), (0,), ((0,),), None)
        assert pta_workers._WORKER_GPU_DEVICE_ID == 0
    else:
        pta_workers._render_worker_initializer((0, 1), (), (), None)
        assert pta_workers._WORKER_GPU_DEVICE_ID is None
    assert pta_workers._WORKER_GPU_RUNTIME is None and telemetry.spans == []


@pytest.mark.parametrize('prewarm', ['1', '0'])
def test_warmup_failure_is_raised_by_the_first_task_like_a_lazy_build(monkeypatch, telemetry, clean_worker,
                                                                      capsys, prewarm):
    # Real builder: the missing GPU definition fails before any CUDA import.
    monkeypatch.setenv('YOLO_TTA_PTA_GPU_OWNER_PREWARM', prewarm)
    pta_workers._render_worker_initializer((), (0,), (), None)  # must not raise into Pool
    stored = pta_workers._WORKER_GPU_RUNTIME_ERROR
    assert (stored is not None) == (prewarm == '1')
    assert ('owner warm-up failed on cuda:0' in capsys.readouterr().out) == (prewarm == '1')
    assert [name for name, _exc in telemetry.fallbacks] == (
        ['pta.worker.gpu_runtime_prewarm'] if prewarm == '1' else [])
    payload = dict(gen=0, volume_shm='v', mask_shm='m')
    for _attempt in range(2):  # every task rebuilds and raises, as before
        with pytest.raises(RuntimeError, match='without a GPU augmentation definition') as raised:
            pta_workers._prepare_gpu_payload_task(payload)
        assert raised.value is not stored
        assert pta_workers._WORKER_GPU_RUNTIME_ERROR is None and pta_workers._WORKER_GPU_RUNTIME is None


@pytest.fixture
def fake_cuda_build(monkeypatch, clean_worker):
    """Run the real runtime builder against fake torch/cuda_backend modules."""
    record = SimpleNamespace(policies=[], set_device=[], fail_policy=[])

    def policy_builder(*, device, batch_size):
        if record.fail_policy:
            raise record.fail_policy.pop(0)
        record.policies.append((device, batch_size))
        return SimpleNamespace(apply_batch=lambda *args, **kwargs: None)

    torch = SimpleNamespace(set_num_threads=lambda count: None, cuda=SimpleNamespace(
        is_available=lambda: True, set_device=record.set_device.append))
    backend = SimpleNamespace(_GpuWorkerRenderEngine=lambda device: SimpleNamespace(device=f'fake:{device}'))
    modules = {'torch': torch, '.cuda_backend': backend}
    monkeypatch.setattr(pta_workers, 'importlib', SimpleNamespace(
        import_module=lambda name, package=None: modules[name]))
    augmentation = pta_workers.LoadedGpuAugmentation(Path('policy.py'), 'sha', 'build', 'fake', policy_builder)
    pta_workers._WORKER_STATIC.update(augmentation=augmentation, gpu_batch_size=7, image_format='png',
                                      jpeg_encode_backend='auto', gpu_device_ids=(3, 5))
    return record


def test_initializer_runs_the_real_builder_once(telemetry, clean_worker, fake_cuda_build):
    pta_workers._render_worker_initializer((9,), (3, 5), ((0, 1, 2), (3, 4)), None)
    runtime = pta_workers._WORKER_GPU_RUNTIME
    assert fake_cuda_build.set_device == [5] and fake_cuda_build.policies == [('cuda:5', 7)]
    assert runtime['device_id'] == 5 and runtime['azimuthal_renderer'].device == 'fake:cuda:5'
    # The first task reuses the warm runtime.
    assert pta_workers._gpu_runtime_for_worker() is runtime
    assert fake_cuda_build.policies == [('cuda:5', 7)]


def test_transient_warmup_failure_is_rebuilt_by_the_first_task(telemetry, clean_worker, fake_cuda_build):
    fake_cuda_build.fail_policy.append(RuntimeError('CUDA busy during warm-up'))
    pta_workers._render_worker_initializer((), (0,), (), None)
    assert pta_workers._WORKER_GPU_RUNTIME is None
    assert 'CUDA busy during warm-up' in str(pta_workers._WORKER_GPU_RUNTIME_ERROR)
    runtime = pta_workers._gpu_runtime_for_worker()  # the lazy build, as before
    assert runtime is pta_workers._WORKER_GPU_RUNTIME and runtime['device_id'] == 0
    assert fake_cuda_build.policies == [('cuda:0', 7)]
    assert pta_workers._WORKER_GPU_RUNTIME_ERROR is None


def test_failure_after_publishing_the_runtime_fails_only_the_first_task(monkeypatch, telemetry, clean_worker,
                                                                        fake_cuda_build):
    from XTA import spherical_host_cache

    monkeypatch.setattr(spherical_host_cache, 'install_spherical_host_cache',
                        mock.Mock(side_effect=MemoryError('host cache install failed')))
    pta_workers._render_worker_initializer((), (0,), (), None)
    runtime = pta_workers._WORKER_GPU_RUNTIME
    assert runtime is not None and isinstance(pta_workers._WORKER_GPU_RUNTIME_ERROR, MemoryError)
    # As with a lazy build: the first task fails, later tasks use the published runtime.
    with pytest.raises(MemoryError, match='host cache install failed'):
        pta_workers._gpu_runtime_for_worker()
    assert pta_workers._gpu_runtime_for_worker() is runtime
    assert fake_cuda_build.policies == [('cuda:0', 7)]


def test_initializer_reset_clears_a_stored_warmup_error(monkeypatch, telemetry, clean_worker):
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_RUNTIME_ERROR', RuntimeError('old pool'))
    pta_workers._render_worker_initializer()
    assert pta_workers._WORKER_GPU_RUNTIME_ERROR is None


def _first_task_runtime_probe():
    try:
        pta_workers._gpu_runtime_for_worker()
    except RuntimeError as exc:
        return str(exc)
    return 'built'


def _warmup_state_probe():
    error = pta_workers._WORKER_GPU_RUNTIME_ERROR
    return os.getpid(), None if error is None else str(error)


@pytest.mark.parametrize('method', ['fork', 'spawn'])
def test_pool_initializer_failure_reaches_tasks_without_respawning(monkeypatch, tmp_path, method):
    # Production GPU owners fork; spawn exercises the same initializer contract on Windows.
    if method == 'fork' and not pta_workers.gpu_fork_render_backend_available():
        pytest.skip('fork-only GPU pool')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '0')
    monkeypatch.setattr(pta_workers, '_WORKER_STATIC', {})
    pta_workers.set_worker_static_context(
        out_dir=tmp_path, split_active=False, image_format='png', png_compression=1, jpeg_quality=95,
        jpeg_encode_backend='auto', gpu_batch_size=1, gpu_render_threads=1, gpu_device_ids=(0,),
        augmentation=None)
    static = pta_workers._spawn_worker_static_payload() if method == 'spawn' else None
    context = multiprocessing.get_context(method)
    with context.Pool(1, initializer=pta_workers._render_worker_initializer,
                      initargs=((), (0,), (), static)) as pool:
        states = [pool.apply(_warmup_state_probe) for _ in range(2)]
        results = [pool.apply(_first_task_runtime_probe) for _ in range(2)]
    # One worker that was never respawned; its warm-up failure waited for the first task.
    assert len({pid for pid, _error in states}) == 1
    assert 'without a GPU augmentation definition' in states[0][1]
    assert all('without a GPU augmentation definition' in result for result in results)


# Early native-T intensity upload ------------------------------------------

class FakeEngine:
    def __init__(self, events, *, mode='resident', fail=None):
        self.events, self.mode, self.fail = events, mode, fail
        self._volume_key = None
        self._volume_gpu = None
        self.calls = []

    def ensure_volume_array(self, source, *, identity, resize_to_t=None, require_azimuthal_texture=False):
        self.calls.append(dict(source=source, identity=identity, resize_to_t=resize_to_t,
                               require_azimuthal_texture=require_azimuthal_texture,
                               thread=threading.current_thread().name))
        if self.fail is not None:
            raise self.fail
        key = (identity, tuple(source.shape), int(resize_to_t or source.shape[0]))
        if self._volume_key != key:
            self.events.append('upload')
            self._volume_key = key
            self._volume_gpu = SimpleNamespace(nbytes=source.nbytes) if self.mode == 'resident' else None
        return self.mode

    def retire_volume_array(self):
        self.events.append('renderer_retire')
        self._volume_key = self._volume_gpu = None

    def arm_residency_retry(self):
        self.events.append('arm_retry')
        return False


class FakeCuda:
    def __init__(self, events, free=64 * GIB):
        self.events, self.free = events, free

    def device(self, _device):
        return nullcontext()

    def synchronize(self, _device):
        self.events.append('fence')

    def mem_get_info(self, _device):
        return self.free, 80 * GIB

    def memory_allocated(self, _device):
        return 0

    def memory_reserved(self, _device):
        return 0


def _cartesian_plan():
    view = SimpleNamespace(name='transverse', family='transverse', physical_view_name='transverse')
    return SimpleNamespace(view=SimpleNamespace(shared_view=view), channel_variant=SimpleNamespace(kind='gray'))


@pytest.fixture
def staged(monkeypatch, telemetry):
    events = []
    logical = np.zeros((6, 5, 7), np.uint8)
    native = np.arange(3 * 5 * 7, dtype=np.uint8).reshape(3, 5, 7)
    pointer = int(logical.__array_interface__['data'][0])
    monkeypatch.setattr(pta_workers, '_WORKER_NATIVE_INTENSITY_BY_POINTER', {pointer: (native, 6)})
    monkeypatch.setattr(pta_workers, '_WORKER_VOLUME_IDENTITY_BY_POINTER', {pointer: 'shm:logical'})
    monkeypatch.setattr(cuda_backend, 'gpu_render_reserve_bytes', lambda: 12 * GIB)
    monkeypatch.delenv('YOLO_TTA_PTA_GPU_INTENSITY_PRESTAGE', raising=False)
    engine = FakeEngine(events)
    owner = SimpleNamespace(mask_gpu=object(), coverage_gpu=None,
                            arm_residency_retry=lambda: events.append('category_arm_retry'))
    key = (0, ('shm-logical', 'shm-mask', None, 'shm-native'))
    runtime = dict(torch=SimpleNamespace(cuda=FakeCuda(events)), device_id=0, azimuthal_renderer=engine,
                   azimuthal_render_lock=threading.Lock(), categorical_volume_owner=owner,
                   gpu_volume_task_key=key)
    yield SimpleNamespace(events=events, logical=logical, native=native, engine=engine, owner=owner,
                          runtime=runtime, key=key, telemetry=telemetry)
    pta_workers._join_intensity_prestage(runtime)
    finalizer = runtime.get('intensity_prestage_finalizer')
    if finalizer is not None:
        finalizer.cancel()


def _start(state, plans=None):
    """Start an early upload, if eligible, and join it as the next render boundary would."""
    pta_workers._maybe_prestage_render_intensity(state.runtime, [_cartesian_plan()] if plans is None else plans,
                                                 state.logical)
    thread = state.runtime.get('intensity_prestage')
    pta_workers._join_intensity_prestage(state.runtime)
    assert thread is None or not thread.is_alive()
    return thread


def test_prestage_makes_the_render_paths_first_call_in_the_background(staged):
    assert _start(staged) is not None
    call, = staged.engine.calls
    assert call['source'] is staged.native and call['identity'] == 'shm:logical'
    assert call['resize_to_t'] == 6 and call['thread'] == 'pta-intensity-prestage'
    assert staged.events == ['upload']
    gauge = dict(staged.telemetry.gauges)['pta.worker.intensity_prestage']
    assert gauge['mode'] == 'resident' and gauge['source_bytes'] == staged.native.nbytes
    assert staged.telemetry.spans == ['pta.worker.intensity_prestage']
    # The render path's own call then finds the source resident without a second upload.
    assert staged.engine.ensure_volume_array(staged.native, identity='shm:logical', resize_to_t=6,
                                             require_azimuthal_texture=True) == 'resident'
    assert staged.events == ['upload']
    # One attempt per source generation, even once the engine is retired again.
    staged.engine.retire_volume_array()
    assert _start(staged) is None and len(staged.engine.calls) == 2


@pytest.mark.parametrize('case', ['option_off', 'no_native_t', 'categorical_not_resident', 'source_already_resident',
                                  'no_renderer', 'no_lock', 'no_cuda_family', 'family_disabled', 'no_key'])
def test_prestage_preconditions(monkeypatch, staged, case):
    plans = [_cartesian_plan()]
    if case == 'option_off':
        monkeypatch.setenv('YOLO_TTA_PTA_GPU_INTENSITY_PRESTAGE', '0')
    elif case == 'no_native_t':
        pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER.clear()
    elif case == 'categorical_not_resident':
        staged.owner.mask_gpu = None
    elif case == 'source_already_resident':
        staged.engine._volume_key = ('other', (3, 5, 7), 6)
    elif case == 'no_renderer':
        staged.runtime['azimuthal_renderer'] = None
    elif case == 'no_lock':
        staged.runtime.pop('azimuthal_render_lock')
    elif case == 'no_cuda_family':
        monkeypatch.setenv('YOLO_TTA_PTA_GPU_CARTESIAN_RENDER', 'off')
    elif case == 'family_disabled':
        staged.runtime['cuda_projection_disabled_families'] = {'cartesian'}
    else:
        staged.runtime.pop('gpu_volume_task_key')
    assert _start(staged, plans) is None
    assert staged.engine.calls == [] and 'intensity_prestage_key' not in staged.runtime


@pytest.mark.parametrize('render_reserve_gib, categorical_mib, margin_gib', [
    (12, None, 12), (1, None, 2), (1, '4096', 4), (1, '0', 1)])
def test_tight_vram_leaves_admission_to_the_first_render_task(monkeypatch, staged, render_reserve_gib,
                                                              categorical_mib, margin_gib):
    # The larger of the render reserve and classification's categorical reserve stays free.
    monkeypatch.setattr(cuda_backend, 'gpu_render_reserve_bytes', lambda: render_reserve_gib * GIB)
    if categorical_mib is None:
        monkeypatch.delenv('PTA_GPU_CATEGORICAL_RESERVE_MIB', raising=False)
    else:
        monkeypatch.setenv('PTA_GPU_CATEGORICAL_RESERVE_MIB', categorical_mib)
    staged.runtime['torch'].cuda.free = staged.native.nbytes + margin_gib * GIB - 1
    _start(staged)
    assert staged.engine.calls == [] and staged.engine._volume_key is None
    assert dict(staged.telemetry.gauges)['pta.worker.intensity_prestage']['mode'] == 'skipped'
    staged.runtime.pop('intensity_prestage_key')
    staged.runtime['torch'].cuda.free += 1
    _start(staged)
    assert len(staged.engine.calls) == 1 and staged.engine._volume_key is not None


@pytest.mark.parametrize('mode', ['stream', 'unresolved'])
def test_declined_upload_resets_so_the_render_task_decides_afresh(staged, mode):
    staged.engine.mode = mode
    _start(staged)
    assert staged.events == ['upload', 'renderer_retire']
    assert staged.engine._volume_key is None


def test_upload_exception_is_reported_and_never_escapes(staged, capsys):
    staged.engine.fail = RuntimeError('fence failed')
    _start(staged)
    name, error = staged.telemetry.fallbacks[0]
    assert name == 'pta.worker.intensity_prestage' and str(error) == 'fence failed'
    assert 'early intensity upload skipped' in capsys.readouterr().out
    pta_workers._prepare_gpu_volume_task(staged.runtime, identity=staged.key[1], generation=0)


class Gate:
    """A prestage thread that finishes only when released."""

    def __init__(self, events):
        self.release = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(events,), name='pta-intensity-prestage')
        self.thread.start()

    def run(self, events):
        assert self.release.wait(10)
        events.append('prestage_done')


def test_same_source_classification_runs_beside_the_upload(staged):
    gate = Gate(staged.events)
    staged.runtime['intensity_prestage'] = gate.thread
    try:
        pta_workers._prepare_gpu_volume_task(staged.runtime, identity=staged.key[1], generation=0,
                                             wait_prestage=False)
        assert staged.events == ['category_arm_retry']  # no join and no renderer retry arm
        assert staged.runtime['intensity_prestage'] is gate.thread
    finally:
        gate.release.set()
    gate.thread.join(10)
    pta_workers._prepare_gpu_volume_task(staged.runtime, identity=staged.key[1], generation=0,
                                         wait_prestage=False)
    assert 'intensity_prestage' not in staged.runtime
    assert staged.events == ['category_arm_retry', 'prestage_done', 'arm_retry', 'category_arm_retry']


def test_render_boundary_waits_for_the_upload_before_any_render_work(staged):
    gate = Gate(staged.events)
    staged.runtime['intensity_prestage'] = gate.thread
    done = threading.Event()

    def render_boundary():
        pta_workers._prepare_gpu_volume_task(staged.runtime, identity=staged.key[1], generation=0)
        done.set()

    boundary = threading.Thread(target=render_boundary)
    boundary.start()
    assert not done.wait(0.2)
    gate.release.set()
    boundary.join(10)
    assert done.is_set()
    assert staged.events == ['prestage_done', 'arm_retry', 'category_arm_retry']
    assert 'pta.worker.intensity_prestage_wait' in staged.telemetry.spans


@pytest.mark.parametrize('wait_prestage', [False, True])
def test_generation_change_joins_the_upload_before_fencing_and_retiring(monkeypatch, staged, wait_prestage):
    from XTA import pta_cuda_masks

    monkeypatch.setattr(pta_cuda_masks, 'retire_gpu_categorical_volume',
                        lambda runtime: staged.events.append('category_retire'))
    gate = Gate(staged.events)
    staged.runtime['intensity_prestage'] = gate.thread
    staged.runtime['intensity_prestage_key'] = staged.key
    gate.release.set()
    pta_workers._prepare_gpu_volume_task(staged.runtime, identity=('next',), generation=1,
                                         wait_prestage=wait_prestage)
    assert staged.events == ['prestage_done', 'fence', 'category_retire', 'renderer_retire']
    assert 'intensity_prestage' not in staged.runtime and 'intensity_prestage_key' not in staged.runtime


@pytest.fixture
def classification_entry(monkeypatch, staged):
    from XTA import pta_gpu_classification

    payload = dict(gen=0, plans=[_cartesian_plan()], volume_shm='shm-logical', mask_shm='shm-mask',
                   semantic_coverage_shm=None, native_intensity_shm='shm-native')
    monkeypatch.setattr(pta_workers, '_worker_load_payload', lambda *_args: payload)
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_DEVICE_ID', 0)
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_RUNTIME', staged.runtime)
    monkeypatch.setattr(pta_workers, '_WORKER_GPU_RUNTIME_ERROR', None)
    monkeypatch.setattr(pta_workers, '_worker_volume_arrays', lambda _payload: (staged.logical, staged.logical, None))

    def classify(runtime, *_args, **_kwargs):
        staged.events.append('classify')
        assert 'intensity_prestage' not in runtime
        return 'classified'

    monkeypatch.setattr(pta_gpu_classification, 'classify_task', classify)
    return lambda: pta_workers._classification_task_entry(('payload', 1, object()))


def test_classification_entry_starts_upload_after_categorical_admission(staged, classification_entry):
    assert classification_entry() == 'classified'
    pta_workers._join_intensity_prestage(staged.runtime)
    assert staged.events == ['arm_retry', 'category_arm_retry', 'classify', 'upload']


@pytest.mark.parametrize('failure', ['thread_start', 'gate'])
def test_prestage_start_failure_never_fails_classification(monkeypatch, staged, classification_entry, failure):
    error = RuntimeError("can't start new thread")
    if failure == 'thread_start':
        monkeypatch.setattr(threading.Thread, 'start', lambda self: (_ for _ in ()).throw(error))
    else:
        monkeypatch.setattr(pta_workers, '_render_uses_cuda_intensity', lambda *_args: (_ for _ in ()).throw(error))
    assert classification_entry() == 'classified'
    assert staged.telemetry.fallbacks == [('pta.worker.intensity_prestage', error)]
    assert 'intensity_prestage' not in staged.runtime and staged.engine.calls == []
    # The next render boundary has nothing to join and proceeds as before.
    pta_workers._prepare_gpu_volume_task(staged.runtime, identity=staged.key[1], generation=0)
    assert staged.events == ['arm_retry', 'category_arm_retry', 'classify', 'arm_retry', 'category_arm_retry']


# The static gate must agree with the actual render path ---------------------

def _cpr_plan(tmp_path):
    geometry = build_cpr_geometry(np.array([[2, 4, 5], [9, 4, 5]]), (12, 10, 10), raster_size=8)
    view = pta.ViewInfo('cpr', 'CPR', 'cpr', geometry.num_slices, 8, 8, 'clamp', 12, 10, 10,
                        cpr_geometry=geometry)
    variant = pta.expand_channel_variants(pta.resolve_channel_formats('gray'))[0]
    return pta.build_plans_for_views(views=[view], out_dir=tmp_path / 'cpr', stem='tube',
        tile_configs=[], save_overlay=False, imgsz=8, label_enabled=True, channel_variants=(variant,),
        publish_images=False, publish_labels=False)[0]


@pytest.mark.parametrize('setting', [None, 'SPHERICAL', 'RADIAL', 'AZIMUTHAL', 'TILTED', 'CARTESIAN', 'CPR',
                                     'source_mode', 'disabled'])
def test_static_gate_matches_render_path_calls(tmp_path, monkeypatch, telemetry, setting):
    from tools.qualify_pta_gpu_masks import build_cases

    if setting == 'source_mode':
        monkeypatch.setenv('YOLO_TTA_AZIMUTHAL_SOURCE_MODE', 'nearest_xy_linear_t')
    elif setting is not None and setting != 'disabled':
        monkeypatch.setenv(f'YOLO_TTA_PTA_GPU_{setting}_RENDER', '0')
    cases = [(family, plan, frame) for family, plan, frame, item in build_cases(tmp_path, 24, 32, 16)
             if item == 'full']
    cases.append(('cpr', _cpr_plan(tmp_path), 3))
    cases.append(('custom_cartesian', replace(cases[0][1], channel_variant=pta.ChannelVariant(
        'custom', 'custom', 3, 3, False, (-1, 0, 1))), cases[0][2]))
    volume = np.zeros((24, 24, 24), np.uint8)
    seen = set()
    for family, plan, frame in cases:
        engine = FakeEngine([], mode='stream')
        runtime = dict(azimuthal_renderer=engine, azimuthal_render_lock=threading.Lock(),
                       cuda_projection_disabled_families={'radial', 'cpr'} if setting == 'disabled' else set())
        expected = pta_workers._render_uses_cuda_intensity(runtime, [plan])
        with mock.patch('builtins.print'):
            pta_workers._gpu_projected_item_image(runtime, volume, plan, frame, 'full')
        assert bool(engine.calls) == expected, (setting, family)
        seen.add(expected)
    assert seen == ({True, False} if setting is not None else {True})


# Real CUDA: the early upload is the render path's upload ---------------------

@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_PRESTAGE_CUDA') != '1',
                    reason='explicit CUDA qualification only')
@pytest.mark.parametrize('source', ['ramp13', 'zeros7', 'full23', 'random1'])
def test_cuda_prestaged_native_intensity_renders_identically(tmp_path, monkeypatch, source):
    lock = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert lock['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    from XTA import media, pta_cuda_masks
    from tools.qualify_pta_gpu_masks import build_cases, make_volumes

    monkeypatch.setenv('YOLO_TTA_GPU_RENDER_RESERVE_GIB', '1')
    monkeypatch.setenv('YOLO_TTA_FAST_GEOMETRY', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_FP32', '0')
    intensity, mask, coverage = make_volumes(24)
    # Edge native depths (1 and 23 of 24) and constant/random content.
    native = {'ramp13': intensity[:13].copy(), 'zeros7': np.zeros((7, 24, 24), np.uint8),
              'full23': np.full((23, 24, 24), 255, np.uint8),
              'random1': np.random.default_rng(3).integers(0, 256, (1, 24, 24), dtype=np.uint8)}[source]
    logical = media.resize_volume_to_processing_cube_gray8(native, (24, 24, 24), tmp_path / 'logical.uint8',
                                                          workers=2, prefer_memory=True, reserve_bytes=0)
    pointer = int(logical.__array_interface__['data'][0])
    monkeypatch.setitem(pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER, pointer, (native, 24))
    monkeypatch.setitem(pta_workers._WORKER_VOLUME_IDENTITY_BY_POINTER, pointer, 'shm:logical')
    cases = build_cases(tmp_path, 24, 32, 16)

    def runtime_for():
        return dict(torch=torch, device_id=0, azimuthal_renderer=cuda_backend._GpuWorkerRenderEngine('cuda:0'),
                    azimuthal_render_lock=threading.RLock(), azimuthal_texture_required=True,
                    gpu_volume_task_key=(0, ('logical',)))

    lazy, early = runtime_for(), runtime_for()
    # Classification first owns the categorical source on the shared render stream.
    for runtime in (lazy, early):
        family, plan, frame, item = cases[0]
        assert pta_cuda_masks.render_gpu_categorical_item(runtime, mask, coverage, plan, frame, item,
                                                          identity='logical-mask') is not None
    pta_workers._maybe_prestage_render_intensity(early, [case[1] for case in cases], logical)
    assert early.get('intensity_prestage') is not None
    # Categorical sampling keeps running while the upload proceeds.
    expected_masks = []
    for family, plan, frame, item in cases:
        rendered = pta_cuda_masks.render_gpu_categorical_item(early, mask, coverage, plan, frame, item,
                                                              identity='logical-mask')
        rendered[2].synchronize()
        expected_masks.append(rendered[0].cpu().numpy())
    pta_workers._prepare_gpu_volume_task(early, identity=('logical',), generation=0)
    engine = early['azimuthal_renderer']
    staged_tensor = engine._volume_gpu
    assert staged_tensor is not None and engine._mode == 'resident'
    np.testing.assert_array_equal(staged_tensor.cpu().numpy(), native)
    for index, (family, plan, frame, item) in enumerate(cases):
        images = []
        for runtime in (lazy, early):
            image, ready = pta_workers._gpu_projected_item_image(runtime, logical, plan, frame, item)
            ready.synchronize()
            images.append(image.cpu().numpy())
        np.testing.assert_array_equal(images[0], images[1], err_msg=f'{family} {item}')
        rendered = pta_cuda_masks.render_gpu_categorical_item(lazy, mask, coverage, plan, frame, item,
                                                              identity='logical-mask')
        rendered[2].synchronize()
        np.testing.assert_array_equal(rendered[0].cpu().numpy(), expected_masks[index])
    assert engine._volume_gpu is staged_tensor  # never uploaded again
    assert lazy['azimuthal_renderer']._source_residency_timings.keys() == engine._source_residency_timings.keys()
    for runtime in (lazy, early):
        pta_cuda_masks.retire_gpu_categorical_volume(runtime)
        runtime['azimuthal_renderer'].release_inference_assets()
        finalizer = runtime.get('intensity_prestage_finalizer')
        if finalizer is not None:
            finalizer.cancel()
