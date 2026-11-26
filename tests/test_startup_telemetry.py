"""Startup boundaries preserve lazy execution, failure propagation and readiness."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import pickle
import queue
import shutil
import subprocess
import sys
import weakref
from types import SimpleNamespace
from unittest import mock

import pytest
import numpy as np

from XTA import inference, media, runtime, workers
from tests import test_worker_raster_plan_provenance as raster_helpers


@pytest.mark.parametrize('kind, target', [
    ('cpu', '_cpu_inference_worker_main'),
    ('gpu', '_gpu_inference_worker_main'),
])
def test_bootstrap_measures_import_and_decodes_worker_config(kind, target, monkeypatch, capsys):
    from XTA import tta_worker_bootstrap

    entry = mock.Mock()
    monkeypatch.setattr(workers, target, entry)
    config = {'view': SimpleNamespace(name='transverse'), 'numa_affinity_cpus': (1, 3)}
    task_channel, result_channel = object(), object()
    tta_worker_bootstrap.start_detector_worker(
        kind, 2, 'model', pickle.dumps(config), task_channel, result_channel)

    assert entry.call_args.args == (2, 'model', config, task_channel, result_channel)
    receipt = entry.call_args.kwargs['bootstrap_receipt']
    assert receipt['bootstrap_wall_time_ns'] > 0 and receipt['bootstrap_monotonic_ns'] > 0
    assert receipt['import_seconds'] >= 0 and receipt['init_decode_seconds'] >= 0
    stdout = capsys.readouterr().out
    assert stdout.index('bootstrap entered') < stdout.index('worker runtime imported')


def test_runtime_versions_only_inspects_loaded_modules(monkeypatch):
    from XTA.tta_worker_bootstrap import loaded_runtime_versions

    monkeypatch.setitem(sys.modules, 'cupy', None)
    monkeypatch.setitem(sys.modules, 'tensorrt', SimpleNamespace(__version__='0' * 200))
    versions = loaded_runtime_versions()
    assert versions['python'] == sys.version.split()[0]
    assert versions['numpy'] == np.__version__
    assert len(versions['tensorrt']) == 128
    assert 'cupy' not in versions


@pytest.fixture
def diagnostics(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '0')
    telemetry = runtime.RuntimeTelemetry()
    telemetry.flush_seconds = 1e9
    events = []
    for module in (runtime, media, workers, inference):
        monkeypatch.setattr(module, 'runtime_telemetry', lambda: telemetry)
        if hasattr(module, 'runtime_trace_event'):
            monkeypatch.setattr(module, 'runtime_trace_event',
                                lambda name, **fields: events.append((name, fields)))
    return telemetry, events


@pytest.mark.parametrize('failed', [False, True])
def test_parent_kernel_reports_work_and_flush_visible_progress(diagnostics, monkeypatch, capsys, failed):
    from XTA import pipeline

    telemetry, events = diagnostics
    monkeypatch.setattr(pipeline, 'runtime_telemetry', lambda: telemetry)
    monkeypatch.setattr(pipeline, 'runtime_trace_event',
                        lambda name, **fields: events.append((name, fields)))

    def warm():
        assert 'Warming parent kernel sample.' in capsys.readouterr().out
        assert events == [('startup_parent_warmup_start', {'kernel': 'sample'})]
        if failed:
            raise OSError('compiler failed')
        return {'signatures': 1}

    if failed:
        with pytest.raises(OSError, match='compiler failed'):
            pipeline._warm_parent_kernel('sample', warm)
    else:
        assert pipeline._warm_parent_kernel('sample', warm)['signatures'] == 1
    assert events[-1][0] == 'startup_parent_warmup_done'
    assert events[-1][1]['failed'] is failed
    outcome = 'failed' if failed else 'finished'
    assert f'Parent kernel sample {outcome} in' in capsys.readouterr().out


@pytest.mark.skipif(not (shutil.which('ffmpeg') and shutil.which('ffprobe')), reason='Requires FFmpeg')
def test_cpu_cli_warms_after_worker_registration_and_before_admission(tmp_path):
    pytest.importorskip('openvino')
    from tools.qualify_tta_reconciliation import create_cpu_fixtures, runtime_environment

    fixture = tmp_path / 'fixture'
    create_cpu_fixtures(fixture, size=64, frames=6)
    telemetry = tmp_path / 'telemetry'
    env = runtime_environment(tmp_path)
    env.update(CUDA_VISIBLE_DEVICES='', YOLO_TTA_TASK_TRACE='1', YOLO_TTA_TELEMETRY='1',
               YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER='0', YOLO_TTA_TELEMETRY_DIR=str(telemetry))
    env.setdefault('XTA_NATIVE_SCHEDULER_CACHE', str(tmp_path / 'native-cache'))
    output = tmp_path / 'output'
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'XTA', '--mode', 'tta',
               '--input', str(fixture / 'input.mkv'), '--model', f'cpu:{fixture / "model.xml"}',
               '--device', 'cpu', '--quantize', 'cpu:fp32', '--batch', 'cpu:1', '--channel_format', 'gray',
               '--imgsz', '64', '--cpu_threads', '2', '--cpu_streams', '1', '--cpu_infer_requests', '2',
               '--conf', '.80', '--min_conf', '0', '--min_radius', '0', '--enable_cartesian', 'transverse',
               '--output', str(output), '--temp', str(tmp_path / 'runtime'), '--save', 'voxel_volume']
    log = tmp_path / 'startup.log'
    with log.open('wb') as stream:
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    assert completed.returncode == 0, log.read_text(encoding='utf-8')[-4000:]
    assert json.loads((output / 'manifest.json').read_bytes())['status'] == 'complete'

    events = [event for path in telemetry.glob('*.jsonl')
              for row in map(json.loads, path.read_bytes().splitlines()) for event in row['events']]
    registered = next(event for event in events if event['event'] == 'startup_detector_workers_registered')
    parent_events = sorted((event for event in events if event['pid'] == registered['pid']),
                           key=lambda event: event['sequence'])
    names = [event['event'] for event in parent_events]
    assert names.index('startup_worker_spawn_done') < names.index('startup_detector_workers_registered')
    assert names.index('startup_detector_workers_registered') < names.index('startup_parent_warmup_start')
    assert max(i for i, name in enumerate(names) if name == 'startup_parent_warmup_done') < (
        names.index('startup_detector_inventory_start'))
    admission = parent_events[names.index('startup_detector_inventory_start')]['monotonic_ns']
    dequeued = min(event['monotonic_ns'] for event in events if event['event'] == 'worker_dequeue')
    assert admission <= dequeued


@pytest.mark.parametrize('metadata_count', ['17', None])
def test_probe_records_only_the_subprocesses_needed_for_exact_count(diagnostics, metadata_count):
    telemetry, events = diagnostics
    response = dict(width=80, height=64, avg_frame_rate='30/1')
    if metadata_count is not None:
        response['nb_frames'] = metadata_count
    responses = [SimpleNamespace(stdout=json.dumps(dict(streams=[response]))),
                 SimpleNamespace(stdout=json.dumps(dict(streams=[dict(nb_read_packets='17')])))]
    with (mock.patch.object(media, '_require_bin'),
          mock.patch.object(media.subprocess, 'run', side_effect=responses) as run):
        result = media.ffprobe_info(Path('source.mkv'))
    assert result == dict(width=80, height=64, fps=30., num_frames=17)
    assert run.call_count == (1 if metadata_count else 2)
    phases = telemetry.snapshot()['phases']
    assert phases['startup.source_probe']['calls'] == 1
    assert phases['startup.source_probe.metadata']['calls'] == 1
    assert ('startup.source_probe.packet_count' in phases) == (metadata_count is None)
    assert [fields['phase'] for name, fields in events if name.endswith('_done')] == (
        ['metadata'] if metadata_count else ['metadata', 'packet_count'])
    assert all(not fields['failed'] for name, fields in events if name.endswith('_done'))


@pytest.mark.parametrize('failing_phase', ['metadata', 'packet_count'])
def test_probe_failure_keeps_the_original_error_and_reports_failed_phase(diagnostics, failing_phase):
    telemetry, events = diagnostics
    failure = OSError('source unreadable')
    metadata = SimpleNamespace(stdout=json.dumps(dict(streams=[dict(
        width=80, height=64, avg_frame_rate='30/1')])))
    responses = [failure] if failing_phase == 'metadata' else [metadata, failure]
    with (mock.patch.object(media, '_require_bin'),
          mock.patch.object(media, '_spawn_subprocess_with_retry', side_effect=responses),
          pytest.raises(OSError) as caught):
        media.ffprobe_info(Path('source.mkv'))
    assert caught.value is failure
    assert events[-1] == ('startup_source_probe_done', dict(phase=failing_phase, failed=True))
    counters = telemetry.snapshot()['counters']
    assert counters['startup.source_probe.errors'] == 1
    assert counters[f'startup.source_probe.{failing_phase}.errors'] == 1


def test_backend_setup_is_cold_only_and_never_iterates_prediction(diagnostics):
    telemetry, events = diagnostics
    model = SimpleNamespace(predictor=None)

    def predict(**kwargs):
        assert kwargs['stream'] is True
        model.predictor = SimpleNamespace(model=object())
        def infer():
            raise AssertionError('diagnostics forced warmup or inference')
            yield
        return infer()

    model.predict = mock.Mock(side_effect=predict)
    cfg = inference.PredictConfig(imgsz=16, conf=.5, device='cuda:0', quantize=16, batch=1)
    with (mock.patch.object(inference, 'ensure_yolo_ready_for_predict'),
          mock.patch.object(inference, 'validate_yolo_model_input_channels')):
        assert inference._ensure_predictor_for_direct_predict(model, cfg) is model.predictor
        assert inference._ensure_predictor_for_direct_predict(model, cfg) is model.predictor
    model.predict.assert_called_once()
    assert telemetry.snapshot()['phases']['startup.yolo.backend_setup']['calls'] == 1
    assert [name for name, _ in events] == ['startup_backend_setup_start', 'startup_backend_setup_done']
    assert events[-1][1]['failed'] is False


def test_failed_backend_setup_preserves_fallback_and_marks_failure(diagnostics):
    telemetry, events = diagnostics
    model = SimpleNamespace(predictor=None, predict=mock.Mock(side_effect=RuntimeError('load failed')))
    cfg = inference.PredictConfig(imgsz=16, conf=.5, device='cuda:0', quantize=16, batch=1)
    with redirect_stdout(io.StringIO()):
        assert inference._ensure_predictor_for_direct_predict(model, cfg) is None
    assert events[-1][1]['failed'] is True
    assert telemetry.snapshot()['counters']['startup.yolo.backend_setup.errors'] == 1


@pytest.fixture
def in_process_worker_io(monkeypatch):
    """Supply descriptors to the model-startup unit without a separate actor process."""
    from XTA import tta_worker_transport
    from XTA.native_detector_handles import install_detector_handles

    class WorkerIO:
        def __init__(self, incoming, outgoing):
            self.get = incoming.get
            self.put = outgoing.put

        @staticmethod
        def materialize(task, persistent):
            return install_detector_handles(task, persistent, {})

        def installed(self, _task_id):
            pass

        def compute_done(self, _task_id):
            pass

    monkeypatch.setattr(tta_worker_transport, 'DetectorActorPort', WorkerIO)


@pytest.fixture
def primed_worker(monkeypatch, diagnostics, in_process_worker_io):
    telemetry, events = diagnostics
    replacements = dict(configure_pipeline_modes=None, initialize_runtime_observability=telemetry,
        set_retina_mask_processor=None, set_gpu_worker_fused_preflight_specs=None,
        set_angle_variant_gpu_fastpath=None, ensure_yolo_ready_for_predict=None,
        validate_yolo_model_input_channels=None, require_channel_aware_yolo_preprocess_patch=None,
        ensure_gpu_retina_proto_union_predictor_patch=True, cpu_retina_masks_enabled=False,
        _arm_worker_gpu_union_retirement_manager=None, _init_worker_gpu_render_engine=None,
        d1_owner_pipeline_enabled=False, _shutdown_d1_worker_pipeline=None,
        shutdown_radial_owners=None, _shutdown_resident_trt_pipeline_cache=None,
        _shutdown_gpu_union_retirement_manager=None, flush_runtime_task_trace=None,
        _worker_gpu_render_engine=None, _worker_initialized_torch=None,
        _resident_trt_pipeline_retirement_ready=0, _release_resident_trt_pipeline_cache=0,
        _direct_predict_applicable=True)
    for name, value in replacements.items():
        monkeypatch.setattr(workers, name, mock.Mock(return_value=value))
    monkeypatch.setattr(inference, 'ensure_yolo_ready_for_predict', mock.Mock())
    monkeypatch.setattr(inference, 'validate_yolo_model_input_channels', mock.Mock())
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,7')

    class Backend:
        pass

    class Predictor:
        pass

    class Model:
        def __init__(self, fail_once=False):
            self.model = 'model.engine'
            self.predictor = None
            self.calls = self.iterations = 0
            self.references = []
            self.fail_once = fail_once

        def predict(self, **kwargs):
            assert kwargs['stream'] is True
            self.calls += 1
            events.append(('backend_load', dict(call=self.calls)))
            self.predictor = Predictor()
            self.predictor.model = None
            self.references.append(weakref.ref(self.predictor))
            if self.fail_once and self.calls == 1:
                raise RuntimeError('transient backend load failure')
            self.predictor.model = Backend()
            self.references.append(weakref.ref(self.predictor.model))
            def stream():
                self.iterations += 1
                raise AssertionError('startup forced model inference')
                yield
            return stream()

    def run(model, *, tasks=(), init=None, path='model.engine', outgoing=None):
        incoming = queue.Queue()
        for task in tasks:
            incoming.put(task)
        incoming.put(None)
        monkeypatch.setattr(workers, 'load_ultralytics_model', mock.Mock(return_value=model))
        config = dict(imgsz=16, conf=.5, batch=1, quantize=16, cpu_workers=1, task='segment')
        config.update(init or {})
        with redirect_stdout(io.StringIO()):
            workers._gpu_inference_worker_main(1, path, config, incoming,
                                               outgoing if outgoing is not None else queue.Queue())
        return incoming

    return Model, run


@pytest.mark.parametrize('external_policy', [False, True])
def test_backend_prime_precedes_ready_and_first_task_reuses_backend_then_retires_all_aliases(
        monkeypatch, diagnostics, primed_worker, external_policy):
    _telemetry, events = diagnostics
    Model, run = primed_worker
    model = Model()

    def first_task(model_owner, cfg, task):
        assert model.calls == 1
        assert task['_startup_first_task'] is True
        assert inference._ensure_predictor_for_direct_predict(model_owner, cfg) is model.predictor
        assert model.calls == 1
        return {}

    def renderer(_device):
        assert model.calls == 1 and model.predictor.model is not None
        events.append(('renderer_initialized', {}))

    class Outgoing(queue.Queue):
        def put(self, value, *args, **kwargs):
            if value['type'] == 'ready':
                assert model.calls == 1 and model.predictor.model is not None
                events.append(('ready_sent', {}))
            if value['type'] == 'inference_assets_released':
                assert value['ok'] and value['stats']['released']
                assert all(reference() is None for reference in model.references)
            return super().put(value, *args, **kwargs)

    monkeypatch.setattr(workers, 'run_prediction_volume_in_worker', first_task)
    monkeypatch.setattr(workers, '_init_worker_gpu_render_engine', renderer)
    init = {}
    if external_policy:
        from XTA import tta_augmentation
        init['augmentation_settings'] = SimpleNamespace(enabled=True)
        def initialize_policy(*args, **kwargs):
            assert model.calls == 0
            events.append(('policy_initialized', {}))
        monkeypatch.setattr(tta_augmentation, 'worker_policy', initialize_policy)
    outgoing = Outgoing()
    run(model, tasks=[dict(task_id=1, slice_count=1),
        dict(task_id=-1, task_type='control', op='release_inference_assets', inference_drained=True)],
        outgoing=outgoing, init=init)
    assert model.calls == 1 and model.iterations == 0
    names = [name for name, _ in events]
    assert names.index('backend_load') < names.index('renderer_initialized') < names.index('ready_sent')
    if external_policy:
        assert names.index('policy_initialized') < names.index('backend_load')
    receipt = next(fields for name, fields in events if name == 'worker_backend_prime_done')
    assert receipt == dict(device='cuda:1', primed=True, declined=False)
    assert all(reference() is None for reference in model.references)


def test_declined_prime_drops_only_incomplete_predictor_and_first_task_can_retry(
        monkeypatch, diagnostics, primed_worker):
    _telemetry, events = diagnostics
    Model, run = primed_worker
    model = Model(fail_once=True)

    def first_task(model_owner, cfg, task):
        assert model.calls == 1 and model.predictor is None
        assert all(reference() is None for reference in model.references)
        assert inference._ensure_predictor_for_direct_predict(model_owner, cfg) is model.predictor
        assert model.calls == 2 and model.predictor.model is not None
        return {}

    monkeypatch.setattr(workers, 'run_prediction_volume_in_worker', first_task)
    run(model, tasks=[dict(task_id=1, slice_count=1)])
    assert model.calls == 2 and model.iterations == 0
    receipt = next(fields for name, fields in events if name == 'worker_backend_prime_done')
    assert receipt == dict(device='cuda:1', primed=False, declined=True)


def test_declined_prime_preserves_complete_backend_for_retryable_validation(
        monkeypatch, diagnostics, primed_worker):
    _telemetry, events = diagnostics
    Model, run = primed_worker
    model = Model()
    monkeypatch.setattr(inference, 'validate_yolo_model_input_channels',
                        mock.Mock(side_effect=[OSError('metadata temporarily unavailable'), None]))

    def first_task(model_owner, cfg, task):
        assert model.calls == 1 and model.predictor.model is not None
        assert inference._ensure_predictor_for_direct_predict(model_owner, cfg) is model.predictor
        assert model.calls == 1
        return {}

    monkeypatch.setattr(workers, 'run_prediction_volume_in_worker', first_task)
    run(model, tasks=[dict(task_id=1, slice_count=1)])
    assert model.calls == 1 and model.iterations == 0
    receipt = next(fields for name, fields in events if name == 'worker_backend_prime_done')
    assert receipt == dict(device='cuda:1', primed=False, declined=True)


@pytest.mark.parametrize('mode', ['pytorch', 'semantic', 'direct_disabled'])
def test_backend_prime_skips_routes_outside_existing_direct_engine_contract(
        monkeypatch, diagnostics, primed_worker, mode):
    _telemetry, events = diagnostics
    Model, run = primed_worker
    model = Model()
    init = {}
    path = 'model.pt' if mode == 'pytorch' else 'model.engine'
    if mode == 'semantic':
        init['task'] = 'semantic'
    elif mode == 'direct_disabled':
        monkeypatch.setattr(workers, '_direct_predict_applicable', lambda cfg: False)
    run(model, init=init, path=path)
    assert model.calls == 0 and model.predictor is None
    assert 'worker_ready' in [name for name, _ in events]
    assert 'worker_backend_prime_done' not in [name for name, _ in events]


def test_backend_prime_channel_mismatch_is_fatal_before_ready(monkeypatch, diagnostics, primed_worker):
    telemetry, events = diagnostics
    Model, run = primed_worker
    failure = inference.ModelInputChannelMismatchError('engine expects different channels')
    monkeypatch.setattr(inference, 'validate_yolo_model_input_channels', mock.Mock(side_effect=failure))
    outgoing = queue.Queue()
    incoming = run(Model(), tasks=[dict(task_id=1, slice_count=1)], outgoing=outgoing)
    assert outgoing.get()['type'] == 'fatal'
    assert incoming.get()['task_id'] == 1
    assert 'worker_ready' not in [name for name, _ in events]
    assert telemetry.snapshot()['counters']['startup.worker.backend_prime.errors'] == 1


@pytest.mark.parametrize('fail_model', [False, True])
def test_worker_ready_is_traced_before_send_and_never_sent_after_failed_initialization(
        diagnostics, in_process_worker_io, fail_model):
    telemetry, events = diagnostics
    incoming = queue.Queue()
    incoming.put(None)

    class Outgoing(queue.Queue):
        def put(self, value, *args, **kwargs):
            events.append(('message', dict(type=value['type'])))
            return super().put(value, *args, **kwargs)

    outgoing = Outgoing()
    init = dict(imgsz=16, conf=.5, batch=1, quantize=16, cpu_workers=1, task='semantic')
    replacements = dict(configure_pipeline_modes=None, initialize_runtime_observability=telemetry,
        set_retina_mask_processor=None, set_gpu_worker_fused_preflight_specs=None,
        set_angle_variant_gpu_fastpath=None, ensure_yolo_ready_for_predict=None,
        validate_yolo_model_input_channels=None, require_channel_aware_yolo_preprocess_patch=None,
        _arm_worker_gpu_union_retirement_manager=None, _init_worker_gpu_render_engine=None,
        d1_owner_pipeline_enabled=False, _shutdown_d1_worker_pipeline=None,
        shutdown_radial_owners=None, _shutdown_resident_trt_pipeline_cache=None,
        _shutdown_gpu_union_retirement_manager=None, flush_runtime_task_trace=None)
    with ExitStack() as stack, redirect_stdout(io.StringIO()):
        stack.enter_context(mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,7'}))
        for name, value in replacements.items():
            stack.enter_context(mock.patch.object(workers, name, return_value=value))
        load = stack.enter_context(mock.patch.object(workers, 'load_ultralytics_model',
            side_effect=RuntimeError('model unavailable') if fail_model else None, return_value=object()))
        workers._gpu_inference_worker_main(1, 'model.engine', init, incoming, outgoing)
    load.assert_called_once()
    names = [name for name, _ in events]
    assert events[0] == ('worker_startup_begin', dict(device='cuda:1'))
    if fail_model:
        assert 'worker_ready' not in names
        assert events[-2][0] == 'worker_startup_failed'
        assert events[-2][1]['failed'] is True
        assert outgoing.get()['type'] == 'fatal'
        assert telemetry.snapshot()['counters']['startup.worker.model_wrapper.errors'] == 1
    else:
        assert names.index('worker_ready') < names.index('message')
        assert outgoing.get()['type'] == 'ready'
        assert telemetry.snapshot()['phases']['startup.worker.model_wrapper']['calls'] == 1


@pytest.mark.parametrize('identity', ['known', 'uninitialized', 'missing', 'unsupported'])
def test_worker_ready_uuid_uses_initialized_local_device_and_keeps_logical_identity(
        monkeypatch, diagnostics, primed_worker, identity):
    _telemetry, events = diagnostics
    Model, run = primed_worker
    expected = 'GPU-actual-device-uuid' if identity == 'known' else None

    def properties(index):
        assert index == 0
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'
        if identity == 'unsupported':
            raise RuntimeError('UUID unavailable on this Torch build')
        return SimpleNamespace(uuid=expected) if identity == 'known' else SimpleNamespace()

    cuda = SimpleNamespace(get_device_properties=mock.Mock(side_effect=properties))
    monkeypatch.setattr(workers, '_worker_initialized_torch',
                        lambda: None if identity == 'uninitialized' else SimpleNamespace(cuda=cuda))
    outgoing = queue.Queue()
    run(Model(), outgoing=outgoing)
    event = next(fields for name, fields in events if name == 'worker_ready')
    ready = outgoing.get_nowait()
    assert event['device'] == ready['device'] == 'cuda:1'
    assert event['cuda_device_uuid'] == ready['cuda_device_uuid'] == expected
    assert ready['type'] == 'ready' and ready['gpu_index'] == 1
    if identity == 'uninitialized':
        cuda.get_device_properties.assert_not_called()
    else:
        cuda.get_device_properties.assert_called_once_with(0)


@pytest.mark.parametrize('uuid', ['GPU-actual-device-uuid', None, 123])
def test_parent_ready_identity_gauge_uses_registered_processes_and_preserves_all_devices(
        tmp_path, diagnostics, uuid):
    from XTA import tta_scheduler
    from tests.native_detector_test_support import DetectorRun

    telemetry, _events = diagnostics
    with DetectorRun(tmp_path) as run, mock.patch.object(tta_scheduler, 'runtime_telemetry',
                                                        return_value=telemetry):
        for index in (2, 7):
            run.worker(index, pid=1000 + index)
        run.start_transport()
        for index in (2, 7):
            with mock.patch.object(tta_scheduler.time, 'monotonic_ns', return_value=100 + index):
                run.state.gpu_result_queue.put(dict(type='ready', gpu_index=index, pid=1000 + index,
                                                    cuda_device_uuid=uuid))
                run.drain_until(lambda: index in run.state.gpu_worker_ready_details_by_id)
        gauge = telemetry.snapshot()['gauges']['gpu.device_identity']
        assert gauge == dict(parent_pid=os.getpid(), sample_monotonic_ns=107, source='managed_worker_ready',
            devices={f'cuda:{index}': dict(worker_pid=1000 + index,
                cuda_device_uuid=uuid if isinstance(uuid, str) else None) for index in (2, 7)})
        assert run.state.gpu_worker_results_collected == 0


@pytest.mark.parametrize('mutation,registered', [
    (dict(gpu_index=9), True), (dict(gpu_index=True), True),
    (dict(pid=999), True), (dict(pid=None), True), (dict(pid=True), True), ({}, False),
])
def test_parent_ready_identity_cannot_authorize_unmanaged_or_mismatched_worker(
        tmp_path, diagnostics, mutation, registered):
    from XTA import tta_scheduler
    from tests.native_detector_test_support import DetectorRun

    telemetry, _events = diagnostics
    with DetectorRun(tmp_path) as run, mock.patch.object(tta_scheduler, 'runtime_telemetry',
                                                        return_value=telemetry):
        if registered:
            run.worker(2, pid=1002)
        run.start_transport()
        message = dict(type='ready', gpu_index=2, pid=1002, cuda_device_uuid='GPU-foreign')
        message.update(mutation)
        run.state.gpu_result_queue.put(message)
        assert run.state.scheduler_wake.wait(3)
        with pytest.raises(RuntimeError, match='unregistered worker identity'):
            run.scheduler.drain_process_inference_results()
        assert 'gpu.device_identity' not in telemetry.snapshot()['gauges']
        assert run.state.gpu_worker_ready_details_by_id == {}


@pytest.mark.parametrize('fail_upload', [False, True])
def test_first_source_preparation_reports_existing_host_timers_and_failed_upload(tmp_path, diagnostics, fail_upload):
    _telemetry, events = diagnostics
    from XTA.backprojection import _ResidentTensorRTRingFatalError
    task, _ = raster_helpers.WorkerRasterPlanProvenanceTests._task(tmp_path, with_image_sink=False)
    task.update(_startup_first_task=True, _startup_device='cuda:2')
    failure = _ResidentTensorRTRingFatalError('source fence failed')
    engine = SimpleNamespace(ensure_volume=mock.Mock(return_value='resident',
        side_effect=failure if fail_upload else None),
        _source_residency_timings=dict(allocation_host_seconds=.25, upload_host_seconds=.75))
    source = SimpleNamespace(close=mock.Mock())
    cfg = inference.PredictConfig(imgsz=2, conf=.5, device='cuda:0', quantize=16, batch=1)

    class ArrayBackings:
        def memmap(self, *args, **kwargs):
            return np.zeros(kwargs['shape'], np.uint8)
        def __getattr__(self, name):
            return getattr(np, name)

    with (mock.patch.object(workers, '_worker_gpu_render_engine', return_value=engine),
          mock.patch.object(workers, 'GpuRenderedYoloSource', return_value=source),
          mock.patch.object(workers, 'np', ArrayBackings()),
          mock.patch.object(workers, 'predict_source_and_accumulate', return_value={})):
        if fail_upload:
            with pytest.raises(_ResidentTensorRTRingFatalError) as caught:
                workers.run_prediction_volume_in_worker(object(), cfg, task)
            assert caught.value is failure
        else:
            workers.run_prediction_volume_in_worker(object(), cfg, task)
            task['_startup_first_task'] = False
            task['result_mask_path'] = tmp_path / 'later.mask.u8.dat'
            workers.run_prediction_volume_in_worker(object(), cfg, task)
    selected = [(name, fields) for name, fields in events if name.startswith('worker_source_prepare_')]
    assert [name for name, _ in selected] == ['worker_source_prepare_start', 'worker_source_prepare_done']
    fields = selected[-1][1]
    assert fields['failed'] == fail_upload
    assert fields['device'] == 'cuda:2'
    assert fields['host_seconds'] >= 0.
    if not fail_upload:
        assert fields['allocation_host_seconds'] == .25
        assert fields['upload_host_seconds'] == .75
        assert fields['render_path'] == 'gpu_resident'
        assert source.close.call_count == 2
