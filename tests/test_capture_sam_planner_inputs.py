"""CPU-only checks for exact pre-SAM diagnostic capture and its failure boundary."""
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import sys
import weakref

import numpy as np
import pytest

from tools import capture_sam_planner_inputs as tool
from XTA import config, geometry, pipeline
from XTA.interpolation import RawBBoxMaskStore
from XTA.unification.runtime import CompiledPhysicalViews


def _view(name='azimuthal_transverse'):
    return geometry.ViewInfo(name=name, num_slices=3, src_h=5, src_w=11,
        pad_mode='pad', family='azimuthal', full_t=7, full_h=9, full_w=13,
        sampling_policy='coverage', sampling_certificate='exact-fixture')


def _variant(view):
    return geometry.expand_views_into_tta_variants([view], [0])[0]


def _prepare_kwargs(view, array, scores=None):
    return dict(model_name='best', view=view, union_mm=array, confmap_mm=scores,
        confidence_owner=None, confidence_retired_callback=None, confmap_path=None,
        min_conf=0., min_radius=0., slice_workers=1, precleaned_slice_cleanup=True,
        hole_fill_done_on_device=True, slice_meta=None, sam_context=None,
        preinterpolation_layer_already_published=False)


@pytest.mark.parametrize('value', [0, 1, 255])
def test_native_binary_roundtrip_preserves_literal_bytes_and_geometry(tmp_path, value):
    view = _variant(_view())
    array = np.zeros((3, 5, 11), np.uint8)
    array[0, 0, 0] = value
    array[2, 4, 10] = value  # Non-byte-aligned width and disjoint original frames.
    original = array.copy()
    capture = tool.PlannerInputCapture(tmp_path, [view.name], {'fixture': True})
    with pytest.raises(tool._CaptureFinished):
        capture.prepare(**_prepare_kwargs(view, array))
    manifest = json.loads((tmp_path / capture.captured[view.name]).read_text())
    assert manifest['view'] == json.loads(json.dumps(asdict(view)))
    assert manifest['literal_bytes_sha256'] == hashlib.sha256(original.tobytes()).hexdigest()
    assert manifest['working_canvas_kind'] == 'native_view_grid'
    store = RawBBoxMaskStore.open((tmp_path / capture.captured[view.name]).parent / manifest['store'])
    try:
        restored = np.stack([(store.decode_slice(i) != 0).astype(manifest['dtype'])
            * manifest['nonzero_value'] for i in range(3)])
    finally:
        store.close()
    np.testing.assert_array_equal(restored, original)
    np.testing.assert_array_equal(array, original)


def test_mixed_values_leave_no_scope_success_marker(tmp_path):
    view = _variant(_view())
    data = np.zeros((3, 5, 11), np.uint8)
    data[0, 1, 1], data[2, 1, 1] = 1, 255
    capture = tool.PlannerInputCapture(tmp_path, [view.name], {})
    with pytest.raises(ValueError, match='mixed'):
        capture.prepare(**_prepare_kwargs(view, data))
    assert not capture.captured and not list(tmp_path.rglob('manifest.json'))


def test_transferred_confidence_retirement_preserves_unrelated_file(tmp_path):
    view = _variant(_view())
    confidence_path = tmp_path / 'shared-confidence.dat'
    unrelated = tmp_path / 'unrelated.dat'
    unrelated.write_bytes(b'keep')
    scores = np.memmap(confidence_path, mode='w+', dtype=np.uint8, shape=(3, 5, 11))
    scores[:] = 17
    capture = tool.PlannerInputCapture(tmp_path, ['radial_other__tta_a0'], {})
    retired = []
    kwargs = _prepare_kwargs(view, np.zeros(scores.shape, np.uint8))
    kwargs.update(confidence_owner=[scores], confmap_path=unrelated,
        confidence_retired_callback=lambda *args: retired.append(args))
    del scores
    result = capture.prepare(**kwargs)
    gc.collect()
    assert kwargs['confidence_owner'] == []
    assert result.native_support_mm is result.final_view_volume_mm is None
    assert unrelated.read_bytes() == b'keep' and confidence_path.exists()
    assert retired == [('best', view.name, 165)]


def _arguments(tmp_path):
    source, model = tmp_path / 'source.mkv', tmp_path / 'best.engine'
    source.write_bytes(b'source')
    model.write_bytes(b'model')
    sam = tmp_path / 'sam'
    sam.mkdir()
    return ['--input', str(source), '--model', f'gpu:{model}', f'sam:{sam}',
        '--output', str(tmp_path / 'original-output'), '--device', '0', '--batch', 'gpu:1',
        '--channel_format', 'gray', '--angle', '0', '--enable_azimuthal', 'transverse',
        '--min_conf', '0', '--min_radius', '0', '--save', 'nrrd', 'summary',
        '--interpolation_backend', 'sam', '--interpolation_distance', '30',
        '--extrapolation_distance', '30']


def _fake_pipeline(arguments, *, mode='capture', capture_count=None):
    original = config.build_argparser().parse_args(arguments)
    settings = config.resolve_interpolation_settings(original,
        config.resolve_backend_models(original.model), config.resolve_backend_devices(original.device))
    physical = (_view(), replace(_view('radial_transverse_patch_u0_h0'), family='radial'))
    fake = SimpleNamespace(**{name: getattr(pipeline, name) for name in (
        'build_argparser', 'resolve_interpolation_settings', 'prepare_view_volume_after_fullframe',
        'compile_physical_views', 'expand_views_into_tta_variants', 'choose_scratch_dir',
        'assemble_final_union_after_view_union', '_finalize_physical_view_volume_group',
        'build_tta_run_manifest', '_publish_complete_tta_manifest', 'NrrdLayerSink',
        '_cleanup_tta_selected_run_scratch')})
    fake.compile_physical_views = lambda **kwargs: CompiledPhysicalViews(physical, (), (), ())
    fake.seen = []

    def run():
        args = fake.build_argparser().parse_args()
        for key, value in vars(original).items():
            if key not in {'output', 'temp', 'capture_component_replay'}:
                assert getattr(args, key) == value
        resolved = fake.resolve_interpolation_settings(args, None, None)
        assert asdict(resolved) == asdict(settings)
        assert not resolved.sam_enabled
        selected = fake.compile_physical_views(augmentation_angles=[0])
        variants = fake.expand_views_into_tta_variants(selected.views, [0])
        assert all(v.sampling_certificate == 'exact-fixture' for v in variants)
        if mode == 'forbidden-sam':
            from XTA.sam_integration import SamInterpolationContext
            SamInterpolationContext()
        if mode == 'zero':
            return
        for index, view in enumerate(variants):
            if capture_count is not None and index >= capture_count:
                raise OSError('inference failed')
            fake.seen.append(view.name)
            fake.prepare_view_volume_after_fullframe(**_prepare_kwargs(view,
                np.full((3, 5, 11), index % 2, np.uint8)))
    fake.main = run
    return fake, [_variant(view).name for view in physical]


def test_selection_suppresses_sam_and_restores_patches(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, targets = _fake_pipeline(arguments)
    original_prepare = fake.prepare_view_volume_after_fullframe
    report = tool.run_capture(tmp_path / 'diagnostic', targets[1:], arguments, pipeline=fake)
    assert report['status'] == 'capture_complete' and fake.seen == targets[1:]
    assert report['scientific_tta_complete'] is False and report['sam_executed'] is False
    assert fake.prepare_view_volume_after_fullframe is original_prepare
    assert not list((tmp_path / 'diagnostic').rglob('nrrd'))
    assert not (tmp_path / 'original-output').exists()


@pytest.mark.parametrize('mode,count,status', [('zero', None, 'capture_incomplete'),
    ('capture', 1, 'capture_failed'), ('forbidden-sam', None, 'capture_failed')])
def test_zero_partial_and_forbidden_sam_cannot_claim_complete(tmp_path, monkeypatch, mode, count, status):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, targets = _fake_pipeline(arguments, mode=mode, capture_count=count)
    report = tool.run_capture(tmp_path / 'diagnostic', targets, arguments, pipeline=fake)
    assert report['status'] == status and report['missing_views']
    assert report['scientific_tta_complete'] is False


def test_unknown_targets_fail_before_fake_inference(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, _ = _fake_pipeline(arguments)
    report = tool.run_capture(tmp_path / 'diagnostic', ['radial_missing__tta_a0'], arguments, pipeline=fake)
    assert report['status'] == 'capture_failed' and not fake.seen
    assert 'Unknown exact runtime' in report['error']


def test_output_must_be_fresh_and_disjoint(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, targets = _fake_pipeline(arguments)
    with pytest.raises(ValueError, match='fresh and disjoint'):
        tool.run_capture(tmp_path / 'original-output' / 'nested', targets, arguments, pipeline=fake)
    assert not fake.seen


def test_concurrent_capture_failure_is_joined_before_failed_marker(tmp_path, monkeypatch):
    from XTA import interpolation
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, targets = _fake_pipeline(arguments)
    original_writer = interpolation.write_raw_bbox_mask_store
    joined = []

    def write(array, path, **kwargs):
        if 'radial_' in kwargs['desc']:
            raise OSError('capture write failed')
        return original_writer(array, path, **kwargs)

    def run():
        selected = fake.compile_physical_views(augmentation_angles=[0])
        views = fake.expand_views_into_tta_variants(selected.views, [0])
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(fake.prepare_view_volume_after_fullframe,
                    **_prepare_kwargs(view, np.zeros((3, 5, 11), np.uint8))) for view in views]
                for future in futures:
                    future.result()
        finally:
            joined.append(True)
    fake.main = run
    monkeypatch.setattr(interpolation, 'write_raw_bbox_mask_store', write)
    report = tool.run_capture(tmp_path / 'diagnostic', targets, arguments, pipeline=fake)
    assert joined == [True] and report['status'] == 'capture_failed'
    assert 'capture write failed' in report['error']
    assert len(report['captures']) == 1


def test_cleanup_failure_and_changed_input_cannot_publish_complete(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, targets = _fake_pipeline(arguments)
    original_main = fake.main
    original_cleanup = fake._cleanup_tta_selected_run_scratch
    teardown = []

    def run():
        args = fake.build_argparser().parse_args()
        fake.choose_scratch_dir(args.temp, Path(args.output), 'fixture')
        try:
            original_main()
        finally:
            teardown.append(True)
    fake.main = run
    fake._cleanup_tta_selected_run_scratch = lambda **kwargs: (_ for _ in ()).throw(OSError('cleanup failure'))
    report = tool.run_capture(tmp_path / 'diagnostic', targets, arguments, pipeline=fake)
    assert teardown == [True] and not report['missing_views']
    assert report['status'] == 'capture_failed' and 'cleanup failure' in report['error']
    assert 'cleanup failure' in (tmp_path / 'diagnostic' / report['failure_traceback']).read_text()
    fake._cleanup_tta_selected_run_scratch = original_cleanup

    def changed():
        try:
            original_main()
        finally:
            (tmp_path / 'source.mkv').write_bytes(b'changed source bytes')
    fake.main = changed
    report = tool.run_capture(tmp_path / 'changed-input', targets, arguments, pipeline=fake)
    assert report['status'] == 'capture_failed' and 'Source identity changed' in report['error']
    assert 'Source identity revalidation failure' in (
        tmp_path / 'changed-input' / report['failure_traceback']).read_text()


def test_original_reconciliation_policy_source_is_retained(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path) + ['--reconciliation',
        str(tool.ROOT / 'XTA/examples/external_reconciliation/union.py')]
    fake, targets = _fake_pipeline(arguments)
    report = tool.run_capture(tmp_path / 'diagnostic', targets, arguments, pipeline=fake)
    assert report['status'] == 'capture_complete'
    copied = tmp_path / 'diagnostic' / 'reconciliation-policy.py'
    assert copied.read_bytes() == Path(arguments[-1]).read_bytes()
    assert report['provenance']['original_arguments']['save'] == ['nrrd', 'summary']
    assert report['provenance']['original_arguments']['reconciliation'] == arguments[-1]


def test_postcapture_confidence_failure_cannot_lose_race_to_completion(tmp_path, monkeypatch):
    from XTA import runtime
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, targets = _fake_pipeline(arguments)
    closing, release = threading.Event(), threading.Event()
    original_close = runtime.close_memmap_array_without_flush

    def close(scores, **kwargs):
        if scores is not None:
            closing.set()
            assert release.wait(5)
            raise OSError('confidence retirement failed after capture')
        return original_close(scores, **kwargs)

    def run():
        selected = fake.compile_physical_views(augmentation_angles=[0])
        views = fake.expand_views_into_tta_variants(selected.views, [0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(fake.prepare_view_volume_after_fullframe, **_prepare_kwargs(
                views[0], np.zeros((3, 5, 11), np.uint8), np.zeros((3, 5, 11), np.uint8)))
            assert closing.wait(5)
            last = pool.submit(fake.prepare_view_volume_after_fullframe,
                **_prepare_kwargs(views[1], np.zeros((3, 5, 11), np.uint8)))
            try:
                last.result()
            finally:
                release.set()
    fake.main = run
    monkeypatch.setattr(runtime, 'close_memmap_array_without_flush', close)
    report = tool.run_capture(tmp_path / 'diagnostic', targets, arguments, pipeline=fake)
    assert not report['missing_views'] and report['status'] == 'capture_failed'
    assert 'confidence retirement failed' in report['error']


def test_capture_follows_real_detector_hole_cleanup(tmp_path, monkeypatch):
    from XTA import workspace
    monkeypatch.setattr(workspace, '_YOLO_HOLE_FILL_ACTIVE', True)
    from XTA.assembly import prepare_detector_view_cpu
    view = _variant(_view())
    array = np.zeros((3, 5, 11), np.uint8)
    array[:, 1:4, 3:8] = 1
    array[:, 2, 4:7] = 0
    original = array.copy()
    expected = array.copy()
    prepare_detector_view_cpu(expected, None, view, 0., 0., workers=1,
        precleaned_slice_cleanup=False, hole_fill_done_on_device=False)
    assert np.count_nonzero(expected) > np.count_nonzero(original)
    capture = tool.PlannerInputCapture(tmp_path, [view.name], {})
    kwargs = _prepare_kwargs(view, array)
    kwargs.update(precleaned_slice_cleanup=False, hole_fill_done_on_device=False)
    with pytest.raises(tool._CaptureFinished):
        capture.prepare(**kwargs)
    record = json.loads((tmp_path / capture.captured[view.name]).read_text())
    assert record['literal_bytes_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()


def test_real_admitted_mapped_owner_retires_before_capture_completion(tmp_path, monkeypatch):
    from tests.test_view_prepare_input_retirement import _task
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    arguments = _arguments(tmp_path)
    fake, targets = _fake_pipeline(arguments)

    def run():
        args = fake.build_argparser().parse_args()
        scratch = fake.choose_scratch_dir(args.temp, Path(args.output), 'fixture')
        selected = fake.compile_physical_views(augmentation_angles=[0])
        view = fake.expand_views_into_tta_variants(selected.views, [0])[0]
        task, _ = _task(scratch)
        task.view = view
        task.prepare = fake.prepare_view_volume_after_fullframe
        task.interpolation_backend, task.interpolation_distance = 'sam', 30
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(task)
            del task
            future.result()
    fake.main = run
    report = tool.run_capture(tmp_path / 'diagnostic', targets[:1], arguments, pipeline=fake)
    assert report['status'] == 'capture_complete' and not report['error']
    assert not list((tmp_path / 'diagnostic' / 'runtime-temp').rglob('original.u8.dat'))


@pytest.mark.parametrize('mode', ['capture', 'sam_disabled', 'sdf_disabled'])
def test_capture_reaches_real_pipeline_gpu_task_inventory_without_sam(tmp_path, monkeypatch, request, mode):
    from XTA import assembly
    source_shape = assembly.final_source_output_shape()
    request.addfinalizer(lambda: assembly.set_final_source_output_shape(source_shape))

    class StartupReached(RuntimeError):
        pass

    class IdleWorker:
        pid = 123456789
        exitcode = 0
        def start(self):
            pass
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    monkeypatch.setenv('YOLO_TTA_CPU_INPUT_PREFETCH', '1')
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    monkeypatch.setenv('YOLO_TTA_GPU_WORKER_CPU', '1')
    monkeypatch.setattr(pipeline, '_pin_cuda_visible_device_token', lambda index: '-1')
    monkeypatch.setattr(pipeline, 'plan_gpu_feeder_core_reservations', lambda tokens: [[]])
    monkeypatch.setattr(pipeline, 'plan_gpu_worker_affinity', lambda *args, **kwargs: [None])
    monkeypatch.setattr(pipeline.mp, 'get_context', lambda kind: SimpleNamespace(
        Queue=queue.Queue, Process=lambda **kwargs: IdleWorker()))
    monkeypatch.setattr(pipeline, 'ffprobe_info', lambda path: dict(
        width=32, height=32, num_frames=16, fps=10.))
    monkeypatch.setattr(pipeline, '_cube_t_axis_resize_backend', lambda: 'slab')
    cubes = []
    original_cube = pipeline.LazyProcessingCube

    class ObservedCube(original_cube):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cubes.append((self, weakref.ref(self.source)))
    monkeypatch.setattr(pipeline, 'LazyProcessingCube', ObservedCube)

    def decode(**kwargs):
        volume = np.memmap(kwargs['out_dat'], mode='w+', dtype=np.uint8,
            shape=(kwargs['num_frames'], kwargs['height'], kwargs['width']))
        volume[:] = 0
        return volume
    monkeypatch.setattr(pipeline, 'decode_video_to_memmap_gray8', decode)
    monkeypatch.setattr(pipeline, 'decode_video_to_memmap_gray8_streaming', decode)
    reached = []

    def prefetch_boundary():
        frame = sys._getframe(1)
        if frame.f_code is pipeline._main_impl.__code__:
            state = frame.f_locals
            assert state['sam_context'] is None
            assert isinstance(state['volume_rgb'], ObservedCube)
            settings = state['interpolation_settings']
            assert not settings.sam_enabled
            assert settings.backend == ('sdf' if mode == 'sdf_disabled' else 'sam')
            assert state['args'].interpolation_distance == settings.extrapolation_distance == (
                30 if mode == 'capture' else 0)
            tasks = list(state['gpu_worker_tasks_by_id'].values())
            assert tasks and {task['view'].name for task in tasks} == {'azimuthal_transverse__tta_a0'}
            reached.append(len(tasks))
            raise StartupReached('real task inventory reached before GPU dispatch')
        return False
    monkeypatch.setattr(pipeline, 'memfd_workspace_enabled', prefetch_boundary)
    arguments = _arguments(tmp_path) + ['--imgsz', '64']
    if mode == 'capture':
        report = tool.run_capture(tmp_path / 'diagnostic', ['azimuthal_transverse__tta_a0'], arguments)
        assert reached, report['error']
        assert report['error'] == 'StartupReached: real task inventory reached before GPU dispatch'
        assert report['status'] == 'capture_failed' and report['scientific_tta_complete'] is False
        failure = (tmp_path / 'diagnostic' / report['failure_traceback']).read_text()
        assert '_main_impl' in failure and 'prefetch_boundary' in failure
    else:
        arguments += ['--interpolation_backend', 'sdf' if mode == 'sdf_disabled' else 'sam',
            '--interpolation_distance', '0', '--extrapolation_distance', '0']
        monkeypatch.setattr(sys, 'argv', ['tta', *arguments])
        with pytest.raises(StartupReached, match='real task inventory reached'):
            pipeline.main()
        assert reached
        gc.collect()
        pipeline._cleanup_tta_selected_run_scratch(temp_dir=tmp_path / 'original-output' / 'temp',
            out_dir=tmp_path / 'original-output')
    assert not pipeline._PIPELINE_RUN_LOCK.locked()
    gc.collect()
    assert len(cubes) == 1
    cube, source = cubes[0]
    assert cube._stop.is_set() and cube.source is None
    cube._watcher.join(timeout=1)
    assert not cube._watcher.is_alive() and source() is None
    assert not list(tmp_path.rglob('input_volume.gray8.dat'))
