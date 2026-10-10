"""Full native Radial rasters keep exact pixels with bounded source-T uploads."""
from contextlib import nullcontext
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_backend, geometry
from XTA.config import resolve_tilted_view_groups
from XTA.cylindrical_geometry import shell_coordinates
from XTA.sam_gpu_rendering import (SamGpuCropRenderer, SamGpuRenderingUnavailable,
                                  radial_source_t_window, try_gpu_crop_renderer)


def _views(base, tilt=0, direction='vertical'):
    tilted = resolve_tilted_view_groups([f'{base}:{abs(tilt)}:both']) if tilt else ()
    views = geometry.get_view_infos(41, 47, 53, cartesian_views=(), tilt_groups=tilted,
        radial_views=(('tilted_'+base) if tilt else base,), radial_patch_size=8, radial_min_radius=1.)
    matching = [view for view in views if view.family == 'radial'
        and (not tilt or view.tilt_angle_deg == tilt and view.tilt_direction == direction)]
    arc = max(view.radial_patch_index for view in matching)
    band = max(view.radial_height_index for view in matching)
    return [view for view in matching if view.radial_patch_index in (0, arc)
            and view.radial_height_index in (0, band)]


def _frames(view):
    return tuple(sorted({0, view.num_slices//2, view.num_slices-1}))


def _engine(source, logical_t, *, window=None, device='cpu'):
    import torch
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    start, stop = window or (0, source.shape[0])
    engine.torch = torch
    engine.device = device
    engine._volume_gpu = torch.as_tensor(source[start:stop].copy(), device=device)
    engine._volume_flat = engine._volume_gpu.reshape(-1)
    engine._logical_t = logical_t
    engine._radial_source_t = (source.shape[0], start) if window else None
    engine._native_t_map_cache = {}
    return engine


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('tilt,direction', ((0, 'vertical'), (-45, 'vertical'), (45, 'vertical'),
                                          (-45, 'horizontal'), (45, 'horizontal')))
@pytest.mark.parametrize('native_t', (31, 41, 53))
def test_planned_window_contains_every_nested_native_tap(base, tilt, direction, native_t):
    for view in _views(base, tilt, direction):
        frames = _frames(view)
        start, stop = radial_source_t_window(view, (native_t, 47, 53), 41, frames)
        assert 0 <= start < stop <= native_t
        for frame in frames:
            tt, _yy, _xx, _valid = shell_coordinates(view, frame)
            outer = np.floor(tt).astype(np.int64)
            for offset in (0, 1):
                logical = np.clip(outer+offset, 0, 40)
                positions = (logical+.5)*(native_t/41.)-.5
                lower = np.clip(np.floor(positions).astype(np.int64), 0, native_t-1)
                upper = np.minimum(lower+1, native_t-1)
                assert int(lower.min()) >= start
                assert int(upper.max()) < stop


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('tilt,direction', ((0, 'vertical'), (-45, 'vertical'), (45, 'vertical'),
                                          (-45, 'horizontal'), (45, 'horizontal')))
@pytest.mark.parametrize('native_t', (31, 41, 53))
def test_torch_reference_full_and_window_sources_are_byte_exact(base, tilt, direction, native_t):
    source = np.random.default_rng(613).integers(0, 256, (native_t, 47, 53), np.uint8)
    full = _engine(source, 41)
    for view in _views(base, tilt, direction):
        frames = _frames(view)
        window = radial_source_t_window(view, source.shape, 41, frames)
        cropped = _engine(source, 41, window=window)
        for frame in frames:
            expected = full._render_radial_native_resident_torch(view, frame).numpy()
            actual = cropped._render_radial_native_resident_torch(view, frame).numpy()
            np.testing.assert_array_equal(actual, expected)


def test_window_keeps_global_t_map_and_ignores_invalid_endpoint_masks():
    source = np.zeros((53, 47, 53), np.uint8)
    full = _engine(source, 41)
    cropped = _engine(source, 41, window=(17, 39))
    for expected, actual in zip(full._native_t_indices(), cropped._native_t_indices()):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    view = replace(_views('transverse', 45)[0], src_h=53)
    # Both height endpoints can be invalid while interior pixels still sample T.
    window = radial_source_t_window(view, source.shape, 41, _frames(view))
    assert window == (0, 53)


def test_full_t_envelope_stops_scanning_remaining_radii():
    view = replace(_views('transverse')[0], src_h=41)
    with mock.patch('XTA.cylindrical_geometry.shell_coordinates', wraps=shell_coordinates) as coordinates:
        assert radial_source_t_window(view, (41, 47, 53), 41, _frames(view)) == (0, 41)
    assert coordinates.call_count == 1


def test_nested_logical_gray8_taps_are_not_direct_native_interpolation():
    outer = np.floor(1.9).astype(int)
    logical = np.array([outer, outer+1])
    lower = np.floor((logical+.5)*(8/5.)-.5).astype(int)
    upper = lower+1
    assert tuple(lower) == (1, 3) and tuple(upper) == (2, 4)
    direct = (1.9+.5)*(8/5.)-.5
    assert int(np.floor(direct)) == 3  # Would omit the logical lower voxel's native taps.


@pytest.mark.parametrize('change', ({'family': 'spherical'}, {'center_x': 1.25},
    {'tilt_angle_deg': 46., 'radial_tilted_source': True}, {'radial_arc_origin': float('inf')},
    {'radial_radii': (.5,)}, {'full_t': 5000}))
def test_unqualified_recipe_retains_complete_source(change):
    view = replace(_views('transverse')[0], **change)
    assert radial_source_t_window(view, (41, 47, 53), 41, (0,)) == (0, 41)
    assert radial_source_t_window(_views('transverse')[0], (5000, 47, 53), 41, (0,)) == (0, 5000)
    assert radial_source_t_window(_views('transverse')[0], (41, 47, 53), 41, ('0',)) == (0, 41)


def test_actual_job_height_band_and_late_arc_window_reduce_resident_bytes():
    shape = (2911, 3064, 3022)
    views = geometry.get_view_infos(*shape, cartesian_views=(), radial_views=('transverse', 'sagittal'),
                                    radial_patch_size=2048)
    transverse = next(view for view in views if view.name == 'radial_transverse_patch_u3_h0')
    sagittal = next(view for view in views if view.name == 'radial_sagittal_patch_u3_h0')
    a = radial_source_t_window(transverse, shape, shape[0], range(31))
    b = radial_source_t_window(sagittal, shape, shape[0], range(sagittal.num_slices-31, sagittal.num_slices))
    assert (a[1]-a[0])/shape[0] < .71
    assert (b[1]-b[0])/shape[0] < .26


def test_partial_source_frame_binding_precedes_gpu_work():
    source = np.zeros((41, 47, 53), np.uint8)
    view = _views('transverse')[0]
    context = SimpleNamespace(_check_image_lifetime=mock.Mock(), source_identity='frozen')
    renderer = SamGpuCropRenderer(context, view, source, 41, None, object(), 4096,
        source_t_window=(1, 17), source_frames=(0,))
    other = replace(view, radial_arc_origin=view.radial_arc_origin+1.)
    for descriptor, frame in ((view, 1), (other, 0)):
        with pytest.raises(SamGpuRenderingUnavailable, match='does not cover'):
            renderer.render(descriptor, frame, np.eye(2, 3), output_origin_yx=(0, 0),
                            output_height=4, output_width=4)
    engine = _engine(source, 41, window=(1, 17))
    ordinary = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    with pytest.raises(RuntimeError, match='another physical family'):
        engine._render_native_plane(ordinary, 0)


def test_admission_and_upload_charge_resident_window_and_keep_global_geometry(monkeypatch):
    import torch
    source = np.zeros((41, 47, 53), np.uint8)
    view = _views('transverse')[0]
    frames = _frames(view)
    context = SimpleNamespace(source_volume=source, detector_retirement_ready=True,
        _resource_local=SimpleNamespace(profile=object()), source_identity='immutable',
        _check_image_lifetime=mock.Mock())
    plan = SimpleNamespace(frame_crop_bounds={frame: (0, 0, 8, 8) for frame in frames})
    counters, gauges = {}, {}
    telemetry = SimpleNamespace(add=lambda name, value: counters.__setitem__(name, counters.get(name, 0)+value),
        gauge=lambda name, value: gauges.__setitem__(name, value))
    monkeypatch.setenv('YOLO_TTA_SAM_GPU_IMAGES', '1')
    monkeypatch.setattr('XTA.sam_resources.validate_live_sam_resource_profile',
        lambda _profile: {'base_non_cpu_allowance_bytes': 2**30})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cuda_backend, 'gpu_render_reserve_bytes', lambda: 1234)
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    renderer = try_gpu_crop_renderer(context, view, (view.num_slices, 8, 8), plan)
    assert renderer is not None and renderer.lease is None
    start, stop = renderer.source_t_window
    resident = (stop-start)*47*53
    assert resident < source.nbytes
    assert renderer.required_gpu == resident+256*8*8+96*8*8+1234
    engine = SimpleNamespace(ensure_volume_array=mock.Mock(return_value='resident'),
        ensure_volume=mock.Mock(side_effect=AssertionError('whole source was uploaded')),
        _native_t_map_cache={'old': object()}, _source_residency_timings={})
    monkeypatch.setattr(cuda_backend, '_GpuWorkerRenderEngine', lambda _device: engine)
    renderer.lease = SimpleNamespace(device_index=0)
    renderer.device_index = 0
    assert renderer._start() is engine
    uploaded = engine.ensure_volume_array.call_args.args[0]
    assert uploaded.shape == (stop-start, 47, 53) and np.shares_memory(uploaded, source)
    assert engine._radial_source_t == (41, start) and engine._logical_t == 41
    assert engine._azimuthal_texture_admitted is False
    assert engine._native_t_map_cache == {}
    assert counters['sam.gpu_images.source_upload_bytes'] == resident
    assert counters['sam.gpu_images.source_window_upload_bytes_avoided'] == source.nbytes-resident
    assert gauges['sam.gpu_images.source_window_headroom_bytes_saved'] == source.nbytes-resident
    identity = renderer.sampling_identity()
    assert identity['native_shape_tyx'] == list(source.shape)
    assert identity['planned_source_t_window'] == [start, stop]
    renderer.engine = None
    renderer.resident_t_window = (0, 41)
    renderer.source_resident_bytes = source.nbytes
    renderer.required_gpu += source.nbytes-resident
    assert renderer._start() is engine
    assert renderer.resident_t_window == renderer.source_t_window
    assert renderer.source_resident_bytes == resident
    assert renderer.required_gpu == resident+256*8*8+96*8*8+1234
    assert renderer.sampling_identity() == identity


def _handoff_pair(tmp_path, resident=(0, 7), planned=(2, 5), free=2**30):
    from tests.test_sam_gpu_rendering import ready_handoff_pair
    context, first, second, engine, lease = ready_handoff_pair(tmp_path, free=free)
    first.torch.__version__ = 'test'
    first.torch.version = SimpleNamespace(cuda='test')
    for renderer, window in ((first, resident), (second, planned)):
        renderer.source_t_window = renderer.resident_t_window = window
        renderer.source_planned_bytes = renderer.source_resident_bytes = (window[1]-window[0])*9*11
        renderer.required_gpu = renderer.source_planned_bytes+512
        renderer.source_frames = frozenset((0,))
    engine._radial_source_t = (7, resident[0]) if resident != (0, 7) else None
    first._uploaded_source_key = first._source_handoff_key()
    return context, first, second, engine, lease


def _close_handoff(context, *renderers):
    context.cancel('test complete')
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
        for renderer in renderers:
            renderer.close()
    context.close()


@pytest.mark.parametrize('resident,planned', (((0, 7), (2, 5)), ((1, 6), (2, 5))))
def test_resident_superset_preserves_planned_identity_and_counts_avoided_upload(tmp_path, resident, planned):
    context, first, second, engine, lease = _handoff_pair(tmp_path, resident, planned)
    identity, original_origin = second.sampling_identity(), engine._radial_source_t
    telemetry = SimpleNamespace(add=mock.Mock(), gauge=mock.Mock())
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch('XTA.runtime.runtime_telemetry', return_value=telemetry):
            assert first._try_handoff()
        assert second.engine is engine and second.lease is lease
        assert second.source_t_window == planned and second.source_frames == frozenset((0,))
        assert second.resident_t_window == resident
        assert second.source_resident_bytes == (resident[1]-resident[0])*9*11
        assert second.required_gpu == second.source_resident_bytes+512
        assert second.sampling_identity() == identity
        assert engine._radial_source_t == original_origin
        assert second._uploaded_source_key == second._source_handoff_key()
        assert second._start() is engine  # Adopted storage is not replaced by the smaller plan.
        telemetry.add.assert_any_call('sam.gpu_images.source_upload_bytes_saved', second.source_planned_bytes)
    finally:
        _close_handoff(context, first, second)


def test_superset_handoff_requires_all_projector_headroom(tmp_path):
    context, first, second, engine, _lease = _handoff_pair(tmp_path, free=511)
    identity = second.sampling_identity()
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            assert not first._try_handoff()  # Subtracting the larger old source would wrongly admit this.
            assert first.engine is engine and second.engine is None
            assert second.resident_t_window == second.source_t_window
            second.torch.cuda.mem_get_info.return_value = (512, 2**40)
            assert first._try_handoff()
        assert second.required_gpu == 7*9*11+512
        assert second.source_planned_bytes == 3*9*11
        assert second.sampling_identity() == identity
    finally:
        _close_handoff(context, first, second)


@pytest.mark.parametrize('planned,refusal', (((2, 7), 'extent'), ((0, 5), 'extent'),
    ((0, 7), 'full_source'), ((2, 5), 'cancel'), ((2, 5), 'not_ready'), ((2, 5), 'source')))
def test_handoff_refusals_leave_planned_and_actual_ownership_unchanged(tmp_path, planned, refusal):
    import threading
    context, first, second, engine, _lease = _handoff_pair(tmp_path, (1, 6), planned)
    if refusal == 'cancel':
        second._gpu_prefetch_cancel = threading.Event()
        second._gpu_prefetch_cancel.set()
    elif refusal == 'not_ready':
        second._gpu_wait_deadline = None
    elif refusal == 'source':
        second.source = second.source.copy()
    before = (second.resident_t_window, second.source_resident_bytes, second.required_gpu,
              second.source_t_window, second.source_frames)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            assert not first._try_handoff()
        assert first.engine is engine and second.engine is None
        assert before == (second.resident_t_window, second.source_resident_bytes, second.required_gpu,
                          second.source_t_window, second.source_frames)
    finally:
        _close_handoff(context, first, second)


def test_nested_handoffs_keep_actual_origin_and_each_frozen_plan(tmp_path):
    context, first, second, engine, _lease = _handoff_pair(tmp_path, (0, 7), (1, 6))
    third = SamGpuCropRenderer(context, None, context.source_volume, 7, None, first.torch, 2**20,
                              required_gpu=3*9*11+512)
    third.source_t_window = third.resident_t_window = (2, 5)
    third.source_planned_bytes = third.source_resident_bytes = 3*9*11
    third.source_frames = frozenset((0,))
    identity = third.sampling_identity()
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch.object(context, '_can_extend_gpu_image_burst', return_value=True), \
                mock.patch.object(context, '_runtime', SimpleNamespace(idle_image_handoff=lambda: nullcontext(True))):
            assert first._try_handoff()
            context._queue_gpu_image(third)
            assert second._try_handoff()
        assert third.engine is engine and third.resident_t_window == (0, 7)
        assert second.source_t_window == (1, 6) and third.source_t_window == (2, 5)
        assert third.required_gpu == 7*9*11+512
        assert third._image_burst_count == 3 and engine._radial_source_t is None
        assert third._uploaded_source_key == third._source_handoff_key()
        assert third.sampling_identity() == identity
    finally:
        _close_handoff(context, first, second, third)


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned resident-superset qualification')
def test_tiny_cuda_superset_handoff_keeps_planned_identity_and_actual_origin(tmp_path):
    import torch
    from tests.test_sam_view_image_cache import context_for
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file(), 'caller must hold GPU_LOCK for explicit CUDA qualification'
    assert torch.cuda.is_available()
    source = np.random.default_rng(631).integers(0, 256, (61, 47, 53), np.uint8)
    view = next(item for item in _views('transverse') if item.radial_height_index > 0)
    frames = _frames(view)
    planned = radial_source_t_window(view, source.shape, 41, frames)
    actual = (max(0, planned[0]-5), min(61, planned[1]+5))
    assert actual[0] > 0 and actual != planned
    context = context_for(tmp_path, source)
    lease = SimpleNamespace(device_index=0, release=mock.Mock())
    first = SamGpuCropRenderer(context, view, source, 41, lease, torch, 2**20,
        source_t_window=actual, source_frames=frames)
    second = SamGpuCropRenderer(context, view, source, 41, None, torch, 2**20,
        required_gpu=(planned[1]-planned[0])*47*53+2**20,
        source_t_window=planned, source_frames=frames)
    identity = second.sampling_identity()
    try:
        expected = first.render(view, frames[0], np.eye(2, 3, dtype=np.float32),
            output_origin_yx=(1, 2), output_height=3, output_width=4)
        engine = first.engine
        original_origin = engine._radial_source_t
        context._queue_gpu_image(first)
        context._finish_gpu_image_wait(first, granted=True)
        context._queue_gpu_image(second)
        assert first._try_handoff()
        received = second.render(view, frames[0], np.eye(2, 3, dtype=np.float32),
            output_origin_yx=(1, 2), output_height=3, output_width=4)
        np.testing.assert_array_equal(received, expected)
        assert second.engine is engine and engine._radial_source_t == original_origin
        assert second.source_t_window == planned and second.resident_t_window == actual
        assert second.source_resident_bytes == (actual[1]-actual[0])*47*53
        assert second.required_gpu == second.source_resident_bytes+2**20
        assert second.sampling_identity() == identity
        print({'superset_source_t_window': actual, 'planned_source_t_window': planned,
               'avoided_upload_bytes': second.source_planned_bytes,
               'actual_resident_bytes': second.source_resident_bytes, 'exact_crop_pixels': received.size})
    finally:
        _close_handoff(context, first, second)


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned Radial source window qualification')
def test_tiny_cuda_source_windows_match_full_scalar_columns_and_torch(monkeypatch):
    import torch
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file(), 'caller must hold GPU_LOCK for explicit CUDA qualification'
    assert torch.cuda.is_available()
    full = cuda_backend._GpuWorkerRenderEngine('cuda:0')
    cropped = cuda_backend._GpuWorkerRenderEngine('cuda:0')
    cases = partial_cases = crop_cases = 0
    try:
        for native_t in (31, 41, 53):
            source = np.random.default_rng(617+native_t).integers(0, 256, (native_t, 47, 53), np.uint8)
            assert full.ensure_volume_array(source, identity=f'full:{native_t}') == 'resident'
            full._logical_t = 41
            full._native_t_map_cache.clear()
            for base in ('transverse', 'sagittal', 'coronal'):
                for tilt, direction in ((0, 'vertical'), (-45, 'vertical'), (45, 'vertical'),
                                        (-45, 'horizontal'), (45, 'horizontal')):
                    for view in _views(base, tilt, direction):
                        frames = _frames(view)
                        start, stop = radial_source_t_window(view, source.shape, 41, frames)
                        assert cropped.ensure_volume_array(source[start:stop],
                            identity=f'window:{native_t}:{start}:{stop}') == 'resident'
                        cropped._radial_source_t = (native_t, start)
                        cropped._logical_t = 41
                        cropped._native_t_map_cache.clear()
                        partial_cases += stop-start < native_t
                        for frame in frames:
                            for columns in (False, True):
                                with mock.patch.object(cuda_backend, 'radial_column_geometry_for_shape', return_value=columns):
                                    with torch.cuda.stream(full._stream):
                                        expected_tensor = full._render_radial_native_resident_cuda(view, frame)
                                        expected = expected_tensor.cpu().numpy()
                                    with torch.cuda.stream(cropped._stream):
                                        actual_tensor = cropped._render_radial_native_resident_cuda(view, frame)
                                        actual = actual_tensor.cpu().numpy()
                                np.testing.assert_array_equal(actual, expected)
                                if columns:
                                    from XTA.sam_gpu_rendering import crop_inverse
                                    matrix = crop_inverse(np.array([[.83, .027, -.31], [-.019, .79, .23]], np.float32), (1, 2))
                                    with torch.cuda.stream(full._stream):
                                        expected_crop = full.warp_native_uint8_frame(expected_tensor.to(torch.uint8), matrix, 6, 7).cpu().numpy()
                                    with torch.cuda.stream(cropped._stream):
                                        actual_crop = cropped.warp_native_uint8_frame(actual_tensor.to(torch.uint8), matrix, 6, 7).cpu().numpy()
                                    np.testing.assert_array_equal(actual_crop, expected_crop)
                                    crop_cases += 1
                                cases += 1
                            with torch.cuda.stream(full._stream):
                                reference = full._render_radial_native_resident_torch(view, frame).cpu().numpy()
                            with torch.cuda.stream(cropped._stream):
                                actual = cropped._render_radial_native_resident_torch(view, frame).cpu().numpy()
                            np.testing.assert_array_equal(actual, reference)
                            cases += 1
        assert partial_cases > 30
        assert full._azimuthal_texture_ref is None and cropped._azimuthal_texture_ref is None
        print({'exact_full_window_native_rasters': cases, 'partial_source_windows': partial_cases,
               'exact_nonzero_origin_crops': crop_cases, 'maximum_source_upload_bytes': 53*47*53})
    finally:
        full.release_inference_assets()
        cropped.release_inference_assets()
        torch.cuda.empty_cache()
