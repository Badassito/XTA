"""Independent exactness, budget and lifetime oracles for SAM Azimuthal batching."""
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from XTA import geometry
from XTA import inference
from XTA import sam_azimuthal_batch as batch
from XTA.cuda_backend import _GpuWorkerRenderEngine as scalar
from XTA.sam_gpu_rendering import crop_inverse, SamGpuRenderingUnavailable
from XTA.sam_transverse_cache_rendering import native_crop_bbox
from tests.test_sam_azimuthal_native_roi import engine_for, views_for


@pytest.fixture(scope='module')
def device():
    value = os.environ.get('XTA_SAM_BATCH_ORACLE_DEVICE', 'cpu')
    if value != 'cpu':
        assert os.environ.get('XTA_RUN_SAM_BATCH_CUDA_ORACLE') == '1'
        assert (Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK').is_file()
        assert torch.cuda.is_available()
    return value


def stream_context(engine):
    return torch.cuda.stream(engine._stream) if engine.device.type == 'cuda' else nullcontext()


def finish(engine):
    if engine.device.type == 'cuda':
        engine._stream.synchronize()


def bit_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if actual.dtype == torch.float32:
        assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))
    else:
        assert torch.equal(actual, expected)


def source_for(native_t, pattern):
    if pattern == 'random':
        return np.random.default_rng(157546).integers(0, 256, (native_t, 11, 17), np.uint8)
    tt, yy, xx = np.indices((native_t, 11, 17))
    return ((tt % 2)*31 + ((yy + xx) % 2)*128).astype(np.uint8)


@pytest.mark.parametrize('size', [1, 2, 3, 4])
@pytest.mark.parametrize('native_t,logical_t', [(7, 7), (5, 9), (9, 5)])
@pytest.mark.parametrize('pattern', ['random', 'half_ties'])
def test_native_batch_is_bit_exact_to_full_plane_and_scalar_roi(size, native_t, logical_t, pattern, device):
    source = source_for(native_t, pattern)
    before = source.copy()
    engine = engine_for(source, logical_t, device)
    source_pointer = engine._volume_gpu.data_ptr()
    source_version = engine._volume_gpu._version
    frames = (0, 2, 4, 7)[:size]
    try:
        with torch.inference_mode(), stream_context(engine):
            for original in views_for(source, logical_t):
                stack = geometry.azimuthal_stack_length(original)
                for rows in (stack, stack + 3):
                    view = replace(original, src_h=rows, src_w=original.src_w+1,
                                   center_x=original.center_x+.5, center_y=original.center_y-.5)
                    bbox = (max(0, rows-4), 1, rows, min(view.src_w, 6))
                    y0, x0, y1, x1 = bbox
                    wanted = torch.stack([scalar._render_azimuthal_native_resident_torch(engine, view, frame)[y0:y1, x0:x1]
                                          for frame in frames])
                    existing = torch.stack([engine._render_azimuthal_native_crop_resident_torch(view, frame, bbox)
                                            for frame in frames])
                    bit_equal(existing, wanted)
                    for block in (1, 3, 512):
                        actual = batch.project_native_batch(engine, view, frames, bbox, block)
                        bit_equal(actual, wanted)
                        bit_equal(actual.round().clamp(0, 255).to(torch.uint8),
                                  wanted.round().clamp(0, 255).to(torch.uint8))
            assert engine._volume_gpu.data_ptr() == source_pointer
            assert engine._volume_gpu._version == source_version
    finally:
        finish(engine)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize('size', [1, 2, 3, 4])
@pytest.mark.parametrize('shape', [(31, 47), (1, 257), (257, 1)])
def test_batched_grid_sample_preserves_global_coordinates_borders_and_identity(size, shape, device):
    engine = engine_for(np.zeros((3, 11, 17), np.uint8), 3, device)
    yy, xx = np.indices(shape)
    planes = np.stack([((yy*73+xx*151+index*17) % 256).astype(np.uint8) for index in range(size)])
    native = torch.as_tensor(planes, device=engine.device)
    matrices = [np.array([[1., 0., 0.], [0., 1., 0.]], np.float32),
        np.array([[1.000001, 0., .000001], [0., .999999, -.000001]], np.float32)]
    matrices += [np.array([[.73, .04, x], [-.03, .81, y]], np.float32)
                 for x, y in [(-.500001, -.499999), (.5, .5), (shape[1]-.5, shape[0]-.5), (-100., 300.)]]
    try:
        with torch.inference_mode(), stream_context(engine):
            for matrix in matrices:
                for output_shape in (shape, (1 if shape[0] == 1 else 5, 1 if shape[1] == 1 else 7)):
                    expected = torch.stack([scalar.warp_native_uint8_frame(engine, item, matrix, *output_shape)
                                            for item in native])
                    actual = batch.warp_native_batch(engine, native, matrix, *output_shape)
                    bit_equal(actual, expected)
    finally:
        finish(engine)
    np.testing.assert_array_equal(native.cpu().numpy(), planes)


@pytest.mark.parametrize('size', [1, 2, 3, 4])
@pytest.mark.parametrize('native_t,logical_t', [(7, 7), (5, 9), (9, 5)])
def test_native_roi_quantization_then_batch_warp_equals_full_plane_pipeline(size, native_t, logical_t, device):
    source = source_for(native_t, 'half_ties')
    engine = engine_for(source, logical_t, device)
    frames = (0, 2, 4, 7)[:size]
    try:
        with torch.inference_mode(), stream_context(engine):
            for original in views_for(source, logical_t):
                view = replace(original, src_h=41, src_w=53)
                for matrix in (np.array([[.73, .04, 17.5], [-.03, .81, 11.5]], np.float32),
                               np.array([[-.77, .13, 54.3], [.07, .81, -3.5]], np.float32)):
                    height, width = 7, 9
                    bbox = native_crop_bbox((0, 0, height, width), matrix, (view.src_h, view.src_w))
                    assert bbox is not None
                    y0, x0, y1, x1 = bbox
                    patch = batch.project_native_batch(engine, view, frames, bbox, 3)
                    canvas = torch.zeros((size, view.src_h, view.src_w), dtype=torch.uint8, device=engine.device)
                    canvas[:, y0:y1, x0:x1] = patch.round().clamp(0, 255).to(torch.uint8)
                    actual = batch.warp_native_batch(engine, canvas, matrix, height, width)
                    wanted = []
                    for frame in frames:
                        full = scalar._render_azimuthal_native_resident_torch(engine, view, frame)
                        wanted.append(scalar.warp_native_uint8_frame(engine,
                            full.round().clamp(0, 255).to(torch.uint8), matrix, height, width))
                    bit_equal(actual, torch.stack(wanted))
    finally:
        finish(engine)


def metadata_case(scratch=256*1024**2):
    small = np.zeros((7, 11, 17), np.uint8)
    view = replace(views_for(small, 7)[0], full_t=128, full_h=2911, full_w=4300,
                   src_h=2911, src_w=4300, center_x=2149.5, center_y=1455., roi_radius=1455.)
    renderer = SimpleNamespace(view=view, source=SimpleNamespace(shape=(128, 2911, 4300)),
        logical_t=128, scratch_bytes=scratch, source_t_window=(0, 128))
    matrix = np.array([[1., 0., 1400.], [0., 1., 800.]], np.float32)
    crop = dict(output_origin_yx=(0, 0), output_height=128, output_width=128)
    return renderer, view, matrix, crop


def independent_ledger(renderer, view, frames, inverse, crop):
    matrix = crop_inverse(inverse, crop['output_origin_yx'])
    height, width = crop['output_height'], crop['output_width']
    bbox = native_crop_bbox((0, 0, height, width), matrix, (view.src_h, view.src_w))
    h, c = bbox[2]-bbox[0], bbox[3]-bbox[1]
    count, a, p = len(frames), view.src_h*view.src_w, height*width
    k = 4*view.src_w+20*renderer.logical_t+32*view.src_h
    fixed_projection = k+256*count*view.src_w+count*a+32*count*h*c
    warp = k+6*count*a+16*count*p+32*p
    host = 2*count*p+128*(view.src_h+renderer.logical_t+view.src_w+c)+64*count
    device = min(renderer.scratch_bytes, 256*a+96*p)
    rows = 2*h  # This fixture deliberately folds 128 native rows into 2911 rows.
    q = min(512, rows, max(0, device-fixed_projection)//(256*count*c))
    return dict(bbox=bbox, warp=warp, host=host, projection=fixed_projection+256*count*c*q,
                q=q, device=device, rows=rows, minimum_projection=fixed_projection+256*count*c)


def test_real_dataset_geometry_chooses_b3_when_b4_exceeds_256mib_without_allocations():
    renderer, view, matrix, crop = metadata_case()
    for size in (2, 3, 4):
        frames = tuple(range(size))
        expected = independent_ledger(renderer, view, frames, matrix, crop)
        actual = batch.plan_batch(renderer, view, frames, matrix, retained_cache_bytes=0, **crop)
        assert actual.eligible is (size < 4)
        if actual.eligible:
            assert actual.bbox_yx == expected['bbox']
            assert actual.row_block == expected['q']
            assert actual.projection_bytes == expected['projection']
            assert actual.warp_bytes == expected['warp']
            assert actual.host_bytes == expected['host']
            assert actual.device_cap_bytes == expected['device']
            assert actual.selected_rows == expected['rows']
    chosen = batch.choose_batch_prefix(renderer, view, iter(range(1000000)), matrix, retained_cache_bytes=0, **crop)
    assert chosen.eligible and chosen.batch_size == 3


@pytest.mark.parametrize('size', [2, 3])
def test_real_geometry_scratch_boundary_accepts_exact_limit_and_rejects_one_byte_less(size):
    renderer, view, matrix, crop = metadata_case()
    frames = tuple(range(size))
    terms = independent_ledger(renderer, view, frames, matrix, crop)
    limit = max(terms['warp'], terms['host'], terms['minimum_projection'])
    for delta in (-1, 0, 1):
        renderer.scratch_bytes = limit+delta
        plan = batch.plan_batch(renderer, view, frames, matrix, retained_cache_bytes=0, **crop)
        assert plan.eligible is (delta >= 0)
        if plan.eligible:
            assert max(plan.projection_bytes, plan.warp_bytes) <= plan.device_cap_bytes
            assert plan.host_bytes <= plan.host_cap_bytes


def test_prefix_reads_no_more_than_four_requests_and_private_source_guard_precedes_work():
    renderer, view, matrix, crop = metadata_case()
    reads = []
    def requests():
        for index in range(4):
            reads.append(index)
            yield index
        raise AssertionError('prefix read beyond its bounded inventory')
    assert batch.choose_batch_prefix(renderer, view, requests(), matrix, retained_cache_bytes=0, **crop).batch_size == 3
    assert reads == [0, 1, 2, 3]
    renderer.source_t_window = (0, 64)
    assert not batch.plan_batch(renderer, view, (0, 1), matrix, retained_cache_bytes=0, **crop).eligible
    renderer.source_t_window = (0, 128)
    assert not batch.plan_batch(renderer, replace(view), (0, 1), matrix, retained_cache_bytes=0, **crop).eligible


@pytest.mark.parametrize('frames', [(0,), (0, 0), (0, 1, 2, 3, 4), (-1, 0)])
def test_unsupported_cohorts_are_declined_without_mutating_owner(frames):
    renderer, view, matrix, crop = metadata_case()
    before = dict(vars(renderer))
    result = batch.plan_batch(renderer, view, frames, matrix, **crop)
    assert not result.eligible
    assert vars(renderer) == before


@pytest.fixture
def lifecycle(monkeypatch):
    """Real CPU arithmetic with inert CUDA contexts and observable copy boundaries."""
    monkeypatch.setattr(inference, '_AFFINE_GRID_CACHE', OrderedDict())
    source = source_for(7, 'half_ties')
    engine = engine_for(source, 7)
    engine._stream = object()
    engine._fused_azimuthal_taps = OrderedDict()
    view = replace(views_for(source, 7)[0], src_h=41, src_w=53)
    crop = dict(output_origin_yx=(0, 0), output_height=7, output_width=9)
    inverse = np.array([[.73, .04, 17.5], [-.03, .81, 11.5]], np.float32)
    state = SimpleNamespace(events=[], cancelled=False, cancel_after_warp=False,
        fail_stage=None, failure=RuntimeError('original controlled failure'))

    def check():
        state.events.append('check')
        if state.cancelled:
            raise state.failure

    class TorchProxy:
        cuda = SimpleNamespace(device=lambda *_args: nullcontext(), stream=lambda *_args: nullcontext())
        def __getattr__(self, name):
            return getattr(torch, name)

    owner = object()
    from XTA.cuda_backend import gpu_render_reserve_bytes
    renderer = SimpleNamespace(view=view, source=source, logical_t=7, engine=engine,
        torch=TorchProxy(), source_t_window=(0, 7), source_frames=frozenset(range(view.num_slices)),
        source_resident_bytes=source.nbytes, source_planned_bytes=source.nbytes,
        required_gpu=source.nbytes+256*view.src_h*view.src_w+96*63+gpu_render_reserve_bytes(),
        device_index=0, scratch_bytes=256*1024**2, lease=owner, _render_failed=False,
        context=SimpleNamespace(_check_image_lifetime=check))

    def start():
        state.events.append('start')
        return engine

    def scalar_render(candidate, frame, affine, **output):
        state.events.append(('scalar', frame))
        check()
        if renderer.source_t_window != (0, source.shape[0]) and (
                candidate is not renderer.view or frame not in renderer.source_frames):
            raise SamGpuRenderingUnavailable('SAM partial source does not cover this physical frame demand')
        full = scalar._render_azimuthal_native_resident_torch(engine, candidate, frame)
        result = scalar.warp_native_uint8_frame(engine, full.round().clamp(0, 255).to(torch.uint8),
            crop_inverse(affine, output['output_origin_yx']), output['output_height'], output['output_width'])
        return result.numpy()

    renderer._start, renderer.render = start, scalar_render
    renderer.close = lambda: pytest.fail('borrowed helper retired the caller owner')
    engine.clear_native_plane_cache = lambda: state.events.append('clear')
    project, warp = batch.project_native_batch, batch.warp_native_batch

    def project_checked(*args, **kwargs):
        state.events.append('project')
        if state.fail_stage == 'project':
            raise state.failure
        return project(*args, **kwargs)

    class Pixels:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, position):
            return Pixels(self.value[position])
        def cpu(self):
            state.events.append('copy')
            if state.fail_stage == 'copy':
                raise state.failure
            return self.value

    def warp_checked(*args, **kwargs):
        state.events.append('warp')
        if state.fail_stage == 'warp':
            raise state.failure
        result = warp(*args, **kwargs)
        if state.cancel_after_warp:
            state.cancelled = True
        return Pixels(result)

    monkeypatch.setattr(batch, 'project_native_batch', project_checked)
    monkeypatch.setattr(batch, 'warp_native_batch', warp_checked)
    return SimpleNamespace(renderer=renderer, view=view, crop=crop, inverse=inverse,
        state=state, owner=owner, engine=engine)


def call(case, frames=(0, 2, 4, 7), consume=None):
    return batch.render_compatible_batch(case.renderer, case.view, frames, case.inverse,
        consume=consume, **case.crop)


def test_successful_batch_consumes_every_frame_once_in_order_and_keeps_source_owner(lifecycle):
    case = lifecycle
    saved = case.engine._volume_gpu.data_ptr()
    consumed = []
    result = call(case, consume=lambda position, image: consumed.append((position, image.copy())))
    assert not result.images
    assert [position for position, _ in consumed] == [0, 1, 2, 3]
    assert result.receipt['completed_images'] == result.receipt['requested_batch_size'] == 4
    assert result.receipt['effective_batch_size'] == 4 and result.receipt['batch_path_used'] is True
    assert case.state.events.count('copy') == 4
    assert case.state.events.count('clear') == 1
    assert case.renderer.lease is case.owner and case.renderer.engine is case.engine
    assert case.engine._volume_gpu.data_ptr() == saved and not case.renderer._render_failed
    for (_, image), frame in zip(consumed, (0, 2, 4, 7)):
        expected = case.renderer.render(case.view, frame, case.inverse, **case.crop)
        np.testing.assert_array_equal(image, expected)


def test_collecting_success_returns_all_charged_views_that_outlive_source_retirement(lifecycle):
    case = lifecycle
    first = call(case)
    assert len(first.images) == first.receipt['completed_images'] == 4
    assert first.receipt['planned_host_scratch_peak_bytes'] >= 2*4*7*9
    saved = [image.copy() for image in first.images]
    # A single returned view pins the fully charged BHW host allocation. It is
    # independent of the source/engine, but must be dropped before a new chunk.
    owner = first.images[0]
    while isinstance(getattr(owner, 'base', None), np.ndarray):
        owner = owner.base
    assert isinstance(owner.base, torch.Tensor) and owner.base.numel() == 4*7*9
    case.engine._volume_gpu = None  # Simulate subsequent caller-owned retirement.
    case.engine._fold_cache.clear()
    case.engine._native_t_map_cache.clear()
    assert case.state.events.count('copy') == 1
    for before, after in zip(saved, first.images):
        np.testing.assert_array_equal(after, before)


def test_unsupported_collecting_mode_is_an_explicit_prelaunch_decline_never_empty_success(lifecycle):
    case = lifecycle
    with pytest.raises(batch.BatchDeclined) as caught:
        call(case, (0,))
    assert caught.value.reason == 'batch_size'
    assert 'start' not in case.state.events and 'project' not in case.state.events
    assert not any(isinstance(item, tuple) and item[0] == 'scalar' for item in case.state.events)


def test_singleton_sink_fallback_preserves_exactly_one_requested_frame(lifecycle):
    case = lifecycle
    consumed = []
    result = call(case, (4,), consume=lambda position, image: consumed.append((position, image.copy())))
    assert not result.images and len(consumed) == 1 and consumed[0][0] == 0
    assert result.receipt['batch_path_used'] is False
    assert result.receipt['requested_batch_size'] == result.receipt['effective_batch_size'] == 1
    assert [item for item in case.state.events if isinstance(item, tuple)] == [('scalar', 4)]
    assert 'start' not in case.state.events and 'project' not in case.state.events


@pytest.mark.parametrize('stage', ['project', 'warp', 'copy'])
def test_batch_runtime_failure_is_not_retried_and_owner_is_retained_for_retirement(lifecycle, stage):
    case = lifecycle
    case.state.fail_stage = stage
    with pytest.raises(SamGpuRenderingUnavailable) as caught:
        call(case)
    assert caught.value.__cause__ is case.state.failure
    assert case.renderer._render_failed
    assert case.renderer.lease is case.owner and case.renderer.engine is case.engine
    assert not any(isinstance(item, tuple) and item[0] == 'scalar' for item in case.state.events)


@pytest.mark.parametrize('position', [0, 1, 3])
def test_private_sink_failure_has_original_identity_and_no_complete_result(lifecycle, position):
    case = lifecycle
    visits = []
    failure = RuntimeError('private cache write failed')
    def consume(index, image):
        visits.append(index)
        if index == position:
            raise failure
    with pytest.raises(RuntimeError) as caught:
        call(case, consume=consume)
    assert caught.value is failure
    assert visits == list(range(position+1))
    assert case.state.events.count('copy') == position+1
    assert case.renderer.lease is case.owner and case.renderer.engine is case.engine
    assert not any(isinstance(item, tuple) and item[0] == 'scalar' for item in case.state.events)


@pytest.mark.parametrize('point', ['before_start', 'after_enqueue', 'after_first_sink'])
def test_cancellation_never_returns_complete_or_releases_the_source(lifecycle, point):
    case = lifecycle
    visited = []
    if point == 'before_start':
        case.state.cancelled = True
    if point == 'after_enqueue':
        case.state.cancel_after_warp = True
    def consume(position, image):
        visited.append(position)
        if point == 'after_first_sink':
            case.state.cancelled = True
    with pytest.raises(RuntimeError) as caught:
        call(case, consume=consume)
    assert caught.value is case.state.failure
    assert visited == ([0] if point == 'after_first_sink' else [])
    assert case.state.events.count('copy') == len(visited)
    assert case.renderer.lease is case.owner and case.renderer.engine is case.engine
    if point == 'before_start':
        assert 'start' not in case.state.events
    else:
        assert case.renderer._render_failed


def test_unmeasured_cache_never_admits_even_when_cold_scenario_would_fit():
    renderer, view, matrix, crop = metadata_case()
    unknown = batch.plan_batch(renderer, view, (0, 1, 2), matrix, **crop)
    cold = batch.plan_batch(renderer, view, (0, 1, 2), matrix, retained_cache_bytes=0, **crop)
    assert not unknown.eligible and unknown.reason == 'cache_measurement_required'
    assert unknown.retained_cache_bytes is None and cold.eligible


def test_actual_dataset_retained_cache_boundary_is_exact_to_one_byte():
    renderer, view, matrix, crop = metadata_case()
    frames = (0, 1, 2)
    terms = independent_ledger(renderer, view, frames, matrix, crop)
    allowance = 256*view.src_h*view.src_w+96*crop['output_height']*crop['output_width']
    assert terms['warp'] > terms['minimum_projection']
    cache_limit = allowance-terms['warp']
    for delta in (-1, 0, 1):
        retained = cache_limit+delta
        plan = batch.plan_batch(renderer, view, frames, matrix, retained_cache_bytes=retained, **crop)
        assert plan.eligible is (delta <= 0)
        assert plan.retained_cache_bytes == retained
        assert plan.existing_gpu_allowance_bytes == allowance
        assert plan.total_device_bound_bytes == retained+max(plan.projection_bytes, plan.warp_bytes)
        if plan.eligible:
            assert plan.total_device_bound_bytes <= allowance


def test_cache_snapshot_counts_full_storages_once_and_excludes_resident_source_aliases(lifecycle):
    case = lifecycle
    source = case.engine._volume_gpu
    case.engine._volume_flat = source.view(-1)
    case.engine._native_plane_cache['source-alias'] = source[0]
    backing = torch.empty(2048, dtype=torch.uint8)
    distinct = torch.empty(128, dtype=torch.float32)
    case.engine._fold_cache['same-owner'] = (backing[:1], backing[1:2])
    case.engine._native_t_map_cache['same-owner'] = backing[4:7]
    inference._AFFINE_GRID_CACHE[('cpu', 'same-owner')] = backing[7:9]
    inference._AFFINE_GRID_CACHE[('cpu', 'distinct-owner')] = distinct[:1]
    inference._AFFINE_GRID_CACHE[('cuda:7', 'other-device')] = torch.empty(3, device='meta')
    measured = batch.retained_cache_snapshot(case.engine)
    assert measured['retained_cache_bytes'] == 2048+128*4
    assert measured['retained_cache_storage_count'] == 2
    assert case.engine._volume_gpu is source


def test_global_affine_cache_uses_resolved_tensor_device_not_requested_key(monkeypatch):
    """Pure fake storage metadata: never queries or allocates a CUDA device."""
    class Storage:
        def __init__(self, pointer, size):
            self.pointer, self.size = pointer, size
        def data_ptr(self):
            return self.pointer
        def nbytes(self):
            return self.size

    class Tensor:
        def __init__(self, device, storage):
            self.device, self.storage = device, storage
        def untyped_storage(self):
            return self.storage

    shared = Storage(100, 128)
    source = Tensor('cuda:0', Storage(1, 1000))
    engine = SimpleNamespace(torch=SimpleNamespace(Tensor=Tensor), device='cuda:0',
                             _volume_gpu=source, _volume_flat=Tensor('cuda:0', source.storage))
    monkeypatch.setattr(inference, '_AFFINE_GRID_CACHE', OrderedDict([
        (('cuda', 'unresolved-request'), Tensor('cuda:0', shared)),
        (('cuda', 'same-owner-alias'), Tensor('cuda:0', shared)),
        (('cuda:9', 'key-differs-from-actual'), Tensor('cuda:0', Storage(300, 64))),
        (('cuda:0', 'actual-other-device'), Tensor('cuda:1', Storage(200, 512))),
        (('cpu', 'actual-cpu'), Tensor('cpu', Storage(400, 1024))),
    ]))
    measured = batch.retained_cache_snapshot(engine)
    assert measured['device'] == 'cuda:0'
    assert measured['retained_cache_bytes'] == 128+64
    assert measured['retained_cache_storage_count'] == 2


@pytest.mark.parametrize('bad_cache', ['opaque', 'source_alias'])
def test_unprovable_cache_or_source_alias_is_an_explicit_decline(lifecycle, bad_cache):
    case = lifecycle
    if bad_cache == 'opaque':
        case.engine._fused_azimuthal_taps['unproven'] = object()
    else:
        case.engine._volume_flat = case.engine._volume_gpu.clone().view(-1)
    with pytest.raises(batch.BatchDeclined):
        batch.retained_cache_snapshot(case.engine)


@pytest.mark.parametrize('sink', [False, True])
def test_post_start_cache_growth_declines_before_batch_work_without_failure_or_retry(lifecycle, sink):
    case = lifecycle
    original_start = case.renderer._start
    seen = []
    def start_with_growth():
        engine = original_start()
        # Half a megabyte is enough to exhaust this small owner's G. Views are
        # tiny, so counting tensor numel rather than backing storage would miss it.
        owner = torch.empty(600000, dtype=torch.uint8)
        engine._fold_cache['new-owner'] = (owner[:1], owner[1:2])
        return engine
    case.renderer._start = start_with_growth
    consume = (lambda index, image: seen.append(index)) if sink else None
    with pytest.raises(batch.BatchDeclined) as caught:
        call(case, consume=consume)
    assert caught.value.reason == 'scratch_or_cache_bound'
    assert caught.value.receipt['retained_cache_bytes'] == 600000
    assert caught.value.receipt['batch_path_used'] is False
    assert 'project' not in case.state.events and 'warp' not in case.state.events
    assert 'copy' not in case.state.events and not seen
    assert not case.renderer._render_failed
    assert case.renderer.lease is case.owner and case.renderer.engine is case.engine
    assert not any(isinstance(item, tuple) and item[0] == 'scalar' for item in case.state.events)


def test_cold_grid_and_fold_cache_growth_is_inside_the_new_work_bound(lifecycle):
    case = lifecycle
    before = batch.retained_cache_snapshot(case.engine)['retained_cache_bytes']
    result = call(case, consume=lambda *_args: None)
    after = batch.retained_cache_snapshot(case.engine)['retained_cache_bytes']
    h, w = case.view.src_h, case.view.src_w
    p = case.crop['output_height']*case.crop['output_width']
    independent_cold_bound = 4*w+20*case.renderer.logical_t+32*h+32*p
    assert 0 < after-before <= independent_cold_bound
    assert result.receipt['retained_cache_bytes'] == before
    assert result.receipt['total_device_bound_bytes'] <= result.receipt['existing_gpu_allowance_bytes']
