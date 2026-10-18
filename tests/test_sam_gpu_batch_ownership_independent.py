"""Private-cache batch transaction oracles, using CPU pixels and fake renderers."""
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, sam_resources as resources
from XTA.sam_cyclic import build_cyclic_frame_addressing
from XTA.sam_gpu_rendering import SamGpuCropRenderer, SamGpuRenderingUnavailable, live_image_sampling
from tests.test_sam_gpu_rendering import FakeRenderer
from tests.test_sam_image_prefetch import account
from tests.test_sam_view_image_cache import context_for, crop_from, demand
from tests.test_sam_view_orientations import SOURCE_SHAPE


@pytest.fixture(autouse=True)
def cpu_geometry_only(monkeypatch):
    for flag in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT',
                 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', 'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT'):
        monkeypatch.setenv(flag, '0')


def pixels(frame, height, width, origin):
    y0, x0 = origin
    return ((frame * 17 + 3 * (y0 + np.arange(height))[:, None]
             + 5 * (x0 + np.arange(width))[None, :]) % 251).astype(np.uint8)


class PrefixRenderer(FakeRenderer):
    def __init__(self, context, *, size=2, behavior='success', index=0, before_prefix=None):
        super().__init__(context)
        self.size, self.behavior, self.index = size, behavior, index
        self.before_prefix = before_prefix
        self.prefixes, self.scalar_frames = [], []

    def render(self, view, frame, inverse, **crop):
        self.calls += 1
        self.scalar_frames.append(frame)
        return pixels(frame, crop['output_height'], crop['output_width'], crop['output_origin_yx'])

    def render_compatible_prefix(self, view, frames, inverse, *, consume, **crop):
        from XTA.sam_azimuthal_batch import BatchDeclined
        frames = tuple(frames)
        self.prefixes.append(frames)
        if self.before_prefix is not None:
            self.before_prefix(self, frames)
        if self.behavior == 'decline' or len(frames) < 2:
            raise BatchDeclined('controlled pre-batch refusal')
        count = min(self.size, len(frames))
        receipt = dict(completed_images=count, effective_batch_size=count,
                       batch_path_used=True, host_result_mode='sink')
        if self.behavior == 'empty':
            return receipt
        positions = list(range(count))
        if self.behavior == 'partial':
            positions.pop()
        elif self.behavior == 'duplicate':
            positions[1] = positions[0]
        elif self.behavior == 'out_of_order':
            positions[:2] = [1, 0]
        for position in positions:
            if self.behavior == 'cancel' and position == self.index:
                self.context.cancel(f'controlled batch cancellation {position}')
            image = pixels(frames[position], crop['output_height'], crop['output_width'],
                           crop['output_origin_yx'])
            if self.behavior == 'bad_dtype' and position == self.index:
                image = image.astype(np.float32)
            consume(position, image)
            if self.behavior == 'gpu_failure' and position == self.index:
                raise SamGpuRenderingUnavailable('controlled launched batch failure')
            if self.behavior == 'late_decline' and position == self.index:
                raise BatchDeclined('controlled decline after sink entry')
        if self.behavior == 'bad_receipt':
            receipt['completed_images'] = count - 1
        elif self.behavior == 'float_count':
            receipt['completed_images'] = receipt['effective_batch_size'] = float(count)
        elif self.behavior == 'false_batch_marker':
            receipt['batch_path_used'] = False
        elif self.behavior == 'wrong_host_mode':
            receipt['host_result_mode'] = 'collected'
        return receipt


def make_case(tmp_path, monkeypatch, **renderer_kwargs):
    monkeypatch.setenv('YOLO_TTA_SAM_IMAGE_CACHE_MAX_BYTES', '4096')
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '4096')
    source = np.zeros(SOURCE_SHAPE, np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(),
        azimuthal_views=('transverse',), azimuthal_azimuth_angles=(30.,))[0]
    shape = (view.num_slices, 8, 8)
    assert shape[0] >= 4
    renderer = PrefixRenderer(context, **renderer_kwargs)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *a, **k: renderer)
    return context, view, shape, renderer


def assert_unpublished(context):
    assert not context._caches and not context._cache_entries and not context._cache_owners
    assert not context._image_builds
    assert context.cache_logical_bytes == 0
    assert not list((context.temp_dir / 'sam_image_cache').glob('*.gray8.dat'))


def test_batch_enters_only_after_existing_image_credit_and_private_mapping_are_owned(tmp_path, monkeypatch):
    context, view, shape, renderer = make_case(tmp_path, monkeypatch)
    context._runtime = SimpleNamespace(
        release_source_cache=lambda ref: dict(status='retired', workers_finished=True, gray_mappings_retired=True),
        close=lambda: None, cancel=lambda reason: None, residency_released=True, dispatch_stats={})
    pool = account(64 * resources.GIB)
    box = (1, 2, 6, 7)
    try:
        with resources.admit_sam_parent_resources(pool, 4 * resources.GIB, 'batch-parent',
                base_allowance_bytes=4 * resources.GIB,
                image_phase_bytes=context.mandatory_image_phase_bytes(),
                headroom_probe=lambda: 128 * resources.GIB) as parent:
            charged = pool.in_use
            def admitted(_renderer, frames):
                assert frames[:2] == (0, 1)
                assert pool.in_use == charged and parent._lease.scope_holds == 1
                assert len(context._image_builds) == 1
                ticket = next(iter(context._image_builds.values()))
                assert ticket['payload_bytes'] == 50 and ticket['credit_bytes'] >= 50 + 4096
                assert ticket['private_path'].exists() and ticket['private_mapping_ref']() is not None
                assert ticket['parent_image_credit_release'] is not None
                assert not context._caches and not context._cache_entries
            renderer.before_prefix = admitted
            with context.resource_scope(parent):
                with context.image_cohort_provider(view, shape, demand(shape, {0: box, 1: box})) as ref:
                    assert live_image_sampling(ref)['backend'] == 'cuda'
                    for frame in (0, 1):
                        np.testing.assert_array_equal(crop_from(ref, frame, box), pixels(frame, 5, 5, box[:2]))
                    assert context.rendered_frames == 2 and context.rendered_pixels == 50
                    assert renderer.closed
                assert parent._lease.scope_holds == 0 and pool.in_use == charged
        assert pool.in_use == 0
    finally:
        context.close()


def test_adjacent_prefixes_and_scalar_tail_write_every_offset_and_count_once(tmp_path, monkeypatch):
    context, view, shape, renderer = make_case(tmp_path, monkeypatch, size=2)
    boxes = {0: (1, 1, 5, 6), 1: (1, 1, 5, 6), 2: (2, 2, 7, 6),
             3: (2, 2, 7, 6), 4: (1, 1, 5, 6)}
    try:
        ref = context.image_provider(view, shape, demand(shape, boxes))
        assert [prefix[:2] for prefix in renderer.prefixes if len(prefix) >= 2] == [(0, 1), (2, 3)]
        assert renderer.scalar_frames == [4]
        offset = 0
        for record in ref.frame_crops:
            frame, y0, x0, y1, x1, recorded_offset = record
            assert recorded_offset == offset
            np.testing.assert_array_equal(crop_from(ref, frame, boxes[frame]),
                pixels(frame, y1 - y0, x1 - x0, (y0, x0)))
            offset += (y1 - y0) * (x1 - x0)
        assert offset == ref.size_bytes == 100
        assert context.rendered_frames == 5 and context.rendered_pixels == 100
        assert renderer.closed and len(context._cache_entries) == 1
    finally:
        context.close()


@pytest.mark.parametrize('behavior', ['empty', 'partial', 'duplicate', 'out_of_order',
                                    'bad_receipt', 'bad_dtype', 'late_decline', 'float_count',
                                    'false_batch_marker', 'wrong_host_mode'])
def test_partial_or_invalid_sink_never_publishes_or_falls_through_to_scalar(tmp_path, monkeypatch, behavior):
    context, view, shape, renderer = make_case(tmp_path, monkeypatch, behavior=behavior, index=1)
    box = (1, 1, 6, 6)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            context.image_provider(view, shape, demand(shape, {0: box, 1: box, 2: box}))
        assert len(renderer.prefixes) == 1 and not renderer.scalar_frames
        assert renderer.closed
        assert_unpublished(context)
    finally:
        context.close()


@pytest.mark.parametrize('position', [0, 1, 2, 3])
def test_cancellation_at_each_sink_position_discards_the_whole_private_cache(tmp_path, monkeypatch, position):
    context, view, shape, renderer = make_case(tmp_path, monkeypatch,
        size=4, behavior='cancel', index=position)
    box = (1, 1, 6, 6)
    try:
        with pytest.raises(RuntimeError, match='controlled batch cancellation'):
            context.image_provider(view, shape, demand(shape, {frame: box for frame in range(4)}))
        assert len(renderer.prefixes) == 1 and not renderer.scalar_frames and renderer.closed
        assert_unpublished(context)
    finally:
        context.close()


def test_launched_failure_retires_before_whole_cpu_rebuild_without_mixed_pixels(tmp_path, monkeypatch):
    context, view, shape, renderer = make_case(tmp_path, monkeypatch, behavior='gpu_failure', index=0)
    box = (1, 1, 6, 6)
    original = context._image_provider
    def checked_build(*args, **kwargs):
        if renderer.prefixes:
            assert renderer.closed
            assert_unpublished(context)
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_image_provider', checked_build)
    try:
        ref = context.image_provider(view, shape, demand(shape, {0: box, 1: box, 2: box}))
        assert live_image_sampling(ref)['backend'] == 'cpu'
        for frame in (0, 1, 2):
            np.testing.assert_array_equal(crop_from(ref, frame, box), 0)
        assert len(context._cache_entries) == 1
        assert len(list((context.temp_dir / 'sam_image_cache').glob('*.gray8.dat'))) == 1
        assert len(renderer.prefixes) == 1 and not renderer.scalar_frames
    finally:
        context.close()


def test_pre_sink_decline_uses_scalar_without_skipping_work(tmp_path, monkeypatch):
    context, view, shape, renderer = make_case(tmp_path, monkeypatch, behavior='decline')
    box = (1, 1, 6, 6)
    try:
        ref = context.image_provider(view, shape, demand(shape, {frame: box for frame in range(3)}))
        assert renderer.scalar_frames == [0, 1, 2]
        for frame in range(3):
            np.testing.assert_array_equal(crop_from(ref, frame, box), pixels(frame, 5, 5, box[:2]))
        assert context.rendered_frames == 3 and context.rendered_pixels == 75
    finally:
        context.close()


def test_later_cyclic_alias_stays_out_of_batch_and_reuses_completed_native_bytes(tmp_path, monkeypatch):
    context, view, shape, renderer = make_case(tmp_path, monkeypatch)
    box = (0, 0, 8, 8)
    alias = shape[0]
    try:
        ref = context.image_provider(view, shape, demand(shape, {0: box, 1: box, alias: box},
            build_cyclic_frame_addressing(shape, 1)))
        assert renderer.prefixes == [(0, 1)] and not renderer.scalar_frames
        np.testing.assert_array_equal(crop_from(ref, alias, box), crop_from(ref, 0, box)[:, ::-1])
        assert context.rendered_frames == 2 and context.rendered_pixels == 128
        assert context.image_cache_reused_pixels == 64
        assert len(ref.frame_crops) == 3
    finally:
        context.close()


def test_same_demand_cross_parent_donor_copies_without_batch_or_scalar_rendering(tmp_path, monkeypatch):
    context, view, shape, first = make_case(tmp_path, monkeypatch)
    context._runtime = SimpleNamespace(
        release_source_cache=lambda ref: dict(status='retired', workers_finished=True, gray_mappings_retired=True),
        close=lambda: None, cancel=lambda reason: None, residency_released=True, dispatch_stats={})
    pool, box = account(64 * resources.GIB), (1, 1, 7, 7)
    planned = demand(shape, {0: box, 1: box, 2: box})
    second = PrefixRenderer(context)
    def admitted(name):
        return resources.admit_sam_parent_resources(pool, 4 * resources.GIB, name,
            base_allowance_bytes=4 * resources.GIB,
            image_phase_bytes=context.mandatory_image_phase_bytes(),
            headroom_probe=lambda: 128 * resources.GIB)
    try:
        with admitted('donor-parent') as parent:
            with context.resource_scope(parent):
                with context.image_cohort_provider(view, shape, planned) as donor:
                    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *a, **k: second)
                    with admitted('copy-parent') as recipient:
                        with context.resource_scope(recipient):
                            with context.image_cohort_provider(view, shape, planned) as copied:
                                assert copied.path != donor.path
                                assert copied.identity_sha256 == donor.identity_sha256
                                assert not second.prefixes and not second.scalar_frames
                                for frame in range(3):
                                    np.testing.assert_array_equal(crop_from(copied, frame, box),
                                        pixels(frame, 6, 6, box[:2]))
                            assert not copied.path.exists() and donor.path.exists()
        assert context.image_cache_reused_pixels == 108
        assert pool.in_use == 0
    finally:
        context.close()


def test_cpu_fragmented_holes_preserve_existing_scalar_coverage(tmp_path, monkeypatch):
    context, view, shape, unused = make_case(tmp_path, monkeypatch)
    inner, outer = (2, 2, 5, 5), (1, 1, 7, 7)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *a, **k: None)
    render_frames = []
    render = context._render_demand_crop
    def observed(*args, **kwargs):
        render_frames.append(args[1])
        return render(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', observed)
    try:
        context.image_provider(view, shape, demand(shape, {0: inner}))
        render_frames.clear()
        ref = context.image_provider(view, shape, demand(shape, {0: outer, 1: outer, 2: outer}))
        assert not unused.prefixes
        assert render_frames.count(0) > 1, 'the missing border must retain scalar fragment semantics'
        for frame in range(3):
            np.testing.assert_array_equal(crop_from(ref, frame, outer), 0)
        assert context.image_cache_reused_pixels == 9
        assert context.rendered_pixels == 108
    finally:
        context.close()


def test_impossible_native_roi_declines_before_starting_a_gpu_owner(tmp_path, monkeypatch):
    from XTA.sam_azimuthal_batch import BatchDeclined
    source = np.zeros((31, 41, 43), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(),
        azimuthal_views=('transverse',), azimuthal_azimuth_angles=(30.,))[0]
    renderer = SamGpuCropRenderer(context, view, source, source.shape[0], None,
        SimpleNamespace(), 64 * 1024**2)
    start = mock.Mock(side_effect=AssertionError('preflight acquired a GPU owner'))
    monkeypatch.setattr(renderer, '_start', start)
    try:
        with pytest.raises(BatchDeclined):
            renderer.render_compatible_prefix(view, (0, 1), np.eye(2, 3, dtype=np.float32),
                output_origin_yx=(0, 0), output_height=view.src_h, output_width=view.src_w,
                consume=lambda *args: pytest.fail('preflight called its sink'))
        start.assert_not_called()
        assert renderer.lease is None and renderer.engine is None
    finally:
        renderer.close()
        context.close()


@pytest.mark.parametrize('mode', ['cold', 'warm', 'blocked', 'failed_fence'])
def test_real_prefix_dispatch_reselects_warm_cache_and_retains_the_same_owner_until_fence(
        tmp_path, monkeypatch, mode):
    from XTA import sam_azimuthal_batch as batching
    source = np.zeros((31, 41, 43), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(),
        azimuthal_views=('transverse',), azimuthal_azimuth_angles=(30.,))[0]
    lease = SimpleNamespace(device_index=0, release=mock.Mock())
    renderer = SamGpuCropRenderer(context, view, source, source.shape[0], lease,
        SimpleNamespace(device=lambda token: token), 64 * 1024**2)
    source_owner = SimpleNamespace(shape=source.shape)
    retire = mock.Mock(side_effect=[RuntimeError('controlled batch fence failure'), {}]
                       if mode == 'failed_fence' else None)
    engine = SimpleNamespace(device='cpu', _volume_gpu=source_owner, _volume_mm=None,
        _logical_t=source.shape[0], release_inference_assets=retire)
    starts, calls, delivered, counts = [], [], [], {}
    matrix = np.eye(2, 3, dtype=np.float32)
    crop = dict(output_origin_yx=(3, 3), output_height=3, output_width=3)
    frames = (0, 1, 2, 3)
    cold = batching.choose_batch_prefix(renderer, view, frames, matrix, retained_cache_bytes=0, **crop)
    assert cold.eligible and cold.batch_size == 4
    retained = 0
    if mode == 'warm':
        two = batching.plan_batch(renderer, view, frames[:2], matrix, retained_cache_bytes=0, **crop)
        allowed_loops = 2 * two.scalar_gather_iterations_per_image
        minimum_rows = (two.selected_rows + allowed_loops - 1) // allowed_loops
        roi_width = two.bbox_yx[3] - two.bbox_yx[1]
        projection_minimum = two.projection_bytes - 256 * 2 * roi_width * (two.row_block - minimum_rows)
        retained = two.existing_gpu_allowance_bytes - max(two.warp_bytes, projection_minimum)
    elif mode == 'blocked':
        retained = cold.existing_gpu_allowance_bytes
    selected = batching.choose_batch_prefix(renderer, view, frames, matrix,
        retained_cache_bytes=retained, **crop)
    if mode == 'warm':
        assert selected.eligible and selected.batch_size == 2
    elif mode == 'blocked':
        assert not selected.eligible
    def start():
        starts.append(True)
        renderer.engine = engine
        return engine
    def snapshot(actual_engine):
        assert starts and actual_engine is engine
        assert renderer.lease is lease and renderer.source is source
        return dict(retained_cache_bytes=retained, device='cpu')
    def execute(actual_renderer, actual_view, actual_frames, inverse, *, consume, **actual_crop):
        assert actual_renderer is renderer and actual_view is view and actual_crop == crop
        assert renderer.engine is engine and engine._volume_gpu is source_owner
        lease.release.assert_not_called()
        calls.append(tuple(actual_frames))
        for position in range(len(actual_frames)):
            consume(position, np.zeros((3, 3), np.uint8))
        receipt = dict(asdict(selected), completed_images=len(actual_frames),
            effective_batch_size=len(actual_frames), batch_path_used=True, host_result_mode='sink',
            enqueue_host_seconds=0., copy_and_sink_host_seconds=0.)
        return batching.BatchResult((), receipt)
    def add(name, value):
        counts[name] = counts.get(name, 0) + value
    telemetry = SimpleNamespace(add=add, set_scheduler_gauge=lambda *a, **k: None,
                                trace_event=lambda *a, **k: None)
    monkeypatch.setattr(renderer, '_start', start)
    monkeypatch.setattr(batching, 'retained_cache_snapshot', snapshot)
    monkeypatch.setattr(batching, 'render_compatible_batch', execute)
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda: telemetry)
    monkeypatch.setattr('XTA.backprojection._trim_main_process_cuda_device', lambda *a, **k: None)
    try:
        if mode == 'blocked':
            with pytest.raises(batching.BatchDeclined, match='no_funded_prefix'):
                renderer.render_compatible_prefix(view, frames, matrix, **crop,
                    consume=lambda *args: pytest.fail('unfunded batch called its sink'))
            assert not calls
        else:
            receipt = renderer.render_compatible_prefix(view, frames, matrix, **crop,
                consume=lambda position, image: delivered.append(position))
            assert calls == [frames[:selected.batch_size]]
            assert delivered == list(range(selected.batch_size))
            assert receipt['completed_images'] == selected.batch_size
            assert counts['sam.gpu_images.frames'] == selected.batch_size
            assert counts['sam.gpu_images.pixels'] == 9 * selected.batch_size
            assert counts['sam.gpu_images.batches'] == 1
        assert starts == [True]
        assert renderer.lease is lease and renderer.engine is engine
        lease.release.assert_not_called()
        if mode == 'failed_fence':
            with pytest.raises(RuntimeError, match='controlled batch fence failure'):
                renderer.close()
            assert renderer.lease is lease and renderer.engine is engine
            assert engine._volume_gpu is source_owner
            assert context._unsettled_image_renderers == [renderer]
            lease.release.assert_not_called()
        renderer.close()
        assert renderer.lease is None and renderer.engine is None
        assert lease.release.call_count == 1 and not context._unsettled_image_renderers
    finally:
        renderer.close()
        context.close()
