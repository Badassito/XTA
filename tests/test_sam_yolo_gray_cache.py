"""Canonical captured YOLO pixels feed SAM without another source projection."""
from dataclasses import replace
import sqlite3
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, gray_cache
from XTA.gray_cache import GrayCacheError, GrayCacheReader, GrayCacheWriter, gray_cache_path
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.sam_cyclic import build_cyclic_frame_addressing
from XTA.sam_gpu_rendering import live_image_sampling, same_live_image_geometry
from XTA.sam_integration import SAM_YOLO_GRAY_IMAGE_CONTRACT
from XTA.sam_view_geometry import sam_gray_cache_geometry_record
from tests.test_sam_view_image_cache import context_for, crop_from, demand
from tests.test_sam_view_orientations import SOURCE_SHAPE, VIEWS


CACHED_VIEWS = tuple(next(view for view in VIEWS if view.family == family)
                     for family in ('tilted', 'azimuthal', 'radial', 'spherical'))


def cache_spec(context, view, shape):
    return dict(source_identity=context.source_identity,
        source_resize_semantics=context.source_resize_semantics,
        physical_view_record=sam_gray_cache_geometry_record(view, shape, context.source_grid_shape,
            source_processing_shape_tyx=context.source_volume.shape),
        native_centers=list(range(shape[0])), processing_shape=list(context.source_volume.shape),
        canvas_shape=list(shape), output_to_processing_affine=[[1., 0., 0.], [0., 1., 0.]],
        capture_contract=dict(contract='xta.cuda_yolo_center_gray8/1', backend='cuda_normalized_yolo',
            channel_token='gray3', channel_offsets=[-1, 0, 1], channel_index=1,
            input_dtype='float16_or_float32', output_dtype='uint8',
            conversion='float32_clamp_0_1_multiply_255_round_to_nearest_even'))


def install_cache(context, view, shape, pixels=None, *, sealed=True, frames=None, root=None, codec='auto'):
    context.gray_cache_root = root or context.temp_dir/'captured'
    spec = cache_spec(context, view, shape)
    context.gray_cache_specs[geometry.physical_view_name(view)] = spec
    if pixels is None:
        t, y, x = np.indices(shape)
        pixels = ((t*31+y*13+x*7+51) % 251).astype(np.uint8)
    writer = GrayCacheWriter(gray_cache_path(context.gray_cache_root, spec), spec, codec=codec, min_free_bytes=0)
    try:
        indices = list(range(shape[0])) if frames is None else list(frames)
        for start in range(0, len(indices), 8):
            batch = indices[start:start+8]
            assert writer.write_frames(batch, pixels[batch])
        if sealed:
            assert writer.seal()
    finally:
        writer.close()
    return pixels


@pytest.mark.parametrize('view', CACHED_VIEWS, ids=lambda view: view.family)
@pytest.mark.parametrize('angle', (0., 31.), ids=('canonical', 'unwarped_tta'))
def test_canonical_capture_bypasses_gpu_and_every_native_preparation(tmp_path, view, angle):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    pixels = install_cache(context, view, shape)
    actual_view = geometry.expand_views_into_tta_variants((view,), (angle,))[0]
    frame, bbox = shape[0]//2, (1, 2, 7, 6)
    context._resource_local.image_prefetch_cancel = mock.Mock(is_set=lambda: False)
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', side_effect=AssertionError('GPU admission')), \
                mock.patch.object(context, '_render_demand_crop', side_effect=AssertionError('source render')), \
                mock.patch.object(context, '_transverse_batch_iterator', side_effect=AssertionError('native batching')), \
                mock.patch.object(context, '_can_prefetch_cpu_images', side_effect=AssertionError('native prefetch')):
            reference = context.image_provider(actual_view, shape, demand(shape, {frame: bbox}))
        np.testing.assert_array_equal(crop_from(reference, frame, bbox), pixels[frame, 1:7, 2:6])
        sampling = live_image_sampling(reference)
        assert sampling['contract'] == SAM_YOLO_GRAY_IMAGE_CONTRACT
        assert sampling['capture']['content_digest']
        assert sampling['capture']['codecs']
        assert context.native_sampling_calls == 0
    finally:
        context._resource_local.image_prefetch_cancel = None
        context.close()


def test_azimuthal_aliases_reflect_canonical_capture_then_enlarge_without_new_identity(tmp_path):
    view = CACHED_VIEWS[1]
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 270, 270)
    pixels = install_cache(context, view, shape)
    addressing = build_cyclic_frame_addressing(shape, 1)
    alias = shape[0]
    small, large = (2, 1, 266, 257), (1, 0, 269, 269)
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', side_effect=AssertionError('GPU admission')), \
                mock.patch.object(context, '_render_demand_crop', side_effect=AssertionError('source render')):
            first = context.image_provider(view, shape, demand(shape, {alias: small}, addressing))
            enlarged = context.image_provider(view, shape, demand(shape, {alias: large}, addressing))
            native = context.image_provider(view, shape, demand(shape, {0: (1, 1, 269, 270)}))
        np.testing.assert_array_equal(crop_from(first, alias, small), pixels[0, :, ::-1][2:266, 1:257])
        np.testing.assert_array_equal(crop_from(enlarged, alias, large), pixels[0, :, ::-1][1:269, :269])
        np.testing.assert_array_equal(crop_from(native, 0, (1, 1, 269, 270)), pixels[0, 1:269, 1:270])
        assert first.identity_sha256 == enlarged.identity_sha256
        assert context.image_cache_reused_pixels >= 16
    finally:
        context.close()


@pytest.mark.parametrize('reason', ('missing', 'unsealed', 'source', 'geometry', 'canvas'))
def test_missing_or_incompatible_capture_uses_existing_source_path(tmp_path, reason):
    view = CACHED_VIEWS[0]
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    if reason == 'missing':
        context.gray_cache_root = tmp_path/'absent'
        context.gray_cache_specs[geometry.physical_view_name(view)] = cache_spec(context, view, shape)
    else:
        install_cache(context, view, shape, sealed=reason != 'unsealed',
            frames=(0,) if reason == 'unsealed' else None)
    if reason == 'source':
        context.source_identity = 'different-source'
    elif reason == 'geometry':
        view = replace(view, center_x=view.center_x+.125)
    elif reason == 'canvas':
        shape = (shape[0], 10, 10)
    bbox = (1, 1, 5, 5)
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', return_value=None) as gpu, \
                mock.patch.object(context, '_render_demand_crop', return_value=np.full((4, 4), 7, np.uint8)) as render:
            reference = context.image_provider(view, shape, demand(shape, {0: bbox}))
        assert gpu.call_count == render.call_count == 1
        np.testing.assert_array_equal(crop_from(reference, 0, bbox), 7)
        assert live_image_sampling(reference)['backend'] == 'cpu'
    finally:
        context.close()


@pytest.mark.parametrize('state', ('busy', 'uninitialized'))
def test_temporarily_unavailable_cache_records_reason_and_uses_source(tmp_path, state):
    view = CACHED_VIEWS[0]
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    if state == 'busy':
        install_cache(context, view, shape)
    else:
        context.gray_cache_root = tmp_path/'initializing'
        context.gray_cache_specs[geometry.physical_view_name(view)] = cache_spec(context, view, shape)
    spec = context.gray_cache_specs[geometry.physical_view_name(view)]
    path = gray_cache_path(context.gray_cache_root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    if state == 'busy':
        connection.execute('BEGIN EXCLUSIVE')
    else:
        connection.close()
        connection = None
    telemetry = mock.Mock()
    try:
        with mock.patch('XTA.sam_integration.runtime_telemetry', return_value=telemetry), \
                mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', return_value=None), \
                mock.patch.object(context, '_render_demand_crop', return_value=np.full((4, 4), 7, np.uint8)) as render:
            reference = context.image_provider(view, shape, demand(shape, {0: (1, 1, 5, 5)}))
        assert render.call_count == 1
        np.testing.assert_array_equal(crop_from(reference, 0, (1, 1, 5, 5)), 7)
        assert live_image_sampling(reference)['backend'] == 'cpu'
        telemetry.add.assert_any_call('sam.gray_cache.unavailable_fallbacks', 1)
        reason = next(call.args[1] for call in telemetry.gauge.call_args_list
                      if call.args[0] == 'sam.gray_cache.last_unavailable_reason')
        assert ('busy' if state == 'busy' else 'not committed') in reason
    finally:
        if connection is not None:
            connection.rollback()
            connection.close()
        context.close()


def test_active_decode_failure_discards_entire_private_cohort_and_closes_reader(tmp_path):
    view = CACHED_VIEWS[0]
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    assert shape[0] > 1
    install_cache(context, view, shape)
    calls, closed = [], []
    original_read, original_close = GrayCacheReader.read_crop, GrayCacheReader.close
    def read(reader, *args):
        calls.append(args)
        if len(calls) == 2:
            raise GrayCacheError('corrupt block after successful first frame')
        return original_read(reader, *args)
    def close(reader):
        closed.append(reader)
        return original_close(reader)
    try:
        with mock.patch.object(GrayCacheReader, 'read_crop', new=read), \
                mock.patch.object(GrayCacheReader, 'close', new=close), \
                mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', side_effect=AssertionError('GPU fallback')), \
                mock.patch.object(context, '_render_demand_crop', side_effect=AssertionError('source fallback')):
            with pytest.raises(GrayCacheError, match='corrupt block'):
                context.image_provider(view, shape, demand(shape, {0: (1, 1, 7, 7), 1: (1, 1, 7, 7)}))
        assert len(calls) == 2 and len(closed) == 1
        assert not context._cache_entries and not context._image_builds
        assert context._image_build_credit_bytes == 0
        assert not list((context.temp_dir/'sam_image_cache').glob('*.dat'))
    finally:
        context.close()


def test_content_identity_separates_donors_and_authenticates_gray_source_retries(tmp_path):
    view = CACHED_VIEWS[0]
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    small, large = (2, 2, 6, 6), (1, 1, 7, 7)
    try:
        install_cache(context, view, shape, np.full(shape, 123, np.uint8), root=tmp_path/'first')
        first = context.image_provider(view, shape, demand(shape, {0: small}))
        install_cache(context, view, shape, np.full(shape, 231, np.uint8), root=tmp_path/'second')
        second = context.image_provider(view, shape, demand(shape, {0: large}))
        assert first.identity_sha256 != second.identity_sha256
        np.testing.assert_array_equal(crop_from(second, 0, large), 231)
        assert same_live_image_geometry(first.identity_sha256, second)
        context.gray_cache_root = None
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', return_value=None):
            source = context.image_provider(view, shape, demand(shape, {0: large}))
        np.testing.assert_array_equal(crop_from(source, 0, large), 0)
        assert source.identity_sha256 != second.identity_sha256
        assert same_live_image_geometry(first.identity_sha256, source)
        assert not same_live_image_geometry(first.identity_sha256, LtaPhysicalViewCacheRef.from_payload(source.payload()))
    finally:
        context.close()


def test_actual_codec_labels_are_part_of_capture_identity(tmp_path):
    view = CACHED_VIEWS[0]
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    pixels = np.full(shape, 123, np.uint8)
    planned = demand(shape, {0: (1, 1, 7, 7)})
    try:
        install_cache(context, view, shape, pixels, root=tmp_path/'compressed', codec='zlib')
        compressed = context.image_provider(view, shape, planned)
        install_cache(context, view, shape, pixels, root=tmp_path/'raw', codec='raw')
        raw = context.image_provider(view, shape, planned)
        left, right = live_image_sampling(compressed)['capture'], live_image_sampling(raw)['capture']
        assert left['content_digest'] == right['content_digest']
        assert left['codecs'] == ['zlib'] and right['codecs'] == ['raw']
        assert compressed.identity_sha256 != raw.identity_sha256
        np.testing.assert_array_equal(crop_from(raw, 0, (1, 1, 7, 7)), 123)
    finally:
        context.close()


@pytest.mark.parametrize('view', CACHED_VIEWS, ids=lambda view:view.name)
def test_decoder_workspace_is_charged_before_read_and_insufficient_profile_falls_back(tmp_path, view):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    install_cache(context, view, shape)
    bbox = (1, 1, 7, 7)
    payload = 36
    profile = object()
    context._resource_local.profile = profile
    original_read = GrayCacheReader.read_crop
    seen = []
    def read(reader, *args):
        credit = context._image_build_credit_bytes
        assert reader.max_cache_bytes == 64*1024**2
        assert credit >= payload+reader.max_cache_bytes
        assert credit >= payload+reader.crop_workspace_bytes(bbox)
        seen.append(credit)
        return original_read(reader, *args)
    try:
        with mock.patch('XTA.sam_resources.validate_live_sam_resource_profile',
                return_value=dict(base_non_cpu_allowance_bytes=2**30, lease_id='one-parent')), \
                mock.patch.object(GrayCacheReader, 'read_crop', new=read), \
                mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', side_effect=AssertionError('GPU admission')):
            context.image_provider(view, shape, demand(shape, {0: bbox}))
        assert len(seen) == 1 and context._image_build_credit_bytes == 0
        with mock.patch('XTA.sam_resources.validate_live_sam_resource_profile',
                return_value=dict(base_non_cpu_allowance_bytes=payload+64*1024**2-1, lease_id='one-parent')), \
                mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', return_value=None) as gpu, \
                mock.patch.object(GrayCacheReader, 'read_crop', side_effect=AssertionError('unfunded decoder read')), \
                mock.patch.object(context, '_render_demand_crop', return_value=np.zeros((6, 6), np.uint8)) as render:
            source = context.image_provider(view, shape, demand(shape, {0: bbox}))
        # Qualified shell crops use the bounded CPU route when no idle SDK
        # handoff is available; Azimuthal retains its GPU-first fallback.
        assert gpu.call_count == int(view.family == 'azimuthal')
        assert render.call_count == 1 and context._image_build_credit_bytes == 0
        assert live_image_sampling(source)['backend'] == 'cpu'
    finally:
        context._resource_local.profile = None
        context.close()


def test_adjacent_native_frames_reuse_one_decoded_3d_chunk(tmp_path):
    view = CACHED_VIEWS[0]
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    shape = (view.num_slices, 8, 8)
    assert shape[0] > 1
    pixels = install_cache(context, view, shape)
    bbox = (1, 1, 7, 7)
    try:
        with mock.patch.object(gray_cache, '_decode', wraps=gray_cache._decode) as decode, \
                mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer', side_effect=AssertionError('GPU admission')):
            reference = context.image_provider(view, shape, demand(shape, {0: bbox, 1: bbox}))
        assert decode.call_count == 1
        for frame in (0, 1):
            np.testing.assert_array_equal(crop_from(reference, frame, bbox), pixels[frame, 1:7, 1:7])
    finally:
        context.close()
