"""Canonical all-view SAM crops and amortized source/cache ownership, CPU only."""
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry, sam_integration
from XTA.media import LazyProcessingCube
from XTA.sam_cyclic import build_cyclic_frame_addressing
from tests.test_sam_view_orientations import SOURCE_SHAPE, VIEWS


def context_for(tmp_path, source, **kwargs):
    return sam_integration.SamInterpolationContext(model_path='unused', device_ids=(0,),
        temp_dir=tmp_path/'runtime', evidence_root=tmp_path/'evidence',
        source_volume=source, source_identity='immutable-source', **kwargs)


def demand(shape, frames, addressing=None):
    addressing = dict(addressing or {})
    logical = tuple(addressing.get('evidence_shape_tyx', shape))
    return SimpleNamespace(frame_crop_bounds=frames,
        plan=SimpleNamespace(virtual_shape_tyx=logical, frame_addressing=addressing))


def crop_from(reference, frame, bbox):
    data = reference.open()
    try:
        if reference.frame_crops:
            record = next(item for item in reference.frame_crops if item[0] == frame)
            _, y0, x0, y1, x1, offset = record
            plane = data[offset:offset+(y1-y0)*(x1-x0)].reshape(y1-y0, x1-x0)
        else:
            plane = data[frame]
            y0 = x0 = 0
        cy0, cx0, cy1, cx1 = bbox
        return plane[cy0-y0:cy1-y0, cx0-x0:cx1-x0].copy()
    finally:
        data._mmap.close()


def expected_canvas(context, view, shape, frame):
    affine, inverse, _ = context._canvas_transform(view, shape)
    return geometry.render_intensity_frame_on_grid(context.source_volume, view, frame,
        M_src_to_out=affine, M_out_to_src=inverse, output_height=shape[1], output_width=shape[2])


@pytest.mark.parametrize('view', VIEWS, ids=lambda view: view.name)
@pytest.mark.parametrize('processing', (False, True), ids=('native', 'processing'))
def test_needed_crop_matches_established_tta_sampler_on_canonical_grid(tmp_path, view, processing):
    source = (np.arange(np.prod(SOURCE_SHAPE), dtype=np.uint16)*17 % 251).astype(np.uint8).reshape(SOURCE_SHAPE)
    # Detector augmentation was inverted before SAM's observed canvas.
    view = geometry.expand_views_into_tta_variants((view,), (31.,))[0]
    shape = (view.num_slices, 8, 8) if processing else (view.num_slices, view.src_h, view.src_w)
    frame = view.num_slices//2
    bbox = (1, 1, shape[1]-1, shape[2]-1)
    context = context_for(tmp_path, source)
    try:
        planned = demand(shape, {frame: bbox})
        reference = context.image_provider(view, shape, planned)
        expected = expected_canvas(context, view, shape, frame)
        np.testing.assert_array_equal(crop_from(reference, frame, bbox), expected[1:-1, 1:-1])
        assert context.rendered_pixels == (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        assert context.image_provider(view, shape, planned) is reference
        assert context.rendered_frames == 1
        assert context._canvas_transform(view, shape)[2]['detector_augmentation_angle_deg'] == 31.
    finally:
        context.close()


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_cubic_source_backing_reuse_requires_actual_transverse_axes(tmp_path, base):
    path = tmp_path/'source.dat'
    source = np.memmap(path, dtype=np.uint8, mode='w+', shape=(5, 5, 5))
    source[:] = np.arange(125).reshape(source.shape)
    source.flush()
    view = geometry.get_view_infos(*source.shape, cartesian_views=(base,))[0]
    context = context_for(tmp_path, source)
    try:
        reference = context.image_provider(view, source.shape)
        assert (reference.path == path.resolve()) == (base == 'transverse')
        assert context.exact_backing_reuses == int(base == 'transverse')
        expected = {'transverse': source, 'sagittal': source.transpose(1, 0, 2),
                    'coronal': source.transpose(2, 0, 1)}[base]
        actual = reference.open()
        try:
            np.testing.assert_array_equal(actual, expected)
        finally:
            actual._mmap.close()
    finally:
        context.close()
        source._mmap.close()


@pytest.mark.parametrize('base', ('sagittal', 'coronal'))
def test_non_z_lazy_orientation_materializes_shared_cube_once(tmp_path, base):
    source = np.arange(3*4*5, dtype=np.uint8).reshape(3, 4, 5)
    lazy = LazyProcessingCube(source, (5, 6, 7), tmp_path/'cube.dat', workers=1,
        request_path=tmp_path/'request', ready_path=tmp_path/'ready', failed_path=tmp_path/'failed')
    context = context_for(tmp_path, lazy)
    view = geometry.get_view_infos(*lazy.shape, cartesian_views=(base,))[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    bbox = (0, 0, view.src_h, view.src_w)
    try:
        with mock.patch.object(lazy, 'materialize', wraps=lazy.materialize) as materialize:
            first = context.image_provider(view, shape, demand(shape, {0: bbox}))
            # A different frame needs the same shared processing cube, not a
            # decoded Z-slice substituted for this orientation's native plane.
            second = context.image_provider(view, shape, demand(shape, {1: bbox}))
            constructions = materialize.call_count
            assert lazy.materialized
            assert context.source_materializations == 1
            expected = lazy._array.transpose(1, 0, 2) if base == 'sagittal' else lazy._array.transpose(2, 0, 1)
            np.testing.assert_array_equal(crop_from(first, 0, bbox), expected[0])
            np.testing.assert_array_equal(crop_from(second, 1, bbox), expected[1])
            # Getter calls may ask materialize for the already-built array;
            # the actual construction and retained source are shared once.
            assert constructions >= 1
    finally:
        context.close()
        lazy.close()


def test_transverse_lazy_crop_preserves_unmaterialized_fast_path(tmp_path):
    source = np.arange(3*4*5, dtype=np.uint8).reshape(3, 4, 5)
    lazy = LazyProcessingCube(source, (5, 4, 5), tmp_path/'cube.dat', workers=1,
        request_path=tmp_path/'request', ready_path=tmp_path/'ready', failed_path=tmp_path/'failed')
    view = geometry.get_view_infos(*lazy.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, lazy)
    shape = lazy.shape
    try:
        reference = context.image_provider(view, shape, demand(shape, {2: (1, 1, 4, 5)}))
        assert not lazy.materialized
        assert context.source_materializations == 0
        assert crop_from(reference, 2, (1, 1, 4, 5)).shape == (3, 4)
    finally:
        context.close()
        lazy.close()


def test_superset_and_partial_overlap_render_only_missing_pixels(tmp_path):
    source = np.arange(3*12*13, dtype=np.uint16).astype(np.uint8).reshape(3, 12, 13)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, source)
    shape = source.shape
    try:
        with mock.patch.object(context, '_render_demand_crop', wraps=context._render_demand_crop) as render:
            original = context.image_provider(view, shape, demand(shape, {1: (2, 3, 8, 9)}))
            assert context.rendered_pixels == 36
            assert context.image_provider(view, shape, demand(shape, {1: (3, 4, 7, 8)})) is original
            assert context.image_cache_superset_hits == 1
            assert render.call_count == 1
            expanded = context.image_provider(view, shape, demand(shape, {1: (1, 2, 9, 10)}))
            assert context.rendered_pixels == 64  # disjoint uncovered strips
            assert context.image_cache_reused_pixels == 36
            np.testing.assert_array_equal(crop_from(expanded, 1, (1, 2, 9, 10)), source[1, 1:9, 2:10])
            before = render.call_count
            assert context.image_provider(view, shape, demand(shape, {1: (1, 2, 9, 10)})) is expanded
            assert render.call_count == before
            original.revalidate()
    finally:
        context.close()


def test_augmentation_identity_reuses_canonical_pixels_but_axis_recipe_does_not(tmp_path):
    source = np.arange(125, dtype=np.uint8).reshape(5, 5, 5)
    transverse, sagittal = geometry.get_view_infos(*source.shape, cartesian_views=('transverse', 'sagittal'))
    variants = geometry.expand_views_into_tta_variants((transverse,), (0., 31.))
    context = context_for(tmp_path, source)
    try:
        first = context.image_provider(variants[0], source.shape)
        assert context.image_provider(variants[1], source.shape) is first
        other = context.image_provider(sagittal, source.shape)
        assert other.identity_sha256 != first.identity_sha256
        assert other.path != first.path
    finally:
        context.close()


@pytest.mark.parametrize('processing', (False, True))
def test_cyclic_alias_matches_working_canvas_flip_and_reuses_native_pixels(tmp_path, processing):
    view = next(view for view in VIEWS if view.family == 'azimuthal' and not view.azimuthal_tilted_source)
    source = np.arange(np.prod(SOURCE_SHAPE), dtype=np.uint16).astype(np.uint8).reshape(SOURCE_SHAPE)
    shape = (view.num_slices, 8, 8) if processing else (view.num_slices, view.src_h, view.src_w)
    addressing = build_cyclic_frame_addressing(shape, 1)
    bbox = (1, 1, shape[1]-1, shape[2]-2)
    mirrored = (bbox[0], shape[2]-bbox[3], bbox[2], shape[2]-bbox[1])
    context = context_for(tmp_path, source)
    try:
        # Render physical support once, then another logical positive address
        # reads those same working pixels and reflects the cropped columns.
        context.image_provider(view, shape, demand(shape, {0: mirrored}))
        before = context.rendered_pixels
        alias = context.image_provider(view, shape, demand(shape, {shape[0]: bbox}, addressing))
        assert alias.shape == tuple(addressing['evidence_shape_tyx'])
        assert context.rendered_pixels == before
        expected = expected_canvas(context, view, shape, 0)[:, ::-1]
        np.testing.assert_array_equal(crop_from(alias, shape[0], bbox), expected[bbox[0]:bbox[2], bbox[1]:bbox[3]])
    finally:
        context.close()


def test_native_and_alias_in_one_transaction_reuse_the_completed_crop(tmp_path):
    view = next(view for view in VIEWS if view.family == 'azimuthal' and not view.azimuthal_tilted_source)
    source = np.arange(np.prod(SOURCE_SHAPE), dtype=np.uint16).astype(np.uint8).reshape(SOURCE_SHAPE)
    shape = (view.num_slices, view.src_h, view.src_w)
    bbox = (0, 0, shape[1], shape[2])
    context = context_for(tmp_path, source)
    try:
        reference = context.image_provider(view, shape, demand(shape, {0: bbox, shape[0]: bbox},
            build_cyclic_frame_addressing(shape, 1)))
        assert context.rendered_frames == 1
        np.testing.assert_array_equal(crop_from(reference, shape[0], bbox), crop_from(reference, 0, bbox)[:, ::-1])
    finally:
        context.close()


def test_same_name_and_shape_sampler_change_invalidates_image_cache(tmp_path):
    source = np.arange(np.prod(SOURCE_SHAPE), dtype=np.uint16).astype(np.uint8).reshape(SOURCE_SHAPE)
    view = next(view for view in VIEWS if view.family == 'azimuthal' and not view.azimuthal_tilted_source)
    shape = (view.num_slices, view.src_h, view.src_w)
    context = context_for(tmp_path, source)
    try:
        planned = demand(shape, {0: (0, 0, shape[1], shape[2])})
        first = context.image_provider(view, shape, planned)
        modified = replace(view, roi_radius=view.roi_radius*.75)
        second = context.image_provider(modified, shape, planned)
        assert second.path != first.path
        assert second.identity_sha256 != first.identity_sha256
        assert context.rendered_frames == 2
    finally:
        context.close()


def test_cyclic_image_demand_refuses_non_azimuthal_routing(tmp_path):
    source = np.zeros((3, 6, 6), np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, source)
    try:
        with pytest.raises(ValueError, match='Azimuthal'):
            context.image_provider(view, source.shape, demand(source.shape, {3: (0, 0, 6, 6)},
                build_cyclic_frame_addressing(source.shape, 1)))
        assert not (tmp_path/'runtime'/'sam_image_cache').exists()
    finally:
        context.close()


def test_failed_render_publishes_no_partial_descriptor_and_can_retry(tmp_path):
    source = np.zeros((3, 6, 6), np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, source)
    try:
        with mock.patch.object(context, '_render_demand_crop', side_effect=RuntimeError('render failed')):
            with pytest.raises(RuntimeError, match='render failed'):
                context.image_provider(view, source.shape, demand(source.shape, {0: (0, 0, 4, 4)}))
        assert not context._caches
        # Failure retirement preserves live NumPy/traceback aliases, and the
        # owned scratch unlink runs asynchronously after the last alias dies.
        from XTA.runtime import wait_for_retired_memmap_unlinks
        for path in (tmp_path/'runtime'/'sam_image_cache').glob('*.dat'):
            wait_for_retired_memmap_unlinks(path=path)
        assert not list((tmp_path/'runtime'/'sam_image_cache').glob('*.dat'))
        assert context.image_provider(view, source.shape, demand(source.shape, {0: (0, 0, 4, 4)}))
    finally:
        context.close()


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('side', (1024, 1535, 1536))
def test_awkward_scale_global_coordinates_are_exact_for_new_expanded_and_partial_crops(tmp_path, base, side):
    # Independent review found 1--8 one-LSB mismatches here when a float32
    # affine was rebased before warp. These coordinates retain its witness.
    source = np.random.default_rng(337).integers(0, 256, size=(17, 67, 113), dtype=np.uint8)
    original = geometry.get_view_infos(*source.shape, cartesian_views=(base,))[0]
    view = geometry.expand_views_into_tta_variants((original,), (31.,))[0]
    shape, frame = (view.num_slices, side, side), view.num_slices//2
    inner = (side//2+17, side-199, side//2+69, side-123)
    expanded = (inner[0]-19, inner[1]-23, inner[2]+17, inner[3]+29)
    partial = (inner[0]-41, inner[1]+11, inner[2]+31, inner[3]+55)
    context = context_for(tmp_path, source)
    try:
        expected = expected_canvas(context, view, shape, frame)
        for bbox in (inner, expanded, partial):
            before = context.native_sampling_calls
            ref = context.image_provider(view, shape, demand(shape, {frame: bbox}))
            np.testing.assert_array_equal(crop_from(ref, frame, bbox),
                expected[bbox[0]:bbox[2], bbox[1]:bbox[3]])
            # Several disjoint holes reuse one native plane in this physical
            # frame transaction, rather than repeating a shell/oriented render.
            assert context.native_sampling_calls-before == 1
        ref = context.image_provider(view, shape, demand(shape, {frame: inner}))
        np.testing.assert_array_equal(crop_from(ref, frame, inner), expected[inner[0]:inner[2], inner[1]:inner[3]])
        assert context.native_sampling_calls == 3
        assert context.canonical_sampling_pixels >= context.rendered_pixels
    finally:
        context.close()


@pytest.mark.parametrize('family', ('tilted', 'azimuthal', 'radial', 'spherical'))
@pytest.mark.parametrize('side', (129, 1535))
def test_awkward_scale_nonlinear_views_match_full_canonical_grid(tmp_path, family, side):
    source = np.random.default_rng(339).integers(0, 256, size=SOURCE_SHAPE, dtype=np.uint8)
    view = next(view for view in VIEWS if view.family == family)
    shape, frame = (view.num_slices, side, side), view.num_slices//2
    bbox = (side//4+1, side//3+3, side-1, side-2)
    context = context_for(tmp_path, source)
    try:
        expected = expected_canvas(context, view, shape, frame)
        reference = context.image_provider(view, shape, demand(shape, {frame: bbox}))
        np.testing.assert_array_equal(crop_from(reference, frame, bbox), expected[bbox[0]:bbox[2], bbox[1]:bbox[3]])
        assert context.native_sampling_calls == 1
    finally:
        context.close()


def test_cartesian_crop_budget_uses_only_demanded_native_pixels(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '2048')
    source = np.zeros((17, 67, 113), np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('sagittal',))[0]
    context = context_for(tmp_path, source)
    try:
        with mock.patch.object(geometry, 'get_view_frame_by_index',
                wraps=geometry.get_view_frame_by_index) as sampler:
            reference = context.image_provider(view, (view.num_slices, 1024, 1024),
                demand((view.num_slices, 1024, 1024), {0: (500, 500, 508, 508)}))
        assert sampler.call_args.kwargs['view_frames'] is not None
        assert not crop_from(reference, 0, (500, 500, 508, 508)).any()
        assert context.native_sampling_calls == 1
        assert context.native_sampling_pixels < int(view.src_h)*int(view.src_w)
    finally:
        context.close()


def test_global_grid_phase_matches_scalar_opencv_backend(tmp_path):
    from XTA._deps import cv2
    previous = cv2.useOptimized()
    cv2.setUseOptimized(False)
    source = np.random.default_rng(337).integers(0, 256, size=(17, 67, 113), dtype=np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('sagittal',))[0]
    context = context_for(tmp_path, source)
    shape, frame = (view.num_slices, 1535, 1535), view.num_slices//2
    bbox = (766, 1314, 854, 1442)
    try:
        expected = expected_canvas(context, view, shape, frame)
        reference = context.image_provider(view, shape, demand(shape, {frame: bbox}))
        np.testing.assert_array_equal(crop_from(reference, frame, bbox), expected[766:854, 1314:1442])
    finally:
        context.close()
        cv2.setUseOptimized(previous)


@pytest.mark.parametrize('pad_mode', ('pad', 'clamp'))
def test_lazy_transverse_native_roi_retains_awkward_global_sampling_phase(tmp_path, pad_mode):
    from XTA.media import resize_volume_to_processing_cube_gray8
    source = np.random.default_rng(341).integers(0, 256, size=(7, 19, 29), dtype=np.uint8)
    out_shape = (9, 19, 29)
    expected_source = resize_volume_to_processing_cube_gray8(source, out_shape, tmp_path/'expected.dat',
                                                            workers=1, prefer_memory=False)
    lazy = LazyProcessingCube(source, out_shape, tmp_path/'unused.dat', workers=1,
        request_path=tmp_path/'request', ready_path=tmp_path/'ready', failed_path=tmp_path/'failed')
    view = replace(geometry.get_view_infos(*out_shape, cartesian_views=('transverse',))[0], pad_mode=pad_mode)
    context = context_for(tmp_path, lazy)
    expected_context = context_for(tmp_path/'expected', expected_source)
    shape, frame, bbox = (9, 1535, 1535), 4, (766, 1007, 937, 1339)
    try:
        expected = expected_canvas(expected_context, view, shape, frame)
        ref = context.image_provider(view, shape, demand(shape, {frame: bbox}))
        np.testing.assert_array_equal(crop_from(ref, frame, bbox), expected[766:937, 1007:1339])
        assert not lazy.materialized
        assert context.native_sampling_calls == 1
    finally:
        context.close()
        expected_context.close()
        lazy.close()
        expected_source._mmap.close()


def test_phase_self_check_is_small_cached_and_backend_bound(monkeypatch):
    from XTA import sam_canvas_rendering as renderer
    from XTA._deps import cv2
    monkeypatch.setattr(renderer, '_PHASE_SELF_CHECKS', {})
    backend = renderer.canonical_sampling_backend()
    with mock.patch.object(cv2, 'warpAffine', wraps=cv2.warpAffine) as warp:
        first = renderer.ensure_canonical_phase_supported(backend)
        second = renderer.ensure_canonical_phase_supported(backend)
        assert warp.call_count == 4
        assert first == second
        assert first['status'] == 'passed'
        assert first['synthetic_comparisons'] == 8
        assert first['largest_synthetic_canvas_bytes'] < 128*1024
        changed_backend = {**backend, 'build_sha256': 'changed-build-identity'}
        changed = renderer.ensure_canonical_phase_supported(changed_backend)
        assert changed['backend_key_sha256'] != first['backend_key_sha256']
        assert warp.call_count == 8
        monkeypatch.setattr(renderer, 'IMPLEMENTATION_SHA256', 'changed-helper-identity')
        changed_helper = renderer.ensure_canonical_phase_supported(backend)
        assert changed_helper['backend_key_sha256'] != first['backend_key_sha256']
        assert warp.call_count == 12
    first['backend']['opencv_version'] = 'receipt-consumer-mutation'
    assert renderer._PHASE_SELF_CHECKS[second['backend_key_sha256']]['backend']['opencv_version'] == backend['opencv_version']


def test_native_backing_identity_needs_no_numeric_phase_probe(tmp_path):
    from XTA import sam_canvas_rendering as renderer
    source = np.memmap(tmp_path/'source.dat', mode='w+', dtype=np.uint8, shape=(3, 6, 7))
    source[:] = 31
    source.flush()
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    view = geometry.expand_views_into_tta_variants((view,), (31.,))[0]
    context = context_for(tmp_path, source)
    try:
        with mock.patch.object(renderer, 'ensure_canonical_phase_supported', side_effect=AssertionError('no identity probe')):
            reference = context.image_provider(view, source.shape)
        assert reference.path == (tmp_path/'source.dat').resolve()
        assert context.canonical_phase_self_check_receipt == {'status': 'not_required'}
    finally:
        context.close()
        source._mmap.close()


def test_detector_retirement_signal_is_readonly_and_has_no_startup_side_effects(tmp_path):
    source = np.zeros((3, 6, 7), np.uint8)
    context = context_for(tmp_path, source, detector_device_ids=(0,))
    dedicated = context_for(tmp_path/'dedicated', source, detector_device_ids=(1,))
    try:
        with mock.patch.object(context, '_start', side_effect=AssertionError('readiness started SAM')):
            assert not context.detector_retirement_ready
            context.detector_assets_retired()
            assert context.detector_retirement_ready
            assert dedicated.detector_retirement_ready
            with pytest.raises(AttributeError):
                context.detector_retirement_ready = False
        assert context._runtime is None and dedicated._runtime is None
        assert context.rendered_frames == dedicated.rendered_frames == 0
        assert not context._caches and not dedicated._caches
        context.cancel('test cancellation')
        assert context._ready.is_set()
        assert not context.detector_retirement_ready
    finally:
        context.close()
        dedicated.close()
    assert not dedicated.detector_retirement_ready


def test_unsupported_phase_fails_closed_before_model_admission(tmp_path, monkeypatch):
    from XTA import sam_canvas_rendering as renderer
    monkeypatch.setattr(renderer, '_PHASE_SELF_CHECKS', {})
    source = np.arange(3*7*9, dtype=np.uint8).reshape(3, 7, 9)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('sagittal',))[0]
    observed = np.zeros((7, 32, 32), np.uint8)
    observed[(0, 6), 14:19, 14:19] = 1
    context = context_for(tmp_path, source)

    def wrong_phase(_native, _matrix, *, output_height, output_width, **_kwargs):
        return np.zeros((output_height, output_width), np.uint8)

    try:
        with mock.patch.object(renderer, '_remap_global_affine_crop', side_effect=wrong_phase), \
             mock.patch.object(context, '_start', side_effect=AssertionError('model must remain unadmitted')) as start:
            with pytest.raises(RuntimeError, match='Unsupported OpenCV canonical SAM crop numerical phase'):
                context.interpolate(observed, view=view, scope='bad-phase', gap_distance=7,
                    min_radius=0, interpolation_walk_back=0, work_dir=tmp_path/'evidence')
        start.assert_not_called()
        assert not context._caches
        assert not (tmp_path/'runtime'/'sam_image_cache').exists()
        assert (tmp_path/'evidence'/'context_preparation_failure.json').exists()
        assert all(receipt['status'] == 'unsupported' for receipt in renderer._PHASE_SELF_CHECKS.values())
        # Failed numerical rules remain rejected; the guard never tunes itself
        # from source data or silently tries a different sampling formula.
        with pytest.raises(RuntimeError, match='SAM model admission was refused'):
            renderer.ensure_canonical_phase_supported()
    finally:
        context.close()


@pytest.mark.parametrize('pad_mode', ('pad', 'clamp'))
@pytest.mark.parametrize('side', (17, 129, 1535))
def test_odd_rectangular_azimuthal_alias_keeps_global_canvas_phase(tmp_path, pad_mode, side):
    source = np.random.default_rng(337).integers(0, 256, size=(17, 67, 113), dtype=np.uint8)
    view = geometry.get_view_infos(*source.shape, cartesian_views=(), azimuthal_views=('sagittal',),
                                   azimuthal_azimuth_angles=(60.,))[0]
    view = replace(view, pad_mode=pad_mode)
    shape = (view.num_slices, side, side)
    native = (2, max(1, side//3), side-1, side-2)
    mirror = (native[0], side-native[3], native[2], side-native[1])
    addressing = build_cyclic_frame_addressing(shape, 1)
    context = context_for(tmp_path, source)
    try:
        expected = expected_canvas(context, view, shape, 0)
        first = context.image_provider(view, shape, demand(shape, {0: native}))
        before = context.rendered_pixels
        second = context.image_provider(view, shape, demand(shape, {shape[0]: mirror}, addressing))
        np.testing.assert_array_equal(crop_from(first, 0, native), expected[native[0]:native[2], native[1]:native[3]])
        np.testing.assert_array_equal(crop_from(second, shape[0], mirror),
                                      expected[:, ::-1][mirror[0]:mirror[2], mirror[1]:mirror[3]])
        assert context.rendered_pixels == before
        assert context.native_sampling_calls == 1
    finally:
        context.close()
