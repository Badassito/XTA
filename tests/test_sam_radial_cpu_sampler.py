"""SAM's compiled Radial intensity feed retains the independent NumPy oracle."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import radial_rendering_cpu as sampler, sam_canvas_rendering as feed
from XTA.cylindrical_geometry import render_shell_frame


SOURCE = np.random.default_rng(156023).integers(0, 256, (37, 43, 47), np.uint8)
TILTS = (('', 0.), ('vertical', 31.), ('vertical', -31.), ('horizontal', 41.), ('horizontal', -41.))


def view(shape=SOURCE.shape, base='transverse', tilt=('', 0.)):
    plane_h, plane_w = ((shape[1], shape[2]) if base == 'transverse' else
                       (shape[0], shape[2]) if base == 'sagittal' else (shape[0], shape[1]))
    return SimpleNamespace(family='radial', name='sam_radial_cpu_test',
        full_t=shape[0], full_h=shape[1], full_w=shape[2], src_h=65, src_w=65,
        radial_base_view=base, center_x=(plane_w-1)/2., center_y=(plane_h-1)/2.,
        radial_arc_origin=117.125, radial_height_origin=0., radial_tilted_source=bool(tilt[0]),
        tilt_direction=tilt[0], tilt_angle_deg=tilt[1], radial_radii=(.7, 9.7, 18., 24., -.7, 0.))


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('tilt', TILTS)
def test_gray8_forward_crops_match_numpy_axes_tilts_wraps_and_zero_extension(base, tilt):
    physical = view(base=base, tilt=tilt)
    for frame in range(6):
        for bbox in ((0, 0, 65, 65), (31, 11, 34, 63), (62, 61, 65, 65)):
            actual = sampler.render_radial_gray8_crop(SOURCE, physical, frame, bbox)
            with np.errstate(divide='ignore', invalid='ignore'):
                expected = render_shell_frame(SOURCE, physical, frame, bbox_yx=bbox)
            assert actual.dtype == np.uint8 and actual.flags.c_contiguous and actual.shape == expected.shape
            assert np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max(initial=0) <= 1


@pytest.mark.parametrize('base,shape', (('transverse', (1, 5, 7)),
    ('sagittal', (5, 1, 7)), ('coronal', (5, 7, 1))))
@pytest.mark.parametrize('tilt', TILTS[1:])
def test_partial_taps_on_thin_sources_and_height_padding_are_preserved(base, shape, tilt):
    source = np.random.default_rng(156023).integers(0, 256, shape, np.uint8)
    physical = view(shape, base, tilt)
    physical.radial_radii = (.7, 1.7, 2.)
    physical.radial_arc_origin = 4.125
    for frame in range(3):
        bbox = (0, 0, 7, 7)
        actual = sampler.render_radial_gray8_crop(source, physical, frame, bbox)
        expected = render_shell_frame(source, physical, frame, bbox_yx=bbox)
        np.testing.assert_array_equal(actual, expected)
        assert not actual[1:].any(), 'Height padding must not shear back into the source'


@pytest.mark.parametrize('gray', (0, 1, 127, 255))
def test_constant_source_fractional_global_origins_and_signed_radii(gray):
    source = np.full(SOURCE.shape, gray, np.uint8)
    physical = view(tilt=('horizontal', -41.))
    physical.radial_arc_origin, physical.radial_height_origin = -21.125, -.5
    bbox = (0, 0, 65, 65)
    for frame in range(6):
        actual = sampler.render_radial_gray8_crop(source, physical, frame, bbox)
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = render_shell_frame(source, physical, frame, bbox_yx=bbox)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('reverse', (False, True))
def test_sampler_borrows_readonly_strided_sources(reverse):
    backing = np.empty((74, 43, 94), np.uint8)
    backing[::2, :, ::2] = SOURCE
    source = backing[::2, :, ::2]
    if reverse:
        source = source[::-1, ::-1, ::-1]
    source.flags.writeable = False
    before = source.tobytes()
    physical, bbox = view(base='coronal', tilt=('vertical', -31.)), (5, 7, 35, 59)
    actual = sampler.render_radial_gray8_crop(source, physical, 2, bbox)
    expected = render_shell_frame(source, physical, 2, bbox_yx=bbox)
    np.testing.assert_array_equal(actual, expected)
    assert source.tobytes() == before and not source.flags.writeable and not source.flags.c_contiguous


@pytest.mark.parametrize('field,value', (('family', 'spherical'), ('full_t', 38), ('src_h', 0),
    ('radial_base_view', 'unsupported'), ('radial_radii', (float('nan'),)),
    ('radial_radii', (float('inf'),)), ('center_x', float('nan')),
    ('radial_arc_origin', float('inf')), ('radial_height_origin', float('nan')),
    ('radial_tilted_source', True)))
def test_bad_geometry_is_rejected_before_native_entry(monkeypatch, field, value):
    physical = view()
    setattr(physical, field, value)
    if field == 'radial_tilted_source':
        physical.tilt_angle_deg = float('inf')
    monkeypatch.setattr(sampler, '_compiled_render', lambda *args: pytest.fail('Unchecked native entry'))
    with pytest.raises(ValueError):
        sampler.render_radial_gray8_crop(SOURCE, physical, 0, (0, 0, 2, 2))


@pytest.mark.parametrize('bbox', ((-1, 0, 4, 4), (0, 0, 66, 4), (4, 0, 4, 4),
                                (0., 0, 4, 4), (False, 0, 4, 4), (0, 0, 4), 4))
def test_bad_crop_is_rejected_before_native_entry(monkeypatch, bbox):
    monkeypatch.setattr(sampler, '_compiled_render', lambda *args: pytest.fail('Unchecked native entry'))
    with pytest.raises(ValueError, match='crop bounds'):
        sampler.render_radial_gray8_crop(SOURCE, view(), 0, bbox)


def test_sam_only_dispatch_keeps_general_categorical_nonuint8_and_empty_reference(monkeypatch):
    physical, bbox = view(), (0, 0, 2, 2)
    expected = render_shell_frame(SOURCE, physical, 0, bbox_yx=bbox)
    np.testing.assert_array_equal(feed._render_native_shell_crop(SOURCE, physical, 0, bbox), expected)
    monkeypatch.setattr(sampler, 'render_radial_gray8_crop', lambda *args: pytest.fail('Unexpected native call'))
    np.testing.assert_array_equal(feed._render_native_shell_crop(SOURCE, physical, 0, None), np.zeros((1, 1), np.uint8))
    np.testing.assert_array_equal(feed._render_native_shell_crop(SOURCE.astype(np.float32), physical, 0, bbox), expected)
    np.testing.assert_array_equal(render_shell_frame(SOURCE, physical, 0, bbox_yx=bbox), expected)
    categorical = render_shell_frame(SOURCE, physical, 0, categorical=True, bbox_yx=bbox)
    assert set(np.unique(categorical)) <= {0, 1}


def test_lazy_source_is_refused_without_materialization_and_native_failure_propagates(monkeypatch):
    class LazySource:
        def __array__(self, *args):
            pytest.fail('Unfunded source materialization')
    with pytest.raises(ValueError, match='materialized'):
        sampler.render_radial_gray8_crop(LazySource(), view(), 0, (0, 0, 2, 2))
    failure = RuntimeError('Native Radial failed')
    def fail(*args):
        raise failure
    monkeypatch.setattr(sampler, '_compiled_render', fail)
    with pytest.raises(RuntimeError) as caught:
        feed._render_native_shell_crop(SOURCE, view(), 0, (0, 0, 2, 2))
    assert caught.value is failure


def test_native_identity_changes_actual_image_cache_key(tmp_path, monkeypatch):
    from tests.test_sam_native_crop_feed import case
    from tests.test_sam_view_image_cache import context_for, demand
    source, physical = case('radial')
    context = context_for(tmp_path, source)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args: None)
    shape = (physical.num_slices, physical.src_h, physical.src_w)
    plan = demand(shape, {physical.num_slices//2: (13, 17, 29, 35)})
    try:
        first = context.image_provider(physical, shape, plan)
        monkeypatch.setattr(sampler, 'IMPLEMENTATION_SHA256', 'f'*64)
        second = context.image_provider(physical, shape, plan)
        assert first.identity_sha256 != second.identity_sha256 and first.path != second.path
    finally:
        context.close()


def test_kernel_identity_records_actual_compiler_and_has_no_parallel_pool():
    identity = sampler.radial_gray8_sampling_identity()
    assert identity['backend'] == 'numba_nogil_f64_radial_f32_trilinear'
    assert identity['implementation_sha256'] == sampler.IMPLEMENTATION_SHA256
    assert identity['numba_version'] and identity['llvmlite_version'] and identity['llvm_version']
    assert identity['host_cpu_name'] and isinstance(identity['using_svml'], bool)
    assert sampler._compiled_render.targetoptions['nogil'] is True
    assert sampler._compiled_render.targetoptions['fastmath'] is False
    assert not sampler._compiled_render.targetoptions.get('parallel', False)
