"""SAM's compiled intensity feed retains the independent native NumPy oracle."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sam_canvas_rendering as feed, spherical_rendering_cpu as sampler
from XTA.spherical_geometry import cube_rotation, render_shell_frame


SOURCE = np.random.default_rng(155997).integers(0, 256, (37, 43, 47), np.uint8)


def view(face=0, rotation=None):
    return SimpleNamespace(family='spherical', name='sam_cpu_test',
        full_t=37, full_h=43, full_w=47, src_h=161, src_w=161, num_slices=4,
        spherical_face=face, spherical_face_intervals=137,
        spherical_u_origin=-17, spherical_v_origin=-21,
        spherical_rotation_xyz=rotation or cube_rotation(),
        spherical_radii=(.7, 10., 18., 24.))


@pytest.mark.parametrize('face', range(6))
@pytest.mark.parametrize('rotation', (cube_rotation(), cube_rotation('vertical', 31.),
                                   cube_rotation('horizontal', -41.)))
def test_gray8_forward_crops_match_numpy_faces_rotation_padding_and_zero_extension(face, rotation):
    physical = view(face, rotation)
    for frame in range(4):
        for bbox in ((0, 0, 161, 161), (31, 11, 34, 123), (144, 139, 161, 161)):
            actual = sampler.render_spherical_gray8_crop(SOURCE, physical, frame, bbox)
            expected = render_shell_frame(SOURCE, physical, frame, bbox_yx=bbox)
            assert actual.flags.c_contiguous and actual.dtype == np.uint8
            assert actual.shape == expected.shape
            assert np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max(initial=0) <= 1


@pytest.mark.parametrize('reverse', (False, True))
def test_sampler_borrows_readonly_noncontiguous_source(reverse):
    backing = np.empty((74, 43, 94), np.uint8)
    backing[::2, :, ::2] = SOURCE
    source = backing[::2, :, ::2]
    if reverse:
        source = source[::-1, ::-1, ::-1]
    source.flags.writeable = False
    before = source.tobytes()
    bbox = (5, 7, 145, 153)
    actual = sampler.render_spherical_gray8_crop(source, view(), 2, bbox)
    expected = render_shell_frame(source, view(), 2, bbox_yx=bbox)
    assert np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max(initial=0) <= 1
    assert source.tobytes() == before and not source.flags.writeable
    assert not source.flags.c_contiguous


@pytest.mark.parametrize('field,value', (('family', 'radial'), ('full_t', 38),
    ('src_h', 0), ('spherical_face', -1), ('spherical_face', 6), ('spherical_face', True),
    ('spherical_face_intervals', 0), ('spherical_radii', (float('nan'),)),
    ('spherical_radii', (float('inf'),)),
    ('spherical_rotation_xyz', (float('nan'),)*9)))
def test_invalid_physical_geometry_fails_before_native_entry(monkeypatch, field, value):
    physical = view()
    setattr(physical, field, value)
    monkeypatch.setattr(sampler, '_compiled_render', lambda *args: pytest.fail('Unchecked native entry'))
    with pytest.raises(ValueError):
        sampler.render_spherical_gray8_crop(SOURCE, physical, 0, (0, 0, 2, 2))


@pytest.mark.parametrize('bbox', ((-1, 0, 4, 4), (0, 0, 162, 4), (4, 0, 4, 4),
                                (0., 0, 4, 4), (False, 0, 4, 4), (0, 0, 4), 4))
def test_invalid_crop_fails_before_native_entry(monkeypatch, bbox):
    monkeypatch.setattr(sampler, '_compiled_render', lambda *args: pytest.fail('Unchecked native entry'))
    with pytest.raises(ValueError, match='crop bounds'):
        sampler.render_spherical_gray8_crop(SOURCE, view(), 0, bbox)


def test_sam_only_route_preserves_numpy_categorical_nonuint8_and_empty_planes(monkeypatch):
    physical, bbox = view(), (21, 17, 23, 19)
    expected = render_shell_frame(SOURCE, physical, 0, bbox_yx=bbox)
    np.testing.assert_array_equal(feed._render_native_shell_crop(SOURCE, physical, 0, bbox), expected)
    def fail(*args, **kwargs):
        pytest.fail('Unexpected native kernel call')
    monkeypatch.setattr(sampler, 'render_spherical_gray8_crop', fail)
    np.testing.assert_array_equal(feed._render_native_shell_crop(SOURCE, physical, 0, None),
                                  np.zeros((1, 1), np.uint8))
    np.testing.assert_array_equal(feed._render_native_shell_crop(SOURCE.astype(np.float32), physical, 0, bbox), expected)
    np.testing.assert_array_equal(render_shell_frame(SOURCE, physical, 0, bbox_yx=bbox), expected)
    categorical = render_shell_frame(SOURCE, physical, 0, categorical=True, bbox_yx=bbox)
    assert set(np.unique(categorical)) <= {0, 1}


def test_native_failure_propagates_without_reference_fallback(monkeypatch):
    failure = RuntimeError('Native sampler failed')
    def fail(*args):
        raise failure
    monkeypatch.setattr(sampler, '_compiled_render', fail)
    with pytest.raises(RuntimeError) as caught:
        feed._render_native_shell_crop(SOURCE, view(), 0, (0, 0, 2, 2))
    assert caught.value is failure


@pytest.mark.parametrize('face', range(6))
@pytest.mark.parametrize('constant', (None, 127))
def test_zero_and_signed_radius_preserve_reference_center_and_pixels(face, constant):
    physical = view(face, cube_rotation('horizontal', -41.))
    physical.spherical_radii = (0., -3.)
    source = SOURCE if constant is None else np.full(SOURCE.shape, constant, np.uint8)
    bbox = (21, 17, 155, 155)
    for frame in (0, 1):
        np.testing.assert_array_equal(sampler.render_spherical_gray8_crop(source, physical, frame, bbox),
                                      render_shell_frame(source, physical, frame, bbox_yx=bbox))


def test_sampler_refuses_lazy_source_without_materializing_it():
    class LazySource:
        def __array__(self, *args):
            pytest.fail('An unfunded source was materialized')
    with pytest.raises(ValueError, match='materialized NumPy'):
        sampler.render_spherical_gray8_crop(LazySource(), view(), 0, (0, 0, 2, 2))


def test_numerical_sampler_identity_tracks_compiler_and_source(monkeypatch):
    identity = sampler.spherical_gray8_sampling_identity()
    assert identity['backend'] == 'numba_nogil_f64_qsc_f32_trilinear'
    assert identity['implementation_sha256'] == sampler.IMPLEMENTATION_SHA256
    assert identity['numba_version'] and identity['llvmlite_version'] and identity['host_cpu_name']
    assert identity['llvm_version'] and isinstance(identity['using_svml'], bool)
    assert 'host_cpu_features' in identity and identity['fastmath'] is False
    monkeypatch.setattr(sampler, 'IMPLEMENTATION_SHA256', 'changed native bytes')
    assert sampler.spherical_gray8_sampling_identity()['implementation_sha256'] != identity['implementation_sha256']


def test_kernel_is_serial_nogil_and_keeps_float_order():
    assert sampler._compiled_render.targetoptions['nogil'] is True
    assert sampler._compiled_render.targetoptions['fastmath'] is False
    assert not sampler._compiled_render.targetoptions.get('parallel', False)


def test_native_sampler_source_change_changes_actual_image_cache_identity(tmp_path, monkeypatch):
    from tests.test_sam_native_crop_feed import case
    from tests.test_sam_view_image_cache import context_for, demand
    source, physical = case('spherical')
    context = context_for(tmp_path, source)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *args: None)
    shape = (physical.num_slices, physical.src_h, physical.src_w)
    plan = demand(shape, {physical.num_slices//2: (13, 17, 29, 35)})
    try:
        first = context.image_provider(physical, shape, plan)
        monkeypatch.setattr(sampler, 'IMPLEMENTATION_SHA256', 'f'*64)
        second = context.image_provider(physical, shape, plan)
        assert first.identity_sha256 != second.identity_sha256
        assert first.path != second.path
    finally:
        context.close()
