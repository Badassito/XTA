"""Categorical shell CUDA output agrees with the native CPU geometry oracle."""
from __future__ import annotations

import numpy as np
import pytest

from XTA.geometry import ViewInfo
from XTA.cylindrical_geometry import render_shell_frame as render_radial_cpu
from XTA.spherical_geometry import cube_rotation, render_shell_frame as render_spherical_cpu
from XTA.pta_cuda_shells import _inverse_affine, render_shell_categorical_pair


def _gpu():
    torch = pytest.importorskip('torch')
    pytest.importorskip('cupy')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for resident shell sampling')
    return torch


def _sources():
    rng = np.random.default_rng(9162)
    mask = (rng.random((23, 27, 29)) < 0.22).astype(np.uint8)
    coverage = (rng.random(mask.shape) < 0.67).astype(np.uint8)
    return mask, coverage


def _radial(base, *, tilted=False):
    return ViewInfo(
        name=f'radial_{base}', family='radial', num_slices=1,
        src_h=17, src_w=19, pad_mode='pad', full_t=23, full_h=27, full_w=29,
        center_x=14.0 if base != 'sagittal' else 14.0,
        center_y=13.0 if base == 'transverse' else 11.0,
        radial_base_view=base, radial_radii=(8.25,),
        radial_arc_origin=3.0, radial_height_origin=5,
        radial_tilted_source=tilted,
        tilt_direction='vertical' if tilted else '',
        tilt_angle_deg=31.0 if tilted else 0.0,
    )


def _spherical(face, *, tilted=False):
    return ViewInfo(
        name=f'spherical_{face}', family='spherical', num_slices=1,
        src_h=17, src_w=19, pad_mode='pad', full_t=23, full_h=27, full_w=29,
        spherical_face=face, spherical_face_intervals=18,
        spherical_u_origin=1, spherical_v_origin=2,
        spherical_radii=(9.25,),
        spherical_tilted_source=tilted,
        spherical_rotation_xyz=cube_rotation('vertical', 23.0) if tilted else cube_rotation(),
    )


@pytest.mark.parametrize('base,tilted', [
    ('transverse', False), ('sagittal', False), ('coronal', False),
    ('transverse', True), ('sagittal', True), ('coronal', True),
])
def test_radial_native_identity_matches_cpu(base, tilted):
    torch = _gpu()
    mask, coverage = _sources()
    view = _radial(base, tilted=tilted)
    expected_mask = render_radial_cpu(mask, view, 0, categorical=True)
    expected_coverage = render_radial_cpu(coverage, view, 0, categorical=True)
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    actual_mask, actual_coverage = render_shell_categorical_pair(
        torch.as_tensor(mask, device='cuda'), torch.as_tensor(coverage, device='cuda'),
        view, 0, M_src_to_out=identity, out_h=view.src_h, out_w=view.src_w,
        stream=torch.cuda.current_stream(),
    )
    np.testing.assert_array_equal(actual_mask.cpu().numpy(), expected_mask)
    np.testing.assert_array_equal(actual_coverage.cpu().numpy(), expected_coverage)


@pytest.mark.parametrize('face,tilted', [(0, False), (1, False), (2, False),
                                          (3, False), (4, True), (5, True)])
def test_spherical_native_identity_matches_cpu(face, tilted):
    torch = _gpu()
    mask, coverage = _sources()
    view = _spherical(face, tilted=tilted)
    expected_mask = render_spherical_cpu(mask, view, 0, categorical=True)
    expected_coverage = render_spherical_cpu(coverage, view, 0, categorical=True)
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    actual_mask, actual_coverage = render_shell_categorical_pair(
        torch.as_tensor(mask, device='cuda'), torch.as_tensor(coverage, device='cuda'),
        view, 0, M_src_to_out=identity, out_h=view.src_h, out_w=view.src_w,
        stream=torch.cuda.current_stream(),
    )
    np.testing.assert_array_equal(actual_mask.cpu().numpy(), expected_mask)
    np.testing.assert_array_equal(actual_coverage.cpu().numpy(), expected_coverage)


def test_shell_affine_translation_and_padding():
    torch = _gpu()
    cv2 = pytest.importorskip('cv2')
    mask, coverage = _sources()
    view = _radial('transverse')
    forward = np.array([[1, 0, 3], [0, 1, -2]], dtype=np.float32)
    expected_mask = cv2.warpAffine(render_radial_cpu(mask, view, 0, categorical=True),
                                   forward, (view.src_w, view.src_h), flags=cv2.INTER_NEAREST)
    expected_coverage = cv2.warpAffine(render_radial_cpu(coverage, view, 0, categorical=True),
                                       forward, (view.src_w, view.src_h), flags=cv2.INTER_NEAREST)
    actual_mask, actual_coverage = render_shell_categorical_pair(
        torch.as_tensor(mask, device='cuda'), torch.as_tensor(coverage, device='cuda'),
        view, 0, M_src_to_out=forward, out_h=view.src_h, out_w=view.src_w,
    )
    np.testing.assert_array_equal(actual_mask.cpu().numpy(), expected_mask)
    np.testing.assert_array_equal(actual_coverage.cpu().numpy(), expected_coverage)


@pytest.mark.parametrize('angle,scale,offset', [
    (17.0, 1.0, (2.0, -3.0)),
    (-31.0, 0.8, (-6.0, 4.0)),
    (0.0, 1.4, (1.0, 2.0)),
    (0.0, 2.0 / 3.0, (0.0, 0.0)),
])
def test_forward_affine_inverse_matches_opencv_nearest(angle, scale, offset):
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(571)
    source = rng.integers(0, 2, size=(61, 67), dtype=np.uint8)
    forward = cv2.getRotationMatrix2D((33.0, 30.0), angle, scale)
    forward[:, 2] += offset
    forward = forward.astype(np.float32)
    expected = cv2.warpAffine(source, forward, (71, 59), flags=cv2.INTER_NEAREST)
    m00, m01, m02, m10, m11, m12 = _inverse_affine(None, forward)
    y, x = np.indices(expected.shape, dtype=np.float64)
    sx = np.rint(m00 * x + m01 * y + m02).astype(np.int64)
    sy = np.rint(m10 * x + m11 * y + m12).astype(np.int64)
    valid = (sx >= 0) & (sx < source.shape[1]) & (sy >= 0) & (sy < source.shape[0])
    actual = np.zeros_like(expected)
    actual[valid] = source[sy[valid], sx[valid]]
    np.testing.assert_array_equal(actual, expected)


def test_opencv_half_pixel_affine_uses_even_ties():
    cv2 = pytest.importorskip('cv2')
    source = np.arange(8, dtype=np.uint8).reshape(1, 8)
    forward = np.array([[1, 0, -0.5], [0, 1, 0]], dtype=np.float32)
    actual = cv2.warpAffine(source, forward, (6, 1), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(actual[0], [0, 2, 2, 4, 4, 6])
    inverse = _inverse_affine(None, forward)
    np.testing.assert_array_equal(np.rint(np.arange(6) + inverse[2]), actual[0])


@pytest.mark.parametrize('family', ['radial', 'spherical'])
@pytest.mark.parametrize('affine', ['half_pixel', 'tile_scale'])
def test_shell_affine_ties_match_cpu_mask_and_coverage(family, affine):
    torch = _gpu()
    cv2 = pytest.importorskip('cv2')
    mask, coverage = _sources()
    view = _radial('transverse', tilted=True) if family == 'radial' else _spherical(2, tilted=True)
    native = render_radial_cpu if family == 'radial' else render_spherical_cpu
    if affine == 'half_pixel':
        forward = np.array([[1, 0, -0.5], [0, 1, -0.5]], dtype=np.float32)
    else:
        # Representative 1536 -> 1024 tile scale, translated onto a crop.
        forward = np.array([[2.0 / 3.0, 0, 3], [0, 2.0 / 3.0, 2]], dtype=np.float32)
    expected_mask = cv2.warpAffine(native(mask, view, 0, categorical=True), forward,
                                   (view.src_w, view.src_h), flags=cv2.INTER_NEAREST)
    expected_coverage = cv2.warpAffine(native(coverage, view, 0, categorical=True), forward,
                                       (view.src_w, view.src_h), flags=cv2.INTER_NEAREST)
    actual_mask, actual_coverage = render_shell_categorical_pair(
        torch.as_tensor(mask, device='cuda'), torch.as_tensor(coverage, device='cuda'),
        view, 0, M_src_to_out=forward, out_h=view.src_h, out_w=view.src_w,
    )
    np.testing.assert_array_equal(actual_mask.cpu().numpy(), expected_mask)
    np.testing.assert_array_equal(actual_coverage.cpu().numpy(), expected_coverage)
