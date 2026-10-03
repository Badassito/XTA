"""Radial voxel-cell coverage through actual TTA grids, without CUDA admission."""
from dataclasses import replace
import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from XTA import cylindrical_projection as cp, geometry
from XTA.config import resolve_tilted_view_groups
from tests.reference_backends.radial import pull_radial_chunk


def views(shape, base, *, patch=8, minimum=.7, tilt=None):
    return [view for view in geometry.get_view_infos(*shape, cartesian_views=(),
        radial_views=((('tilted_' if tilt else '') + base),), radial_patch_size=patch,
        radial_min_radius=minimum,
        tilt_groups=resolve_tilted_view_groups([f'{base}:{abs(tilt)}:both']) if tilt else ())
        if view.family == 'radial' and (not tilt or view.tilt_angle_deg == tilt)]


def analytic_domain(view, shape):
    """Independent centered source geometry; height owns half-open voxel cells."""
    work = np.array((view.full_t, view.full_h, view.full_w), np.float64)
    coords = [(np.arange(n)*2+1-n)*w/(2*n) + (w-1)/2 for n, w in zip(shape, work)]
    t, y, x = np.meshgrid(*coords, indexing='ij')
    base = view.radial_base_view
    stack, py, px, length = {'transverse': (t, y, x, view.full_t),
                             'sagittal': (y, t, x, view.full_h),
                             'coronal': (x, t, y, view.full_w)}[base]
    rho = np.hypot(px-view.center_x, py-view.center_y)
    height = stack.copy()
    if view.radial_tilted_source:
        height -= math.tan(math.radians(view.tilt_angle_deg)) * (
            py-view.center_y if view.tilt_direction == 'vertical' else px-view.center_x)
    return ((rho >= view.radial_min_radius) & (rho <= view.radial_max_radius)
            & (height >= -.5) & (height < length-.5))


def project_cpu(data, view, shape, *, numeric=False, direct=False):
    radii = np.asarray(geometry.radial_global_radii(view), np.float64)
    if direct:
        return np.stack([cp._pull_radial_chunk_compiled(data, view, radii, shape, z, 0,
                          shape[1]*shape[2], scalar_max=numeric).reshape(shape[1:]) for z in range(shape[0])])
    plan, _ = cp._radial_plane_plan(view, radii, shape)
    centers, ideal, sampled, rows, columns, length, vertical = cp._radial_projection_metadata(view, data.shape, shape, plan)
    boxes = np.zeros((len(data), 4), np.int64)
    return cp._project_radial_block(data, plan.shell_index, plan.column_offsets, plan.native_columns,
        sampled, rows, columns, centers, ideal, length, view.radial_height_origin, view.src_h,
        plan.base_id, vertical, plan.plane_shape[1], shape[1], shape[2], 0, shape[0], boxes, False, numeric)


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('shape', ((11, 15, 19), (5, 7, 8), (7, 9, 11)))
@pytest.mark.parametrize('processing', ((3, 4), (8, 8)))
def test_reduced_and_native_all_foreground_cover_height_caps_without_radius_growth(base, shape, processing):
    built = views((7, 9, 11), base)
    results = [np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)]
    for view in built:
        data = np.ones((view.num_slices, *processing), np.uint8)
        for direct, target in zip((False, True), results):
            target |= project_cpu(data, view, shape, direct=direct)
    wanted = analytic_domain(built[0], shape)
    for result in results:
        np.testing.assert_array_equal(result, wanted.astype(np.uint8))


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('tilt', (-45., -23., 23., 45.))
def test_tilted_half_cell_height_limits_match_exact_source_domain_and_score_support(base, tilt):
    shape = (11, 15, 19)
    built = views((7, 9, 11), base, tilt=tilt)
    for direction in ('vertical', 'horizontal'):
        group = [view for view in built if view.tilt_direction == direction]
        binary, numeric = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
        for view in group:
            data = np.full((view.num_slices, 4, 4), 173, np.uint8)
            binary |= project_cpu(data, view, shape)
            numeric = np.maximum(numeric, project_cpu(data, view, shape, numeric=True, direct=True))
        wanted = analytic_domain(group[0], shape)
        np.testing.assert_array_equal(binary, wanted.astype(np.uint8))
        np.testing.assert_array_equal(numeric != 0, wanted)
        assert np.all(numeric[wanted] == 173)


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_rational_restoration_retains_exact_closed_annulus_boundary_and_inner_hole(base):
    shape = (21, 21, 21)
    built = views((7, 7, 7), base, minimum=1.)
    combined = np.zeros(shape, np.uint8)
    for view in built:
        combined |= project_cpu(np.ones((view.num_slices, 8, 8), np.uint8), view, shape)
    indices = np.arange(21) - 10
    a, b = np.meshgrid(indices, indices, indexing='ij')
    exact_plane = ((a*a+b*b) >= 9) & ((a*a+b*b) <= 81)
    if base == 'transverse':
        wanted = np.broadcast_to(exact_plane, shape)
    elif base == 'sagittal':
        wanted = np.broadcast_to(exact_plane[:, None, :], shape)
    else:
        wanted = np.broadcast_to(exact_plane[:, :, None], shape)
    np.testing.assert_array_equal(combined.astype(bool), wanted)


def test_closed_radius_roundoff_is_local_to_each_limit_and_never_expands_the_hole():
    eps = np.finfo(np.float64).eps
    minimum, maximum = 1e-12, 1e6
    values = np.array((0., minimum*(1-32*eps), minimum*(1-4*eps), minimum,
                       maximum, maximum*(1+4*eps), maximum*(1+32*eps)))
    np.testing.assert_array_equal(cp._inside_closed_radial_limits(values, minimum, maximum),
                                 (False, False, True, True, True, True, False))


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('processing', ((4, 4), (8, 8)))
def test_sharp_shell_gaps_localized_patches_and_numeric_max_have_the_same_oracle_footprint(base, processing):
    rng = np.random.default_rng(425)
    shape = (11, 15, 19)
    for view in views((7, 9, 11), base, patch=8, tilt=23.)[::3]:
        data = np.zeros((view.num_slices, *processing), np.uint8)
        data[::2, 1:3, 1:3] = rng.integers(10, 255, (len(data[::2]), 2, 2), dtype=np.uint8)
        radii = np.asarray(geometry.radial_global_radii(view))
        expected = np.stack([pull_radial_chunk(data, view, radii, shape, z, 0, shape[1]*shape[2],
                            scalar_max=True).reshape(shape[1:]) for z in range(shape[0])])
        for direct in (False, True):
            actual = project_cpu(data, view, shape, numeric=True, direct=direct)
            binary = project_cpu(data, view, shape, direct=direct)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(binary != 0, expected != 0)


@pytest.mark.parametrize('size,processing,angle', ((7, 3, 0.), (8, 4, 31.), (9, 5, 120.), (8, 8, 31.)))
def test_actual_model_affine_inverse_then_project_matches_native_restored_raster(size, processing, angle):
    view = views((7, 17, 19), 'transverse', patch=size)[0]
    view = geometry.expand_views_into_tta_variants((view,), (angle,))[0]
    native = np.zeros((size, size), np.uint8)
    native[1:size-1, size//2] = 1
    native[size//2, 1:size-1] = 1
    aff = geometry.build_affine(view=view.name, src_w=size, src_h=size, out_size=processing,
                                angle_deg=angle, pad_mode=view.pad_mode)
    model = cv2.warpAffine(native, aff.M_src_to_out, (processing, processing), flags=cv2.INTER_NEAREST)
    restore = geometry.output_to_view_processing_affine(view, aff.M_out_to_src, processing)
    h, w = geometry.view_processing_plane_shape(view, processing)
    accumulated = cv2.warpAffine(model, restore, (w, h), flags=cv2.INTER_NEAREST)
    canonical = geometry.build_affine(view=view.name, src_w=size, src_h=size, out_size=processing,
                                     angle_deg=0., pad_mode=view.pad_mode)
    restored_native = (cv2.warpAffine(accumulated, canonical.M_out_to_src, (size, size), flags=cv2.INTER_NEAREST)
                       if (h, w) != (size, size) else accumulated)
    a = np.broadcast_to(accumulated, (view.num_slices, h, w)).copy()
    b = np.broadcast_to(restored_native, (view.num_slices, size, size)).copy()
    np.testing.assert_array_equal(project_cpu(a, view, (11, 21, 25)), project_cpu(b, view, (11, 21, 25)))


@pytest.mark.parametrize('tile_size,stride', ((3, 1), (3, 3), (4, 2), (4, 4)))
@pytest.mark.parametrize('angle', (0., 31., 120.))
def test_actual_dense_ring_tiles_restore_full_parent_without_new_seam_holes(tmp_path, tile_size, stride, angle):
    # The real Radial factory uses --imgsz as native intrinsic patch size. Tile
    # magnification changes only its model-to-parent footprint, not shell axes.
    view = views((7, 17, 19), 'transverse', patch=8)[0]
    view = geometry.expand_views_into_tta_variants((view,), (angle,))[0]
    aff = geometry.build_affine(view=view.name, src_w=8, src_h=8, out_size=8,
                                angle_deg=angle, pad_mode=view.pad_mode)
    aug = geometry.AugJob(aug_id='fixture', angle_deg=angle, aff=aff, meta_path=tmp_path/'aug.json')
    config = geometry.TileConfig(config_id='fixture', tile_size=tile_size, tile_stride=stride)
    jobs = geometry.build_dense_tile_jobs_for_aug(view, aug, config, 8, tmp_path)
    native = np.ones((8, 8), np.uint8)
    parent = np.zeros_like(native)
    for job in jobs:
        model = cv2.warpAffine(native, job.M_src_to_out, (8, 8), flags=cv2.INTER_NEAREST)
        restored = geometry.output_to_view_processing_affine(view, job.M_out_to_src, 8)
        parent |= cv2.warpAffine(model, restored, (8, 8), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(parent, native)
    a = np.broadcast_to(parent, (view.num_slices, 8, 8)).copy()
    b = np.ones_like(a)
    np.testing.assert_array_equal(project_cpu(a, view, (11, 21, 25)), project_cpu(b, view, (11, 21, 25)))


@pytest.mark.parametrize('tile_size,stride', ((3, 1), (3, 3), (4, 2), (4, 4)))
def test_unrotated_magnified_ring_tiles_preserve_a_real_one_pixel_gap(tmp_path, tile_size, stride):
    view = views((7, 17, 19), 'transverse', patch=8)[0]
    aff = geometry.build_affine(view=view.name, src_w=8, src_h=8, out_size=8,
                                angle_deg=0., pad_mode=view.pad_mode)
    aug = geometry.AugJob(aug_id='fixture', angle_deg=0., aff=aff, meta_path=tmp_path/'aug.json')
    config = geometry.TileConfig(config_id='fixture', tile_size=tile_size, tile_stride=stride)
    native = np.ones((8, 8), np.uint8)
    native[:, 4] = 0
    parent = np.zeros_like(native)
    for job in geometry.build_dense_tile_jobs_for_aug(view, aug, config, 8, tmp_path):
        model = cv2.warpAffine(native, job.M_src_to_out, (8, 8), flags=cv2.INTER_NEAREST)
        parent |= cv2.warpAffine(model, job.M_out_to_src, (8, 8), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(parent, native)
