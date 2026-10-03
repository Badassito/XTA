"""Independent physical-support checks of completed native projection routes.

The all-foreground oracle is analytic geometry, not the production projector or
an older scatter implementation. Tests deliberately use reduced model grids.
"""
from __future__ import annotations

from dataclasses import replace
import itertools
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.outputs import _read_layer_slice_in_output_shape
from XTA.interpolation import RawBBoxMaskStore


@pytest.fixture(autouse=True)
def cpu_projection_environment(monkeypatch):
    for name in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT',
                 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', 'YOLO_TTA_GPU_SLICE_LABELING'):
        monkeypatch.setenv(name, '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')


def centered_work_points(work, target):
    # Keep exact integer numerators until division. This intentionally avoids
    # the cancellation-prone scale-then-subtract implementation under review.
    axes = [np.asarray([(2*i+1-out)*inside for i in range(out)], dtype=np.longdouble)
            / np.longdouble(2*out) for inside, out in zip(work, target)]
    return np.meshgrid(*axes, indexing='ij')


def radial_views(base, *, work=(7, 9, 11), minimum=1., size=8):
    return [view for view in geometry.get_view_infos(*work, cartesian_views=(),
        radial_views=(base,), radial_min_radius=minimum, radial_patch_size=size)
        if view.family == 'radial']


def spherical_views(*, work=(7, 9, 11), minimum=.7, size=8, rotation=None):
    from XTA.spherical_geometry import build_spherical_view_infos
    result = build_spherical_view_infos(*work, targets=('transverse',),
        min_radius=minimum, patch_size=size, tilted_views=())
    if rotation is not None:
        result = [replace(view, spherical_rotation_xyz=tuple(np.asarray(rotation).reshape(-1)))
                  for view in result]
    return result


def analytic_annulus(view, target, *, spherical=False):
    tt, yy, xx = centered_work_points((view.full_t, view.full_h, view.full_w), target)
    if spherical:
        square = xx*xx + yy*yy + tt*tt
        minimum, maximum = view.spherical_min_radius, view.spherical_max_radius
    else:
        py, px = {'transverse': (yy, xx), 'sagittal': (tt, xx), 'coronal': (tt, yy)}[view.radial_base_view]
        square = px*px + py*py
        minimum, maximum = view.radial_min_radius, view.radial_max_radius
    return (square >= np.longdouble(minimum)**2) & (square <= np.longdouble(maximum)**2)


def direct_native(data, view, target, root):
    path = Path(root)/'direct-native.dat'
    projected = assembly.project_view_volume_to_orthogonal_volume(data, view, path,
        'independent native coverage', out_shape_tyx=target, workers=1,
        prefer_memory=False, reserve_bytes=0)
    try:
        return np.stack([_read_layer_slice_in_output_shape(projected, target, z)
                         for z in range(target[0])])
    finally:
        if isinstance(projected, np.memmap):
            projected._mmap.close()
        path.unlink(missing_ok=True)


def published_native(data, view, target, root):
    previous = assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(target)
    try:
        reference = assembly.materialize_nrrd_view_layer(data, model_name='coverage-witness',
            view=view, source='fullframe', mask_kind='yolo', stage='pre_interpolation',
            temp_dir=Path(root), workers=1, known_has_foreground=bool(data.any()),
            submit_to_sink=False, force_path_backed_store=True, internal_packbits_store=True,
            emit_empty=True)
    finally:
        assembly.set_final_source_output_shape(previous)
    assert reference is not None
    assert tuple(reference.shape) == tuple(target), 'A completed nonlinear layer retained processing geometry'
    stored = RawBBoxMaskStore.open(reference.path)
    try:
        return np.stack([_read_layer_slice_in_output_shape(stored, target, z)
                         for z in range(target[0])])
    finally:
        close = getattr(stored, 'close', None)
        if callable(close):
            close()


def model_masks(view, *, gap_radius=None):
    result = np.ones((view.num_slices, 3, 4), np.uint8)
    if gap_radius is not None:
        radii = view.spherical_radii if view.family == 'spherical' else view.radial_radii
        for frame, radius in enumerate(radii):
            if radius == gap_radius:
                result[frame] = 0
    return result


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_radial_reduced_model_grid_fills_native_annulus_including_height_caps(tmp_path, base):
    target = (11, 15, 19)
    views = radial_views(base)
    expected = analytic_annulus(views[0], target)
    union = np.zeros(target, np.uint8)
    for index, view in enumerate(views):
        data = model_masks(view)
        original = data.copy()
        actual = published_native(data, view, target, tmp_path/str(index))
        union |= actual
        np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(union != 0, expected,
        err_msg=f'{base}: full-height sampled cells must cover native end caps without radius dilation')


@pytest.mark.parametrize('work,target', [((7, 7, 7), (21, 21, 21)),
                                       ((7, 9, 11), (21, 27, 33))])
@pytest.mark.parametrize('angle', [0., 31., -23.])
def test_all_spherical_faces_cover_exact_physical_annulus_after_model_downsampling(tmp_path, work, target, angle):
    from XTA.spherical_geometry import cube_rotation
    views = spherical_views(work=work, rotation=cube_rotation('vertical', angle))
    expected = analytic_annulus(views[0], target, spherical=True)
    union = np.zeros(target, np.uint8)
    for index, view in enumerate(views):
        union |= published_native(model_masks(view), view, target, tmp_path/str(index))
    np.testing.assert_array_equal(union != 0, expected,
        err_msg='Six face/patch publications must fill physical shell support without growing annulus')


@pytest.mark.parametrize('family', ['radial', 'spherical'])
def test_declared_black_radius_shell_leaves_sharp_native_gap_and_save_route_agrees(tmp_path, family):
    target = (11, 15, 19)
    views = radial_views('transverse') if family == 'radial' else spherical_views()
    radii = sorted({float(radius) for view in views for radius in
                    (view.radial_radii if family == 'radial' else view.spherical_radii)})
    assert len(radii) >= 3
    gap = radii[len(radii)//2]
    tt, yy, xx = centered_work_points((7, 9, 11), target)
    distance = np.sqrt(tt*tt+yy*yy+xx*xx if family == 'spherical' else yy*yy+xx*xx)
    closest = np.argmin(np.abs(distance[..., None]-np.asarray(radii, dtype=np.longdouble)), axis=-1)
    expected = analytic_annulus(views[0], target, spherical=family == 'spherical') & (closest != radii.index(gap))
    direct = np.zeros(target, np.uint8)
    published = np.zeros(target, np.uint8)
    for index, view in enumerate(views):
        data = model_masks(view, gap_radius=gap)
        root = tmp_path/str(index)
        root.mkdir(parents=True)
        direct |= direct_native(data, view, target, root)
        published |= published_native(data, view, target, root)
    np.testing.assert_array_equal(direct, published)
    np.testing.assert_array_equal(published != 0, expected,
        err_msg='A black sampled shell cannot be filled by densification or double restoration')


@pytest.mark.parametrize('face', range(6))
def test_single_rotated_spherical_face_fills_only_its_closed_physical_wedge(tmp_path, face):
    from XTA.qsc import QSC_FACE_BASES
    from XTA.spherical_geometry import cube_rotation
    target = (11, 15, 19)
    rotation = cube_rotation('horizontal', 31.)
    views = [view for view in spherical_views(rotation=rotation) if view.spherical_face == face]
    expected = analytic_annulus(views[0], target, spherical=True)
    tt, yy, xx = centered_work_points((7, 9, 11), target)
    local = np.stack((xx, yy, tt), axis=-1) @ np.asarray(rotation, dtype=np.longdouble).reshape(3, 3)
    normal, right, up = (local @ np.asarray(axis, dtype=np.longdouble) for axis in QSC_FACE_BASES[face])
    tolerance = np.longdouble(8*np.finfo(np.float64).eps) * np.max(np.abs(local), axis=-1)
    expected &= (normal > 0) & (normal+tolerance >= np.maximum(np.abs(right), np.abs(up)))
    union = np.zeros(target, np.uint8)
    for index, view in enumerate(views):
        union |= published_native(model_masks(view), view, target, tmp_path/str(index))
    np.testing.assert_array_equal(union != 0, expected,
        err_msg='A face publication must neither leave holes in its wedge nor fill another face')


@pytest.mark.parametrize('family', ['radial', 'spherical'])
@pytest.mark.parametrize('split', ['frames', 'model_columns'])
def test_disjoint_input_shards_or_to_dense_physical_coverage_without_patch_seams(tmp_path, family, split):
    target = (11, 15, 19)
    views = radial_views('transverse') if family == 'radial' else spherical_views()
    expected = analytic_annulus(views[0], target, spherical=family == 'spherical')
    union = np.zeros(target, np.uint8)
    for index, view in enumerate(views):
        first, second = model_masks(view), model_masks(view)
        if split == 'frames':
            first[1::2] = 0
            second[::2] = 0
        else:
            first[:, :, 2:] = 0
            second[:, :, :2] = 0
        union |= published_native(first, view, target, tmp_path/f'{index}-first')
        union |= published_native(second, view, target, tmp_path/f'{index}-second')
    np.testing.assert_array_equal(union != 0, expected,
        err_msg='Split source contributions must retain every native cell across frame/tile/patch seams')


@pytest.mark.parametrize('family,base', [('radial', 'transverse'), ('radial', 'sagittal'),
                                       ('radial', 'coronal'), ('spherical', 'transverse')])
def test_binary_native_coverage_and_numeric_confidence_share_same_support(tmp_path, family, base):
    from XTA.confidence_projection import score_projection_reader
    target = (11, 15, 19)
    views = radial_views(base) if family == 'radial' else spherical_views()
    radii = sorted({float(radius) for view in views for radius in
                    (view.radial_radii if family == 'radial' else view.spherical_radii)})
    gap = radii[len(radii)//2]
    mask_union = np.zeros(target, np.uint8)
    score_union = np.zeros(target, np.uint8)
    for index, view in enumerate(views):
        mask = model_masks(view, gap_radius=gap)
        score = mask*np.uint8(173)
        mask_union |= published_native(mask, view, target, tmp_path/f'{index}-binary')
        with score_projection_reader(score, view, target, tmp_path/f'{index}-scores',
                                     memory_bytes=8*1024**2) as read:
            projected = np.stack([read(z) for z in range(target[0])])
        score_union = np.maximum(score_union, projected)
    np.testing.assert_array_equal(score_union > 0, mask_union != 0)
    np.testing.assert_array_equal(score_union, mask_union*np.uint8(173))


def tilted_view(base, direction, angle, work=(7, 9, 11)):
    from XTA.config import TiltedViewGroup
    return next(view for view in geometry.get_view_infos(*work, cartesian_views=(),
        tilt_groups=(TiltedViewGroup((base,), (abs(angle),), (direction,)),))
        if view.family == 'tilted' and view.tilt_angle_deg == angle)


def tilted_physics_oracle(data, view, target, *, numeric=False):
    """Scalar inverse geometry from public view/input descriptors only."""
    work = (view.full_t, view.full_h, view.full_w)
    stack, vertical, horizontal = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2),
                                    'coronal': (2, 0, 1)}[view.tilt_base_view]
    shear = vertical if view.tilt_direction == 'vertical' else horizontal
    tangent = math.tan(math.radians(view.tilt_angle_deg))
    if data.shape[1:] == (view.src_h, view.src_w):
        sy, sx, ty, tx = 1., 1., 0., 0.
    else:
        assert view.pad_mode == 'clamp' and data.shape[1] == data.shape[2]
        size = data.shape[1]
        # The producer contract stores float32 coefficients. Derive that
        # simple zero-angle affine independently, without the inverse helper.
        sy, sx = float(np.float32(size/view.src_h)), float(np.float32(size/view.src_w))
        ty = float(np.float32((size-1)/2-(size/view.src_h)*(view.src_h-1)/2))
        tx = float(np.float32((size-1)/2-(size/view.src_w)*(view.src_w-1)/2))
    result = np.zeros(target, np.uint8)
    for position in np.ndindex(target):
        contributors = []
        for index, inside, outside in zip(position, work, target):
            if outside < inside:
                contributors.append(range(index*inside//outside,
                                          ((index+1)*inside+outside-1)//outside))
            else:
                contributors.append(((index+.5)*inside/outside-.5,))
        for point in itertools.product(*contributors):
            local = point[stack]-tangent*(point[shear]-(work[shear]-1)/2)-view.tilt_frame_start
            if not -.5 <= local < view.num_slices-.5:
                continue
            source_frame = min(view.num_slices-1, max(0, math.floor(local+.5)))
            native_y = min(view.src_h-1, max(0., point[vertical]))
            native_x = min(view.src_w-1, max(0., point[horizontal]))
            row, col = sy*native_y+ty, sx*native_x+tx
            if not (-.5 <= row < data.shape[1]-.5 and -.5 <= col < data.shape[2]-.5):
                continue
            row = min(data.shape[1]-1, max(0, math.floor(row+.5)))
            col = min(data.shape[2]-1, max(0, math.floor(col+.5)))
            value = int(data[source_frame, row, col])
            result[position] = max(int(result[position]), value if numeric else int(value != 0))
    return result


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('angle', [-23., 23.])
def test_tilted_full_foreground_fills_inverse_physical_fov_without_scatter_holes(tmp_path, base, direction, angle):
    target = (11, 15, 19)
    view = tilted_view(base, direction, angle)
    data = np.ones((view.num_slices, 5, 5), np.uint8)
    expected = tilted_physics_oracle(data, view, target)
    assert expected.any() and not expected.all()  # Unsampled wedge remains background.
    direct = direct_native(data, view, target, tmp_path)
    published = published_native(data, view, target, tmp_path)
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(published, expected)


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('target', [(11, 15, 19), (4, 5, 6)])
def test_tilted_sharp_model_gaps_contractions_and_confidence_are_physical(tmp_path, base, target):
    from XTA.confidence_projection import score_projection_reader
    view = tilted_view(base, 'horizontal', 23.)
    data = np.ones((view.num_slices, 5, 5), np.uint8)
    data[:, :, 2] = 0
    data[1::3] = 0
    expected = tilted_physics_oracle(data, view, target)
    actual = published_native(data, view, target, tmp_path/'published')
    np.testing.assert_array_equal(actual, expected)
    score = np.broadcast_to(np.arange(view.num_slices, dtype=np.uint8)[:, None, None]+np.uint8(100), data.shape).copy()
    score *= data
    score_expected = tilted_physics_oracle(score, view, target, numeric=True)
    with score_projection_reader(score, view, target, tmp_path/'scores', memory_bytes=8*1024**2) as read:
        score_actual = np.stack([read(z) for z in range(target[0])])
    np.testing.assert_array_equal(score_actual, score_expected)
    np.testing.assert_array_equal(score_actual > 0, expected != 0)


def test_single_spherical_model_cell_localizes_inside_one_face_instead_of_flooding_it(tmp_path):
    target = (21, 21, 21)
    source_view = next(view for view in spherical_views(work=(7, 7, 7), minimum=1.)
                       if view.spherical_face == 0)
    intervals = source_view.spherical_face_intervals
    view = replace(source_view, spherical_u_origin=0, spherical_v_origin=0,
                   src_h=intervals+1, src_w=intervals+1, spherical_patch_size=intervals+1)
    data = np.zeros((view.num_slices, 2, 2), np.uint8)
    data[:, 1, 1] = 1
    actual = published_native(data, view, target, tmp_path)
    tt, yy, xx = centered_work_points((7, 7, 7), target)
    face = analytic_annulus(view, target, spherical=True) & (xx > 0) & (xx >= np.maximum(np.abs(yy), np.abs(tt)))
    positive = face & (yy > xx/4) & (tt < -xx/4)
    negative = face & ((yy < -xx/4) | (tt > xx/4))
    assert positive.any() and negative.any()
    assert np.all(actual[positive] == 1)
    assert not actual[negative].any()
    assert not actual[~face].any()


def azimuthal_view(base, *, tilted=False, direction='vertical', angle=23.,
                   step=30., native_raster=9, work=(7, 9, 11)):
    from XTA.config import TiltedViewGroup
    token = ('tilted_' if tilted else '')+base
    views = geometry.get_view_infos(*work, cartesian_views=(), azimuthal_views=(token,),
        azimuthal_azimuth_angles=(step,), azimuthal_native_raster=native_raster,
        tilt_groups=(TiltedViewGroup((base,), (abs(angle),), (direction,)),) if tilted else ())
    return next(view for view in views if view.family == 'azimuthal'
                and (not tilted or view.tilt_angle_deg == angle))


def azimuthal_physical_fov(view, target):
    tt, yy, xx = centered_work_points((view.full_t, view.full_h, view.full_w), target)
    centers = (tt, yy, xx)
    stack, vertical, horizontal = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2),
                                    'coronal': (2, 0, 1)}[view.azimuthal_base_view]
    # Radius R denotes the last radial sample center. The inherited physical
    # pixel-circle boundary is R+.5; explicit Radial/Spherical annuli differ.
    fov = centers[vertical]**2 + centers[horizontal]**2 <= np.longdouble(view.roi_radius+.5)**2
    work = (view.full_t, view.full_h, view.full_w)
    local = centers[stack]+np.longdouble((work[stack]-1)/2)
    if view.azimuthal_tilted_source:
        shear = vertical if view.tilt_direction == 'vertical' else horizontal
        local -= np.longdouble(math.tan(math.radians(view.tilt_angle_deg)))*centers[shear]
        local -= view.tilt_frame_start
    return fov & (local >= -.5) & (local < work[stack]-.5)


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('native_raster', [4, 9])
@pytest.mark.parametrize('tilted', [False, True])
def test_dense_and_coarse_azimuthal_model_masks_cover_only_native_physical_fov(tmp_path, base, native_raster, tilted):
    view = azimuthal_view(base, tilted=tilted, direction='horizontal', angle=-23., native_raster=native_raster)
    target = (11, 15, 19)
    data = np.ones((view.num_slices, 5, 5), np.uint8)
    expected = azimuthal_physical_fov(view, target)
    direct = direct_native(data, view, target, tmp_path)
    published = published_native(data, view, target, tmp_path)
    np.testing.assert_array_equal(direct != 0, expected)
    np.testing.assert_array_equal(published != 0, expected)


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_nonzero_nonuniform_azimuthal_plan_keeps_angular_background_localized(tmp_path, base):
    view = azimuthal_view(base, step=5.)
    # The nominal spacing is below the declared coverage bound, so these are
    # the exact reconstruction-plan angles (no virtual densification needed).
    view = replace(view, azimuths_deg=(17., 18., 19., 20.), num_slices=4)
    data = np.zeros((4, 5, 5), np.uint8)
    data[0] = 1
    target = (11, 15, 19)
    expected = azimuthal_physical_fov(view, target)
    tt, yy, xx = centered_work_points((7, 9, 11), target)
    vertical, horizontal = {'transverse': (yy, xx), 'sagittal': (tt, xx), 'coronal': (tt, yy)}[base]
    theta = np.mod(np.degrees(np.arctan2(vertical.astype(np.float64), horizontal.astype(np.float64))), 180.)
    errors = np.abs((theta[..., None]-np.array(view.azimuths_deg)+90.) % 180.-90.)
    expected &= np.argmin(errors, axis=-1) == 0
    actual = published_native(data, view, target, tmp_path)
    np.testing.assert_array_equal(actual != 0, expected,
        err_msg='Nonuniform plan origin cannot be silently treated as zero-start uniform angular samples')


def azimuthal_row_band_oracle(data, view, target, *, numeric=False):
    """Inverse physical height/categorical cells; all angle/column values agree."""
    work = (view.full_t, view.full_h, view.full_w)
    stack, vertical, horizontal = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2),
                                    'coronal': (2, 0, 1)}[view.azimuthal_base_view]
    shear = vertical if view.tilt_direction == 'vertical' else horizontal
    tangent = math.tan(math.radians(view.tilt_angle_deg)) if view.azimuthal_tilted_source else 0.
    canvas = max(view.src_h, view.src_w)
    scale = float(np.float32(data.shape[1]/canvas))
    offset = float(np.float32((data.shape[1]-1)/2-(data.shape[1]/canvas)*(view.src_h-1)/2))
    result = np.zeros(target, np.uint8)
    for position in np.ndindex(target):
        point = [(index+.5)*inside/outside-.5 for index, inside, outside in zip(position, work, target)]
        delta_y, delta_x = point[vertical]-(work[vertical]-1)/2, point[horizontal]-(work[horizontal]-1)/2
        if math.hypot(delta_y, delta_x) > view.roi_radius+.5:
            continue
        shift = -tangent*(point[shear]-(work[shear]-1)/2)-view.tilt_frame_start
        local = point[stack]+shift
        if view.src_h > target[stack]:
            first = max(0, math.floor(position[stack]*view.src_h/target[stack]+shift*view.src_h/work[stack]))
            stop = min(view.src_h, math.ceil((position[stack]+1)*view.src_h/target[stack]+shift*view.src_h/work[stack]))
            rows = range(first, stop)
        else:
            if not -.5 <= local < work[stack]-.5:
                continue
            row = round((local+.5)*view.src_h/work[stack]-.5)
            rows = (min(view.src_h-1, max(0, row)),)
        for native_row in rows:
            # Input is a square processing canvas; the source-native chart is
            # padded to max(H,W), then the declared float32 scale applies.
            model_row = min(data.shape[1]-1, max(0, round(scale*native_row+offset)))
            value = int(data[0, model_row, 0])
            result[position] = max(int(result[position]), value if numeric else int(value != 0))
    return result


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
@pytest.mark.parametrize('target', [(11, 15, 19), (4, 5, 6)])
def test_tilted_azimuthal_sharp_model_row_gap_and_numeric_max_survive_native_handoff(tmp_path, base, direction, target):
    from XTA.confidence_projection import score_projection_reader
    view = azimuthal_view(base, tilted=True, direction=direction, angle=23., native_raster=4)
    data = np.ones((view.num_slices, 3, 3), np.uint8)
    data[:, 1] = 0
    expected = azimuthal_row_band_oracle(data, view, target)
    assert expected.any() and not expected.all()
    actual = published_native(data, view, target, tmp_path/'published')
    direct = direct_native(data, view, target, tmp_path)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(direct, expected)
    score = np.broadcast_to(np.arange(3, dtype=np.uint8)[None, :, None]+np.uint8(100), data.shape).copy()*data
    score_expected = azimuthal_row_band_oracle(score, view, target, numeric=True)
    with score_projection_reader(score, view, target, tmp_path/'scores', memory_bytes=8*1024**2) as read:
        score_actual = np.stack([read(z) for z in range(target[0])])
    np.testing.assert_array_equal(score_actual, score_expected)
    np.testing.assert_array_equal(score_actual > 0, expected != 0)
