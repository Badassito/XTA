"""Native confidence values follow physical cells, then share binary support."""
from itertools import product
from dataclasses import replace
import math

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.config import TiltedViewGroup
from XTA.confidence_projection import score_projection_reader


def _axis_cells(index, inside, outside):
    if outside < inside:
        return range(index*inside//outside, ((index+1)*inside+outside-1)//outside)
    return ((index+.5)*inside/outside-.5,)


def _scalar_tilted_scores(values, view, target):
    """Scalar inverse-shear oracle; never calls a projection address helper."""
    base = geometry.tilted_base_view_name(view)
    stack, row, column = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2),
                          'coronal': (2, 0, 1)}[base]
    work = (view.full_t, view.full_h, view.full_w)
    axis = row if view.tilt_direction == 'vertical' else column
    tangent = math.tan(math.radians(view.tilt_angle_deg))
    if values.shape[1:] == (view.src_h, view.src_w):
        matrix = np.array(((1., 0., 0.), (0., 1., 0.)))
    else:
        matrix = geometry.build_affine(view.name, view.src_w, view.src_h,
            values.shape[2], 0., view.pad_mode).M_src_to_out
    result = np.zeros(target, np.uint8)
    for destination in np.ndindex(target):
        for coordinate in product(*[_axis_cells(i, n, o)
                                    for i, n, o in zip(destination, work, target)]):
            frame = coordinate[stack]-tangent*(coordinate[axis]-(work[axis]-1)/2)
            frame -= view.tilt_frame_start
            if not -.5 <= frame < values.shape[0]-.5:
                continue
            v = min(view.src_h-1, max(0., coordinate[row]))
            u = min(view.src_w-1, max(0., coordinate[column]))
            x, y = matrix @ np.array((u, v, 1.))
            if not (-.5 <= y < values.shape[1]-.5 and -.5 <= x < values.shape[2]-.5):
                continue
            f, y, x = (math.floor(q+.5) for q in (frame, y, x))
            result[destination] = max(result[destination], values[f, y, x])
    return result


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for name in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT',
                 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT'):
        monkeypatch.setenv(name, '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('direction', ('vertical', 'horizontal'))
@pytest.mark.parametrize('reduced', (False, True))
@pytest.mark.parametrize('target', ((10, 14, 18), (3, 4, 5)))
def test_tilted_values_match_scalar_native_cell_oracle(tmp_path, base, direction, reduced, target):
    view = next(v for v in geometry.get_view_infos(5, 7, 9, cartesian_views=(),
        tilt_groups=(TiltedViewGroup((base,), (23.,), (direction,)),)) if v.tilt_angle_deg < 0)
    shape = (view.num_slices, 3, 3) if reduced else (view.num_slices, view.src_h, view.src_w)
    values = np.random.default_rng(83).choice(np.array([0, 17, 89, 231], np.uint8), shape)
    expected = _scalar_tilted_scores(values, view, target)
    with score_projection_reader(values, view, target, tmp_path, memory_bytes=2*1024**2) as read:
        actual = np.stack([read(z) for z in range(target[0])])
    np.testing.assert_array_equal(actual, expected)


def _views():
    views = geometry.get_view_infos(5, 7, 9, cartesian_views=(),
        tilt_groups=(TiltedViewGroup(('transverse', 'sagittal', 'coronal'),
                                    (23.,), ('vertical', 'horizontal')),),
        azimuthal_views=('transverse', 'sagittal', 'coronal',
                         'tilted_transverse', 'tilted_sagittal', 'tilted_coronal'),
        azimuthal_azimuth_angles=(45.,)*6, azimuthal_native_raster=0,
        radial_views=('transverse', 'sagittal', 'coronal'), radial_min_radius=.7,
        radial_patch_size=5, spherical_views=('transverse',), spherical_min_radius=.7,
        spherical_patch_size=5)
    selected, seen = [], set()
    for view in views:
        key = view.name if view.family not in ('radial', 'spherical') else (
            view.family, view.radial_base_view, view.spherical_face)
        if key not in seen:
            selected.append(view)
            seen.add(key)
    return selected


@pytest.mark.parametrize('view', _views(), ids=lambda v: v.name)
def test_native_score_positive_support_equals_binary_projection(tmp_path, view):
    target = (8, 10, 12)
    shape = (view.num_slices, 3, 3)
    values = np.random.default_rng(87).choice(np.array([0, 0, 19, 97, 203], np.uint8), shape)
    original = values.copy()
    with score_projection_reader(values, view, target, tmp_path/'scores', memory_bytes=4*1024**2) as read:
        actual = np.stack([read(z) for z in range(target[0])])
    path = tmp_path/'binary.dat'
    binary = assembly.project_view_volume_to_orthogonal_volume((values > 0).astype(np.uint8),
        view, path, 'confidence support oracle', out_shape_tyx=target,
        workers=1, prefer_memory=False, reserve_bytes=0)
    try:
        np.testing.assert_array_equal(actual > 0, np.asarray(binary) > 0)
        assert set(np.unique(actual)).issubset({0, 19, 97, 203})
        np.testing.assert_array_equal(values, original)
    finally:
        if isinstance(binary, np.memmap):
            binary._mmap.close()
        path.unlink(missing_ok=True)


def test_contracted_destination_takes_max_without_inventing_known_background(tmp_path):
    view = geometry.get_view_infos(3, 5, 7, cartesian_views=(),
        tilt_groups=(TiltedViewGroup(('transverse',), (23.,), ('vertical',)),))[0]
    view = replace(view, tilt_angle_deg=0., tilt_frame_start=0, num_slices=3)
    values = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    values[0, 0, 0], values[1, 1, 1] = 71, 229
    with score_projection_reader(values, view, (1, 2, 3), tmp_path, memory_bytes=1024**2) as read:
        actual = read(0)
    assert actual[0, 0] == 229
    assert np.count_nonzero(actual) == 1


def test_offset_nonuniform_azimuthal_angles_have_actual_nearest_owners(tmp_path):
    view = geometry.get_view_infos(3, 3, 3, cartesian_views=(),
        azimuthal_views=('transverse',), azimuthal_azimuth_angles=(45.,),
        azimuthal_native_raster=0)[0]
    angles = (7., 19., 62., 144.)
    view = replace(view, azimuths_deg=angles, num_slices=len(angles))
    levels = np.array([17, 59, 101, 211], np.uint8)
    values = np.broadcast_to(levels[:, None, None], (4, 3, 3)).copy()
    expected = np.zeros((3, 3, 3), np.uint8)
    for y, x in product(range(3), repeat=2):
        dx, dy = x-view.center_x, y-view.center_y
        if math.hypot(dx, dy) > view.roi_radius+.5:
            continue
        theta = math.degrees(math.atan2(dy, dx)) % 180.
        owner = min(range(4), key=lambda i: abs((theta-angles[i]+90.) % 180.-90.))
        expected[:, y, x] = levels[owner]
    for budget in (None, 1024**2):
        with score_projection_reader(values, view, expected.shape, tmp_path,
                                     memory_bytes=budget) as read:
            actual = np.stack([read(z) for z in range(3)])
        np.testing.assert_array_equal(actual, expected)

