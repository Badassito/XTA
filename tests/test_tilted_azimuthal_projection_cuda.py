"""Independent native-cell coverage checks; no retired scatter oracle is used."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import tracemalloc
from unittest import mock

import numpy as np
import pytest

from XTA import backprojection as bp, geometry as g
from XTA.projection_coverage import iter_destination_samples, nearest_plan_indices
from XTA.tilted_azimuthal_projection import (
    TiltedAzimuthalPlanUnavailable, build_tilted_azimuthal_plan,
    tilted_azimuthal_cuda_capability,
)
from XTA.tilted_azimuthal_projection_cuda import (
    TiltedAzimuthalCudaProjector, TiltedAzimuthalCudaProjectionUnavailable,
)


@pytest.fixture(autouse=True)
def cpu_only_routes(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT_RESIDENT', '0')


def tilted_views(shape=(5, 7, 9), angles=(30.,), raster=0):
    result = []
    for base in ('transverse', 'sagittal', 'coronal'):
        for tilted in g._build_tilted_view_infos(
                *shape, tilt_views=(base,), tilt_angles=angles,
                tilt_directions=('vertical', 'horizontal')):
            result.append(g._build_azimuthal_view_info(
                *shape, base_view=base, azimuth_angle=30.,
                azimuthal_native_raster=raster, request_token='test', tilted_source=tilted))
    return result


def source_for(view, value=1, processing_size=None):
    shape = (view.num_slices, view.src_h, view.src_w)
    if processing_size is not None:
        shape = (view.num_slices, processing_size, processing_size)
    return np.full(shape, value, np.uint8)


def positive_support(view, shape):
    """Physical inverse-shear/circular FOV, independent of address generator."""
    base = g.azimuthal_base_view_name(view)
    stack, vertical, horizontal = {'transverse': (0, 1, 2),
        'sagittal': (1, 0, 2), 'coronal': (2, 0, 1)}[base]
    work = (view.full_t, view.full_h, view.full_w)
    coords = np.indices(shape, dtype=np.float64)
    for axis in range(3):
        coords[axis] = (coords[axis] + .5) * work[axis] / shape[axis] - .5
    shear_axis = vertical if view.tilt_direction == 'vertical' else horizontal
    shear = np.tan(np.deg2rad(view.tilt_angle_deg)) * (coords[shear_axis] - (work[shear_axis] - 1) / 2)
    local = coords[stack] - shear - view.tilt_frame_start
    physical = (local >= -.5) & (local < work[stack] - .5)
    radius = view.roi_radius if view.roi_radius > 0 else max(1., (view.diameter - 1) / 2)
    circle = ((coords[horizontal] - view.center_x)**2
              + (coords[vertical] - view.center_y)**2) <= (radius + .5)**2
    return physical & circle


@pytest.mark.parametrize('view', tilted_views(), ids=lambda view: view.name)
@pytest.mark.parametrize('shape', [(5, 7, 9), (8, 11, 14)])
@pytest.mark.parametrize('processing_size', [None, 4])
def test_all_positive_native_coverage_preserves_physical_wedge_circle_and_borders(tmp_path, view, shape, processing_size):
    source = source_for(view, processing_size=processing_size)
    actual = bp.backproject_azimuthal_volume_to_volume(source, view, tmp_path/'dense.dat', 'coverage',
        out_shape_tyx=shape, reserve_bytes=0, prefer_memory=True)
    np.testing.assert_array_equal(actual, positive_support(view, shape))
    assert actual.any() and not actual.all()


@pytest.mark.parametrize('view', tilted_views(), ids=lambda view: view.name)
def test_sink_native_blocks_equal_dense_and_chunk_partition_without_cuda(tmp_path, view):
    rng = np.random.default_rng(411)
    source = (rng.random((view.num_slices, view.src_h, view.src_w)) < .25).astype(np.uint8)
    shape = (8, 11, 14)
    dense = bp.backproject_azimuthal_volume_to_volume(source, view, tmp_path/'dense.dat', 'dense',
        out_shape_tyx=shape, reserve_bytes=0)
    pieces = np.zeros(shape, np.uint8)
    def callback(z, block):
        assert block.shape[0] == 1
        pieces[z:z + len(block)] = block
    with mock.patch.object(bp, '_try_acquire_main_process_gpu_stage', side_effect=AssertionError('GPU admission')):
        result = bp.backproject_azimuthal_volume_to_volume(source, view, tmp_path/'sink.dat', 'sink',
            out_shape_tyx=shape, reserve_bytes=0, sink_only=True, projection_block_callback=callback)
    assert result.shape == shape
    np.testing.assert_array_equal(pieces, dense)
    other = np.zeros(shape, np.uint8).reshape(-1)
    for dest, frame, row, col in iter_destination_samples(view, source.shape, shape, chunk_voxels=7):
        assert len(dest) <= 7
        other[dest[source[frame, row, col] != 0]] = 1
    np.testing.assert_array_equal(other.reshape(shape), dense)


def test_nonuniform_offset_actual_angle_ownership_and_half_turn_seam():
    plan = [SimpleNamespace(angle_deg=angle) for angle in (73., 11., 151., 179., 191.)]
    theta = np.asarray([0., 7., 39., 100., 155., 177.])
    distances = np.abs((theta[:, None] - np.asarray([p.angle_deg for p in plan])[None, :] + 90.) % 180. - 90.)
    np.testing.assert_array_equal(nearest_plan_indices(theta, plan), distances.argmin(axis=1))


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('spacing', [7., 40.])
@pytest.mark.parametrize('shape', [(6, 8, 10), (13, 16, 18)])
def test_upright_dense_coarse_all_base_native_roi_coverage(tmp_path, base, spacing, shape):
    view = g._build_azimuthal_view_info(9, 11, 13, base_view=base,
        azimuth_angle=spacing, azimuthal_native_raster=4, request_token='test')
    source = source_for(view)
    output = bp.backproject_azimuthal_volume_to_volume(source, view, tmp_path/'az.dat',
        'upright native coverage', out_shape_tyx=shape, reserve_bytes=0)
    np.testing.assert_array_equal(output, positive_support(view, shape))


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_zero_shear_agrees_with_upright_native_center_plane_and_stack_or(tmp_path, base):
    work = (9, 11, 13)
    upright = g._build_azimuthal_view_info(*work, base_view=base, azimuth_angle=8.,
        azimuthal_native_raster=6, request_token='test')
    tilted = replace(upright, azimuthal_tilted_source=True, tilt_direction='horizontal', tilt_angle_deg=0.)
    rng = np.random.default_rng(319)
    source = (rng.random((upright.num_slices, upright.src_h, upright.src_w)) < .19).astype(np.uint8)
    for index, shape in enumerate(((9, 11, 13), (6, 8, 10), (13, 16, 18))):
        expected = bp.backproject_azimuthal_volume_to_volume(source, upright, tmp_path/f'upright-{index}.dat', 'upright',
            out_shape_tyx=shape, reserve_bytes=0)
        actual = bp.backproject_azimuthal_volume_to_volume(source, tilted, tmp_path/f'tilted-{index}.dat', 'tilted',
            out_shape_tyx=shape, reserve_bytes=0)
        np.testing.assert_array_equal(actual, expected)


def test_legacy_cuda_is_typed_unavailable_before_geometry_gpu_lease_or_allocation():
    source = np.ones((6, 5, 7), np.uint8)
    view = tilted_views()[0]
    assert not tilted_azimuthal_cuda_capability()[0]
    with mock.patch.object(bp, 'build_dense_azimuthal_backprojection_map', side_effect=AssertionError('geometry allocation')):
        with pytest.raises(TiltedAzimuthalPlanUnavailable, match='native destination'):
            build_tilted_azimuthal_plan(source, view, (8, 11, 14))
    with mock.patch.object(TiltedAzimuthalCudaProjector, '_initialize_cuda', side_effect=AssertionError('CUDA initialized')):
        with pytest.raises(TiltedAzimuthalCudaProjectionUnavailable, match='native destination'):
            TiltedAzimuthalCudaProjector(source, object())
    with mock.patch.object(bp, '_try_acquire_main_process_gpu_stage', side_effect=AssertionError('lease acquired')):
        assert bp._try_tilted_azimuthal_cuda_stage(source, view, (8, 11, 14)) is None


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction', ['horizontal', 'vertical'])
@pytest.mark.parametrize('angle', [30., 45.])
def test_ordinary_tilted_native_pull_has_no_forward_half_tie_holes(tmp_path, base, direction, angle):
    work = (6, 8, 10)
    view = g._build_tilted_view_infos(*work, tilt_views=(base,), tilt_angles=(angle,),
                                    tilt_directions=(direction,))[0]
    source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
    shape = (11, 15, 19)
    actual = bp.backproject_tilted_volume_to_volume(source, view, tmp_path/'tilt.dat', 'tilt',
        out_shape_tyx=shape, reserve_bytes=0)
    stack, va, ua = {'transverse':(0,1,2), 'sagittal':(1,0,2), 'coronal':(2,0,1)}[base]
    c = np.indices(shape, dtype=np.float64)
    for a in range(3): c[a] = (c[a] + .5) * work[a] / shape[a] - .5
    sa = va if direction == 'vertical' else ua
    frame = c[stack] - np.tan(np.deg2rad(angle)) * (c[sa] - (work[sa] - 1) / 2)
    expected = (frame >= -.5) & (frame < work[stack] - .5)
    np.testing.assert_array_equal(actual, expected)


def test_lazy_strong_contraction_has_bounded_coefficient_storage_and_last_contributor():
    view = g._build_tilted_view_infos(901, 7, 9, tilt_views=('transverse',), tilt_angles=(30.,),
                                    tilt_directions=('horizontal',))[0]
    view = replace(view, tilt_angle_deg=0.)
    tracemalloc.start()
    highest = -1
    for dest, frame, row, col in iter_destination_samples(view, (901,7,9), (3,21,27), chunk_voxels=1701):
        assert len(dest) <= 1701
        highest = max(highest, int(frame.max()))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert highest == 900
    assert peak < 8 * 1024**2


def test_bbox_visit_and_first_stop_are_exact_subset_of_complete_address_stream():
    view = tilted_views()[0]
    source = source_for(view)
    shape = (8,11,14)
    bbox = (2,3,4,5,8,10)
    result = list(iter_destination_samples(view, source.shape, shape, destination_bbox_tyx=bbox, chunk_voxels=13))
    addresses = np.concatenate([part[0] for part in result])
    z, y, x = np.unravel_index(addresses, shape)
    assert np.all((z>=2)&(z<5)&(y>=3)&(y<8)&(x>=4)&(x<10))
    assert len(addresses) <= (5-2)*(8-3)*(10-4)


def test_conservative_occupied_band_skips_empty_native_work_without_losing_support(tmp_path):
    view = g._build_tilted_view_infos(301, 7, 9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]
    source = np.zeros((301, 7, 9), np.uint8)
    source[150, 2:5, 3:6] = 1
    shape = (501, 11, 14)
    bbox = bp._native_pull_occupied_stack_bbox(source, view, shape)
    assert bbox[3] - bbox[0] < 30
    expected = np.zeros(shape, np.uint8).reshape(-1)
    for dest, frame, row, col in iter_destination_samples(view, source.shape, shape):
        expected[dest[source[frame, row, col] != 0]] = 1
    output = bp.backproject_tilted_volume_to_volume(source, view, tmp_path/'bounded.dat',
        'bounded native coverage', out_shape_tyx=shape, reserve_bytes=0)
    np.testing.assert_array_equal(output, expected.reshape(shape))


@pytest.mark.parametrize('base,shape', [
    ('transverse', (1, 5, 5)), ('sagittal', (5, 1, 5)), ('coronal', (5, 5, 1))])
def test_degenerate_azimuthal_radius_does_not_inflate_disk_and_scores(tmp_path, base, shape):
    from XTA.confidence_projection import score_projection_reader
    from XTA.projection_coverage import effective_azimuthal_radius
    view = g._build_azimuthal_view_info(1,1,1,base_view=base,azimuth_angle=45.,
                                      azimuthal_native_raster=0,request_token='test')
    assert view.diameter == 1 and view.roi_radius == 0
    assert effective_azimuthal_radius(view) == 0
    source = source_for(view, value=173)
    # The production factory's zero radius is a legitimate single-cell disk.
    stack, va, ua = {'transverse':(0,1,2), 'sagittal':(1,0,2), 'coronal':(2,0,1)}[base]
    coords = np.indices(shape, dtype=np.float64)
    for axis in range(3): coords[axis] = (coords[axis]+.5)/shape[axis]-.5
    expected = coords[va]**2 + coords[ua]**2 <= .25
    assert expected.sum() == 21
    output = bp.backproject_azimuthal_volume_to_volume((source > 0).astype(np.uint8),view,tmp_path/'degenerate.dat',
        'degenerate disk',out_shape_tyx=shape,reserve_bytes=0)
    np.testing.assert_array_equal(output,expected)
    with score_projection_reader(source,view,shape,tmp_path/'scores',memory_bytes=1024**2) as reader:
        scores=np.stack([reader(z) for z in range(shape[0])])
    np.testing.assert_array_equal(scores,expected.astype(np.uint8)*173)
    assert effective_azimuthal_radius(replace(view,roi_radius=2.)) == 2.


@pytest.mark.parametrize('base,work', [
    ('transverse',(3,1,5)), ('sagittal',(1,3,5)), ('coronal',(1,5,3))])
def test_degenerate_nonzero_tilted_azimuthal_preserves_disk_and_physical_wedge(tmp_path, base, work):
    from XTA.confidence_projection import score_projection_reader
    tilted = g._build_tilted_view_infos(*work,tilt_views=(base,),tilt_angles=(30.,),
                                      tilt_directions=('horizontal',))[0]
    view = g._build_azimuthal_view_info(*work,base_view=base,azimuth_angle=45.,
        azimuthal_native_raster=0,request_token='test',tilted_source=tilted)
    shape=tuple(axis*5 for axis in work)
    coords=np.indices(shape,dtype=np.float64)
    for axis in range(3):coords[axis]=(coords[axis]+.5)*work[axis]/shape[axis]-.5
    stack,va,ua={'transverse':(0,1,2),'sagittal':(1,0,2),'coronal':(2,0,1)}[base]
    frame=coords[stack]-np.tan(np.deg2rad(30.))*(coords[ua]-view.center_x)
    expected=((frame>=-.5)&(frame<work[stack]-.5)
              &((coords[ua]-view.center_x)**2+(coords[va]-view.center_y)**2<=.25))
    source=source_for(view,value=173)
    output=bp.backproject_azimuthal_volume_to_volume(source,view,tmp_path/'degenerate.dat',
        'degenerate wedge',out_shape_tyx=shape,reserve_bytes=0)
    np.testing.assert_array_equal(output,expected)
    with score_projection_reader(source,view,shape,tmp_path/'scores',memory_bytes=1024**2) as reader:
        scores=np.stack([reader(z) for z in range(shape[0])])
    np.testing.assert_array_equal(scores,expected.astype(np.uint8)*173)
