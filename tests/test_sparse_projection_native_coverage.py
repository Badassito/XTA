"""Selected component bounds preserve corrected native pull coverage."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import gc
import io
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import backprojection, geometry, sparse_projection
from XTA.config import TiltedViewGroup
from XTA.interpolation import CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.runtime import wait_for_retired_memmap_directory_cleanup


def _views(shape=(11, 13, 15), native_raster=0):
    return [view for view in geometry.get_view_infos(
        *shape, cartesian_views=[],
        azimuthal_views=['transverse', 'sagittal', 'coronal',
                         'tilted_transverse', 'tilted_sagittal', 'tilted_coronal'],
        azimuthal_azimuth_angles=[45.0] * 6,
        tilt_groups=[TiltedViewGroup(('transverse', 'sagittal', 'coronal'),
                                    (-37.0, 37.0), ('vertical', 'horizontal'))],
        azimuthal_native_raster=native_raster,
    ) if view.family == 'azimuthal']


_VIEWS = _views()


def _full_destination_reference(data, view, target):
    from XTA.projection_coverage import iter_destination_samples

    expected = np.zeros(target, np.uint8)
    for destinations, frames, rows, columns in iter_destination_samples(view, data.shape, target):
        np.maximum.at(expected.reshape(-1), destinations, data[frames, rows, columns])
    return expected


@pytest.mark.parametrize('view', _VIEWS, ids=lambda view: view.name)
@pytest.mark.parametrize('target', [(11, 13, 15), (19, 21, 23), (7, 9, 11)])
def test_sparse_selected_cells_equal_full_destination_pull(tmp_path, view, target):
    """Full-output dense oracle detects any support omitted by sparse bounds."""
    shape = (view.num_slices, 7, 7)
    data = np.zeros(shape, np.uint8)
    data[0, 1, 2] = 1
    data[-1, -2, -3] = 1
    data[1, 3, 3] = 1
    source_path = tmp_path / 'selected.cvol'
    with mock.patch.dict(os.environ, {'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1',
                                    'YOLO_TTA_GPU_BACKPROJECT': '0'}), \
            mock.patch.object(backprojection, 'gpu_backproject_enabled', return_value=False), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                                 desc='original selected additions')
        expected = _full_destination_reference(data, view, target)
        source = RawBBoxMaskStore.open(source_path, mmap_payload=False)
        try:
            with mock.patch.object(RawBBoxMaskStore, 'decode_slice', side_effect=AssertionError('dense input')), \
                    mock.patch.object(RawBBoxMaskStore, 'decode_slice_crop', side_effect=AssertionError('dense crop')):
                result = sparse_projection.project_azimuthal_sparse_store(
                    source, view, tmp_path / 'bounded.cvol', out_shape_tyx=target,
                )
            # A borrowed adapter retains ownership and remains usable after projection.
            assert source.shape == shape
        finally:
            source.close()
    actual_store = RawBBoxMaskStore.open(tmp_path / 'bounded.cvol', mmap_payload=True)
    try:
        actual = np.stack([actual_store.decode_slice(frame) for frame in range(target[0])])
        assert actual_store.meta['projection_geometry_contract'] == 'xta.native_destination_pull/1'
    finally:
        actual_store.close()
    np.testing.assert_array_equal(actual, expected)
    assert result['foreground_voxels'] == np.count_nonzero(expected)
    assert result['input_foreground_samples'] == 3
    assert result['max_destination_address_count'] <= sparse_projection._MAP_STRIP_PIXELS
    assert result['map_bytes'] > 0
    if geometry.is_tilted_azimuthal_view(view):
        assert result['backend'].startswith('compiled_sparse_')


@pytest.mark.parametrize('view', _views((61, 73, 89))[::3], ids=lambda view: view.name)
@pytest.mark.parametrize('position', ['center', 'edge'])
def test_tiny_component_native_upscale_visits_bounded_destination(tmp_path, view, position):
    """A small selected cell requests a geometry subvolume, not a full scan."""
    import XTA.projection_coverage as coverage

    shape = (view.num_slices, view.src_h, view.src_w)
    data = np.zeros(shape, np.uint8)
    column = shape[2] // 2 if position == 'center' else shape[2] - 2
    data[0, shape[1] // 2, column] = 1
    source_path = tmp_path / 'tiny.cvol'
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        write_raw_bbox_mask_store(data, source_path, format_name=CVOL_FORMAT, desc='tiny selected component')
        original = coverage.iter_destination_samples
        seen = []

        def bounded(*args, **kwargs):
            bounds = kwargs.get('destination_bbox_tyx')
            assert bounds is not None
            seen.append(bounds)
            yield from original(*args, **kwargs)

        target = (91, 109, 133)
        with mock.patch.object(coverage, 'iter_destination_samples', side_effect=bounded):
            result = sparse_projection.project_azimuthal_sparse_store(
                source_path, view, tmp_path / 'tiny-projected.cvol', out_shape_tyx=target,
            )
    assert not seen  # Tilted jobs now use the fused compiled destination pull.
    assert result['destination_candidate_voxels'] < np.prod(target) // 2
    if position == 'center':
        assert result['destination_candidate_voxels'] < np.prod(target) // 100
    assert result['max_destination_address_count'] <= sparse_projection._MAP_STRIP_PIXELS
    assert result['foreground_voxels'] > 0


@pytest.mark.parametrize('view', _views(native_raster=9), ids=lambda view: view.name)
def test_compact_native_raster_nonuniform_seam_angles_preserve_all_support(tmp_path, view):
    view = replace(view, num_slices=4, azimuths_deg=(1.0, 37.0, 88.0, 179.0))
    data = np.zeros((4, 5, 5), np.uint8)
    data[0, 1, 2] = 1
    data[-1, 3, 1] = 1
    target = (23, 17, 29)
    with mock.patch.dict(os.environ, {'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1'}), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        expected = _full_destination_reference(data, view, target)
        source_path = tmp_path / 'compact.cvol'
        write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                                 desc='compact native seam fixture')
        result = sparse_projection.project_azimuthal_sparse_store(
            source_path, view, tmp_path / 'compact-projected.cvol', out_shape_tyx=target,
        )
    store = RawBBoxMaskStore.open(result['path'], mmap_payload=True)
    try:
        actual = np.stack([store.decode_slice(z) for z in range(target[0])])
    finally:
        store.close()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('view', _views((41, 49, 57), native_raster=3)[3::4], ids=lambda view: view.name)
@pytest.mark.parametrize('row', [0, 2])
def test_extreme_compact_native_row_cells_do_not_clip_sheared_coverage(tmp_path, view, row):
    data = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    data[0, row, 1] = 1
    target = (47, 53, 59)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        expected = _full_destination_reference(data, view, target)
        source_path = tmp_path / 'extreme-compact.cvol'
        write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                                 desc='extreme compact row cell')
        result = sparse_projection.project_azimuthal_sparse_store(
            source_path, view, tmp_path / 'extreme-projected.cvol', out_shape_tyx=target,
        )
    store = RawBBoxMaskStore.open(result['path'], mmap_payload=True)
    try:
        actual = np.stack([store.decode_slice(z) for z in range(target[0])])
    finally:
        store.close()
    np.testing.assert_array_equal(actual, expected)


def test_encoded_sampler_rejects_invalid_addresses_and_corrupt_payload(tmp_path):
    source_path = tmp_path / 'raw.cvol'
    data = np.zeros((2, 5, 9), np.uint8)
    data[1, 1:4, 2:8] = 1
    with contextlib.redirect_stdout(io.StringIO()):
        write_raw_bbox_mask_store(data, source_path, desc='sampler input')
    source = RawBBoxMaskStore.open(source_path, mmap_payload=True)
    try:
        sampler = sparse_projection._SparseMaskSampler(source)
        try:
            with pytest.raises(ValueError, match='declared input'):
                sampler.sample([2], [1], [1])
            with pytest.raises(ValueError, match='shape'):
                sampler.sample([1, 1], [1], [2])
            assert sampler.sample([-1], [-1], [-1], [False]).tolist() == [0]
        finally:
            sampler.close()
        source.index[1]['payload_size'] += 1
        with pytest.raises(ValueError, match='payload size'):
            sparse_projection._SparseMaskSampler(source)
    finally:
        source.close()


def test_tilted_pull_failure_preserves_source_and_never_publishes_partial_store(tmp_path):
    view = _VIEWS[3]
    data = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
    source_path, target = tmp_path / 'source.cvol', tmp_path / 'failed.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        write_raw_bbox_mask_store(data, source_path, desc='failure source')

    def fail(*_args, **_kwargs):
        raise RuntimeError('injected pull failure')

    with mock.patch('XTA.projection_coverage_cpu.pull_native_encoded_flat_into', new=fail):
        with pytest.raises(RuntimeError, match='injected pull failure'):
            sparse_projection.project_azimuthal_sparse_store(
                source_path, view, target, out_shape_tyx=(11, 13, 15),
            )
    gc.collect()
    for pending in tmp_path.glob('.*.projection-*'):
        wait_for_retired_memmap_directory_cleanup(pending)
    assert not target.exists()
    assert not list(tmp_path.glob('.*.projection-*'))
    source = RawBBoxMaskStore.open(source_path)
    try:
        np.testing.assert_array_equal(source.decode_slice(0), data[0])
    finally:
        source.close()


@pytest.mark.parametrize('view', _views((1, 1, 1)), ids=lambda view: view.name)
@pytest.mark.parametrize('processing_size', [1, 3])
def test_one_column_zero_roi_has_analytic_half_cell_disk_not_inflated_square(tmp_path, view, processing_size):
    """An independent physical disk oracle catches shared radius inflation."""
    assert view.diameter == view.src_w == 1
    assert view.roi_radius == 0.0
    data = np.ones((view.num_slices, processing_size, processing_size), np.uint8)
    coordinates = (np.arange(5, dtype=np.float64) + 0.5) / 5.0 - 0.5
    disk = ((coordinates[:, None] ** 2 + coordinates[None, :] ** 2) <= 0.5 ** 2).astype(np.uint8)
    assert np.count_nonzero(disk) == 21
    base = geometry.azimuthal_base_view_name(view)
    expected = {'transverse': disk[None, :, :], 'sagittal': disk[:, None, :],
                'coronal': disk[:, :, None]}[base]
    source_path = tmp_path / 'one-column.cvol'
    with mock.patch.dict(os.environ, {'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1',
                                    'YOLO_TTA_GPU_BACKPROJECT': '0'}), \
            mock.patch.object(backprojection, 'gpu_backproject_enabled', return_value=False), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        write_raw_bbox_mask_store(data, source_path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
                                 desc='actual diameter-one native producer')
        dense = backprojection.backproject_azimuthal_volume_to_volume(
            data, view, tmp_path / 'one-column-dense.dat', 'analytic disk dense',
            out_shape_tyx=expected.shape, reserve_bytes=0, workers=1,
        )
        try:
            np.testing.assert_array_equal(np.asarray(dense), expected)
        finally:
            from XTA.runtime import close_memmap_array
            close_memmap_array(dense)
        result = sparse_projection.project_azimuthal_sparse_store(
            source_path, view, tmp_path / 'one-column-projected.cvol', out_shape_tyx=expected.shape,
        )
    store = RawBBoxMaskStore.open(result['path'], mmap_payload=True)
    try:
        actual = np.stack([store.decode_slice(z) for z in range(expected.shape[0])])
    finally:
        store.close()
    np.testing.assert_array_equal(actual, expected)
    assert result['foreground_voxels'] == 21
