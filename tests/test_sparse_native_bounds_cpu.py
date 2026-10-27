"""Exact frozen bounds and independent discrete-address checks for numeric setup."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry, sparse_projection
from XTA.azimuthal_trajectory_cpu import azimuthal_backprojection_arrays
from XTA.interpolation import CTILE_INDEX_DTYPE
from XTA.sparse_bounds_cpu import _native_span, _physical_row_extents
from tests.reference_backends.sparse_bounds import legacy_destination_bbox


def make_view(base, tilt, *, work=(11, 13, 17), raster=7, spacing=45.):
    physical = None
    if tilt is not None:
        direction, angle = tilt
        physical = next(v for v in geometry._build_tilted_view_infos(*work,
            tilt_views=(base,), tilt_angles=(abs(angle),), tilt_directions=(direction,))
            if v.tilt_angle_deg == angle)
    return geometry._build_azimuthal_view_info(*work, base_view=base,
        azimuth_angle=spacing, azimuthal_native_raster=raster,
        request_token='bounds-equivalence', tilted_source=physical)


def make_store(view, pattern, *, side=None):
    h, w = (view.src_h, view.src_w) if side is None else (side, side)
    records = np.zeros(view.num_slices, CTILE_INDEX_DTYPE)
    rng = np.random.default_rng(160661)
    for frame in range(view.num_slices):
        if pattern == 'empty' or (pattern == 'scattered' and frame%3 == 1):
            continue
        records[frame]['kind'] = 1
        if pattern == 'edge':
            y, x = (0 if frame%2 else h-1), (0 if frame%3 else w-1)
            records[frame]['y0'], records[frame]['y1'] = y, y+1
            records[frame]['x0'], records[frame]['x1'] = x, x+1
        elif pattern == 'production':
            records[frame]['y0'], records[frame]['y1'] = frame%100, frame%100+900
            records[frame]['x0'], records[frame]['x1'] = frame%91, frame%91+1056
        else:
            y0, x0 = int(rng.integers(h)), int(rng.integers(w))
            records[frame]['y0'], records[frame]['y1'] = y0, int(rng.integers(y0+1, h+1))
            records[frame]['x0'], records[frame]['x1'] = x0, int(rng.integers(x0+1, w+1))
    records.flags.writeable = False
    return SimpleNamespace(shape=(view.num_slices, h, w), index=records)


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('tilt', [None, ('vertical', 30.), ('vertical', -30.),
                                ('horizontal', 37.), ('horizontal', -37.)])
@pytest.mark.parametrize('raster', [0, 7])
@pytest.mark.parametrize('target', [(4, 7, 9), (19, 23, 29)])
@pytest.mark.parametrize('pattern', ['empty', 'edge', 'scattered'])
def test_numeric_bounds_equal_frozen_scalar_geometry(base, tilt, raster, target, pattern):
    view = make_view(base, tilt, raster=raster)
    store = make_store(view, pattern)
    before = store.index.copy()
    expected = legacy_destination_bbox(store, view, target)
    actual = sparse_projection._destination_bbox_for_sparse_input(store, view, target)
    assert actual == expected
    np.testing.assert_array_equal(store.index, before)


@pytest.mark.parametrize('base,spacing,count', [('transverse', .050581033, 3559),
    ('sagittal', .054816042, 3284), ('coronal', .054816042, 3284)])
def test_actual_coarsened_geometry_preserves_frozen_bounds_without_source_volume(base, spacing, count):
    view = make_view(base, None, work=(2911, 3064, 3023), raster=2048, spacing=spacing)
    assert view.num_slices == count
    store = make_store(view, 'production')
    target = (1930, 3064, 3023)
    expected = legacy_destination_bbox(store, view, target)
    trajectory = azimuthal_backprojection_arrays(view)
    assert sparse_projection._destination_bbox_for_sparse_input(store, view, target) == expected
    assert sparse_projection._destination_bbox_for_sparse_input(store, view, target,
        trajectory=trajectory) == expected


@pytest.mark.parametrize('angles', [(73.,), (179., 11., 73., 151., 191.),
    (-90., 0., 90., 180., 270.), (0., 0., 90., 90., 180.)])
@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_nonuniform_duplicate_and_seam_trajectories_keep_exact_bounds(base, angles):
    view = replace(make_view(base, ('horizontal', -37.)),
        num_slices=len(angles), azimuths_deg=angles)
    store = make_store(view, 'edge')
    assert sparse_projection._destination_bbox_for_sparse_input(store, view, (17, 9, 23)) == \
        legacy_destination_bbox(store, view, (17, 9, 23))


@pytest.mark.parametrize('values', [np.array([0, 0, 1, 3, 3, 6], np.int32),
                                  np.array([3, 0, 6, 1, 3, 0], np.int32)])
def test_native_lookup_span_uses_binary_search_only_after_monotonicity_proof(values):
    monotone = bool(np.all(values[1:] >= values[:-1]))
    for first in range(8):
        for stop in range(first, 9):
            indices = np.flatnonzero((values >= first)&(values < stop))
            expected = (int(indices[0]), int(indices[-1])) if len(indices) else (len(values), -1)
            actual = _native_span(values, first, stop, monotone)
            # Empty binary-search spans may return insertion,insertion-1; both
            # represent the same empty support. Nonempty extents must be exact.
            assert actual == expected if indices.size else actual[1] < actual[0]


@pytest.mark.parametrize('native,physical', [(1, 9), (2, 3), (4, 3), (7, 13),
    (13, 7), (1930, 2911), (2911, 1930)])
def test_fused_row_inversion_preserves_independent_original_row_address_rule(native, physical):
    from XTA.backprojection import _azimuthal_processing_rows_for_output
    rows = np.arange(native, dtype=np.int32)//2
    processing = int(rows.max())+1
    grid = SimpleNamespace(native_h=native, native_row_to_processing=rows)
    first, last = np.full(processing, physical, np.int32), np.full(processing, -1, np.int32)
    for frame in range(physical):
        mapped = _azimuthal_processing_rows_for_output(grid, physical, frame)
        first[mapped] = np.minimum(first[mapped], frame)
        last[mapped] = np.maximum(last[mapped], frame)
    actual_first, actual_last = _physical_row_extents(rows, processing, physical)
    np.testing.assert_array_equal(actual_first, first)
    np.testing.assert_array_equal(actual_last, last)


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('radius,diameter', [(0., 1), (1e9, 7), (.125, 7)])
def test_degenerate_and_large_circle_exact_bounds(base, radius, diameter):
    view = replace(make_view(base, ('vertical', -30.)), roi_radius=radius,
                   diameter=diameter, center_y=-2.25, center_x=19.125)
    store = make_store(view, 'edge', side=1)
    assert sparse_projection._destination_bbox_for_sparse_input(store, view, (21, 8, 19)) == \
        legacy_destination_bbox(store, view, (21, 8, 19))


@pytest.mark.parametrize('changes', [{'kind': 2}, {'y0': 4, 'y1': 3}, {'y1': 8},
                                   {'x0': 7}, {'x1': 8}, {'x1': 0}])
def test_malformed_records_rejected_before_native_indexing(changes, monkeypatch):
    from XTA import sparse_bounds_cpu
    view = make_view('transverse', None)
    store = make_store(view, 'edge')
    records = store.index.copy()
    for field, value in changes.items():
        records[0][field] = value
    store.index = records
    monkeypatch.setattr(sparse_bounds_cpu, 'sparse_native_bounds',
        lambda *a: pytest.fail('invalid record reached unchecked native bounds'))
    with pytest.raises(ValueError, match='record bounds'):
        sparse_projection._destination_bbox_for_sparse_input(store, view, (9, 9, 9))


def test_unused_empty_record_coordinates_still_ignored():
    view = make_view('transverse', None)
    store = make_store(view, 'empty')
    records = store.index.copy()
    for field in ('y0', 'x0', 'y1', 'x1'):
        records[field] = 999999
    store.index = records
    assert sparse_projection._destination_bbox_for_sparse_input(store, view, (9, 9, 9)) is None


@pytest.mark.parametrize('invalid', ['nan', 'negative_owner', 'past_owner', 'length', 'f32'])
def test_invalid_borrowed_trajectory_rejected_before_native_indexing(invalid, monkeypatch):
    from XTA import sparse_bounds_cpu
    view = make_view('transverse', None)
    store = make_store(view, 'edge')
    angles, owners, reverses = [a.copy() for a in azimuthal_backprojection_arrays(view)]
    if invalid == 'nan': angles[0] = np.nan
    elif invalid == 'negative_owner': owners[0] = -1
    elif invalid == 'past_owner': owners[0] = len(store.index)
    elif invalid == 'length': reverses = reverses[:-1]
    else: angles = angles.astype(np.float32)
    monkeypatch.setattr(sparse_bounds_cpu, 'sparse_native_bounds',
        lambda *a: pytest.fail('invalid trajectory reached unchecked native bounds'))
    with pytest.raises(ValueError, match='exact finite numeric trajectory'):
        sparse_projection._destination_bbox_for_sparse_input(store, view, (9, 9, 9),
            trajectory=(angles, owners, reverses))


def test_encoded_caller_borrows_exact_trajectory_arrays(monkeypatch):
    from XTA.packed_tilted_projection import _encoded_destination_bbox
    view = make_view('transverse', None)
    store = make_store(view, 'edge')
    trajectory = azimuthal_backprojection_arrays(view)
    def inspect(received_store, received_view, output_shape, *, trajectory):
        assert received_store is store and received_view is view
        assert all(a is b for a, b in zip(trajectory, expected))
        return (0, 0, 0, *output_shape)
    expected = trajectory
    monkeypatch.setattr(sparse_projection, '_destination_bbox_for_sparse_input', inspect)
    assert _encoded_destination_bbox(store, view, (9, 9, 9), trajectory=trajectory) == (0, 0, 0, 9, 9, 9)
