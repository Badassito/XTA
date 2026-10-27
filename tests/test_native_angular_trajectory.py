"""Exact angular owner parity against the original per-angle NumPy builder."""
from dataclasses import replace

import numpy as np
import pytest

from XTA import backprojection as bp, geometry
from XTA.azimuthal_trajectory_cpu import (azimuthal_backprojection_arrays,
    nearest_azimuthal_sources, prepare_azimuthal_backprojection_arrays)


def _view(angles, diameter=13):
    view = geometry._build_azimuthal_view_info(9, diameter, diameter,
        base_view='transverse', azimuth_angle=30., azimuthal_native_raster=0,
        request_token='trajectory-parity')
    return replace(view, azimuths_deg=tuple(angles), num_slices=len(angles))


def _original(view):
    """Frozen original algorithm; deliberately uses one NumPy argmin per angle."""
    source = np.asarray([float(angle) for angle in view.azimuths_deg], np.float64)
    coverage = bp.azimuthal_full_coverage_angle_deg(int(view.diameter))
    if not source.size:
        return (source, np.empty(0, np.int32), np.empty(0, bool)), dict(
            provided_spacing_deg=0., coverage_spacing_deg=coverage,
            effective_spacing_deg=0., source_frames=0., backprojection_angles=0., densified=0.)
    provided = bp._azimuthal_view_nominal_spacing_deg(view)
    effective = min(float(provided), float(coverage))
    if provided <= coverage * (1. + 1e-9):
        angles = source
        indices = np.arange(source.size, dtype=np.int32)
        reverses = np.zeros(source.size, bool)
        densified = False
    else:
        angles = np.asarray(geometry.build_azimuthal_azimuths(effective), np.float64)
        owners = [bp._nearest_azimuthal_source_for_backprojection(float(angle), source)
                  for angle in angles]
        indices = np.asarray([item[0] for item in owners], np.int32)
        reverses = np.asarray([item[1] for item in owners], bool)
        densified = True
    return (angles, indices, reverses), dict(provided_spacing_deg=float(provided),
        coverage_spacing_deg=float(coverage), effective_spacing_deg=float(effective),
        source_frames=float(source.size), backprojection_angles=float(angles.size),
        densified=float(densified))


@pytest.mark.parametrize('angles', [(), (0.,), (0., 1., 2.), (0., 90.),
    (0., 30., 30., 179., 180.), (-180., -120., 0., 180., 360.),
    (179., 0., 90., 45., 45.), (0., 20., np.nextafter(90., 0.), 160.)])
def test_numeric_arrays_and_public_records_match_original(angles):
    view = _view(angles)
    expected, expected_metadata = _original(view)
    actual, metadata = prepare_azimuthal_backprojection_arrays(view)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
        assert not got.flags.writeable
    assert metadata == expected_metadata
    records, public_metadata = bp.build_azimuthal_backprojection_plan(view)
    assert public_metadata == expected_metadata
    assert [(item.angle_deg, item.source_index, item.reverse_u) for item in records] == list(
        zip(expected[0].tolist(), expected[1].tolist(), expected[2].tolist()))


@pytest.mark.parametrize('sources', [
    [-1., 1., 89., 91., 179., 181.], [180., 0., 360., -180.],
    [77., 77., -13., 167., 257.], [1.e308, -1.e308, 0.],
    [np.nan, 0., 90.], [0., np.inf, 90.], [-np.inf, 0., 90.]])
def test_nearest_preserves_wrap_first_ties_reversal_and_nonfinite_fallback(sources):
    source = np.asarray(sources, np.float64)
    targets = np.asarray([-1.e308, -720., -180., -90., -1.e-13, 0.,
        np.nextafter(0., 1.), 45., np.nextafter(90., 0.), 90.,
        np.nextafter(90., np.inf), 180., 360., 1.e308], np.float64)
    source_before, target_before = source.copy(), targets.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        expected = [bp._nearest_azimuthal_source_for_backprojection(float(target), source)
                    for target in targets]
        indices, reverses = nearest_azimuthal_sources(targets, source)
    np.testing.assert_array_equal(indices, [item[0] for item in expected])
    np.testing.assert_array_equal(reverses, [item[1] for item in expected])
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(targets, target_before)


@pytest.mark.parametrize('base,spacing,source_count,dense_count', [
    ('transverse', .050581033, 3559, 4749),
    ('sagittal', .054816042, 3284, 4573),
    ('coronal', .054816042, 3284, 4573)])
def test_actual_coarsened_160661_trajectory_matches_every_original_owner(
        base, spacing, source_count, dense_count):
    view = geometry._build_azimuthal_view_info(2911, 3064, 3023,
        base_view=base, azimuth_angle=spacing, azimuthal_native_raster=2048,
        request_token='160661-exact-coarsened')
    assert view.num_slices == source_count
    expected, _ = _original(view)
    actual = azimuthal_backprojection_arrays(view)
    assert actual[0].size == dense_count
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


def test_empty_source_and_multidimensional_angles_rejected():
    with pytest.raises(ValueError, match='No azimuthal source'):
        nearest_azimuthal_sources([0.], [])
    with pytest.raises(ValueError, match='one-dimensional'):
        nearest_azimuthal_sources([[0.]], [0.])


def test_adversarial_midpoint_neighbors_match_original_argmin():
    rng = np.random.default_rng(160661)
    sources = rng.uniform(-720., 720., 257)
    sources[::19] = 37.
    midpoints = (sources[:-1] + sources[1:]) * .5
    targets = np.r_[sources, midpoints, np.nextafter(midpoints, -np.inf),
                    np.nextafter(midpoints, np.inf), rng.uniform(-1080., 1080., 1024)]
    expected = [bp._nearest_azimuthal_source_for_backprojection(float(target), sources)
                for target in targets]
    indices, reverses = nearest_azimuthal_sources(targets, sources)
    np.testing.assert_array_equal(indices, [item[0] for item in expected])
    np.testing.assert_array_equal(reverses, [item[1] for item in expected])


@pytest.mark.parametrize('cached', [False, True])
def test_plan_retains_exact_raw_angles_and_charges_their_bytes(cached):
    from XTA import projection_coverage_cpu as cpu
    view = _view((-180., -120., 0., 180., 360.))
    plan = cpu.prepare_native_pull_plan(view, (view.num_slices, 9, 13), (9, 13, 13),
        max_plan_bytes=4 * 1024**2, cache_plane=cached)
    expected, _ = _original(view)
    np.testing.assert_array_equal(plan.raw_angles_f64, expected[0])
    assert not plan.raw_angles_f64.flags.writeable
    arrays = (plan.integers, plan.coefficients, plan.keys, plan.plan_angles, plan.plan_sources,
        plan.plan_reverses, plan.sorted_angles, plan.owner_order, plan.row_lookup,
        plan.column_lookup, plan.raw_angles_f64)
    assert plan.persistent_bytes == cpu._CONTROL_BYTES + sum(array.nbytes for array in arrays)
