"""Rectangle painting must equal the original point raster, including clipping."""
import numpy as np
import pytest

from XTA import sam_bridge_planning as planning
from XTA.sam_observation_masks import immutable_observation_mask


def literal_paint(destination, observation, crop, shift_y, shift_x, *, raster_origin_yx=None):
    ys, xs = np.nonzero(observation.mask_crop)
    origin_y, origin_x = crop[:2] if raster_origin_yx is None else raster_origin_yx
    iy = (np.rint(ys + observation.bbox_yx[0] + shift_y - origin_y).astype(np.int64)
          + int(origin_y) - crop[0])
    ix = (np.rint(xs + observation.bbox_yx[1] + shift_x - origin_x).astype(np.int64)
          + int(origin_x) - crop[1])
    inside = (iy >= 0) & (iy < destination.shape[0]) & (ix >= 0) & (ix < destination.shape[1])
    destination[iy[inside], ix[inside]] = True


def observation(mask, y=-20, x=-13):
    h, w = mask.shape
    return planning.SamObservation('a', 0, 1, 1, (y, x, y + h, x + w), mask)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('shift', [(0., 0.), (.2, -.3), (1.5, -.5), (-100., 80.)])
def test_rectangle_paint_matches_literal_points_on_noncontiguous_prefilled_outputs(packed, shift):
    rng = np.random.default_rng(44)
    mask = rng.random((64, 72)) < .22
    record = observation(immutable_observation_mask(mask) if packed else mask)
    original = rng.random((160, 170)) < .1
    expected, actual = original.copy(), original.copy()
    arguments = (record, (-10, -8, 70, 77), *shift)
    literal_paint(expected[::2, ::2], *arguments, raster_origin_yx=(-31, 41))
    planning._paint_shifted(actual[::2, ::2], *arguments, raster_origin_yx=(-31, 41))
    np.testing.assert_array_equal(actual, expected)


def test_regular_raster_avoids_point_coordinates_but_half_ties_fall_back(monkeypatch):
    record = observation(immutable_observation_mask(np.ones((64, 64), bool)))
    crop = (-25, -22, 80, 90)
    point_calls = []
    original = planning._paint_shifted_points

    def tracked(*args, **kwargs):
        point_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(planning, '_paint_shifted_points', tracked)
    for shift, wanted_calls in [((.25, -.125), 0), ((.5, -.5), 1)]:
        actual = np.zeros((105, 112), bool)
        expected = actual.copy()
        literal_paint(expected, record, crop, *shift, raster_origin_yx=(-31, 7))
        planning._paint_shifted(actual, record, crop, *shift, raster_origin_yx=(-31, 7))
        np.testing.assert_array_equal(actual, expected)
        assert len(point_calls) == wanted_calls


def test_sparse_large_source_tiny_destination_does_not_allocate_axis_vectors(monkeypatch):
    mask = np.zeros((4096, 1), bool)
    mask[0] = mask[-1] = True
    record = observation(immutable_observation_mask(mask))
    actual = np.zeros((2, 2), bool)
    expected = actual.copy()
    literal_paint(expected, record, (0, 0, 2, 2), 0., 0.)
    monkeypatch.setattr(planning.np, 'arange', lambda *_a, **_k: pytest.fail('unbounded axis allocation'))
    planning._paint_shifted(actual, record, (0, 0, 2, 2), 0., 0.)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('shift,origin', [((float('nan'), 0.), (0, 0)),
    ((float('inf'), 0.), (0, 0)), ((.2, .3), (2**54, -(2**54)))])
def test_nonfinite_and_large_geometry_preserve_point_fallback(shift, origin):
    record = observation(np.ones((64, 64), bool))
    expected = np.zeros((90, 90), bool)
    actual = expected.copy()
    with np.errstate(invalid='ignore'):
        literal_paint(expected, record, (-30, -30, 60, 60), *shift, raster_origin_yx=origin)
        planning._paint_shifted(actual, record, (-30, -30, 60, 60), *shift, raster_origin_yx=origin)
    np.testing.assert_array_equal(actual, expected)


def test_nonboolean_destination_preserves_existing_values_and_true_assignment():
    record = observation(np.ones((64, 64), bool))
    actual = np.full((90, 90), 7, np.int16)
    expected = actual.copy()
    literal_paint(expected, record, (-30, -30, 60, 60), .2, .3)
    planning._paint_shifted(actual, record, (-30, -30, 60, 60), .2, .3)
    np.testing.assert_array_equal(actual, expected)


def test_readonly_destination_keeps_original_failure_even_without_inside_pixels():
    record = observation(np.ones((64, 64), bool))
    target = np.zeros((90, 90), bool)
    target.flags.writeable = False
    for function in (literal_paint, planning._paint_shifted):
        with pytest.raises(ValueError):
            function(target, record, (1000, 1000, 1090, 1090), .2, .3)


def test_random_large_sparse_packed_and_dense_paints_match_independent_oracle():
    rng = np.random.default_rng(621)
    for case in range(200):
        h, w = rng.integers(32, 100, size=2).tolist()
        mask = rng.random((h, w)) < rng.choice([0., .001, .1, .9, 1.])
        if case % 2:
            mask = immutable_observation_mask(mask)
        record = observation(mask, *rng.integers(-70, 70, size=2).tolist())
        ch, cw = rng.integers(1, 150, size=2).tolist()
        y, x = rng.integers(-70, 70, size=2).tolist()
        crop = (y, x, y + ch, x + cw)
        origin = tuple(rng.integers(-90, 90, size=2).tolist())
        shifts = rng.uniform(-20., 20., size=2)
        actual = rng.random((ch, cw)) < .2
        expected = actual.copy()
        literal_paint(expected, record, crop, *shifts, raster_origin_yx=origin)
        planning._paint_shifted(actual, record, crop, *shifts, raster_origin_yx=origin)
        np.testing.assert_array_equal(actual, expected, err_msg=f'case {case}')
