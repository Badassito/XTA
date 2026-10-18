"""Independent overlap and floating-lattice checks for rectangle painting."""
import numpy as np
import pytest

from XTA import sam_bridge_planning as planning
from tests.test_sam_shifted_paint import literal_paint


@pytest.mark.parametrize('layout', ['contiguous', 'transposed', 'reversed'])
@pytest.mark.parametrize('shift', [(0., 1.), (0., -1.), (1., 0.), (-1., 0.), (2.25, -3.125)])
def test_rectangle_paint_preserves_frozen_point_semantics_when_source_aliases_destination(monkeypatch, layout, shift):
    original = np.random.default_rng(738).random((96, 96)) < .2
    expected, actual = original.copy(), original.copy()

    def view(array):
        return array.T if layout == 'transposed' else array[::-1, ::-1] if layout == 'reversed' else array

    wanted, target = view(expected), view(actual)
    bbox = (16, 10, 48, 58)
    source_slice = (slice(16, 48), slice(10, 58))
    expected_observation = planning.SamObservation('alias', 0, 1, 1, bbox, wanted[source_slice])
    actual_observation = planning.SamObservation('alias', 0, 1, 1, bbox, target[source_slice])
    literal_paint(wanted, expected_observation, (0, 0, 96, 96), *shift)
    monkeypatch.setattr(planning, '_paint_shifted_points',
        lambda *_args, **_kwargs: pytest.fail('Consecutive large alias raster should use rectangle path'))
    planning._paint_shifted(target, actual_observation, (0, 0, 96, 96), *shift)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('shift', [np.nextafter(.5, 0.), np.nextafter(.5, 1.), -.5, 2.**-12, -(2.**-12)])
def test_large_safe_geometry_and_near_half_ties_keep_literal_operation_order(sign, shift):
    random = np.random.default_rng(113)
    mask = random.random((64, 72)) < .2
    top, left = sign * (2**40 - 128), -sign * (2**40 - 128)
    observation = planning.SamObservation('large', 0, 1, 1,
        (top, left, top + 64, left + 72), mask)
    crop = (top - 10, left - 12, top + 86, left + 84)
    origin = (-sign * 2**40, sign * 2**40)
    actual = random.random((96, 96)) < .1
    expected = actual.copy()
    literal_paint(expected, observation, crop, shift, -shift, raster_origin_yx=origin)
    planning._paint_shifted(actual, observation, crop, shift, -shift, raster_origin_yx=origin)
    np.testing.assert_array_equal(actual, expected)
