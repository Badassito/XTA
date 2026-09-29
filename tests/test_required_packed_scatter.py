"""Required compiled scatter preserves duplicate-coordinate bit accumulation."""
import numpy as np
import pytest
from unittest import mock

from XTA import backprojection
from tests.reference_backends.packed_scatter import scatter


def test_compiled_scatter_matches_oracle_for_strides_duplicates_and_edge_bits():
    for width in (1, 7, 8, 9, 33):
        rng = np.random.default_rng(width)
        t = rng.integers(0, 3, 128, dtype=np.int32)[::-1]
        y = rng.integers(0, 5, 128, dtype=np.int32)[::-1]
        x = rng.integers(0, width, 128, dtype=np.int32)[::-1]
        packed_w = (width + 7) // 8
        expected = np.zeros(3 * 5 * packed_w, np.uint8)
        actual = expected.copy()
        scatter(expected, t, y, x, out_h=5, packed_w=packed_w)
        backprojection._or_tilted_azimuthal_coordinates_into_packed(
            actual, t, y, x, out_h=5, packed_w=packed_w)
        np.testing.assert_array_equal(actual, expected)


def test_compiled_scatter_failure_is_not_replayed_on_cpu():
    destination = np.zeros(8, np.uint8)
    coordinates = np.zeros(1, np.int32)
    with mock.patch.object(backprojection, '_numba_or_tilted_azimuthal_coordinates_into_packed',
                           side_effect=RuntimeError('compiled scatter failed')):
        with pytest.raises(RuntimeError, match='compiled scatter failed'):
            backprojection._or_tilted_azimuthal_coordinates_into_packed(
                destination, coordinates, coordinates, coordinates, out_h=1, packed_w=1)
    np.testing.assert_array_equal(destination, 0)
