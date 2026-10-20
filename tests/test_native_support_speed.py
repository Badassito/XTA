import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from xta_native_filter import PreviewSupportCheck


def brute_support(a, shape):
    result = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        bounds = tuple(slice(i*n//m, ((i+1)*n+m-1)//m)
                       for i, n, m in zip(index, a.shape, shape))
        result[index] = np.any(a[bounds])
    return result


@pytest.mark.parametrize('full,low', [
    ((7, 11, 13), (3, 4, 5)),
    ((3, 7, 5), (3, 7, 5)),
    ((5, 11, 13), (2, 11, 4)),
    ((5, 11, 13), (5, 3, 13)),
    ((7, 11, 13), (1, 1, 1)),
    ((1, 1, 1), (1, 1, 1)),
])
@pytest.mark.parametrize('density', [0, .01, .5, 1])
def test_support_matches_direct_footprints_across_slabs(full, low, density):
    rng = np.random.default_rng(8123)
    native = (rng.random(full) < density).astype(np.uint8)
    expected = brute_support(native, low)
    check = PreviewSupportCheck(full, expected)
    for z in range(0, full[0], 2):
        check.observe(z, native[z:z+2])
    np.testing.assert_array_equal(check.seen, expected)
    result = check.finish()
    assert result['unsupported_preview_foreground_cells'] == 0
    assert result['native_support_cells_absent_from_preview'] == 0


def test_support_includes_both_sides_of_fractional_boundaries():
    full, low = (7, 11, 13), (3, 4, 5)
    for axis in range(3):
        native = np.zeros(full, np.uint8)
        position = [0, 0, 0]
        position[axis] = full[axis] // low[axis]
        native[tuple(position)] = 1
        check = PreviewSupportCheck(full, np.zeros(low, np.uint8))
        check.observe(0, native)
        expected = brute_support(native, low)
        np.testing.assert_array_equal(check.seen, expected)
        assert check.finish()['native_support_cells_absent_from_preview'] == np.count_nonzero(expected)


def test_support_rejects_preview_foreground_without_native_support():
    original = np.zeros((2, 3, 4), np.uint8)
    original[1, 2, 3] = 1
    check = PreviewSupportCheck((5, 7, 9), original)
    check.observe(0, np.zeros((5, 7, 9), np.uint8))
    with pytest.raises(ValueError, match='1 original preview foreground cells have NO'):
        check.finish()


@pytest.mark.parametrize('density', [0, .01, .5, 1])
def test_point_support_matches_direct_footprints_with_duplicates(density):
    full, low = (7, 11, 13), (3, 4, 5)
    native = np.random.default_rng(123).random(full) < density
    points = np.nonzero(native)
    points = tuple(np.concatenate((axis, axis[:5])) for axis in points)
    check = PreviewSupportCheck(full, np.zeros(low, np.uint8))
    check.observe_points(*points)
    np.testing.assert_array_equal(check.seen, brute_support(native, low))


def test_point_support_accepts_scalar_slice_coordinate():
    full, low = (7, 11, 13), (3, 4, 5)
    native = np.zeros(full, np.uint8)
    native[2, [3, 4], [5, 6]] = 1
    check = PreviewSupportCheck(full, np.zeros(low, np.uint8))
    check.observe_points(2, np.array([3, 4]), np.array([5, 6]))
    np.testing.assert_array_equal(check.seen, brute_support(native, low))


def test_support_handles_nonbinary_uint8_foreground():
    native = np.random.default_rng(731).integers(0, 256, (7, 11, 13), dtype=np.uint8)
    expected = brute_support(native, (3, 4, 5))
    check = PreviewSupportCheck(native.shape, expected)
    check.observe(0, native)
    np.testing.assert_array_equal(check.seen, expected)
