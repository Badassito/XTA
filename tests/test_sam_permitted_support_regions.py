"""Sparse permitted contributions must not scan unrelated owner-plane regions."""
import numpy as np
import pytest

from XTA.sam_interpolation_contracts import _include_permitted_support


class CountedOwner(np.ndarray):
    def __new__(cls, shape, comparisons):
        value = np.full(shape, -1, dtype=np.int32).view(cls)
        value.comparisons = comparisons
        return value
    def __array_finalize__(self, original):
        self.comparisons = getattr(original, 'comparisons', None)
    def __eq__(self, other):
        self.comparisons.append(self.shape)
        return np.asarray(self) == other
    def __ne__(self, other):
        self.comparisons.append(self.shape)
        return np.asarray(self) != other


@pytest.mark.parametrize('shape,region', [
    ((23, 29), (slice(7, 9), slice(11, 15))),
    ((1, 17), (slice(0, 1), slice(5, 8))),
    ((17, 1), (slice(5, 8), slice(0, 1))),
    ((1, 1), (slice(0, 1), slice(0, 1))),
])
def test_owner_comparisons_are_bounded_to_actual_contributor_support(shape, region):
    plane = np.zeros(shape, bool)
    plane[region] = True
    comparisons = []
    first, second = CountedOwner(shape, comparisons), CountedOwner(shape, comparisons)
    _include_permitted_support(first, second, plane, 3, 7)
    measured = list(comparisons)
    assert len(measured) == 5 and set(measured) == {plane[region].shape}
    expected_first, expected_second = np.full(shape, -1), np.full(shape, -1)
    expected_first[region], expected_second[region] = 3, 7
    np.testing.assert_array_equal(np.asarray(first), expected_first)
    np.testing.assert_array_equal(np.asarray(second), expected_second)


@pytest.mark.parametrize('owners', [(1, 2), (1, -2), (-2, -2)])
def test_empty_contributor_does_not_read_or_change_owner_state(owners):
    shape = (19, 31)
    comparisons = []
    first, second = CountedOwner(shape, comparisons), CountedOwner(shape, comparisons)
    first[1, 2], second[1, 2] = 17, -2
    before = np.asarray(first).copy(), np.asarray(second).copy()
    _include_permitted_support(first, second, np.zeros(shape, bool), *owners)
    assert not comparisons
    np.testing.assert_array_equal(np.asarray(first), before[0])
    np.testing.assert_array_equal(np.asarray(second), before[1])


def test_bbox_holes_do_not_erase_other_support_or_mutate_contributor():
    shape = (9, 13)
    first, second = np.full(shape, -1, np.int32), np.full(shape, -1, np.int32)
    first[4, 6], second[4, 6] = 9, 12
    plane = np.zeros(shape, bool)
    plane[2, 3] = plane[6, 9] = True
    original = plane.copy()
    plane.setflags(write=False)
    _include_permitted_support(first, second, plane, -2)
    assert first[4, 6] == 9 and second[4, 6] == 12
    assert first[2, 3] == second[2, 3] == -2
    assert first[6, 9] == second[6, 9] == -2
    assert first[0, 0] == second[0, 0] == -1
    np.testing.assert_array_equal(plane, original)
