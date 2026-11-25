import math

import numpy as np
import pytest
from scipy import ndimage

from XTA.native_review_flood import BasinTooLarge, claim_window
from tests.slicer_cutoff_store_support import flood_python, load_store_module, unload_store_module


def _bounds(mask):
    bounds = ndimage.find_objects(mask, max_label=1)[0]
    return tuple(s.start for s in bounds), tuple(s.stop for s in bounds)


def _select(shape, before, after, footprint, kept, *, flood, max_voxels, margin=1):
    received = []

    def read(lo, hi):
        received.append((lo, hi))
        slices = tuple(slice(a, b) for a, b in zip(lo, hi))
        return before[slices], after[slices], footprint[slices], kept[slices]

    out, lo, hi = claim_window(shape, _bounds(footprint), read, flood, max_voxels=max_voxels, margin=margin)
    restored = np.zeros(shape, bool)
    restored[tuple(slice(a, b) for a, b in zip(lo, hi))] = out
    return restored, received


def test_basin_finishes_inside_a_large_connected_input():
    shape = (1, 1, 300)
    before = np.ones(shape, dtype=np.uint8)
    before[..., 2:10] = 3
    footprint = before >= 3
    after = np.where(footprint, 0, before).astype(np.uint8)
    kept = np.zeros(shape, dtype=bool)
    kept[..., 1] = True
    kept[..., 10] = True

    clipped, reads = _select(shape, before, after, footprint, kept, flood=flood_python, max_voxels=32)
    assert np.array_equal(clipped, footprint)
    assert all(hi[2] - lo[2] < 32 for lo, hi in reads)


def test_invalidated_local_basin_grows_or_fails_before_large_read():
    shape = (1, 1, 200)
    before = np.full(shape, 2, dtype=np.uint8)
    footprint = np.zeros(shape, dtype=bool)
    footprint[..., 2] = True
    after = before.copy()
    after[footprint] = 0
    kept = np.zeros(shape, dtype=bool)
    reads = []

    def read(lo, hi):
        slices = tuple(slice(a, b) for a, b in zip(lo, hi))
        assert math.prod(b - a for a, b in zip(lo, hi)) <= 20
        reads.append((lo, hi))
        return before[slices], after[slices], footprint[slices], kept[slices]

    with pytest.raises(BasinTooLarge, match="budget 20"):
        claim_window(shape, _bounds(footprint), read, flood_python, max_voxels=20, margin=1)
    assert len(reads) >= 2


@pytest.mark.parametrize("shape", [(7, 11, 17), (1, 19, 26)])
def test_window_matches_legacy_full_basin_in_three_dimensions_and_slices(shape):
    pytest.importorskip("SimpleITK")
    reference = load_store_module()
    rng = np.random.default_rng(617)
    try:
        for _ in range(24):
            before = rng.integers(0, 4, size=shape, dtype=np.uint8)
            kept = before > 2
            anchor = tuple(n // 2 for n in shape)
            before[anchor] = 3
            footprint = np.zeros(shape, dtype=bool)
            footprint[anchor] = True
            kept[anchor] = False
            after = before.copy()
            after[footprint] = 0
            expected = reference._xta_store_flood_in_window(before, after, footprint, kept, "sitk")[0]
            actual, reads = _select(shape, before, after, footprint, kept, flood=None, max_voxels=math.prod(shape))
            assert reads
            np.testing.assert_array_equal(actual, expected)
    finally:
        unload_store_module(reference)
