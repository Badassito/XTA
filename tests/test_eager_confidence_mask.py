"""Exact eager score masking, including borrowed buffer views and aliases."""
import numpy as np
import pytest

from XTA.confidence_capture_cpu import mask_eager_scores_and_bound


def scalar_reference(mask, scores):
    """Track the footprint as it was written in forward z/y/x order."""
    bounds = np.zeros((scores.shape[0], 4), dtype=np.int64)
    for z in range(scores.shape[0]):
        first_y = scores.shape[1]
        first_x = scores.shape[2]
        stop_y = stop_x = 0
        for y in range(scores.shape[1]):
            for x in range(scores.shape[2]):
                if not mask[z, y, x]:
                    scores[z, y, x] = 0
                elif scores[z, y, x]:
                    first_y = min(first_y, y)
                    first_x = min(first_x, x)
                    stop_y = y + 1
                    stop_x = max(stop_x, x + 1)
        if stop_y:
            bounds[z] = first_y, stop_y, first_x, stop_x
    return bounds


def off_vector_boundary(shape):
    backing = np.empty(int(np.prod(shape)) + 16, dtype=np.uint8)
    offset = (1 - backing.ctypes.data) % 16
    return backing[offset:offset + int(np.prod(shape))].reshape(shape)


@pytest.mark.parametrize('width', [0, 1, 15, 16, 17, 31, 32, 33, 65, 517])
@pytest.mark.parametrize('mask_dtype', [np.uint8, np.bool_])
def test_unaligned_rows_and_tails_match_the_scalar_oracle(width, mask_dtype):
    mask_bytes = off_vector_boundary((3, 5, width))
    scores = off_vector_boundary(mask_bytes.shape)
    assert not width or mask_bytes.ctypes.data % 16 == scores.ctypes.data % 16 == 1
    rng = np.random.default_rng(8514)
    mask_bytes[:] = rng.choice(np.asarray([0, 0, 0, 1, 2, 128, 255], np.uint8), size=mask_bytes.shape)
    scores[:] = rng.choice(np.asarray([0, 1, 127, 128, 255], np.uint8), size=mask_bytes.shape)
    mask_bytes[1] = 0
    mask_bytes[2] = 0
    if width:
        mask_bytes[2, -1, -1] = 128
        scores[2, -1, -1] = 255
    mask = mask_bytes.view(mask_dtype)
    mask.flags.writeable = False
    expected = scores.copy()
    before = mask_bytes.copy()
    bounds = mask_eager_scores_and_bound(mask, scores)
    assert bounds.dtype == np.int64
    np.testing.assert_array_equal(bounds, scalar_reference(mask, expected))
    np.testing.assert_array_equal(scores, expected)
    np.testing.assert_array_equal(mask_bytes, before)


def test_signed_strides_match_the_scalar_oracle():
    rng = np.random.default_rng(8515)
    mask = rng.choice(np.asarray([0, 2, 128], np.uint8), size=(3, 5, 67))[::-1, :, ::-2].view(bool)
    mask.flags.writeable = False
    backing = rng.integers(0, 256, (3, 5, 68), dtype=np.uint8)
    expected = backing.copy()
    oracle = scalar_reference(mask, expected[:, ::-1, 1::2])
    bounds = mask_eager_scores_and_bound(mask, backing[:, ::-1, 1::2])
    np.testing.assert_array_equal(bounds, oracle)
    np.testing.assert_array_equal(backing, expected)


@pytest.mark.parametrize('shift', [0, 1, -1, 33])
def test_aliases_retain_scalar_row_order(shift):
    shape = (3, 5, 33)
    size = int(np.prod(shape))
    rng = np.random.default_rng(8516)
    backing = rng.integers(1, 256, size + abs(shift), dtype=np.uint8)
    backing[0] = 0
    backing[7 + max(0, shift)] = 0
    expected = backing.copy()

    def views(storage):
        low = max(0, -shift)
        mask = storage[low:low + size].reshape(shape).view(bool)
        mask.flags.writeable = False
        scores = storage[low + shift:low + shift + size].reshape(shape)
        return mask, scores

    oracle = scalar_reference(*views(expected))
    bounds = mask_eager_scores_and_bound(*views(backing))
    np.testing.assert_array_equal(bounds, oracle)
    np.testing.assert_array_equal(backing, expected)
