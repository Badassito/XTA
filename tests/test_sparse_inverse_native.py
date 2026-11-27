import numpy as np
import pytest

from XTA import sparse_projection as sparse
from XTA.interpolation import CTILE_INDEX_DTYPE


def _reversed_unaligned(values):
    values = np.asarray(values)
    result = np.ndarray(values.shape, dtype=values.dtype,
                        buffer=bytearray(values.nbytes + 1), offset=1)[::-1]
    result[...] = values
    return result


@pytest.mark.parametrize("packed", [False, True])
def test_encoded_inverse_reads_strided_cvol_without_copying(packed):
    records = _reversed_unaligned(np.zeros(3, dtype=CTILE_INDEX_DTYPE))[::2]
    crop = np.array([[0, 231, 0, 3, 0, 255, 1, 0, 1],
                     [7, 0, 1, 0, 0, 1, 0, 5, 0]], np.uint8)
    body = np.packbits(crop != 0, axis=1, bitorder="little") if packed else crop
    encoded = np.full(2 * (body.size + 3), 0xaa, dtype=np.uint8)[::-2]
    encoded[3:] = body.ravel()
    records[0]["kind"] = 1
    records[0]["offset"] = 3
    records[0]["payload_size"] = body.size
    records[0]["payload_nbytes"] = crop.size
    records[0]["y0"], records[0]["y1"] = 5, 7
    records[0]["x0"], records[0]["x1"] = 7, 16
    angles = _reversed_unaligned(np.array([0, 0, 1, 0, 0, -500, 0], np.int64))
    rows = _reversed_unaligned(np.array([5, 6, 5, 5, 4, 5, 6], np.int32))
    columns = _reversed_unaligned(np.array([8, 15, 8, 15, 8, 8, 14], np.int32))
    valid = _reversed_unaligned(np.array([1, 1, 1, 1, 1, 0, 1], dtype=bool))
    for source in (encoded, records, angles, rows, columns, valid):
        source.setflags(write=False)
    observed, reads = sparse._sample_encoded_mask(encoded, records, angles, rows, columns, valid, packed)
    np.testing.assert_array_equal(observed, [1, 0, 0, 1, 0, 0, 1])
    assert observed.dtype == np.uint8
    assert reads == 4

    with pytest.raises(ValueError, match="encoded payload"):
        sparse._sample_encoded_mask(encoded[:3], records, angles, rows, columns, valid, packed)


def test_inverse_owner_prefix_and_fill_preserve_scatter_order_and_capacity():
    keys = _reversed_unaligned(np.array([3, 0, 3, 2, 1, 0, 2], np.int64))
    positions = _reversed_unaligned(np.array([9, 6, 8, 2, 0, 1, 12], np.uint32))
    keys.setflags(write=False)
    positions.setflags(write=False)
    counts = _reversed_unaligned(np.zeros(4, np.uint32))
    sparse._count_keys(keys, counts)
    np.testing.assert_array_equal(counts, [2, 1, 2, 2])
    offsets = sparse._prefix_counts(counts)
    np.testing.assert_array_equal(offsets, [0, 2, 3, 5, 7])
    assert offsets.dtype == np.uint32

    counts[:] = offsets[:-1]
    owners = _reversed_unaligned(np.full(int(offsets[-1]), 91, np.uint32))
    sparse._fill_owners(keys, positions, counts, owners)
    np.testing.assert_array_equal(owners, [6, 1, 0, 2, 12, 9, 8])
    np.testing.assert_array_equal(counts, offsets[1:])
    with pytest.raises(ValueError, match="uint32 owner capacity"):
        sparse._prefix_counts(np.array([0xffffffff, 1], dtype=np.uint32))
    with pytest.raises(ValueError, match="map key"):
        sparse._count_keys(np.array([-1], dtype=np.int32), counts)
    with pytest.raises(ValueError, match="owner cursor"):
        sparse._fill_owners(keys[:1], positions[:1], counts, owners)
    np.testing.assert_array_equal(owners, [6, 1, 0, 2, 12, 9, 8])


@pytest.mark.parametrize("base", [0, 1, 2])
def test_sparse_inverse_scatter_matches_bytes_and_counts_for_each_base(base):
    out_t, out_h, out_w = shape = (4, 3, 11)
    plane_width = out_h if base == 2 else out_w
    crop = _reversed_unaligned(np.array([[11, 0], [1, 2]], dtype=np.uint8))
    selected_rows = [[], [1, 2], [2]]
    selected_owners = [[], [], [], [], [0, 1, plane_width + 2], [2, 2]]
    row_offsets = _reversed_unaligned(np.cumsum([0] + [len(row) for row in selected_rows], dtype=np.uint32))
    frames = _reversed_unaligned(np.array([value for row in selected_rows for value in row], np.uint32))
    key_offsets = _reversed_unaligned(np.cumsum([0] + [len(row) for row in selected_owners], dtype=np.uint32))
    owners = _reversed_unaligned(np.array([value for row in selected_owners for value in row], np.uint32))
    for source in (crop, row_offsets, frames, key_offsets, owners):
        source.setflags(write=False)
    packed = np.zeros(2 * out_t * out_h * ((out_w + 7) // 8), np.uint8)[::-2]
    bounds = _reversed_unaligned(np.tile([out_h, 0, out_w, 0], (out_t, 1)).astype(np.int32))
    counts = _reversed_unaligned(np.zeros(out_t, np.uint64))

    expected = np.zeros(shape, dtype=bool)
    expected_bounds = bounds.copy()
    candidates = 0
    for row, col in np.argwhere(crop):
        for frame in selected_rows[1 + row]:
            for owner in selected_owners[4 + col]:
                v, u = divmod(owner, plane_width)
                z, y, x = [(frame, v, u), (v, frame, u), (v, u, frame)][base]
                candidates += 1
                expected[z, y, x] = True
                expected_bounds[z, 0] = min(expected_bounds[z, 0], y)
                expected_bounds[z, 1] = max(expected_bounds[z, 1], y + 1)
                expected_bounds[z, 2] = min(expected_bounds[z, 2], x)
                expected_bounds[z, 3] = max(expected_bounds[z, 3], x + 1)

    arguments = (crop, 1, 1, 1, key_offsets, owners, row_offsets, frames,
                 plane_width, 3, base, out_h, out_w, packed, bounds, counts)
    assert sparse._scatter_upright_crop(*arguments) == (3, candidates, int(expected.sum()))
    np.testing.assert_array_equal(packed, np.packbits(expected, axis=2, bitorder="little").ravel())
    np.testing.assert_array_equal(bounds, expected_bounds)
    np.testing.assert_array_equal(counts, expected.sum(axis=(1, 2)))
    assert sparse._scatter_upright_crop(*arguments) == (3, candidates, 0)

    original = packed.copy()
    bad_offsets = key_offsets.copy()
    bad_offsets[5] = len(owners) + 1
    with pytest.raises(ValueError, match="offsets exceed"):
        sparse._scatter_upright_crop(*arguments[:4], bad_offsets, *arguments[5:])
    np.testing.assert_array_equal(packed, original)
