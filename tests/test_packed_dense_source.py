"""Independent crop/bitstream parity and bounded borrowed-source ownership."""
import gc
from pathlib import Path
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import packed_dense_source as dense
from XTA.interpolation import (INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter,
                                RawBBoxMaskStore, write_raw_bbox_mask_store)


def _oracle(source, first, count):
    records, payload = [], bytearray()
    for z in range(first, first + count):
        plane = np.asarray(source[z]) != 0
        ys, xs = np.nonzero(plane)
        if not len(xs):
            records.append((z, 0, 0, 0, 0, 0, len(payload), 0))
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        data = np.packbits(plane[y0:y1, x0:x1], axis=1, bitorder='little').tobytes()
        records.append((z, y0, y1, x0, x1, int(plane.sum()), len(payload), len(data)))
        payload.extend(data)
    return records, bytes(payload)


def _actual(records):
    return [(r.z, r.y0, r.y1, r.x0, r.x1, r.foreground, r.offset, r.size) for r in records]


@pytest.mark.parametrize('width', [1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 65, 129])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'fortran', 'reversed'])
def test_exact_crop_payload_for_native_strided_and_reversed_masks(width, layout):
    rng = np.random.default_rng(914)
    data = np.zeros((7, 11, width), np.uint8)
    selected = rng.random(data.shape) < .22
    data[selected] = rng.choice(np.array([1, 7, 128, 255], np.uint8), selected.sum())
    data[1] = 0
    data[3] = 0
    data[3, -1, -1] = 255
    if layout == 'strided':
        base = np.zeros((14, 22, width * 3), np.uint8)
        source = base[::2, ::2, ::3]
        source[:] = data
    elif layout == 'fortran':
        source = np.asfortranarray(data)
    elif layout == 'reversed':
        source = data[::-1, ::-1, ::-1]
    else:
        source = data
    before = source.copy()
    records, payload = dense.encode_dense_u8_packed_block(source, 1, 5)
    expected_records, expected_payload = _oracle(source, 1, 5)
    assert _actual(records) == expected_records
    assert payload.tobytes() == expected_payload
    assert source.flags.writeable
    np.testing.assert_array_equal(source, before)


def test_simd_membership_masks_cover_every_byte_pattern_and_uint8_nonzero_value():
    data = np.empty((1, 256, 8), np.uint8)
    for mask in range(256):
        for bit in range(8):
            data[0, mask, bit] = (17 + 29 * bit) if mask & (1 << bit) else 0
    records, payload = dense.encode_dense_u8_packed_block(data, 0, 1)
    expected_records, expected_payload = _oracle(data, 0, 1)
    assert _actual(records) == expected_records
    assert payload.tobytes() == expected_payload == bytes(range(1, 256))


def test_reusable_payload_slot_matches_reference_index_and_chunk_bytes(tmp_path):
    rng = np.random.default_rng(151)
    data = rng.integers(0, 2, (9, 23, 47), dtype=np.uint8)
    data[3:5] = 0
    data[7] = 0
    data[7, 3, 8] = 1
    capacity = 3 * 23 * ((47 + 7) // 8)
    slot = np.empty(capacity, np.uint8)
    writer = IncrementalRawBBoxMaskStoreWriter(shape=data.shape,
        store_dir=tmp_path / 'encoded.cvol', format_name=INTERNAL_PACKED_CVOL_FORMAT,
        desc='dense block parity')
    try:
        for first in range(0, len(data), 3):
            records, payload = dense.encode_dense_u8_packed_block(data, first, 3,
                payload_buffer=slot, max_payload_bytes=capacity)
            if payload.size:
                assert np.shares_memory(slot, payload)
            writer.consume_encoded_block(first, records, payload, packed=True)
            slot.fill(0xCD)  # A finished synchronous append must own its bytes.
        actual_stats = writer.finalize()
    except BaseException as error:
        writer.abort(error)
        writer.discard()
        raise
    expected_stats = write_raw_bbox_mask_store(data, tmp_path / 'reference.cvol',
        format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='independent dense reference', workers=1)
    assert (tmp_path / 'encoded.cvol' / 'chunks.bin').read_bytes() == (tmp_path / 'reference.cvol' / 'chunks.bin').read_bytes()
    assert (tmp_path / 'encoded.cvol' / 'index.bin').read_bytes() == (tmp_path / 'reference.cvol' / 'index.bin').read_bytes()
    for key in ('foreground_voxels', 'raw_payload_bytes', 'segment_extent_ijk', 'nonempty_slices'):
        assert actual_stats[key] == expected_stats[key]


def test_results_do_not_retain_source_mapping_with_gc_disabled(tmp_path):
    source = np.memmap(tmp_path / 'native.dat', mode='w+', dtype=np.uint8, shape=(5, 19, 27))
    source[:] = 0
    source[2, 1:18, 3:26] = 255
    owner = weakref.ref(source._mmap)
    enabled = gc.isenabled()
    gc.disable()
    try:
        records, payload = dense.encode_dense_u8_packed_block(source, 0, 5)
        source = None
        assert owner() is None
        assert records[2].foreground == 17 * 23
        assert np.count_nonzero(payload) > 0
    finally:
        if enabled:
            gc.enable()


def test_worst_case_and_control_bounds_are_checked_before_scanning():
    virtual = np.broadcast_to(np.uint8(0), (64, 32768, 32768))
    with mock.patch.object(dense, '_dense_strided_metadata', side_effect=AssertionError('scanned unadmitted volume')):
        with pytest.raises(MemoryError, match='workspace'):
            dense.encode_dense_u8_packed_block(virtual, 0, 64)
    with pytest.raises(ValueError, match='allowance'):
        dense.encode_dense_u8_packed_block(np.zeros((65, 1, 1), np.uint8), 0, 65)
    with pytest.raises(ValueError, match='allowance'):
        dense.encode_dense_u8_packed_block(np.zeros((1, 1, 1), np.uint8), 0, 1,
            max_payload_bytes=dense.DENSE_PACKED_PAYLOAD_BYTES + 1)


def test_payload_cannot_alias_source_and_undersized_slots_fail_without_mutation():
    source = np.ones((2, 5, 9), np.uint8)
    before = source.copy()
    with pytest.raises(ValueError, match='alias'):
        dense.encode_dense_u8_packed_block(source, 0, 2, payload_buffer=source.reshape(-1))
    with pytest.raises(MemoryError, match='slot'):
        dense.encode_dense_u8_packed_block(source, 0, 2, payload_buffer=np.empty(1, np.uint8))
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize('first,count', [(-1, 1), (0, -1), (2, 2), (4, 0)])
def test_invalid_slice_windows_are_rejected(first, count):
    with pytest.raises(ValueError):
        dense.encode_dense_u8_packed_block(np.zeros((3, 5, 7), np.uint8), first, count)


def test_empty_block_and_readonly_source_preserve_caller_flags():
    source = np.zeros((3, 5, 7), np.uint8)
    source.setflags(write=False)
    records, payload = dense.encode_dense_u8_packed_block(source, 3, 0, max_payload_bytes=0)
    assert records == () and payload.size == 0
    assert not source.flags.writeable
