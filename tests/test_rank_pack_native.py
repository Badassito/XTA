"""Exact multilevel rank bytes and bounded ordered writer ownership."""
from contextlib import contextmanager
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import sweep_backprojection as sweep
from XTA.native_cpu import kernels


def _rank_reference(region, z, top, left, selected):
    fields, offsets, parts = [], [0], []
    for level in selected:
        keep = region >= level
        yy, xx = np.nonzero(keep)
        if not yy.size:
            break
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        part = np.packbits(keep[y0:y1, x0:x1], axis=1, bitorder="little").reshape(-1)
        fields.append((z, top + y0, top + y1, left + x0, left + x1,
                       yy.size, 0, part.size))
        offsets.append(offsets[-1] + part.size)
        parts.append(part)
    return (np.asarray(fields, np.int64).reshape(-1, 8), np.asarray(offsets, np.int64),
            np.concatenate(parts) if parts else np.empty(0, np.uint8))


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("layout", ["unaligned", "reverse_rows", "reversed_stride2"])
def test_pack_rank_levels_preserves_wire_bytes(width, dense, layout):
    rng = np.random.default_rng(width)
    values = (np.full((4, 2 * width + 3), 255, np.uint8) if dense else
              rng.integers(0, 245, (4, 2 * width + 3), dtype=np.uint8))
    region = values[:, 3:3 + width]
    if layout == "reverse_rows":
        region = region[::-1]
    elif layout == "reversed_stride2":
        region = values[::-1, 2 * width:0:-2]
    region.setflags(write=False)
    expected_values = values.copy()
    selected = np.arange(1, 256, dtype=np.int64)
    if not dense:
        selected = selected[::2]
    selected.setflags(write=False)
    budget, lane, shared = sweep._rank_pack_credit((1, *region.shape), 255, len(selected))
    assert budget == 4 * ((width + 7) // 8) * len(selected)
    assert lane == region.size + budget + 8 * (256 * 5 + 10 * len(selected) + 1)
    assert shared == 8 * len(selected) + 64
    params = (3, 11, 29, 255, selected)
    result = kernels.pack_rank_levels(region, *params, budget)
    expected = _rank_reference(region, *params[:3], selected)
    for actual, ref in zip(result, expected):
        assert actual.dtype == ref.dtype
        assert actual.flags.c_contiguous and actual.shape == ref.shape
        np.testing.assert_array_equal(actual, ref)
    np.testing.assert_array_equal(values, expected_values)
    assert not np.shares_memory(result[2], region)
    if dense and width:
        assert result[2].size == budget
        with pytest.raises(MemoryError, match="payload credit"):
            kernels.pack_rank_levels(region, *params, budget - 1)


@pytest.mark.parametrize("level", [-1, 0, 1, 127, 128, 129, 254, 255, 256])
def test_single_level_threshold_preserves_unsigned_order_and_padding(level):
    pattern = np.array([0, 1, 126, 127, 128, 129, 254, 255], np.uint8)
    region = np.tile(pattern, (4, 7))[::-1, 1:]
    actual = kernels.pack_rank_level(region, 1, 4, 2, 35, level)
    crop = region[1:4, 2:35].astype(np.int16)
    np.testing.assert_array_equal(actual, np.packbits(crop >= level, axis=1, bitorder="little"))


def test_pack_rank_levels_rejects_invalid_controls_and_empty_output():
    selected = np.array([1, 3], np.int64)
    for shape in ((0, 1), (4, 0), (4, 1)):
        ranks = np.zeros(shape, np.uint8)
        fields, offsets, payload = kernels.pack_rank_levels(ranks, 5, 6, 7, 3, selected, 0)
        assert fields.shape == (0, 8) and payload.shape == (0,)
        np.testing.assert_array_equal(offsets, [0])
    ranks = np.array([[4]], np.uint8)
    with pytest.raises(ValueError, match="exceeds its declared cutoff rank"):
        kernels.pack_rank_levels(ranks, 0, 0, 0, 3, selected, 1)
    with pytest.raises(ValueError, match="ascending and in bounds"):
        kernels.pack_rank_levels(ranks, 0, 0, 0, 3, selected[::-1], 1)


@pytest.mark.parametrize("failure", [None, "read", "write"])
def test_rank_commit_uses_local_offsets_and_releases_full_owner_before_refill(tmp_path, monkeypatch, failure):
    values = np.asarray([[[3], [2]], [[0], [0]], [[1], [2]]], dtype=np.uint8)
    cutoffs = (40, 60, 80)
    selected = np.arange(1, 4, dtype=np.int64)
    products = tmp_path / "stores"
    products.mkdir()
    shape = values.shape
    view = SimpleNamespace(name="transverse", family="orthogonal")
    read_calls, payload_owners, writes, reader_budgets = [], [], [], []
    result, diagnostics = {}, []
    failure_cause = OSError("controlled rank plane failure")

    @contextmanager
    def reader(*args, memory_bytes, workers, **kwargs):
        assert workers == 1
        reader_budgets.append(memory_bytes)

        def read_plane(z):
            read_calls.append(z)
            assert all(ref() is None for ref in payload_owners)
            if failure == "read" and z == 2:
                raise failure_cause
            return (0, shape[1], 0, shape[2], values[z])

        read_plane.read_crop = read_plane
        yield read_plane

    encode_plane = sweep._encode_rank_plane
    consume = sweep.IncrementalRawBBoxMaskStoreWriter.consume_encoded_fields

    def encode(*args, **kwargs):
        delta, batch = encode_plane(*args, **kwargs)
        if batch is not None:
            payload_owners.append(weakref.ref(batch[2]))
        return delta, batch

    def write(writer, z, fields, payload, *, packed):
        cutoff = int(writer.store_dir.stem.rsplit("_", 1)[1])
        position = cutoffs.index(cutoff)
        expected_fields, offsets, bits = _rank_reference(values[z], z, 0, 0, selected)
        np.testing.assert_array_equal(fields, expected_fields[position:position+1])
        np.testing.assert_array_equal(payload, bits[offsets[position]:offsets[position+1]])
        assert fields[0, 6] == 0 and packed
        writes.append((z, cutoff))
        if failure == "write" and z == 2 and cutoff == 60:
            raise failure_cause
        consume(writer, z, fields, payload, packed=packed)

    monkeypatch.setattr(sweep, "sweep_rank_projection_reader", reader)
    monkeypatch.setattr(sweep, "_encode_rank_plane", encode)
    monkeypatch.setattr(sweep.IncrementalRawBBoxMaskStoreWriter, "consume_encoded_fields", write)
    monkeypatch.setattr(sweep, "_ref", lambda *args: args[-1])
    disk = SimpleNamespace(add=lambda amount: None)

    def project(credit):
        sweep._project_chain(values, None, cutoffs, range(1, 4), set(cutoffs), view, shape,
                             tmp_path, products, products, "test", "key", credit,
                             disk, result, diagnostics, 1)

    _, lane, shared = sweep._rank_pack_credit(shape, 3, 3)
    with pytest.raises(MemoryError, match="one packed output lane"):
        project(lane + shared)
    assert not reader_budgets and not list(products.iterdir())

    request = 524288
    if failure:
        with pytest.raises(OSError) as raised:
            project(request)
        assert raised.value is failure_cause
        assert not result and not list(products.iterdir())
    else:
        project(request)
        assert set(result) == set(cutoffs)
        assert writes == [(0, 40), (0, 60), (0, 80), (2, 40), (2, 60)]
        assert diagnostics[0]["rank_pack_credit_bytes"] + reader_budgets[0] == request
        assert diagnostics[0]["rank_pack_payload_bytes_per_lane"] == 6
        assert all(ref() is None for ref in payload_owners)
    assert reader_budgets == [request - lane - shared]
    assert read_calls == [0, 1, 2]
