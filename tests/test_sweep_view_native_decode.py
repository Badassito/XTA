"""The batched, GIL-free journal frame decoder equals the v26.0.0 stream reader byte for byte.

Journals are written by the real NativeTaskFanoutWriter (multi-range tasks,
strips, aliases, explicit empties, padding, absent/zero/real score lanes,
scalar-owned floors, background encoding, integrity mode). Every reader API
and the projection rank input are compared against a verbatim copy of the
original seek/read reader on both the Python and the native route, including
partial yields and messages on corrupted payloads.
"""
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from XTA import sweep_views as views
from XTA import sweep_backprojection as backprojection
from XTA import sweep_view_decode as decode


VIEW = 'native_decode__tta_a0'


def reference_crops(reader, z, include_scores, metrics):
    """The v26.0.0 NativeViewReader._iter_crops, on a private stream per row."""
    for row, path in reader._rows(z):
        if row['state'] == 1:
            continue
        try:
            with open(path, 'rb') as stream:
                stream.seek(int(row['offset']))
                consumed = 0
                verify = reader._verification[path]
                frame_digest = hashlib.sha256() if verify else None
                for _ in range(int(row['chunks'])):
                    header = stream.read(views._CHUNK.size)
                    if len(header) != views._CHUNK.size:
                        raise views.SweepViewIntegrityError('Truncated native crop header')
                    y0, y1, x0, x1, ml, sl, mc, sc, mraw, sraw = views._CHUNK.unpack(header)
                    area = (y1-y0)*(x1-x0)
                    if not (int(row['y0']) <= y0 < y1 <= int(row['y1']) <= reader.shape[1] and int(row['x0']) <= x0 < x1 <= int(row['x1']) <= reader.shape[2]) or area > views._CHUNK_BYTES or mraw != (area+7)//8 or sraw != (area if row['score'] else 0):
                        raise views.SweepViewIntegrityError('Native crop exceeds bounded geometry')
                    if ml > mraw+1024 or sl > sraw+1024 or bool(sl) != bool(row['score']) or ml+sl+views._CHUNK.size > int(row['length'])-consumed:
                        raise views.SweepViewIntegrityError('Native crop encoded lengths are invalid')
                    mb = stream.read(ml)
                    if include_scores or verify:
                        sb = stream.read(sl)
                    else:
                        stream.seek(sl, 1)
                        sb = None
                    metrics['payload_read_bytes'] += views._CHUNK.size+len(mb)+(0 if sb is None else len(sb))
                    if len(mb) != ml or (sb is not None and len(sb) != sl):
                        raise views.SweepViewIntegrityError('Native crop bytes changed')
                    if verify:
                        prefix = struct.pack('<8I', y0, y1, x0, x1, ml, sl, mraw, sraw)
                        checksum = zlib.crc32(prefix)
                        if zlib.crc32(mb, checksum) != mc or zlib.crc32(sb, checksum) != sc:
                            raise views.SweepViewIntegrityError('Native crop bytes changed')
                        frame_digest.update(header); frame_digest.update(mb); frame_digest.update(sb)

                    def inflate(raw, expected):
                        decoder = zlib.decompressobj(); value = decoder.decompress(raw, expected+1)
                        if len(value) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                            raise views.SweepViewIntegrityError('Native crop compression is malformed')
                        return value
                    mask = np.unpackbits(np.frombuffer(inflate(mb, mraw), np.uint8), bitorder='little', count=area).reshape(y1-y0, x1-x0)
                    score = None if not include_scores or not row['score'] else np.frombuffer(inflate(sb, sraw), np.uint8).reshape(mask.shape)
                    mask.setflags(write=False)
                    consumed += views._CHUNK.size+ml+sl
                    yield y0, y1, x0, x1, mask, score
                if consumed != int(row['length']):
                    raise views.SweepViewIntegrityError('Native frame has unindexed payload bytes')
                if verify and frame_digest.digest() != row['sha256'].tobytes():
                    raise views.SweepViewIntegrityError('Native frame checksum differs from its acknowledged index')
        except (OSError, zlib.error, ValueError, struct.error) as exc:
            raise views.SweepViewIntegrityError('Cannot read acknowledged native crop') from exc


def collect(iterator):
    """(crops yielded, (exception type, message) or None)."""
    crops = []
    try:
        for crop in iterator:
            crops.append(crop)
    except (views.SweepViewIntegrityError, RuntimeError, IndexError) as error:
        return crops, (type(error), str(error))
    return crops, None


def assert_same_crops(actual, expected, *, scores=True):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert len(got) == len(want)
        assert got[:4] == want[:4] and all(type(value) is int for value in got[:4])
        for got_array, want_array in zip(got[4:], want[4:]):
            if want_array is None:
                assert got_array is None
                continue
            assert got_array.dtype == want_array.dtype == np.uint8
            assert got_array.shape == want_array.shape
            assert got_array.tobytes() == want_array.tobytes()
            assert not got_array.flags.writeable


def library_path():
    if os.name != 'nt':
        return None
    import importlib.util
    spec = importlib.util.find_spec('torchvision')
    if spec is None or not spec.origin:
        return None
    candidate = Path(spec.origin).parent / 'zlib.dll'
    return str(candidate) if candidate.exists() else None


@pytest.fixture
def family(tmp_path, monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_ID', 'native-decode')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_GIB', '1')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_VERIFY', '0')
    return tmp_path / 'native_fanout'


@pytest.fixture(params=['python', 'native'])
def route(request, monkeypatch):
    if request.param == 'python':
        monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_DECODE', 'python')
        return request.param
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_DECODE', 'auto')
    monkeypatch.setenv('YOLO_TTA_CONFIDENCE_NATIVE_INFLATE', 'auto')
    if os.name == 'nt':
        library = library_path()
        if library is None:
            pytest.skip('no zlib library to qualify on this Windows host')
        monkeypatch.setenv('XTA_QUALIFY_NATIVE_ZLIB', '1')
        monkeypatch.setenv('XTA_TEST_NATIVE_ZLIB_LIBRARY', library)
    decoder, receipt = decode.native_view_decoder()
    if decoder is None:
        pytest.skip(f'native journal decode unavailable: {receipt}')
    return request.param


def build(root, rng, shape, cutoffs, *, tasks, padding, scalar_owned, background):
    """Write and seal one random family; returns the decision."""
    receipts = []
    owned = cutoffs[0] if scalar_owned else None
    edges = sorted(set([0, shape[0], *rng.integers(1, shape[0], tasks-1).tolist()]))
    for start, stop in zip(edges, edges[1:]):
        count = stop-start
        pads = 0 if scalar_owned else int(rng.integers(0, padding+1))
        writer = views.NativeTaskFanoutWriter(root, 'detector', VIEW, shape, cutoffs, frame_start=start,
            expected_frames=count, padding_count=pads, backend='test', scalar_owned_cutoff=owned)
        if background:
            writer.start_background_encoding(max_pending_bytes=int(rng.integers(0, 4096)))
        previous = {}
        for position, cent in enumerate(cutoffs):
            if cent == owned:
                writer.put_scalar_owned_range(cent, start, stop, prediction_counts=np.zeros(count, np.int64),
                                              frames_with_predictions=np.zeros(count, np.int64))
                continue
            lane = bool(rng.integers(0, 2))
            z = start
            while z < stop:
                kind = rng.choice(['crop', 'crop', 'crop', 'nested', 'full', 'empty', 'run', 'alias'])
                source = previous.get(z)
                if kind == 'alias' and source is not None and source != owned:
                    writer.put_alias(cent, z, source_cutoff=source, prediction_count=int(rng.integers(0, 4)))
                    z += 1
                    continue
                if kind == 'run':
                    run = int(rng.integers(1, stop-z+1))
                    writer.put_empty_range(cent, z, z+run, score_lane=lane)
                    for frame in range(z, z+run):
                        previous[frame] = cent
                    z += run
                    continue
                if kind == 'empty':
                    empty = np.empty((0, 0), np.uint8)
                    writer.put(cent, z, empty, empty if lane else None, prediction_count=0)
                elif kind == 'full':
                    plane = (rng.random(shape[1:]) < rng.random()*0.3).astype(np.uint8)
                    scores = rng.integers(0, 256, shape[1:], dtype=np.uint8) if lane else None
                    writer.put(cent, z, plane, scores, prediction_count=1)
                else:
                    y0 = int(rng.integers(0, shape[1])); y1 = int(rng.integers(y0+1, shape[1]+1))
                    x0 = int(rng.integers(0, shape[2])); x1 = int(rng.integers(x0+1, shape[2]+1))
                    mask = (rng.random((y1-y0, x1-x0)) < rng.choice([0.0, 0.05, 0.5, 1.0])).astype(np.uint8)
                    if lane:
                        scores = (np.zeros(mask.shape, np.uint8) if rng.random() < 0.2
                                  else rng.integers(0, 256, mask.shape, dtype=np.uint8))
                    else:
                        scores = None
                    writer.put(cent, z, mask, scores, bbox=(y0, y1, x0, x1), prediction_count=1)
                previous[z] = cent
                z += 1
            for ordinal in range(pads):
                frame = int(rng.integers(0, shape[0]))
                y0 = int(rng.integers(0, shape[1])); y1 = int(rng.integers(y0+1, shape[1]+1))
                x0 = int(rng.integers(0, shape[2])); x1 = int(rng.integers(x0+1, shape[2]+1))
                mask = (rng.random((y1-y0, x1-x0)) < 0.4).astype(np.uint8)
                scores = rng.integers(0, 256, mask.shape, dtype=np.uint8) if lane else None
                writer.put(cent, frame, mask, scores, bbox=(y0, y1, x0, x1), is_padding=True,
                           padding_ordinal=ordinal, prediction_count=5)
        receipts.append(writer.finish())
    return views.seal_view_family('detector', VIEW, shape, cutoffs, receipts=receipts, root=root)


def rank_input(cutoffs, shape, directory, read_mask_crops):
    budget = backprojection._BackingBudget(directory, 0, None, 1 << 30)
    with backprojection._rank_input(cutoffs, shape, directory, None, read_mask_crops, budget) as (
            ranks, boxes, present, breaks):
        return (None if ranks is None else np.array(ranks)), boxes.copy(), set(present), list(breaks)


CASES = [
    # seed, shape, cutoffs, tasks, padding, scalar_owned, background, verify, chunk_bytes
    (1, (9, 13, 17), (50, 60, 80), 2, 1, False, False, False, None),
    (2, (12, 17, 23), (50, 70, 90), 3, 2, True, True, False, 64),
    (3, (7, 31, 9), (40, 50, 60, 95), 1, 0, False, True, False, 37),
    (4, (10, 16, 16), (50, 80), 4, 2, False, False, True, 48),
    (5, (6, 40, 33), (50, 55, 65, 85), 2, 1, True, False, False, None),
    (6, (11, 23, 29), (10, 50, 99), 3, 3, False, True, True, 100),
]


@pytest.mark.parametrize('case', CASES, ids=lambda case: f'seed{case[0]}')
def test_every_reader_api_equals_the_stream_reader(family, monkeypatch, route, case, tmp_path):
    seed, shape, cutoffs, tasks, padding, scalar_owned, background, verify, chunk_bytes = case
    if chunk_bytes is not None:
        monkeypatch.setattr(views, '_CHUNK_BYTES', chunk_bytes)
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_VERIFY', '1' if verify else '0')
    decision = build(family, np.random.default_rng(seed), shape, cutoffs, tasks=tasks, padding=padding,
                     scalar_owned=scalar_owned, background=background)
    assert decision['status'] == 'ready'
    readable = [c for c in cutoffs if not (scalar_owned and c == cutoffs[0])]
    with ExitStack() as stack:
        readers = {c: stack.enter_context(views.open_view('detector', VIEW, c, native_shape=shape, root=family))
                   for c in readable}
        python_rows, fused_frames, level_merges = [], [], []
        frame_merge, level_merge = views._MaskFrameCrops.merge_rank_frame, views._MaskFrameCrops.merge_rank_level
        monkeypatch.setattr(views._MaskFrameCrops, 'merge_rank_frame', lambda self, *args: (
            lambda result: fused_frames.append(result is not None) or result)(frame_merge(self, *args)))
        monkeypatch.setattr(views._MaskFrameCrops, 'merge_rank_level', lambda self, *args: (
            level_merges.append(1) or level_merge(self, *args)))
        for reader in readers.values():
            assert (reader._decoder is not None) == (route == 'native' and not verify)
            original = reader._python_row_crops
            monkeypatch.setattr(reader, '_python_row_crops',
                                lambda *args, original=original: python_rows.append(1) or original(*args))
        for cent, reader in readers.items():
            for z in range(shape[0]):
                for include_scores in (True, False):
                    metrics = dict(payload_read_bytes=0)
                    expected, error = collect(reference_crops(reader, z, include_scores, metrics))
                    assert error is None
                    before = reader.io_metrics['payload_read_bytes']
                    if include_scores:
                        actual = list(reader.iter_crops(z))
                    else:
                        actual = [(*crop, None) for crop in reader.iter_mask_crops(z)]
                    assert reader.io_metrics['payload_read_bytes']-before == metrics['payload_read_bytes']
                    assert_same_crops(actual, expected)
                    assert_same_crops(reader.read_frame_crops(z, include_scores), expected)
                mask_plane = np.zeros(shape[1:], np.uint8)
                score_plane = np.zeros(shape[1:], np.uint8)
                for y0, y1, x0, x1, mask, score in collect(reference_crops(reader, z, True, metrics))[0]:
                    mask_plane[y0:y1, x0:x1] |= mask
                    if score is not None:
                        np.maximum(score_plane[y0:y1, x0:x1], score, out=score_plane[y0:y1, x0:x1])
                assert reader.read_mask_plane(z).tobytes() == mask_plane.tobytes()
                scores = reader.read_score_plane(z)
                assert (scores is None) == (not reader.has_score_lane)
                if scores is not None:
                    assert scores.tobytes() == score_plane.tobytes()
        # Projection rank input: fused native merge versus the original per-crop loop.
        for first in range(len(readable)):
            for stop in range(first+1, len(readable)+1):
                cuts = tuple(readable[first:stop])
                work = tmp_path / f'rank-{first}-{stop}'
                work.mkdir()
                native = rank_input(cuts, shape, work, lambda c, z: readers[c].iter_mask_crops(z))
                oracle = rank_input(cuts, shape, work, lambda c, z: (
                    crop[:5] for crop in reference_crops(readers[c], z, False, dict(payload_read_bytes=0))))
                assert (native[0] is None) == (oracle[0] is None)
                if native[0] is not None:
                    assert native[0].tobytes() == oracle[0].tobytes()
                np.testing.assert_array_equal(native[1], oracle[1])
                assert native[2:] == oracle[2:]
        # Valid rows, including every rank merge, never need the Python reader natively.
        assert (len(python_rows) == 0) == (route == 'native' and not verify)
        # Frames merge every level in one call; a frame with rows in several
        # payloads (padding from another task range) merges level by level.
        assert any(fused_frames) == (route == 'native' and not verify)
        assert bool(level_merges) == (route == 'native' and not verify and not all(fused_frames))


def test_native_route_does_no_python_inflation_or_stream_io(family, monkeypatch, route, tmp_path):
    if route != 'native':
        pytest.skip('native route only')
    shape = (5, 21, 19)
    build(family, np.random.default_rng(11), shape, (50, 70), tasks=2, padding=1,
          scalar_owned=False, background=False)
    calls = []
    original = views.zlib.decompressobj

    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(views.zlib, 'decompressobj', counted)
    with views.open_view('detector', VIEW, 70, native_shape=shape, root=family) as reader:
        crops = [crop for z in range(shape[0]) for crop in reader.iter_crops(z)]
        assert crops and not calls
        plane = np.zeros(shape[1:], np.uint8)
        for z in range(shape[0]):
            frame = reader.iter_mask_crops(z)
            if frame.wants_rank_plane:
                frame.merge_rank_level(plane, 1, False)
                assert list(frame) == []
        assert not calls and reader.io_metrics['payload_open_calls'] <= len(reader._ranges)*2


@pytest.mark.parametrize('seed', range(24))
def test_corrupted_payloads_fail_exactly_like_the_stream_reader(family, monkeypatch, route, seed, tmp_path):
    rng = np.random.default_rng(100+seed)
    shape = (4, 18, 15)
    monkeypatch.setattr(views, '_CHUNK_BYTES', int(rng.choice([40, 64, 1024])))
    decision = build(family, rng, shape, (50, 75), tasks=1, padding=1, scalar_owned=False, background=False)
    payload = next(family.glob('ranges/*/payload.bin'))
    raw = bytearray(payload.read_bytes())
    if not raw:
        pytest.skip('random family wrote no payload bytes')
    for _ in range(int(rng.integers(1, 4))):
        position = int(rng.integers(0, len(raw)))
        if rng.random() < 0.5:
            raw[position] ^= 1 << int(rng.integers(0, 8))
        else:
            raw[position] = int(rng.integers(0, 256))
    payload.write_bytes(bytes(raw))
    for cent in (50, 75):
        with views.open_view('detector', VIEW, cent, native_shape=shape, root=family) as reader:
            for z in range(shape[0]):
                for include_scores in (True, False):
                    expected = collect(reference_crops(reader, z, include_scores, dict(payload_read_bytes=0)))
                    if include_scores:
                        actual = collect(reader.iter_crops(z))
                    else:
                        actual = collect((*crop, None) for crop in reader.iter_mask_crops(z))
                    assert actual[1] == expected[1]
                    assert_same_crops(actual[0], expected[0])
                    try:
                        batch, batch_error = reader.read_frame_crops(z, include_scores), None
                    except views.SweepViewIntegrityError as error:
                        batch, batch_error = None, (type(error), str(error))
                    assert batch_error == expected[1]
                    if batch is not None:
                        assert_same_crops(batch, expected[0])
                frame = reader.iter_mask_crops(z)
                if frame.wants_rank_plane:
                    plane = np.zeros(shape[1:], np.uint8)
                    frame.merge_rank_level(plane, 1, False)
                    expected = collect(reference_crops(reader, z, False, dict(payload_read_bytes=0)))
                    leftover = collect(frame)
                    # Declined rows raise the stream reader's first error; the
                    # rank input then discards its plane, as it always did.
                    assert leftover[1] == expected[1]
                    if expected[1] is None:
                        reference_plane = np.zeros(shape[1:], np.uint8)
                        for y0, y1, x0, x1, mask, _ in expected[0]:
                            reference_plane[y0:y1, x0:x1] |= mask
                        for y0, y1, x0, x1, mask in leftover[0]:
                            plane[y0:y1, x0:x1] |= mask
                        assert plane.tobytes() == reference_plane.tobytes()


REWRITES = {
    'short_mask_stream': 'Native crop compression is malformed',
    'short_score_stream': 'Native crop compression is malformed',
    'unindexed_tail': 'Native frame has unindexed payload bytes',
    'extra_chunk_at_end_of_file': 'Truncated native crop header',
    'extra_chunk_reads_next_row': 'Native crop encoded lengths are invalid',
}


@pytest.mark.parametrize('rewrite', sorted(REWRITES))
def test_rewritten_rows_fail_exactly_like_the_stream_reader(family, monkeypatch, route, rewrite):
    """Deterministic acknowledged-but-inconsistent rows, sealed through the real writer."""
    monkeypatch.setattr(views, '_CHUNK_BYTES', 120)
    shape = (2, 30, 12)
    writer = views.NativeTaskFanoutWriter(family, 'detector', VIEW, shape, (50, 80), frame_start=0, expected_frames=2)
    rng = np.random.default_rng(5)
    # Cutoff 80 first, so the row for (50, frame 1) is the last one in the payload.
    for cent, z in ((80, 0), (80, 1), (50, 0), (50, 1)):
        mask = (rng.random((30, 12)) < 0.5).astype(np.uint8)
        writer.put(cent, z, mask, rng.integers(1, 256, (30, 12), dtype=np.uint8), bbox=(0, 30, 0, 12))
    target = 0 if rewrite == 'extra_chunk_reads_next_row' else 1
    row = writer.index[0, target]
    assert row['chunks'] == 3
    if rewrite == 'unindexed_tail':
        assert target == 1  # appended bytes stay inside this row's acknowledged range
        writer._stream.write(bytes(5))
        writer._used += 5
        row['length'] += 5
    elif rewrite.startswith('extra_chunk'):
        row['chunks'] += 1
    receipt = writer.finish()
    views.seal_view_family('detector', VIEW, shape, (50, 80), receipts=[receipt], root=family)
    payload = Path(receipt['path']) / 'payload.bin'
    raw = bytearray(payload.read_bytes())
    if rewrite.startswith('short'):
        # Replace the last strip's stream by a complete shorter one, padded to its indexed length.
        position = int(row['offset'])
        for _ in range(2):
            fields = views._CHUNK.unpack_from(raw, position)
            position += views._CHUNK.size+fields[4]+fields[5]
        fields = views._CHUNK.unpack_from(raw, position)
        start, ml, sl = position+views._CHUNK.size, fields[4], fields[5]
        if rewrite == 'short_score_stream':
            start, ml = start+ml, sl
        stream = zlib.compress(zlib.decompress(bytes(raw[start:start+ml]))[:-1], 9)
        raw[start:start+ml] = stream+bytes(ml-len(stream))
        payload.write_bytes(bytes(raw))
    with views.open_view('detector', VIEW, 50, native_shape=shape, root=family) as reader:
        for include_scores in (True, False):
            expected = collect(reference_crops(reader, target, include_scores, dict(payload_read_bytes=0)))
            actual = collect(reader.iter_crops(target) if include_scores else
                             ((*crop, None) for crop in reader.iter_mask_crops(target)))
            assert actual[1] == expected[1]
            assert_same_crops(actual[0], expected[0])
            if rewrite == 'short_score_stream' and not include_scores:
                # Mask-only reads never inspect score streams, natively or not.
                assert expected[1] is None and len(expected[0]) == 3
            else:
                assert expected[1] == (views.SweepViewIntegrityError, REWRITES[rewrite])
                assert len(expected[0]) == (2 if rewrite.startswith('short') else 3)
        frame = reader.iter_mask_crops(target)
        assert frame.wants_rank_plane == (route == 'native')
        if route == 'native':
            plane = np.zeros(shape[1:], np.uint8)
            populated, *_ = frame.merge_rank_level(plane, 1, False)
            leftover = collect(frame)
            if rewrite == 'short_score_stream':
                assert populated and leftover == ([], None)
            else:
                assert not plane.any() and leftover[1] == (views.SweepViewIntegrityError, REWRITES[rewrite])


def test_closed_reader_and_suspended_iterators(family, route):
    shape = (3, 12, 12)
    build(family, np.random.default_rng(7), shape, (50, 80), tasks=1, padding=0,
          scalar_owned=False, background=False)
    reader = views.open_view('detector', VIEW, 80, native_shape=shape, root=family)
    pending = reader.iter_crops(0)
    frames = [list(reader.iter_crops(z)) for z in range(shape[0])]
    reader.close()
    with pytest.raises(RuntimeError, match='closed'):
        list(reader.iter_crops(1))
    with pytest.raises(RuntimeError, match='closed'):
        list(reader.iter_mask_crops(1))
    with pytest.raises(RuntimeError, match='closed'):
        reader.read_frame_crops(1)
    with pytest.raises(RuntimeError, match='closed'):
        list(pending)
    # Crops returned before close stay readable after the mapping is released.
    for crops in frames:
        for crop in crops:
            assert crop[4].sum() >= 0


def test_mapping_cache_reuses_two_payloads_and_counts_opens(family, monkeypatch, route):
    shape = (8, 10, 11)
    receipts = []
    for start in range(0, 8, 2):
        writer = views.NativeTaskFanoutWriter(family, 'detector', VIEW, shape, (50, 80), frame_start=start,
                                              expected_frames=2)
        for cent in (50, 80):
            for z in range(start, start+2):
                writer.put(cent, z, np.ones((2, 3), np.uint8), np.full((2, 3), 9, np.uint8), bbox=(1, 3, 2, 5))
        receipts.append(writer.finish())
    views.seal_view_family('detector', VIEW, shape, (50, 80), receipts=receipts, root=family)
    with views.open_view('detector', VIEW, 80, native_shape=shape, root=family) as reader:
        for z in range(shape[0]):
            assert len(list(reader.iter_crops(z))) == 1
        assert reader.io_metrics['payload_open_calls'] == 4
        for z in (6, 7, 4, 5, 6):
            assert len(reader.read_frame_crops(z)) == 1
        assert reader.io_metrics['payload_open_calls'] == 4
        assert len(reader._payloads) == 2


def test_mask_only_reads_skip_score_bytes_on_both_routes(family, monkeypatch, route):
    """Route-aware form of the stream reader's mask-only check in tests/test_sweep_views.py.

    Mask-only reads account for the header and mask bytes only and never inflate
    the score stream. The Python route inflates through zlib.decompressobj (one
    call per stream read); the native route never calls it once its decoder is
    qualified (the ``route`` fixture qualifies it before the counter is installed,
    because the qualification probe itself compares against Python zlib).
    """
    shape = (1, 4, 5)
    writer = views.NativeTaskFanoutWriter(family, 'detector', VIEW, shape, (50, 80), frame_start=0, expected_frames=1)
    for cent in (50, 80):
        writer.put(cent, 0, np.ones((2, 3), np.uint8), np.full((2, 3), 200, np.uint8), bbox=(1, 3, 1, 4))
    receipt = writer.finish()
    views.seal_view_family('detector', VIEW, shape, (50, 80), receipts=[receipt], root=family)
    fields = views._CHUNK.unpack_from((Path(receipt['path']) / 'payload.bin').read_bytes())
    calls = []
    original = views.zlib.decompressobj

    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(views.zlib, 'decompressobj', counted)
    per_stream = 0 if route == 'native' else 1
    with views.open_view('detector', VIEW, 50, native_shape=shape, root=family) as reader:
        assert (reader._decoder is not None) == (route == 'native')
        crops = list(reader.iter_mask_crops(0))
        assert len(crops) == 1 and crops[0][:4] == (1, 3, 1, 4) and crops[0][4].tolist() == [[1]*3]*2
        assert len(calls) == per_stream
        assert reader.io_metrics['payload_read_bytes'] == views._CHUNK.size+fields[4]
        assert [crop[:4] for crop in reader.read_frame_crops(0, False)] == [(1, 3, 1, 4)]
        assert len(calls) == 2*per_stream
    with views.open_view('detector', VIEW, 50, native_shape=shape, root=family) as reader:
        crops = list(reader.iter_crops(0))
        assert len(crops) == 1 and crops[0][5].tolist() == [[200]*3]*2
        assert len(calls) == 4*per_stream
        assert reader.io_metrics['payload_read_bytes'] == views._CHUNK.size+fields[4]+fields[5]


def test_decoder_disabled_by_configuration_reports_python_route(monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_DECODE', 'python')
    decoder, receipt = decode.native_view_decoder()
    assert decoder is None and receipt['fallback_reason'] == 'disabled_by_configuration'


def test_rank_merge_reads_unaligned_64_bit_windows_without_consuming_tail_padding():
    decoder, receipt = decode.native_view_decoder()
    if decoder is None:
        pytest.skip(f'native journal decode unavailable: {receipt}')
    height, width = 3, 69
    selected = np.zeros((height, width), dtype=bool)
    selected[:2, :64] = True
    selected[2, 1:64:3] = True
    selected[:, -1] = True
    packed = np.packbits(selected.ravel(), bitorder='little')
    # The 207 pixels occupy seven bits of the last byte.
    packed[-1] |= 0b10000000
    body = zlib.compress(packed)
    header = views._CHUNK.pack(0, height, 1, width + 1, len(body), 0, 0, 0, len(packed), 0)
    payload = np.frombuffer(header + body, np.uint8)
    output = np.full((height, width + 2), 3, np.uint8)
    output[1, 5] = 1
    expected = output.copy()
    expected[:, 1:-1][selected] = 4
    metadata = np.array(((0, len(payload), 1, 0, height, 1, width + 1,
                          height, width + 2, 0, height * width, 4, 1),), np.int64)
    status, report, top, bottom, left, right = decoder.merge_rank_rows(payload, metadata, output)
    assert (status, top, bottom, left, right) == (0, 0, height, 1, width + 1)
    np.testing.assert_array_equal(report, [[1, 1, len(payload)]])
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize('case', CASES, ids=lambda case: f'seed{case[0]}')
def test_sibling_family_matches_cutoff_readers_and_skips_consumed_iterators(
        family, monkeypatch, route, case, tmp_path):
    seed, shape, cuts, tasks, padding, scalar_owned, background, verify, chunk_bytes = case
    if chunk_bytes is not None:
        monkeypatch.setattr(views, '_CHUNK_BYTES', chunk_bytes)
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_VERIFY', str(int(verify)))
    build(family, np.random.default_rng(seed), shape, cuts, tasks=tasks, padding=padding,
          scalar_owned=scalar_owned, background=background)
    with ExitStack() as stack:
        ordinary = {c: stack.enter_context(views.open_view('detector', VIEW, c, native_shape=shape, root=family))
                    for c in cuts[1:]}
        opened, verified = [], []
        decision, check_range = views._decision, views._range
        with monkeypatch.context() as audit:
            def acquire(*args, **kwargs):
                opened.append(1)
                return decision(*args, **kwargs)
            def validate(root, receipt, product, cutoffs):
                verified.append(receipt['path'])
                return check_range(root, receipt, product, cutoffs)
            audit.setattr(views, '_decision', acquire)
            audit.setattr(views, '_range', validate)
            joint = stack.enter_context(views.open_sibling_views('detector', VIEW, cuts,
                                                                 native_shape=shape, root=family))
        assert len(opened) == 1
        assert verified == [receipt['path'] for receipt in joint.metadata['ranges']]
        ordinary_wrappers, family_wrappers = [], []
        new_iterator = views._MaskFrameCrops.__init__
        def count_iterator(iterator, reader, z):
            counts = family_wrappers if reader is joint.readers[reader.cutoff] else ordinary_wrappers
            counts.append((reader.cutoff, z))
            new_iterator(iterator, reader, z)
        monkeypatch.setattr(views._MaskFrameCrops, '__init__', count_iterator)
        subsets = (cuts[1:], cuts[-1:], cuts[1::2])
        for selected in subsets:
            separate = rank_input(selected, shape, tmp_path, lambda c, z: ordinary[c].iter_mask_crops(z))
            grouped = rank_input(selected, shape, tmp_path, joint)
            if separate[0] is None:
                assert grouped[0] is None
            else:
                np.testing.assert_array_equal(grouped[0], separate[0])
            np.testing.assert_array_equal(grouped[1], separate[1])
            assert grouped[2:] == separate[2:]
        assert len(ordinary_wrappers) == shape[0]*sum(map(len, subsets))
        if route == 'native' and not verify:
            padded = set().union(*(reader._padding for reader in joint.readers.values()))
            assert all(z in padded for _, z in family_wrappers)
            assert len(family_wrappers) < len(ordinary_wrappers)
        assert {c: r.io_metrics for c, r in joint.readers.items()} == {c: r.io_metrics for c, r in ordinary.items()}
        with pytest.raises(ValueError):
            joint.rank_merger((cuts[-1], cuts[-1]), shape)
        with pytest.raises(ValueError):
            joint.rank_merger((cuts[0],), shape)
        merge = joint.rank_merger(cuts[1:], shape)
        with pytest.raises(ValueError, match='gap check'):
            merge(0, None, np.empty(0, bool))
        joint.readers[cuts[-1]].close()
        with pytest.raises(RuntimeError, match='closed'):
            merge(0, None, np.zeros(len(cuts)-1, bool))


def test_sibling_family_declines_corrupt_late_lane_without_writing_or_charging(family, route, tmp_path):
    shape = (2, 3, 5)
    cuts = (50, 70, 90)
    writer = views.NativeTaskFanoutWriter(family, 'detector', VIEW, shape, cuts, frame_start=0,
                                          expected_frames=2, backend='test')
    for c in cuts:
        writer.put_empty_range(c, 0, 1)
        writer.put(c, 1, np.ones((2, 3), np.uint8), bbox=(0, 2, 1, 4))
    offset = int(writer.index[-1, 1]['offset'])
    receipt = writer.finish()
    views.seal_view_family('detector', VIEW, shape, cuts, receipts=[receipt], root=family)
    payload = Path(receipt['path']) / 'payload.bin'
    with payload.open('r+b') as stream:
        stream.seek(offset + views._CHUNK.size)
        stream.write(b'\x00\x00')
    with views.open_sibling_views('detector', VIEW, cuts, native_shape=shape, root=family) as joint:
        merge = joint.rank_merger(cuts[1:], shape)
        checks = np.array([False, True])
        empty = merge(0, None, checks)
        assert empty is not None and empty[0] is None
        plane = np.full(shape[1:], 7, np.uint8)
        assert merge(1, plane, checks) is None
        assert (plane == 7).all()
        assert all(r.io_metrics['payload_read_bytes'] == 0 for r in joint.readers.values())
        with pytest.raises(views.SweepViewIntegrityError) as failure:
            rank_input(cuts[1:], shape, tmp_path, joint)
    with ExitStack() as stack:
        separate = {c: stack.enter_context(views.open_view('detector', VIEW, c, root=family)) for c in cuts[1:]}
        with pytest.raises(views.SweepViewIntegrityError) as expected:
            rank_input(cuts[1:], shape, tmp_path, lambda c, z: separate[c].iter_mask_crops(z))
    assert str(failure.value) == str(expected.value)
