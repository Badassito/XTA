"""Fused NRRD layer producer: byte identity with the per-slice reference path.

The reference paths are the pre-fusion loops kept verbatim as fallbacks
(``_stream_native_crop_spans_per_band``, the original bodies of
``iter_native_sparse_members`` / ``iter_restored_sparse_members`` and
``_tee_sparse``); forcing ``fused_store_view`` to return ``None`` selects them.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import nrrd_compression as nc
from XTA import nrrd_sparse_kernels as nk
from XTA import outputs
from XTA.interpolation import (
    CVOL_FORMAT,
    INTERNAL_PACKED_CVOL_FORMAT,
    NrrdLayerRef,
    RawBBoxMaskStore,
    write_raw_bbox_mask_store,
)
from XTA.nrrd_spans import _stream_native_crop_spans_per_band, stream_native_crop_spans

FORMATS = (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT)


def _volume(seed, shape, *, empty_every=4, full_slice=True):
    rng = np.random.default_rng(seed)
    t, h, w = shape
    vol = np.zeros(shape, np.uint8)
    for z in range(t):
        if z % empty_every == 1:
            continue
        y0 = int(rng.integers(0, h - 1)); y1 = int(rng.integers(y0 + 1, h + 1))
        x0 = int(rng.integers(0, w - 1)); x1 = int(rng.integers(x0 + 1, w + 1))
        vol[z, y0:y1, x0:x1] = (rng.random((y1 - y0, x1 - x0)) < 0.6).astype(np.uint8)
        vol[z, y0, x0] = 1
        vol[z, y1 - 1, x1 - 1] = 1
    if full_slice and t > 2:
        vol[2] = (rng.random((h, w)) < 0.5).astype(np.uint8)
        vol[2, 0, 0] = vol[2, -1, -1] = 1
    return vol


def _store(tmp_path, vol, fmt, name='layer'):
    path = Path(tmp_path) / f'{name}-{fmt}.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        write_raw_bbox_mask_store(vol, path, format_name=fmt, desc='gil-test', workers=1)
    return path


class _Sink(io.RawIOBase):
    def __init__(self):
        self.parts = []

    def writable(self):
        return True

    def write(self, b):
        self.parts.append(bytes(b))
        return len(b)


def _codec():
    return nc._nrrd_member_codec_spec('zlib')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope='module', autouse=True)
def _warm():
    info = nk.warm_nrrd_producer_kernels()
    assert info['crop_row_signatures'] >= 2 and info['tee_signatures'] >= 2


# ----------------------------------------------------------------------------- crop rows

@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('member_bytes', [86, 1000, 16 << 20])
@pytest.mark.parametrize('batch_bytes', [1, 700, 5000, None])
def test_crop_row_stream_bytes_stats_and_observations_match_reference(tmp_path, fmt, member_bytes, batch_bytes):
    vol = _volume(11, (13, 37, 43))
    store = RawBBoxMaskStore.open(_store(tmp_path, vol, fmt), mmap_payload=True)
    try:
        assert nk.fused_store_view(store) is not None
        for first, stop in ((0, 13), (2, 9), (12, 13), (1, 2), (5, 5)):
            results = []
            for fused in (False, True):
                sink = _Sink()
                writer = nc._MemberParallelGzipPayloadWriter(sink, codec_spec=_codec())
                seen = []

                def observe(z, y, x, crop):
                    seen.append((z, y, x, crop.shape, crop.tobytes()))

                if fused:
                    encoded = stream_native_crop_spans(store, writer, first, stop, member_bytes, observe,
                                                       batch_bytes=batch_bytes)
                else:
                    encoded = _stream_native_crop_spans_per_band(store, writer, first, stop, member_bytes, observe)
                writer.close()
                stats = writer._writer_stats
                results.append((b''.join(sink.parts), encoded, seen,
                                stats['canonical_zero_logical_bytes'], stats['canonical_zero_members']))
            ref, new = results
            assert _sha(new[0]) == _sha(ref[0])
            assert new[1:] == ref[1:]
            import gzip
            assert gzip.decompress(new[0]) == vol[first:stop].tobytes()
    finally:
        store.close()


def test_writer_without_batch_method_replays_per_member_calls(tmp_path):
    vol = _volume(12, (9, 21, 30))
    store = RawBBoxMaskStore.open(_store(tmp_path, vol, INTERNAL_PACKED_CVOL_FORMAT), mmap_payload=True)
    try:
        ref_sink, new_sink = _Sink(), _Sink()
        ref = nc._MemberParallelGzipPayloadWriter(ref_sink, codec_spec=_codec())
        _stream_native_crop_spans_per_band(store, ref, 0, 9, 200)
        ref.close()

        class NoBatch:
            def __init__(self, inner):
                self.inner = inner
                self.write_canonical_zeros = inner.write_canonical_zeros
                self.write_owned_known_nonzero = inner.write_owned_known_nonzero

        new = nc._MemberParallelGzipPayloadWriter(new_sink, codec_spec=_codec())
        stream_native_crop_spans(store, NoBatch(new), 0, 9, 200, batch_bytes=300)
        new.close()
        assert b''.join(new_sink.parts) == b''.join(ref_sink.parts)
    finally:
        store.close()


def test_member_batch_hardware_minimum_replays_existing_policy():
    calls = []
    codec = mock.Mock(minimum_input_bytes=65536)
    writer = nc._MemberParallelGzipPayloadWriter(io.BytesIO(), codec_spec=('qat', 1, codec))
    with mock.patch.object(writer, 'write_canonical_zeros', side_effect=lambda n: calls.append(('z', n)) or n), \
            mock.patch.object(writer, 'write_owned_known_nonzero',
                              side_effect=lambda mv: calls.append(('d', bytes(mv))) or len(mv)):
        assert writer.write_owned_member_batch(b'abcdef', [(5, 0, 2), (0, 2, 6), (7, 6, 6)]) == 18
    assert calls == [('z', 5), ('d', b'ab'), ('d', b'cdef'), ('z', 7)]


def test_member_batch_rejects_bad_segments_and_closed_writer():
    writer = nc._MemberParallelGzipPayloadWriter(io.BytesIO(), codec_spec=_codec())
    with pytest.raises(ValueError):
        writer.write_owned_member_batch(b'abc', [(0, 2, 9)])
    with pytest.raises(ValueError):
        writer.write_owned_member_batch(b'abc', [(-1, 0, 1)])
    writer.close()
    with pytest.raises(RuntimeError):
        writer.write_owned_member_batch(b'abc', [(0, 0, 1)])


def test_member_batch_failure_abandons_writer():
    def boom(_payload):
        raise OSError('codec failed')

    sink = io.BytesIO()
    writer = nc._MemberParallelGzipPayloadWriter(sink, codec_spec=('zlib', 1, boom))
    with pytest.raises(OSError, match='codec failed'):
        writer.write_owned_member_batch(b'abcdef', [(3, 0, 6)])
        writer.close()
    assert writer.closed and not writer._pending


def test_canonical_zero_run_bytes_match_writer():
    for n in (1, 7, 1 << 20, (1 << 20) + 1, 5 * (1 << 20) + 12345):
        sink = io.BytesIO()
        writer = nc._MemberParallelGzipPayloadWriter(sink, codec_spec=_codec())
        writer.write_canonical_zeros(n)
        writer.close()
        assert nc.canonical_zero_run_bytes(n) == sink.getvalue()
        assert nc.canonical_zero_run_member_count(n) == writer._writer_stats['canonical_zero_members']


# ----------------------------------------------------------------------------- fallbacks

def test_invalid_index_takes_reference_path_and_raises_same_error(tmp_path):
    vol = _volume(13, (6, 17, 19))
    path = _store(tmp_path, vol, CVOL_FORMAT)
    errors = []
    for fused in (False, True):
        store = RawBBoxMaskStore.open(path, mmap_payload=True)
        try:
            index = store.index.copy()
            index['x1'][3] = 99  # out-of-frame bbox at slice 3
            store.index = index
            assert not nk.fused_store_view(store).valid(np.arange(6))
            writer = nc._MemberParallelGzipPayloadWriter(io.BytesIO(), codec_spec=_codec())
            fn = stream_native_crop_spans if fused else _stream_native_crop_spans_per_band
            with pytest.raises(ValueError) as caught:
                fn(store, writer, 0, 6, 100)
            errors.append(str(caught.value))
            writer.close()
        finally:
            store.close()
    assert errors[0] == errors[1] and 'invalid bbox' in errors[0]


def test_replaced_decode_keeps_reference_path(tmp_path):
    vol = _volume(14, (4, 9, 11))
    store = RawBBoxMaskStore.open(_store(tmp_path, vol, CVOL_FORMAT), mmap_payload=True)
    try:
        assert nk.fused_store_view(store) is not None
        with mock.patch.object(RawBBoxMaskStore, 'decode_slice_crop', side_effect=AssertionError('decode')):
            assert nk.fused_store_view(store) is None
        store.decode_slice_crop = lambda *a, **k: None
        assert nk.fused_store_view(store) is None
    finally:
        store.close()


# ----------------------------------------------------------------------------- members

def _collect(iterator):
    out = []
    for ln, member in iterator:
        out.append((ln, None if member is None else (member.shape, member.tobytes())))
    return out


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('member_bytes', [1, 37 * 43 * 2, 37 * 43 * 5])
def test_native_sparse_members_match_reference(tmp_path, fmt, member_bytes, monkeypatch):
    vol = _volume(21, (13, 37, 43))
    store = RawBBoxMaskStore.open(_store(tmp_path, vol, fmt), mmap_payload=True)
    monkeypatch.setenv('YOLO_TTA_NRRD_PRODUCER_BATCH_MIB', '16')
    try:
        for first, stop in ((0, 13), (3, 11)):
            got = []
            for fused in (False, True):
                seen = []

                def observe(z, y, x, crop):
                    seen.append((z, y, x, crop.shape, crop.tobytes()))

                ctx = contextlib.nullcontext() if fused else mock.patch.object(nk, 'fused_store_view', return_value=None)
                with ctx:
                    members = _collect(store.iter_native_sparse_members(first, stop, member_bytes=member_bytes,
                                                                         sparse_consumer=observe))
                got.append((members, seen))
            assert got[0] == got[1]
    finally:
        store.close()


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('out_shape', [(13, 37, 43), (7, 19, 23), (29, 61, 71), (13, 20, 80), (5, 37, 43), (1, 9, 9)])
def test_restored_sparse_members_match_reference(tmp_path, fmt, out_shape):
    vol = _volume(31, (13, 37, 43))
    store = RawBBoxMaskStore.open(_store(tmp_path, vol, fmt), mmap_payload=True)
    try:
        slice_bytes = out_shape[1] * out_shape[2]
        for member_bytes in (1, slice_bytes * 3):
            for first, stop in ((0, out_shape[0]), (out_shape[0] // 3, out_shape[0])):
                got = []
                for fused in (False, True):
                    seen = []

                    def observe(z, y, x, crop):
                        seen.append((z, y, x, crop.shape, crop.tobytes()))

                    ctx = (contextlib.nullcontext() if fused
                           else mock.patch.object(nk, 'fused_store_view', return_value=None))
                    with ctx:
                        members = _collect(store.iter_restored_sparse_members(
                            out_shape, first, stop, member_bytes=member_bytes, sparse_consumer=observe))
                    got.append((members, seen))
                assert got[0] == got[1]
    finally:
        store.close()


# ----------------------------------------------------------------------------- whole layer + mirrors

def _write_layer(tmp_path, tag, store_path, fmt, store_shape, out_shape, mirrors, z_shards, fused):
    ref = NrrdLayerRef(key='k', name='layer', path=store_path, shape=store_shape, storage_format=fmt)
    root = Path(tmp_path) / tag  # same file names: Slicer segment names derive from them
    root.mkdir()
    out = root / 'layer.nrrd'
    mirror_specs = [(shape, root / f'layer-lq{i}.nrrd') for i, shape in enumerate(mirrors)]
    ctx = contextlib.nullcontext() if fused else mock.patch.object(nk, 'fused_store_view', return_value=None)
    tee = mock.patch.object(nk.FusedSparseMirrorTee, 'tee_records', autospec=True,
                            side_effect=nk.FusedSparseMirrorTee.tee_records)
    with ctx, tee as tee_calls, contextlib.redirect_stdout(io.StringIO()):
        outputs.write_layer_nrrd_with_low_quality_mirrors(ref, out_shape, out, mirror_specs, z_shards=z_shards)
    assert bool(tee_calls.call_count) == bool(fused)
    return [_sha(out.read_bytes())] + [_sha(p.read_bytes()) for _s, p in mirror_specs]


@pytest.mark.parametrize('fmt', FORMATS)
@pytest.mark.parametrize('out_shape', [(16, 41, 47), (9, 23, 29), (31, 60, 90)])
@pytest.mark.parametrize('z_shards', [1, 3])
def test_layer_nrrd_and_mirror_files_are_byte_identical(tmp_path, fmt, out_shape, z_shards, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_PRODUCER_BATCH_MIB', '16')
    vol = _volume(41, (16, 41, 47))
    store_path = _store(tmp_path, vol, fmt)
    mirrors = [(5, 9, 11), (8, 41, 47), (4, 50, 20)]
    ref = _write_layer(tmp_path, 'ref', store_path, fmt, vol.shape, out_shape, mirrors, z_shards, False)
    new = _write_layer(tmp_path, 'new', store_path, fmt, vol.shape, out_shape, mirrors, z_shards, True)
    assert new == ref


def test_fused_tee_failure_marks_only_that_job(tmp_path):
    vol = _volume(51, (6, 20, 24))
    store = RawBBoxMaskStore.open(_store(tmp_path, vol, CVOL_FORMAT), mmap_payload=True)
    try:
        jobs = []
        for shape in ((3, 10, 12), (3, 10, 12)):
            jobs.append({'shape': shape, 'volume': np.zeros(shape, np.uint8),
                         'src_to_m': [(z // 2,) for z in range(6)],
                         'locks': [threading.Lock() for _ in range(16)], 'failed': False})
        jobs[1]['volume'] = None  # OR into None raises inside the kernel call
        tee = nk.FusedSparseMirrorTee(jobs, (20, 24))

        def consumer(*_a):
            raise AssertionError('generic path used')

        consumer.fused_sparse_mirror_tee = tee
        writer = nc._MemberParallelGzipPayloadWriter(io.BytesIO(), codec_spec=_codec())
        stream_native_crop_spans(store, writer, 0, 6, 1 << 20, consumer)
        writer.close()
        assert jobs[1]['failed'] and not jobs[0]['failed']
        assert jobs[0]['volume'].any()
        assert not any(lock.locked() for job in jobs for lock in job['locks'])
    finally:
        store.close()



# ----------------------------------------------------------------------------- tee lock discipline

def _reference_tee(jobs, source_hw, buf2d, records):
    """_tee_sparse semantics per record (outputs.py), used as the identity oracle."""
    for row0, y0, x0, y1, x1, z in records.tolist():
        crop = buf2d[row0:row0 + (y1 - y0), x0:x1]
        for job in jobs:
            targets = job['src_to_m'][z]
            if not targets:
                continue
            resized = outputs._resize_sparse_binary_crop_to_output_region(
                crop, source_shape=source_hw, source_bbox=(y0, x0, y1, x1),
                output_shape=tuple(job['shape'][1:]))
            if resized is None:
                continue
            my0, mx0, my1, mx1, mirror_crop = resized
            dst = job['volume'][:, my0:my1, mx0:mx1]
            for mz in targets:
                np.bitwise_or(dst[mz], mirror_crop, out=dst[mz])


def _tee_case(seed, n_stripes=16):
    rng = np.random.default_rng(seed)
    h, w, t = 37, 45, 12
    buf = np.zeros((t * h, w), np.uint8)
    records = []
    for z in range(t):
        if z % 5 == 3:
            continue
        y0 = int(rng.integers(0, h - 1)); y1 = int(rng.integers(y0 + 1, h + 1))
        x0 = int(rng.integers(0, w - 1)); x1 = int(rng.integers(x0 + 1, w + 1))
        row0 = z * h + y0
        if z % 5 != 4:  # z % 5 == 4 keeps an all-zero crop (skipped by both tees)
            buf[row0:row0 + y1 - y0, x0:x1] = rng.random((y1 - y0, x1 - x0)) < 0.4
        records.append((row0, y0, x0, y1, x1, z))
    records = np.asarray(records, np.int64)

    def jobs():
        out = []
        for shape in ((6, 9, 11), (12, 37, 45), (4, 60, 20), (20, 5, 70)):
            m_t = shape[0]
            src_to_m = [() if z % 7 == 6 else tuple(sorted({(z * m_t) // t, ((z + 1) * m_t - 1) // t}))
                        for z in range(t)]
            out.append({'shape': shape, 'volume': np.zeros(shape, np.uint8), 'src_to_m': src_to_m,
                        'locks': [threading.Lock() for _ in range(n_stripes)], 'failed': False})
        return out
    return (h, w), buf, records, jobs


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('n_stripes', [16, 3, 1])
def test_fused_tee_matches_reference_tee(seed, n_stripes):
    hw, buf, records, jobs = _tee_case(seed, n_stripes)
    ref_jobs, new_jobs = jobs(), jobs()
    _reference_tee(ref_jobs, hw, buf, records)
    nk.FusedSparseMirrorTee(new_jobs, hw).tee_records(buf, records)
    for ref, new in zip(ref_jobs, new_jobs):
        assert not new['failed']
        assert ref['volume'].any()
        np.testing.assert_array_equal(new['volume'], ref['volume'])


class _CountingLock:
    held = 0
    guard = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        with _CountingLock.guard:
            _CountingLock.held += 1
        return self

    def __exit__(self, *exc):
        with _CountingLock.guard:
            _CountingLock.held -= 1
        self._lock.release()

    acquire = __enter__

    def release(self):
        self.__exit__()

    def locked(self):
        return self._lock.locked()


def test_fused_tee_resizes_unlocked_and_locks_only_the_or(monkeypatch):
    hw, buf, records, jobs = _tee_case(4)
    job_list = jobs()
    for job in job_list:
        job['locks'] = [_CountingLock() for _ in range(16)]
    seen = {'resize_held': [], 'or_ok': []}
    real_resize, real_or = nk.resize_records_for_tee, nk.or_tee_records

    def resize(*a):
        seen['resize_held'].append(_CountingLock.held)
        return real_resize(*a)

    def or_records(regions, results, recs, ptr, index, volume):
        job = next(j for j in job_list if j['volume'] is volume)
        live = regions[:, 2] > regions[:, 0]
        want = sorted({int(mz) % 16 for z in recs[live, 5].tolist() for mz in job['src_to_m'][z]})
        locked = [i for i, lk in enumerate(job['locks']) if lk.locked()]
        seen['or_ok'].append(locked == want and _CountingLock.held == len(want))
        return real_or(regions, results, recs, ptr, index, volume)

    monkeypatch.setattr(nk, 'resize_records_for_tee', resize)
    monkeypatch.setattr(nk, 'or_tee_records', or_records)
    nk.FusedSparseMirrorTee(job_list, hw).tee_records(buf, records)
    assert seen['resize_held'] == [0] * len(job_list)
    assert len(seen['or_ok']) == len(job_list) and all(seen['or_ok'])
    assert _CountingLock.held == 0
    ref_jobs = jobs()
    _reference_tee(ref_jobs, hw, buf, records)
    for ref, new in zip(ref_jobs, job_list):
        np.testing.assert_array_equal(new['volume'], ref['volume'])


def test_concurrent_shards_resize_in_parallel(monkeypatch):
    """Two z-shards of one layer teeing into the same mirror must not serialize on the
    resize: each waits inside its resize for the other one to arrive there too. If the
    resize ran under the stripe locks (the pre-fix tee), the second shard would block on
    the locks and the barrier would time out."""
    hw, buf, records, jobs = _tee_case(5)
    job_list = jobs()[:1]
    tee = nk.FusedSparseMirrorTee(job_list, hw)
    barrier = threading.Barrier(2, timeout=10)
    real_resize = nk.resize_records_for_tee

    def resize(*a):
        barrier.wait()
        return real_resize(*a)

    monkeypatch.setattr(nk, 'resize_records_for_tee', resize)
    errors = []

    def shard(recs):
        try:
            tee.tee_records(buf, recs)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    # interleaved shards share mirror slices, hence stripes
    threads = [threading.Thread(target=shard, args=(np.ascontiguousarray(r),))
               for r in (records[0::2], records[1::2])]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert not errors and not job_list[0]['failed']  # a BrokenBarrierError would mark failed
    ref_jobs = jobs()[:1]
    _reference_tee(ref_jobs, hw, buf, records)
    np.testing.assert_array_equal(job_list[0]['volume'], ref_jobs[0]['volume'])


def test_warm_compiles_every_kernel_signature():
    info = nk.warm_nrrd_producer_kernels()
    assert info['seconds'] < 5.0
    for key in ('crop_row_signatures', 'native_signatures', 'restored_signatures',
                'tee_signatures', 'tee_or_signatures'):
        assert info[key] >= 2
