"""GIL fix #5: the nogil encoded-block writer is byte-identical to the scalar original.

``LegacyWriter.consume_encoded_block`` below is the pre-fix method copied
verbatim from ``XTA/interpolation.py`` (HEAD). Every comparison runs the same
inputs through both writers and requires identical exceptions (type and
message), index/payload/meta bytes, slice states, offsets and counters.
"""
import contextlib
import io
import json
import operator
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import interpolation as ip
from XTA.packed_dense_source import encode_dense_u8_packed_block


class LegacyWriter(ip.IncrementalRawBBoxMaskStoreWriter):
    def consume_encoded_block(self, z0, records, payload, *, packed):
        first = operator.index(z0)
        count = len(records)
        if first < 0 or first + count > self.shape[0]:
            raise IndexError(f'{self.desc}: encoded block is outside source slices')
        if bool(packed) != self._packbits_payload:
            raise ValueError(f'{self.desc}: encoded payload format does not match the store')
        data = np.asarray(payload)
        if data.dtype != np.uint8 or data.ndim != 1 or not data.flags.c_contiguous:
            raise ValueError(f'{self.desc}: encoded payload must be contiguous uint8 bytes')
        entries = []
        cursor = 0
        for index, record in enumerate(records):
            z, y0, y1, x0, x1, foreground, offset, size = (
                operator.index(getattr(record, field))
                for field in ('z', 'y0', 'y1', 'x0', 'x1', 'foreground', 'offset', 'size')
            )
            if z != first + index or not (0 <= y0 <= y1 <= self.shape[1] and 0 <= x0 <= x1 <= self.shape[2]):
                raise ValueError(f'{self.desc}: encoded slice coordinates are invalid')
            area = (y1 - y0) * (x1 - x0)
            row_bytes = ((x1 - x0 + 7) // 8) if packed else (x1 - x0)
            expected_size = (y1 - y0) * row_bytes
            if (offset != cursor or size != expected_size or foreground < 0 or foreground > area
                    or (foreground == 0 and (area != 0 or (y0, y1, x0, x1) != (0, 0, 0, 0)))
                    or (foreground > 0 and area == 0)):
                raise ValueError(f'{self.desc}: encoded crop size/count/offset is invalid')
            cursor += size
            if cursor > data.size:
                raise ValueError(f'{self.desc}: encoded payload ends inside a slice')
            if packed and size and (x1 - x0) % 8:
                tail_mask = (0xff << ((x1 - x0) % 8)) & 0xff
                tail_bytes = data[offset:cursor].reshape(y1 - y0, row_bytes)[:, -1]
                if np.any(tail_bytes & tail_mask):
                    raise ValueError(f'{self.desc}: encoded row padding bits must be zero')
            entries.append((z, y0, y1, x0, x1, foreground, offset, size, area))
        if cursor != data.size:
            raise ValueError(f'{self.desc}: encoded payload has unclaimed bytes')
        with self._lock:
            if self._failed_reason is not None:
                return
            if self._finalized or self._fd is None:
                raise RuntimeError(f'{self.desc}: encoded block arrived after store closure')
            if np.any(self._slice_state[first:first + count] != 0):
                raise ValueError(f'{self.desc}: encoded slices were delivered more than once')
            fd, base_offset = self._fd, self._next_offset
            self._next_offset += int(data.size)
            self._slice_state[first:first + count] = np.uint8(1)
            self._active_callbacks += 1
        try:
            if data.size:
                self._pwrite_all(fd, memoryview(data).cast('B'), int(base_offset))
            with self._lock:
                if self._failed_reason is not None:
                    return
                for z, y0, y1, x0, x1, foreground, offset, size, area in entries:
                    rec = self.index[z]
                    if foreground:
                        rec['kind'] = np.uint8(1)
                        rec['offset'] = np.uint64(base_offset + offset)
                        rec['payload_size'] = np.uint64(size)
                        rec['y0'], rec['y1'] = np.uint32(y0), np.uint32(y1)
                        rec['x0'], rec['x1'] = np.uint32(x0), np.uint32(x1)
                        rec['payload_nbytes'] = np.uint64(area)
                        self._nonempty_slices += 1
                        self._foreground_voxels += foreground
                        self._min_t, self._max_t = min(self._min_t, z), max(self._max_t, z)
                        self._min_y, self._max_y = min(self._min_y, y0), max(self._max_y, y1 - 1)
                        self._min_x, self._max_x = min(self._min_x, x0), max(self._max_x, x1 - 1)
                    self._slice_state[z] = np.uint8(2)
        except BaseException as exc:
            self.abort(exc)
            raise
        finally:
            with self._lock:
                self._active_callbacks -= 1


@pytest.fixture(autouse=True)
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def encode(source, first=0, packed=False):
    records, chunks, offset = [], [], 0
    for local, plane in enumerate(source):
        ys, xs = np.nonzero(plane)
        if len(ys):
            y0, y1, x0, x1 = int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)
            crop = np.ascontiguousarray(plane[y0:y1, x0:x1] != 0, dtype=np.uint8)
            data = (np.packbits(crop, axis=1, bitorder='little') if packed else crop).reshape(-1)
        else:
            y0 = y1 = x0 = x1 = 0
            data = np.empty(0, np.uint8)
        records.append(SimpleNamespace(z=first + local, y0=y0, y1=y1, x0=x0, x1=x1,
                                       foreground=len(ys), offset=offset, size=data.size))
        chunks.append(data)
        offset += data.size
    return tuple(records), (np.concatenate(chunks) if chunks else np.empty(0, np.uint8))


def random_volume(rng, shape, density=0.3):
    vol = np.zeros(shape, np.uint8)
    for z in range(shape[0]):
        if rng.random() < 0.25:
            continue  # empty slice
        y0, x0 = rng.integers(0, shape[1]), rng.integers(0, shape[2])
        y1, x1 = rng.integers(y0 + 1, shape[1] + 1), rng.integers(x0 + 1, shape[2] + 1)
        vol[z, y0:y1, x0:x1] = (rng.random((y1 - y0, x1 - x0)) < density) * rng.integers(1, 256)
    return vol


def make_pair(tmp_path, shape, packed, tag='s'):
    fmt = ip.INTERNAL_PACKED_CVOL_FORMAT if packed else ip.CVOL_FORMAT
    new = ip.IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=tmp_path / f'{tag}-new',
                                               format_name=fmt, desc='enc')
    old = LegacyWriter(shape=shape, store_dir=tmp_path / f'{tag}-old', format_name=fmt, desc='enc')
    return new, old


def state(w):
    return (w.index.tobytes(), w._slice_state.tobytes(), w._next_offset, w._nonempty_slices,
            w._foreground_voxels, w._min_t, w._max_t, w._min_y, w._max_y, w._min_x, w._max_x,
            w._active_callbacks, w._failed_reason is not None, w._finalized)


def call(w, *args, **kwargs):
    try:
        w.consume_encoded_block(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - compared exactly
        return (type(exc), str(exc))
    return None


def assert_same(new, old, *args, **kwargs):
    a, b = call(new, *args, **kwargs), call(old, *args, **kwargs)
    assert a == b
    assert state(new) == state(old)
    return a


def finalize_bytes(w):
    stats = w.finalize()
    meta = json.loads(Path(w.meta_path).read_text())
    return (Path(w.index_path).read_bytes(), Path(w.chunks_path).read_bytes(),
            {k: v for k, v in stats.items() if 'path' not in k and 'seconds' not in k},
            {k: v for k, v in meta.items() if 'path' not in k})


@pytest.mark.parametrize('packed', [False, True])
def test_random_blocks_publish_identical_bytes(tmp_path, packed):
    rng = np.random.default_rng(5 + packed)
    shape = (70, 23, 37)
    vol = random_volume(rng, shape)
    new, old = make_pair(tmp_path, shape, packed)
    bounds = sorted(set([0, shape[0]] + list(rng.integers(1, shape[0], 9))))
    blocks = list(zip(bounds[:-1], bounds[1:]))
    rng.shuffle(blocks)
    for a, b in blocks:
        records, data = encode(vol[a:b], a, packed)
        if rng.random() < 0.5:
            data.flags.writeable = False
        assert assert_same(new, old, a, records, data, packed=packed) is None
    assert finalize_bytes(new) == finalize_bytes(old)


def test_real_dense_encoder_output_identical(tmp_path):
    rng = np.random.default_rng(11)
    shape = (130, 61, 77)
    vol = random_volume(rng, shape, 0.5)
    new, old = make_pair(tmp_path, shape, True)
    slot = np.empty(64 * shape[1] * ((shape[2] + 7) // 8), np.uint8)
    for z in range(0, shape[0], 64):
        count = min(64, shape[0] - z)
        records, payload = encode_dense_u8_packed_block(vol, z, count, payload_buffer=slot)
        assert assert_same(new, old, z, records, payload, packed=True) is None
    assert finalize_bytes(new) == finalize_bytes(old)


FIELDS = ('z', 'y0', 'y1', 'x0', 'x1', 'foreground', 'offset', 'size')


def test_mutated_records_raise_identically_and_leave_state(tmp_path):
    rng = np.random.default_rng(3)
    py = random.Random(3)
    shape = (12, 17, 21)
    raised = set()
    for packed in (False, True):
        vol = random_volume(rng, shape, 0.6)
        new, old = make_pair(tmp_path, shape, packed, tag=f'm{packed}')
        records, data = encode(vol, 0, packed)
        for trial in range(500):
            altered = [SimpleNamespace(**vars(r)) for r in records]
            payload = data.copy()
            for _ in range(py.choice((1, 1, 2))):
                kind = py.random()
                rec = py.choice(altered)
                if kind < 0.7:
                    name = py.choice(FIELDS)
                    setattr(rec, name, getattr(rec, name) + py.choice((-9, -2, -1, 1, 2, 9, 64)))
                elif kind < 0.85 and payload.size:
                    payload[py.randrange(payload.size)] ^= np.uint8(1 << py.randrange(8))
                elif kind < 0.92:
                    payload = np.append(payload, np.uint8(0))
                elif payload.size:
                    payload = payload[:-1].copy()
            err = assert_same(new, old, 0, altered, payload, packed=packed)
            if err is not None:
                raised.add(err)
            else:  # mutation kept the contract (e.g. a crop data bit): both published it
                assert finalize_bytes(new) == finalize_bytes(old)
                new, old = make_pair(tmp_path, shape, packed, tag=f'm{packed}-{trial}')
        assert assert_same(new, old, 0, records, data, packed=packed) is None
        assert finalize_bytes(new) == finalize_bytes(old)
    messages = {m.split(': ', 1)[1] for _, m in raised}
    assert {'encoded slice coordinates are invalid', 'encoded crop size/count/offset is invalid',
            'encoded payload ends inside a slice', 'encoded row padding bits must be zero',
            'encoded payload has unclaimed bytes'} <= messages


class Weird:
    def __init__(self, base, **over):
        self.__dict__.update(vars(base))
        self.__dict__.update(over)


@pytest.mark.parametrize('case', ['missing', 'float', 'huge', 'neg_huge', 'numpy', 'bool_like',
                                  'huge_after_bad', 'string', 'short_len'])
def test_unusual_records_take_the_scalar_path_identically(tmp_path, case):
    shape = (4, 9, 13)
    vol = np.zeros(shape, np.uint8)
    vol[0, 2:5, 3:12] = 1
    vol[2, 1, 1] = 1
    vol[3, 0:9, 0:13] = 1
    new, old = make_pair(tmp_path, shape, True)
    records, data = encode(vol, 0, True)
    recs = [SimpleNamespace(**vars(r)) for r in records]
    if case == 'missing':
        recs[2] = SimpleNamespace(**{k: v for k, v in vars(recs[2]).items() if k != 'size'})
    elif case == 'float':
        recs[1] = Weird(recs[1], y0=0.0)
    elif case == 'huge':
        recs[3] = Weird(recs[3], size=1 << 70)
    elif case == 'neg_huge':
        recs[0] = Weird(recs[0], offset=-(1 << 80))
    elif case == 'numpy':
        recs = [Weird(r, **{k: np.uint64(getattr(r, k)) for k in FIELDS}) for r in recs]
    elif case == 'bool_like':
        recs[2] = Weird(recs[2], foreground=True)
    elif case == 'huge_after_bad':
        recs[0] = Weird(recs[0], z=7)  # validation error precedes the overflow below
        recs[3] = Weird(recs[3], x1=1 << 90)
    elif case == 'string':
        recs[3] = Weird(recs[3], z='3')
    elif case == 'short_len':
        class Lying(list):
            def __len__(self):
                return 3
        recs = Lying(recs)
    assert_same(new, old, 0, recs, data, packed=True)
    if case == 'numpy':
        assert finalize_bytes(new) == finalize_bytes(old)


@pytest.mark.parametrize('z0', [0, 1, 2])
@pytest.mark.parametrize('n_iter', [2, 3, 4, 5])
@pytest.mark.parametrize('fg', ['none', 'last', 'all'])
def test_len_disagreeing_with_iteration_never_writes_out_of_bounds(tmp_path, z0, n_iter, fg):
    """Review issue: len()==2 but iteration yields n_iter records starting at z0;
    rows past the index end must raise the original IndexError, abort the writer
    and leave the memory beyond index/_slice_state untouched."""
    shape = (4, 4, 4)

    class Lying(list):
        def __len__(self):
            return 2

    def build():
        out, offset = [], 0
        for i in range(n_iter):
            if fg == 'all' or (fg == 'last' and i == n_iter - 1):
                out.append(SimpleNamespace(z=z0 + i, y0=1, y1=2, x0=i % 4, x1=i % 4 + 1,
                                           foreground=1, offset=offset, size=1))
                offset += 1
            else:
                out.append(SimpleNamespace(z=z0 + i, y0=0, y1=0, x0=0, x1=0,
                                           foreground=0, offset=0, size=0))
        return Lying(out), np.ones(offset, np.uint8)

    results, guards = [], []
    for w in make_pair(tmp_path, shape, False, tag=f'{z0}-{n_iter}-{fg}'):
        big_index = np.zeros(shape[0] + 4, ip.CTILE_INDEX_DTYPE)
        big_state = np.zeros(shape[0] + 4, np.uint8)
        w.index, w._slice_state = big_index[:shape[0]], big_state[:shape[0]]
        records, data = build()
        results.append((call(w, z0, records, data, packed=False), state(w)))
        guards.append((big_index[shape[0]:].tobytes(), big_state[shape[0]:].tobytes()))
    assert results[0] == results[1]
    assert guards[0] == guards[1] == (np.zeros(4, ip.CTILE_INDEX_DTYPE).tobytes(), bytes(4))
    if z0 + n_iter > shape[0]:
        err, st = results[0]
        assert err == (IndexError, f'index {shape[0]} is out of bounds for axis 0 with size {shape[0]}')
        assert st[12] is True  # writer aborted


def test_bad_bool_value_matches_scalar_semantics(tmp_path):
    # foreground=True is index 1; validates for a 1-pixel crop in both versions.
    shape = (2, 4, 4)
    vol = np.zeros(shape, np.uint8)
    vol[1, 2, 3] = 1
    new, old = make_pair(tmp_path, shape, False)
    records, data = encode(vol, 0, False)
    recs = [records[0], Weird(records[1], foreground=True)]
    assert assert_same(new, old, 0, recs, data, packed=False) is None
    assert finalize_bytes(new) == finalize_bytes(old)


def test_duplicate_closed_failed_and_pwrite_error_paths(tmp_path):
    shape = (6, 8, 8)
    vol = np.zeros(shape, np.uint8)
    vol[:, 1:5, 2:7] = 1
    new, old = make_pair(tmp_path, shape, True)
    records, data = encode(vol[:3], 0, True)
    assert assert_same(new, old, 0, records, data, packed=True) is None
    assert assert_same(new, old, 0, records, data, packed=True)[1].endswith('more than once')
    assert assert_same(new, old, 9, records, data, packed=True)[0] is IndexError
    assert assert_same(new, old, 3, records, data, packed=False)[0] is ValueError
    assert assert_same(new, old, 3, records, data.astype(np.int16), packed=True)[0] is ValueError
    for w in (new, old):
        w._pwrite_all = lambda *a: (_ for _ in ()).throw(OSError('disk error'))
    tail, tdata = encode(vol[3:], 3, True)
    assert assert_same(new, old, 3, tail, tdata, packed=True) == (OSError, 'disk error')
    assert new.failed and old.failed
    assert assert_same(new, old, 3, tail, tdata, packed=True) is None  # failed writers ignore
    new.discard()
    old.discard()


def test_empty_block_and_all_empty_slices(tmp_path):
    shape = (3, 5, 5)
    new, old = make_pair(tmp_path, shape, True)
    assert assert_same(new, old, 1, (), np.empty(0, np.uint8), packed=True) is None
    records, data = encode(np.zeros(shape, np.uint8), 0, True)
    assert assert_same(new, old, 0, records, data, packed=True) is None
    assert finalize_bytes(new) == finalize_bytes(old)


def test_warm_compiles_every_production_signature():
    info = ip.warm_encoded_block_kernels()
    assert info['validate_signatures'] >= 2  # writable and readonly payloads
    assert info['commit_signatures'] >= 1
    for kernel in (ip._encoded_block_validate_kernel, ip._encoded_block_commit_kernel):
        assert kernel.targetoptions.get('nogil') is True
        assert type(kernel._cache).__name__ == 'NullCache'
    before = (len(ip._encoded_block_validate_kernel.nopython_signatures),
              len(ip._encoded_block_commit_kernel.nopython_signatures))
    # A production-shaped call (writer index, readonly encoder payload) adds no signature.
    shape = (64, 33, 45)
    rng = np.random.default_rng(1)
    vol = random_volume(rng, shape)
    records, payload = encode_dense_u8_packed_block(vol, 0, 64)
    with contextlib.ExitStack() as stack:
        import tempfile
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        w = ip.IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=root / 's',
                                                 format_name=ip.INTERNAL_PACKED_CVOL_FORMAT, desc='w')
        payload.flags.writeable = False
        w.consume_encoded_block(0, records, payload, packed=True)
        raw = ip.IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=root / 'r',
                                                   format_name=ip.CVOL_FORMAT, desc='w')
        rr, rd = encode(vol, 0, False)
        raw.consume_encoded_block(0, rr, rd, packed=False)
        w.discard()
        raw.discard()
    after = (len(ip._encoded_block_validate_kernel.nopython_signatures),
             len(ip._encoded_block_commit_kernel.nopython_signatures))
    assert after == before
