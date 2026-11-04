"""Batched nogil score-block restore is identical to the per-block Python reader."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import types
import zlib

import numpy as np
import pytest

from XTA import confidence_native_inflate as inflate
from XTA import confidence_storage as storage
from XTA.confidence_storage import BLOCK_DTYPE, BLOCK_LAYOUT, BlockScoreReader


def _old_iter_crops(self, z):
    """Verbatim pre-change BlockScoreReader.iter_crops (git HEAD)."""
    if self._closed:
        raise RuntimeError('Confidence block reader is closed')
    z = int(z)
    if not 0 <= z < self.shape[0]:
        raise IndexError('Confidence crop is outside its payload grid')
    first,stop = np.searchsorted(self.index['z'],(z,z+1))
    with (self.reference.path/'scores.u8.zlib').open('rb') as stream:
        for number in range(int(first),int(stop)):
            if self._closed:
                raise RuntimeError('Confidence block reader is closed')
            record=self.index[number]
            y,x,h,w,offset,length = (int(record[k]) for k in ('y','x','h','w','offset','length'))
            stream.seek(offset)
            encoded = stream.read(length)
            decoder = zlib.decompressobj()
            raw = decoder.decompress(encoded,h*w+1)
            if len(raw)!=h*w or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError('Confidence score block is truncated or malformed')
            yield y,y+h,x,x+w,np.frombuffer(raw,np.uint8).reshape(h,w)


def _old_restore(reader, shape, stop=None):
    """Verbatim pre-change sam_parent_staging score_blocks loop."""
    output = np.zeros(shape, np.uint8)
    decoded_bytes = 0
    error = None
    try:
        for z in range(shape[0]):
            if stop is not None and stop.is_set():
                raise RuntimeError('SAM compact parent restore cancelled')
            for y0,y1,x0,x1,crop in _old_iter_crops(reader, z):
                output[z,y0:y1,x0:x1] = crop
                decoded_bytes += int(crop.nbytes)
    except Exception as exc:
        error = (type(exc), str(exc))
    return output, decoded_bytes, error


def _library():
    if os.name != 'nt':
        return None
    import torchvision
    candidate = Path(torchvision.__file__).parent / 'zlib.dll'
    return str(candidate) if candidate.exists() else None


@pytest.fixture
def native(monkeypatch):
    library = _library()
    if os.name == 'nt':
        if library is None:
            pytest.skip('no qualified zlib library on this Windows host')
        monkeypatch.setenv('XTA_QUALIFY_NATIVE_ZLIB', '1')
        monkeypatch.setenv('XTA_TEST_NATIVE_ZLIB_LIBRARY', library)
    monkeypatch.setenv('YOLO_TTA_CONFIDENCE_NATIVE_INFLATE', 'auto')
    inflater, receipt = inflate.native_confidence_inflater()
    if inflater is None:
        pytest.skip(f'native inflate unavailable: {receipt}')
    return inflater


def _write_fixture(root, shape, blocks):
    """blocks: (z, y, x, raw2d, encoded) in stored order."""
    root.mkdir(parents=True, exist_ok=True)
    rows, offset = [], 0
    with open(root / 'scores.u8.zlib', 'wb') as stream:
        for z, y, x, raw, encoded in blocks:
            stream.write(encoded)
            rows.append((z, y, x, raw.shape[0], raw.shape[1], offset, len(encoded)))
            offset += len(encoded)
    np.array(rows, BLOCK_DTYPE).tofile(root / 'index.bin')
    return types.SimpleNamespace(storage_shape=shape, path=root, metadata=dict(
        layout=BLOCK_LAYOUT, index_record_bytes=BLOCK_DTYPE.itemsize, block_count=len(rows),
        payload_bytes=offset))


def _random_blocks(seed, shape, *, overlap=False, block=128):
    rng = np.random.default_rng(seed)
    blocks = []
    for z in range(shape[0]):
        if rng.random() < 0.2:
            continue  # empty frame
        for y in range(0, shape[1], block):
            for x in range(0, shape[2], block):
                if rng.random() < 0.35:
                    continue
                h = int(rng.integers(1, min(block, shape[1]-y)+1))
                w = int(rng.integers(1, min(block, shape[2]-x)+1))
                y0 = y+int(rng.integers(0, min(block, shape[1]-y)-h+1))
                x0 = x+int(rng.integers(0, min(block, shape[2]-x)-w+1))
                raw = rng.integers(0, 256, (h, w), dtype=np.uint8)
                raw[rng.random((h, w)) < rng.random()] = 0
                level = int(rng.integers(0, 10))
                blocks.append((z, y0, x0, raw, zlib.compress(raw.tobytes(), level)))
        if overlap and blocks and blocks[-1][0] == z:
            # Accepted index: overlapping crops are assigned in record order.
            raw = rng.integers(1, 256, (40, 50), dtype=np.uint8)
            blocks.append((z, 3, 5, raw, zlib.compress(raw.tobytes(), 3)))
    return blocks


def _compare_iter(reader, shape):
    for z in range(shape[0]):
        old = list(_old_iter_crops(reader, z))
        new = list(reader.iter_crops(z))
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert [type(v) for v in a[:4]] == [type(v) for v in b[:4]] == [int]*4
            assert a[:4] == b[:4]
            assert b[4].dtype == np.uint8 and b[4].shape == a[4].shape and b[4].flags.c_contiguous
            assert not b[4].flags.writeable
            assert a[4].tobytes() == b[4].tobytes()


@pytest.mark.parametrize('seed,shape,overlap', [(1, (6, 300, 257), False), (2, (5, 129, 1), False),
                                                 (3, (7, 256, 384), True), (4, (3, 1, 513), False)])
def test_iter_and_scatter_identical(native, tmp_path, monkeypatch, seed, shape, overlap):
    reference = _write_fixture(tmp_path / 'e', shape, _random_blocks(seed, shape, overlap=overlap))
    reader = BlockScoreReader(reference)
    try:
        _compare_iter(reader, shape)
        assert reader.decode_backend == 'native_uncompress2'
        expected, expected_bytes, error = _old_restore(reader, shape)
        assert error is None
        for limit in (storage._INFLATE_SCATTER_BATCH_BYTES, 1):
            monkeypatch.setattr(storage, '_INFLATE_SCATTER_BATCH_BYTES', limit)
            output, progress, calls = np.zeros(shape, np.uint8), {}, []
            reader.scatter_into(output, before_batch=lambda: calls.append(1), progress=progress)
            assert output.tobytes() == expected.tobytes()
            assert progress['decoded_bytes'] == expected_bytes
            assert progress['decode_backend'] == 'native_uncompress2' and len(calls) >= 1
        monkeypatch.setattr(storage, '_INFLATE_DECODED_BATCH_BYTES', 1)
        _compare_iter(reader, shape)
    finally:
        reader.close()


def test_memmap_destination_and_python_fallback(native, tmp_path, monkeypatch):
    shape = (5, 260, 300)
    reference = _write_fixture(tmp_path / 'e', shape, _random_blocks(9, shape, overlap=True))
    reader = BlockScoreReader(reference)
    try:
        expected, expected_bytes, _ = _old_restore(reader, shape)
        output = np.memmap(tmp_path / 'out.u8', mode='w+', dtype=np.uint8, shape=shape)
        progress = {}
        reader.scatter_into(output, progress=progress)
        assert np.asarray(output).tobytes() == expected.tobytes()
        assert progress['decoded_bytes'] == expected_bytes and progress['decode_batches'] >= 1
        output._mmap.close()
        # Non-contiguous destinations keep the per-crop assignment loop.
        wide = np.zeros((shape[0], shape[1], shape[2]*2), np.uint8)
        progress = {}
        reader.scatter_into(wide[:, :, ::2], progress=progress)
        assert wide[:, :, ::2].tobytes() == expected.tobytes() and progress['decoded_bytes'] == expected_bytes
        assert progress['decode_backend'] == 'python_zlib'
    finally:
        reader.close()
    monkeypatch.setenv('YOLO_TTA_CONFIDENCE_NATIVE_INFLATE', 'off')
    reader = BlockScoreReader(reference)
    try:
        progress, output = {}, np.zeros(shape, np.uint8)
        reader.scatter_into(output, progress=progress)
        assert output.tobytes() == expected.tobytes() and progress['decode_backend'] == 'python_zlib'
        _compare_iter(reader, shape)
        assert reader.decode_backend == 'python_zlib'
    finally:
        reader.close()


def _corrupt_cases(rng):
    raw = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    good = zlib.compress(raw.tobytes(), 3)
    dictionary = zlib.compressobj(3, zdict=b'xyz'*30)
    with_dictionary = dictionary.compress(raw.tobytes())+dictionary.flush()
    flipped = bytearray(good); flipped[len(good)//2] ^= 0x55
    checksum = bytearray(good); checksum[-1] ^= 1
    return {
        'flipped': (raw, bytes(flipped)),
        'checksum': (raw, bytes(checksum)),
        'trailing': (raw, good+b'\0'),
        'truncated': (raw, good[:-3]),
        'too_long': (raw[:, :63], good),
        'too_short': (np.zeros((64, 65), np.uint8), good),
        'dictionary': (raw, with_dictionary),
        'header': (raw, b'\x79'+good[1:]),
    }


@pytest.mark.parametrize('case', ['flipped', 'checksum', 'trailing', 'truncated', 'too_long',
                                  'too_short', 'dictionary', 'header'])
def test_malformed_block_fails_like_python(native, tmp_path, case):
    rng = np.random.default_rng(11)
    shape = (4, 256, 256)
    blocks = [(0, 0, 0, *(lambda r: (r, zlib.compress(r.tobytes(), 3)))(
                  rng.integers(0, 256, (100, 90), dtype=np.uint8)))]
    raw, encoded = _corrupt_cases(rng)[case]
    good = rng.integers(0, 256, (30, 30), dtype=np.uint8)
    blocks += [(1, 0, 0, good, zlib.compress(good.tobytes())), (2, 128, 0, good, zlib.compress(good.tobytes())),
               (2, 0, 128, raw, encoded), (2, 200, 200, good, zlib.compress(good.tobytes())),
               (3, 0, 0, good, zlib.compress(good.tobytes()))]
    reference = _write_fixture(tmp_path / 'e', shape, blocks)
    reader = BlockScoreReader(reference)
    try:
        def drain(iterator):
            items = []
            try:
                for item in iterator:
                    items.append((item[:4], item[4].tobytes()))
            except Exception as exc:
                return items, (type(exc), str(exc))
            return items, None
        for z in range(shape[0]):
            assert drain(reader.iter_crops(z)) == drain(_old_iter_crops(reader, z))
        expected, expected_bytes, error = _old_restore(reader, shape)
        output, progress = np.zeros(shape, np.uint8), {}
        try:
            reader.scatter_into(output, progress=progress)
            got = None
        except Exception as exc:
            got = (type(exc), str(exc))
        assert got == error
        assert output.tobytes() == expected.tobytes()
        assert progress['decoded_bytes'] == expected_bytes
    finally:
        reader.close()


def test_payload_shrunk_after_open_fails_like_python(native, tmp_path):
    shape = (3, 128, 128)
    blocks = _random_blocks(5, shape)
    blocks = [b for b in blocks] or pytest.skip('empty fixture')
    reference = _write_fixture(tmp_path / 'e', shape, blocks)
    reader = BlockScoreReader(reference)
    try:
        payload = tmp_path / 'e' / 'scores.u8.zlib'
        data = payload.read_bytes()
        payload.write_bytes(data[:len(data)-7])
        expected, expected_bytes, error = _old_restore(reader, shape)
        output, progress = np.zeros(shape, np.uint8), {}
        with pytest.raises(error[0]) as caught:
            reader.scatter_into(output, progress=progress)
        assert str(caught.value) == error[1]
        assert output.tobytes() == expected.tobytes() and progress['decoded_bytes'] == expected_bytes
    finally:
        reader.close()


def test_cancellation_and_closed_reader(native, tmp_path):
    shape = (6, 256, 256)
    reference = _write_fixture(tmp_path / 'e', shape, _random_blocks(7, shape))
    reader = BlockScoreReader(reference)
    stop = threading.Event()
    stop.set()
    def check():
        if stop.is_set():
            raise RuntimeError('SAM compact parent restore cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        reader.scatter_into(np.zeros(shape, np.uint8), before_batch=check)
    iterator = reader.iter_crops(next(z for z in range(shape[0]) if any(True for _ in _old_iter_crops(reader, z))))
    next(iterator)
    reader.close()
    with pytest.raises(RuntimeError, match='closed'):
        list(iterator)
    with pytest.raises(RuntimeError, match='closed'):
        list(reader.iter_crops(0))
    with pytest.raises(RuntimeError, match='closed'):
        reader.scatter_into(np.zeros(shape, np.uint8))
    assert reader._payload is None


def test_concurrent_iterators_share_one_reader(native, tmp_path):
    shape = (8, 256, 256)
    reference = _write_fixture(tmp_path / 'e', shape, _random_blocks(13, shape))
    reader = BlockScoreReader(reference)
    try:
        expected = [[(c[:4], c[4].tobytes()) for c in _old_iter_crops(reader, z)] for z in range(shape[0])]
        failures = []
        def work(offset):
            for repeat in range(20):
                for z in range(shape[0]):
                    zz = (z+offset) % shape[0]
                    got = [(c[:4], c[4].tobytes()) for c in reader.iter_crops(zz)]
                    if got != expected[zz]:
                        failures.append(zz)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not failures
    finally:
        reader.close()


def test_write_blocks_round_trip_and_staging_restore(native, tmp_path):
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    rng = np.random.default_rng(21)
    volume = rng.integers(0, 256, (5, 300, 270), dtype=np.uint8)
    volume[rng.random(volume.shape) < 0.6] = 0
    volume[2] = 0
    storage.write_blocks(tmp_path / 'ev', volume.shape, lambda z: volume[z], layer_key='k',
                         model_name='m', provenance={})
    reference = ConfidenceEvidenceRef.open(tmp_path / 'ev')
    with reference.native_reader() as reader:
        for z in range(volume.shape[0]):
            plane = np.zeros(volume.shape[1:], np.uint8)
            for y0, y1, x0, x1, crop in reader.iter_crops(z):
                plane[y0:y1, x0:x1] = crop
            assert plane.tobytes() == volume[z].tobytes()
    reader = BlockScoreReader(reference)
    try:
        output = np.zeros(volume.shape, np.uint8)
        reader.scatter_into(output)
        assert output.tobytes() == volume.tobytes()
    finally:
        reader.close()


def test_warm_function_compiles_production_signatures(native):
    receipt = inflate.warm_confidence_inflate_kernels()
    assert receipt['backend'] == 'native_uncompress2'
    assert receipt['blocks_signatures'] == 1 and receipt['scatter_signatures'] == 1
    # Production argument types: fresh payload/rows/decode arrays and a C uint8 3-D destination.
    signatures = native._inflate_scatter.nopython_signatures
    output_type = signatures[0].args[3]
    assert signatures[0].args[1].layout == 'C' and not signatures[0].args[1].mutable
    assert output_type.dtype.name == 'uint8' and output_type.ndim == 3 and output_type.layout == 'C'
    assert output_type.mutable


def test_positioned_read_path_with_partial_reads(native, tmp_path, monkeypatch):
    """The POSIX preadv branch (emulated here) loops over partial reads."""
    calls = []
    def fake_preadv(fd, buffers, offset):
        (buffer,) = buffers
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, min(len(buffer), 1000))  # deliberately partial
        buffer[:len(data)] = data
        calls.append(offset)
        return len(data)
    monkeypatch.setattr(storage, '_PREADV', fake_preadv)
    shape = (4, 256, 300)
    reference = _write_fixture(tmp_path / 'e', shape, _random_blocks(17, shape, overlap=True))
    reader = BlockScoreReader(reference)
    try:
        _compare_iter(reader, shape)
        expected, expected_bytes, _ = _old_restore(reader, shape)
        output, progress = np.zeros(shape, np.uint8), {}
        reader.scatter_into(output, progress=progress)
        assert output.tobytes() == expected.tobytes() and progress['decoded_bytes'] == expected_bytes
        assert len(calls) > progress['decode_batches']
        # A short positioned read hands the failure to the Python reader.
        reader._payload.close()
        reader._payload = None
        payload = tmp_path / 'e' / 'scores.u8.zlib'
        payload.write_bytes(payload.read_bytes()[:-5])
        expected, expected_bytes, error = _old_restore(reader, shape)
        output, progress = np.zeros(shape, np.uint8), {}
        with pytest.raises(error[0]) as caught:
            reader.scatter_into(output, progress=progress)
        assert str(caught.value) == error[1]
        assert output.tobytes() == expected.tobytes() and progress['decoded_bytes'] == expected_bytes
    finally:
        reader.close()
