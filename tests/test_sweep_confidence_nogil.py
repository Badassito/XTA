"""Compiled sweep observed-confidence cells write the generic writer's exact bytes.

The reference readers below are the per-block NumPy implementations that the
compiled frame encoder replaced. Driving the same writer with either must
produce byte-identical payload, index, metadata and catalogue files.
"""
from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np
import pytest

from XTA import geometry
from XTA import sweep_confidence
from XTA.confidence_storage import (
    BLOCK_DTYPE,
    ConfidenceStageLimit,
    _RECORD,
    checked_shape,
    crop_blocks,
    write_blocks,
)
from XTA.sweep_confidence import SweepObservedConfidenceWriter


class ReferenceMaskedScorePlanes:
    """Previous dense plane reader (``write_blocks`` scans it with plane_blocks)."""

    def __init__(self, shape, read_mask, read_scores):
        if not callable(read_mask) or not callable(read_scores):
            raise TypeError('Observed confidence requires mask and score plane readers')
        self.shape = checked_shape(shape)
        self.read_mask, self.read_scores = read_mask, read_scores

    def __call__(self, z):
        mask = self.read_mask(z)
        if mask is None:
            return None
        mask = np.asarray(mask)
        if mask.shape != self.shape[1:] or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
            raise ValueError('Sweep confidence mask plane differs from its native grid')
        if not np.any(mask):
            return None
        values = self.read_scores(z)
        if values is None:
            return None
        values = np.asarray(values)
        if values.dtype != np.uint8 or values.shape != self.shape[1:]:
            raise ValueError('Sweep confidence score plane differs from its native grid')
        return np.where(mask != 0, values, np.uint8(0))


class ReferenceMaskedScoreCrops:
    """Previous per-block strip merger (``write_blocks`` partitions it with crop_blocks)."""

    def __init__(self, shape, read_native_crops, block_size=128):
        if not callable(read_native_crops):
            raise TypeError('Observed confidence needs a native crop reader')
        self.shape = checked_shape(shape)
        self.read_native_crops = read_native_crops
        self.block_size = block_size

    def iter_crops(self, z):
        blocks = {}
        height, width = self.shape[1:]
        block = self.block_size
        for y0, y1, x0, x1, mask, scores in self.read_native_crops(z):
            if scores is None:
                continue
            y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
            mask, scores = np.asarray(mask), np.asarray(scores)
            if (not 0 <= y0 < y1 <= height or not 0 <= x0 < x1 <= width
                    or mask.shape != (y1-y0, x1-x0) or scores.shape != mask.shape
                    or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8))
                    or scores.dtype != np.uint8):
                raise ValueError('Sweep confidence native crop differs from its grid')
            values = np.where(mask != 0, scores, np.uint8(0))
            for by in range(y0//block*block, y1, block):
                top, bottom = max(by, y0), min(by+block, y1)
                for bx in range(x0//block*block, x1, block):
                    left, right = max(bx, x0), min(bx+block, x1)
                    part = values[top-y0:bottom-y0, left-x0:right-x0]
                    if not np.any(part):
                        continue
                    key = (by, bx)
                    if key not in blocks:
                        blocks[key] = np.zeros((min(block, height-by), min(block, width-bx)), np.uint8)
                    target = blocks[key][top-by:bottom-by, left-bx:right-bx]
                    np.maximum(target, part, out=target)
        for (y, x), values in sorted(blocks.items()):
            yield y, y+values.shape[0], x, x+values.shape[1], values


def reference_reader(shape, crops):
    planes = getattr(crops, '__self__', None)
    if isinstance(planes, sweep_confidence._MaskedScorePlanes):
        return ReferenceMaskedScorePlanes(shape, planes.read_mask, planes.read_scores)
    return ReferenceMaskedScoreCrops(shape, crops)


def digests(root):
    return {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*')) if path.is_file()}


def publish(root, shape, *, reference, monkeypatch, cutoff=60, metrics=None, **readers):
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    with monkeypatch.context() as patch:
        if reference:
            patch.setattr(sweep_confidence, '_MaskedCropFrameEncoder', reference_reader)
        writer = SweepObservedConfidenceWriter(root, cutoff_hundredths=cutoff,
            source_shape_tyx=shape, run_id='nogil', task='segment', min_conf=0.0,
            confidence_floor_by_backend={'gpu': cutoff / 100})
        ref = writer.publish_view(view, 'model', native_shape=shape, metrics=metrics, **readers)
        writer.finish()
    return ref, digests(root)


SHAPES = [(4, 300, 410), (3, 128, 256), (2, 129, 257), (5, 1, 700), (3, 700, 1),
          (6, 255, 383), (3, 64, 64), (2, 1000, 130), (1, 127, 129), (2, 520, 777)]


def layout(rng, values):
    """Randomize memory layout and writeability without changing values."""
    kind = rng.integers(0, 4)
    if kind == 1:
        values = np.asfortranarray(values)
    elif kind == 2:
        padded = np.zeros((values.shape[0] * 2, values.shape[1] * 3), values.dtype)
        padded[::2, ::3] = values
        values = padded[::2, ::3]
    if rng.integers(0, 3) == 0:
        values = values.view()
        values.setflags(write=False)
    return values


def random_mask(rng, shape):
    support = rng.random(shape) < rng.choice([0.0, 0.01, 0.3, 0.9, 1.0])
    kind = rng.integers(0, 4)
    if kind == 0:
        return support
    scale = (1, 255, None)[kind - 1]
    return (support * (rng.integers(1, 4, shape) if scale is None else scale)).astype(np.uint8)


def random_scores(rng, shape):
    kind = rng.integers(0, 6)
    if kind == 0:
        return None
    if kind == 1:
        return np.zeros(shape, np.uint8)
    values = rng.integers(0, 256, shape, dtype=np.uint8)
    if kind == 2:
        values[rng.random(shape) < 0.5] = 0
    elif kind == 3:
        values[:] = rng.integers(1, 256)
    return values


def random_strips(seed):
    rng = np.random.default_rng(seed)
    shape = SHAPES[seed % len(SHAPES)]
    height, width = shape[1:]
    frames = []
    for _ in range(shape[0]):
        crops = []
        for _ in range(int(rng.integers(0, 7))):
            style = rng.integers(0, 4)
            if style == 0:
                y0, x0 = int(rng.integers(0, height)), int(rng.integers(0, width))
                y1, x1 = y0 + 1, x0 + 1
            elif style == 1:
                y0, y1, x0, x1 = 0, height, 0, width
            elif style == 2:
                y0 = int(rng.integers(0, height))
                y1, x0, x1 = int(min(height, y0 + rng.integers(1, 200))), 0, width
            else:
                y0, x0 = int(rng.integers(0, height)), int(rng.integers(0, width))
                y1 = int(rng.integers(y0 + 1, min(height, y0 + 300) + 1))
                x1 = int(rng.integers(x0 + 1, min(width, x0 + 300) + 1))
            mask = layout(rng, random_mask(rng, (y1 - y0, x1 - x0)))
            scores = random_scores(rng, mask.shape)
            crops.append((y0, y1, x0, x1, mask, None if scores is None else layout(rng, scores)))
        frames.append(crops)
    return shape, frames


@pytest.mark.parametrize('seed', range(40))
def test_random_overlapping_strips_write_identical_files(tmp_path, monkeypatch, seed):
    shape, frames = random_strips(seed)
    read = lambda z: list(frames[z])
    _, expected = publish(tmp_path / 'reference', shape, reference=True,
                          monkeypatch=monkeypatch, read_native_crops=read)
    metrics = {}
    _, actual = publish(tmp_path / 'compiled', shape, reference=False,
                        monkeypatch=monkeypatch, metrics=metrics, read_native_crops=read)
    assert actual == expected
    assert len(actual) == 4  # manifest, metadata, index, payload
    assert metrics['frame_encoder'] == 'caller_compiled_cells'
    assert metrics['frame_encoder_codec'] in ('python_zlib', 'native_compress2')
    assert metrics['frame_encoder_codec_fallback_frames'] == 0


@pytest.mark.parametrize('seed', range(20))
def test_random_dense_planes_write_identical_files(tmp_path, monkeypatch, seed):
    rng = np.random.default_rng(1000 + seed)
    shape = SHAPES[seed % len(SHAPES)]
    masks, scores = [], []
    for _ in range(shape[0]):
        kind = rng.integers(0, 5)
        masks.append(None if kind == 0 else np.zeros(shape[1:], bool) if kind == 1
                     else layout(rng, random_mask(rng, shape[1:])))
        value = random_scores(rng, shape[1:]) if rng.integers(0, 4) else None
        scores.append(None if value is None else layout(rng, value))
    readers = dict(read_mask=lambda z: masks[z], read_scores=lambda z: scores[z])
    _, expected = publish(tmp_path / 'reference', shape, reference=True,
                          monkeypatch=monkeypatch, **readers)
    _, actual = publish(tmp_path / 'compiled', shape, reference=False,
                        monkeypatch=monkeypatch, **readers)
    assert actual == expected


def journal_like_frames(seed, shape, chunk_rows=37):
    """Tight-bbox detector support split into read-only row strips, as journal readers yield."""
    rng = np.random.default_rng(seed)
    frames = []
    height, width = shape[1:]
    for _ in range(shape[0]):
        if rng.random() < 0.2:
            frames.append([])
            continue
        y0, x0 = int(rng.integers(0, height // 2)), int(rng.integers(0, width // 2))
        y1, x1 = int(rng.integers(y0 + 1, height + 1)), int(rng.integers(x0 + 1, width + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        centre_y, centre_x = (y0 + y1) / 2, (x0 + x1) / 2
        radius = max(1.0, min(y1 - y0, x1 - x0) / 2)
        support = ((yy - centre_y) ** 2 + (xx - centre_x) ** 2 <= radius ** 2)
        support &= rng.random(support.shape) > 0.02
        score = np.where(support, rng.integers(128, 256, support.shape), 0).astype(np.uint8)
        crops = []
        for top in range(y0, y1, chunk_rows):
            bottom = min(y1, top + chunk_rows)
            packed = np.packbits(support[top - y0:bottom - y0], axis=None, bitorder='little')
            mask = np.unpackbits(packed, bitorder='little',
                                 count=(bottom - top) * (x1 - x0)).reshape(bottom - top, x1 - x0)
            mask.setflags(write=False)
            values = np.frombuffer(score[top - y0:bottom - y0].tobytes(), np.uint8).reshape(mask.shape)
            crops.append((top, bottom, x0, x1, mask, values))
        frames.append(crops)
    return frames


@pytest.mark.parametrize('cutoff', [60, 70, 80, 90])
def test_journal_shaped_read_only_strips_write_identical_files(tmp_path, monkeypatch, cutoff):
    shape = (6, 900, 1100)
    frames = journal_like_frames(cutoff, shape)
    read = lambda z: iter(frames[z])
    _, expected = publish(tmp_path / 'reference', shape, reference=True, cutoff=cutoff,
                          monkeypatch=monkeypatch, read_native_crops=read)
    ref, actual = publish(tmp_path / 'compiled', shape, reference=False, cutoff=cutoff,
                          monkeypatch=monkeypatch, read_native_crops=read)
    assert actual == expected
    assert ref.metadata['block_count'] > 5
    catalogue = json.loads((tmp_path / 'compiled' / 'reconciliation_evidence' / 'manifest.json').read_text())
    assert catalogue['observed_confidence']['capture_confidence_floor'] == cutoff / 100


@pytest.mark.parametrize('seed', [7, 13, 24])
def test_frame_encoder_records_equal_crop_blocks_and_leave_clean_workspace(seed):
    shape, frames = random_strips(seed)
    encoder = sweep_confidence._MaskedCropFrameEncoder(shape, lambda z: list(frames[z]))
    reference = ReferenceMaskedScoreCrops(shape, lambda z: list(frames[z]))
    for z in range(shape[0]):
        records, payload, known, timings = encoder.encode_caller_frame(z, 128)
        index, chunks, expected_known, offset = bytearray(), [], 0, 0
        for y0, y1, x0, x1, crop in crop_blocks(reference.iter_crops(z), shape[1:], 128):
            encoded = zlib.compress(crop.tobytes(), level=3)
            index += _RECORD.pack(z, y0, x0, y1 - y0, x1 - x0, offset, len(encoded))
            chunks.append(encoded)
            offset += len(encoded)
            expected_known += int(np.count_nonzero(crop))
        assert records.dtype == BLOCK_DTYPE and records.tobytes() == bytes(index)
        assert payload == b''.join(chunks) and known == expected_known
        assert set(timings) >= {'read_seconds', 'scan_seconds', 'compression_seconds'}
        assert not encoder._plane.any() and not encoder._touched.any()
    with pytest.raises(ValueError, match='grid differs'):
        encoder.encode_caller_frame(0, 64)


def test_numeric_staging_limit_matches_reference_at_the_exact_boundary(tmp_path):
    shape = (4, 520, 777)
    frames = journal_like_frames(3, shape)
    read = lambda z: iter(frames[z])
    common = dict(layer_key='layer', model_name='model', provenance={},
                  coordinate_space='native_view_processing', source_shape=shape)
    exact = write_blocks(tmp_path / 'measure', shape, ReferenceMaskedScoreCrops(shape, read), **common)
    limit = exact['payload_bytes'] + exact['block_count'] * _RECORD.size
    for name, make in (('reference', ReferenceMaskedScoreCrops),
                       ('compiled', sweep_confidence._MaskedCropFrameEncoder)):
        fits = write_blocks(tmp_path / f'{name}-fits', shape, make(shape, read),
                            max_numeric_bytes=limit, **common)
        assert fits == exact
        with pytest.raises(ConfidenceStageLimit):
            write_blocks(tmp_path / f'{name}-over', shape, make(shape, read),
                         max_numeric_bytes=limit - 1, **common)
        assert not any((tmp_path / f'{name}-over').iterdir())


def qualified_zlib_library():
    """A zlib library at Python's runtime version for the explicit-qualification route."""
    import ctypes
    import importlib.util
    import os
    from pathlib import Path
    candidates = [os.environ.get('XTA_TEST_SWEEP_NATIVE_ZLIB_LIBRARY', ''), getattr(zlib, '__file__', '')]
    spec = importlib.util.find_spec('torchvision')
    if spec is not None and spec.origin:
        candidates.append(str(Path(spec.origin).parent / 'zlib.dll'))
    for path in filter(None, candidates):
        try:
            library = ctypes.CDLL(path)
            library.zlibVersion.restype = ctypes.c_char_p
            if library.zlibVersion().decode() == zlib.ZLIB_RUNTIME_VERSION:
                return path
        except (OSError, AttributeError):
            continue
    return None


@pytest.fixture
def native_compression(monkeypatch):
    path = qualified_zlib_library()
    if path is None:
        pytest.skip('No zlib library at the Python zlib runtime version for native qualification')
    monkeypatch.setenv('YOLO_TTA_CONFIDENCE_NATIVE_CODEC', 'auto')
    monkeypatch.setenv('XTA_QUALIFY_NATIVE_ZLIB', '1')
    monkeypatch.setenv('XTA_TEST_NATIVE_ZLIB_LIBRARY', path)
    kernel, backend = sweep_confidence._native_cell_compressor()
    assert backend == 'native_compress2' and kernel is not None
    return kernel


@pytest.mark.parametrize('seed', [1, 4, 9, 16, 25, 36])
def test_native_frame_compression_writes_identical_files(tmp_path, monkeypatch, native_compression, seed):
    shape, frames = random_strips(seed)
    read = lambda z: list(frames[z])
    _, expected = publish(tmp_path / 'reference', shape, reference=True,
                          monkeypatch=monkeypatch, read_native_crops=read)
    metrics = {}
    _, actual = publish(tmp_path / 'compiled', shape, reference=False,
                        monkeypatch=monkeypatch, metrics=metrics, read_native_crops=read)
    assert actual == expected
    assert metrics['frame_encoder_codec'] == 'native_compress2'


def test_native_journal_shaped_frames_write_identical_files(tmp_path, monkeypatch, native_compression):
    shape = (6, 900, 1100)
    frames = journal_like_frames(11, shape)
    read = lambda z: iter(frames[z])
    _, expected = publish(tmp_path / 'reference', shape, reference=True,
                          monkeypatch=monkeypatch, read_native_crops=read)
    ref, actual = publish(tmp_path / 'compiled', shape, reference=False,
                          monkeypatch=monkeypatch, read_native_crops=read)
    assert actual == expected and ref.metadata['block_count'] > 5


def test_failed_native_frame_falls_back_to_exact_python_zlib(tmp_path, monkeypatch):
    shape, frames = random_strips(12)
    read = lambda z: list(frames[z])
    _, expected = publish(tmp_path / 'reference', shape, reference=True,
                          monkeypatch=monkeypatch, read_native_crops=read)
    failing = lambda raw, table: (np.empty(0, np.uint8), np.zeros(len(table), np.int64), 0, -5)
    monkeypatch.setattr(sweep_confidence, '_native_cell_compressor', lambda: (failing, 'native_compress2'))
    metrics = {}
    _, actual = publish(tmp_path / 'compiled', shape, reference=False,
                        monkeypatch=monkeypatch, metrics=metrics, read_native_crops=read)
    assert actual == expected
    assert metrics['frame_encoder_codec_fallback_frames'] > 0


@pytest.mark.parametrize('version, source', [('0.0.0', 'python_zlib_extension'),
                                             (zlib.ZLIB_RUNTIME_VERSION, 'system_zlib')])
def test_native_compression_requires_python_zlib_identity(monkeypatch, version, source):
    from XTA import confidence_native_codec

    class Codec:
        pass

    codec = Codec()
    codec.version = version
    monkeypatch.setattr(confidence_native_codec, 'native_confidence_codec',
                        lambda: (codec, dict(backend='native_compress2', library_source=source)))
    assert sweep_confidence._native_cell_compressor() == (None, 'python_zlib')


@pytest.mark.parametrize('mask_readonly', [False, True])
@pytest.mark.parametrize('score_readonly', [False, True])
def test_sweep_crop_kernels_accept_readonly_inputs(mask_readonly, score_readonly):
    mask = np.array([[1, 0], [2, 255]], np.uint8)
    scores = np.array([[7, 9], [0, 13]], np.uint8)
    mask.setflags(write=not mask_readonly)
    scores.setflags(write=not score_readonly)
    plane = np.zeros((6, 8), np.uint8)
    touched = np.zeros((2, 2), np.uint8)
    sweep_confidence._merge_masked_crop(plane, touched, mask, scores, 1, 4, 4)
    table, raw = sweep_confidence._take_frame_cells(plane, touched, 4)
    np.testing.assert_array_equal(table, [[1, 4, 2, 2, 2, 0]])
    assert raw.tobytes() == bytes((7, 0, 0, 13))
    assert not np.any(plane) and not np.any(touched)


@pytest.mark.parametrize('bad', ['outside', 'shape', 'mask_dtype', 'score_dtype'])
def test_invalid_strip_fails_like_reference_without_files(tmp_path, monkeypatch, bad):
    shape = (2, 260, 270)
    mask = np.ones((10, 12), np.uint8)
    scores = np.full(mask.shape, 99, np.uint8)
    crop = dict(outside=(255, 265, 0, 12, mask, scores),
                shape=(0, 10, 0, 11, mask, scores),
                mask_dtype=(0, 10, 0, 12, mask.astype(np.int16), scores),
                score_dtype=(0, 10, 0, 12, mask, scores.astype(np.float32)))[bad]
    good = (100, 110, 100, 112, mask, scores)
    read = lambda z: [good, crop] if z == 1 else [good]
    for reference in (True, False):
        root = tmp_path / str(reference)
        with pytest.raises(ValueError, match='native crop differs from its grid'):
            publish(root, shape, reference=reference, monkeypatch=monkeypatch, read_native_crops=read)
        assert not (root / 'reconciliation_evidence' / 'manifest.json').exists()
        assert not list(root.rglob('metadata.json'))
        assert not list(root.rglob('*.partial'))
