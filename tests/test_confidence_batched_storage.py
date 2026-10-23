"""Bounded confidence batches preserve exact stream bytes and numeric limits."""
from pathlib import Path
import zlib

import numpy as np
import pytest

from XTA import confidence_storage as storage
from XTA.confidence_capture import confidence_capture_resources, plan_confidence_capture
from XTA.confidence_evidence import _MaskedNativeScoreReader
from tests.test_confidence_parallel_capture import fixture


class NoTellWriter:
    def __init__(self, stream, writes):
        self.stream, self.writes = stream, writes
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return self.stream.__exit__(*args)
    def write(self, value):
        self.writes.append(len(value))
        return self.stream.write(value)
    def flush(self):
        return self.stream.flush()
    def tell(self):
        raise AssertionError('Confidence offsets must not query the raw file per cell')


@pytest.mark.parametrize('block', [8, 128])
def test_bounded_batches_keep_payload_index_metadata_exact_without_tell(tmp_path, monkeypatch, block):
    mask, scores, active, boxes = fixture()
    options = dict(layer_key='k', model_name='m', provenance={},
        coordinate_space='native_view_processing', source_shape=scores.shape, block_size=block)
    storage.write_blocks(tmp_path/'reference', scores.shape,
        lambda z:np.where(mask[z], scores[z], np.uint8(0)), **options)
    plan = plan_confidence_capture(scores.shape, 3, workspace_bytes=64*1024**2, block_size=block)
    payload_writes, index_writes = [], []
    original_open = Path.open
    def controlled_open(path, mode='r', *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if mode == 'wb' and path.parent == tmp_path/'actual':
            if path.name == 'scores.u8.zlib.partial':
                return NoTellWriter(stream, payload_writes)
            if path.name == 'index.bin.partial':
                return NoTellWriter(stream, index_writes)
        return stream
    monkeypatch.setattr(Path, 'open', controlled_open)
    metrics = {}
    with confidence_capture_resources(plan):
        reader = _MaskedNativeScoreReader(mask, scores, active, boxes)
        storage.write_blocks(tmp_path/'actual', scores.shape, reader, metrics=metrics, **options)
    for name in ('index.bin', 'scores.u8.zlib', 'metadata.json'):
        assert (tmp_path/'actual'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()
    assert len(payload_writes) == metrics['storage_payload_write_calls'] < metrics['block_count']
    assert len(index_writes) == metrics['storage_index_write_calls'] == len(payload_writes)
    assert 0 < metrics['storage_batch_limit_bytes'] <= 1024**2
    assert metrics['storage_batch_reserved_bytes'] <= plan.consumer_bytes


def test_large_single_cell_streams_without_a_consumer_sized_duplicate(tmp_path):
    shape = (3, 128, 128)
    mask = np.ones(shape, np.uint8)
    scores = np.random.default_rng(160591).integers(0, 256, shape, np.uint8)
    active = np.ones(shape[0], bool)
    boxes = np.tile(np.array((0, 128, 0, 128), np.int64), (shape[0], 1))
    plan = plan_confidence_capture(shape, 1)
    options = dict(layer_key='k', model_name='m', provenance={})
    expected = storage.write_blocks(tmp_path/'reference', shape, lambda z:scores[z], **options)
    metrics = {}
    with confidence_capture_resources(plan):
        reader = _MaskedNativeScoreReader(mask, scores, active, boxes)
        actual = storage.write_blocks(tmp_path/'actual', shape, reader, metrics=metrics, **options)
    assert actual == expected
    assert metrics['storage_batch_limit_bytes'] < shape[1]*shape[2]
    assert metrics['storage_payload_write_calls'] == metrics['block_count'] == shape[0]
    for name in ('index.bin', 'scores.u8.zlib'):
        assert (tmp_path/'actual'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()


def test_numeric_stage_limit_includes_buffered_payload_and_records(tmp_path):
    mask, scores, active, boxes = fixture()
    plan = plan_confidence_capture(scores.shape, 2, block_size=8)
    with confidence_capture_resources(plan):
        reader = _MaskedNativeScoreReader(mask, scores, active, boxes)
        with pytest.raises(storage.ConfidenceStageLimit):
            storage.write_blocks(tmp_path/'limited', scores.shape, reader,
                layer_key='k', model_name='m', provenance={}, block_size=8,
                max_numeric_bytes=1024)
    assert not (tmp_path/'limited/metadata.json').exists()
    assert not list((tmp_path/'limited').glob('*.partial'))


def test_writer_reports_executed_mixed_codecs_and_native_batch_counts(tmp_path, monkeypatch):
    shape = (2, 4, 5)
    mask, scores = np.zeros(shape, np.uint8), np.full(shape, 173, np.uint8)
    mask[:, 0, 0] = 1
    active = np.ones(2, bool)
    boxes = np.tile(np.array((0, 1, 0, 1), np.int64), (2, 1))
    plan = plan_confidence_capture(shape, 2)
    metrics = {}
    def encode(z, block):
        return [(0, 1, 0, 1, 1, zlib.compress(bytes([173]), level=3))], dict(
            scan_seconds=0., compression_seconds=0., copy_kernel_calls=1,
            copy_batch_peak_bytes=64 if z == 0 else 32,
            native_compression_batches=1 if z == 0 else 0,
            native_batch_seconds=.003 if z == 0 else 0.,
            codec_backend='native' if z == 0 else 'python',
            codec_version='1.3.1' if z == 0 else '1.3.2',
            codec_python_version='1.3.2', codec_fallback_reason=None if z == 0 else 'declined')
    with confidence_capture_resources(plan):
        reader = _MaskedNativeScoreReader(mask, scores, active, boxes)
        monkeypatch.setattr(reader, 'encode_frame', encode)
        storage.write_blocks(tmp_path/'actual', shape, reader, metrics=metrics,
            layer_key='k', model_name='m', provenance={})
    assert metrics['codec_backend'] == metrics['codec_version'] == 'mixed'
    assert metrics['codec_backend_values'] == ['native', 'python']
    assert metrics['capture_codec_frame_counts'] == {'native': 1, 'python': 1}
    assert metrics['codec_python_version'] == '1.3.2'
    assert metrics['native_compression_batches'] == 1
    assert metrics['native_batch_seconds'] == .003
    assert metrics['capture_copy_kernel_calls'] == 2
    assert metrics['capture_copy_batch_peak_bytes'] == 64
