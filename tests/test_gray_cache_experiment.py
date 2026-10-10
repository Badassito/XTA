"""Small correctness checks for the standalone lossless gray8 experiment."""
import json

import numpy as np
import pytest

from tools import benchmark_gray_cache as experiment


@pytest.mark.parametrize('mode', experiment.MODES, ids=lambda mode: mode.name)
def test_codecs_preserve_edge_chunks_and_temporal_uint8_overflow(mode):
    values = (np.arange(9*257*259, dtype=np.uint32).reshape(9, 257, 259) * 127).astype(np.uint8)
    packets, metrics = experiment.encode_sample(values, mode)
    assert metrics['exact_roundtrip']
    assert metrics['estimated_total_bytes'] == metrics['payload_bytes'] + 32*len(packets)
    restored = np.empty_like(values)
    for packet in packets.values():
        target = tuple(slice(start, start+size) for start, size in zip(packet.start, packet.shape))
        restored[target] = experiment._decode(packet, mode)
    np.testing.assert_array_equal(restored, values)
    if mode.depth == 8:
        assert packets[(8, 0, 0)].shape[0] == 1
    if mode.tile:
        assert packets[(0, 256, 256)].shape[1:] == (1, 3)


def test_temporal_prediction_wraps_both_increasing_and_decreasing_values():
    mode = next(mode for mode in experiment.MODES if mode.name == '3d_zstd1_delta')
    values = np.array([[254, 0], [255, 255], [0, 254], [1, 1]], np.uint8)[:, None, :]
    packet = experiment.Packet((0, 0, 0), values.shape, experiment._encode(values, mode))
    np.testing.assert_array_equal(experiment._decode(packet, mode), values)


def test_unaligned_crops_and_edge_clipping_cross_multiple_chunks():
    mode = experiment.Mode('test', 'zstd', 1, 3, 4, True)
    values = np.arange(7*11*13, dtype=np.uint16).reshape(7, 11, 13).astype(np.uint8)
    packets, _ = experiment.encode_sample(values, mode)
    reader = experiment.CropReader(packets, mode, values.shape, cache_bytes=256)
    for request in (((1, 6), (2, 10), (3, 12)), ((-4, 3), (-2, 6), (10, 19)),
                    ((5, 12), (8, 18), (-5, 4)), ((10, 12), (0, 2), (0, 2))):
        actual, bounds = reader.read(request)
        np.testing.assert_array_equal(actual, values[tuple(slice(first, stop) for first, stop in bounds)])
    assert reader.stats['cache_evictions'] > 0
    assert reader.stats['peak_cache_bytes'] <= 256


def test_lru_eviction_reloads_the_old_chunk_and_a_repeat_hits():
    mode = experiment.Mode('test', 'deflate', 1, 3, 4)
    values = np.arange(3*4*8, dtype=np.uint8).reshape(3, 4, 8)
    packets, _ = experiment.encode_sample(values, mode)
    reader = experiment.CropReader(packets, mode, values.shape, cache_bytes=48)
    left, right = ((0, 3), (0, 4), (0, 4)), ((0, 3), (0, 4), (4, 8))
    for request in (left, right, left, left):
        actual, bounds = reader.read(request)
        np.testing.assert_array_equal(actual, values[tuple(slice(first, stop) for first, stop in bounds)])
    assert reader.stats['cache_misses'] == 3
    assert reader.stats['cache_hits'] == 1
    assert reader.stats['cache_evictions'] == 2
    assert reader.stats['peak_cache_bytes'] == 48
    assert reader.stats['decoded_bytes'] == 144
    cold = experiment.CropReader(packets, mode, values.shape, cache_bytes=0)
    for _ in range(2):
        cold.read(left)
    assert cold.stats['cache_hits'] == 0 and cold.stats['cache_bytes'] == 0
    assert cold.stats['decoded_bytes'] == 96


def test_deterministic_crop_workload_retains_logical_access_metrics():
    mode = experiment.Mode('test', 'lz4', None, 3, 4)
    values = np.arange(7*11*13, dtype=np.uint16).reshape(7, 11, 13).astype(np.uint8)
    packets, _ = experiment.encode_sample(values, mode)
    cold = experiment.measure_crops(values, packets, mode, cache_bytes=0)
    warm = experiment.measure_crops(values, packets, mode, cache_bytes=experiment.CACHE_BYTES)
    assert cold['requested_crop_bytes'] == warm['requested_crop_bytes'] > 0
    assert cold['encoded_bytes_fetched'] > warm['encoded_bytes_fetched'] > 0
    assert cold['decoded_bytes'] > warm['decoded_bytes'] > 0
    assert cold['requests'] == 6 and warm['cache_hits'] > 0


def test_manifest_outputs_and_optional_thread_scaling_use_only_reports(tmp_path):
    values = np.arange(9*12*14, dtype=np.uint16).reshape(9, 12, 14).astype(np.uint8)
    np.save(tmp_path/'sample.npy', values)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'name':'small', 'path':'sample.npy', 'family':'orthogonal'}]))
    result = experiment.run(manifest, tmp_path/'reports', workers=(1, 4), warmup_rounds=1)
    sample = result['samples'][0]
    assert sample['raw_bytes'] == values.nbytes
    assert set(sample['modes']) == {mode.name for mode in experiment.MODES}
    scaling = sample['modes']['3d_zstd1_delta']['thread_scaling']
    assert [row['workers'] for row in scaling] == [1, 4]
    assert all(row['max_inflight_chunks'] == row['workers'] for row in scaling)
    assert {path.name for path in (tmp_path/'reports').iterdir()} == {'results.json', 'summary.txt'}
    assert json.loads((tmp_path/'reports'/'results.json').read_text()) == result
    assert 'no physical storage I/O was measured' in (tmp_path/'reports'/'summary.txt').read_text()


def test_invalid_samples_and_resource_bounds_are_rejected_before_encoding(tmp_path):
    mode = experiment.MODES[0]
    for values in (np.zeros((2, 3), np.uint8), np.zeros((2, 3, 4), np.float32),
                   np.zeros((0, 3, 4), np.uint8),
                   np.broadcast_to(np.zeros((1, 1, 1), np.uint8), (256, 1024, 1024))):
        with pytest.raises(ValueError):
            experiment.encode_sample(values, mode)
    with pytest.raises(ValueError, match='positive'):
        experiment.encode_sample(np.zeros((1, 2, 3), np.uint8), mode, repeats=0)
    with pytest.raises(ValueError, match='Cache'):
        experiment.CropReader({}, mode, (1, 2, 3), cache_bytes=-1)
    with pytest.raises(ValueError, match='Crop bounds'):
        experiment.CropReader({}, mode, (1, 2, 3)).read(((0, 1), (2, 2), (0, 1)))
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='nonempty list'):
        experiment.run(manifest, tmp_path/'reports')
    assert not (tmp_path/'reports').exists()


def test_manifest_hash_refuses_changed_sample_before_benchmark(tmp_path):
    np.save(tmp_path/'sample.npy', np.zeros((1, 2, 3), np.uint8))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'path': 'sample.npy', 'sha256': '0'*64}]))
    with pytest.raises(ValueError, match='differ from the preparation manifest'):
        experiment.run(manifest, tmp_path/'reports')
    assert not (tmp_path/'reports').exists()


def test_optional_heat_soak_roundtrips_a_bounded_number_of_chunks():
    values = np.arange(2*12*14, dtype=np.uint16).reshape(2, 12, 14).astype(np.uint8)
    assert experiment.heat_soak(values, 0) is None
    with pytest.raises(ValueError, match='Heat soak'):
        experiment.heat_soak(values, -1)
    result = experiment.heat_soak(values, .02)
    assert result['threads'] == 1 and result['iterations'] > 0
    assert result['elapsed_seconds'] >= .02
