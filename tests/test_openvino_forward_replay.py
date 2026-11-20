"""Raw OpenVINO replay preserves cutoff-independent outputs and ownership."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from XTA import workers


class _MemoryReplay:
    def __init__(self):
        self.records = {}
        self.inputs = []
        self.stores = 0

    def input_key(self, value, *, source, batch_index):
        self.inputs.append((source, batch_index))
        digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
        digest.update(memoryview(value).cast('B'))
        return digest.hexdigest()

    def load(self, key):
        arrays = self.records.get(key)
        return None if arrays is None else tuple(value.copy() for value in arrays)

    def store(self, key, arrays):
        self.records[key] = tuple(value.copy() for value in arrays)
        self.stores += 1


class _InferQueue:
    """Delay callbacks, then immediately reuse every output allocation."""
    def __init__(self, *, semantic=False):
        self.semantic = semantic
        self.pending = []
        self.calls = 0

    def set_callback(self, callback):
        self.callback = callback

    def start_async(self, inputs, *, userdata):
        self.calls += 1
        self.pending.append((next(iter(inputs.values())).shape[0], userdata))

    def wait_all(self):
        for batch, userdata in reversed(self.pending):
            if self.semantic:
                outputs = [np.full((batch, 1, 2, 2), .1, np.float32)]
            else:
                head = np.zeros((batch, 6, 10), np.float32)
                head[:, :, 0] = [4, 4, 8, 8, .4, 1]
                outputs = [head, np.ones((batch, 1, 2, 2), np.float32)]
            request = SimpleNamespace(
                get_output_tensor=lambda index: SimpleNamespace(data=outputs[index]),
            )
            self.callback(request, userdata)
            for output in outputs:
                output.fill(-12345)
        self.pending.clear()


def _runner(cache, *, semantic=False):
    runner = workers._OpenVinoCpuSegmenter.__new__(workers._OpenVinoCpuSegmenter)
    runner.request_count = 2
    runner.batch = 1
    runner.input_channels = 1
    runner.input_layout = 'NCHW'
    runner.input_element_type = 'f32'
    runner.input_name = 'images'
    runner.task = 'semantic' if semantic else 'segment'
    runner.infer_queue = _InferQueue(semantic=semantic)
    runner.output_ports = (object(),) if semantic else (object(), object())
    runner.expected_class_count = 1
    runner.resolved_precision = 'fp32'
    runner.model_int8_quantized = False
    runner.forward_replay = cache
    return runner


def _source(value=20, count=1):
    return [(None, [np.full((8, 8, 1), value, np.uint8)], None)
            for _ in range(count)]


def _run(runner, *, source=None, conf=.15, min_conf=0.):
    source = _source() if source is None else source
    mask = np.zeros((len(source), 8, 8), np.uint8)
    scores = np.zeros_like(mask)
    stats = runner.infer_source_to_union(
        source, num_frames=len(source), out_size=8,
        conf_threshold=conf, min_conf=min_conf, min_radius=0.,
        view_union_mm=mask, view_confmap_mm=scores,
        M_out_to_native=np.asarray([[1, 0, 0], [0, 1, 0]], np.float32),
        native_h=8, native_w=8,
    )
    return stats, mask, scores


def test_persistent_replay_survives_runner_recreation_and_preserves_cutoffs(tmp_path, monkeypatch):
    from XTA.forward_replay import replay_cache_summary

    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_GIB', '.001')
    xml = tmp_path / 'model.xml'
    xml.write_text('raw-output model identity fixture', encoding='utf8')
    xml.with_suffix('.bin').write_bytes(b'weights')

    def persistent_runner():
        runner = _runner(None)
        runner.model_xml = xml
        runner.ov = SimpleNamespace(get_version=lambda: '2026.fixture')
        runner.imgsz = 8
        runner.compile_config = {'INFERENCE_NUM_THREADS': 2}
        runner.cpu_flags = frozenset()
        runner.capabilities = {'FP32'}
        runner.forward_replay = runner._create_forward_replay()
        assert runner.forward_replay is not None
        return runner

    first = persistent_runner()
    fresh, mask, _scores = _run(first, min_conf=.3)
    assert mask.sum() == 64
    assert fresh['openvino_forward_replay_misses'] == 1
    assert first.infer_queue.calls == 1
    second = persistent_runner()
    hit, mask, _scores = _run(second, min_conf=.6)
    assert mask.sum() == 0
    assert hit['openvino_forward_replay_hits'] == 1
    assert second.infer_queue.calls == 0
    assert hit['openvino_output_copy_bytes'] == 0
    summary = replay_cache_summary(tmp_path / 'cache')
    assert summary['entries'] == 1
    assert summary['by_backend']['openvino_cpu']['hit'] == 1


def test_real_cpu_openvino_raw_outputs_replay_without_another_output_copy(tmp_path, monkeypatch):
    import pytest
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops

    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_GIB', '.001')
    inputs = ops.parameter([1, 1, 8, 8], np.float32, name='images')
    head = np.zeros((1, 6, 10), np.float32)
    head[0, :, 0] = [4, 4, 8, 8, .4, 1]
    model = ov.Model([ops.constant(head), ops.subtract(inputs, ops.constant(np.float32(.5)))],
                     [inputs], 'raw_replay_cpu')
    path = tmp_path / 'model.xml'
    ov.save_model(model, path)

    def real_runner():
        return workers._OpenVinoCpuSegmenter(
            str(path), imgsz=8, batch=1, input_channels=1, requested_precision='fp32',
            inference_threads=1, physical_cores=1, streams=1, infer_requests=1,
        )

    fresh, mask, _scores = _run(real_runner(), source=_source(255), min_conf=.3)
    assert mask.sum() == 64
    assert fresh['openvino_forward_replay_misses'] == 1
    assert fresh['openvino_output_copy_bytes'] > 0
    hit, mask, _scores = _run(real_runner(), source=_source(255), min_conf=.6)
    assert mask.sum() == 0
    assert hit['openvino_forward_replay_hits'] == 1
    assert hit['openvino_output_copy_bytes'] == 0


def test_annotated_source_uses_metadata_key_but_cpu_augmentation_hashes_pixels(tmp_path, monkeypatch):
    from XTA.forward_replay import bind_worker_replay_source, replay_for_model
    from XTA.geometry import AffineSpec, AugJob, ViewInfo
    from XTA.tta_augmentation_cpu_runtime import CpuPolicyBatchSource

    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_DIR', str(tmp_path / 'cache'))
    model = tmp_path / 'model.xml'
    model.write_text('fixture', encoding='utf8')
    cache = replay_for_model([model], backend='openvino_cpu', settings={'precision': 'fp32'})
    assert cache is not None
    source = SimpleNamespace()
    affine = np.eye(2, 3, dtype=np.float32)
    view = ViewInfo('transverse', 1, 8, 8, 'clamp', full_t=1, full_h=8, full_w=8)
    job = AugJob('a0', 0., AffineSpec(
        'transverse', 0., 8, 8, 8, 8, 8, 8, 0., 0., affine, affine, affine, affine,
    ))
    task = {'sweep_source_stamp': ['source', 123], 'kind': 'fullframe', 'job_id': 1,
            'view': view, 'job': job}
    bind_worker_replay_source(source, task, 'openvino_cpu')
    pixels = np.zeros((1, 1, 8, 8), np.float32)
    logical = cache.input_key(pixels, source=source, batch_index=0)
    changed = pixels + np.float32(1)
    assert cache.input_key(changed, source=source, batch_index=0) == logical
    augmented = CpuPolicyBatchSource([], [], None, [], None)
    assert cache.input_key(changed, source=augmented, batch_index=0) != cache.input_key(
        pixels, source=augmented, batch_index=0,
    )


class OpenVinoForwardReplayTests(unittest.TestCase):
    def test_raw_segment_outputs_replay_at_changed_conf_and_min_conf(self):
        cache = _MemoryReplay()
        runner = _runner(cache)
        fresh, mask, _scores = _run(runner, min_conf=.3)
        self.assertEqual(int(mask.sum()), 64)
        self.assertEqual(runner.infer_queue.calls, 1)
        self.assertEqual(fresh['openvino_output_copy_count'], 2)
        self.assertEqual(fresh['openvino_output_copy_bytes'], 256)
        raw = next(iter(cache.records.values()))
        self.assertEqual(raw[0].shape, (1, 6, 10))
        self.assertEqual(float(raw[0][0, 4, 0]), float(np.float32(.4)))
        for conf, minimum in ((.5, 0.), (.15, .6), (.15, .3)):
            with self.subTest(conf=conf, min_conf=minimum):
                hit, actual, actual_scores = _run(runner, conf=conf, min_conf=minimum)
                _stats, expected, expected_scores = _run(
                    _runner(None), conf=conf, min_conf=minimum,
                )
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual_scores, expected_scores)
                self.assertEqual(hit['openvino_forward_replay_hits'], 1)
                self.assertEqual(hit['openvino_output_copy_bytes'], 0)
        self.assertEqual(runner.infer_queue.calls, 1)
        self.assertEqual(cache.stores, 1)

    def test_semantic_logits_are_retained_before_probability_thresholding(self):
        cache = _MemoryReplay()
        runner = _runner(cache, semantic=True)
        _fresh, mask, _scores = _run(runner, conf=.2)
        self.assertEqual(int(mask.sum()), 64)
        raw = next(iter(cache.records.values()))
        np.testing.assert_array_equal(raw[0], np.full((1, 1, 2, 2), .1, np.float32))
        hit, mask, _scores = _run(runner, conf=.8)
        self.assertEqual(int(mask.sum()), 0)
        self.assertEqual(hit['openvino_forward_replay_hits'], 1)
        self.assertEqual(runner.infer_queue.calls, 1)

    def test_repeated_submission_index_uses_prepared_pixels(self):
        cache = _MemoryReplay()
        runner = _runner(cache)
        _run(runner, source=_source(20))
        _run(runner, source=_source(30))
        stats, _mask, _scores = _run(runner, source=_source(20))
        self.assertEqual(len(cache.records), 2)
        self.assertEqual(runner.infer_queue.calls, 2)
        self.assertEqual(stats['openvino_forward_replay_hits'], 1)
        self.assertEqual([index for _source_value, index in cache.inputs], [0, 0, 0])

    def test_store_precedes_adaptation_and_load_arrays_are_independent(self):
        cache = _MemoryReplay()
        runner = _runner(cache)
        original = workers._openvino_cpu_payloads_from_outputs

        def mutate_after_adaptation(outputs, **kwargs):
            payloads = original(outputs, **kwargs)
            outputs[0].fill(0)
            return payloads

        with mock.patch.object(workers, '_openvino_cpu_payloads_from_outputs',
                               side_effect=mutate_after_adaptation):
            _run(runner)
            _run(runner)
        raw = next(iter(cache.records.values()))
        self.assertEqual(float(raw[0][0, 4, 0]), float(np.float32(.4)))
        self.assertEqual(runner.infer_queue.calls, 1)

    def test_out_of_order_callbacks_preserve_destinations_and_copy_ownership(self):
        runner = _runner(_MemoryReplay())
        stats, mask, _scores = _run(runner, source=_source(20) + _source(30))
        np.testing.assert_array_equal(mask, np.ones((2, 8, 8), np.uint8))
        self.assertEqual(stats['prediction_count'], 2)
        self.assertEqual(stats['openvino_output_copy_count'], 4)

    def test_model_identity_covers_companion_files_and_resolved_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / 'model.xml'
            for path in (xml, xml.with_suffix('.bin'), root / 'metadata.yaml'):
                path.write_text('fixture', encoding='utf8')
            runner = _runner(None)
            runner.model_xml = xml
            runner.ov = SimpleNamespace(get_version=lambda: '2026.fixture')
            runner.imgsz = 8
            runner.compile_config = {'INFERENCE_NUM_THREADS': 2,
                                     'INFERENCE_PRECISION_HINT': 'f32'}
            runner.cpu_flags = frozenset({'fixture_flag'})
            runner.capabilities = {'FP32'}
            factory = mock.Mock(return_value='cache')
            with mock.patch('XTA.forward_replay.replay_for_model', factory):
                self.assertEqual(runner._create_forward_replay(), 'cache')
                xml.with_suffix('.bin').unlink()
                self.assertIsNone(runner._create_forward_replay())
            self.assertEqual(factory.call_count, 1)
            files = factory.call_args.args[0]
            self.assertEqual({path.name for path in files},
                             {'model.xml', 'model.bin', 'metadata.yaml'})
            settings = factory.call_args.kwargs['settings']
            self.assertEqual(settings['openvino_version'], '2026.fixture')
            self.assertEqual(settings['input_layout'], 'NCHW')
            self.assertFalse({'conf', 'min_conf', 'pid'} & set(settings))


if __name__ == '__main__':
    unittest.main()
