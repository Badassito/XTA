"""OpenVINO emits exact independent native planes from one rendered forward."""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import workers
from XTA.geometry import AffineSpec, AugJob, ViewInfo
from XTA.inference import PredictConfig


class _ForwardQueue:
    def __init__(self, semantic=False):
        self.semantic, self.pending, self.calls = semantic, [], 0

    def set_callback(self, callback):
        self.callback = callback

    def start_async(self, inputs, *, userdata):
        self.calls += 1
        self.pending.append((next(iter(inputs.values())).copy(), userdata))

    def wait_all(self):
        for images, userdata in reversed(self.pending):
            prototype = images * np.float32(2) - np.float32(1)
            head = np.zeros((len(images), 6, 10), np.float32)
            head[:, :, 0] = [2, 4, 4, 8, .4, 1]
            head[:, :, 1] = [6, 4, 4, 8, .9, 1]
            outputs = [prototype] if self.semantic else [head, prototype]
            self.callback(SimpleNamespace(get_output_tensor=lambda i: SimpleNamespace(data=outputs[i])),
                          userdata)
            for array in outputs:
                array.fill(-12345)
        self.pending.clear()


def _runner(*, semantic=False, batch=2):
    runner = workers._OpenVinoCpuSegmenter.__new__(workers._OpenVinoCpuSegmenter)
    runner.request_count, runner.batch, runner.input_channels = 2, batch, 1
    runner.input_layout, runner.input_element_type, runner.input_name = 'NCHW', 'f32', 'images'
    runner.task = 'semantic' if semantic else 'segment'
    runner.output_ports = (object(),) if semantic else (object(), object())
    runner.expected_class_count, runner.resolved_precision = 1, 'fp32'
    runner.model_int8_quantized, runner.forward_replay = False, None
    runner.infer_queue = _ForwardQueue(semantic)
    return runner


class _Source:
    def __init__(self, batch=2):
        full = np.full((8, 8, 1), 255, np.uint8)
        half = full.copy()
        half[:4] = 0
        self.images = [full, half, np.zeros_like(full)]
        self.batch, self.rendered = batch, 0

    def __iter__(self):
        for start in range(0, len(self.images), self.batch):
            images = self.images[start:start + self.batch]
            images += [images[-1]] * (self.batch - len(images))
            self.rendered += 1
            yield None, images, None


class _Planes:
    floor_cent, cutoffs = 15, (40, 70, 90, 100)

    def __init__(self, fail_at=None):
        self.records, self.reason, self.finish_calls, self.abort_calls = {}, None, 0, 0
        self.fail_at, self.lock = fail_at, threading.Lock()
        self.work = dict(rendered_frames=0, render_batches=0, forward_calls=0, mask_decodes=0)

    def record_detector_work(self, **counts):
        with self.lock:
            for key, value in counts.items():
                self.work[key] += value

    def put(self, cent, index, mask, score, **counts):
        with self.lock:
            if self.reason is not None:
                return
            if cent == self.fail_at:
                raise OSError('sibling journal unavailable')
            assert (cent, index) not in self.records
            self.records[cent, index] = (mask.copy(), None if score is None else score.copy(), counts)

    def disable(self, reason):
        self.reason = str(reason)

    def finish(self):
        self.finish_calls += 1
        return {'status': 'complete' if self.reason is None else 'unsupported'}

    def abort(self, reason):
        self.abort_calls += 1
        self.disable(reason)


def _infer(runner, source, *, cent=15, writer=None, scores=True, min_conf=0.):
    mask = np.zeros((3, 7, 9), np.uint8)
    conf = np.zeros_like(mask) if scores else None
    stats = runner.infer_source_to_union(
        source, num_frames=3, out_size=8, conf_threshold=cent / 100.,
        view_union_mm=mask, view_confmap_mm=conf,
        M_out_to_native=np.asarray([[.8, .1, -1], [0, .8, 1]], np.float32),
        native_h=7, native_w=9, min_conf=min_conf, min_radius=0., native_fanout=writer,
    )
    return stats, mask, conf


@pytest.mark.parametrize('semantic', [False, True])
@pytest.mark.parametrize('scores', [False, True])
def test_one_source_forward_emits_every_scalar_native_plane(semantic, scores):
    source, runner, writer = _Source(), _runner(semantic=semantic), _Planes()
    writer.cutoffs = (90, 40, 100, 70)
    with mock.patch.object(workers, '_openvino_cpu_payloads_from_outputs',
                           wraps=workers._openvino_cpu_payloads_from_outputs) as decode:
        floor_stats, floor_mask, floor_scores = _infer(runner, source, writer=writer, scores=scores)
    assert decode.call_count == (0 if semantic else 2)
    assert source.rendered == 2 and runner.infer_queue.calls == 2
    assert writer.work['rendered_frames'] == 3
    assert writer.work['render_batches'] == 2
    assert writer.work['forward_calls'] == 2
    assert writer.work['mask_decodes'] == (15 if semantic else 9)
    assert len(writer.records) == 3 * (len(writer.cutoffs) + 1)
    for cent in (writer.floor_cent, *writer.cutoffs):
        scalar_stats, expected, expected_scores = _infer(
            _runner(semantic=semantic), _Source(), cent=cent, scores=scores,
        )
        records = [writer.records[cent, frame] for frame in range(3)]
        np.testing.assert_array_equal(np.stack([value[0] for value in records]), expected)
        if scores:
            np.testing.assert_array_equal(np.stack([value[1] for value in records]), expected_scores)
        else:
            assert all(value[1] is None for value in records)
        assert sum(value[2]['prediction_count'] for value in records) == scalar_stats['prediction_count']
        assert sum(value[2]['frames_with_predictions'] for value in records) == scalar_stats['frames_with_predictions']
        assert all(value[2]['proto_treated'] is False for value in records)
    np.testing.assert_array_equal(floor_mask, np.stack([writer.records[15, i][0] for i in range(3)]))
    assert floor_stats['prediction_count'] == sum(writer.records[15, i][2]['prediction_count'] for i in range(3))


@pytest.mark.parametrize('failed_cent', [15, 70])
def test_optional_native_write_failure_keeps_scalar_result(failed_cent):
    writer = _Planes(fail_at=failed_cent)
    actual_stats, actual, scores = _infer(_runner(), _Source(), writer=writer)
    expected_stats, expected, expected_scores = _infer(_runner(), _Source())
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(scores, expected_scores)
    assert actual_stats['prediction_count'] == expected_stats['prediction_count']
    assert 'sibling journal unavailable' in writer.reason


def test_nonzero_cleanup_disables_optional_native_decoding():
    writer = _Planes()
    _stats, actual, actual_scores = _infer(_runner(), _Source(), writer=writer, min_conf=.5)
    _stats, expected, expected_scores = _infer(_runner(), _Source(), min_conf=.5)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_scores, expected_scores)
    assert not writer.records and writer.reason


def test_sibling_below_inference_cutoff_declines_optimized_fanout():
    writer = _Planes()
    _, actual, _ = _infer(_runner(), _Source(), cent=50, writer=writer)
    _, expected, _ = _infer(_runner(), _Source(), cent=50)
    np.testing.assert_array_equal(actual, expected)
    assert not writer.records and 'exceeds a native sibling cutoff' in writer.reason


def _worker_task(tmp_path, mode):
    matrix = np.eye(2, 3, dtype=np.float32)
    view = ViewInfo('transverse', 5, 8, 8, 'clamp', full_t=5, full_h=8, full_w=8)
    job = AugJob('a0', 0., AffineSpec(
        'transverse', 0., 8, 8, 8, 8, 8, 8, 0., 0., matrix, matrix, matrix, matrix,
    ))
    source = tmp_path / 'source.dat'
    np.full((5, 8, 8), 255, np.uint8).tofile(source)
    task = dict(view=view, job=job, kind='fullframe', model_name='tiny', job_id='a0',
                out_size=8, channel_format='gray', slice_start=1, slice_count=3,
                processing_shape=(5, 8, 8), source_volume_path=str(source), source_shape=(5, 8, 8),
                result_mode=mode, union_num_slices=5, M_out_to_processing=matrix,
                result_mask_path=str(tmp_path / 'mask.dat'), result_conf_path=str(tmp_path / 'scores.dat'),
                streaming_cleanup_min_conf=0., streaming_cleanup_min_radius=0.)
    if mode == 'direct_union':
        initial = np.full((5, 8, 8), 7, np.uint8)
        initial[1:4] = 0
        for name in ('result_mask_path', 'result_conf_path'):
            initial.tofile(task[name])
    return task


@pytest.mark.parametrize('mode', ['file', 'direct_union'])
def test_worker_finishes_native_receipt_and_preserves_shared_window(tmp_path, mode):
    from XTA import native_fanout
    task, writer = _worker_task(tmp_path, mode), _Planes()
    cfg = PredictConfig(imgsz=8, conf=.15, device='cpu', quantize=None, batch=1, input_channels=1)
    with mock.patch.object(native_fanout, 'open_worker_fanout', return_value=writer):
        stats = workers.run_prediction_volume_in_openvino_worker(_runner(batch=1), cfg, task)
    assert stats['sweep_native_fanout']['status'] == 'complete'
    assert writer.finish_calls == 1 and writer.abort_calls == 0
    expected = np.stack([writer.records[15, index][0] for index in range(3)])
    actual = np.fromfile(task['result_mask_path'], np.uint8).reshape((-1, 8, 8))
    if mode == 'direct_union':
        np.testing.assert_array_equal(actual[1:4], expected)
        assert (actual[[0, 4]] == 7).all()
    else:
        np.testing.assert_array_equal(actual, expected)


def test_worker_aborts_once_on_true_source_failure(tmp_path):
    from XTA import native_fanout
    task, writer = _worker_task(tmp_path, 'file'), _Planes()
    cfg = PredictConfig(imgsz=8, conf=.15, device='cpu', quantize=None, batch=1, input_channels=1)
    with mock.patch.object(native_fanout, 'open_worker_fanout', return_value=writer), mock.patch.object(
        workers, '_worker_render_callable', side_effect=ValueError('source failed'),
    ):
        with pytest.raises(ValueError, match='source failed'):
            workers.run_prediction_volume_in_openvino_worker(_runner(batch=1), cfg, task)
    assert writer.abort_calls == 1 and writer.finish_calls == 0


def test_optional_finish_failure_keeps_floor_without_double_abort(tmp_path):
    from XTA import native_fanout

    class FailedFinish(_Planes):
        def finish(self):
            self.finish_calls += 1
            raise OSError('journal finish unavailable')

    task, writer = _worker_task(tmp_path, 'file'), FailedFinish()
    cfg = PredictConfig(imgsz=8, conf=.15, device='cpu', quantize=None, batch=1, input_channels=1)
    with mock.patch.object(native_fanout, 'open_worker_fanout', return_value=writer):
        stats = workers.run_prediction_volume_in_openvino_worker(_runner(batch=1), cfg, task)
    assert stats['sweep_native_fanout']['status'] == 'unsupported'
    assert writer.finish_calls == 1 and writer.abort_calls == 0
    expected = np.stack([writer.records[15, index][0] for index in range(3)])
    actual = np.fromfile(task['result_mask_path'], np.uint8).reshape((3, 8, 8))
    np.testing.assert_array_equal(actual, expected)


def test_declined_task_reports_canonical_unsupported_receipt(tmp_path):
    from XTA import native_fanout

    task = _worker_task(tmp_path, 'file')
    cfg = PredictConfig(imgsz=8, conf=.15, device='cpu', quantize=None, batch=1, input_channels=1)

    def decline(task, *_args, **_kwargs):
        task['_sweep_native_decline_reason'] = 'unqualified native window'

    with mock.patch.object(native_fanout, 'open_worker_fanout', side_effect=decline):
        stats = workers.run_prediction_volume_in_openvino_worker(_runner(batch=1), cfg, task)
    receipt = stats['sweep_native_fanout']
    assert receipt['status'] == 'unsupported'
    assert receipt['reason'] == 'unqualified native window'
    assert receipt['slice_count'] == 3


def test_native_integrity_failure_is_not_optional_fallback():
    from XTA.sweep_views import SweepViewIntegrityError

    class CorruptPlanes(_Planes):
        def put(self, *_args, **_kwargs):
            raise SweepViewIntegrityError('acknowledged native packet changed')

    with pytest.raises(RuntimeError, match='acknowledged native packet changed') as raised:
        _infer(_runner(), _Source(), writer=CorruptPlanes())
    assert isinstance(raised.value.__cause__, SweepViewIntegrityError)


def test_actual_native_bridge_crops_and_commits_every_cutoff_plane():
    from XTA.native_fanout import WorkerCutoffPlanes

    class Journal:
        cutoffs = (15, 40, 70, 90, 100)

        def __init__(self):
            self.records = {}

        def put(self, cent, frame, mask, scores, **metadata):
            self.records[cent, frame] = (mask.copy(), scores.copy(), metadata)

        def finish(self):
            return {'status': 'complete'}

        def abort(self, reason):
            raise AssertionError(reason)

    journal = Journal()
    writer = WorkerCutoffPlanes(journal, floor_cent=15, offset=7, shape=(7, 9),
                                has_scores=True, expected_frames=3)
    _infer(_runner(), _Source(), writer=writer)
    receipt = writer.finish()
    assert receipt['status'] == 'complete'
    assert receipt['produced_cutoff_frames'] == 15
    assert receipt['counters']['forward_calls'] == 2
    assert receipt['counters']['mask_decodes'] == 9
    for cent in journal.cutoffs:
        _stats, expected, scores = _infer(_runner(), _Source(), cent=cent)
        for frame in range(3):
            crop, values, metadata = journal.records[cent, frame + 7]
            y0, y1, x0, x1 = metadata['bbox']
            actual = np.zeros((7, 9), np.uint8)
            actual_scores = np.zeros_like(actual)
            actual[y0:y1, x0:x1] = crop
            actual_scores[y0:y1, x0:x1] = values
            np.testing.assert_array_equal(actual, expected[frame])
            np.testing.assert_array_equal(actual_scores, scores[frame])


@pytest.mark.parametrize('background', [False, True])
def test_concurrent_openvino_aliases_use_the_correct_durable_frame(tmp_path, background):
    from XTA.native_fanout import WorkerCutoffPlanes
    from XTA.sweep_views import NativeTaskFanoutWriter, hydrate, seal_view_family

    shape = (3, 7, 9)
    cuts = (15, 40, 70, 80, 90, 100)
    journal = NativeTaskFanoutWriter(tmp_path, 'model', 'view', shape, cuts,
                                    frame_start=0, expected_frames=3, backend='openvino',
                                    scalar_owned_cutoff=15)
    if background:
        journal.start_background_encoding()
    barrier = threading.Barrier(2, timeout=10)

    class InterleavedPlanes(WorkerCutoffPlanes):
        def put(self, cutoff, index, *args, **kwargs):
            super().put(cutoff, index, *args, **kwargs)
            if cutoff == 70 and index in (0, 2):
                barrier.wait()

    source = _Source()
    source.images[2] = source.images[1][::-1].copy()
    writer = InterleavedPlanes(journal, floor_cent=15, offset=0, shape=shape[1:],
                               has_scores=True, expected_frames=shape[0])
    _infer(_runner(), source, writer=writer)
    assert not writer._last_crops
    receipt = writer.finish()
    assert receipt['status'] == 'complete'
    assert receipt['counters']['native_packet_aliases'] == 6
    assert receipt['counters']['bulk_empty_frames'] == 3
    decision = seal_view_family('model', 'view', shape, cuts, receipts=[receipt], root=tmp_path)
    for cent in cuts[1:]:
        expected_stats, expected, expected_scores = _infer(_runner(), source, cent=cent)
        actual = np.zeros(shape, dtype=np.uint8)
        scores = np.zeros_like(actual)
        hydrate('model', 'view', cent, actual, scores, root=tmp_path)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scores, expected_scores)
        assert decision['prediction_counts'][cent] == expected_stats['prediction_count']
        assert decision['frames_with_predictions'][cent] == expected_stats['frames_with_predictions']


def test_real_openvino_cpu_fanout_uses_one_forward_per_rendered_batch(tmp_path, monkeypatch):
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops

    monkeypatch.delenv('XTA_YOLO_SWEEP_CACHE_DIR', raising=False)
    inputs = ops.parameter([1, 1, 8, 8], np.float32, name='images')
    head = np.zeros((1, 6, 10), np.float32)
    head[0, :, 0] = [2, 4, 4, 8, .4, 1]
    head[0, :, 1] = [6, 4, 4, 8, .9, 1]
    model = ov.Model([ops.constant(head), ops.subtract(inputs, ops.constant(np.float32(.5)))],
                     [inputs], 'native_fanout_cpu')
    path = tmp_path / 'model.xml'
    ov.save_model(model, path)
    runner = workers._OpenVinoCpuSegmenter(
        str(path), imgsz=8, batch=1, input_channels=1, requested_precision='fp32',
        inference_threads=1, physical_cores=1, streams=1, infer_requests=2,
    )
    source, writer = _Source(batch=1), _Planes()
    _infer(runner, source, writer=writer)
    assert source.rendered == writer.work['render_batches'] == 3
    assert writer.work['forward_calls'] == 3
    assert len(writer.records) == 15
    for cent in (writer.floor_cent, *writer.cutoffs):
        expected_stats, expected, scores = _infer(runner, _Source(batch=1), cent=cent)
        records = [writer.records[cent, index] for index in range(3)]
        np.testing.assert_array_equal(np.stack([value[0] for value in records]), expected)
        np.testing.assert_array_equal(np.stack([value[1] for value in records]), scores)
        assert sum(value[2]['prediction_count'] for value in records) == expected_stats['prediction_count']


def test_native_fanout_bypasses_seeded_raw_cache_and_scalar_fallback_still_reuses_it(tmp_path, monkeypatch):
    from XTA.forward_replay import replay_for_model

    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('XTA_YOLO_SWEEP_CACHE_GIB', '.001')
    model = tmp_path / 'fixture.xml'
    model.write_text('fixture', encoding='utf8')
    cache = replay_for_model([model], backend='openvino_cpu', settings={'fanout_fixture': 1})
    assert cache is not None
    first = _runner()
    first.forward_replay = cache
    _infer(first, _Source())  # Seed valid raw packets through the ordinary scalar route.
    second, writer = _runner(), _Planes()
    cache_api = mock.Mock(wraps=cache)
    second.forward_replay = cache_api
    stats, _mask, _scores = _infer(second, _Source(), writer=writer)
    cache_api.input_key.assert_not_called()
    cache_api.load.assert_not_called()
    cache_api.store.assert_not_called()
    assert stats['openvino_forward_replay_hits'] == 0
    assert stats['openvino_forward_replay_misses'] == 0
    assert second.infer_queue.calls == writer.work['forward_calls'] == 2
    assert writer.work['mask_decodes'] == 9
    assert writer.work['rendered_frames'] == 3
    assert writer.work['render_batches'] == 2

    fallback = _runner()
    fallback.forward_replay = cache_api
    scalar_stats, _mask, _scores = _infer(fallback, _Source())
    assert scalar_stats['openvino_forward_replay_hits'] == 2
    assert fallback.infer_queue.calls == 0
    assert cache_api.load.call_count == 2
    cache_api.store.assert_not_called()
