"""Background native-journal encoding and batched sibling reads stay byte-exact."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import sweep_views as views
from XTA.native_fanout import WorkerCutoffPlanes


@pytest.fixture
def family(tmp_path, monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_WORK_DIR', str(tmp_path))
    monkeypatch.setenv('XTA_YOLO_SWEEP_ID', 'async-family')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce')
    monkeypatch.setenv('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS', '50,60,70,80')
    return tmp_path / 'native_fanout'


def _writer(shape, *, floor=None):
    task = {'view': SimpleNamespace(name='transverse__tta_a0'), 'model_name': 'detector',
            'slice_start': 0, 'sweep_fanout_identity': {'test': True}}
    return views.begin_task_fanout(task, SimpleNamespace(conf=.5), backend='cuda',
                                   expected_frames=shape[0], native_shape=shape,
                                   scalar_owned_cutoff=floor)


def _workload(seed, shape):
    rng = np.random.default_rng(seed)
    ops = []
    for z in range(shape[0]):
        previous = None
        for cent in (60, 70, 80):
            kind = rng.choice(['crop', 'alias', 'empty', 'plane'], p=[.5, .2, .1, .2])
            if kind == 'alias' and previous is not None:
                ops.append(('alias', cent, z, previous))
            elif kind == 'empty':
                ops.append(('empty', cent, z))
            else:
                y0 = int(rng.integers(0, shape[1] - 1)); y1 = int(rng.integers(y0 + 1, shape[1] + 1))
                x0 = int(rng.integers(0, shape[2] - 1)); x1 = int(rng.integers(x0 + 1, shape[2] + 1))
                mask = (rng.random((y1 - y0, x1 - x0)) < .4).astype(np.uint8)
                score = rng.integers(0, 256, mask.shape, dtype=np.uint8)
                if kind == 'plane':
                    full = np.zeros(shape[1:], np.uint8); conf = np.zeros_like(full)
                    full[y0:y1, x0:x1] = mask; conf[y0:y1, x0:x1] = score
                    ops.append(('plane', cent, z, full, conf, int(mask.sum() > 0)))
                else:
                    ops.append(('crop', cent, z, mask, score, (y0, y1, x0, x1), int(rng.integers(0, 5))))
            previous = cent
    return ops


def _replay(writer, ops, shape):
    for op in ops:
        if op[0] == 'alias':
            writer.put_alias(op[1], op[2], source_cutoff=op[3], prediction_count=3, frames_with_predictions=1)
        elif op[0] == 'empty':
            writer.put_empty_range(op[1], op[2], op[2] + 1, score_lane=True)
        elif op[0] == 'plane':
            writer.put(op[1], op[2], op[3].copy(), op[4].copy(), prediction_count=op[5])
        else:
            writer.put(op[1], op[2], op[3].copy(), op[4].copy(), bbox=op[5], prediction_count=op[6])
    writer.put_scalar_owned_range(50, 0, shape[0], prediction_counts=np.arange(shape[0]),
                                  frames_with_predictions=np.ones(shape[0], np.int64))
    return writer.finish()


def _journal(receipt):
    path = Path(receipt['path'])
    meta = json.loads((path / 'metadata.json').read_bytes())
    meta.pop('range_id')
    return (path / 'payload.bin').read_bytes(), (path / 'index.bin').read_bytes(), meta


@pytest.mark.parametrize('verify', ['0', '1'])
@pytest.mark.parametrize('queue_bytes', [None, 1])
def test_background_encoding_journal_is_byte_identical(family, monkeypatch, verify, queue_bytes):
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_VERIFY', verify)
    monkeypatch.setattr(views, '_CHUNK_BYTES', 96)  # several strips per crop
    shape = (6, 23, 31)
    ops = _workload(7, shape)
    synchronous = _journal(_replay(_writer(shape, floor=50), ops, shape))
    writer = _writer(shape, floor=50)
    writer.start_background_encoding(queue_bytes)
    assert writer.background_encoding
    background = _journal(_replay(writer, ops, shape))
    assert background[0] == synchronous[0] and background[0]
    assert background[1] == synchronous[1]
    assert background[2] == synchronous[2]


def test_background_encode_failure_surfaces_on_next_call_and_abort_discards(family):
    shape = (3, 4, 5)
    writer = _writer(shape)
    writer.start_background_encoding()
    bad = np.full((1, 1), 2, np.uint8)
    writer.put(60, 0, bad, None, bbox=(0, 1, 0, 1))
    with pytest.raises(ValueError, match='only 0/1'):
        writer.finish()
    with pytest.raises(ValueError, match='only 0/1'):
        writer.put(70, 0, np.ones((1, 1), np.uint8), None, bbox=(0, 1, 0, 1))
    receipt = writer.abort('encode failed')
    assert receipt['status'] == 'fallback'
    assert not (Path(receipt['path']) / 'payload.bin').exists()
    assert views.seal_view_family('detector', 'transverse__tta_a0', shape, [50, 60, 70, 80],
                                  receipts=[receipt], identity={'test': True})['status'] == 'fallback'


def test_background_capacity_refusal_becomes_worker_fallback(family, monkeypatch):
    shape = (1, 4, 5)
    writer = _writer(shape, floor=50)
    writer.start_background_encoding()
    def refuse(_):
        raise views.SweepViewCapacityError('test budget pressure')
    monkeypatch.setattr(writer, '_fund', refuse)
    sink = WorkerCutoffPlanes(writer, floor_cent=50, offset=0, shape=(4, 5), has_scores=False,
                              expected_frames=1)
    sink.consume_empty(0, 50, 0, frames=0)
    sink.put(60, 0, np.ones((4, 5), np.uint8), None, prediction_count=1, frames_with_predictions=1)
    sink.put(70, 0, np.ones((4, 5), np.uint8), None, prediction_count=1, frames_with_predictions=1)
    sink.consume_empty(0, 80, 0)
    receipt = sink.finish()
    assert receipt['status'] == 'fallback' and not receipt['supported']
    assert 'budget pressure' in receipt['reason']


def test_worker_plane_put_never_retains_caller_buffer(family, monkeypatch):
    import threading
    shape = (1, 4, 5)
    writer = _writer(shape, floor=50)
    writer.start_background_encoding()
    release = threading.Event()
    run = writer._run_encoder_job
    monkeypatch.setattr(writer, '_run_encoder_job', lambda job: (release.wait(10), run(job)))
    sink = WorkerCutoffPlanes(writer, floor_cent=50, offset=0, shape=(4, 5), has_scores=True,
                              expected_frames=1)
    # Full-width support makes the crop a contiguous view of the caller plane.
    plane = np.zeros((4, 5), np.uint8); plane[1:3] = 1
    conf = plane * 77
    expected, expected_conf = plane.copy(), conf.copy()
    sink.consume_empty(0, 50, 1, frames=1)
    for cent in (60, 70, 80):
        plane[:], conf[:] = expected, expected_conf
        sink.put(cent, 0, plane, conf, prediction_count=1, frames_with_predictions=1)
        plane[:], conf[:] = 1, 255  # the caller reuses its buffer before the encoder runs
    release.set()
    receipt = sink.finish()
    assert receipt['status'] == 'complete'
    views.seal_view_family('detector', 'transverse__tta_a0', shape, [50, 60, 70, 80],
                           receipts=[receipt], identity={'test': True})
    for cent in (60, 70, 80):
        out = np.zeros(shape, np.uint8); scores = np.zeros_like(out)
        views.hydrate('detector', 'transverse__tta_a0', cent, out, scores, identity={'test': True})
        assert np.array_equal(out[0], expected) and np.array_equal(scores[0], expected_conf)


class _Recorder:
    cutoffs = (20, 40, 60, 80, 90)

    def __init__(self):
        self.calls = []

    def put(self, cutoff, index, mask, score, **metadata):
        self.calls.append(('put', cutoff, index, mask.copy(), None if score is None else score.copy(),
                           tuple(sorted(metadata.items()))))

    def put_empty_range(self, cutoff, start, stop, **metadata):
        self.calls.append(('empty', cutoff, start, stop, repr(sorted(metadata.items()))))

    def finish(self):
        return {'status': 'complete'}

    def abort(self, reason):
        raise AssertionError(reason)


def _calls_equal(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
                assert x.dtype == y.dtype and np.array_equal(x, y)
            else:
                assert x == y


_CUDA_QUALIFICATION = pytest.mark.skipif(os.environ.get('XTA_TEST_NATIVE_FANOUT_CUDA') != '1',
                                         reason='explicit coordinated native-fanout CUDA qualification only')


_GEOMETRIES = {  # out_size, decoded plane side, native shape, output-to-native matrix
    'affine': (12, 12, (10, 11), [[1, 0, 1], [0, 1, -1]]),
    'identity': (12, 12, (12, 12), [[1, 0, 0], [0, 1, 0]]),
    'resized': (12, 9, (10, 11), [[.9, .1, 1], [-.1, .95, 0]]),
}


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=_CUDA_QUALIFICATION)])
@pytest.mark.parametrize('device_counts', [True, False])
@pytest.mark.parametrize('scores', [True, False])
@pytest.mark.parametrize('geometry', sorted(_GEOMETRIES))
@pytest.mark.parametrize('group_planes', [None, 3])
def test_batched_sibling_results_issue_the_sequential_writes(device_counts, scores, device, geometry,
                                                             group_planes, monkeypatch):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from XTA import inference, native_fanout
    out_size, side, native, matrix = _GEOMETRIES[geometry]
    matrix = np.array(matrix, np.float32)
    if group_planes is not None:  # groups of one and two payloads
        monkeypatch.setattr(native_fanout, '_SIBLING_WARP_GROUP_BYTES', group_planes * 4 * side * side)
    rng = np.random.default_rng(3)

    def frame_payloads(frame):
        payloads = []
        for k, cent in enumerate((40, 60, 80, 90)):
            if (frame + k) % 4 == 3:
                payloads.append((cent, inference.GpuFlattenedRetinaPayload(None, None, instance_count=0)))
                continue
            union = torch.zeros((side, side))
            if (frame + k) % 4 != 2:  # decoded but empty after the warp
                y0, x0 = int(rng.integers(1, 4)), int(rng.integers(1, 4))
                union[y0:y0 + 4, x0:x0 + 5] = 1
            conf = torch.from_numpy(rng.random((side, side)).astype(np.float32)).to(device)
            union = union.to(device)
            if (frame + k) % 4 == 2 and frame % 2 == 0:
                conf = None  # an empty sibling without a score plane joins a scored group
            count = torch.tensor([k + frame], dtype=torch.int32, device=device)
            payload = inference.GpuFlattenedRetinaPayload(
                union, conf, instance_count=-1 if device_counts else k + frame,
                instance_count_device=count if device_counts else None)
            payloads.append((cent, payload))
        return payloads

    frames = [frame_payloads(frame) for frame in range(4)]
    recorded = []
    for batched in (False, True):
        writer = _Recorder()
        sink = WorkerCutoffPlanes(writer, floor_cent=20, offset=5, shape=native,
                                  has_scores=scores, expected_frames=4)
        sink.configure_warp(out_size, matrix, native)
        sink.device_counts = device_counts
        for frame, payloads in enumerate(frames):
            sink.consume_empty(frame, 20, 1, frames=1)
            if batched:
                sink.consume_results(frame, iter(payloads))
            else:
                for cent, payload in payloads:
                    sink.consume_result(frame, cent, payload)
        receipt = sink.finish()
        assert receipt['status'] == 'complete', receipt
        recorded.append((writer.calls, receipt))
    _calls_equal(recorded[0][0], recorded[1][0])
    assert recorded[0][1] == recorded[1][1]
    assert any(call[0] == 'put' for call in recorded[0][0])


def test_crop_transfer_converts_masks_exactly_like_uint8_cast():
    torch = pytest.importorskip('torch')
    mask = torch.zeros((5, 6), dtype=torch.bool); mask[1:4, 2:5] = True
    raw = torch.zeros((5, 6), dtype=torch.uint8); raw[0:2, 0:3] = 3
    score = torch.arange(30, dtype=torch.uint8).reshape(5, 6)
    crops = WorkerCutoffPlanes._crops_to_host(((mask, score, (1, 4, 2, 5)), (raw, None, (0, 2, 0, 3))))
    assert np.array_equal(crops[0][0], mask[1:4, 2:5].to(torch.uint8).numpy())
    assert np.array_equal(crops[0][1], score[1:4, 2:5].numpy())
    assert crops[1][1] is None and np.array_equal(crops[1][0], raw[0:2, 0:3].numpy())
    raw.fill_(0)
    assert crops[1][0].max() == 3  # private storage
