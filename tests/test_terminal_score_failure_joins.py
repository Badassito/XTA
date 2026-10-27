"""Terminal score-only failures settle real readers before returning ownership."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gc
from threading import Event
import weakref

import numpy as np
import pytest

from XTA import assembly, confidence_evidence as evidence, pipeline, runtime
from XTA.confidence_capture import confidence_capture_resources, plan_confidence_capture
from XTA.confidence_orchestration import (
    ConfidenceCaptureQueue, RetiringConfidenceCapture,
)
from XTA.projection_queue import settle_prepared_view_components
from tests.test_packed_tilted_projection import _task, _VIEWS


@pytest.fixture(autouse=True)
def isolated_cpu_capture(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '0')
    evidence.configure_confidence_evidence(None, enabled=False)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        evidence.shutdown_confidence_publication(raise_errors=False)
        evidence.configure_confidence_evidence(None, enabled=False)
        if was_enabled:
            gc.enable()


def _configure(root, shape, *, slots=1, capture_workers=1):
    plan = plan_confidence_capture(shape, capture_workers)
    evidence.configure_confidence_evidence(root/'out', enabled=True,
        defer_projection=True, background_publication=True)
    owner = evidence.configure_terminal_confidence_capture(slots=slots,
        workspace_bytes=slots*plan.workspace_bytes, source_bytes=4*np.prod(shape))
    assert owner is not False and owner is not None
    return plan, owner


def _owned_prepare(root, *, use_roi=False, capture_workers=1):
    """Use genuinely owned heap roots, rather than pretending disk mmaps qualify."""
    root.mkdir(parents=True, exist_ok=True)
    task, expected, _, _, _ = _task(root)
    old, task.union_mm = task.union_mm, None
    runtime.close_memmap_array_without_flush(old, unlink_path=task.union_path)
    old = None
    task.union_mm = np.array(expected, dtype=np.uint8, copy=True)
    task.confmap_mm = np.full(expected.shape, 143, dtype=np.uint8)
    task.confmap_path = None
    if use_roi:
        active = np.any(expected != 0, axis=(1, 2))
        boxes = np.zeros((expected.shape[0], 4), dtype=np.int64)
        for z in np.flatnonzero(active):
            rows, columns = np.nonzero(expected[z])
            boxes[z] = rows.min(), rows.max()+1, columns.min(), columns.max()+1
        task.slice_meta = dict(valid=True, slice_any=active, slice_bboxes=boxes)
    roots = weakref.ref(task.union_mm), weakref.ref(task.confmap_mm)
    retired = []
    task.confidence_retired_callback = lambda *args: retired.append(args)
    plan, queue = _configure(root, expected.shape, capture_workers=capture_workers)
    task.prepare = partial(pipeline._prepare_parent_with_confidence_capture,
        assembly.prepare_view_volume_after_fullframe, capture_plan=plan)
    return task, expected, roots, retired, plan, queue


def _score_request(plan, capture, retire):
    mask = np.ones(plan.shape, dtype=np.uint8)
    mask.reshape(-1)[::3] = 0
    scores = np.full(plan.shape, 91, dtype=np.uint8)
    roots = weakref.ref(mask), weakref.ref(scores)
    with confidence_capture_resources(plan):
        prepared = evidence.prepare_terminal_confidence_scores(mask, scores)
    assert prepared is not None and prepared.scores is scores
    request = RetiringConfidenceCapture(prepared.scores, prepared.scores,
        plan=plan, capture=capture, retire=retire, kwargs={}, score_only_source=True)
    return request, roots


def _assert_empty(queue):
    state = queue.snapshot()
    assert state['source_bytes'] == state['working_bytes'] == 0


def _observe_prepared_requests(queue, monkeypatch):
    observations = []
    original = queue.try_submit

    def submit(request):
        prepared = reader = None
        try:
            prepared = request.kwargs.get('prepared')
            if prepared is not None:
                reader = prepared.reader
                observations.append((weakref.ref(request), weakref.ref(prepared),
                    None if reader is None else weakref.ref(reader)))
            return original(request)
        finally:
            request = prepared = reader = None

    monkeypatch.setattr(queue, 'try_submit', submit)
    return observations


def _assert_prepared_requests_released(observations, *, use_roi):
    assert observations
    for request_ref, prepared_ref, reader_ref in observations:
        request = request_ref()
        if request is not None:
            assert request.kwargs is request.mask is request.scores is None
        assert prepared_ref() is None
        if use_roi:
            assert reader_ref is not None
        if reader_ref is not None:
            assert reader_ref() is None


@pytest.mark.parametrize('use_roi', [False, True], ids=['full-grid', 'validated-roi'])
def test_refused_terminal_capture_uses_the_same_sanitized_source_synchronously(tmp_path, monkeypatch, use_roi):
    task, expected, roots, retired, plan, queue = _owned_prepare(tmp_path, use_roi=use_roi)
    entered, release = Event(), Event()
    original = evidence.publish_confidence_scores
    original_try = evidence.try_capture_terminal_confidence_scores
    original_sync = evidence.capture_terminal_confidence_scores
    handoffs = []

    def try_capture(prepared, **kwargs):
        result = original_try(prepared, **kwargs)
        if kwargs['model_name'] == 'model':
            assert result is None
            handoffs.append(('refused', id(prepared), id(prepared.reader)))
        return result

    def synchronous_capture(prepared, **kwargs):
        if kwargs['model_name'] == 'model':
            handoffs.append(('sync', id(prepared), id(prepared.reader)))
        return original_sync(prepared, **kwargs)

    def publish(values, **kwargs):
        if kwargs['model_name'] == 'occupant':
            entered.set()
            assert release.wait(10)
        return original(values, **kwargs)

    monkeypatch.setattr(evidence, 'publish_confidence_scores', publish)
    monkeypatch.setattr(evidence, 'try_capture_terminal_confidence_scores', try_capture)
    monkeypatch.setattr(evidence, 'capture_terminal_confidence_scores', synchronous_capture)
    mask = np.ones(expected.shape, dtype=np.uint8)
    scores = np.full(expected.shape, 71, dtype=np.uint8)
    with confidence_capture_resources(plan):
        occupant = evidence.prepare_terminal_confidence_scores(mask, scores)
        first = evidence.try_capture_terminal_confidence_scores(occupant,
            retire=lambda _value: None, view=task.view, model_name='occupant', temp_dir=tmp_path)
    mask = scores = occupant = None
    try:
        assert entered.wait(10)
        result = task()
        # Queue refusal must not try to remask/recover an original dense source.
        assert roots[0]() is roots[1]() is None and len(retired) == 1
        assert queue.snapshot()['source_bytes'] == expected.nbytes
        assert result.pending_component_layers == []
        assert [entry[0] for entry in handoffs] == ['refused', 'sync']
        assert handoffs[0][1:] == handoffs[1][1:]
    finally:
        release.set()
        first.result(timeout=10)
    evidence.drain_confidence_publication()
    reference, = [ref for ref in evidence._REGISTRY.values() if ref.model_name == 'model']
    with reference.native_reader() as reader:
        actual, _known = reader(0, expected.shape[0])
    np.testing.assert_array_equal(actual, np.where(expected != 0, 143, 0).astype(np.uint8))
    _assert_empty(queue)


def test_aborted_queued_score_reader_never_runs_or_keeps_a_mask_root(tmp_path):
    shape = (2, 8, 8)
    plan, _ = _configure(tmp_path, shape)
    entered, release = Event(), Event()
    retired = []

    def block(mask, scores):
        assert mask is scores and not scores.flags.writeable
        entered.set()
        assert release.wait(10)

    queue = ConfidenceCaptureQueue(slots=2, workspace_bytes=plan.workspace_bytes,
        source_bytes=4*np.prod(shape))
    first_request, _ = _score_request(plan, block, lambda _value: None)
    first = queue.try_submit(first_request)
    try:
        assert entered.wait(10)
        second_request, roots = _score_request(plan,
            lambda *_args: pytest.fail('aborted score reader ran'),
            lambda score: retired.append(score.nbytes))
        assert roots[0]() is None  # No mask retained even before queue cancellation.
        second = queue.try_submit(second_request)
        assert second is not None
        queue._queue.abort()
        with pytest.raises(RuntimeError, match='aborted'):
            second.result(timeout=10)
        assert roots[1]() is None and retired == [np.prod(shape)]
        assert second_request.mask is second_request.scores is None
        release.set()
        first.result(timeout=10)
        _assert_empty(queue)
    finally:
        release.set()
        queue.close(raise_errors=False)


def test_cancelled_unstarted_score_future_closes_its_real_request(tmp_path):
    plan, _ = _configure(tmp_path, (2, 8, 8))
    queue = ConfidenceCaptureQueue(slots=2, workspace_bytes=plan.workspace_bytes,
        source_bytes=1024)
    entered, executor_occupied, release = Event(), Event(), Event()
    retired = []

    def active_reader(*_args):
        entered.set()
        assert release.wait(10)

    def occupy_last_executor_thread():
        executor_occupied.set()
        assert release.wait(10)

    first_request, _ = _score_request(plan, active_reader, lambda _score: None)
    first = queue.try_submit(first_request)
    assert entered.wait(10)
    occupant = queue._queue._executor.submit(occupy_last_executor_thread)
    assert executor_occupied.wait(10)
    task, roots = _score_request(plan,
        lambda *_args: pytest.fail('cancelled reader started'),
        lambda score: retired.append(score.nbytes))
    try:
        accepted = queue.try_submit(task)
        assert accepted is not None and roots[0]() is None
        assert queue.snapshot()['source_bytes'] == 256
        assert accepted.cancel()
        # Keep the request and real cancelled executor Future alive.
        assert task.mask is task.scores is None
        assert roots[1]() is None and retired == [128]
        assert queue.snapshot()['source_bytes'] == 128
        release.set()
        first.result(timeout=10)
        occupant.result(timeout=10)
        _assert_empty(queue)
    finally:
        release.set()
        queue.close(raise_errors=False)


@pytest.mark.parametrize('reader_started', [False, True])
def test_submit_then_error_revokes_or_joins_score_only_reader(tmp_path, monkeypatch, reader_started):
    task, _expected, roots, retired, _plan, queue = _owned_prepare(tmp_path, use_roi=True)
    observed_requests = _observe_prepared_requests(queue, monkeypatch)
    entered, release = Event(), Event()
    deferred = []
    original_writer = evidence.write_block_confidence_evidence

    def writer(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(evidence, 'write_block_confidence_evidence', writer)
    original_submit = queue._queue._executor.submit

    def fail_after_enqueue(fn):
        if reader_started:
            original_submit(fn)
            assert entered.wait(10)
        else:
            deferred.append(fn)
        raise RuntimeError('injected score submit failure after enqueue')

    monkeypatch.setattr(queue._queue._executor, 'submit', fail_after_enqueue)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            failed_submit = caller.submit(task)
            if reader_started:
                assert entered.wait(10)
                assert not failed_submit.done() and not retired
                release.set()
            with pytest.raises(RuntimeError, match='score submit failure') as caught:
                failed_submit.result(timeout=10)
        assert caught.value.__traceback__ is not None and failed_submit.done()
        assert roots[0]() is roots[1]() is None
        assert len(retired) == 1
        _assert_prepared_requests_released(observed_requests, use_roi=True)
        for wrapper in deferred:
            with pytest.raises(Exception, match='revoked'):
                wrapper()
        _assert_empty(queue)
    finally:
        release.set()
        queue.close(raise_errors=False)


@pytest.mark.parametrize('capture_fails', [False, True])
@pytest.mark.parametrize('use_roi', [False, True], ids=['full-grid', 'validated-roi'])
def test_binary_pack_failure_joins_running_score_reader_and_preserves_primary(tmp_path, monkeypatch, capture_fails, use_roi):
    from XTA import packed_tilted_projection
    task, _expected, roots, retired, _plan, queue = _owned_prepare(tmp_path, use_roi=use_roi)
    observed_requests = _observe_prepared_requests(queue, monkeypatch)
    entered, release, pack_failed = Event(), Event(), Event()
    original_writer = evidence.write_block_confidence_evidence

    def writer(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        if capture_fails:
            raise ValueError('secondary score codec failure')
        return original_writer(*args, **kwargs)

    def fail_pack(*args, **kwargs):
        assert entered.wait(10)
        pack_failed.set()
        raise RuntimeError('primary binary pack failure')

    monkeypatch.setattr(evidence, 'write_block_confidence_evidence', writer)
    monkeypatch.setattr(packed_tilted_projection, 'pack_terminal_native_source', fail_pack)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            failed_parent = caller.submit(task)
            assert pack_failed.wait(10)
            assert not failed_parent.done() and roots[1]() is not None and not retired
            release.set()
            with pytest.raises(RuntimeError, match='primary binary pack failure') as caught:
                failed_parent.result(timeout=10)
        if capture_fails:
            assert any('secondary score codec failure' in note for note in caught.value.__notes__)
        assert roots[0]() is roots[1]() is None
        assert task.union_mm is task.confmap_mm is None and len(retired) == 1
        _assert_prepared_requests_released(observed_requests, use_roi=use_roi)
        _assert_empty(queue)
    finally:
        release.set()


@pytest.mark.parametrize('use_roi', [False, True], ids=['full-grid', 'validated-roi'])
def test_partial_inplace_mask_failure_aborts_without_original_score_retry(tmp_path, monkeypatch, use_roi):
    task, expected, roots, retired, _plan, queue = _owned_prepare(tmp_path, use_roi=use_roi)
    if use_roi:
        # Keep the real bbox unchanged but include an unknown score inside it.
        expected[:, 2, 2] = task.union_mm[:, 2, 2] = 0
    original_copy = evidence.np.copyto
    original_prepare = evidence.prepare_terminal_confidence_scores
    observed, writes = [], []

    def partial_copy(destination, source, *args, **kwargs):
        if not np.shares_memory(destination, roots[1]()):
            return original_copy(destination, source, *args, **kwargs)
        writes.append(1)
        if use_roi and len(writes) == 1:
            return original_copy(destination, source, *args, **kwargs)
        where = kwargs['where']
        halfway = max(1, len(destination)//2)
        original_copy(destination[:halfway], source, where=where[:halfway])
        raise ValueError('partial terminal score mutation failure')

    def prepare(mask, scores, **kwargs):
        try:
            return original_prepare(mask, scores, **kwargs)
        except ValueError:
            np.testing.assert_array_equal(mask, expected)
            observed.append(np.array(scores, copy=True))
            raise

    def no_retry(*args, **kwargs):
        pytest.fail('partially transformed scores were retried as a legacy source')

    monkeypatch.setattr(evidence.np, 'copyto', partial_copy)
    monkeypatch.setattr(evidence, 'prepare_terminal_confidence_scores', prepare)
    monkeypatch.setattr(evidence, 'try_capture_prediction_confidence', no_retry)
    monkeypatch.setattr(evidence, 'capture_prediction_confidence', no_retry)
    with ThreadPoolExecutor(max_workers=1) as caller:
        failed_parent = caller.submit(task)
        with pytest.raises(ValueError, match='partial terminal score mutation') as caught:
            failed_parent.result(timeout=10)
    assert failed_parent.done() and caught.value.__traceback__ is not None
    assert len(observed) == 1 and np.any(observed[0] == 0) and np.any(observed[0] == 143)
    if use_roi:
        assert len(writes) == 2  # One ROI plane sealed before the next partial write failed.
    assert roots[0]() is roots[1]() is None and len(retired) == 1
    assert not evidence._REGISTRY
    _assert_empty(queue)


@pytest.mark.parametrize('use_roi', [False, True], ids=['full-grid', 'validated-roi'])
def test_score_codec_failure_settles_child_without_retaining_dense_roots(tmp_path, monkeypatch, use_roi):
    task, _expected, roots, retired, _plan, queue = _owned_prepare(tmp_path, use_roi=use_roi)
    observed_requests = _observe_prepared_requests(queue, monkeypatch)

    def fail_codec(*args, **kwargs):
        # A real writer call is reached after real score masking/queue admission.
        reader = args[2]
        if hasattr(reader, 'iter_crops'):
            list(reader.iter_crops(0))
        else:
            reader(0)
        raise OSError('injected terminal score codec failure')

    monkeypatch.setattr(evidence, 'write_block_confidence_evidence', fail_codec)
    result = task()
    child, = result.pending_component_layers
    with pytest.raises(OSError, match='terminal score codec failure') as caught:
        child.result(timeout=10)
    assert caught.value.__traceback__ is not None and child.done()
    assert roots[0]() is roots[1]() is None and len(retired) == 1
    _assert_prepared_requests_released(observed_requests, use_roi=use_roi)
    with pytest.raises(OSError, match='terminal score codec failure'):
        settle_prepared_view_components(result)
    assert not evidence._REGISTRY
    _assert_empty(queue)


def test_roi_encode_failure_joins_real_shared_producers_without_retaining_sources(tmp_path, monkeypatch):
    from XTA.native_pull_scheduling import (
        configure_native_pull_scheduler, shutdown_native_pull_scheduler,
    )
    scheduler = configure_native_pull_scheduler(workers=4, max_workers=4)
    task, expected, roots, retired, _plan, queue = _owned_prepare(
        tmp_path, use_roi=True, capture_workers=4)
    observed_requests = _observe_prepared_requests(queue, monkeypatch)
    borrower_entered, producer_failed, release = Event(), Event(), Event()
    encoded_frames = []
    original = evidence._MaskedNativeScoreReader.encode_frame

    def encode_then_fail(reader, z, block_size):
        encoded = original(reader, z, block_size)
        assert encoded[0]  # Actual source reads and numeric block encoding occurred.
        encoded_frames.append(z)
        if z == 0:
            assert borrower_entered.wait(10)
            producer_failed.set()
            raise OSError('injected ROI shared producer failure after encoding')
        borrower_entered.set()
        assert release.wait(10)
        return encoded

    monkeypatch.setattr(evidence._MaskedNativeScoreReader, 'encode_frame', encode_then_fail)
    try:
        result = task()
        child, = result.pending_component_layers
        assert producer_failed.wait(10)
        assert not child.done() and roots[1]() is not None and not retired
        release.set()
        with pytest.raises(OSError, match='ROI shared producer failure') as caught:
            child.result(timeout=10)
        assert encoded_frames and caught.value.__traceback__ is not None and child.done()
        assert roots[0]() is roots[1]() is None
        assert retired == [('model', task.view.name, expected.nbytes)]
        _assert_prepared_requests_released(observed_requests, use_roi=True)
        assert not evidence._REGISTRY
        _assert_empty(queue)
        state = scheduler.snapshot()
        assert state['active_jobs'] == state['active_operations'] == state['retained_operations'] == 0
    finally:
        release.set()
        shutdown_native_pull_scheduler()


def test_numeric_publication_failure_joins_after_real_codec_releases_dense_sources(tmp_path, monkeypatch):
    from XTA import confidence_publication
    task, _expected, roots, retired, _plan, queue = _owned_prepare(tmp_path)
    copied = []

    def fail_copy(stage, destination, **kwargs):
        # Opening the actual numeric stage proves codec ownership is independent.
        reference = evidence.ConfidenceEvidenceRef.open(stage)
        copied.append(reference.metadata['layout'])
        raise OSError('injected terminal numeric publication failure')

    monkeypatch.setattr(confidence_publication, 'copy_staged_blocks', fail_copy)
    result = task()
    child, = result.pending_component_layers
    child.result(timeout=10)
    assert roots[0]() is roots[1]() is None and len(retired) == 1
    with pytest.raises(OSError, match='terminal numeric publication failure') as caught:
        evidence.drain_confidence_publication()
    assert copied and caught.value.__traceback__ is not None
    assert not evidence._REGISTRY
    assert evidence._PUBLICATION.snapshot()['reserved_numeric_bytes'] == 0
    assert not list((tmp_path/'confidence_publication').glob('confidence-publish-*'))
    _assert_empty(queue)
