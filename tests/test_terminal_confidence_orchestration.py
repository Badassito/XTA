"""Terminal binary projection and confidence own independent, bounded work."""
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import gc
from threading import Event
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA.confidence_capture import plan_confidence_capture, current_confidence_capture_plan
from XTA.confidence_orchestration import (ConfidenceCaptureQueue, RetiringConfidenceCapture,
    retire_confidence_scores, terminal_capture_budget, join_failed_capture, ConfidenceSourceRetirement)
from XTA.projection_queue import settle_prepared_view_components


def request(plan, capture, retire=lambda value:None):
    return RetiringConfidenceCapture(np.ones(plan.shape, np.uint8),
        np.full(plan.shape, 91, np.uint8), plan=plan, capture=capture, retire=retire, kwargs={})


def test_primary_failure_is_preserved_without_python311_exception_notes():
    class LegacyError(ValueError):
        add_note = None
    primary = LegacyError('primary failure')
    future = Future()
    future.set_exception(RuntimeError('secondary capture failure'))
    join_failed_capture(future, primary)
    def capture(*_args):
        raise primary
    def retire(_scores):
        raise RuntimeError('secondary cleanup failure')
    task = request(plan_confidence_capture((1, 8, 8), 1), capture, retire)
    mask_ref, score_ref = weakref.ref(task.mask), weakref.ref(task.scores)
    with pytest.raises(LegacyError) as caught:
        task()
    assert caught.value is primary
    assert mask_ref() is score_ref() is None


def test_independent_grant_and_context_survive_parent_release():
    plan = plan_confidence_capture((3, 8, 8), 2)
    entered, release = Event(), Event()
    def capture(mask, scores):
        assert current_confidence_capture_plan() is plan
        assert mask.sum() == 192 and scores.sum() == 192*91
        entered.set()
        assert release.wait(5)
    queue = ConfidenceCaptureQueue(slots=1, workspace_bytes=plan.workspace_bytes, source_bytes=1000)
    try:
        future = queue.try_submit(request(plan, capture))
        assert entered.wait(5)
        # The parent's ordinary reservation/context may now end. This startup
        # promise and active grant stay live until the independent reader ends.
        assert current_confidence_capture_plan() is None
        snapshot = queue.snapshot()
        assert snapshot['working_bytes'] == snapshot['workspace_reserve_bytes'] == plan.workspace_bytes
        assert snapshot['source_bytes'] == 384
        release.set()
        assert future.result(timeout=5) is None
        queue.drain()
        assert queue.snapshot()['working_bytes'] == queue.snapshot()['source_bytes'] == 0
    finally:
        release.set()
        queue.close()


@pytest.mark.parametrize('failed', [False, True])
def test_gc_disabled_future_and_callbacks_do_not_retain_dense_sources(failed):
    plan = plan_confidence_capture((2, 8, 8), 1)
    queue = ConfidenceCaptureQueue(slots=1, workspace_bytes=plan.workspace_bytes, source_bytes=1000)
    events = []
    def capture(mask, scores):
        nested_values = scores[0]
        assert nested_values.sum() > 0 and mask.sum() > 0
        if failed:
            raise RuntimeError('capture failure')
    task = request(plan, capture, lambda value:events.append(value.nbytes))
    mask_ref, scores_ref = weakref.ref(task.mask), weakref.ref(task.scores)
    prior = gc.isenabled()
    gc.disable()
    try:
        future = queue.try_submit(task)
        if failed:
            with pytest.raises(RuntimeError, match='capture failure'):
                future.result(timeout=5)
        else:
            future.result(timeout=5)
        assert mask_ref() is scores_ref() is None
        assert events == [128]
        assert task.mask is task.scores is task.retire is task.capture is task.kwargs is None
        # Keep both the completed Future (including its callbacks/traceback)
        # and request alive to prove source release without cyclic collection.
        assert future.done() and task is not None
    finally:
        queue.close(raise_errors=False)
        if prior:
            gc.enable()


def test_full_queue_declines_without_taking_source_ownership():
    plan = plan_confidence_capture((1, 8, 8), 1)
    entered, release = Event(), Event()
    def capture(*args):
        entered.set()
        assert release.wait(5)
    queue = ConfidenceCaptureQueue(slots=1, workspace_bytes=plan.workspace_bytes, source_bytes=1000)
    try:
        first = queue.try_submit(request(plan, capture))
        assert entered.wait(5)
        events = []
        declined = request(plan, lambda *args:None, lambda score:events.append(score.nbytes))
        assert queue.try_submit(declined) is None
        assert declined.mask is not None and declined.scores is not None and not events
        # The caller retains the original synchronous capture/retirement path.
        declined()
        assert events == [64]
        assert queue.snapshot()['peak_pending'] == 1
        release.set()
        first.result(timeout=5)
    finally:
        release.set()
        queue.close()


def test_projection_failure_joins_reader_before_source_cleanup():
    plan = plan_confidence_capture((1, 8, 8), 1)
    entered, release, cleaned = Event(), Event(), Event()
    queue = ConfidenceCaptureQueue(slots=1, workspace_bytes=plan.workspace_bytes, source_bytes=1000)
    def capture(*args):
        entered.set()
        assert release.wait(5)
    future = queue.try_submit(request(plan, capture, lambda score:cleaned.set()))
    assert entered.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            joined = executor.submit(join_failed_capture, future, ValueError('binary projection failed'))
            assert not joined.done() and not cleaned.is_set()
            release.set()
            joined.result(timeout=5)
            assert cleaned.is_set() and future.done()
    finally:
        release.set()
        queue.close()


def test_ownership_only_child_blocks_settlement_then_contributes_no_layer():
    child = Future()
    prepared = SimpleNamespace(pending_component_layers=[child], nrrd_layers=['binary'])
    assert not settle_prepared_view_components(prepared)
    child.set_result(None)
    assert settle_prepared_view_components(prepared)
    assert prepared.nrrd_layers == ['binary'] and not prepared.pending_component_layers


def test_failed_ownership_child_propagates_before_component_registration():
    child = Future()
    child.set_exception(RuntimeError('confidence failed'))
    prepared = SimpleNamespace(pending_component_layers=[child], nrrd_layers=['binary'])
    with pytest.raises(RuntimeError, match='confidence failed'):
        settle_prepared_view_components(prepared)
    assert prepared.nrrd_layers == ['binary']


@pytest.mark.parametrize('parent,physical,workers,eligible', [(60,315,21,30), (192,1200,21,30),
    (1,1,320,1000), (60,0,21,30), (60,315,21,0), (60,315,1,30)])
def test_startup_budget_is_bounded_by_existing_parent_and_physical_promises(parent,physical,workers,eligible):
    gib = 1024**3
    budget = terminal_capture_budget(parent_capacity=parent*gib, physical_headroom=physical*gib,
        parent_workers=workers, eligible_parents=eligible)
    assert 0 <= budget['slots'] <= min(workers, eligible)
    assert budget['capture_workspace_bytes']+budget['numeric_reserve_bytes'] <= budget['allowance_bytes']
    assert budget['allowance_bytes'] <= min(parent*gib//8,physical*gib//32)
    if parent == 60 and physical == 315 and workers == 21 and eligible == 30:
        assert budget['slots'] == 21


def test_score_credit_waits_for_the_last_borrowed_view_with_gc_disabled(tmp_path):
    scores = np.memmap(tmp_path/'score.dat', dtype=np.uint8, mode='w+', shape=(2,8,8))
    scores[:] = 91
    alias = scores[0]
    reference = weakref.ref(scores._mmap)
    events, carrier = [], [scores]
    prior = gc.isenabled()
    gc.disable()
    try:
        retire_confidence_scores(scores, model_name='model', view_name='view', path=tmp_path/'score.dat',
            keep_temp=False, retired_callback=lambda *v:events.append(v), carrier=carrier)
        scores = None
        assert not carrier and not events and reference() is not None
        assert alias.sum() == 64*91
        alias = None
        assert events == [('model','view',128)] and reference() is None
    finally:
        if prior:
            gc.enable()


def test_terminal_binary_pack_and_projection_do_not_wait_for_capture(tmp_path, monkeypatch):
    from XTA import assembly, confidence_evidence as evidence, pipeline
    from tests.test_packed_tilted_projection import _task
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    task, expected, mask_ref, _, mask_retired = _task(tmp_path)
    scores = np.memmap(tmp_path/'scores.dat', dtype=np.uint8, mode='w+', shape=expected.shape)
    scores[:] = 143
    score_ref = weakref.ref(scores._mmap)
    task.confmap_mm, task.confmap_path = scores, tmp_path/'scores.dat'
    task.confidence_retired_callback = lambda *args:None
    scores = None
    plan = plan_confidence_capture(expected.shape, 2)
    task.prepare = partial(pipeline._prepare_parent_with_confidence_capture,
        assembly.prepare_view_volume_after_fullframe, capture_plan=plan)
    entered, release, projected = Event(), Event(), Event()
    evidence.configure_confidence_evidence(tmp_path/'out', enabled=True,
        defer_projection=True, background_publication=True)
    evidence.configure_terminal_confidence_capture(slots=2, workspace_bytes=2*plan.workspace_bytes,
        source_bytes=4*expected.nbytes)
    def capture(mask, score, **kwargs):
        assert current_confidence_capture_plan() is plan
        entered.set()
        assert release.wait(5)
        np.testing.assert_array_equal(mask, expected)
        assert np.all(score == 143)
    def project(*args, **kwargs):
        assert entered.wait(5) and not release.is_set()
        # Confidence is still a real dense owner; packing may not claim its
        # mask bytes returned merely because the binary reader detached.
        assert mask_ref() is not None and score_ref() is not None and not mask_retired
        projected.set()
        return None
    monkeypatch.setattr(evidence, 'capture_prediction_confidence', capture)
    monkeypatch.setattr(assembly, '_materialize_terminal_packed_tilted_view_layer', project)
    prior = gc.isenabled()
    gc.disable()
    try:
        result = task()
        assert projected.is_set() and len(result.pending_component_layers) == 1
        assert not result.pending_component_layers[0].done()
        assert not settle_prepared_view_components(result)
        release.set()
        result.pending_component_layers[0].result(timeout=5)
        assert mask_ref() is score_ref() is None
        assert len(mask_retired) == 1
        assert settle_prepared_view_components(result) and result.nrrd_layers == []
    finally:
        release.set()
        evidence.shutdown_confidence_publication(raise_errors=False)
        evidence.configure_confidence_evidence(None, enabled=False)
        if prior:
            gc.enable()


def test_explicit_numeric_publication_limits_are_not_expanded(tmp_path):
    from XTA import confidence_evidence as evidence
    from XTA.confidence_publication import ConfidencePublicationQueue
    evidence.configure_confidence_evidence(tmp_path, enabled=True,
        defer_projection=True, background_publication=True)
    evidence._PUBLICATION.close()
    explicit = ConfidencePublicationQueue(max_pending_bytes=8*1024**2, stage_bytes=4*1024**2)
    evidence._PUBLICATION = explicit
    try:
        assert not evidence.configure_terminal_confidence_capture(slots=21,
            workspace_bytes=21*256*1024**2, source_bytes=1024**3)
        assert evidence._PUBLICATION is explicit
        assert explicit.max_pending_bytes == 8*1024**2 and explicit.stage_bytes == 4*1024**2
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)


def test_real_async_confidence_artifact_is_byte_identical_to_synchronous_capture(tmp_path):
    from XTA import confidence_evidence as evidence
    from XTA.confidence_capture import confidence_capture_resources
    from tests.test_packed_tilted_projection import _VIEWS
    view = _VIEWS[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    original_mask = np.random.default_rng(4160676).integers(0,2,shape,dtype=np.uint8)
    original_scores = np.random.default_rng(160676).integers(0,256,shape,dtype=np.uint8)
    plan = plan_confidence_capture(shape, 2)
    artifacts = []
    prior = gc.isenabled()
    gc.disable()
    try:
        for asynchronous in (False, True):
            root = tmp_path/str(asynchronous)
            evidence.configure_confidence_evidence(root/'out', enabled=True,
                defer_projection=True, background_publication=True)
            if asynchronous:
                evidence.configure_terminal_confidence_capture(slots=2,
                    workspace_bytes=2*plan.workspace_bytes, source_bytes=4*original_mask.nbytes)
            mask, scores = original_mask.copy(), original_scores.copy()
            refs = weakref.ref(mask), weakref.ref(scores)
            with confidence_capture_resources(plan):
                if asynchronous:
                    future = evidence.try_capture_prediction_confidence(mask,scores,retire=lambda v:None,
                        view=view,model_name='model',temp_dir=root,output_shape=(5,7,9))
                else:
                    evidence.capture_prediction_confidence(mask,scores,view=view,
                        model_name='model',temp_dir=root,output_shape=(5,7,9))
            mask = scores = None
            if asynchronous:
                assert future.result(timeout=10) is None
                assert refs[0]() is refs[1]() is None
            evidence.drain_confidence_publication()
            ref, = evidence._REGISTRY.values()
            artifacts.append({p.name:p.read_bytes() for p in ref.path.iterdir() if p.suffix == '.bin'})
            assert artifacts[-1]
            evidence.shutdown_confidence_publication()
        assert artifacts[0] == artifacts[1]
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)
        if prior:
            gc.enable()


@pytest.mark.parametrize('capture_fails', [False, True])
def test_retirement_failure_releases_sources_and_preserves_primary_error_without_gc(capture_fails):
    plan = plan_confidence_capture((1,8,8), 1)
    def capture(mask, scores):
        if capture_fails:
            raise ValueError('primary capture failure')
    def retire(scores):
        alias = scores[0]
        assert alias.sum() > 0
        raise RuntimeError('source retirement failure')
    task = request(plan,capture,retire)
    refs = weakref.ref(task.mask), weakref.ref(task.scores)
    queue = ConfidenceCaptureQueue(slots=1,workspace_bytes=plan.workspace_bytes,source_bytes=1024)
    prior = gc.isenabled()
    gc.disable()
    try:
        future = queue.try_submit(task)
        error_type = ValueError if capture_fails else RuntimeError
        with pytest.raises(error_type) as caught:
            future.result(timeout=5)
        if capture_fails:
            assert any('source retirement failure' in n for n in caught.value.__notes__)
        assert refs[0]() is refs[1]() is None
        assert queue.snapshot()['source_bytes'] == 0
    finally:
        queue.close(raise_errors=False)
        if prior:
            gc.enable()


def test_abort_before_reader_starts_releases_queued_source_holders_without_gc():
    plan = plan_confidence_capture((1,8,8), 1)
    entered, release = Event(), Event()
    def capture(*args):
        entered.set()
        assert release.wait(5)
    queue = ConfidenceCaptureQueue(slots=2,workspace_bytes=plan.workspace_bytes,source_bytes=4096)
    first = queue.try_submit(request(plan,capture))
    assert entered.wait(5)
    task = request(plan,lambda *a:pytest.fail('aborted reader started'))
    refs = weakref.ref(task.mask),weakref.ref(task.scores)
    prior = gc.isenabled()
    gc.disable()
    try:
        second = queue.try_submit(task)
        assert second is not None
        queue._queue.abort()
        with pytest.raises(RuntimeError,match='aborted'):
            second.result(timeout=5)
        assert refs[0]() is refs[1]() is None
        release.set()
        first.result(timeout=5)
    finally:
        release.set()
        queue.close(raise_errors=False)
        if prior:
            gc.enable()


def test_run_resource_close_joins_captures_before_shared_cpu_pool():
    from XTA.tta_lifecycle import PipelineRunResources
    plan = plan_confidence_capture((1,8,8), 1)
    entered, release = Event(), Event()
    events = []
    def capture(*args):
        entered.set()
        assert release.wait(5)
        events.append('capture_finished')
    queue = ConfidenceCaptureQueue(slots=1,workspace_bytes=plan.workspace_bytes,source_bytes=1024)
    resources = PipelineRunResources()
    resources.track_closeable(SimpleNamespace(close=lambda:events.append('cpu_pool_closed')))
    resources.track_closeable(queue)
    resources.track_executor(SimpleNamespace(shutdown=lambda **kwargs:events.append('parent_joined')))
    future = queue.try_submit(request(plan,capture))
    assert entered.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            closed = executor.submit(resources.close,failed=False)
            assert not closed.done() and 'cpu_pool_closed' not in events
            release.set()
            closed.result(timeout=5)
        assert events == ['parent_joined','capture_finished','cpu_pool_closed']
        assert future.done()
    finally:
        release.set()
        queue.close(raise_errors=False)


@pytest.mark.parametrize('reader_started', [False, True])
def test_enqueue_then_submit_failure_revokes_or_joins_before_returning_sources(monkeypatch, reader_started):
    plan = plan_confidence_capture((1,8,8), 1)
    queue = ConfidenceCaptureQueue(slots=1,workspace_bytes=plan.workspace_bytes,source_bytes=1024)
    entered, release = Event(), Event()
    retired = []
    retirement = ConfidenceSourceRetirement(lambda value:retired.append(value.nbytes))
    def capture(*args):
        entered.set()
        assert release.wait(5)
    task = request(plan,capture,retirement)
    refs = weakref.ref(task.mask),weakref.ref(task.scores)
    original_submit = queue._queue._executor.submit
    deferred = []
    def fail_after_enqueue(fn):
        if reader_started:
            original_submit(fn)
            assert entered.wait(5)
        else:
            deferred.append(fn)
        raise RuntimeError('thread start failure after enqueue')
    monkeypatch.setattr(queue._queue._executor,'submit',fail_after_enqueue)
    prior = gc.isenabled()
    gc.disable()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            submitted = executor.submit(queue.try_submit,task)
            if reader_started:
                assert entered.wait(5)
                assert not submitted.done() and not retired
                release.set()
            with pytest.raises(RuntimeError,match='thread start failure'):
                submitted.result(timeout=5)
        # The caller's exception finally shares the same retirement ticket.
        retirement(None)
        assert retired == [64] and refs[0]() is refs[1]() is None
        for wrapper in deferred:
            with pytest.raises(Exception,match='revoked'):
                wrapper()
        assert queue.snapshot()['source_bytes'] == queue.snapshot()['working_bytes'] == 0
    finally:
        release.set()
        queue.close(raise_errors=False)
        if prior:
            gc.enable()
