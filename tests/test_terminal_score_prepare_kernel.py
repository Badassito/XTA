"""Compiled, threaded terminal score masking matches the previous numpy loop exactly."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from XTA import confidence_capture_cpu as cpu
from XTA import confidence_evidence as evidence
from XTA.confidence_capture import confidence_capture_resources, plan_confidence_capture


def _inputs(seed, shape=(9, 61, 47), *, boolean=False, empty_frames=(2, 5)):
    rng = np.random.default_rng(seed)
    mask = rng.choice(np.asarray([0, 0, 1, 2, 255], np.uint8), shape)
    for z in empty_frames:
        mask[z] = 0
    mask[3, :, :] = 0
    mask[3, 10:20, 4:9] = 7
    scores = rng.integers(0, 256, shape, dtype=np.uint8)
    if boolean:
        mask = mask.astype(bool)
    active = np.any(mask != 0, axis=(1, 2))
    boxes = np.full((shape[0], 4), -1, np.int64)
    for z in np.flatnonzero(active):
        yy, xx = np.nonzero(mask[z])
        boxes[z] = (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1)
    return mask, scores, active, boxes


def _reference(mask, scores, active=None, boxes=None):
    """The previous chunked numpy semantics."""
    result = scores.copy()
    if active is None:
        result[mask == 0] = 0
        return result
    for z in np.flatnonzero(active):
        y0, y1, x0, x1 = map(int, boxes[z])
        crop = result[z, y0:y1, x0:x1]
        crop[mask[z, y0:y1, x0:x1] == 0] = 0
    return result


@pytest.mark.parametrize('boolean', [False, True])
@pytest.mark.parametrize('threads', [1, 3, 8, 64])
def test_kernels_match_reference_for_any_split(boolean, threads):
    mask, scores, active, boxes = _inputs(17, boolean=boolean)
    flat = scores.copy()
    evidence._run_terminal_score_masking(cpu.zero_unmasked_scores_flat,
        (mask.reshape(-1), flat.reshape(-1)), flat.size, threads)
    np.testing.assert_array_equal(flat, _reference(mask, scores))
    boxed = scores.copy()
    evidence._run_terminal_score_masking(cpu.zero_unmasked_scores_boxes,
        (mask, boxed, active, boxes), boxed.shape[0], threads)
    np.testing.assert_array_equal(boxed, _reference(mask, scores, active, boxes))
    # Outside the support boxes nothing is touched.
    outside = np.ones(scores.shape, bool)
    for z in np.flatnonzero(active):
        y0, y1, x0, x1 = boxes[z]
        outside[z, y0:y1, x0:x1] = False
    np.testing.assert_array_equal(boxed[outside], scores[outside])


def test_masking_runner_covers_every_index_once():
    seen = np.zeros(1001, np.int64)
    lock = threading.Lock()

    def task(first, stop):
        with lock:
            seen[first:stop] += 1
    for threads in (1, 2, 7, 500):
        seen[:] = 0
        evidence._run_terminal_score_masking(task, (), seen.size, threads)
        assert np.all(seen == 1)
    evidence._run_terminal_score_masking(task, (), 0, 4)  # no work, no error


def test_masking_runner_waits_for_every_piece_before_raising():
    finished = []
    release = threading.Event()

    def task(first, stop):
        if first == 0:
            raise RuntimeError('piece failed')
        release.wait(0.2)
        finished.append(first)
    with pytest.raises(RuntimeError, match='piece failed'):
        evidence._run_terminal_score_masking(task, (), 40, 4)
    assert len(finished) == 3  # 4 pieces; all others ended before the raise


def test_thread_count_override(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS', '3')
    assert evidence._terminal_score_prepare_threads() == 3
    monkeypatch.setenv('YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS', 'bogus')
    assert evidence._terminal_score_prepare_threads() == 1
    monkeypatch.setenv('YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS', '0')
    assert evidence._terminal_score_prepare_threads() == 1
    monkeypatch.delenv('YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS')
    assert evidence._terminal_score_prepare_threads() == 1  # one nogil call by default


@pytest.mark.parametrize('metadata', [True, False])
@pytest.mark.parametrize('threads', ['1', '5'])
@pytest.mark.parametrize('boolean', [False, True])
def test_prepare_terminal_scores_matches_previous_result(tmp_path, monkeypatch, metadata, threads, boolean):
    monkeypatch.setenv('YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS', threads)
    mask, scores, active, boxes = _inputs(91, boolean=boolean)
    expected = _reference(mask, scores, active, boxes) if metadata else _reference(mask, scores)
    evidence.configure_confidence_evidence(tmp_path, enabled=True, defer_projection=True)
    plan = plan_confidence_capture(mask.shape, 2, workspace_bytes=32 * 1024**2)
    try:
        with confidence_capture_resources(plan):
            kwargs = dict(known_slice_any=active, known_slice_bboxes=boxes) if metadata else {}
            prepared = evidence.prepare_terminal_confidence_scores(mask, scores, **kwargs)
            assert prepared is not None and prepared.scores is scores
            np.testing.assert_array_equal(scores, expected)
            assert not scores.flags.writeable
            if metadata:
                assert prepared.reader.mask is prepared.reader.scores
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)


@pytest.mark.parametrize('metadata', [True, False])
@pytest.mark.parametrize('readonly_mask', [False, True])
@pytest.mark.parametrize('boolean', [False, True])
def test_startup_warmup_covers_every_production_signature(tmp_path, metadata, readonly_mask, boolean):
    """No terminal view may compile in the busy main process after startup warmup."""
    receipt = cpu.warm_terminal_score_kernels()
    assert receipt['boxes_signatures'] >= 4 and receipt['flat_signatures'] >= 4
    before = (len(cpu.zero_unmasked_scores_flat.nopython_signatures),
              len(cpu.zero_unmasked_scores_boxes.nopython_signatures))
    mask, scores, active, boxes = _inputs(33, boolean=boolean)
    expected = _reference(mask, scores, active, boxes) if metadata else _reference(mask, scores)
    if readonly_mask:
        mask.setflags(write=False)
    evidence.configure_confidence_evidence(tmp_path, enabled=True, defer_projection=True)
    plan = plan_confidence_capture(mask.shape, 2, workspace_bytes=32 * 1024**2)
    try:
        with confidence_capture_resources(plan):
            kwargs = dict(known_slice_any=active, known_slice_bboxes=boxes) if metadata else {}
            assert evidence.prepare_terminal_confidence_scores(mask, scores, **kwargs) is not None
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)
    np.testing.assert_array_equal(scores, expected)
    assert (len(cpu.zero_unmasked_scores_flat.nopython_signatures),
            len(cpu.zero_unmasked_scores_boxes.nopython_signatures)) == before


@pytest.mark.parametrize('metadata', [True, False])
def test_threaded_failure_releases_dense_arrays_after_frame_clearing(tmp_path, monkeypatch, metadata):
    """A failed piece re-raises on the caller; cleared frames keep no array alive."""
    import gc
    import traceback
    import weakref
    monkeypatch.setenv('YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS', '4')
    real_flat, real_boxes = cpu.zero_unmasked_scores_flat, cpu.zero_unmasked_scores_boxes

    def failing(real):
        def kernel(*args):
            if args[-2] == 0:
                raise ValueError('piece failed')
            return real(*args)
        return kernel
    monkeypatch.setattr(cpu, 'zero_unmasked_scores_flat', failing(real_flat))
    monkeypatch.setattr(cpu, 'zero_unmasked_scores_boxes', failing(real_boxes))
    mask, scores, active, boxes = _inputs(5)
    roots = weakref.ref(mask), weakref.ref(scores)
    evidence.configure_confidence_evidence(tmp_path, enabled=True, defer_projection=True)
    plan = plan_confidence_capture(mask.shape, 2, workspace_bytes=32 * 1024**2)
    error = None
    try:
        with confidence_capture_resources(plan):
            kwargs = dict(known_slice_any=active, known_slice_bboxes=boxes) if metadata else {}
            try:
                evidence.prepare_terminal_confidence_scores(mask, scores, **kwargs)
            except ValueError as caught:
                error = caught
                traceback.clear_frames(caught.__traceback__)
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)
    assert error is not None and error.__traceback__ is not None
    del mask, scores, kwargs
    gc.collect()
    assert roots[0]() is None and roots[1]() is None
