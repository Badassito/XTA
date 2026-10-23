"""Actual confidence staging jobs share the run CPU and ordered-buffer budget."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import threading
import time
import weakref

import numpy as np
import pytest

from XTA import confidence_evidence as evidence, geometry, pipeline
from XTA.confidence_capture import confidence_capture_resources
from XTA.confidence_capture_cpu import warm_confidence_capture_kernels
from XTA.confidence_publication import ConfidencePublicationQueue
from XTA.confidence_storage import _ordered_encoded_frames
from XTA.native_pull_scheduling import (
    configure_native_pull_scheduler, shutdown_native_pull_scheduler,
)


def wait_until(predicate, timeout=10):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for bounded capture progress')
        time.sleep(.005)


@pytest.fixture(autouse=True)
def clean_run_owners():
    evidence.configure_confidence_evidence(None, enabled=False)
    yield
    evidence.shutdown_confidence_publication(raise_errors=False)
    evidence.configure_confidence_evidence(None, enabled=False)
    shutdown_native_pull_scheduler()


def test_shared_window_is_charged_before_admission_without_changing_standalone_workers():
    shape = (1931, 2048, 2048)
    standalone = pipeline._parent_confidence_capture_plan(shape, 1, enabled=True)
    assert standalone.workers == standalone.requested_workers == 1
    scheduler = configure_native_pull_scheduler(workers=32, max_workers=128)
    shared = pipeline._parent_confidence_capture_plan(shape, 1, enabled=True)
    assert shared.requested_workers == scheduler.max_workers
    assert 32 <= shared.workers < 128
    assert standalone.workspace_bytes < shared.workspace_bytes <= 256*1024**2
    assert shared.workspace_bytes >= (shared.fixed_bytes+shared.consumer_bytes
                                     + shared.workers*shared.worker_bytes)
    # Tail resizing changes CPU admission, never this already charged window.
    scheduler.resize(128)
    assert pipeline._parent_confidence_capture_plan(shape, 1, enabled=True) == shared


def test_actual_four_staging_slots_share_32_workers_with_21_parent_threads(tmp_path, monkeypatch):
    warm_confidence_capture_kernels()
    shape = (64, 32, 32)
    scores = np.random.default_rng(160587).integers(0, 256, shape, dtype=np.uint8)
    mask = np.ones(shape, np.uint8)
    active = np.ones(shape[0], bool)
    boxes = np.tile(np.array((0, shape[1], 0, shape[2]), np.int64), (shape[0], 1))
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    # Start with one credit so all four real numeric slots can register before
    # admission grows to32. This exercises the same live resizing used at drain.
    scheduler = configure_native_pull_scheduler(workers=1, max_workers=128)
    plan = pipeline._parent_confidence_capture_plan(shape, 1, enabled=True)
    assert plan.workers == shape[0] and plan.workspace_bytes <= 256*1024**2
    evidence.configure_confidence_evidence(tmp_path/'out', enabled=True, defer_projection=True)
    queue = ConfidencePublicationQueue()  # Unchanged256MiB /64MiB = four slots.
    monkeypatch.setattr(evidence, '_PUBLICATION', queue)
    release = threading.Event()
    lock = threading.Lock()
    running = {}
    original = evidence._MaskedNativeScoreReader.encode_frame

    def controlled(reader, z, block):
        with lock:
            running[id(reader)] = running.get(id(reader), 0)+1
        try:
            assert release.wait(10)
            return original(reader, z, block)
        finally:
            with lock:
                running[id(reader)] -= 1

    monkeypatch.setattr(evidence._MaskedNativeScoreReader, 'encode_frame', controlled)
    parents = ThreadPoolExecutor(max_workers=21)

    def capture(number):
        current = replace(view, name=f'capture_parent_{number}')
        with confidence_capture_resources(plan):
            return evidence.capture_prediction_confidence(mask, scores, view=current,
                model_name='m', temp_dir=tmp_path, known_slice_any=active,
                known_slice_bboxes=boxes)

    try:
        captures = [parents.submit(capture, number) for number in range(21)]
        wait_until(lambda: scheduler.snapshot()['active_confidence_captures'] == 4)
        assert scheduler.snapshot()['active_jobs'] == 4
        assert scheduler.snapshot()['active_projections'] == 0
        assert queue.snapshot()['reserved_numeric_bytes'] == 256*1024**2
        scheduler.resize(32)
        wait_until(lambda: scheduler.snapshot()['active_operations'] == 32)
        def four_equal_shares():
            with lock:
                return sorted(running.values()) == [8, 8, 8, 8]
        wait_until(four_equal_shares)
        state = scheduler.snapshot()
        assert state['retained_operations'] == state['peak_active_operations'] == 32
        release.set()
        publications = [capture.result(timeout=30) for capture in captures]
        references = [future.result(timeout=30) for future in publications]
        assert queue.drain()['reserved_numeric_bytes'] == 0
        assert queue.snapshot()['peak_reserved_numeric_bytes'] == 256*1024**2
        assert scheduler.snapshot()['active_jobs'] == 0
        assert scheduler.snapshot()['active_operations'] == 0
        assert scheduler.snapshot()['retained_operations'] == 0
        # Real numeric format/order/unknown values must match serial publication.
        reference = references[0]
        expected = evidence.write_block_confidence_evidence(tmp_path/'serial', shape,
            lambda z:np.where(mask[z], scores[z], np.uint8(0)),
            model_name='m', layer_key=reference.layer_key,
            coordinate_space='native_view_processing', source_shape_tyx=shape)
        for result in references:
            for name in ('scores.u8.zlib', 'index.bin'):
                assert (result.path/name).read_bytes() == (expected.path/name).read_bytes()
    finally:
        release.set()
        parents.shutdown(wait=True, cancel_futures=True)


def test_failed_capture_joins_borrowers_then_releases_source_and_cpu_credits():
    scheduler = configure_native_pull_scheduler(workers=4, max_workers=4)
    entered, release = threading.Event(), threading.Event()
    class Reader:
        def __init__(self):
            self.source = np.ones((8, 7, 9), np.uint8)
        def encode_frame(self, z, block):
            if z == 0:
                assert entered.wait(10)
                raise ValueError('injected capture failure')
            entered.set()
            assert release.wait(10)
            return self.source[z].copy(), {}
    reader = Reader()
    source_ref, reader_ref = weakref.ref(reader.source), weakref.ref(reader)
    frames = _ordered_encoded_frames(reader, 0, 8, 128, 4)
    def consume_failure():
        try:
            next(frames)
        except ValueError as error:
            return str(error)  # Discard the exception and its source traceback.
        raise AssertionError('Injected capture failure was lost')
    consumer = ThreadPoolExecutor(max_workers=1)
    try:
        future = consumer.submit(consume_failure)
        assert entered.wait(10)
        assert not future.done()  # Readers still borrow source; lease cannot retire.
        release.set()
        assert future.result(timeout=10) == 'injected capture failure'
        frames.close()
        assert scheduler.snapshot()['active_jobs'] == 0
        assert scheduler.snapshot()['active_operations'] == 0
        assert scheduler.snapshot()['retained_operations'] == 0
        del reader, frames
        gc.collect()
        assert reader_ref() is None and source_ref() is None
    finally:
        release.set()
        consumer.shutdown(wait=True)
