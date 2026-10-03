"""Byte-exact fused/parallel confidence capture and conservative task admission."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from contextlib import contextmanager
import json
import threading
import time
from unittest import mock

import numpy as np
import pytest

from XTA import confidence_storage as storage
from XTA.confidence_capture import (plan_confidence_capture, confidence_capture_resources,
                                     current_confidence_capture_plan)
from XTA.confidence_evidence import _MaskedNativeScoreReader


def fixture(shape=(7, 273, 311), *, boolean=False, strided=False):
    rng = np.random.default_rng(1074)
    scores = rng.integers(0, 256, shape, dtype=np.uint8)
    mask = (rng.random(shape) < .2).astype(bool if boolean else np.uint8)
    mask[0] = 0
    mask[2, :19] = 0
    mask[3, -22:] = 0
    scores[4, :, :23] = 0
    if strided:
        mask, scores = mask[:, ::-1, ::-1], scores[:, ::-1, ::-1]
    active = np.any(mask, axis=(1, 2))
    boxes = np.full((shape[0], 4), -1, np.int32)
    for z in np.flatnonzero(active):
        y, x = np.nonzero(mask[z])
        boxes[z] = (y.min(), y.max()+1, x.min(), x.max()+1)
    return mask, scores, active, boxes


@pytest.mark.parametrize('workers', (1, 2, 4))
@pytest.mark.parametrize('block', (8, 128))
@pytest.mark.parametrize('boolean,strided', ((False, False), (True, True)))
def test_fused_ordered_capture_is_byte_identical_to_original_full_plane(tmp_path, workers, block, boolean, strided):
    mask, scores, active, boxes = fixture(boolean=boolean, strided=strided)
    original_mask, original_scores = mask.copy(), scores.copy()
    plan = plan_confidence_capture(scores.shape, workers, workspace_bytes=64*1024**2, block_size=block)
    options = dict(layer_key='capture', model_name='model', provenance={'same': 'input'},
                   coordinate_space='native_view_processing', source_shape=(7,273,311), block_size=block)
    reference = storage.write_blocks(tmp_path/'reference', scores.shape,
        lambda z:np.where(mask[z] != 0, scores[z], np.uint8(0)), **options)
    metrics = {}
    with confidence_capture_resources(plan):
        reader = _MaskedNativeScoreReader(mask, scores, active, boxes)
        assert not reader.mask.flags.writeable and not reader.scores.flags.writeable
        with mock.patch.object(reader, 'iter_crops', side_effect=AssertionError('whole-hull copy')):
            actual = storage.write_blocks(tmp_path/'actual', scores.shape, reader, metrics=metrics, **options)
    assert actual == reference
    for name in ('index.bin', 'scores.u8.zlib', 'metadata.json'):
        assert (tmp_path/'actual'/name).read_bytes() == (tmp_path/'reference'/name).read_bytes()
    assert metrics['capture_backend'] == 'compiled_masked_cells_ordered_frames'
    assert metrics['capture_workers'] == plan.workers
    assert metrics['capture_workspace_bytes'] == plan.workspace_bytes
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(scores, original_scores)


def test_admission_declines_before_numeric_arrays_and_matches_exact_window():
    plan = plan_confidence_capture((4747,2048,2048),32,workspace_bytes=256*1024**2)
    assert plan.workers == 32
    assert plan.workspace_bytes == max(plan.initialization_peak_bytes,
        plan.fixed_bytes + plan.consumer_bytes + plan.workers*plan.worker_bytes)
    assert plan.workspace_bytes <= plan.workspace_limit_bytes
    with pytest.raises(MemoryError):
        plan_confidence_capture((3,3064,3022),32,workspace_bytes=1024**2)
    with pytest.raises(ValueError):
        plan_confidence_capture((1,65535,65535),1,workspace_bytes=2**40,block_size=65535)
    with pytest.raises(MemoryError,match='metadata initialization'):
        plan_confidence_capture((60000,1,1),1,workspace_bytes=4*1024**2)


def test_explicit_grid_or_source_mismatch_cannot_upgrade_admitted_workspace(tmp_path):
    mask,scores,active,boxes=fixture()
    plan=plan_confidence_capture(scores.shape,4,workspace_bytes=32*1024**2,block_size=128)
    with confidence_capture_resources(plan):
        reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
        with pytest.raises(ValueError,match='geometry/grid'):
            storage.write_blocks(tmp_path/'changed',scores.shape,reader,layer_key='k',model_name='m',provenance={},block_size=8)
        with pytest.raises(ValueError,match='geometry differs'):
            _MaskedNativeScoreReader(mask[:,:2,:2],scores,active,boxes)
    assert not (tmp_path/'changed'/'metadata.json').exists()


def test_bounds_snapshot_and_thread_local_scope_are_independent():
    mask,scores,active,boxes=fixture()
    plan=plan_confidence_capture(scores.shape,3,workspace_bytes=64*1024**2)
    with confidence_capture_resources(plan):
        reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
        boxes[:] = -1
        active[:] = False
        assert reader.active.any()
        assert (reader.boxes[reader.active] >= 0).all()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(current_confidence_capture_plan).result() is None
    assert current_confidence_capture_plan() is None
    with pytest.raises(ValueError,match='admission proof'):
        with confidence_capture_resources(replace(plan,workers=plan.workers+1)):
            pass


def test_ordered_source_reader_failure_joins_all_running_reads_and_never_publishes(tmp_path):
    mask,scores,active,boxes=fixture()
    plan=plan_confidence_capture(scores.shape,3,workspace_bytes=64*1024**2)
    with confidence_capture_resources(plan):
        reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
    real=reader.encode_frame
    lock=threading.Lock()
    running=0
    maximum=0
    def encoded(z,block):
        nonlocal running,maximum
        with lock:
            running+=1;maximum=max(maximum,running)
        try:
            time.sleep(.01)
            if z==1:raise OSError('injected frame failure')
            return real(z,block)
        finally:
            with lock:running-=1
    with mock.patch.object(reader,'encode_frame',side_effect=encoded):
        with pytest.raises(OSError,match='injected frame failure'):
            storage.write_blocks(tmp_path/'failed',scores.shape,reader,layer_key='k',model_name='m',provenance={})
    assert running==0 and maximum<=plan.workers
    assert not (tmp_path/'failed'/'metadata.json').exists()
    assert not list((tmp_path/'failed').glob('*.partial'))


def test_stage_capacity_failure_preserves_original_error_and_settles_workers(tmp_path):
    mask,scores,active,boxes=fixture()
    plan=plan_confidence_capture(scores.shape,3,workspace_bytes=64*1024**2)
    with confidence_capture_resources(plan):reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
    with pytest.raises(storage.ConfidenceStageLimit):
        storage.write_blocks(tmp_path/'limited',scores.shape,reader,layer_key='k',model_name='m',provenance={},max_numeric_bytes=1)
    assert list((tmp_path/'limited').iterdir())==[]
