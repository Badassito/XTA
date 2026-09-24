"""Immutable staging, backpressure and joined failure of retained confidence."""
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import threading
from unittest import mock

import numpy as np
import pytest

from XTA import confidence_evidence as evidence, geometry
from XTA.confidence_publication import ConfidencePublicationQueue, copy_staged_blocks


@pytest.fixture(autouse=True)
def reset_publication():
    evidence.configure_confidence_evidence(None, enabled=False)
    yield
    evidence.shutdown_confidence_publication(raise_errors=False)
    evidence.configure_confidence_evidence(None, enabled=False)


def write(path, values, reader=None, **kwargs):
    return evidence.write_block_confidence_evidence(path, values.shape, reader or (lambda z:values[z]),
        model_name='m',layer_key='k',**kwargs)


def test_cropped_reader_preserves_global_blocks_compressed_bytes_and_unknowns(tmp_path):
    values = np.zeros((3, 271, 315),np.uint8)
    rng = np.random.default_rng(414)
    values[1,17:256,37:300] = rng.integers(0,256,(239,263),dtype=np.uint8)
    values[1,110:140,110:140] = 0
    class Reader:
        known_z_bounds = (1,2)
        def iter_crops(self,z):
            assert z == 1
            yield 17,256,37,300,values[z,17:256,37:300]
    reference = write(tmp_path/'full', values)
    metrics = {}
    cropped = write(tmp_path/'cropped',values,Reader(),metrics=metrics)
    for name in ('scores.u8.zlib','index.bin'):
        assert (reference.path/name).read_bytes() == (cropped.path/name).read_bytes()
    with cropped.reader() as reader:
        actual,known = reader(0,3)
    np.testing.assert_array_equal(actual,values)
    np.testing.assert_array_equal(known,values>0)
    assert metrics['block_count'] == reference.metadata['block_count']
    assert metrics['compression_seconds'] >= 0 and metrics['storage_seconds'] >= 0


def test_cropped_reader_rejects_duplicate_block_cells(tmp_path):
    values = np.ones((1,20,20),np.uint8)
    class Reader:
        def iter_crops(self,z):
            yield 0,5,0,5,values[z,:5,:5]
            yield 10,15,10,15,values[z,10:15,10:15]
    with pytest.raises(ValueError,match='block cells'):
        write(tmp_path/'bad',values,Reader())
    assert not (tmp_path/'bad/metadata.json').exists()


def test_background_publication_owns_compressed_snapshot_not_retired_inputs(tmp_path):
    evidence.configure_confidence_evidence(tmp_path/'out',enabled=True,defer_projection=True,background_publication=True)
    scores = np.full((3,9,11),173,np.uint8)
    mask = np.zeros(scores.shape,np.uint8); mask[:,2:7,3:8] = 1
    scores[1,3,4] = 0
    expected = np.where(mask,scores,0).copy()
    view = geometry.get_view_infos(*scores.shape,cartesian_views=('transverse',))[0]
    entered,finish = threading.Event(),threading.Event()
    original_copy = copy_staged_blocks
    def blocked(*args,**kwargs):
        entered.set()
        assert finish.wait(10)
        return original_copy(*args,**kwargs)
    try:
        with mock.patch('XTA.confidence_publication.copy_staged_blocks',side_effect=blocked):
            future = evidence.capture_prediction_confidence(mask,scores,view=view,model_name='m',temp_dir=tmp_path)
            assert isinstance(future,Future) and entered.wait(5)
            assert not (tmp_path/'out/reconciliation_evidence/manifest.json').exists()
            # Simulate workspace reuse while the output filesystem is stalled.
            scores[:] = 0; mask[:] = 0
            finish.set()
            reference = future.result(timeout=10)
        state = evidence.drain_confidence_publication()
        assert state['completed'] == 1 and state['reserved_numeric_bytes'] == 0
        with reference.native_reader() as reader:
            actual,known = reader(0,3)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(known,expected>0)
        assert evidence.lookup_confidence_evidence(reference.layer_key,'m') == reference
        assert not list((tmp_path/'confidence_publication').iterdir())
    finally:
        finish.set()


def test_stage_credit_blocks_next_dense_snapshot_until_publication_retires(tmp_path):
    queue = ConfidencePublicationQueue(max_pending_bytes=1024,stage_bytes=1024,workers=1)
    entered,release,second_started = threading.Event(),threading.Event(),threading.Event()
    def writer(path,limit):
        (path/'x').write_bytes(b'x')
    def first_publisher(path,waited):
        entered.set()
        assert release.wait(10)
    def second_writer(path,limit):
        second_started.set(); writer(path,limit)
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            first = queue.stage_and_submit(tmp_path,writer,first_publisher)
            assert entered.wait(5)
            second = executor.submit(queue.stage_and_submit,tmp_path,second_writer,lambda *_:None)
            assert not second_started.wait(.1)
            assert not first.cancel()
            release.set(); first.result(timeout=10)
            second.result(timeout=10).result(timeout=10)
            state = queue.drain()
            assert state['peak_reserved_numeric_bytes'] == 1024
            assert state['reserved_numeric_bytes'] == 0
        finally:
            release.set(); queue.close(raise_errors=False)
    assert not list(tmp_path.iterdir())


def test_queued_compressed_owner_cannot_be_cancelled_before_cleanup(tmp_path):
    queue = ConfidencePublicationQueue(max_pending_bytes=2048,stage_bytes=1024,workers=1)
    entered, release = threading.Event(),threading.Event()
    def writer(path,limit):
        (path/'x').write_bytes(b'x')
    def blocked(*_):
        entered.set()
        assert release.wait(10)
    try:
        first = queue.stage_and_submit(tmp_path,writer,blocked)
        assert entered.wait(5)
        queued = queue.stage_and_submit(tmp_path,writer,lambda *_:None)
        assert not queued.cancel()
        release.set()
        first.result(timeout=10); queued.result(timeout=10)
        assert queue.drain()['reserved_numeric_bytes'] == 0
    finally:
        release.set(); queue.close(raise_errors=False)
    assert not list(tmp_path.iterdir())


def test_oversized_stage_uses_direct_streaming_with_identical_values(tmp_path):
    evidence.configure_confidence_evidence(tmp_path/'out',enabled=True,defer_projection=True)
    evidence._PUBLICATION = ConfidencePublicationQueue(max_pending_bytes=128,stage_bytes=128,workers=1)
    values = np.random.default_rng(77).integers(0,256,(2,19,23),dtype=np.uint8)
    original = values.copy()
    view = geometry.get_view_infos(*values.shape,cartesian_views=('transverse',))[0]
    reference = evidence.capture_prediction_confidence(values>0,values,view=view,model_name='m',temp_dir=tmp_path)
    assert not isinstance(reference,Future)
    with reference.native_reader() as reader:
        np.testing.assert_array_equal(reader(0,2)[0],original)
    np.testing.assert_array_equal(values,original)
    state = evidence.drain_confidence_publication()
    assert state['submitted'] == 0 and state['synchronous_large_stage_fallbacks'] == 1
    assert state['reserved_numeric_bytes'] == 0
    assert not list((tmp_path/'confidence_publication').iterdir())


def test_failed_background_copy_is_joined_and_wakes_future_producers(tmp_path):
    evidence.configure_confidence_evidence(tmp_path/'out',enabled=True,defer_projection=True,background_publication=True)
    values = np.full((2,7,9),191,np.uint8)
    view = geometry.get_view_infos(*values.shape,cartesian_views=('transverse',))[0]
    failure = OSError('injected background copy failure')
    with mock.patch('XTA.confidence_publication.copy_staged_blocks',side_effect=failure):
        future = evidence.capture_prediction_confidence(values>0,values,view=view,model_name='m',temp_dir=tmp_path)
        with pytest.raises(OSError,match='background copy'):
            future.result(timeout=10)
    with pytest.raises(OSError,match='background copy'):
        evidence.drain_confidence_publication()
    with pytest.raises(OSError,match='background copy'):
        evidence.capture_prediction_confidence(values>0,values,view=view,model_name='other',temp_dir=tmp_path)
    assert not list((tmp_path/'confidence_publication').iterdir())
    assert not (tmp_path/'out/reconciliation_evidence/manifest.json').exists()


def test_corrupt_stage_cleans_destination_and_existing_destination_is_preserved(tmp_path):
    values = np.full((1,8,10),101,np.uint8)
    reference = write(tmp_path/'stage',values)
    (reference.path/'scores.u8.zlib').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='changed'):
        copy_staged_blocks(reference.path,tmp_path/'destination')
    assert not (tmp_path/'destination').exists()
    existing = tmp_path/'existing'; existing.mkdir(); (existing/'sentinel').write_text('original')
    with pytest.raises(FileExistsError):
        copy_staged_blocks(reference.path,existing)
    assert (existing/'sentinel').read_text() == 'original'


def test_manifest_failure_does_not_register_an_unpublished_reference(tmp_path):
    evidence.configure_confidence_evidence(tmp_path/'out',enabled=True)
    values = np.full((1,8,10),101,np.uint8)
    reference = write(tmp_path/'out/reconciliation_evidence/stage',values)
    with mock.patch('XTA.confidence_evidence._write_json_atomic',side_effect=OSError('manifest write')):
        with pytest.raises(OSError,match='manifest write'):
            evidence._register_confidence_reference(reference)
    assert evidence.lookup_confidence_evidence('k','m') is None
