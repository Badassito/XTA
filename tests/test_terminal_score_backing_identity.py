"""Real-FD mapped-identity checks; file proxies are not Linux memfd evidence."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

from XTA import confidence_evidence as evidence
from XTA import runtime
from XTA.confidence_capture import confidence_capture_resources, plan_confidence_capture


SHAPE = (1, 2, 4)
HAS_LINUX_MEMFD = (sys.platform.startswith('linux') and hasattr(os, 'memfd_create')
                   and Path('/proc/self/fd').is_dir())


@contextmanager
def registered_mapping(path, *, shape=SHAPE, mode='r+', fd=None):
    """Mirror owner registration with a real FD; ordinary files are test proxies."""
    descriptor = os.open(path, os.O_RDWR) if fd is None else fd
    key = f'registered-file-proxy:{uuid.uuid4()}'
    array = None
    runtime._register_memfd_owner(key, descriptor, 'terminal score backing identity test')
    try:
        array = np.memmap(path, dtype=np.uint8, mode=mode, shape=shape)
        array._workspace_memfd_owner_key = key
        array._workspace_memfd_owner_fd = descriptor
        yield array, key, descriptor
    finally:
        if array is not None and not array._mmap.closed:
            array._mmap.close()
        runtime._release_memfd_owner_key(key)


@pytest.fixture(autouse=True)
def enabled_capture(monkeypatch):
    monkeypatch.setattr(evidence, '_ENABLED', True)
    monkeypatch.setattr(evidence, '_DEFER_PROJECTION', True)


def prepare(mask, scores, **kwargs):
    with confidence_capture_resources(plan_confidence_capture(SHAPE, 1)):
        return evidence.prepare_terminal_confidence_scores(mask, scores, **kwargs)


def create_file(path, *, extra=0):
    path.write_bytes(bytes([0]) * (int(np.prod(SHAPE)) + extra))
    return path


def seed(mask, scores):
    mask[:] = [[[255, 0, 1, 0], [1, 1, 0, 255]]]
    scores[:] = [[[0, 200, 9, 211], [13, 0, 251, 255]]]


def assert_declines_unchanged(mask, scores, **kwargs):
    mask_before, score_before = mask.copy(), scores.copy()
    assert prepare(mask, scores, **kwargs) is None
    assert np.array_equal(mask, mask_before)
    assert np.array_equal(scores, score_before)
    assert scores.flags.writeable


def test_registered_file_proxy_real_fstat_accepts_distinct_full_roots(tmp_path):
    """Cross-platform proxy exercises real registry pin/dup/fstat, not memfd creation."""
    with registered_mapping(create_file(tmp_path / 'mask.u8')) as (mask, mk, mfd):
        with registered_mapping(create_file(tmp_path / 'score.u8')) as (scores, sk, sfd):
            seed(mask, scores)
            mask_before = mask.copy()
            expected = np.where(mask != 0, scores, np.uint8(0))
            pins_before = set(runtime._MEMFD_PROBE_PINS)
            proofs = runtime.capture_memfd_owner_proofs({'mask': mask, 'scores': scores})
            assert len(proofs) == 2
            for proof, descriptor in zip(proofs, (mfd, sfd)):
                info = os.fstat(descriptor)
                assert proof[2:] == (info.st_dev, info.st_ino, info.st_size)
            assert proofs[0][2:4] != proofs[1][2:4]
            prepared = prepare(mask, scores)
            assert prepared is not None and prepared.scores is scores
            assert prepared.source_bytes == scores.nbytes
            assert np.array_equal(scores, expected) and np.array_equal(mask, mask_before)
            assert not scores.flags.writeable
            assert set(runtime._MEMFD_PROBE_PINS) == pins_before
            assert mk in runtime._MEMFD_OWNERS and sk in runtime._MEMFD_OWNERS
    assert mk not in runtime._MEMFD_OWNERS and sk not in runtime._MEMFD_OWNERS
    with pytest.raises(OSError):
        os.fstat(mfd)
    with pytest.raises(OSError):
        os.fstat(sfd)


def test_registered_file_proxy_same_physical_file_two_maps_rejected(tmp_path):
    path = create_file(tmp_path / 'same.u8')
    with registered_mapping(path) as (mask, _mk, _mfd):
        with registered_mapping(path) as (scores, _sk, _sfd):
            mask[:] = 91
            assert not np.may_share_memory(mask, scores)
            proofs = runtime.capture_memfd_owner_proofs({'mask': mask, 'scores': scores})
            assert len(proofs) == 2 and proofs[0][2:4] == proofs[1][2:4]
            assert_declines_unchanged(mask, scores)


@pytest.mark.parametrize('source', ('mask', 'scores'))
def test_registered_file_proxy_separate_map_metadata_alias_rejected(tmp_path, source):
    mask_path = create_file(tmp_path / 'mask.u8')
    score_path = create_file(tmp_path / 'score.u8')
    with registered_mapping(mask_path) as (mask, _mk, _mfd):
        with registered_mapping(score_path) as (scores, _sk, _sfd):
            seed(mask, scores)
            with registered_mapping(mask_path if source == 'mask' else score_path) as (other, _ok, _ofd):
                active = other.reshape(-1)[:1]
                assert not np.may_share_memory(active, mask)
                assert not np.may_share_memory(active, scores)
                assert_declines_unchanged(mask, scores, known_slice_any=active,
                    known_slice_bboxes=np.array([[0, 2, 0, 4]], np.int64))


def test_registered_file_proxy_extent_mismatch_rejected(tmp_path):
    mask = np.ones(SHAPE, np.uint8)
    with registered_mapping(create_file(tmp_path / 'score.u8', extra=1)) as (scores, _key, _fd):
        scores[:] = 91
        assert runtime.capture_memfd_owner_proofs({'scores': scores}) == ()
        assert_declines_unchanged(mask, scores)


def test_registered_file_proxy_full_root_view_rejected(tmp_path):
    mask = np.ones(SHAPE, np.uint8)
    with registered_mapping(create_file(tmp_path / 'score.u8')) as (scores, _key, _fd):
        scores[:] = 91
        assert len(runtime.capture_memfd_owner_proofs({'scores': scores})) == 1
        assert_declines_unchanged(mask, scores.view())


def test_registered_file_proxy_copy_on_write_mapping_rejected(tmp_path):
    mask = np.ones(SHAPE, np.uint8)
    with registered_mapping(create_file(tmp_path / 'score.u8'), mode='c') as (scores, _key, _fd):
        scores[:] = 91
        assert runtime.capture_memfd_owner_proofs({'scores': scores}) == ()
        assert_declines_unchanged(mask, scores)


def test_registered_file_proxy_retired_owner_fd_rejected(tmp_path):
    mask = np.ones(SHAPE, np.uint8)
    with registered_mapping(create_file(tmp_path / 'score.u8')) as (scores, key, descriptor):
        scores[:] = 91
        assert len(runtime.capture_memfd_owner_proofs({'scores': scores})) == 1
        assert runtime._release_memfd_owner_key(key)
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert runtime.capture_memfd_owner_proofs({'scores': scores}) == ()
        assert_declines_unchanged(mask, scores)


@pytest.mark.parametrize('shape,bbox,dtype', (
    ((2, 2, 300019), (0, 2, 11, 300017), np.bool_),
    ((2, 2500, 137), (3, 2499, 7, 131), np.uint8),
), ids=('wide-column-stripes', 'strided-row-slabs-with-tail'))
def test_prepared_reader_masking_updates_original_score_roi_at_scratch_boundaries(shape, bbox, dtype):
    """Scoped views must be written directly; flattening their ROI could copy."""
    mask = np.zeros(shape, dtype=dtype)
    scores = np.empty(shape, np.uint8)
    scores.fill(203)
    y0, y1, x0, x1 = bbox
    mask[0, y0:y1:2, x0:x1:3] = True if dtype == np.bool_ else 255
    scores[0, y0, x0] = 0
    mask_before, scores_before = mask.copy(), scores.copy()
    expected = scores.copy()
    expected_roi = expected[0, y0:y1, x0:x1]
    expected_roi[mask[0, y0:y1, x0:x1] == 0] = 0
    active = np.array([True, False])
    boxes = np.array([bbox, (0, 0, 0, 0)], np.int64)
    plan = plan_confidence_capture(shape, 1, workspace_bytes=64 * 1024**2)
    with confidence_capture_resources(plan):
        prepared = evidence.prepare_terminal_confidence_scores(mask, scores,
            known_slice_any=active, known_slice_bboxes=boxes)
    assert prepared is not None and prepared.reader is not None
    assert prepared.reader.mask is prepared.reader.scores
    assert np.array_equal(scores, expected) and np.array_equal(mask, mask_before)
    assert np.array_equal(scores[1], scores_before[1])
    assert scores[0, 0, 0] == scores_before[0, 0, 0] == 203
    assert not scores.flags.writeable
    assert len(list(prepared.reader.iter_crops(0))) == 1
    assert not list(prepared.reader.iter_crops(1))


@pytest.mark.skipif(not HAS_LINUX_MEMFD, reason='Actual Linux memfd/procfs required')
def test_actual_linux_memfd_distinct_backings_and_alias_map(tmp_path):
    """Separate Linux-only evidence; ordinary-file proxy success does not satisfy it."""
    mask_fd = os.memfd_create('terminal-mask', int(getattr(os, 'MFD_CLOEXEC', 0)))
    score_fd = os.memfd_create('terminal-score', int(getattr(os, 'MFD_CLOEXEC', 0)))
    os.ftruncate(mask_fd, int(np.prod(SHAPE)))
    os.ftruncate(score_fd, int(np.prod(SHAPE)))
    mask_path, score_path = Path(f'/proc/self/fd/{mask_fd}'), Path(f'/proc/self/fd/{score_fd}')
    with registered_mapping(mask_path, fd=mask_fd) as (mask, _mk, _mfd):
        with registered_mapping(score_path, fd=score_fd) as (scores, _sk, _sfd):
            seed(mask, scores)
            alias_fd = os.dup(mask_fd)
            with registered_mapping(Path(f'/proc/self/fd/{alias_fd}'), fd=alias_fd) as (alias, _ak, _afd):
                assert_declines_unchanged(mask, alias)
            expected = np.where(mask != 0, scores, np.uint8(0))
            prepared = prepare(mask, scores)
            assert prepared is not None and np.array_equal(scores, expected)
