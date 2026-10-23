"""Numeric RAM-stage ownership, conservative admission and explicit disk fallback.

Portable lifecycle mocks use ordinary owned FDs, not actual memfd IO. The separate
Linux-only test executes the real memfd/proc alias path and is skipped on Windows.
"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading

import numpy as np
import pytest

from XTA import confidence_publication as publication
from XTA import confidence_evidence as evidence


def assert_closed(fds):
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.fixture
def portable_memfd_mock(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(publication, '_memfd_staging_supported', lambda:True)
    monkeypatch.setattr(publication, '_physical_stage_headroom', lambda:1024**3)
    def create(name, flags):
        fd = os.open(tmp_path/f'owned_fd_{len(created)}', os.O_CREAT|os.O_RDWR|os.O_EXCL, 0o600)
        created.append(fd)
        return fd
    monkeypatch.setattr(publication.os, 'memfd_create', create, raising=False)
    # This only tests resource lifetime and failure handling. It does not claim
    # that a portable placeholder file stores its bytes in a Linux memfd.
    monkeypatch.setattr(publication, '_link_memfd_numeric', lambda path,fd:Path(path).touch())
    yield created
    assert_closed(created)
    assert publication._RAM_STAGE_PROMISED_BYTES == 0


def test_ram_keepalive_survives_publication_and_shutdown_until_reader_exits(tmp_path, portable_memfd_mock):
    fds = portable_memfd_mock
    queue = publication.ConfidencePublicationQueue(max_pending_bytes=1024, stage_bytes=1024, workers=1)
    entered, release = threading.Event(), threading.Event()
    metrics = {}
    def writer(path, limit):
        os.write(fds[0], b'payload')
        os.write(fds[1], b'index')
    def publisher(path, waited):
        with os.fdopen(os.dup(fds[0]), 'rb') as reader:
            reader.seek(0)
            assert reader.read() == b'payload'
            entered.set()
            assert release.wait(10)
    closer = ThreadPoolExecutor(max_workers=1)
    try:
        future = queue.stage_and_submit(tmp_path/'stages', writer, publisher, metrics=metrics)
        assert entered.wait(10)
        state = queue.snapshot()
        assert state['active_memory_stages'] == 1 and state['active_disk_stages'] == 0
        assert state['ram_numeric_bytes'] == 12
        assert state['ram_promised_bytes'] == 1024+publication._STAGE_PAGE_ALLOWANCE
        assert state['reserved_numeric_bytes'] == 1024
        assert metrics['stage_backing'] == 'memory'
        assert metrics['stage_backing_backend'] == 'linux_memfd'
        assert metrics['stage_backing_decline_reason'] is None
        closing = closer.submit(queue.close)
        assert not closing.done()
        assert all(os.fstat(fd).st_size > 0 for fd in fds)
        release.set()
        future.result(timeout=10)
        closing.result(timeout=10)
        assert_closed(fds)
        assert queue.snapshot()['ram_promised_bytes'] == 0
        assert queue.snapshot()['reserved_numeric_bytes'] == 0
        assert not list((tmp_path/'stages').iterdir())
    finally:
        release.set()
        closer.shutdown(wait=True)
        queue.close(raise_errors=False)


@pytest.mark.parametrize('failure', ['first_fd', 'second_fd', 'second_link'])
def test_partial_memfd_setup_closes_every_fd_and_falls_back_explicitly(tmp_path, monkeypatch, failure):
    fds = []
    monkeypatch.setattr(publication, '_memfd_staging_supported', lambda:True)
    monkeypatch.setattr(publication, '_physical_stage_headroom', lambda:1024**3)
    def create(name, flags):
        if failure == 'first_fd' or (failure == 'second_fd' and fds):
            raise OSError('injected memfd refusal')
        fd = os.open(tmp_path/f'fd_{len(fds)}', os.O_CREAT|os.O_RDWR|os.O_EXCL, 0o600)
        fds.append(fd)
        return fd
    def link(path, fd):
        if failure == 'second_link' and len(fds) == 2:
            raise PermissionError('injected proc alias refusal')
        Path(path).touch()
    monkeypatch.setattr(publication.os, 'memfd_create', create, raising=False)
    monkeypatch.setattr(publication, '_link_memfd_numeric', link)
    queue = publication.ConfidencePublicationQueue(max_pending_bytes=1024, stage_bytes=1024, workers=1)
    metrics = {}
    def writer(path, limit):
        assert not (path/'scores.u8.zlib.partial').exists()
        assert not (path/'index.bin.partial').exists()
        (path/'ordinary').write_bytes(b'unchanged fallback')
    try:
        queue.stage_and_submit(tmp_path/'stages', writer, lambda *_:None, metrics=metrics).result(timeout=10)
        assert metrics['stage_backing'] == 'disk'
        assert metrics['stage_backing_decline_reason'].startswith('memfd_setup_unavailable:')
        assert queue.snapshot()['active_disk_stages'] == 0
        assert queue.snapshot()['reserved_numeric_bytes'] == 0
        assert publication._RAM_STAGE_PROMISED_BYTES == 0
        assert_closed(fds)
    finally:
        queue.close(raise_errors=False)


@pytest.mark.parametrize('failure', ['writer', 'publisher', 'overflow'])
def test_failure_and_overflow_retire_ram_owners_and_numeric_credit(tmp_path, portable_memfd_mock, failure):
    fds = portable_memfd_mock
    queue = publication.ConfidencePublicationQueue(max_pending_bytes=1024, stage_bytes=1024, workers=1)
    def writer(path, limit):
        os.write(fds[0], b'x'*(limit+1 if failure == 'overflow' else 17))
        if failure == 'writer':
            raise OSError('injected writer failure')
    def publisher(path, waited):
        with os.fdopen(os.dup(fds[0]), 'rb'):
            raise OSError('injected publisher failure')
    try:
        if failure == 'writer':
            with pytest.raises(OSError, match='writer failure'):
                queue.stage_and_submit(tmp_path/'stages', writer, publisher)
        elif failure == 'publisher':
            future = queue.stage_and_submit(tmp_path/'stages', writer, publisher)
            with pytest.raises(OSError, match='publisher failure'):
                future.result(timeout=10)
        else:
            assert queue.stage_and_submit(tmp_path/'stages', writer, publisher) is None
            assert queue.snapshot()['synchronous_large_stage_fallbacks'] == 1
        assert_closed(fds)
        assert queue.snapshot()['ram_promised_bytes'] == 0
        assert queue.snapshot()['active_memory_stages'] == 0
        assert queue.snapshot()['reserved_numeric_bytes'] == 0
        assert not list((tmp_path/'stages').iterdir())
    finally:
        queue.close(raise_errors=False)


def test_global_ram_promises_prevent_two_untouched_slots_spending_same_headroom(tmp_path, portable_memfd_mock, monkeypatch):
    charge = 1024+publication._STAGE_PAGE_ALLOWANCE
    monkeypatch.setattr(publication, '_physical_stage_headroom', lambda:charge+charge//2)
    for name in ('one', 'two', 'three'):
        (tmp_path/name).mkdir()
    first = publication._create_numeric_stage_backing(tmp_path/'one', 1024)
    second = publication._create_numeric_stage_backing(tmp_path/'two', 1024)
    try:
        assert first.memory and not second.memory
        assert second.decline_reason == 'physical_headroom_declined'
        first.close()
        third = publication._create_numeric_stage_backing(tmp_path/'three', 1024)
        try:
            assert third.memory
        finally:
            third.close()
    finally:
        first.close(); second.close()


def test_owner_birth_failure_rolls_back_prebirth_ram_and_numeric_promises(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, '_memfd_staging_supported', lambda:True)
    monkeypatch.setattr(publication, '_physical_stage_headroom', lambda:1024**3)
    def failed_owner(**kwargs):
        raise MemoryError('injected numeric owner allocation failure')
    monkeypatch.setattr(publication, '_NumericStageBacking', failed_owner)
    queue = publication.ConfidencePublicationQueue(max_pending_bytes=1024, stage_bytes=1024, workers=1)
    try:
        with pytest.raises(MemoryError, match='owner allocation'):
            queue.stage_and_submit(tmp_path/'stages', lambda *_:None, lambda *_:None)
        assert publication._RAM_STAGE_PROMISED_BYTES == 0
        assert queue.snapshot()['reserved_numeric_bytes'] == 0
        assert not list((tmp_path/'stages').iterdir())
    finally:
        queue.close(raise_errors=False)


@pytest.mark.parametrize('decline', ['platform', 'headroom'])
def test_real_disk_fallback_keeps_numeric_bytes_and_reports_reason(tmp_path, monkeypatch, decline):
    monkeypatch.setattr(publication, '_memfd_staging_supported', lambda:decline != 'platform')
    monkeypatch.setattr(publication, '_physical_stage_headroom', lambda:0)
    queue = publication.ConfidencePublicationQueue(max_pending_bytes=4096, stage_bytes=4096, workers=1)
    values = np.full((2, 17, 19), 173, np.uint8)
    metrics = {}
    def writer(path, limit):
        return evidence.write_block_confidence_evidence(path, values.shape, lambda z:values[z],
            model_name='m', layer_key='k', max_numeric_bytes=limit)
    try:
        result = queue.stage_and_submit(tmp_path/'stages', writer,
            lambda path,waited:publication.copy_staged_blocks(path, tmp_path/'final'),
            metrics=metrics).result(timeout=10)
        assert metrics['stage_backing'] == 'disk'
        assert metrics['stage_backing_decline_reason'] == (
            'memfd_platform_unavailable' if decline == 'platform' else 'physical_headroom_declined')
        with result.reader() as reader:
            np.testing.assert_array_equal(reader(0, 2)[0], values)
    finally:
        queue.close(raise_errors=False)


@pytest.mark.skipif(not publication._memfd_staging_supported(), reason='Actual Linux memfd/proc IO unavailable')
def test_actual_linux_memfd_path_io_exports_identical_artifact_once(tmp_path, monkeypatch):
    original_create = os.memfd_create
    fds = []
    def create(name, flags):
        fd = original_create(name, flags)
        fds.append(fd)
        return fd
    monkeypatch.setattr(publication.os, 'memfd_create', create)
    monkeypatch.setattr(publication, '_physical_stage_headroom', lambda:1024**3)
    queue = publication.ConfidencePublicationQueue(max_pending_bytes=65536, stage_bytes=65536, workers=1)
    values = np.random.default_rng(160591).integers(0, 256, (3, 31, 43), np.uint8)
    metrics = {}
    def writer(path, limit):
        assert (path/'scores.u8.zlib.partial').is_symlink()
        assert (path/'index.bin.partial').is_symlink()
        return evidence.write_block_confidence_evidence(path, values.shape, lambda z:values[z],
            model_name='m', layer_key='k', max_numeric_bytes=limit)
    def publisher(path, waited):
        assert (path/'scores.u8.zlib').is_symlink()
        assert (path/'index.bin').is_symlink()
        assert all('memfd:' in os.readlink(f'/proc/self/fd/{fd}') for fd in fds)
        return publication.copy_staged_blocks(path, tmp_path/'final')
    try:
        reference = evidence.write_block_confidence_evidence(tmp_path/'reference', values.shape,
            lambda z:values[z], model_name='m', layer_key='k')
        result = queue.stage_and_submit(tmp_path/'stages', writer, publisher,
            metrics=metrics).result(timeout=10)
        assert metrics['stage_backing'] == 'memory'
        assert_closed(fds)
        for name in ('scores.u8.zlib', 'index.bin', 'metadata.json'):
            assert (result.path/name).read_bytes() == (reference.path/name).read_bytes()
            assert not (result.path/name).is_symlink()
        assert queue.snapshot()['reserved_numeric_bytes'] == queue.snapshot()['ram_promised_bytes'] == 0
    finally:
        queue.close(raise_errors=False)
