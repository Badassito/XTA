"""A reader queued before prefix publication can use that prefix after wakeup."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import threading

import pytest

from XTA import artifact_archive as artifacts


@pytest.mark.parametrize('fail_append', [False, True])
def test_queued_readers_recheck_published_prefix_without_exposing_new_members(tmp_path, monkeypatch, fail_append):
    archive = tmp_path/'sam-artifacts.tar'
    old = artifacts.reference(archive, 'existing/old')
    new = artifacts.reference(archive, 'pending/new')
    artifacts.append_members(archive, {'existing/old': b'committed'})
    indexed, publish, appending, finish = [threading.Event() for _ in range(4)]
    waiting = {name: threading.Event() for name in ('old', 'prefix', 'new', 'full')}
    local = threading.local()
    original_index, original_write = artifacts._index, artifacts._write_member
    original_wait = artifacts._READERS_CHANGED.wait

    def index(path, stream=None):
        result = original_index(path, stream)
        if stream is not None:
            indexed.set()
            if not publish.wait(5):
                raise TimeoutError('prefix publication was not released')
        return result

    def write(stream, name, source):
        if name.endswith('/pending/new'):
            appending.set()
            if not finish.wait(5):
                raise TimeoutError('append was not released')
            if fail_append:
                raise OSError('injected append failure')
        return original_write(stream, name, source)

    def wait(timeout=None):
        waiting[local.reader].set()
        return original_wait(timeout)

    def read(name, function, *args):
        local.reader = name
        return function(*args)

    monkeypatch.setattr(artifacts, '_index', index)
    monkeypatch.setattr(artifacts, '_write_member', write)
    monkeypatch.setattr(artifacts._READERS_CHANGED, 'wait', wait)
    with ThreadPoolExecutor(max_workers=5) as pool:
        writer = pool.submit(artifacts.append_members, archive, {'pending/new': b'later'})
        try:
            assert indexed.wait(3)
            assert str(archive) not in artifacts._APPEND_SNAPSHOTS
            readers = {
                'old': pool.submit(read, 'old', artifacts.read_member, old),
                'prefix': pool.submit(read, 'prefix', artifacts.list_members, archive, 'existing'),
                'new': pool.submit(read, 'new', artifacts.read_member, new),
                'full': pool.submit(read, 'full', artifacts.list_members, archive)}
            assert all(event.wait(3) for event in waiting.values())
            publish.set()
            assert appending.wait(3)
            assert readers['old'].result(timeout=2) == b'committed'
            assert readers['prefix'].result(timeout=2) == ['existing/old']
            assert not writer.done()
            for name in ('new', 'full'):
                with pytest.raises(FutureTimeout):
                    readers[name].result(timeout=.05)
        finally:
            publish.set()
            finish.set()
        if fail_append:
            with pytest.raises(OSError, match='injected append failure'):
                writer.result(timeout=3)
            with pytest.raises(FileNotFoundError):
                readers['new'].result(timeout=3)
            assert readers['full'].result(timeout=3) == ['existing/old']
        else:
            writer.result(timeout=3)
            assert readers['new'].result(timeout=3) == b'later'
            assert readers['full'].result(timeout=3) == ['existing/old', 'pending/new']
    assert str(archive) not in artifacts._APPEND_SNAPSHOTS


def test_queued_reader_rejects_inode_change_when_snapshot_becomes_available(tmp_path, monkeypatch):
    archive, replacement = tmp_path/'sam-artifacts.tar', tmp_path/'replacement.tar'
    artifacts.append_members(archive, {'old': b'committed'})
    replacement.write_bytes(archive.read_bytes())
    indexed, publish, waiting, appending, finish = [threading.Event() for _ in range(5)]
    original_index, original_write = artifacts._index, artifacts._write_member
    original_wait = artifacts._READERS_CHANGED.wait

    def index(path, stream=None):
        result = original_index(path, stream)
        if stream is not None:
            indexed.set()
            if not publish.wait(5):
                raise TimeoutError('prefix publication was not released')
        return result

    def write(stream, name, source):
        if name.endswith('/new'):
            appending.set()
            if not finish.wait(5):
                raise TimeoutError('append was not released')
        return original_write(stream, name, source)

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    real_open = open
    def replaced_open(path, mode):
        return real_open(replacement if mode == 'rb' else path, mode)

    monkeypatch.setattr(artifacts, '_index', index)
    monkeypatch.setattr(artifacts, '_write_member', write)
    monkeypatch.setattr(artifacts._READERS_CHANGED, 'wait', wait)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(artifacts.append_members, archive, {'new': b'later'})
        try:
            assert indexed.wait(3)
            reader = pool.submit(artifacts.read_member, artifacts.reference(archive, 'old'))
            assert waiting.wait(3)
            monkeypatch.setattr(artifacts, 'open', replaced_open, raising=False)
            publish.set()
            assert appending.wait(3)
            with pytest.raises(artifacts.ArchiveError, match='prefix changed'):
                reader.result(timeout=2)
            assert not writer.done()
        finally:
            publish.set()
            finish.set()
        writer.result(timeout=3)


def test_interrupted_reader_returns_locally_acquired_archive_lock(tmp_path, monkeypatch):
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'old': b'committed'})
    reference = artifacts.reference(archive, 'old')
    original_writer_lock = artifacts._writer_lock

    @contextmanager
    def interrupted(_path):
        raise KeyboardInterrupt('interrupted before OS lock')
        yield

    monkeypatch.setattr(artifacts, '_writer_lock', interrupted)
    with pytest.raises(KeyboardInterrupt, match='interrupted before OS lock'):
        artifacts.read_member(reference)
    monkeypatch.setattr(artifacts, '_writer_lock', original_writer_lock)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(artifacts.read_member, reference).result(timeout=2) == b'committed'
