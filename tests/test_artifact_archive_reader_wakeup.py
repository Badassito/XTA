"""A reader queued before prefix publication can use that prefix after wakeup."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
import os
import threading

import pytest

from XTA import artifact_archive as artifacts


@pytest.fixture(autouse=True)
def locked_reader_path(monkeypatch):
    """These tests exercise the queued/locked reader path. Committed members of an
    unchanged archive are read lock-free by default (tests/test_artifact_archive_lock_free_reads.py)."""
    monkeypatch.setenv('YOLO_TTA_ARCHIVE_LOCK_FREE_READS', '0')


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
            assert artifacts._path_key(archive) not in artifacts._APPEND_SNAPSHOTS
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
    assert artifacts._path_key(archive) not in artifacts._APPEND_SNAPSHOTS


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
    def interrupted(_path, **_kwargs):
        raise KeyboardInterrupt('interrupted before OS lock')
        yield

    monkeypatch.setattr(artifacts, '_writer_lock', interrupted)
    with pytest.raises(KeyboardInterrupt, match='interrupted before OS lock'):
        artifacts.read_member(reference)
    monkeypatch.setattr(artifacts, '_writer_lock', original_writer_lock)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(artifacts.read_member, reference).result(timeout=2) == b'committed'


def test_slow_snapshot_lookup_does_not_hold_global_condition(tmp_path, monkeypatch):
    archive = tmp_path / 'paused-reader.tar'
    snapshot = object()
    entered, release = threading.Event(), threading.Event()

    def lookup(current):
        assert current is snapshot
        entered.set()
        assert release.wait(3)
        return 'old member'

    def read():
        with artifacts._reader_lock(archive, lookup) as result:
            return result

    monkeypatch.setitem(artifacts._APPEND_SNAPSHOTS, artifacts._path_key(archive), snapshot)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = pool.submit(read)
        try:
            assert entered.wait(1)
            assert artifacts._READERS_CHANGED.acquire(timeout=.25)
            artifacts._READERS_CHANGED.release()
        finally:
            release.set()
        assert reader.result(timeout=1) == 'old member'


@pytest.mark.parametrize('lookup', ['member', 'list'])
def test_reader_retries_when_retired_snapshot_archive_is_replaced(tmp_path, monkeypatch, lookup):
    archive = tmp_path / 'archive.tar'
    replacement = tmp_path / 'replacement.tar'
    artifacts.append_members(archive, {'scope/member': b'old payload'})
    artifacts.append_members(replacement, {'scope/member': b'new payload', 'scope/other': b'new'})
    writing, finish, opening, replaced = [threading.Event() for _ in range(4)]
    write_member, open_prefix = artifacts._write_member, artifacts._open_append_prefix

    def write(stream, name, source):
        if name.endswith('/pending'):
            writing.set()
            assert finish.wait(5)
        return write_member(stream, name, source)

    def open_paused(path, snapshot):
        opening.set()
        assert replaced.wait(5)
        return open_prefix(path, snapshot)

    monkeypatch.setattr(artifacts, '_write_member', write)
    monkeypatch.setattr(artifacts, '_open_append_prefix', open_paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(artifacts.append_members, archive, {'pending': b'later'})
        try:
            assert writing.wait(3)
            if lookup == 'member':
                reader = pool.submit(artifacts.read_member, artifacts.reference(archive, 'scope/member'))
                expected = b'new payload'
            else:
                reader = pool.submit(artifacts.list_members, archive, 'scope')
                expected = ['scope/member', 'scope/other']
            assert opening.wait(3)
            finish.set()
            writer.result(timeout=3)
            os.replace(replacement, archive)
        finally:
            finish.set()
            replaced.set()
        assert reader.result(timeout=3) == expected


@pytest.mark.skipif(os.name != 'nt', reason='Windows path alias')
def test_case_alias_shares_verified_snapshot_and_cache(tmp_path, monkeypatch):
    archive, alias = tmp_path / 'Archive.tar', tmp_path / 'archive.tar'
    artifacts.append_members(archive, {'old': b'payload'})
    if not alias.exists():
        pytest.skip('temporary directory uses case sensitivity')
    key = artifacts._path_key(archive)
    assert key == artifacts._path_key(alias)
    monkeypatch.setenv('YOLO_TTA_ARCHIVE_LOCK_FREE_READS', '1')
    fast = artifacts._fast_index(alias)
    assert fast is not None
    fast[1].close()
    monkeypatch.setenv('YOLO_TTA_ARCHIVE_LOCK_FREE_READS', '0')
    writing, finish = threading.Event(), threading.Event()
    write_member = artifacts._write_member

    def write(stream, name, source):
        if name.endswith('/pending'):
            writing.set()
            assert finish.wait(5)
        return write_member(stream, name, source)

    monkeypatch.setattr(artifacts, '_write_member', write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(artifacts.append_members, archive, {'pending': b'later'})
        try:
            assert writing.wait(3)
            reader = pool.submit(artifacts.read_member, artifacts.reference(alias, 'old'))
            assert reader.result(timeout=3) == b'payload'
            assert key in artifacts._APPEND_SNAPSHOTS
        finally:
            finish.set()
        writer.result(timeout=3)


@pytest.mark.skipif(os.name != 'nt', reason='Windows folded key collision')
def test_casefold_collision_cannot_lend_or_remove_other_files_snapshot(tmp_path, monkeypatch):
    archive, other = tmp_path / 'Straße.tar', tmp_path / 'STRASSE.tar'
    artifacts.append_members(archive, {'old': b'one'})
    if other.exists():
        pytest.skip('filesystem treats these names as the same file')
    artifacts.append_members(other, {'old': b'two'})
    key = artifacts._path_key(archive)
    assert key == artifacts._path_key(other)
    with archive.open('rb') as stream:
        entries, end = artifacts._index(archive, stream)
        snapshot = (artifacts._stamp(os.fstat(stream.fileno()))[:2], entries, end, str(archive))
    monkeypatch.setitem(artifacts._APPEND_SNAPSHOTS, key, snapshot)
    assert artifacts._open_append_prefix(other, snapshot) is None
    assert artifacts._APPEND_SNAPSHOTS[key] is snapshot
    with artifacts._open_append_prefix(archive, snapshot):
        pass
