"""Committed members of an unchanged archive are read without the writer lock."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import os
import shutil

import pytest

from XTA import artifact_archive as artifacts


import threading

_EMULATION_LOCK = threading.Lock()


def _emulated_pread(fd, size, offset):
    with _EMULATION_LOCK:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, size)


@pytest.fixture(autouse=True, params=['native', 'positional'])
def lock_free(monkeypatch, request):
    """'positional' runs the cached-descriptor (Linux) branch, emulating os.pread if absent."""
    monkeypatch.delenv('YOLO_TTA_ARCHIVE_LOCK_FREE_READS', raising=False)
    monkeypatch.setattr(artifacts, '_THREAD_HANDLES', threading.local())
    if request.param == 'positional':
        if not hasattr(os, 'pread'):
            monkeypatch.setattr(os, 'pread', _emulated_pread, raising=False)
    elif hasattr(os, 'pread'):
        monkeypatch.delattr(os, 'pread')
    return request.param


def forbid_lock(monkeypatch):
    @contextmanager
    def refuse(_path):
        raise AssertionError('reader took the writer lock')
        yield
    monkeypatch.setattr(artifacts, '_writer_lock', refuse)


def test_unchanged_archive_reads_committed_members_without_the_lock(tmp_path, monkeypatch):
    archive = tmp_path/'sam-artifacts.tar'
    payloads = {f'scope/run{i}/masks.bin': os.urandom(1000+i) for i in range(5)}
    artifacts.append_members(archive, payloads)
    before = artifacts.archive_read_stats()['fast_reads']
    forbid_lock(monkeypatch)
    for name, data in payloads.items():
        assert artifacts.read_member(artifacts.reference(archive, name)) == data
        with artifacts.open_member(artifacts.reference(archive, name)) as stream:
            stream.seek(7)
            assert stream.read(11) == data[7:18]
    assert artifacts.list_members(archive, 'scope/run2') == ['scope/run2/masks.bin']
    assert artifacts.archive_read_stats()['fast_reads'] - before >= 11


def test_unknown_names_and_full_inventory_keep_the_locked_path(tmp_path, monkeypatch):
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'a/one': b'1'})
    forbid_lock(monkeypatch)
    with pytest.raises(AssertionError, match='writer lock'):
        artifacts.read_member(artifacts.reference(archive, 'a/missing'))
    with pytest.raises(AssertionError, match='writer lock'):
        artifacts.list_members(archive)
    with pytest.raises(AssertionError, match='writer lock'):
        artifacts.list_members(archive, 'not-there')


def test_externally_extended_archive_falls_back_and_sees_new_members(tmp_path):
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'old': b'committed'})
    # Simulate another process appending: the cached index no longer matches.
    other = tmp_path/'other.tar'
    shutil.copyfile(archive, other)
    artifacts._ARCHIVE_INDEX_CACHE.pop(str(artifacts._canonical_path(other)), None)
    artifacts.append_members(other, {'new': b'later'})
    misses = artifacts.archive_read_stats()['fast_misses']
    artifacts._ARCHIVE_INDEX_CACHE[str(artifacts._canonical_path(other))] = \
        artifacts._ARCHIVE_INDEX_CACHE[str(artifacts._canonical_path(archive))]
    assert artifacts.read_member(artifacts.reference(other, 'new')) == b'later'
    assert artifacts.read_member(artifacts.reference(other, 'old')) == b'committed'
    assert artifacts.archive_read_stats()['fast_misses'] > misses


def test_replaced_archive_path_is_not_served_from_an_old_descriptor(tmp_path, monkeypatch, lock_free):
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'m': b'first'})
    assert artifacts.read_member(artifacts.reference(archive, 'm')) == b'first'  # caches a handle
    replacement = tmp_path/'replacement.tar'
    artifacts.append_members(replacement, {'m': b'second'})
    if lock_free == 'positional' and os.name == 'nt':
        # Windows cannot replace a file with an open descriptor; drop it first and
        # check identity validation through a fresh descriptor instead.
        for fd in artifacts._THREAD_HANDLES.handles.values():
            os.close(fd)
        artifacts._THREAD_HANDLES.handles.clear()
    os.replace(replacement, archive)
    # The old cached index/handle must not answer for the new file.
    assert artifacts.read_member(artifacts.reference(archive, 'm')) == b'second'


def test_concurrent_lock_free_reads_return_exact_payloads(tmp_path, monkeypatch):
    archive = tmp_path/'sam-artifacts.tar'
    payloads = {f'run{i}': os.urandom(200_000+i) for i in range(16)}
    artifacts.append_members(archive, payloads)
    forbid_lock(monkeypatch)

    def read_all(_):
        return {name: hashlib.sha256(artifacts.read_member(artifacts.reference(archive, name))).hexdigest()
                for name in payloads}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read_all, range(16)))
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    assert all(result == expected for result in results)


def test_switch_restores_the_locked_reader(tmp_path, monkeypatch):
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'m': b'x'})
    monkeypatch.setenv('YOLO_TTA_ARCHIVE_LOCK_FREE_READS', '0')
    forbid_lock(monkeypatch)
    with pytest.raises(AssertionError, match='writer lock'):
        artifacts.read_member(artifacts.reference(archive, 'm'))


def test_thread_descriptors_close_when_the_thread_ends(tmp_path, lock_free):
    if lock_free != 'positional':
        pytest.skip('only the cached-descriptor branch keeps descriptors')
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'m': b'x'})
    seen = []

    def worker():
        assert artifacts.read_member(artifacts.reference(archive, 'm')) == b'x'
        seen.append(dict(artifacts._THREAD_HANDLES.handles))
    for _ in range(3):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    import gc
    gc.collect()
    for handles in seen:
        for fd in handles.values():
            with pytest.raises(OSError):
                os.fstat(fd)  # closed with its thread
