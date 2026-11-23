"""Committed members of an unchanged archive are read without the writer lock."""
import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import errno
import gc
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
    def refuse(_path, **_kwargs):
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
    artifacts._ARCHIVE_INDEX_CACHE.pop(artifacts._path_key(artifacts._canonical_path(other)), None)
    artifacts.append_members(other, {'new': b'later'})
    misses = artifacts.archive_read_stats()['fast_misses']
    artifacts._ARCHIVE_INDEX_CACHE[artifacts._path_key(artifacts._canonical_path(other))] = \
        artifacts._ARCHIVE_INDEX_CACHE[artifacts._path_key(artifacts._canonical_path(archive))]
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
        seen.append([shared.fd for shared in artifacts._THREAD_HANDLES.handles.values()])
    for _ in range(3):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    import gc
    gc.collect()
    for descriptors in seen:
        for fd in descriptors:
            with pytest.raises(OSError):
                os.fstat(fd)  # closed with its thread


def _positional_only(lock_free):
    if lock_free != 'positional':
        pytest.skip('only the cached-descriptor branch shares descriptors')


def _cached_reader(tmp_path):
    """An archive whose index is cached, plus a reader seeked into its first member."""
    archive = tmp_path/'sam-artifacts.tar'
    payload = os.urandom(100_000)
    artifacts.append_members(archive, {'m1': payload, 'm2': b'second'})
    assert artifacts.read_member(artifacts.reference(archive, 'm1')) == payload  # caches a descriptor
    reader = artifacts.open_member(artifacts.reference(archive, 'm1'))
    reader.seek(10)
    return archive, payload, reader


def test_evicted_descriptor_stays_open_for_a_live_reader(tmp_path, lock_free):
    _positional_only(lock_free)
    _archive, payload, reader = _cached_reader(tmp_path)
    fd = reader._stream._shared.fd
    artifacts._THREAD_HANDLES.handles.clear()
    unrelated = os.open(tmp_path/'unrelated.bin', os.O_CREAT | os.O_RDWR | getattr(os, 'O_BINARY', 0))
    try:
        os.write(unrelated, b'u' * 64)
        assert reader.read(32) == payload[10:42]
        reader.close()
        gc.collect()
        with pytest.raises(OSError):
            os.fstat(fd)  # the last holder closed it
    finally:
        os.close(unrelated)


def test_identity_change_does_not_close_a_live_reader(tmp_path, monkeypatch, lock_free):
    _positional_only(lock_free)
    archive, payload, reader = _cached_reader(tmp_path)
    real_stat, replaced = os.stat, []

    class _OtherInode:
        def __init__(self, info):
            self._info = info

        def __getattr__(self, name):
            value = getattr(self._info, name)
            return value + 1 if name == 'st_ino' else value

    def stat_after_replace(path, *args, **kwargs):
        info = real_stat(path, *args, **kwargs)
        return _OtherInode(info) if replaced and str(path).endswith('sam-artifacts.tar') else info
    monkeypatch.setattr(os, 'stat', stat_after_replace)
    replaced.append(True)
    # Same thread: the identity check evicts the cached descriptor the reader uses.
    assert artifacts.member_exists(artifacts.reference(archive, 'm2'))
    assert not artifacts._THREAD_HANDLES.handles  # the identity check evicted it
    unrelated = os.open(tmp_path/'unrelated.bin', os.O_CREAT | os.O_RDWR | getattr(os, 'O_BINARY', 0))
    try:
        os.write(unrelated, b'u' * 64)
        assert reader.read(32) == payload[10:42]
    finally:
        reader.close()
        os.close(unrelated)


def test_reader_outlives_its_opening_thread(tmp_path, lock_free):
    _positional_only(lock_free)
    archive = tmp_path/'sam-artifacts.tar'
    payload = os.urandom(100_000)
    artifacts.append_members(archive, {'m': payload})
    opened = []
    thread = threading.Thread(target=lambda: opened.append(artifacts.open_member(artifacts.reference(archive, 'm'))))
    thread.start()
    thread.join()
    gc.collect()
    with opened.pop() as reader:
        assert isinstance(reader._stream, artifacts._PositionalStream)
        assert reader.read() == payload


def _refuse_lock_file(monkeypatch, code):
    def unwritable_lock(file, mode='r', *args, **kwargs):
        if mode in ('rb', 'a+b') and str(file).endswith('.tar.lock'):
            raise OSError(code, os.strerror(code), str(file))
        return builtins.open(file, mode, *args, **kwargs)
    monkeypatch.setattr(artifacts, 'open', unwritable_lock, raising=False)


@pytest.mark.parametrize('code', [errno.EROFS, errno.EACCES])
def test_cold_reader_without_a_writable_lock_reads_unlocked(tmp_path, monkeypatch, code):
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'scope/m': b'payload'})
    monkeypatch.setattr(artifacts, '_ARCHIVE_INDEX_CACHE', {})
    _refuse_lock_file(monkeypatch, code)
    assert artifacts.read_member(artifacts.reference(archive, 'scope/m')) == b'payload'
    monkeypatch.setattr(artifacts, '_ARCHIVE_INDEX_CACHE', {})
    assert artifacts.list_members(archive) == ['scope/m']
    assert artifacts.member_exists(artifacts.reference(archive, 'scope/m'))
    assert not artifacts.member_exists(artifacts.reference(archive, 'scope/missing'))
    with pytest.raises(OSError) as refused:
        artifacts.append_members(archive, {'scope/n': b'x'})  # writers still need the lock
    assert refused.value.errno == code


def test_fallback_errors_are_not_chained_to_the_lock_error(tmp_path, monkeypatch):
    archive = tmp_path/'sam-artifacts.tar'
    artifacts.append_members(archive, {'scope/m': b'payload'})
    monkeypatch.setattr(artifacts, '_ARCHIVE_INDEX_CACHE', {})
    _refuse_lock_file(monkeypatch, errno.EROFS)
    with pytest.raises(FileNotFoundError) as missing:
        artifacts.read_member(artifacts.reference(archive, 'scope/missing'))
    context = missing.value.__context__
    while context is not None:
        assert getattr(context, 'errno', None) != errno.EROFS
        context = context.__context__
