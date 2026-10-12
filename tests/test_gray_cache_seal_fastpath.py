from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import sqlite3
import threading

import numpy as np
import pytest

from XTA import gray_cache
from XTA.gray_cache import GrayCacheError, GrayCacheReader, GrayCacheWriter, gray_cache_path


def spec():
    return dict(source_identity='seal-fixture', physical_view_record={'name': 'spherical_fixture'},
                native_centers=[0, 1, 2], canvas_shape=[3, 8, 9], capture_contract={'channel': 0})


def pixels():
    return np.arange(3 * 8 * 9, dtype=np.uint8).reshape(3, 8, 9)


@pytest.mark.parametrize('committed_frames', [0, 1])
def test_incomplete_seal_does_not_wait_for_reserved_writer(tmp_path, committed_frames):
    record = spec()
    path = gray_cache_path(tmp_path, record)
    with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as writer:
        if committed_frames:
            assert writer.write_frames([0], pixels()[:1])
        writer._connection.execute('PRAGMA busy_timeout=25')
        statements = []
        writer._connection.set_trace_callback(statements.append)
        with closing(sqlite3.connect(path)) as holder:
            holder.execute('BEGIN IMMEDIATE')
            try:
                assert writer.seal() is False
                assert writer.disabled_reason is None
                assert not any(sql.upper().startswith('BEGIN') for sql in statements)
            finally:
                holder.rollback()


def test_uncommitted_last_packet_does_not_block_seal_and_later_commit_seals(tmp_path, monkeypatch):
    record = spec()
    path = gray_cache_path(tmp_path, record)
    committed = threading.Event()
    release = threading.Event()
    data = pixels()
    real_connect = sqlite3.connect

    class GatedConnection(sqlite3.Connection):
        block_packet_commit = False

        def commit(self):
            if self.block_packet_commit:
                committed.set()
                assert release.wait(5)
            return super().commit()

    def connect(*args, **kwargs):
        return real_connect(*args, factory=GatedConnection, **kwargs)

    with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as first:
        assert first.write_frames([0], data[:1])
        first._connection.execute('PRAGMA busy_timeout=25')
        monkeypatch.setattr(gray_cache.sqlite3, 'connect', connect)
        with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as last:
            last._connection.block_packet_commit = True
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(last.write_frames, [1, 2], data[1:])
                try:
                    assert committed.wait(5)
                    assert first.seal() is False
                    assert first.disabled_reason is None
                finally:
                    release.set()
                assert future.result(timeout=5)
            assert first.seal()
            last._connection.block_packet_commit = False
            assert last.seal()
    with GrayCacheReader(path, record) as reader:
        first_digest = reader.metadata['content_digest']
        for index, frame in enumerate(data):
            np.testing.assert_array_equal(reader.read_crop(index, (0, 0, 8, 9)), frame)
    reference = gray_cache_path(tmp_path / 'reference', record)
    with GrayCacheWriter(reference, record, 'raw', min_free_bytes=0) as writer:
        assert writer.write_frames([0], data[:1])
        assert writer.write_frames([1, 2], data[1:])
        assert writer.seal()
    with GrayCacheReader(reference, record) as reader:
        assert reader.metadata['content_digest'] == first_digest


def test_sealed_store_can_be_checked_under_reserved_writer(tmp_path):
    record = spec()
    path = gray_cache_path(tmp_path, record)
    with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as writer:
        assert writer.write_frames([0, 1, 2], pixels())
        assert writer.seal()
        writer._connection.execute('PRAGMA busy_timeout=25')
        with closing(sqlite3.connect(path)) as holder:
            holder.execute('BEGIN IMMEDIATE')
            try:
                assert writer.seal()
                assert writer.disabled_reason is None
            finally:
                holder.rollback()


def test_incomplete_seal_still_validates_header(tmp_path):
    record = spec()
    path = gray_cache_path(tmp_path, record)
    with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as writer:
        with closing(sqlite3.connect(path)) as corrupter:
            corrupter.execute('PRAGMA journal_mode=TRUNCATE')
            corrupter.execute("UPDATE header SET spec_digest='bad'")
            corrupter.commit()
        with pytest.raises(GrayCacheError, match='header fields'):
            writer.seal()


def test_incomplete_seal_rejects_replaced_valid_identity(tmp_path):
    record = spec()
    path = gray_cache_path(tmp_path, record)
    with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as writer:
        replacement = dict(record, source_identity='another source')
        encoded = json.dumps(replacement, sort_keys=True, separators=(',', ':')).encode()
        with closing(sqlite3.connect(path)) as corrupter:
            corrupter.execute('PRAGMA journal_mode=TRUNCATE')
            corrupter.execute('UPDATE header SET spec=?,spec_digest=?',
                              (encoded.decode(), hashlib.sha256(encoded).hexdigest()))
            corrupter.commit()
        with pytest.raises(GrayCacheError, match='identity changed'):
            writer.seal()


def test_complete_count_still_requires_valid_chunk_coverage(tmp_path):
    record = spec()
    path = gray_cache_path(tmp_path, record)
    with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as writer:
        assert writer.write_frames([0, 1, 2], pixels())
        with closing(sqlite3.connect(path)) as corrupter:
            corrupter.execute('PRAGMA journal_mode=TRUNCATE')
            corrupter.execute('DELETE FROM chunks')
            corrupter.commit()
        with pytest.raises(GrayCacheError, match='spatial coverage'):
            writer.seal()
        assert writer._connection.execute('SELECT sealed,content_digest FROM header').fetchone() == (0, None)


def test_complete_seal_rechecks_identity_after_read_only_precheck(tmp_path, monkeypatch):
    record = spec()
    path = gray_cache_path(tmp_path, record)
    data = pixels()
    with GrayCacheWriter(path, record, 'raw', min_free_bytes=0) as writer:
        assert writer.write_frames([0, 1, 2], data)
        real_header = gray_cache._header
        calls = 0

        def replace_after_read(connection):
            nonlocal calls
            result = real_header(connection)
            calls += 1
            if calls == 1:
                replacement = dict(record, source_identity='changed during seal')
                encoded = json.dumps(replacement, sort_keys=True, separators=(',', ':')).encode()
                with closing(sqlite3.connect(path)) as corrupter:
                    corrupter.execute('PRAGMA journal_mode=TRUNCATE')
                    corrupter.execute('UPDATE header SET spec=?,spec_digest=?',
                                      (encoded.decode(), hashlib.sha256(encoded).hexdigest()))
                    corrupter.commit()
            return result

        monkeypatch.setattr(gray_cache, '_header', replace_after_read)
        with pytest.raises(GrayCacheError, match='identity changed'):
            writer.seal()
        assert calls == 2
        assert writer._connection.execute('SELECT sealed,content_digest FROM header').fetchone() == (0, None)
