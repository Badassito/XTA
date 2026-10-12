from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import sys
import threading
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from XTA.gray_cache import (GrayCacheError, GrayCacheUnavailable, GrayCacheReader,
    GrayCacheWriter, MAX_METADATA_BYTES, gray_cache_path)
from XTA import gray_cache


@pytest.fixture(autouse=True)
def _clear_gray_encoder_resolution():
    gray_cache._resolve_gray_encoder.cache_clear()
    yield
    gray_cache._resolve_gray_encoder.cache_clear()


def spec(shape=(10,257,259)):
    return dict(source_identity='fixture:bytes:mtime',physical_view_record={'name':'spherical_fixture'},
        source_resize_semantics='center_aligned',native_centers=list(range(shape[0])),
        processing_shape=[12,32,32],canvas_shape=list(shape),
        output_to_processing_affine=[[1,0,0],[0,1,0]],
        capture_contract={'kind':'normalized_cuda_center_gray8_clamp_round_v1','channel':0})


def values(record):
    shape=record['canvas_shape']
    return (np.arange(np.prod(shape),dtype=np.uint32).reshape(shape)*17%256).astype(np.uint8)


def create(path,record,codec='zlib'):
    data=values(record)
    with GrayCacheWriter(path,record,codec,min_free_bytes=0) as writer:
        for start in range(0,len(data),8):
            assert writer.write_frames(range(start,min(start+8,len(data))),data[start:start+8])
        assert writer.seal()
    return data


def test_live_view_missing_until_complete_and_exact_crops(tmp_path):
    record=spec();path=gray_cache_path(tmp_path,record);data=values(record)
    with GrayCacheWriter(path,record,'zlib',min_free_bytes=0) as writer:
        assert writer.write_frames([8,9],data[8:])
        with GrayCacheReader(path,record) as reader:
            assert not writer.seal()
            assert reader.read_crop(8,(0,0,5,5)) is None
            assert writer.write_frames(range(8),data[:8])
            assert writer.seal()
            assert reader.sealed
            for frame in (0,7,8,9):
                actual=reader.read_crop(frame,(250,249,257,259))
                np.testing.assert_array_equal(actual,data[frame,250:257,249:259])
            assert reader.read_crop(10,(0,0,2,2)) is None
            assert reader.read_crop(0,(-1,0,2,2)) is None
            assert len(reader.metadata['content_digest'])==64
            assert 'zlib' in reader.metadata['codecs']
            assert set(reader.metadata['codecs']) <= {'raw','zlib'}
        with pytest.raises(GrayCacheError,match='sealed'):
            writer.write_frames([0],data[:1])
    assert not list(tmp_path.glob('*-wal'))
    assert not list(tmp_path.glob('*-shm'))
    assert len(list(tmp_path.glob('*')))<=2


def test_duplicate_packet_rolls_back_without_partial_frames(tmp_path):
    record=spec((3,5,6));data=values(record);path=gray_cache_path(tmp_path,record)
    with GrayCacheWriter(path,record,'raw',min_free_bytes=0) as writer:
        assert writer.write_frames([0],data[:1])
        with pytest.raises(GrayCacheError,match='transaction'):
            writer.write_frames([1,0],data[[1,0]])
        assert not writer.seal()
        assert writer.write_frames([1,2],data[1:])
        assert writer.seal()
    with GrayCacheReader(path,record) as reader:
        np.testing.assert_array_equal(reader.read_crop(2,(0,0,5,6)),data[2])


def test_concurrent_constructor_and_packet_connections(tmp_path):
    record=spec((2,8,9));data=values(record);path=gray_cache_path(tmp_path,record)
    barrier=threading.Barrier(2)
    def publish(index):
        barrier.wait()
        with GrayCacheWriter(path,record,'zlib',min_free_bytes=0) as writer:
            assert writer.write_frames([index],data[index:index+1])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(publish,index) for index in range(2)]
        for future in futures:future.result(timeout=10)
    with GrayCacheWriter(path,record,'zlib',min_free_bytes=0) as writer:
        assert writer.seal()
    with GrayCacheReader(path,record) as reader:
        np.testing.assert_array_equal(reader.read_crop(1,(0,0,8,9)),data[1])


def test_existing_writer_open_reads_under_another_reserved_writer(tmp_path):
    record=spec((1,8,9));path=gray_cache_path(tmp_path,record)
    with GrayCacheWriter(path,record,min_free_bytes=0):pass
    with closing(sqlite3.connect(path)) as holder:
        holder.execute('BEGIN IMMEDIATE')
        def open_existing(expected):
            with GrayCacheWriter(path,expected,min_free_bytes=0) as writer:
                assert writer._connection.execute('PRAGMA journal_mode').fetchone()==('truncate',)
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                pool.submit(open_existing,record).result(timeout=2)
                changed=dict(record,source_identity='different source')
                with pytest.raises(GrayCacheError,match='identity'):
                    pool.submit(open_existing,changed).result(timeout=2)
            finally:
                holder.rollback()


def test_concurrent_packet_collision_commits_exactly_once(tmp_path):
    record=spec((1,8,9));path=gray_cache_path(tmp_path,record);data=values(record)
    with GrayCacheWriter(path,record,'raw',min_free_bytes=0):pass
    barrier=threading.Barrier(2)
    def publish():
        with GrayCacheWriter(path,record,'raw',min_free_bytes=0) as writer:
            barrier.wait(timeout=5)
            try:
                return writer.write_frames([0],data)
            except GrayCacheError as error:
                assert 'transaction' in str(error)
                return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(publish) for _ in range(2)]
        assert sorted(future.result(timeout=10) for future in futures)==[False,True]
    with GrayCacheWriter(path,record,'raw',min_free_bytes=0) as writer:
        assert writer.seal()
    with closing(sqlite3.connect(path)) as probe:
        assert probe.execute('SELECT count(*) FROM packets').fetchone()==(1,)
        assert probe.execute('SELECT count(*) FROM frames').fetchone()==(1,)
        assert probe.execute('SELECT count(*) FROM chunks').fetchone()==(1,)
    with GrayCacheReader(path,record) as reader:
        np.testing.assert_array_equal(reader.read_crop(0,(0,0,8,9)),data[0])


def test_initial_schema_and_header_are_one_atomic_transaction(tmp_path,monkeypatch):
    record=spec((1,8,9));path=gray_cache_path(tmp_path,record)
    created_header=threading.Event();release_schema=threading.Event();second_started=threading.Event()
    real_connect=sqlite3.connect
    class GatedConnection(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            result=super().execute(sql,*args,**kwargs)
            if 'CREATE TABLE IF NOT EXISTS header' in sql:
                created_header.set()
                assert release_schema.wait(5)
            return result
    def connect(*args,**kwargs):
        kwargs['factory']=GatedConnection
        return real_connect(*args,**kwargs)
    monkeypatch.setattr(gray_cache.sqlite3,'connect',connect)
    def initialize(second=False):
        if second:second_started.set()
        with GrayCacheWriter(path,record,min_free_bytes=0):pass
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(initialize)
        assert created_header.wait(5)
        second=pool.submit(initialize,True)
        assert second_started.wait(5)
        try:
            with closing(real_connect(path)) as probe:
                assert probe.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()==[]
            assert not second.done()
        finally:
            release_schema.set()
        first.result(timeout=10);second.result(timeout=10)
    with GrayCacheReader(path,record) as reader:
        assert not reader.sealed


def test_reader_output_and_metadata_cannot_mutate_cache(tmp_path):
    record=spec((1,3,4));path=gray_cache_path(tmp_path,record);data=create(path,record)
    with GrayCacheReader(path,record) as reader:
        output=reader.read_crop(0,(0,0,3,4));output[:]=0
        meta=reader.metadata;meta['spec']['source_identity']='wrong'
        np.testing.assert_array_equal(reader.read_crop(0,(0,0,3,4)),data[0])


def test_corruption_and_post_seal_mutation_fail(tmp_path):
    record=spec((1,3,4));path=gray_cache_path(tmp_path,record);create(path,record,'raw')
    with GrayCacheReader(path,record) as reader:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE chunks SET data=zeroblob(raw_size)")
        with pytest.raises(GrayCacheError,match='modified|changed'):
            reader.read_crop(0,(0,0,3,4))
    with GrayCacheReader(path,record) as reader:
        with pytest.raises(GrayCacheError,match='content'):
            reader.read_crop(0,(0,0,3,4))


@pytest.mark.parametrize('table,column,length',[
    ('header','spec',MAX_METADATA_BYTES+1),
    ('packets','frames',513),
    ('chunks','codec',10000),
],ids=['header','frames','codec'])
def test_oversized_sql_metadata_rejected_before_use(tmp_path,table,column,length):
    record=spec((1,3,4));path=gray_cache_path(tmp_path,record);create(path,record)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f'UPDATE {table} SET {column}=?',('x'*length,))
    with pytest.raises(GrayCacheError,match='bound'):
        GrayCacheReader(path,record)
    path.unlink()  # Constructor failures must not keep a SQLite handle alive.


def test_identity_output_budget_and_lru_are_charged(tmp_path):
    record=spec((2,257,259));path=gray_cache_path(tmp_path,record);data=create(path,record)
    changed=dict(record,source_identity='other')
    with pytest.raises(GrayCacheError,match='identity'):
        GrayCacheReader(path,changed)
    with GrayCacheReader(path,record,max_crop_bytes=8) as reader:
        with pytest.raises(GrayCacheError,match='scratch'):
            reader.read_crop(0,(0,0,3,3))
    with GrayCacheReader(path,record,max_cache_bytes=131072) as reader:
        np.testing.assert_array_equal(reader.read_crop(0,(0,0,257,259)),data[0])
        assert reader._cached_bytes<=131072
        charged=reader.crop_workspace_bytes((0,0,2,2))
    with GrayCacheReader(path,record,max_cache_bytes=0) as reader:
        assert charged-reader.crop_workspace_bytes((0,0,2,2))==131072


def test_auto_codec_fallback_and_disk_decline(tmp_path,monkeypatch):
    record=spec((1,64,64));data=np.zeros((1,64,64),np.uint8);path=gray_cache_path(tmp_path,record)
    monkeypatch.setitem(sys.modules,'imagecodecs',None)
    with GrayCacheWriter(path,record,min_free_bytes=0) as writer:
        assert writer.write_frames([0],data)
        assert writer.seal()
    with GrayCacheReader(path,record) as reader:
        assert reader.metadata['codecs']==['zlib']
        np.testing.assert_array_equal(reader.read_crop(0,(0,0,64,64)),data[0])
    monkeypatch.setattr('XTA.gray_cache.shutil.disk_usage',lambda _path:SimpleNamespace(free=0))
    other=tmp_path/'missing.sqlite'
    with pytest.raises(GrayCacheUnavailable,match='floor'):
        GrayCacheWriter(other,record)
    assert not other.exists()
    with GrayCacheWriter(path,record,min_free_bytes=0) as writer:
        assert writer.seal()


@pytest.mark.parametrize('compressible',[False,True])
def test_encode_metrics_expose_actual_codec_and_thread_cpu(tmp_path,monkeypatch,compressible):
    record=spec((1,64,64));path=gray_cache_path(tmp_path,record);metrics=[]
    data=(np.zeros((1,64,64),np.uint8) if compressible else
          np.random.default_rng(123).integers(0,256,(1,64,64),dtype=np.uint8))
    monkeypatch.setitem(sys.modules,'imagecodecs',None)
    monkeypatch.setattr(gray_cache,'_metrics',lambda **kwargs:metrics.append(kwargs))
    cpu_times=iter((10.,10.25))
    monkeypatch.setattr(gray_cache.time,'thread_time',lambda:next(cpu_times))
    with GrayCacheWriter(path,record,min_free_bytes=0) as writer:
        assert writer.write_frames([0],data)
    success=[row for row in metrics if 'committed_packets' in row]
    assert len(success)==1
    actual='zlib' if compressible else 'raw';row=success[0]
    assert row['encode_thread_cpu_seconds']==.25 and row['encode_host_seconds']>=0
    for codec in ('raw','zlib','zstd'):
        assert row[codec+'_chunks']==int(codec==actual)
        assert row[codec+'_raw_bytes']==(data.nbytes if codec==actual else 0)
        assert row[codec+'_encoded_bytes']==(row['encoded_bytes'] if codec==actual else 0)
    assert row['encoded_bytes']<(data.nbytes) if compressible else row['encoded_bytes']==data.nbytes


def test_partial_packet_has_no_success_metrics(tmp_path,monkeypatch):
    record=spec((2,5,6));path=gray_cache_path(tmp_path,record);data=values(record);metrics=[]
    monkeypatch.setattr('XTA.gray_cache._metrics',lambda **kwargs:metrics.append(kwargs))
    with GrayCacheWriter(path,record,'raw',min_free_bytes=0) as writer:
        assert writer.write_frames([0],data[:1])
        with pytest.raises(GrayCacheError):writer.write_frames([0],data[:1])
    success=[row for row in metrics if 'committed_packets' in row]
    assert len(success)==1 and success[0]['raw_bytes']==30 and success[0]['committed_frames']==1


@pytest.mark.parametrize('operation',['connect','schema','packet','seal'])
def test_sqlite_full_is_optional_cache_decline(tmp_path,monkeypatch,operation):
    record=spec((1,5,6));path=gray_cache_path(tmp_path,record);data=values(record)
    real_connect=sqlite3.connect
    triggered=False
    class FullConnection(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            nonlocal triggered
            if ((operation=='schema' and 'CREATE TABLE' in sql)
                    or (operation=='seal' and 'UPDATE header SET sealed=1' in sql)):
                triggered=True
                raise sqlite3.OperationalError('database or disk is full')
            return super().execute(sql,*args,**kwargs)
        def executemany(self,sql,*args,**kwargs):
            nonlocal triggered
            if operation=='packet' and 'INSERT INTO chunks' in sql:
                triggered=True
                raise sqlite3.OperationalError('database or disk is full')
            return super().executemany(sql,*args,**kwargs)
    def connect(*args,**kwargs):
        nonlocal triggered
        if operation=='connect':
            triggered=True
            raise sqlite3.OperationalError('database or disk is full')
        kwargs['factory']=FullConnection
        return real_connect(*args,**kwargs)
    monkeypatch.setattr(gray_cache.sqlite3,'connect',connect)
    if operation in ('connect','schema'):
        with pytest.raises(GrayCacheUnavailable,match='disk space'):
            GrayCacheWriter(path,record,min_free_bytes=0)
    else:
        with GrayCacheWriter(path,record,min_free_bytes=0) as writer:
            wrote=writer.write_frames([0],data)
            assert wrote==(operation!='packet')
            assert not writer.seal()
            assert writer.disabled_reason=='disk_full'
        with closing(real_connect(path)) as probe:
            assert probe.execute('SELECT sealed FROM header').fetchone()==(0,)
            if operation=='packet':
                assert probe.execute('SELECT count(*) FROM frames').fetchone()==(0,)
                assert probe.execute('SELECT count(*) FROM packets').fetchone()==(0,)
    assert triggered
    if path.exists():path.unlink()


def test_disk_floor_during_packet_leaves_unsealed_cache(tmp_path,monkeypatch):
    record=spec((1,5,6));path=gray_cache_path(tmp_path,record)
    with GrayCacheWriter(path,record,min_free_bytes=0) as writer:
        monkeypatch.setattr(gray_cache.shutil,'disk_usage',lambda _path:SimpleNamespace(free=0))
        assert not writer.write_frames([0],values(record))
        assert writer.disabled_reason=='disk_space_floor'
        assert not writer.seal()
    with GrayCacheReader(path,record) as reader:
        assert reader.read_crop(0,(0,0,5,6)) is None


def test_decoding_is_bounded_even_for_small_compressed_bomb():
    encoded=zlib.compress(b'x'*100000,1)
    with pytest.raises(GrayCacheError,match='decoder'):
        gray_cache._decode(encoded,'zlib',1024,'0'*64)


def test_read_metrics_count_decoded_chunks_and_lru_hits(tmp_path,monkeypatch):
    record=spec((2,10,11));path=gray_cache_path(tmp_path,record);create(path,record)
    metrics=[]
    monkeypatch.setattr(gray_cache,'_metrics',lambda **kwargs:metrics.append(kwargs))
    with GrayCacheReader(path,record,max_cache_bytes=4096) as reader:
        reader.read_crop(0,(0,0,10,11));reader.read_crop(1,(0,0,10,11))
    assert metrics[0]['decoded_bytes']==220 and metrics[0]['decode_chunks']==1
    assert metrics[1]['decoded_bytes']==0 and metrics[1]['cache_hits']==1


def test_missing_tile_and_oversized_payload_fail(tmp_path):
    record=spec((1,3,4));path=gray_cache_path(tmp_path,record);create(path,record,'raw')
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('UPDATE chunks SET data=zeroblob(raw_size+1)')
    with pytest.raises(GrayCacheError,match='bound'):
        GrayCacheReader(path,record)


def test_foreign_schema_is_corruption_not_unavailable(tmp_path):
    path=tmp_path/'foreign.sqlite'
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute('CREATE TABLE unexpected(value TEXT)')
    with pytest.raises(GrayCacheError,match='incomplete'):
        GrayCacheReader(path)
    path.unlink()


def test_full_view_frame_validation_uses_packet_index(tmp_path):
    record=spec((512,5,6));path=gray_cache_path(tmp_path,record);data=create(path,record,'raw')
    with closing(sqlite3.connect(path)) as probe:
        plan=probe.execute('EXPLAIN QUERY PLAN SELECT frame,plane FROM frames WHERE packet=? ORDER BY plane',(31,)).fetchall()
        assert any('SEARCH frames USING' in row[3] and 'frames_packet_plane' in row[3] for row in plan)
        assert not any('SCAN frames' in row[3] or 'TEMP B-TREE' in row[3] for row in plan)
    with GrayCacheReader(path,record) as reader:
        np.testing.assert_array_equal(reader.read_crop(511,(0,0,5,6)),data[511])


@pytest.mark.parametrize('operation',['initialize','write','header','seal'])
@pytest.mark.parametrize('message',['database is locked','database table is locked'])
def test_writer_busy_is_optional_cache_decline(tmp_path,monkeypatch,operation,message):
    record=spec((1,5,6));path=gray_cache_path(tmp_path,record);data=values(record)
    real_connect=sqlite3.connect;armed=operation=='initialize';triggered=False
    class BusyConnection(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            nonlocal triggered
            if armed and ((operation in ('initialize','write') and sql=='BEGIN IMMEDIATE')
                    or (operation=='header' and 'SELECT name FROM sqlite_master' in sql)
                    or (operation=='seal' and 'UPDATE header SET sealed=1' in sql)):
                triggered=True
                raise sqlite3.OperationalError(message)
            return super().execute(sql,*args,**kwargs)
    def connect(*args,**kwargs):
        kwargs['factory']=BusyConnection
        return real_connect(*args,**kwargs)
    monkeypatch.setattr(gray_cache.sqlite3,'connect',connect)
    if operation=='initialize':
        with pytest.raises(GrayCacheUnavailable,match='lock contention'):
            GrayCacheWriter(path,record,min_free_bytes=0)
    else:
        with GrayCacheWriter(path,record,min_free_bytes=0) as writer:
            if operation=='seal':
                assert writer.write_frames([0],data)
                armed=True
                assert not writer.seal()
            else:
                armed=True
                assert not writer.write_frames([0],data)
            assert writer.disabled_reason=='lock_contention'
        with closing(real_connect(path)) as probe:
            assert probe.execute('SELECT sealed FROM header').fetchone()==(0,)
            if operation!='seal':assert probe.execute('SELECT count(*) FROM frames').fetchone()==(0,)
    assert triggered
    if path.exists():path.unlink()


def test_reader_unsealed_probe_is_nonblocking_under_exclusive_writer(tmp_path,monkeypatch):
    record=spec((1,5,6));path=gray_cache_path(tmp_path,record)
    with GrayCacheWriter(path,record,min_free_bytes=0):pass
    real_connect=sqlite3.connect;timeouts=[]
    def connect(*args,**kwargs):
        timeouts.append(kwargs.get('timeout'))
        return real_connect(*args,**kwargs)
    monkeypatch.setattr(gray_cache.sqlite3,'connect',connect)
    with closing(real_connect(path)) as holder:
        holder.execute('BEGIN EXCLUSIVE')
        with pytest.raises(GrayCacheUnavailable,match='busy'):
            GrayCacheReader(path,record)
        holder.rollback()
    assert timeouts==[0]
    path.unlink()
