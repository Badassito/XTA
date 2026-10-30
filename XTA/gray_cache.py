"""Bounded, lossless per-view gray8 cache with atomic SQLite packet publication."""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import operator
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
import zlib

import numpy as np

SCHEMA = 'xta.gray_view_cache/1'
TILE = 256
DEPTH = 8
MAX_CHUNK_BYTES = DEPTH*TILE*TILE
MAX_PACKET_BYTES = 128*1024**2
MAX_METADATA_BYTES = 1024**2
DEFAULT_DISK_FLOOR_BYTES = 16*1024**3
_GRAY_ENCODER_RESOLUTION_LOCK = threading.Lock()


class GrayCacheError(RuntimeError):
    """Invalid, mismatched or corrupt cached data; never an ordinary cache miss."""


class GrayCacheUnavailable(RuntimeError):
    """Optional cache could not be admitted within disk/lock resource limits."""


def _metrics(**amounts):
    try:
        from .runtime import runtime_telemetry
        telemetry = runtime_telemetry()
        for name,value in amounts.items():
            telemetry.add('sam.gray_cache.'+name,value)
    except Exception:
        pass  # Diagnostics cannot change committed cache ownership.


def _full(error):
    return (getattr(error,'sqlite_errorcode',None) == sqlite3.SQLITE_FULL
            or 'database or disk is full' in str(error).lower())


def _busy(error):
    while error is not None:
        code = getattr(error,'sqlite_errorcode',None)
        if (isinstance(error,sqlite3.Error) and
                (code is not None and code & 255 in (sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED)
                 or 'database is locked' in str(error).lower()
                 or 'database table is locked' in str(error).lower()
                 or 'database schema is locked' in str(error).lower())):
            return True
        error = error.__cause__
    return False


def _integer(value):
    if isinstance(value, (bool, np.bool_)):
        raise GrayCacheError('Gray cache integer fields cannot be Boolean')
    try:
        return operator.index(value)
    except TypeError as error:
        raise GrayCacheError('Gray cache integer field is not integral') from error


def _spec(spec):
    try:
        encoded = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(encoded) > MAX_METADATA_BYTES:
            raise GrayCacheError('Gray cache metadata exceeds its bound')
        record = json.loads(encoded)
        shape = tuple(_integer(x) for x in record['canvas_shape'])
        centers = tuple(_integer(x) for x in record['native_centers'])
        if (len(shape) != 3 or min(shape) <= 0 or not centers or len(set(centers)) != len(centers)
                or min(centers) < 0 or max(centers) >= shape[0]
                or not isinstance(record['source_identity'], str) or not record['source_identity']
                or not isinstance(record['physical_view_record'], dict)
                or not isinstance(record['capture_contract'], dict)):
            raise GrayCacheError('Gray cache spec has invalid shape, coverage or identity')
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        raise GrayCacheError('Gray cache spec is not a bounded JSON identity') from error
    return record, encoded, shape, centers


def gray_cache_path(store_root, spec):
    _record, encoded, _shape, _centers = _spec(spec)
    return Path(store_root)/('gray-'+hashlib.sha256(encoded).hexdigest()[:32]+'.sqlite')


def _header(connection):
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('header','packets','frames','chunks')")}
        if not tables and not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone():
            raise GrayCacheUnavailable('Gray cache initialization has not committed a header')
        if not {'header','packets','frames','chunks'}.issubset(tables):
            raise GrayCacheError('Gray cache schema is incomplete')
        lengths = connection.execute('''SELECT typeof(schema),length(CAST(schema AS BLOB)),
            typeof(spec),length(CAST(spec AS BLOB)),typeof(spec_digest),length(CAST(spec_digest AS BLOB)),
            typeof(sealed),sealed IN (0,1),typeof(content_digest),length(CAST(content_digest AS BLOB))
            FROM header WHERE id=1''').fetchone()
        if (lengths is None or lengths[0] != 'text' or lengths[1] != len(SCHEMA)
                or lengths[2] != 'text' or not 0 < lengths[3] <= MAX_METADATA_BYTES
                or lengths[4] != 'text' or lengths[5] != 64 or lengths[6] != 'integer'
                or not lengths[7] or lengths[8] not in ('null','text')
                or lengths[8] == 'text' and lengths[9] != 64):
            raise GrayCacheError('Gray header fields exceed their type or byte bounds')
        row = connection.execute('SELECT schema,spec,spec_digest,sealed,content_digest FROM header WHERE id=1').fetchone()
        if row is None or row[0] != SCHEMA:
            raise GrayCacheError('Unknown or incomplete gray cache header')
        spec, encoded, shape, centers = _spec(json.loads(row[1]))
        if hashlib.sha256(encoded).hexdigest() != row[2] or row[3] not in (0, 1):
            raise GrayCacheError('Gray cache header identity differs from its digest')
        if row[3] and (not isinstance(row[4], str) or len(row[4]) != 64):
            raise GrayCacheError('Sealed gray cache lacks a content digest')
        return dict(schema=row[0], spec=spec, spec_digest=row[2], sealed=bool(row[3]),
                    content_digest=row[4], chunk_shape=[DEPTH,TILE,TILE]), shape, centers
    except (sqlite3.Error, ValueError, TypeError, RecursionError) as error:
        if _busy(error):
            raise GrayCacheUnavailable('Gray cache header is busy') from error
        raise GrayCacheError('Cannot read gray cache header') from error


def _content_digest(connection, metadata, shape, centers):
    digest = hashlib.sha256(metadata['spec_digest'].encode())
    seen = set()
    declared = set(centers)
    if connection.execute("SELECT 1 FROM packets WHERE typeof(id)!='integer' OR typeof(first_frame)!='integer' LIMIT 1").fetchone():
        raise GrayCacheError('Gray packet identifiers exceed their type bound')
    for packet, frame_type,frame_bytes in connection.execute('SELECT id,typeof(frames),length(CAST(frames AS BLOB)) FROM packets ORDER BY first_frame'):
        if frame_type != 'text' or not 0 < frame_bytes <= 512:
            raise GrayCacheError('Gray packet frame index exceeds its metadata bound')
        frame_json = connection.execute('SELECT frames FROM packets WHERE id=?',(packet,)).fetchone()[0]
        try:
            ids = tuple(_integer(x) for x in json.loads(frame_json))
        except (TypeError, ValueError) as error:
            raise GrayCacheError('Invalid gray packet frame index') from error
        if (not 1 <= len(ids) <= DEPTH or len(set(ids)) != len(ids) or seen.intersection(ids)
                or not set(ids).issubset(declared)):
            raise GrayCacheError('Gray packets have invalid or overlapping coverage')
        seen.update(ids)
        bad_mapping = connection.execute('''SELECT count(*),sum(typeof(frame)!='integer' OR typeof(plane)!='integer')
            FROM frames WHERE packet=?''',(packet,)).fetchone()
        if bad_mapping[0] != len(ids) or bad_mapping[1]:
            raise GrayCacheError('Gray frame index exceeds its type or coverage bound')
        mapping = connection.execute('SELECT frame,plane FROM frames WHERE packet=? ORDER BY plane', (packet,)).fetchall()
        if mapping != [(frame, index) for index, frame in enumerate(ids)]:
            raise GrayCacheError('Gray frame index disagrees with packet coverage')
        digest.update(frame_json.encode())
        count = 0
        bad = connection.execute('''SELECT 1 FROM chunks WHERE packet=? AND
            (typeof(y)!='integer' OR typeof(x)!='integer' OR typeof(h)!='integer' OR typeof(w)!='integer'
             OR typeof(raw_size)!='integer' OR typeof(codec)!='text' OR length(CAST(codec AS BLOB))>8
             OR typeof(sha)!='text' OR length(CAST(sha AS BLOB))!=64 OR typeof(data)!='blob') LIMIT 1''',(packet,)).fetchone()
        if bad:
            raise GrayCacheError('Gray chunk fields exceed their type or metadata bounds')
        for y,x,h,w,raw_size,codec,sha,length in connection.execute(
                'SELECT y,x,h,w,raw_size,codec,sha,length(data) FROM chunks WHERE packet=? ORDER BY y,x', (packet,)):
            if (y < 0 or x < 0 or y % TILE or x % TILE or h != min(TILE,shape[1]-y)
                    or w != min(TILE,shape[2]-x) or not 1 <= h <= TILE or not 1 <= w <= TILE
                    or raw_size != len(ids)*h*w or raw_size > MAX_CHUNK_BYTES
                    or not 0 < length <= raw_size or codec not in ('raw','zlib','zstd')
                    or not isinstance(sha, str) or len(sha) != 64):
                raise GrayCacheError('Gray chunk metadata exceeds its geometry or byte bounds')
            digest.update(json.dumps([y,x,h,w,raw_size,sha], separators=(',', ':')).encode())
            count += 1
        if count != ((shape[1]+TILE-1)//TILE)*((shape[2]+TILE-1)//TILE):
            raise GrayCacheError('Gray packet lacks complete spatial coverage')
    if seen != declared:
        raise GrayCacheError('Gray cache does not contain its complete declared frame coverage')
    return digest.hexdigest()


def _log_gray_encoder(codec, info):
    try:
        from .runtime import runtime_telemetry
        runtime_telemetry().gauge('sam.gray_cache.encoder.'+codec, info)
    except Exception:
        pass  # Diagnostics cannot change the selected codec.


def _checked_gray_encoder(codec, provider, version, compress, fallbacks, requested):
    raw = b'\0'*8192+bytes(range(256))*32+b'\xff'*4096
    try:
        encoded = bytes(compress(raw))
        _decode(encoded, codec, len(raw), hashlib.sha256(raw).hexdigest())
    except (ImportError, AttributeError, NotImplementedError):
        raise
    except Exception as error:
        raise GrayCacheError('Gray '+provider+' encoder failed its bounded known-answer check') from error
    info = dict(requested=requested, backend=provider, codec=codec, version=version,
                level=None if codec == 'raw' else 1, fallback_reasons=list(fallbacks))
    _log_gray_encoder(requested, info)
    return codec, compress, info


@lru_cache(maxsize=4)
def _resolve_gray_encoder(codec):
    if codec not in ('auto','raw','zlib','zstd'):
        raise GrayCacheError('Unknown requested gray codec')
    if codec == 'raw':
        return _checked_gray_encoder('raw','raw',None,lambda raw:raw,(),codec)
    fallbacks = []
    if codec in ('auto','zstd'):
        try:
            import imagecodecs
            encode = imagecodecs.zstd_encode
            return _checked_gray_encoder('zstd','imagecodecs.zstd',getattr(imagecodecs,'__version__',None),
                lambda raw:bytes(encode(raw,level=1)),fallbacks,codec)
        except (ImportError, AttributeError, NotImplementedError) as error:
            if codec == 'zstd':
                raise GrayCacheError('Requested Zstd gray encoder is unavailable') from error
            fallbacks.append('imagecodecs.zstd unavailable: '+type(error).__name__)
    if codec == 'auto':
        try:
            import deflate
            from .nrrd_compression import _known_nrrd_libdeflate_version
            version = _known_nrrd_libdeflate_version(deflate)
            match = re.fullmatch(r'(\d+)\.(\d+)(?:\.\d+)?(?:\+[A-Za-z0-9_.+-]+)?',version or '')
            if match is None or (int(match.group(1)),int(match.group(2))) < (0,9):
                fallbacks.append('libdeflate binding lacks a proven deflate>=0.9.0 version')
            else:
                encode = deflate.zlib_compress
                return _checked_gray_encoder('zlib','libdeflate',version,
                    lambda raw:bytes(encode(raw,1)),fallbacks,codec)
        except (ImportError, AttributeError, NotImplementedError) as error:
            fallbacks.append('libdeflate unavailable: '+type(error).__name__)
        try:
            import isal
            from isal import isal_zlib
            def encode_isal(raw):
                compressor = isal_zlib.compressobj(1,isal_zlib.DEFLATED,15)
                return bytes(compressor.compress(raw)+compressor.flush())
            return _checked_gray_encoder('zlib','isal',getattr(isal,'__version__',None),
                encode_isal,fallbacks,codec)
        except (ImportError, AttributeError, NotImplementedError) as error:
            fallbacks.append('ISA-L unavailable: '+type(error).__name__)
    return _checked_gray_encoder('zlib','stdlib.zlib',zlib.ZLIB_RUNTIME_VERSION,
        lambda raw:zlib.compress(raw,1),fallbacks,codec)


def gray_encoder_info(codec='auto'):
    """Resolve and prove a process-local encoder; return JSON-safe provenance."""
    if codec not in ('auto','raw','zlib','zstd'):
        raise GrayCacheError('Unknown requested gray codec')
    with _GRAY_ENCODER_RESOLUTION_LOCK:
        _actual, _compress, info = _resolve_gray_encoder(codec)
    return dict(info,fallback_reasons=list(info['fallback_reasons']))


def _encode(raw, codec):
    if codec not in ('auto','raw','zlib','zstd'):
        raise GrayCacheError('Unknown requested gray codec')
    with _GRAY_ENCODER_RESOLUTION_LOCK:
        actual, compress, _info = _resolve_gray_encoder(codec)
    encoded = bytes(compress(raw))
    return (actual, encoded) if len(encoded) < len(raw) else ('raw', raw)


def _decode(payload, codec, raw_size, sha):
    if not 0 < raw_size <= MAX_CHUNK_BYTES or not 0 < len(payload) <= raw_size:
        raise GrayCacheError('Gray chunk exceeds its encoded/decoded byte bound')
    try:
        if codec == 'raw':
            raw = payload
        elif codec == 'zlib':
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, raw_size+1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise GrayCacheError('Gray deflate chunk is incomplete or has trailing data')
        elif codec == 'zstd':
            import imagecodecs
            raw = bytes(imagecodecs.zstd_decode(payload, out=raw_size))
        else:
            raise GrayCacheError('Unknown persisted gray codec')
    except (ImportError, AttributeError, ValueError, RuntimeError, zlib.error) as error:
        raise GrayCacheError('Gray chunk decoder failed') from error
    if len(raw) != raw_size or hashlib.sha256(raw).hexdigest() != sha:
        raise GrayCacheError('Gray chunk content differs from its committed hash')
    return raw


class GrayCacheWriter:
    def __init__(self, path, spec, codec='auto', *, min_free_bytes=DEFAULT_DISK_FLOOR_BYTES):
        self.path = Path(path)
        self.spec, encoded, self.shape, self.centers = _spec(spec)
        self.codec, self.min_free_bytes = codec, _integer(min_free_bytes)
        if self.min_free_bytes < 0 or codec not in ('auto','raw','zlib','zstd'):
            raise GrayCacheError('Invalid gray cache writer limits')
        self.disabled_reason = None
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() and shutil.disk_usage(self.path.parent).free < self.min_free_bytes+1024**2:
            _metrics(initialization_disk_declines=1)
            raise GrayCacheUnavailable('Gray cache initialization declined at disk-space floor')
        self._connection = None
        try:
            self._connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            mode = self._connection.execute('PRAGMA journal_mode=TRUNCATE').fetchone()[0]
            if mode.lower() != 'truncate':
                raise GrayCacheError('Gray cache cannot establish rollback-journal mode')
            self._connection.execute('PRAGMA synchronous=NORMAL')
            existing = self._connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='header'").fetchone()
            if not existing:
                self._connection.execute('BEGIN IMMEDIATE')
                # Another packet writer may have initialized the store while this one waited.
                existing = self._connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='header'").fetchone()
                if not existing:
                    schema = '''
                    CREATE TABLE IF NOT EXISTS header(id INTEGER PRIMARY KEY CHECK(id=1),schema TEXT,spec TEXT,spec_digest TEXT,sealed INTEGER,content_digest TEXT);
                    CREATE TABLE IF NOT EXISTS packets(id INTEGER PRIMARY KEY,first_frame INTEGER UNIQUE,frames TEXT);
                    CREATE TABLE IF NOT EXISTS frames(frame INTEGER PRIMARY KEY,packet INTEGER,plane INTEGER);
                    CREATE INDEX IF NOT EXISTS frames_packet_plane ON frames(packet,plane);
                    CREATE TABLE IF NOT EXISTS chunks(packet INTEGER,y INTEGER,x INTEGER,h INTEGER,w INTEGER,raw_size INTEGER,codec TEXT,sha TEXT,data BLOB,PRIMARY KEY(packet,y,x));
                    '''
                    for statement in schema.split(';'):
                        if statement.strip():
                            self._connection.execute(statement)
                    self._connection.execute('INSERT INTO header VALUES(1,?,?,?,0,NULL)',
                        (SCHEMA,encoded.decode(),hashlib.sha256(encoded).hexdigest()))
                    self._connection.commit()
                else:
                    self._connection.rollback()
            metadata, _shape, _centers = _header(self._connection)
            if metadata['spec'] != self.spec:
                raise GrayCacheError('Existing gray cache has a different identity')
        except BaseException as error:
            if self._connection is not None:
                self._connection.close()
            if isinstance(error,sqlite3.Error) and _full(error):
                _metrics(initialization_disk_declines=1)
                raise GrayCacheUnavailable('Gray cache initialization ran out of disk space') from error
            if _busy(error):
                _metrics(initialization_busy_declines=1)
                raise GrayCacheUnavailable('Gray cache initialization declined after lock contention') from error
            raise

    def write_frames(self, frame_ids, gray_u8_NHW):
        ids = tuple(_integer(x) for x in frame_ids)
        values = gray_u8_NHW
        if (not isinstance(values, np.ndarray) or values.dtype != np.uint8 or values.ndim != 3
                or not 1 <= len(ids) <= DEPTH or values.shape != (len(ids),*self.shape[1:])
                or len(set(ids)) != len(ids) or not set(ids).issubset(self.centers)
                or values.nbytes > MAX_PACKET_BYTES):
            raise GrayCacheError('Gray packet has invalid dtype, shape, coverage or size')
        with self._lock:
            try:
                if _header(self._connection)[0]['sealed']:
                    raise GrayCacheError('Cannot mutate a sealed gray cache')
            except GrayCacheUnavailable as error:
                if not _busy(error):
                    raise
                self.disabled_reason = 'lock_contention'
                _metrics(busy_declines=1)
                return False
            if self.disabled_reason is not None:
                return False
        required = self.min_free_bytes+2*values.nbytes+1024**2
        if shutil.disk_usage(self.path.parent).free < required:
            self.disabled_reason = 'disk_space_floor'
            _metrics(disk_declines=1)
            return False
        chunks = []
        encode_started = time.perf_counter()
        encode_cpu_started = time.thread_time()
        for y in range(0,self.shape[1],TILE):
            for x in range(0,self.shape[2],TILE):
                crop = np.ascontiguousarray(values[:,y:y+TILE,x:x+TILE])
                raw = crop.tobytes()
                codec, encoded = _encode(raw,self.codec)
                chunks.append((y,x,crop.shape[1],crop.shape[2],len(raw),codec,hashlib.sha256(raw).hexdigest(),encoded))
        encode_seconds = time.perf_counter()-encode_started
        encode_cpu_seconds = time.thread_time()-encode_cpu_started
        write_started = time.perf_counter()
        with self._lock:
            try:
                self._connection.execute('BEGIN IMMEDIATE')
                if _header(self._connection)[0]['sealed']:
                    raise GrayCacheError('Cannot mutate a sealed gray cache')
                if shutil.disk_usage(self.path.parent).free < required:
                    self._connection.rollback()
                    self.disabled_reason = 'disk_space_floor'
                    _metrics(disk_declines=1)
                    return False
                cursor = self._connection.execute('INSERT INTO packets(first_frame,frames) VALUES(?,?)',
                    (min(ids),json.dumps(ids,separators=(',',':'))))
                packet = cursor.lastrowid
                self._connection.executemany('INSERT INTO frames VALUES(?,?,?)',[(frame,packet,index) for index,frame in enumerate(ids)])
                self._connection.executemany('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)',[(packet,*row) for row in chunks])
                self._connection.commit()
            except sqlite3.Error as error:
                self._connection.rollback()
                if _full(error):
                    self.disabled_reason = 'disk_full'
                    _metrics(disk_declines=1)
                    return False
                if _busy(error):
                    self.disabled_reason = 'lock_contention'
                    _metrics(busy_declines=1)
                    return False
                raise GrayCacheError('Gray packet transaction failed') from error
            except BaseException as error:
                self._connection.rollback()
                if _busy(error):
                    self.disabled_reason = 'lock_contention'
                    _metrics(busy_declines=1)
                    return False
                raise
        write_seconds = time.perf_counter()-write_started
        codec_metrics = {}
        for codec in ('raw','zlib','zstd'):
            selected = [row for row in chunks if row[5] == codec]
            codec_metrics[codec+'_chunks'] = len(selected)
            codec_metrics[codec+'_raw_bytes'] = sum(row[4] for row in selected)
            codec_metrics[codec+'_encoded_bytes'] = sum(len(row[-1]) for row in selected)
        _metrics(raw_bytes=values.nbytes,encoded_bytes=sum(len(row[-1]) for row in chunks),
            committed_packets=1,committed_frames=len(ids),chunks=len(chunks),
            encode_host_seconds=encode_seconds,encode_thread_cpu_seconds=encode_cpu_seconds,
            write_host_seconds=write_seconds,**codec_metrics)
        return True

    def seal(self):
        with self._lock:
            try:
                metadata,_shape,centers = _header(self._connection)
                if metadata['spec'] != self.spec:
                    raise GrayCacheError('Gray cache identity changed before sealing')
                if metadata['sealed']:
                    return True
                # Most task tails have partial coverage; avoid taking their view's writer lock.
                if self._connection.execute('SELECT count(*) FROM frames').fetchone()[0] != len(centers):
                    return False
                self._connection.execute('BEGIN IMMEDIATE')
                metadata,shape,centers = _header(self._connection)
                if metadata['spec'] != self.spec:
                    raise GrayCacheError('Gray cache identity changed before sealing')
                if metadata['sealed']:
                    self._connection.rollback()
                    return True
                if {row[0] for row in self._connection.execute('SELECT frame FROM frames')} != set(centers):
                    self._connection.rollback()
                    return False
                digest = _content_digest(self._connection,metadata,shape,centers)
                self._connection.execute('UPDATE header SET sealed=1,content_digest=? WHERE id=1',(digest,))
                self._connection.commit()
                return True
            except BaseException as error:
                self._connection.rollback()
                if isinstance(error,sqlite3.Error) and _full(error):
                    self.disabled_reason = 'disk_full'
                    _metrics(seal_disk_declines=1)
                    return False
                if _busy(error):
                    self.disabled_reason = 'lock_contention'
                    _metrics(seal_busy_declines=1)
                    return False
                raise

    def close(self):
        with self._lock:
            self._connection.close()

    def __enter__(self): return self
    def __exit__(self,*_args): self.close()


class GrayCacheReader:
    def __init__(self,path,expected_spec=None,max_cache_bytes=0,max_crop_bytes=64*1024**2):
        self.path = Path(path)
        stat = self.path.stat()
        self._file_identity = stat.st_dev,stat.st_ino
        self._lock = threading.RLock()
        self.max_cache_bytes,self.max_crop_bytes = _integer(max_cache_bytes),_integer(max_crop_bytes)
        if self.max_cache_bytes < 0 or self.max_crop_bytes <= 0:
            raise GrayCacheError('Invalid gray cache reader limits')
        self._cache,self._cached_bytes = OrderedDict(),0
        self._sealed_version = None
        self._connection = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro',uri=True,timeout=0,check_same_thread=False)
        try:
            self._metadata,self.shape,self.centers = _header(self._connection)
            if expected_spec is not None and _spec(expected_spec)[0] != self._metadata['spec']:
                raise GrayCacheError('Gray cache identity does not match the requested view')
            self._check()
        except BaseException as error:
            self._connection.close()
            if _busy(error):
                _metrics(reader_busy_declines=1)
                raise GrayCacheUnavailable('Gray cache read admission is busy') from error
            raise

    def _check(self):
        try:
            stat = self.path.stat()
        except OSError as error:
            raise GrayCacheError('Gray cache file disappeared while being read') from error
        if (stat.st_dev,stat.st_ino) != self._file_identity:
            raise GrayCacheError('Gray cache file was replaced while a reader owned it')
        version = self._connection.execute('PRAGMA data_version').fetchone()[0]
        if self._sealed_version is not None and version != self._sealed_version:
            raise GrayCacheError('Sealed gray cache was modified while being read')
        if self._sealed_version is None:
            self._connection.execute('BEGIN')
            try:
                metadata,shape,centers = _header(self._connection)
                if metadata['spec_digest'] != self._metadata['spec_digest']:
                    raise GrayCacheError('Gray cache identity changed while being read')
                if metadata['sealed']:
                    if _content_digest(self._connection,metadata,shape,centers) != metadata['content_digest']:
                        raise GrayCacheError('Sealed gray cache index differs from its content proof')
                    self._sealed_version = self._connection.execute('PRAGMA data_version').fetchone()[0]
                    metadata['codecs'] = [row[0] for row in self._connection.execute('SELECT DISTINCT codec FROM chunks ORDER BY codec')]
                    sealed_stat = self.path.stat()
                    self._sealed_file_state = sealed_stat.st_size,sealed_stat.st_mtime_ns
                self._metadata = metadata
            finally:
                self._connection.rollback()
        elif (stat.st_size,stat.st_mtime_ns) != self._sealed_file_state:
            raise GrayCacheError('Sealed gray cache file bytes changed while being read')

    @property
    def metadata(self):
        with self._lock:
            self._check()
            record = json.loads(json.dumps(self._metadata))
            record.setdefault('codecs',[])
            return record

    @property
    def sealed(self):
        with self._lock:
            self._check()
            return self._metadata['sealed']

    def _bbox(self,bbox):
        box = tuple(_integer(x) for x in bbox)
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise GrayCacheError('Gray crop needs increasing y/x bounds')
        if (box[2]-box[0])*(box[3]-box[1]) > self.max_crop_bytes:
            raise GrayCacheError('Gray crop exceeds its output scratch bound')
        return box

    def crop_workspace_bytes(self,bbox):
        y0,x0,y1,x1 = self._bbox(bbox)
        tiles = ((y1-y0+2*TILE-2)//TILE)*((x1-x0+2*TILE-2)//TILE)
        return (y1-y0)*(x1-x0)+3*MAX_CHUNK_BYTES+tiles*512+16*MAX_METADATA_BYTES+self.max_cache_bytes

    def read_crop(self,native_frame,bbox):
        started = time.perf_counter()
        decoded_bytes = decode_chunks = cache_hits = 0
        decode_seconds = 0.0
        frame = _integer(native_frame)
        y0,x0,y1,x1 = self._bbox(bbox)
        with self._lock:
            self._check()
            if (not self._metadata['sealed'] or frame not in self.centers
                    or y0 < 0 or x0 < 0 or y1 > self.shape[1] or x1 > self.shape[2]):
                return None
            mapping = self._connection.execute('SELECT packet,plane FROM frames WHERE frame=?',(frame,)).fetchone()
            if mapping is None:
                raise GrayCacheError('Sealed gray cache lost a declared frame')
            packet,plane = mapping
            output = np.empty((y1-y0,x1-x0),np.uint8)
            for y in range(y0//TILE*TILE,y1,TILE):
                for x in range(x0//TILE*TILE,x1,TILE):
                    key = packet,y,x
                    values = self._cache.pop(key,None)
                    if values is None:
                        row = self._connection.execute('SELECT h,w,raw_size,codec,sha,length(data) FROM chunks WHERE packet=? AND y=? AND x=?',key).fetchone()
                        if row is None or not 0 < row[2] <= MAX_CHUNK_BYTES or not 0 < row[5] <= row[2]:
                            raise GrayCacheError('Missing or oversized gray chunk')
                        payload = self._connection.execute('SELECT data FROM chunks WHERE packet=? AND y=? AND x=?',key).fetchone()[0]
                        decode_started = time.perf_counter()
                        raw = _decode(payload,row[3],row[2],row[4])
                        decode_seconds += time.perf_counter()-decode_started
                        decoded_bytes += len(raw)
                        decode_chunks += 1
                        values = np.frombuffer(raw,np.uint8).reshape(-1,row[0],row[1])
                        if values.nbytes <= self.max_cache_bytes:
                            while self._cached_bytes+values.nbytes > self.max_cache_bytes:
                                _key,removed = self._cache.popitem(last=False)
                                self._cached_bytes -= removed.nbytes
                            self._cache[key] = values
                            self._cached_bytes += values.nbytes
                    else:
                        cache_hits += 1
                        self._cache[key] = values
                    if not 0 <= plane < values.shape[0]:
                        raise GrayCacheError('Gray frame offset exceeds its chunk')
                    # Cached shape was authenticated on insertion; both sealed checks still apply.
                    top,left,bottom,right = max(y,y0),max(x,x0),min(y+values.shape[1],y1),min(x+values.shape[2],x1)
                    output[top-y0:bottom-y0,left-x0:right-x0] = values[plane,top-y:bottom-y,left-x:right-x]
                    values = raw = payload = None
            self._check()
        read_seconds = time.perf_counter()-started
        _metrics(decoded_bytes=decoded_bytes,decode_chunks=decode_chunks,
            decode_host_seconds=decode_seconds,cache_hits=cache_hits,read_host_seconds=read_seconds)
        return output

    def close(self):
        with self._lock:
            self._cache.clear()
            self._cached_bytes = 0
            self._connection.close()

    def __enter__(self): return self
    def __exit__(self,*_args): self.close()
