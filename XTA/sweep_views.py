"""Bounded, immutable native-view products shared by confidence cutoffs.

Workers publish exact pre-parent masks and observed score bytes. The parent
seals only acknowledged ranges; unsupported routes have an explicit fallback
decision, while a lost or corrupt acknowledged product is an integrity error.
This module owns no inference, cleanup, geometry transform, or GPU state.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict, deque
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import threading
import weakref
from uuid import uuid4
import zlib

import numpy as np

from .json_publication import write_json_atomic
from .sweep_quota import (
    SweepViewCapacityError, native_fanout_root as _root, native_journal_usage, update_leases as _quota_many,
)

SCHEMA = 'xta.native_sweep_views/1'
INDEX = np.dtype([
    ('state', 'u1'), ('padding', 'u1'), ('score', 'u1'), ('reserved', 'u1'),
    ('frame', '<u4'), ('predictions', '<u8'), ('frames_with_predictions', '<u8'),
    ('y0', '<u4'), ('y1', '<u4'), ('x0', '<u4'), ('x1', '<u4'),
    ('offset', '<u8'), ('length', '<u8'), ('chunks', '<u4'),
])
_VERIFY_INDEX=np.dtype(INDEX.descr+[('sha256','u1',(32,))])
_CHUNK = struct.Struct('<10I')
_CHUNK_BYTES = 1024 * 1024
_RESERVE_BYTES = 8 * 1024 * 1024
_ENCODER_QUEUE_MIB = 32
_ENCODER_IDLE_SECONDS = 2.0
_INDEX_CACHE = weakref.WeakValueDictionary()
_INDEX_VALIDATION_CACHE = {}
_INDEX_CACHE_LOCK = threading.Lock()


class SweepViewIntegrityError(RuntimeError):
    """An acknowledged native product cannot be used exactly."""


class SweepViewRetiredError(RuntimeError):
    """The durable downstream product has replaced this native backing."""


class SweepViewScalarOwnedError(RuntimeError):
    """The scalar pipeline owns these pixels; staging contains counts only."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _shape(value):
    result = tuple(int(v) for v in value)
    if len(result) != 3 or min(result) <= 0 or max(result) >= 2**32:
        raise ValueError('Native fanout needs a positive uint32 TYX shape')
    return result


def _cents(values):
    result = tuple(values)
    if not result or any(type(v) is not int or not 0 <= v <= 100 for v in result) or len(set(result)) != len(result):
        raise ValueError('Native fanout cutoffs must be unique integer hundredths in [0,100]')
    return result


def _family_id():
    return str(os.environ.get('XTA_YOLO_SWEEP_ID', ''))


def _product(model_name, view_name, shape, identity=None):
    return dict(family_id=_family_id(), model_name=str(model_name), view_name=str(view_name),
                native_shape=list(_shape(shape)), identity=identity)


def _view_dir(root, model_name, view_name):
    return root / 'views' / _hash([str(model_name), str(view_name)])


def _names(task, cfg):
    view = task.get('view')
    name = getattr(view, 'name', None) or task.get('view_name')
    model = task.get('model_name') or getattr(cfg, 'model_name', None) or 'model'
    if not name:
        raise ValueError('Native fanout task has no logical view name')
    return str(model), str(name)


def _initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    owner = root / 'owner.json'
    if owner.exists():
        if json.loads(owner.read_text(encoding='utf-8')) != {'schema': SCHEMA, 'family_id': _family_id()}:
            raise SweepViewIntegrityError('Native fanout directory belongs to another family')
    else:
        write_json_atomic(owner, {'schema': SCHEMA, 'family_id': _family_id()})


def _quota(root, key, amount):
    _quota_many(root, ((key, amount),))


def cutoffs_for_task(task, cfg, backend=None):
    """All family cuts, including the already-decoded floor, only in produce role."""
    if os.environ.get('XTA_YOLO_SWEEP_NATIVE_ROLE', 'disabled') != 'produce':
        return ()
    raw = os.environ.get('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS', '')
    if not raw or any(re.fullmatch(r'[0-9]+', part) is None for part in raw.split(',')):
        raise ValueError('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS must be comma-separated integer hundredths')
    return _cents(tuple(int(part) for part in raw.split(',')))


def begin_task_fanout(task, cfg, *, backend, expected_frames, padding_count=0,
                      native_shape, identity=None, root=None, scalar_owned_cutoff=None):
    cuts = cutoffs_for_task(task, cfg, backend)
    if not cuts:
        return None
    model, view = _names(task, cfg)
    if identity is None:
        identity=task.get('sweep_fanout_identity')
    return NativeTaskFanoutWriter(_root(root), model, view, native_shape, cuts,
        frame_start=int(task.get('slice_start', 0)), expected_frames=expected_frames,
        padding_count=padding_count, backend=backend, identity=identity,
        scalar_owned_cutoff=scalar_owned_cutoff)


class NativeTaskFanoutWriter:
    """Thread-safe bounded append writer; finish acknowledges the whole task only.

    put accepts a full HxW plane or a crop with a half-open bbox. Supplied bounds
    are authoritative support metadata; no full-plane scan is made for them.
    Empty (0,0) crops are explicit completed empty results. Scores are masked
    to detector support. None preserves an absent score lane independently of
    an explicitly supplied all-zero score array.

    start_background_encoding moves the per-crop encode and append work to one
    FIFO thread; see that method for the ownership and error contract.
    """
    def __init__(self, root, model_name, view_name, native_shape, cutoffs, *,
                 frame_start, expected_frames, padding_count=0, backend='', identity=None,
                 scalar_owned_cutoff=None):
        self.root = _root(root)
        self.shape = _shape(native_shape)
        self.cutoffs = _cents(cutoffs)
        self.primary_cutoff=min(self.cutoffs)
        self.scalar_owned_cutoff=scalar_owned_cutoff
        if scalar_owned_cutoff is not None and (type(scalar_owned_cutoff) is not int or scalar_owned_cutoff!=self.primary_cutoff):
            raise ValueError('Scalar-owned cutoff must be the exact family floor')
        self.start, self.count, self.padding = int(frame_start), int(expected_frames), int(padding_count)
        if self.count < 0 or self.padding < 0 or self.count + self.padding <= 0 or not 0 <= self.start <= self.start + self.count <= self.shape[0]:
            raise ValueError('Invalid native fanout task frame range')
        if scalar_owned_cutoff is not None and self.padding:
            raise ValueError('Scalar-owned staging currently requires a real-frame-only task')
        self.product = _product(model_name, view_name, self.shape, identity)
        self.backend = str(backend)
        self.width = self.count + self.padding
        self.verify_integrity=os.environ.get('XTA_YOLO_SWEEP_NATIVE_VERIFY','0')=='1'
        dtype=_VERIFY_INDEX if self.verify_integrity else INDEX
        if len(self.cutoffs)*self.width*dtype.itemsize > 64*1024**2:
            raise SweepViewCapacityError('Native fanout range index exceeds its bounded workspace')
        self.index = np.zeros((len(self.cutoffs), self.width), dtype=dtype)
        self._cut = {c: i for i, c in enumerate(self.cutoffs)}
        self._lock = threading.RLock()
        self._closed = False
        self._reserved = 0
        self._used = 0
        self._digest = hashlib.sha256() if self.verify_integrity else None
        self._metrics = dict(full_plane_bounds_scans=0, encoded_crop_pixels=0,
                             payload_bytes=0, index_bytes=int(self.index.nbytes),alias_frames=0)
        # Background encoding state; None keeps the original synchronous put.
        self._encoder_jobs = None
        self._encoder_cond = None
        self._encoder_thread = None
        self._encoder_error = None
        self._encoder_discard = False
        self._encoder_pending_bytes = 0
        self._encoder_limit = 0
        _initialize(self.root)
        self.key = uuid4().hex
        self.path = self.root / 'ranges' / self.key
        self.path.mkdir(parents=True)
        try:
            self._fund(int(self.index.nbytes) + 4096)
            self._stream = (self.path / 'payload.bin').open('wb', buffering=1024*1024)
        except BaseException:
            self.path.rmdir()
            raise

    def _fund(self, needed):
        if needed <= self._reserved:
            return
        amount = ((int(needed) + _RESERVE_BYTES - 1) // _RESERVE_BYTES) * _RESERVE_BYTES
        extra = amount - self._reserved
        if shutil.disk_usage(self.root).free < extra + 64*1024**2:
            raise SweepViewCapacityError('Insufficient local disk headroom for native fanout')
        _quota(self.root, self.key, amount)
        self._reserved = amount

    def start_background_encoding(self, max_pending_bytes=None):
        """Encode and append crops on one FIFO thread; put then validates and reserves only.

        A single thread consumes jobs in exactly the order put and put_alias
        reserved them, so payload bytes, index rows and payload offsets are those
        of synchronous puts. From this call on, put takes ownership of its mask
        and score arrays: callers must hand over private arrays and never mutate
        them. Pending arrays are bounded by ``max_pending_bytes`` (default
        XTA_YOLO_SWEEP_NATIVE_WRITE_QUEUE_MIB, 32 MiB); put waits for room. A
        failed encode poisons the task: the next put, put_alias or finish raises
        that error, and abort discards queued work after the job in progress.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError('Native fanout writer is closed')
            if self._encoder_jobs is not None:
                return
            if max_pending_bytes is None:
                try:
                    mib = float(os.environ.get('XTA_YOLO_SWEEP_NATIVE_WRITE_QUEUE_MIB', _ENCODER_QUEUE_MIB))
                except ValueError:
                    mib = _ENCODER_QUEUE_MIB
                if not math.isfinite(mib) or mib < 0:
                    mib = _ENCODER_QUEUE_MIB
                max_pending_bytes = int(mib * 1024**2)
            self._encoder_limit = max(0, int(max_pending_bytes))
            self._encoder_cond = threading.Condition(threading.Lock())
            self._encoder_jobs = deque()

    @property
    def background_encoding(self):
        return self._encoder_jobs is not None

    def _raise_encoder_error(self):
        if self._encoder_error is not None:
            raise self._encoder_error

    def _submit(self, job, nbytes):
        """Queue one reserved job in reservation order; called with self._lock held."""
        cond = self._encoder_cond
        with cond:
            while (self._encoder_jobs and self._encoder_error is None
                   and self._encoder_pending_bytes + nbytes > self._encoder_limit):
                cond.wait()
            self._raise_encoder_error()
            self._encoder_jobs.append((job, nbytes))
            self._encoder_pending_bytes += nbytes
            if self._encoder_thread is None:
                thread = threading.Thread(target=self._encoder_main, daemon=True,
                                          name='xta-native-fanout-encoder')
                self._encoder_thread = thread
                thread.start()
            else:
                cond.notify_all()

    def _encoder_main(self):
        cond, jobs = self._encoder_cond, self._encoder_jobs
        while True:
            with cond:
                while not jobs:
                    # An idle or closed writer's thread exits; a later submit starts another.
                    if self._closed or (not cond.wait(_ENCODER_IDLE_SECONDS) and not jobs):
                        self._encoder_thread = None
                        return
                job, nbytes = jobs[0]
            try:
                if self._encoder_error is None and not self._encoder_discard:
                    self._run_encoder_job(job)
            except BaseException as error:  # surfaced to the producer by put/finish
                if self._encoder_error is None:
                    self._encoder_error = error
            finally:
                with cond:
                    jobs.popleft()
                    self._encoder_pending_bytes -= nbytes
                    cond.notify_all()
                job = None  # an idle thread retains no crop

    def _run_encoder_job(self, job):
        if job[0] == 'crop':
            _, position, slot, values, scores, y0, y1, x0, x1 = job
            self._append_crop(self.index[position, slot], values, scores, y0, y1, x0, x1)
        else:
            _, position, slot, source_position, source_slot, frame, predictions, present = job
            source = self.index[source_position, source_slot]
            if source['state'] not in (1, 2):
                raise ValueError('Native alias source is not complete')
            self.index[position, slot] = source
            destination = self.index[position, slot]
            destination['frame'] = frame
            destination['predictions'], destination['frames_with_predictions'] = predictions, present

    def _wait_encoder_idle(self):
        if self._encoder_jobs is None:
            return
        with self._encoder_cond:
            while self._encoder_jobs:
                self._encoder_cond.wait()

    def _release_encoder(self):
        """Let an idle encoder thread observe the closed writer and exit now."""
        if self._encoder_cond is not None:
            with self._encoder_cond:
                self._encoder_cond.notify_all()

    def _append_crop(self, row, values, scores, y0, y1, x0, x1):
        """Encode one reserved crop at the current payload end and complete its row."""
        row['offset'] = self._used
        frame_digest=hashlib.sha256() if self.verify_integrity else None
        step = max(1, _CHUNK_BYTES // max(1, x1-x0))
        for first in range(0, y1-y0, step):
            stop = min(y1-y0, first+step)
            piece = values[first:stop]
            if np.any(piece > 1):
                raise ValueError('Native detector masks must contain only 0/1')
            if not np.any(piece):
                continue
            packed = np.packbits(piece, axis=None, bitorder='little').tobytes()
            mask_bytes = zlib.compress(packed, 1)
            score_raw = b'' if scores is None else np.where(piece != 0, scores[first:stop], np.uint8(0)).tobytes()
            score_bytes = b'' if scores is None else zlib.compress(score_raw, 1)
            mc=sc=0
            if self.verify_integrity:
                prefix=struct.pack('<8I',y0+first,y0+stop,x0,x1,len(mask_bytes),len(score_bytes),len(packed),len(score_raw))
                checksum=zlib.crc32(prefix)
                mc,sc=zlib.crc32(mask_bytes,checksum),zlib.crc32(score_bytes,checksum)
            head = _CHUNK.pack(y0+first, y0+stop, x0, x1, len(mask_bytes), len(score_bytes),
                mc,sc,len(packed),len(score_raw))
            self._fund(int(self.index.nbytes) + 4096 + self._used + len(head)+len(mask_bytes)+len(score_bytes))
            for data in (head, mask_bytes, score_bytes):
                self._stream.write(data)
                if self.verify_integrity:
                    self._digest.update(data)
                    frame_digest.update(data)
                self._used += len(data)
            row['chunks'] += 1
            self._metrics['encoded_crop_pixels'] += int(piece.size)
        row['length'] = self._used - int(row['offset'])
        if self.verify_integrity:row['sha256']=np.frombuffer(frame_digest.digest(),np.uint8)
        row['state'] = 2 if row['chunks'] else 1

    def put(self, cutoff, frame_index, mask, score=None, *, prediction_count=0,
            frames_with_predictions=None, bbox=None, is_padding=False, padding_ordinal=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('Native fanout writer is closed')
            self._raise_encoder_error()
            if cutoff not in self._cut:
                raise ValueError('Cutoff is outside the native task family')
            if cutoff==self.scalar_owned_cutoff:
                raise SweepViewScalarOwnedError('Record scalar-owned floor counts instead of duplicate pixels')
            frame = int(frame_index)
            if not 0 <= frame < self.shape[0]:
                raise ValueError('Native fanout global frame is out of bounds')
            if is_padding:
                if padding_ordinal is None or not 0 <= int(padding_ordinal) < self.padding:
                    raise ValueError('Padding needs its unique expected ordinal')
                slot = self.count + int(padding_ordinal)
            else:
                if not self.start <= frame < self.start + self.count:
                    raise ValueError('Native fanout real frame is outside its task lease')
                slot = frame - self.start
            row = self.index[self._cut[cutoff], slot]
            if row['state']:
                raise ValueError('Duplicate native fanout frame')
            predictions = int(prediction_count)
            present = int(predictions > 0) if frames_with_predictions is None else int(frames_with_predictions)
            if predictions < 0 or present not in (0, 1):
                raise ValueError('Invalid native prediction statistics')
            values = np.asarray(mask)
            scores = None if score is None else np.asarray(score)
            if values.dtype != np.uint8 or values.ndim != 2 or (scores is not None and (scores.dtype != np.uint8 or scores.shape != values.shape)):
                raise ValueError('Native mask/scores must be matching 2-D uint8 arrays')
            h, w = self.shape[1:]
            if bbox is None:
                if values.shape != (h, w):
                    if values.shape == (0, 0):
                        bbox = (0, 0, 0, 0)
                    else:
                        raise ValueError('Cropped native masks need explicit bounds')
                else:
                    self._metrics['full_plane_bounds_scans'] += 1
                    ys, xs = np.flatnonzero(values.any(axis=1)), np.flatnonzero(values.any(axis=0))
                    bbox = (int(ys[0]), int(ys[-1])+1, int(xs[0]), int(xs[-1])+1) if ys.size else (0, 0, 0, 0)
            y0, y1, x0, x1 = map(int, bbox)
            if not (0 <= y0 <= y1 <= h and 0 <= x0 <= x1 <= w):
                raise ValueError('Native fanout bbox exceeds its grid')
            if values.shape == (h, w):
                values = values[y0:y1, x0:x1]
                scores = None if scores is None else scores[y0:y1, x0:x1]
            if values.shape != (y1-y0, x1-x0):
                raise ValueError('Native fanout crop differs from bbox')
            if x1-x0>_CHUNK_BYTES:
                raise SweepViewCapacityError('Native crop row exceeds the bounded strip workspace')
            row['state'] = 3  # unfinished writes can never be acknowledged
            row['padding'], row['score'], row['frame'] = bool(is_padding), scores is not None, frame
            row['predictions'], row['frames_with_predictions'] = predictions, present
            row['y0'], row['y1'], row['x0'], row['x1'] = y0, y1, x0, x1
            if self._encoder_jobs is not None:
                self._submit(('crop', self._cut[cutoff], slot, values, scores, y0, y1, x0, x1),
                             int(values.nbytes) + (0 if scores is None else int(scores.nbytes)))
                return
            self._append_crop(row, values, scores, y0, y1, x0, x1)

    def put_alias(self,cutoff,frame_index,*,source_cutoff,source_frame_index=None,
                  prediction_count=None,frames_with_predictions=None):
        """Reuse a caller-certified identical real-frame mask AND score product.

        This performs no pixel comparison, scan, compression, or payload write.
        A decoder may call it only when it already knows the native products
        are identical. Prediction statistics can still differ across cutoffs.
        """
        with self._lock:
            if self._closed:raise RuntimeError('Native fanout writer is closed')
            self._raise_encoder_error()
            if cutoff==self.scalar_owned_cutoff or source_cutoff==self.scalar_owned_cutoff:
                raise SweepViewScalarOwnedError('Scalar-owned floor has no staged pixels to alias')
            source_frame=frame_index if source_frame_index is None else source_frame_index
            if cutoff not in self._cut or source_cutoff not in self._cut or not self.start<=int(frame_index)<self.start+self.count or not self.start<=int(source_frame)<self.start+self.count:
                raise ValueError('Native alias is outside the real task lease')
            destination=self.index[self._cut[cutoff],int(frame_index)-self.start]
            source=self.index[self._cut[source_cutoff],int(source_frame)-self.start]
            if destination['state']:raise ValueError('Duplicate native fanout frame')
            # A reserved (3) source is still queued for the FIFO encoder.
            queued=self._encoder_jobs is not None and source['state']==3
            if source['state'] not in (1,2) and not queued:raise ValueError('Native alias source is not complete')
            predictions=int(source['predictions']) if prediction_count is None else int(prediction_count)
            present=int(source['frames_with_predictions']) if frames_with_predictions is None else int(frames_with_predictions)
            if predictions<0 or present not in (0,1):raise ValueError('Invalid native alias prediction statistics')
            if self._encoder_jobs is not None:
                destination['state']=3
                self._metrics['alias_frames']+=1
                self._submit(('alias',self._cut[cutoff],int(frame_index)-self.start,self._cut[source_cutoff],
                              int(source_frame)-self.start,int(frame_index),predictions,present),0)
                return
            self.index[self._cut[cutoff],int(frame_index)-self.start]=source
            destination=self.index[self._cut[cutoff],int(frame_index)-self.start]
            destination['frame']=int(frame_index)
            destination['predictions'],destination['frames_with_predictions']=predictions,present
            self._metrics['alias_frames']+=1

    def put_empty_range(self,cutoff,frame_start,frame_stop,*,score_lane=False,
                        prediction_counts=0,frames_with_predictions=0):
        """Record a caller-known empty real-frame run with one metadata operation."""
        with self._lock:
            if self._closed:raise RuntimeError('Native fanout writer is closed')
            if cutoff==self.scalar_owned_cutoff:
                raise SweepViewScalarOwnedError('Scalar-owned floor is not an empty pixel result')
            first,stop=int(frame_start),int(frame_stop)
            if cutoff not in self._cut or not self.start<=first<=stop<=self.start+self.count:
                raise ValueError('Empty native run is outside its real task lease')
            rows=self.index[self._cut[cutoff],first-self.start:stop-self.start]
            if np.any(rows['state']):raise ValueError('Duplicate native fanout frame')
            counts=np.broadcast_to(np.asarray(prediction_counts),(stop-first,))
            present=np.broadcast_to(np.asarray(frames_with_predictions),(stop-first,))
            if np.any(counts<0) or np.any((present!=0)&(present!=1)):
                raise ValueError('Invalid empty native run prediction statistics')
            rows['frame']=np.arange(first,stop,dtype=np.uint32)
            rows['predictions'],rows['frames_with_predictions']=counts,present
            rows['score']=bool(score_lane)
            rows['state']=1
            if self.verify_integrity:rows['sha256']=np.frombuffer(hashlib.sha256(b'').digest(),np.uint8)

    def put_scalar_owned_range(self,cutoff,frame_start,frame_stop,*,
                               prediction_counts,frames_with_predictions):
        """Record exact scalar-owned counts without reading or storing pixels."""
        with self._lock:
            if self._closed:raise RuntimeError('Native fanout writer is closed')
            first,stop=int(frame_start),int(frame_stop)
            if cutoff!=self.scalar_owned_cutoff or not self.start<=first<=stop<=self.start+self.count:
                raise ValueError('Scalar-owned run differs from its explicit floor/lease')
            counts=np.asarray(prediction_counts);present=np.asarray(frames_with_predictions)
            if counts.shape!=(stop-first,) or present.shape!=(stop-first,) or not np.issubdtype(counts.dtype,np.integer) or not (np.issubdtype(present.dtype,np.integer) or np.issubdtype(present.dtype,np.bool_)) or np.any(counts<0) or np.any((present!=0)&(present!=1)):
                raise ValueError('Scalar-owned runs require exact per-frame integer statistics')
            rows=self.index[self._cut[cutoff],first-self.start:stop-self.start]
            if np.any(rows['state']):raise ValueError('Duplicate native fanout frame')
            rows['frame']=np.arange(first,stop,dtype=np.uint32)
            rows['predictions'],rows['frames_with_predictions']=counts,present
            rows['state']=4

    def _metadata(self, status, reason=''):
        return dict(schema=SCHEMA, status=status, reason=str(reason), product=self.product,
            product_sha256=_hash(self.product), range_id=self.key, backend=self.backend,
            cutoffs=list(self.cutoffs), frame_start=self.start, frame_count=self.count,
            padding_count=self.padding, index_record_bytes=self.index.dtype.itemsize,
            index_shape=list(self.index.shape), payload_bytes=self._used,
            payload_sha256=self._digest.hexdigest() if self.verify_integrity else None,
            verify_integrity=self.verify_integrity,scalar_owned_cutoff=self.scalar_owned_cutoff,
            metrics=dict(self._metrics))

    def _receipt(self, status, metadata):
        return dict(status=status, path=str(self.path), metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                    model_name=self.product['model_name'], view_name=self.product['view_name'],
                    native_shape=list(self.shape), cutoffs=list(self.cutoffs),scalar_owned_cutoff=self.scalar_owned_cutoff)

    def finish(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('Native fanout writer is closed')
            self._wait_encoder_idle()
            self._raise_encoder_error()
            if np.any((self.index['state'] != 1) & (self.index['state'] != 2) & (self.index['state'] != 4)):
                raise SweepViewIntegrityError('Native task did not cover all expected real/padding frames')
            self._stream.close()
            raw = self.index.tobytes()
            with (self.path / 'index.bin').open('wb') as stream:
                stream.write(raw)
            meta = self._metadata('complete')
            meta['index_sha256'] = hashlib.sha256(raw).hexdigest()
            self._metrics['payload_bytes'] = self._used
            meta['metrics'] = dict(self._metrics)
            write_json_atomic(self.path / 'metadata.json', meta)
            _quota(self.root, self.key, self._used + len(raw) + (self.path/'metadata.json').stat().st_size)
            self._closed = True
            self._release_encoder()
            return self._receipt('complete', (self.path / 'metadata.json').read_bytes())

    def abort(self, reason):
        with self._lock:
            if self._closed:
                raise RuntimeError('Cannot abort an acknowledged native task')
            # Queued crops will be deleted; only the job in progress is awaited.
            self._encoder_discard = True
            self._wait_encoder_idle()
            self._stream.close()
            for name in ('payload.bin', 'index.bin'):
                (self.path / name).unlink(missing_ok=True)
            self._closed = True
            self._release_encoder()
            meta = self._metadata('fallback', reason)
            meta['payload_bytes'] = 0
            write_json_atomic(self.path / 'metadata.json', meta)
            _quota(self.root, self.key, (self.path/'metadata.json').stat().st_size)
            return self._receipt('fallback', (self.path / 'metadata.json').read_bytes())


def _range(root, receipt, product, cuts):
    path = Path(receipt.get('path', '')).resolve()
    if path.parent != (root / 'ranges').resolve() or not re.fullmatch('[0-9a-f]{32}', path.name):
        raise SweepViewIntegrityError('Acknowledged native range escapes its owner')
    try:
        raw = (path / 'metadata.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt['metadata_sha256']:
            raise SweepViewIntegrityError('Acknowledged native metadata changed')
        meta = json.loads(raw)
        if meta['schema'] != SCHEMA or meta['product'] != product or meta['product_sha256'] != _hash(product) or tuple(meta['cutoffs']) != cuts:
            raise SweepViewIntegrityError('Native product identity, shape or cutoff family differs')
        if meta['status'] != 'complete':
            raise SweepViewIntegrityError('Complete acknowledgment points to an incomplete native task')
        index_path=path/'index.bin'
        stamp=index_path.stat()
        dtype=_VERIFY_INDEX if meta.get('verify_integrity',False) else INDEX
        expected_shape=(len(cuts), int(meta['frame_count'])+int(meta['padding_count']))
        if tuple(meta['index_shape'])!=expected_shape or meta['index_record_bytes']!=dtype.itemsize or stamp.st_size!=math.prod(expected_shape)*dtype.itemsize:
            raise SweepViewIntegrityError('Native range index shape is invalid')
        cache_key=(str(index_path),receipt['metadata_sha256'],meta['index_sha256'],stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns)
        with _INDEX_CACHE_LOCK:
            flat=_INDEX_CACHE.get(cache_key)
            if flat is None:
                index_raw=index_path.read_bytes()
                if hashlib.sha256(index_raw).hexdigest()!=meta['index_sha256']:
                    raise SweepViewIntegrityError('Native range index changed')
                flat=np.frombuffer(index_raw,dtype=dtype)
                _INDEX_CACHE[cache_key]=flat
        if (path/'payload.bin').stat().st_size != meta['payload_bytes']:
            raise SweepViewIntegrityError('Native range payload is missing or truncated')
        index = flat.reshape(meta['index_shape'])
        with _INDEX_CACHE_LOCK:
            validated=_INDEX_VALIDATION_CACHE.get(cache_key)
            if validated is None or validated() is not flat:
                if np.any((index['state'] != 1) & (index['state'] != 2) & (index['state'] != 4)):
                    raise SweepViewIntegrityError('Native range contains unfinished records')
                owned=meta.get('scalar_owned_cutoff')
                if owned is not None and (type(owned) is not int or owned!=min(cuts)):
                    raise SweepViewIntegrityError('Scalar-owned marker is not the family floor')
                for position,cent in enumerate(cuts):
                    if owned==cent:
                        if np.any(index[position]['state']!=4):
                            raise SweepViewIntegrityError('Scalar-owned floor contains pixel or missing records')
                    elif np.any(index[position]['state']==4):
                        raise SweepViewIntegrityError('Unexpected scalar-owned marker in a pixel lane')
                owned_rows=index[index['state']==4]
                if np.any(owned_rows['length']) or np.any(owned_rows['chunks']) or np.any(owned_rows['score']):
                    raise SweepViewIntegrityError('Scalar-owned marker illegally references pixel data')
                if any(np.any(owned_rows[name]) for name in ('y0','y1','x0','x1')):
                    raise SweepViewIntegrityError('Scalar-owned marker illegally contains pixel bounds')
                if np.any(index['frame']>=product['native_shape'][0]) or np.any(index['offset']>meta['payload_bytes']) or np.any(index['length']>meta['payload_bytes']-index['offset']):
                    raise SweepViewIntegrityError('Native range frame or payload address is out of bounds')
                if np.any((index['state']==1)&((index['chunks']!=0)|(index['length']!=0))) or np.any((index['state']==2)&(index['chunks']==0)) or np.any(index['score']>1):
                    raise SweepViewIntegrityError('Native empty flags or score lane flags are invalid')
                real=int(meta['frame_count'])
                expected_frames=np.arange(int(meta['frame_start']),int(meta['frame_start'])+real,dtype=np.uint32)
                if np.any(index[:,:real]['frame']!=expected_frames[None,:]) or np.any(index[:,:real]['padding']) or np.any(index[:,real:]['padding']!=1):
                    raise SweepViewIntegrityError('Native range logical frame mapping differs')
                _INDEX_VALIDATION_CACHE[cache_key]=weakref.ref(flat,lambda _ref,key=cache_key:_INDEX_VALIDATION_CACHE.pop(key,None))
        return meta, index, dict(receipt)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise SweepViewIntegrityError(f'Cannot verify acknowledged native range: {path}') from exc


def seal_view_family(model_name, view_name, native_shape, expected_cutoffs, *,
                     receipts, identity=None, unsupported_reason=None, root=None):
    root = _root(root)
    _initialize(root)
    product, cuts = _product(model_name, view_name, native_shape, identity), _cents(expected_cutoffs)
    destination = _view_dir(root, model_name, view_name)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'metadata.json').exists():
        raise FileExistsError('Native view family was already decided')
    complete, reasons = [], []
    try:
        # Verify complete acknowledgments even when another route requests fallback.
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise SweepViewIntegrityError('Missing worker native fanout observation')
            if receipt.get('status') == 'complete':
                complete.append(_range(root, receipt, product, cuts))
            elif receipt.get('status') in ('unsupported', 'fallback'):
                for key, expected in [('model_name', str(model_name)), ('view_name', str(view_name)), ('native_shape', list(native_shape))]:
                    if key in receipt and receipt[key] != expected:
                        raise SweepViewIntegrityError('Unsupported observation belongs to another native view')
                reasons.append(str(receipt.get('reason') or 'unsupported producer'))
            else:
                raise SweepViewIntegrityError('Unrecognized worker native fanout status')
        if unsupported_reason:
            reasons.append(str(unsupported_reason))
        if not complete and not reasons:
            raise SweepViewIntegrityError('No acknowledged native ranges or explicit unsupported reason')
        ordered = sorted(complete, key=lambda item: item[0]['frame_start'])
        owners={m.get('scalar_owned_cutoff') for m,_,_ in ordered}
        if len(owners)>1:
            raise SweepViewIntegrityError('Native ranges disagree about scalar pixel ownership')
        scalar_owned=next(iter(owners)) if owners else None
        counts, frames, foreground = {}, {}, {}
        if not reasons:
            expected = 0
            for meta, _index, _receipt in ordered:
                if int(meta['frame_start']) != expected:
                    raise SweepViewIntegrityError('Native view ranges overlap or leave gaps')
                expected += int(meta['frame_count'])
            if expected != product['native_shape'][0]:
                raise SweepViewIntegrityError('Native view ranges do not cover the complete grid')
            for pos, cent in enumerate(cuts):
                counts[cent] = sum(int(index[pos, :meta['frame_count']]['predictions'].sum()) for meta, index, _ in ordered)
                frames[cent] = sum(int(index[pos, :meta['frame_count']]['frames_with_predictions'].sum()) for meta, index, _ in ordered)
                foreground[cent] = None if cent==scalar_owned else any(np.any(index[pos]['state'] == 2) for _meta, index, _ in ordered)
        diagnostic = dict(status='fallback' if reasons else 'ready', prediction_counts=counts,
            frames_with_predictions=frames, has_foreground=foreground,
            reason='; '.join(reasons), stored_bytes=sum(meta['payload_bytes']+index.nbytes for meta,index,_ in ordered))
        metadata = dict(schema=SCHEMA, product=product, cutoffs=list(cuts), ranges=[r for _,_,r in ordered],
                        scalar_owned_cutoff=scalar_owned,**diagnostic)
        write_json_atomic(destination/'metadata.json', metadata)
        if reasons:
            # The fallback decision is durable before disposing of unused,
            # verified complete products. Never touch writing/unobserved ranges.
            retired=[]
            for _meta,_index,receipt in ordered:
                path=Path(receipt['path'])
                for name in ('payload.bin','index.bin','metadata.json'):
                    (path/name).unlink(missing_ok=True)
                path.rmdir()
                _quota(root,path.name,0)
                retired.append(dict(range_id=path.name,metadata_sha256=receipt['metadata_sha256']))
            metadata['ranges']=[]
            metadata['retired_completed_ranges']=retired
            metadata['discarded_bytes']=diagnostic['stored_bytes']
            metadata['stored_bytes']=0
            diagnostic['discarded_bytes']=diagnostic['stored_bytes']
            diagnostic['stored_bytes']=0
            write_json_atomic(destination/'metadata.json',metadata)
        return diagnostic
    except BaseException as exc:
        write_json_atomic(destination/'metadata.json', dict(schema=SCHEMA, product=product,
            cutoffs=list(cuts), status='failed', reason=str(exc)))
        raise


def seal_view(model_name, view_name, cutoff, native_shape, *, receipts, identity=None, root=None):
    return seal_view_family(model_name, view_name, native_shape, (cutoff,), receipts=receipts, identity=identity, root=root)


def _decision(model_name, view_name, cutoff, native_shape=None, identity=None, root=None):
    root = _root(root)
    path = _view_dir(root, model_name, view_name)/'metadata.json'
    if not path.exists():
        if os.environ.get('XTA_YOLO_SWEEP_NATIVE_ROLE') == 'consume':
            raise SweepViewIntegrityError('Missing native view decision in consumer branch')
        return root, None
    try:
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta.get('schema')!=SCHEMA or meta.get('status') not in ('ready','fallback','projected','consumed','failed'):
            raise SweepViewIntegrityError('Native view decision has an invalid schema or state')
        p = meta['product']
        expected_shape = p['native_shape'] if native_shape is None else native_shape
        if p != _product(model_name, view_name, expected_shape, identity) or cutoff not in meta['cutoffs']:
            raise SweepViewIntegrityError('Native view identity, shape or cutoff differs')
        if meta['status'] == 'failed':
            raise SweepViewIntegrityError(meta['reason'])
        return root, meta
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SweepViewIntegrityError('Native view decision is corrupt') from exc


def view_family_status(model_name,view_name,cutoff,*,native_shape=None,identity=None,root=None):
    """Return the explicit native decision; missing consume decisions are fatal."""
    _,meta=_decision(model_name,view_name,cutoff,native_shape,identity,root)
    if meta is not None and meta['status']=='ready' and meta.get('scalar_owned_cutoff')==cutoff:
        return 'scalar_owned'
    return None if meta is None else meta['status']


class _PositionalPayload:
    """Slice-only positional reads of a payload that cannot be memory-mapped.

    Used by the Python reader when a filesystem refuses shared mappings; each
    slice is one positional read, so no stream position is shared.
    """
    __slots__ = ('_path', '_size')

    def __init__(self, path, size):
        self._path, self._size = path, int(size)

    def __len__(self):
        return self._size

    def __getitem__(self, part):
        start, stop, step = part.indices(self._size)
        if step != 1:
            raise ValueError('Positional payload reads are contiguous')
        count = max(0, stop-start)
        with open(self._path, 'rb') as stream:
            if hasattr(os, 'pread'):
                return os.pread(stream.fileno(), count, start)
            stream.seek(start)
            return stream.read(count)


class _PayloadMapping:
    """A read-only mapping of one acknowledged payload file.

    Readers slice it positionally, so no stream position or lock is shared
    between frames or held across yields. Each array slice keeps the mapping
    exported, so ``close`` can never unmap bytes that a decode still reads; it
    then leaves the unmap to the last reference. A filesystem that refuses a
    shared mapping gets positional reads (Python reader only).
    """
    __slots__ = ('array', '_mapping')

    def __init__(self, path):
        with open(path, 'rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            # An empty file cannot be mapped; reading it fails exactly as a stream does.
            try:
                self._mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) if size else None
            except (OSError, ValueError):
                self._mapping = None
                self.array = _PositionalPayload(path, size)
                return
        self.array = np.frombuffer(self._mapping, np.uint8) if size else np.empty(0, np.uint8)

    def close(self):
        self.array = None
        mapping, self._mapping = self._mapping, None
        if mapping is not None:
            try:
                mapping.close()
            except BufferError:
                pass  # a decode still holds a slice; the mapping closes with it


class _MaskFrameCrops:
    """Lazy ``iter_mask_crops`` result: five-tuples, plus a fused native rank merge.

    Iterating it behaves exactly like the generator it replaces. A projection
    rank input may instead call ``merge_rank_level`` first: rows the native
    route accepts are merged into the plane in one nogil call each and only
    the remaining rows (decoded by the Python reader) are then yielded.
    """
    __slots__ = ('_reader', '_z', '_pending', '_iterator')

    def __init__(self, reader, z):
        self._reader, self._z = reader, z
        self._pending = None
        self._iterator = None

    def _rows(self):
        if self._pending is None:
            self._pending = [(row, path) for row, path in self._reader._rows(self._z) if row['state'] != 1]
        return self._pending

    @property
    def wants_rank_plane(self):
        """True when ``merge_rank_level`` can take at least one row of this frame."""
        reader = self._reader
        return (self._iterator is None and reader._decoder is not None
                and any(not reader._verification[path] for _row, path in self._rows()))

    def merge_rank_level(self, plane, level, check_previous):
        """Merge native rows like per-crop ``_merge_rank_crop``; return (populated, gap, box).

        ``box`` is (top, bottom, left, right) over populated strips. Rows of
        one payload merge in one nogil call; rows the native route declines
        stay in this iterator for the caller's own loop.
        """
        if self._iterator is not None:
            raise RuntimeError('Native rank merge must precede crop iteration')
        reader = self._reader
        height, width = reader.shape[1:]
        populated = gap = False
        top, bottom, left, right = height, 0, width, 0
        groups = {}
        for row, path in self._rows():
            groups.setdefault(path, []).append(row)
        remaining = []
        for path, rows in groups.items():
            # Within one level the merge order of rows cannot change any result.
            result = reader._merge_rows(path, [(row, level, check_previous) for row in rows], plane)
            if result is None:
                remaining.extend((row, path) for row in rows)
                continue
            outcome, (row_top, row_bottom, row_left, row_right) = result
            reader._count_read(int(outcome[:, 2].sum()))
            gap = gap or bool(outcome[:, 1].any())
            if outcome[:, 0].any():
                populated = True
                top, bottom = min(top, row_top), max(bottom, row_bottom)
                left, right = min(left, row_left), max(right, row_right)
        self._pending = remaining
        return populated, gap, (top, bottom, left, right)

    def merge_rank_frame(self, frames, plane, checks):
        """Merge every level of one frame in a single nogil call, or return None.

        ``frames[i]`` is level ``i+1``'s ``iter_mask_crops`` result and
        ``checks[i]`` its previous-level gap test; levels merge in order, as
        the per-level loop does. Applies only when every row is native and in
        one payload (a frame's real rows of all cutoffs share their task range);
        otherwise nothing is touched and the caller merges level by level.
        Returns (populated by level, gap by level, box over populated strips).
        """
        entries, owners, path = [], [], None
        for level, (frame, check) in enumerate(zip(frames, checks), 1):
            if type(frame) is not _MaskFrameCrops or frame._iterator is not None:
                return None
            reader = frame._reader
            if reader._decoder is None:
                return None
            for row, row_path in frame._rows():
                if reader._verification[row_path] or (path is not None and row_path != path):
                    return None
                path = row_path
                entries.append((row, level, check))
                owners.append(reader)
        if not entries:
            return None
        result = owners[0]._merge_rows(path, entries, plane)
        if result is None:
            return None
        outcome, box = result
        populated, gaps = [False]*len(frames), [False]*len(frames)
        for (_row, level, _check), owner, (row_populated, row_gap, read) in zip(entries, owners, outcome.tolist()):
            owner._count_read(read)
            populated[level-1] = populated[level-1] or bool(row_populated)
            gaps[level-1] = gaps[level-1] or bool(row_gap)
        for frame in frames:
            frame._pending = []
        return populated, gaps, box

    def _generate(self):
        reader = self._reader
        for row, path in self._rows():
            for y0, y1, x0, x1, mask, _ in reader._row_crops(row, path, False):
                yield y0, y1, x0, x1, mask

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = self._generate()
        return next(self._iterator)

    def close(self):
        if self._iterator is not None:
            self._iterator.close()


class NativeViewReader:
    def __init__(self, root, meta, cutoff):
        self.root, self.metadata, self.cutoff = root, meta, cutoff
        self.shape = tuple(meta['product']['native_shape'])
        self.prediction_count = int(meta['prediction_counts'][str(cutoff)])
        self.frames_with_predictions = int(meta['frames_with_predictions'][str(cutoff)])
        self.has_foreground = bool(meta['has_foreground'][str(cutoff)])
        self._closed, self._ranges = False, []
        # Guards only the mapping cache and metrics; never held across a decode or yield.
        self._state_lock=threading.Lock()
        self._payloads=OrderedDict()
        self._padding={}
        self._verification={}
        self.io_metrics=dict(payload_open_calls=0,payload_read_bytes=0)
        for receipt in meta['ranges']:
            m, index, _ = _range(root, receipt, meta['product'], tuple(meta['cutoffs']))
            pos = m['cutoffs'].index(cutoff)
            rows = index[pos]
            path=Path(receipt['path'])/'payload.bin'
            self._verification[path]=bool(m.get('verify_integrity',False))
            self._ranges.append((m, rows, path))
            for row in rows[m['frame_count']:]:
                self._padding.setdefault(int(row['frame']),[]).append((row,path))
        self._starts = [m['frame_start'] for m,_,_ in self._ranges]
        self.has_score_lane = any(np.any(rows['score']) for _,rows,_ in self._ranges)
        # Integrity mode stays on the Python reader, which owns crc/sha checks.
        self._decoder=None
        if not all(self._verification.values()):
            from .sweep_view_decode import native_view_decoder
            self._decoder=native_view_decoder()[0]

    def _rows(self, z):
        if self._closed:
            raise RuntimeError('Native view reader is closed')
        if not 0 <= int(z) < self.shape[0]:
            raise IndexError('Native view frame is out of range')
        selected = []
        pos = bisect_right(self._starts, int(z))-1
        if pos >= 0:
            meta, rows, path = self._ranges[pos]
            local = int(z)-meta['frame_start']
            if local < meta['frame_count']:
                selected.append((rows[local], path))
        # Snapshot padding rows now: close() clears them, and a suspended
        # iterator resumed after close must fail on its next read, not end early.
        selected.extend(self._padding.get(int(z),()))
        yield from selected

    def _payload(self,path):
        """Return the mapped payload array; at most two mappings stay cached."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError('Native view reader is closed')
            entry=self._payloads.pop(path,None)
            if entry is None:
                entry=_PayloadMapping(path)
                self.io_metrics['payload_open_calls']+=1
            self._payloads[path]=entry
            evicted=[]
            while len(self._payloads)>2:
                evicted.append(self._payloads.popitem(last=False)[1])
            array=entry.array
        for old in evicted:
            old.close()
        return array

    def _count_read(self,nbytes):
        with self._state_lock:
            self.io_metrics['payload_read_bytes']+=nbytes

    def _row_values(self,row):
        """The kernels' row parameters (sweep_view_decode.ROW_PARAMETERS)."""
        return (int(row['offset']),int(row['length']),int(row['chunks']),int(row['y0']),
                int(row['y1']),int(row['x0']),int(row['x1']),self.shape[1],self.shape[2],
                int(row['score']),_CHUNK_BYTES)

    def _row_params(self,row):
        return np.array(self._row_values(row),np.int64)

    def _native_payload(self,path):
        """The mapped payload for the native route, or None to use the Python reader."""
        if self._decoder is None or self._verification[path]:
            return None
        try:
            payload=self._payload(path)
        except OSError:
            return None  # the Python reader reports the failure exactly
        return payload if isinstance(payload,np.ndarray) else None

    def _decode_row(self,row,path,include_scores):
        """Decode one row in a single nogil call; None means use the Python reader."""
        payload=self._native_payload(path)
        if payload is None:
            return None
        status,bounds,starts,masks,scores,read=self._decoder.decode_row(
            payload,self._row_params(row),include_scores)
        if status:
            return None
        self._count_read(int(read))
        masks.setflags(write=False)
        with_scores=bool(include_scores and row['score'])
        if with_scores:
            scores.setflags(write=False)
        crops=[]
        for start,(y0,y1,x0,x1) in zip(starts.tolist(),bounds.tolist()):
            part=slice(start,start+(y1-y0)*(x1-x0))
            mask=masks[part].reshape(y1-y0,x1-x0)
            crops.append((y0,y1,x0,x1,mask,scores[part].reshape(mask.shape) if with_scores else None))
        return crops

    def _merge_rows(self,path,entries,plane):
        """Merge (row, level, check) entries of one payload in one nogil call.

        Returns (outcome rows (populated, gap, read bytes), box) or None when
        declined, in which case the plane is unchanged.
        """
        payload=self._native_payload(path)
        if payload is None or type(plane) is not np.ndarray or plane.dtype!=np.uint8 \
                or plane.ndim!=2 or not plane.flags.c_contiguous or not plane.flags.writeable:
            return None
        table=np.array([(*self._row_values(row),int(level),int(bool(check))) for row,level,check in entries],np.int64)
        status,outcome,top,bottom,left,right=self._decoder.merge_rank_rows(payload,table,plane)
        if status:
            return None
        return outcome,(int(top),int(bottom),int(left),int(right))

    def _python_row_crops(self,row,path,include_scores):
        """Specification reader for one row: positional slices of the mapped payload.

        Reads, checks, messages and partial yields are those of the original
        seek/read stream reader, including reads that run past the row.
        """
        try:
            payload=None
            position=int(row['offset'])
            consumed = 0
            verify=self._verification[path]
            frame_digest=hashlib.sha256() if verify else None
            for _ in range(int(row['chunks'])):
                if payload is None:
                    payload=self._payload(path)
                header = payload[position:position+_CHUNK.size]
                position += len(header)
                if len(header) != _CHUNK.size:
                    raise SweepViewIntegrityError('Truncated native crop header')
                y0,y1,x0,x1,ml,sl,mc,sc,mraw,sraw = _CHUNK.unpack(header)
                area = (y1-y0)*(x1-x0)
                if not (int(row['y0']) <= y0 < y1 <= int(row['y1']) <= self.shape[1] and int(row['x0']) <= x0 < x1 <= int(row['x1']) <= self.shape[2]) or area > _CHUNK_BYTES or mraw != (area+7)//8 or sraw != (area if row['score'] else 0):
                    raise SweepViewIntegrityError('Native crop exceeds bounded geometry')
                if ml > mraw+1024 or sl > sraw+1024 or bool(sl)!=bool(row['score']) or ml+sl+_CHUNK.size > int(row['length'])-consumed:
                    raise SweepViewIntegrityError('Native crop encoded lengths are invalid')
                mb=payload[position:position+ml]
                position += len(mb)
                if include_scores or verify:
                    sb=payload[position:position+sl]
                    position += len(sb)
                else:
                    position += sl
                    sb=None
                self._count_read(_CHUNK.size+len(mb)+(0 if sb is None else len(sb)))
                if len(mb)!=ml or (sb is not None and len(sb)!=sl):
                    raise SweepViewIntegrityError('Native crop bytes changed')
                if verify:
                    prefix=struct.pack('<8I',y0,y1,x0,x1,ml,sl,mraw,sraw)
                    checksum=zlib.crc32(prefix)
                    if zlib.crc32(mb,checksum)!=mc or zlib.crc32(sb,checksum)!=sc:
                        raise SweepViewIntegrityError('Native crop bytes changed')
                    frame_digest.update(header);frame_digest.update(mb);frame_digest.update(sb)
                def inflate(raw, expected):
                    decoder=zlib.decompressobj(); value=decoder.decompress(raw,expected+1)
                    if len(value)!=expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                        raise SweepViewIntegrityError('Native crop compression is malformed')
                    return value
                mask=np.unpackbits(np.frombuffer(inflate(mb,mraw),np.uint8),bitorder='little',count=area).reshape(y1-y0,x1-x0)
                score=None if not include_scores or not row['score'] else np.frombuffer(inflate(sb,sraw),np.uint8).reshape(mask.shape)
                mask.setflags(write=False)
                consumed += _CHUNK.size+ml+sl
                # No mapping slice or array is held across the yield, so a
                # suspended iterator never keeps a closed reader's payload
                # mapped (Windows could not then delete the file).
                header=mb=sb=payload=None
                yield y0,y1,x0,x1,mask,score
            if consumed != int(row['length']):
                raise SweepViewIntegrityError('Native frame has unindexed payload bytes')
            if verify and frame_digest.digest()!=row['sha256'].tobytes():
                raise SweepViewIntegrityError('Native frame checksum differs from its acknowledged index')
        except (OSError, zlib.error, ValueError, struct.error) as exc:
            raise SweepViewIntegrityError('Cannot read acknowledged native crop') from exc

    def _row_crops(self,row,path,include_scores):
        """One row's crops: the native decode, else the Python reader (exact errors)."""
        crops=self._decode_row(row,path,include_scores)
        return self._python_row_crops(row,path,include_scores) if crops is None else crops

    def _iter_crops(self,z,include_scores):
        for row, path in self._rows(z):
            if row['state'] == 1:
                continue
            yield from self._row_crops(row,path,include_scores)

    def iter_crops(self,z):
        """Yield bounded six-tuples (y0,y1,x0,x1,mask,score|None)."""
        yield from self._iter_crops(z,True)

    def iter_mask_crops(self,z):
        """Iterate five-tuples, skipping compressed score I/O in normal mode.

        The lazy iterator also offers the projection rank input a fused
        native merge (``_MaskFrameCrops.merge_rank_level``).
        """
        return _MaskFrameCrops(self,z)

    def read_frame_crops(self,z,include_scores=True):
        """Return all of frame z's six-tuples (score None when not requested) at once.

        Each native row is one nogil decode call; integrity errors are raised
        before any crop is returned.
        """
        return list(self._iter_crops(z,bool(include_scores)))

    def read_mask_plane(self, z):
        output=np.zeros(self.shape[1:],np.uint8)
        for y0,y1,x0,x1,mask in self.iter_mask_crops(z):
            output[y0:y1,x0:x1] |= mask
        return output

    def read_score_plane(self, z):
        if not self.has_score_lane:
            return None
        output=np.zeros(self.shape[1:],np.uint8)
        for y0,y1,x0,x1,_,score in self.iter_crops(z):
            if score is not None:
                np.maximum(output[y0:y1,x0:x1],score,out=output[y0:y1,x0:x1])
        return output

    def close(self):
        with self._state_lock:
            self._closed=True
            payloads=list(self._payloads.values())
            self._payloads.clear()
            self._ranges.clear()
            self._padding.clear()
        for payload in payloads:payload.close()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def open_view(model_name, view_name, cutoff, *, native_shape=None, identity=None, root=None):
    root, meta = _decision(model_name,view_name,cutoff,native_shape,identity,root)
    if meta is None or meta['status']=='fallback':
        raise ValueError('Native view is explicitly unavailable for this route')
    if meta['status'] in ('projected','consumed'):
        raise SweepViewRetiredError('Native view payloads were durably projected and retired')
    if meta.get('scalar_owned_cutoff')==cutoff:
        raise SweepViewScalarOwnedError('Scalar pipeline owns this cutoff; no staged pixel lane exists')
    return NativeViewReader(root,meta,cutoff)


def hydrate(model_name, view_name, cutoff, union, scores=None, *, targets_zeroed=True,
            identity=None, root=None, stop=None):
    """Restore one caller-owned view; default requires freshly zeroed target arrays."""
    if union.dtype!=np.uint8 or union.ndim!=3 or not union.flags.writeable:
        raise ValueError('Native hydration requires a writable uint8 TYX target')
    if scores is not None and (scores.shape!=union.shape or scores.dtype!=np.uint8 or not scores.flags.writeable):
        raise ValueError('Native hydration score target differs')
    with open_view(model_name,view_name,cutoff,native_shape=union.shape,identity=identity,root=root) as reader:
        if scores is not None and not reader.has_score_lane:
            raise ValueError('Native product did not capture a score lane')
        if not targets_zeroed:
            union.fill(0)
            if scores is not None:scores.fill(0)
        active=np.zeros(union.shape[0],bool);boxes=np.zeros((union.shape[0],4),np.int64)
        for z in range(union.shape[0]):
            if stop is not None and stop.is_set():
                raise RuntimeError('Native sweep hydration cancelled')
            contributions=reader.iter_crops(z) if scores is not None else ((*crop,None) for crop in reader.iter_mask_crops(z))
            for y0,y1,x0,x1,mask,score in contributions:
                union[z,y0:y1,x0:x1] |= mask
                if scores is not None and score is not None:
                    np.maximum(scores[z,y0:y1,x0:x1],score,out=scores[z,y0:y1,x0:x1])
                if active[z]:
                    boxes[z]=min(boxes[z,0],y0),max(boxes[z,1],y1),min(boxes[z,2],x0),max(boxes[z,3],x1)
                else:boxes[z]=(y0,y1,x0,x1);active[z]=True
        return dict(valid=True,slice_any=active,slice_bboxes=boxes,prediction_count=reader.prediction_count,
            frames_with_predictions=reader.frames_with_predictions,has_score_lane=reader.has_score_lane,native_fanout_hydrated=True)


def retire_view_payloads(model_name, view_name, expected_cutoffs, *, identity=None, root=None,
                         terminal_state='projected'):
    if terminal_state not in ('projected','consumed'):
        raise ValueError('Native retirement state must be projected or consumed')
    cuts=_cents(expected_cutoffs)
    root,meta=_decision(model_name,view_name,cuts[0],identity=identity,root=root)
    if meta is None or tuple(meta['cutoffs'])!=cuts:
        raise SweepViewIntegrityError('Cannot retire an undecided or different native family')
    if meta['status'] in ('projected','consumed'):return dict(status=meta['status'],freed_bytes=0)
    if meta['status']!='ready':
        raise ValueError('Only durably processed ready native products may be retired')
    freed=0
    updates = []
    for receipt in meta.get('ranges',[]):
        path=Path(receipt['path']).resolve()
        if path.parent!=(root/'ranges').resolve():
            raise SweepViewIntegrityError('Native retirement path escapes its owner')
        for name in ('payload.bin','index.bin'):
            target=path/name
            if target.exists():freed+=target.stat().st_size;target.unlink()
        updates.append((path.name, (path/'metadata.json').stat().st_size))
    if updates:
        _quota_many(root, updates)
    meta['status']=terminal_state
    write_json_atomic(_view_dir(root,model_name,view_name)/'metadata.json',meta)
    return dict(status=terminal_state,freed_bytes=freed)
