"""Sparse numeric confidence blocks with explicit payload coordinates."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np


BLOCK_SCHEMA = 'xta.confidence_evidence/2'
BLOCK_LAYOUT = 'uint8_zlib_blocks'
PIECE_LAYOUT = 'native_pieces'
BLOCK_DTYPE = np.dtype([('z','<u4'), ('y','<u4'), ('x','<u4'), ('h','<u2'),
                       ('w','<u2'), ('offset','<u8'), ('length','<u4')])
_RECORD = struct.Struct('<IIIHHQI')
_CROP_OCCUPANCY_BAND_BYTES = 8 * 1024 * 1024
# Native score-block decode batches. iter_crops keeps at most one bounded
# decoded batch of one frame live; whole-volume scatter batches whole frames by
# stored bytes so cancellation and the failure fallback stay frame-granular.
_INFLATE_DECODED_BATCH_BYTES = 8 * 1024 * 1024
_INFLATE_DECODED_BATCH_RECORDS = 1024
_INFLATE_SCATTER_BATCH_BYTES = 8 * 1024 * 1024
# One positioned read per batch where the platform has it (no seek+read pair).
_PREADV = getattr(os, 'preadv', None)


class ConfidenceStageLimit(RuntimeError):
    """A local compressed stage would exceed its reserved numeric bytes."""


def _measured_frames(frames, timings):
    iterator = iter(frames)
    while True:
        started, cpu_started = time.perf_counter(), time.thread_time()
        try:
            item = next(iterator)
        except StopIteration:
            return
        finally:
            timings['frame_wait_seconds'] += time.perf_counter()-started
            timings['frame_wait_thread_cpu_seconds'] += time.thread_time()-cpu_started
        started, cpu_started = time.perf_counter(), time.thread_time()
        try:
            yield item
        finally:
            timings['frame_consume_seconds'] += time.perf_counter()-started
            timings['frame_consume_thread_cpu_seconds'] += time.thread_time()-cpu_started
            item = None


def _ordered_encoded_frames(reader, first, stop, block_size, workers, metrics=None):
    """Retain at most workers completed frames plus the caller's current frame."""
    from .native_pull_scheduling import get_native_pull_scheduler
    scheduler = get_native_pull_scheduler()
    if scheduler is not None:
        # Register only after the publication queue has admitted its numeric
        # staging slot. Waiting parent executor threads own no CPU lane. Native
        # Projection and confidence share the running CPU allowance; this
        # capture's admitted plan separately funds its ordered result window.
        frames = scheduler.ordered_planes(stop-first,
            lambda offset: reader.encode_frame(first+offset, block_size),
            max_pending=workers, role='confidence_capture')
        try:
            for offset, result in frames:
                yield first+offset, result
        finally:
            # A failed consumer or stage limit must join every borrowed mask/
            # score reader before its parent's dense reservation can retire.
            try:
                if metrics is not None:
                    state = frames.snapshot()
                    metrics.update(capture_scheduler='run_shared',
                        capture_cpu_budget=state['cpu_budget'],
                        capture_peak_active_frames=state['job_peak_active_operations'],
                        capture_peak_retained_frames=state['job_peak_retained_operations'])
            finally:
                frames.close()
                # Failed producer Futures can retain this lambda's closure cell
                # after traceback frames are cleared. Every reader is joined;
                # empty the cell without altering the reader used by a retry.
                reader = None
        return
    if workers <= 1:
        for z in range(first, stop):
            yield z, reader.encode_frame(z, block_size)
        return
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='confidence-capture')
    pending = deque()
    try:
        indices = iter(range(first, stop))
        for _ in range(workers):
            z = next(indices, None)
            if z is not None:
                pending.append((z, pool.submit(reader.encode_frame, z, block_size)))
        while pending:
            z, future = pending.popleft()
            yield z, future.result()
            following = next(indices, None)
            if following is not None:
                pending.append((following, pool.submit(reader.encode_frame, following, block_size)))
    finally:
        for _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def checked_shape(shape):
    values = tuple(shape)
    if (len(values) != 3 or any(isinstance(v, bool) or int(v) != v or not 0 < int(v) < 2**31
                               for v in values)):
        raise ValueError('Confidence grids require three positive int32 dimensions')
    return tuple(map(int, values))


def plane_blocks(plane, block_size=128):
    """Scan each row band once; compress only cropped nonempty grid blocks."""
    values = np.asarray(plane)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError('Confidence planes must be two-dimensional uint8 arrays')
    height, width = values.shape
    block = int(block_size)
    if not 1 <= block <= 65535:
        raise ValueError('Confidence block size must be in [1,65535]')
    starts = np.arange(0, width, block, dtype=np.int64)
    for y in range(0, height, block):
        band = values[y:min(y+block,height)]
        columns = np.any(band, axis=0)
        active = np.logical_or.reduceat(columns, starts)
        for column in np.flatnonzero(active):
            x = int(starts[column])
            tile = band[:, x:min(x+block,width)]
            ys = np.flatnonzero(np.any(tile,axis=1))
            xs = np.flatnonzero(np.any(tile,axis=0))
            y0,y1,x0,x1 = int(ys[0]),int(ys[-1])+1,int(xs[0]),int(xs[-1])+1
            yield y+y0,y+y1,x+x0,x+x1,np.ascontiguousarray(tile[y0:y1,x0:x1])


def crop_blocks(crops, shape, block_size=128):
    """Partition owned crops on the same global grid as a full-plane scan."""
    height, width = shape
    block = int(block_size)
    previous = (-1, -1)
    for y0, y1, x0, x1, crop in crops:
        values = np.asarray(crop)
        y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
        if (values.dtype != np.uint8 or values.shape != (y1-y0, x1-x0)
                or not 0 <= y0 < y1 <= height or not 0 <= x0 < x1 <= width):
            raise ValueError('Confidence reader returned an invalid crop')
        columns = np.arange(x0//block*block, x1, block, dtype=np.int64)
        starts = np.maximum(columns-x0, 0)
        for y in range(y0//block*block, y1, block):
            a, b = max(y, y0), min(y+block, y1)
            band = values[a-y0:b-y0]
            if band.size <= _CROP_OCCUPANCY_BAND_BYTES:
                # Reuse one bounded occupancy scan for every global-grid cell in this
                # row band. Avoid a padded copy of the whole crop and per-cell scans.
                occupied = band != 0
                row_cells = np.logical_or.reduceat(occupied, starts, axis=1)
                column_any = occupied.any(axis=0)
                active = np.flatnonzero(row_cells.any(axis=0))
            else:
                # Very wide crops or unusually large block sizes must not turn a
                # scanner optimization into an unadmitted full-band allocation.
                row_cells = None
                column_any = np.any(band, axis=0)
                active = np.flatnonzero(np.logical_or.reduceat(column_any, starts))
            for column in active:
                x = int(columns[column])
                c, d = max(x, x0), min(x+block, x1)
                local_c, local_d = c-x0, d-x0
                if row_cells is None:
                    tile = band[:, local_c:local_d]
                    ys = np.flatnonzero(np.any(tile, axis=1))
                    xs = np.flatnonzero(np.any(tile, axis=0))
                else:
                    ys = np.flatnonzero(row_cells[:, column])
                    xs = np.flatnonzero(column_any[local_c:local_d])
                if not ys.size:
                    continue
                if (y, x) <= previous:
                    raise ValueError('Confidence reader crops must have disjoint, ordered block cells')
                previous = (y, x)
                top, bottom, left, right = int(ys[0]), int(ys[-1])+1, int(xs[0]), int(xs[-1])+1
                yield a+top, a+bottom, c+left, c+right, np.ascontiguousarray(
                    band[top:bottom, local_c+left:local_c+right])


def write_blocks(directory, shape, slice_reader, *, layer_key, model_name, provenance,
                 coordinate_space='source', source_shape=None, block_size=128,
                 metrics=None, max_numeric_bytes=None, track_minimum_score=False):
    """Stream block index and payload; metadata is the atomic completion marker."""
    from .confidence_evidence import SCORE_SEMANTICS, _write_json_atomic, _json_value, _MaskedNativeScoreReader
    directory = Path(directory)
    shape = checked_shape(shape)
    if coordinate_space not in ('source','native_view_processing'):
        raise ValueError('Unsupported confidence payload coordinate space')
    source_shape = checked_shape(source_shape) if source_shape is not None else (
        shape if coordinate_space == 'source' else None)
    if coordinate_space == 'source' and source_shape != shape:
        raise ValueError('Source-coordinate blocks must use their declared source grid')
    block_size = int(block_size)
    if not 1 <= block_size <= 65535:
        raise ValueError('Invalid confidence block size')
    if max_numeric_bytes is not None and int(max_numeric_bytes) <= 0:
        raise ValueError('Confidence numeric staging limit must be positive')
    first,stop = map(int,getattr(slice_reader,'known_z_bounds',(0,shape[0])))
    if not 0 <= first <= stop <= shape[0]:
        raise ValueError('Confidence known-support bounds are outside their payload grid')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metadata.json').exists():
        raise FileExistsError(f'Confidence evidence already exists: {directory}')
    payload_tmp,index_tmp = directory/'scores.u8.zlib.partial',directory/'index.bin.partial'
    digest,index_digest = hashlib.sha256(),hashlib.sha256()
    blocks,known = 0,0
    minimum_nonzero_score = 256
    read_seconds = scan_seconds = compression_seconds = storage_seconds = 0.0
    codec_diagnostics, codec_frame_counts = {}, {}
    copy_kernel_calls = copy_batch_peak_bytes = 0
    native_compression_batches = 0
    native_batch_seconds = 0.
    started = time.perf_counter()
    next_progress = started + 30.0
    encoded_frames = None
    measured_frames = None
    frame_timings = dict(frame_wait_seconds=0., frame_wait_thread_cpu_seconds=0.,
                         frame_consume_seconds=0., frame_consume_thread_cpu_seconds=0.)
    typed_capture = (isinstance(slice_reader, _MaskedNativeScoreReader)
                     and slice_reader.compiled_capture_capable)
    capture_plan = slice_reader.capture_plan if typed_capture else None
    if capture_plan is not None and (capture_plan.shape != shape or capture_plan.block_size != block_size):
        raise ValueError('Confidence capture writer geometry/grid differs from its admitted task plan')
    if capture_plan is None and typed_capture:
        from .confidence_capture import plan_confidence_capture
        capture_plan = plan_confidence_capture(shape, 1, block_size=block_size)
    # A reader that encodes whole frames on this writer's thread (sweep sibling
    # crops) returns one frame's index records (BLOCK_DTYPE, offsets relative
    # to the frame) and its concatenated zlib payload, in global grid order:
    # one write pair per frame, no per-block Python here. It takes no capture
    # plan, worker pool or scheduler lane.
    caller_encoder = (getattr(slice_reader, 'encode_caller_frame', None)
                      if capture_plan is None else None)
    if capture_plan is not None:
        encoded_frames = _ordered_encoded_frames(slice_reader, first, stop, block_size,
                                                  capture_plan.workers, metrics=metrics)
    elif caller_encoder is not None:
        encoded_frames = ((z, caller_encoder(z, block_size)) for z in range(first, stop))
    # The consumer's already admitted frame allowance also bounds this transient
    # copy. Leave a factor of two plus control space for bytearray overallocation.
    # Generic readers retain their existing cell-at-a-time memory contract.
    batch_limit = (min(1024*1024, max(0, (capture_plan.consumer_bytes-128)//2))
                   if capture_plan is not None else 0)
    payload_offset = 0
    payload_write_calls = index_write_calls = 0
    try:
        with payload_tmp.open('wb') as payload,index_tmp.open('wb') as index:
            payload_batch, index_batch = bytearray(), bytearray()
            def flush_batch():
                nonlocal storage_seconds, payload_write_calls, index_write_calls
                if not index_batch:
                    return 0.
                phase_started = time.perf_counter()
                index.write(index_batch); index_digest.update(index_batch)
                payload.write(payload_batch); digest.update(payload_batch)
                index_write_calls += 1
                payload_write_calls += 1
                index_batch.clear(); payload_batch.clear()
                elapsed = time.perf_counter()-phase_started
                storage_seconds += elapsed
                return elapsed
            frames = ((z, None) for z in range(first, stop)) if encoded_frames is None else encoded_frames
            measured_frames = _measured_frames(frames,frame_timings)
            for z, encoded_result in measured_frames:
                now = time.perf_counter()
                if now >= next_progress:
                    print(f'Confidence write progress {model_name}/{layer_key}: '
                          f'coordinates={coordinate_space}, planes={z-first}/{stop-first}, '
                          f'blocks={blocks}, payload_bytes={payload_offset}, elapsed_s={now-started:.1f}.',
                          flush=True)
                    next_progress = now + 30.0
                if caller_encoder is not None:
                    records, frame_payload, frame_known, phase_metrics = encoded_result
                    read_seconds += phase_metrics['read_seconds']
                    scan_seconds += phase_metrics['scan_seconds']
                    compression_seconds += phase_metrics['compression_seconds']
                    count = len(records)
                    if not count:
                        continue
                    if (records.dtype != BLOCK_DTYPE or records.ndim != 1
                            or int(records['offset'][-1])+int(records['length'][-1]) != len(frame_payload)):
                        raise ValueError('Confidence frame encoder returned inconsistent records')
                    # Cumulative sizes only grow, so the frame total exceeds the
                    # reservation exactly when some per-block prefix would.
                    if (max_numeric_bytes is not None
                            and payload_offset+len(frame_payload)+(blocks+count)*_RECORD.size > int(max_numeric_bytes)):
                        raise ConfidenceStageLimit('Compressed confidence exceeds its local staging reservation')
                    phase_started = time.perf_counter()
                    records['offset'] += np.uint64(payload_offset)
                    index_bytes = records.tobytes()
                    index.write(index_bytes); index_digest.update(index_bytes)
                    payload.write(frame_payload); digest.update(frame_payload)
                    index_write_calls += 1
                    payload_write_calls += 1
                    storage_seconds += time.perf_counter() - phase_started
                    payload_offset += len(frame_payload)
                    blocks += count
                    known += frame_known
                    records = frame_payload = index_bytes = None
                    continue
                if encoded_result is not None:
                    encoded_rows, phase_metrics = encoded_result
                    scan_seconds += phase_metrics['scan_seconds']
                    compression_seconds += phase_metrics['compression_seconds']
                    copy_kernel_calls += int(phase_metrics.get('copy_kernel_calls', 0))
                    copy_batch_peak_bytes = max(copy_batch_peak_bytes,
                        int(phase_metrics.get('copy_batch_peak_bytes', 0)))
                    native_compression_batches += int(phase_metrics.get('native_compression_batches', 0))
                    native_batch_seconds += float(phase_metrics.get('native_batch_seconds', 0.))
                    for field in ('codec_backend', 'codec_version', 'codec_python_version',
                                  'codec_fallback_reason'):
                        if field not in phase_metrics:
                            continue
                        value = phase_metrics[field]
                        if value is not None:
                            value = str(value)[:256]
                        values = codec_diagnostics.setdefault(field, [])
                        if value not in values and len(values) < 4:
                            values.append(value)
                    backend = phase_metrics.get('codec_backend')
                    if backend is not None:
                        backend = str(backend)[:256]
                        if backend not in codec_frame_counts and len(codec_frame_counts) >= 16:
                            backend = 'other'
                        codec_frame_counts[backend] = codec_frame_counts.get(backend, 0)+1
                    iterator = iter(encoded_rows)
                else:
                    phase_started = time.perf_counter()
                    if hasattr(slice_reader, 'iter_crops'):
                        iterator = iter(crop_blocks(slice_reader.iter_crops(z), shape[1:], block_size))
                    else:
                        plane = slice_reader(z)
                        if plane is None:
                            read_seconds += time.perf_counter() - phase_started
                            continue
                        values = np.asarray(plane)
                        if values.dtype != np.uint8 or values.shape != shape[1:]:
                            raise ValueError('Confidence slice reader returned an invalid shape or dtype')
                        iterator = iter(plane_blocks(values,block_size))
                    read_seconds += time.perf_counter() - phase_started
                while True:
                    phase_started = time.perf_counter()
                    item = next(iterator, None)
                    if encoded_result is None:
                        scan_seconds += time.perf_counter() - phase_started
                    if item is None:
                        break
                    if encoded_result is None:
                        y0,y1,x0,x1,crop = item
                        if track_minimum_score:
                            minimum_nonzero_score = min(minimum_nonzero_score,
                                int(crop.min(where=crop != 0, initial=255)))
                        phase_started = time.perf_counter()
                        encoded = zlib.compress(crop.tobytes(),level=3)
                        compression_seconds += time.perf_counter() - phase_started
                        known_in_block = int(np.count_nonzero(crop))
                    else:
                        y0,y1,x0,x1,known_in_block,encoded = item
                    if (max_numeric_bytes is not None
                            and payload_offset+len(encoded)+(blocks+1)*_RECORD.size > int(max_numeric_bytes)):
                        raise ConfidenceStageLimit('Compressed confidence exceeds its local staging reservation')
                    phase_started = time.perf_counter()
                    flushed_seconds = 0.
                    record = _RECORD.pack(z,y0,x0,y1-y0,x1-x0,payload_offset,len(encoded))
                    record_bytes = len(record)+len(encoded)
                    if batch_limit and record_bytes <= batch_limit:
                        if len(index_batch)+len(payload_batch)+record_bytes > batch_limit:
                            flushed_seconds += flush_batch()
                        index_batch.extend(record)
                        payload_batch.extend(encoded)
                    else:
                        flushed_seconds += flush_batch()
                        index.write(record); index_digest.update(record)
                        payload.write(encoded); digest.update(encoded)
                        index_write_calls += 1
                        payload_write_calls += 1
                    payload_offset += len(encoded)
                    storage_seconds += time.perf_counter() - phase_started - flushed_seconds
                    blocks += 1
                    known += known_in_block
                # No copied batch may coexist with both the preceding frame and
                # its successor during an ordered iterator's next() operation.
                flush_batch()
            payload_bytes = payload_offset
            phase_started = time.perf_counter()
            payload.flush(); index.flush()
            storage_seconds += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        payload_tmp.replace(directory/'scores.u8.zlib')
        index_tmp.replace(directory/'index.bin')
        metadata = dict(schema=BLOCK_SCHEMA,layout=BLOCK_LAYOUT,coordinate_space=coordinate_space,
            stored_shape_tyx=list(shape),source_shape_tyx=None if source_shape is None else list(source_shape),
            output_shape_tyx=None if source_shape is None else list(source_shape),
            layer_key=str(layer_key),model_name=str(model_name),dtype='uint8',unknown='score_zero',
            score_semantics=SCORE_SEMANTICS,
            quantization='round-half-even(clip(prediction_score,0,1)*255); quantized zero is unknown',
            payload='scores.u8.zlib',index='index.bin',index_layout='z:u32,y:u32,x:u32,h:u16,w:u16,offset:u64,length:u32; little-endian',
            index_record_bytes=_RECORD.size,block_size=block_size,block_count=blocks,
            payload_bytes=payload_bytes,payload_sha256=digest.hexdigest(),index_sha256=index_digest.hexdigest(),
            known_voxels=known,known_voxels_coordinate_space=coordinate_space,
            stored_axes='(t,Y,X)' if coordinate_space=='source' else '(view_frame,view_row,view_column)',
            exported_axes='(X,Y,t)' if coordinate_space=='source' else None,
            provenance=_json_value(provenance or {}))
        _write_json_atomic(directory/'metadata.json',metadata)
        storage_seconds += time.perf_counter() - phase_started
        if metrics is not None:
            metrics.update(read_seconds=read_seconds, scan_seconds=scan_seconds,
                compression_seconds=compression_seconds, storage_seconds=storage_seconds,
                elapsed_seconds=time.perf_counter()-started, payload_bytes=payload_bytes,
                index_bytes=blocks*_RECORD.size, block_count=blocks,
                storage_batch_limit_bytes=batch_limit,
                storage_batch_reserved_bytes=0 if not batch_limit else 2*batch_limit+128,
                storage_payload_write_calls=payload_write_calls,
                storage_index_write_calls=index_write_calls)
            if track_minimum_score:
                metrics['minimum_nonzero_score'] = minimum_nonzero_score
            if caller_encoder is not None:
                metrics.update(frame_encoder='caller_compiled_cells',
                    frame_encoder_codec=getattr(slice_reader, 'codec_backend', None),
                    frame_encoder_codec_fallback_frames=getattr(slice_reader, 'native_fallback_frames', 0))
            if codec_diagnostics:
                metrics.update(capture_codec_frame_counts=dict(codec_frame_counts),
                    capture_copy_kernel_calls=copy_kernel_calls,
                    capture_copy_batch_peak_bytes=copy_batch_peak_bytes,
                    native_compression_batches=native_compression_batches,
                    native_batch_seconds=native_batch_seconds)
                for field, values in codec_diagnostics.items():
                    metrics[field] = values[0] if len(values) == 1 else 'mixed'
                    if len(values) > 1:
                        metrics[field+'_values'] = list(values)
            if capture_plan is not None:
                metrics.update(capture_backend='compiled_masked_cells_ordered_frames',
                    capture_workers=capture_plan.workers,
                    capture_requested_workers=capture_plan.requested_workers,
                    capture_workspace_bytes=capture_plan.workspace_bytes,
                    capture_workspace_limit_bytes=capture_plan.workspace_limit_bytes,
                    capture_max_pending_frames=capture_plan.max_pending_frames)
        return metadata
    finally:
        if measured_frames is not None:
            measured_frames.close()
        if metrics is not None:
            metrics.update(frame_timings)
        if encoded_frames is not None:
            encoded_frames.close()
        # A retained failed Future owns this traceback. Do not let that frame
        # keep completed numeric windows or borrowed reader aliases alive after
        # every producer has been joined by close().
        encoded_frames = measured_frames = frames = iterator = None
        encoded_result = encoded_rows = slice_reader = caller_encoder = None
        payload_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)


class BlockScoreReader:
    """Read small independent score crops; index mappings are explicitly closable."""
    def __init__(self, reference):
        self._closed=False
        self._payload=None
        self._payload_lock=threading.Lock()
        # Index readers in flight (GIL-free kernels included); close() defers
        # unmapping the index until the last one leaves.
        self._index_pins=0
        self._inflater=None
        self._frame_bounds=None
        self.decode_backend=None
        self.reference = reference
        self.shape = reference.storage_shape
        metadata = reference.metadata
        if metadata.get('layout') != BLOCK_LAYOUT or int(metadata.get('index_record_bytes',0)) != BLOCK_DTYPE.itemsize:
            raise ValueError('Unsupported confidence block index layout')
        count = int(metadata.get('block_count',-1))
        if count < 0:
            raise ValueError('Invalid confidence block count')
        index_path,payload_path = reference.path/'index.bin',reference.path/'scores.u8.zlib'
        if index_path.stat().st_size != count*BLOCK_DTYPE.itemsize:
            raise ValueError('Confidence block index size differs from metadata')
        payload_bytes = payload_path.stat().st_size
        if payload_bytes != int(metadata.get('payload_bytes',-1)):
            raise ValueError('Confidence payload size differs from metadata')
        self.index = np.memmap(index_path,mode='r',dtype=BLOCK_DTYPE,shape=(count,)) if count else np.empty(0,BLOCK_DTYPE)
        try:
            previous_end,previous_z = 0,0
            for first in range(0,count,65536):
                rows = self.index[first:first+65536]
                if (int(rows['offset'][0]) != previous_end or int(rows['z'][0]) < previous_z
                        or np.any(rows['z']>=self.shape[0]) or np.any(rows['z'][1:]<rows['z'][:-1])
                        or np.any(rows['h']==0) or np.any(rows['w']==0) or np.any(rows['length']==0)
                        or np.any(rows['y'].astype(np.uint64)+rows['h']>self.shape[1])
                        or np.any(rows['x'].astype(np.uint64)+rows['w']>self.shape[2])
                        or np.any(rows['offset'][1:]!=rows['offset'][:-1]+rows['length'][:-1])):
                    raise ValueError('Malformed confidence block index')
                previous_end = int(rows['offset'][-1])+int(rows['length'][-1])
                previous_z = int(rows['z'][-1])
            if previous_end != payload_bytes:
                raise ValueError('Confidence payload has unindexed data')
        except BaseException:
            self.close()
            raise

    def iter_crops(self,z):
        if self._closed:
            raise RuntimeError('Confidence block reader is closed')
        z = int(z)
        if not 0 <= z < self.shape[0]:
            raise IndexError('Confidence crop is outside its payload grid')
        first,stop = self._frame_records(z)
        inflater = self._native_inflater() if stop > first else None
        if inflater is None:
            yield from self._python_crops(first,stop)
            return
        number = window_first = window_stop = first
        while number < stop:
            if self._closed:
                raise RuntimeError('Confidence block reader is closed')
            # The pin keeps the index mapped while the GIL-free kernel reads it
            # (a concurrent close() would otherwise unmap it under the kernel);
            # it is never held across a yield.
            with self._index_pin():
                # Reuse one bounded prefix sum across decode batches. The
                # copied prefix sum is independent of the pinned index.
                if number == window_stop:
                    window_first = number
                    window_stop = min(stop,number+_INFLATE_DECODED_BATCH_RECORDS)
                    window = self.index[number:window_stop]
                    ends = np.multiply(window['h'],window['w'],dtype=np.int64)
                    np.cumsum(ends,out=ends)
                    window = None
                begin = number-window_first
                decoded_base = int(ends[begin-1]) if begin else 0
                limit = decoded_base+_INFLATE_DECODED_BATCH_BYTES
                # A first block above the byte budget still makes progress.
                end = max(begin+1,int(ends.searchsorted(limit,side='right')))
                count = end-begin
                total = int(ends[end-1])-decoded_base
                starts = np.empty(count,np.int64)
                starts[0] = 0
                np.subtract(ends[begin:end-1],decoded_base,out=starts[1:])
                records = self.index[number:number+count]
                base = int(records['offset'][0])
                size = int(records['offset'][-1])+int(records['length'][-1])-base
                payload = self._read_payload(base,size)
                if payload is not None:
                    decoded,done,status = inflater.inflate_blocks(payload,records,base,starts,total)
                    # Only the independent coordinates survive the index pin.
                    rows = records[:done][['y','x','h','w']].tolist()
                records = None
            if payload is None:
                # Short read: the original decoder owns the exact failure.
                yield from self._python_crops(number,stop)
                return
            payload = None
            decoded.setflags(write=False)
            for (y,x,h,w),start in zip(rows,starts):
                if self._closed:
                    raise RuntimeError('Confidence block reader is closed')
                yield y,y+h,x,x+w,decoded[start:start+h*w].reshape(h,w)
            decoded = None
            if status:
                # Any native rejection re-runs Python zlib from that block so the
                # exception type, message and preceding yields stay identical.
                yield from self._python_crops(number+done,stop)
                return
            number += count

    def _frame_records(self,z):
        """Index row range of frame z (the index is z-sorted; computed once)."""
        bounds = self._frame_bounds
        if bounds is None:
            with self._index_pin():
                bounds = np.searchsorted(self.index['z'],np.arange(self.shape[0]+1)).tolist()
            self._frame_bounds = bounds
        return bounds[z],bounds[z+1]

    def iter_python_crops(self,z):
        """The original per-block reader for one frame (fallback and oracle)."""
        if self._closed:
            raise RuntimeError('Confidence block reader is closed')
        z = int(z)
        if not 0 <= z < self.shape[0]:
            raise IndexError('Confidence crop is outside its payload grid')
        with self._index_pin():
            first,stop = np.searchsorted(self.index['z'],(z,z+1))
        yield from self._python_crops(int(first),int(stop))

    def _python_crops(self,first,stop):
        """Original per-block Python zlib decode; authoritative for every failure."""
        if first >= stop:
            return
        with (self.reference.path/'scores.u8.zlib').open('rb') as stream:
            for number in range(int(first),int(stop)):
                if self._closed:
                    raise RuntimeError('Confidence block reader is closed')
                with self._index_pin():
                    record=self.index[number]
                    y,x,h,w,offset,length = (int(record[k]) for k in ('y','x','h','w','offset','length'))
                    record=None
                stream.seek(offset)
                encoded = stream.read(length)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(encoded,h*w+1)
                if len(raw)!=h*w or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError('Confidence score block is truncated or malformed')
                yield y,y+h,x,x+w,np.frombuffer(raw,np.uint8).reshape(h,w)

    def _native_inflater(self):
        if self._inflater is None:
            from .confidence_native_inflate import native_confidence_inflater
            inflater,receipt = native_confidence_inflater()
            self._inflater = inflater if inflater is not None else False
            self.decode_backend = receipt['backend']
        return self._inflater or None

    def _read_payload(self,base,size):
        """Read one contiguous stored range; the payload is opened once per reader.

        Returns None on a short read so the caller can let the original decoder
        report that failure.
        """
        payload = np.empty(size,np.uint8)
        view = memoryview(payload)
        with self._payload_lock:
            if self._closed:
                raise RuntimeError('Confidence block reader is closed')
            if self._payload is None:
                self._payload = (self.reference.path/'scores.u8.zlib').open('rb',buffering=0)
            stream = self._payload
            filled = 0
            if _PREADV is not None:
                while filled < size:
                    read = _PREADV(stream.fileno(),[view[filled:]],base+filled)
                    if not read:
                        return None
                    filled += read
                return payload
            stream.seek(base)
            while filled < size:
                read = stream.readinto(view[filled:])
                if not read:
                    return None
                filled += read
        return payload

    def scatter_into(self,output,*,before_batch=None,progress=None):
        """Assign every stored crop into a full-grid array in record order.

        Equivalent to ``output[z,y0:y1,x0:x1] = crop`` for each item of
        ``iter_crops(z)`` over all z. ``before_batch`` runs before each decode
        batch (at least once when the grid has frames). ``progress`` receives
        ``decoded_bytes`` exactly as that loop would count them, also when a
        failure interrupts it, plus ``decode_batches`` and ``decode_backend``.
        """
        progress = {} if progress is None else progress
        progress.setdefault('decoded_bytes',0)
        progress.setdefault('decode_batches',0)
        if self._closed:
            raise RuntimeError('Confidence block reader is closed')
        count = len(self.index)
        inflater = self._native_inflater() if count else None
        if (inflater is None or not isinstance(output,np.ndarray) or output.dtype != np.uint8
                or tuple(output.shape) != tuple(self.shape) or not output.flags.c_contiguous
                or not output.flags.writeable):
            progress['decode_backend'] = 'python_zlib'
            for z in range(self.shape[0]):
                if before_batch is not None:
                    before_batch()
                for y0,y1,x0,x1,crop in self.iter_crops(z):
                    output[z,y0:y1,x0:x1] = crop
                    progress['decoded_bytes'] += int(crop.nbytes)
            return output
        progress['decode_backend'] = self.decode_backend
        with self._index_pin():
            zs = np.asarray(self.index['z'],np.int64)
            offsets = np.asarray(self.index['offset'],np.int64)
            ends = offsets+np.asarray(self.index['length'],np.int64)
        frame_starts = np.concatenate(([0],np.flatnonzero(zs[1:]!=zs[:-1])+1,[count])).tolist()
        frames = len(frame_starts)-1
        frame = 0
        while frame < frames:
            if before_batch is not None:
                before_batch()
            if self._closed:
                raise RuntimeError('Confidence block reader is closed')
            first = frame_starts[frame]
            base = int(offsets[first])
            last = frame+1
            while (last < frames and int(ends[frame_starts[last+1]-1])-base
                   <= _INFLATE_SCATTER_BATCH_BYTES):
                last += 1
            stop = frame_starts[last]
            # The pin keeps the index mapped while the GIL-free kernel reads
            # ``records`` (see iter_crops).
            with self._index_pin():
                payload = self._read_payload(base,int(ends[stop-1])-base)
                records = self.index[first:stop]
                if payload is None:
                    done,status,decoded = 0,-1,0
                else:
                    done,status,decoded = inflater.inflate_scatter(payload,records,base,output)
                payload = None
                progress['decode_batches'] += 1
                if not status:
                    records = None
                    progress['decoded_bytes'] += decoded
                    frame = last
                    continue
                # Count only the whole frames the per-crop loop would have finished,
                # then re-run Python zlib from the failing frame's first block. Its
                # earlier native assignments repeat identically and in order.
                failed_z,last_z = int(zs[first+done]),int(zs[stop-1])
                finished = records[:int(np.searchsorted(zs,failed_z))-first]
                progress['decoded_bytes'] += int((finished['h'].astype(np.int64)*finished['w']).sum())
                records = finished = None
            for z in range(failed_z,last_z+1):
                if z != failed_z and before_batch is not None:
                    before_batch()
                for y0,y1,x0,x1,crop in self.iter_python_crops(z):
                    output[z,y0:y1,x0:x1] = crop
                    progress['decoded_bytes'] += int(crop.nbytes)
            frame = last
        return output

    @contextlib.contextmanager
    def _index_pin(self):
        """Keep the index mapping open for a bounded index read or kernel call.

        Raises the closed-reader error if close() already ran. Never hold a pin
        across a generator yield: close() would defer the unmap until resumed.
        """
        with self._payload_lock:
            if self._closed:
                raise RuntimeError('Confidence block reader is closed')
            self._index_pins += 1
        try:
            yield
        finally:
            with self._payload_lock:
                self._index_pins -= 1
                if self._closed and not self._index_pins:
                    self._release_index()

    def _release_index(self):
        array = getattr(self,'index',None)
        mapping = getattr(array,'_mmap',None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    def close(self):
        self._closed=True
        lock = getattr(self,'_payload_lock',None)
        if lock is None:
            self._release_index()
            return
        with lock:
            payload, self._payload = self._payload, None
            # A pinned GIL-free kernel may still read the index: the last pin
            # to leave unmaps it instead (see _index_pin).
            if not getattr(self,'_index_pins',0):
                self._release_index()
        if payload is not None:
            payload.close()

    def __del__(self):
        self.close()


def assembled_crop(crops):
    """Compatibility bbox access; streaming callers should use iter_crops."""
    values = list(crops)
    if not values:
        return None
    if len(values)==1:
        return values[0]
    y0,y1 = min(v[0] for v in values),max(v[1] for v in values)
    x0,x1 = min(v[2] for v in values),max(v[3] for v in values)
    result = np.zeros((y1-y0,x1-x0),np.uint8)
    for a,b,c,d,crop in values:
        target = result[a-y0:b-y0,c-x0:d-x0]
        np.maximum(target,crop,out=target)
    return y0,y1,x0,x1,result


__all__ = ['BLOCK_SCHEMA','BLOCK_LAYOUT','PIECE_LAYOUT','BlockScoreReader',
           'write_blocks','plane_blocks','checked_shape','assembled_crop']
