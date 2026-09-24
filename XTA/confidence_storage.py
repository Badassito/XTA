"""Sparse numeric confidence blocks with explicit payload coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np


BLOCK_SCHEMA = 'xta.confidence_evidence/2'
BLOCK_LAYOUT = 'uint8_zlib_blocks'
PIECE_LAYOUT = 'native_pieces'
BLOCK_DTYPE = np.dtype([('z','<u4'), ('y','<u4'), ('x','<u4'), ('h','<u2'),
                       ('w','<u2'), ('offset','<u8'), ('length','<u4')])
_RECORD = struct.Struct('<IIIHHQI')


class ConfidenceStageLimit(RuntimeError):
    """A local compressed stage would exceed its reserved numeric bytes."""


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
            occupied = np.any(values[a-y0:b-y0], axis=0)
            active = np.logical_or.reduceat(occupied, starts)
            for column in np.flatnonzero(active):
                x = int(columns[column])
                a, b, c, d = max(y, y0), min(y+block, y1), max(x, x0), min(x+block, x1)
                tile = values[a-y0:b-y0, c-x0:d-x0]
                ys = np.flatnonzero(np.any(tile, axis=1))
                if not ys.size:
                    continue
                xs = np.flatnonzero(np.any(tile, axis=0))
                if (y, x) <= previous:
                    raise ValueError('Confidence reader crops must have disjoint, ordered block cells')
                previous = (y, x)
                top, bottom, left, right = int(ys[0]), int(ys[-1])+1, int(xs[0]), int(xs[-1])+1
                yield a+top, a+bottom, c+left, c+right, np.ascontiguousarray(tile[top:bottom, left:right])


def write_blocks(directory, shape, slice_reader, *, layer_key, model_name, provenance,
                 coordinate_space='source', source_shape=None, block_size=128,
                 metrics=None, max_numeric_bytes=None):
    """Stream block index and payload; metadata is the atomic completion marker."""
    from .confidence_evidence import SCORE_SEMANTICS, _write_json_atomic, _json_value
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
    read_seconds = scan_seconds = compression_seconds = storage_seconds = 0.0
    started = time.perf_counter()
    next_progress = started + 30.0
    try:
        with payload_tmp.open('wb') as payload,index_tmp.open('wb') as index:
            for z in range(first,stop):
                now = time.perf_counter()
                if now >= next_progress:
                    print(f'Confidence write progress {model_name}/{layer_key}: '
                          f'coordinates={coordinate_space}, planes={z-first}/{stop-first}, '
                          f'blocks={blocks}, payload_bytes={payload.tell()}, elapsed_s={now-started:.1f}.',
                          flush=True)
                    next_progress = now + 30.0
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
                    scan_seconds += time.perf_counter() - phase_started
                    if item is None:
                        break
                    y0,y1,x0,x1,crop = item
                    phase_started = time.perf_counter()
                    encoded = zlib.compress(crop.tobytes(),level=3)
                    compression_seconds += time.perf_counter() - phase_started
                    if (max_numeric_bytes is not None
                            and payload.tell()+len(encoded)+(blocks+1)*_RECORD.size > int(max_numeric_bytes)):
                        raise ConfidenceStageLimit('Compressed confidence exceeds its local staging reservation')
                    phase_started = time.perf_counter()
                    record = _RECORD.pack(z,y0,x0,y1-y0,x1-x0,payload.tell(),len(encoded))
                    index.write(record); index_digest.update(record)
                    payload.write(encoded); digest.update(encoded)
                    storage_seconds += time.perf_counter() - phase_started
                    blocks += 1
                    known += int(np.count_nonzero(crop))
            payload_bytes = payload.tell()
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
            quantization='round-half-even(clip(instance_score,0,1)*255); quantized zero is unknown',
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
                index_bytes=blocks*_RECORD.size, block_count=blocks)
        return metadata
    finally:
        payload_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)


class BlockScoreReader:
    """Read small independent score crops; index mappings are explicitly closable."""
    def __init__(self, reference):
        self._closed=False
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
        first,stop = np.searchsorted(self.index['z'],(z,z+1))
        with (self.reference.path/'scores.u8.zlib').open('rb') as stream:
            for number in range(int(first),int(stop)):
                if self._closed:
                    raise RuntimeError('Confidence block reader is closed')
                record=self.index[number]
                y,x,h,w,offset,length = (int(record[k]) for k in ('y','x','h','w','offset','length'))
                stream.seek(offset)
                encoded = stream.read(length)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(encoded,h*w+1)
                if len(raw)!=h*w or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError('Confidence score block is truncated or malformed')
                yield y,y+h,x,x+w,np.frombuffer(raw,np.uint8).reshape(h,w)

    def close(self):
        self._closed=True
        array = getattr(self,'index',None)
        mapping = getattr(array,'_mmap',None)
        if mapping is not None and not mapping.closed:
            mapping.close()

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
