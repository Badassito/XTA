"""Join disjoint native confidence leases without decoding their numeric blocks."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import tempfile

import numpy as np

from .confidence_storage import BLOCK_DTYPE, BLOCK_LAYOUT, BLOCK_SCHEMA, checked_shape


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0, maximum=2**64-1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f'Invalid confidence lease {name}')
    return int(value)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Invalid confidence lease {name} checksum')
    return value


def _metadata(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _contained_file(directory, name):
    path = directory/name
    _require(path.resolve().is_relative_to(directory), 'Confidence lease file escapes its directory')
    _require(path.is_file(), f'Missing confidence lease file: {name}')
    return path


def _preflight(pieces, native_shape, source_shape, layer_key, model_name, destination):
    from .confidence_evidence import ConfidenceEvidenceRef
    leases = []
    for piece in pieces:
        original = piece['reference']
        directory = Path(original.path if isinstance(original, ConfidenceEvidenceRef) else original).resolve()
        _require(not destination.is_relative_to(directory) and not directory.is_relative_to(destination),
                 'Confidence destination must be separate from every input lease')
        metadata_path = _contained_file(directory, 'metadata.json')
        detail, metadata_sha = _metadata(metadata_path)
        reference = ConfidenceEvidenceRef.open(directory)
        _require(reference.metadata == detail and (not isinstance(original, ConfidenceEvidenceRef)
                 or original.metadata == detail), 'Confidence lease metadata changed before copy')
        _require(detail.get('schema') == BLOCK_SCHEMA and detail.get('layout') == BLOCK_LAYOUT,
                 'Disjoint confidence leases require direct schema-2 blocks')
        _require(reference.coordinate_space == 'native_view_processing', 'Confidence lease coordinates differ')
        _require(reference.model_name == str(model_name) and reference.layer_key == str(layer_key),
                 'Confidence lease identity differs')
        _require(reference.source_shape == source_shape and tuple(detail.get('output_shape_tyx', ())) == source_shape,
                 'Confidence lease source geometry differs')
        offset = tuple(piece.get('offset_tyx', (0, 0, 0)))
        _require(len(offset) == 3, 'Invalid confidence lease offset')
        offset = tuple(_integer(v, 'offset') for v in offset)
        shape = reference.storage_shape
        _require(offset[1:] == (0, 0) and shape[1:] == native_shape[1:],
                 'Confidence leases must span the full native XY plane')
        block_size = _integer(detail.get('block_size'), 'block size', 1, 65535)
        _require(isinstance(detail.get('quantization'), str) and bool(detail['quantization']),
                 'Invalid confidence lease quantization')
        _require(detail.get('index_record_bytes') == BLOCK_DTYPE.itemsize
                 and detail.get('payload') == 'scores.u8.zlib' and detail.get('index') == 'index.bin',
                 'Invalid confidence block storage contract')
        _require(detail.get('known_voxels_coordinate_space') == 'native_view_processing',
                 'Confidence known-count coordinates differ')
        count = _integer(detail.get('block_count'), 'block count')
        known = _integer(detail.get('known_voxels'), 'known count', 0, math.prod(shape))
        _require(bool(count) == bool(known), 'Empty confidence lease count differs')
        payload_bytes = _integer(detail.get('payload_bytes'), 'payload bytes')
        payload_sha = _digest(detail.get('payload_sha256'), 'payload')
        index_sha = _digest(detail.get('index_sha256'), 'index')
        payload_path, index_path = (_contained_file(directory, x) for x in ('scores.u8.zlib', 'index.bin'))
        _require(payload_path.stat().st_size == payload_bytes, 'Confidence payload size differs')
        _require(index_path.stat().st_size == count*BLOCK_DTYPE.itemsize, 'Confidence index size differs')
        leases.append(dict(directory=directory, metadata=detail, metadata_sha=metadata_sha,
                           shape=shape, first=offset[0], count=count, known=known, block_size=block_size,
                           payload_bytes=payload_bytes, payload_sha=payload_sha, index_sha=index_sha,
                           payload_path=payload_path, index_path=index_path))
    _require(bool(leases), 'No disjoint confidence leases')
    leases.sort(key=lambda lease: lease['first'])
    expected = 0
    for lease in leases:
        _require(lease['first'] == expected, 'D1 confidence leases overlap or have gaps')
        expected += lease['shape'][0]
        _require(expected <= native_shape[0], 'Confidence lease exceeds native frame range')
    _require(expected == native_shape[0], 'D1 confidence leases do not cover the complete native view')
    _require(len({lease['block_size'] for lease in leases}) == 1, 'Confidence lease block sizes differ')
    _require(len({lease['metadata'].get('quantization') for lease in leases}) == 1,
             'Confidence lease quantization differs')
    _require(sum(lease['payload_bytes'] for lease in leases) < 2**64, 'Combined confidence payload is too large')
    return leases


def _copy_payload(lease, output, global_digest, chunk_bytes):
    digest = hashlib.sha256()
    total = 0
    with lease['payload_path'].open('rb') as stream:
        while chunk := stream.read(chunk_bytes):
            total += len(chunk)
            _require(total <= lease['payload_bytes'], 'Confidence payload grew during copy')
            output.write(chunk)
            digest.update(chunk)
            global_digest.update(chunk)
    _require(total == lease['payload_bytes'] and digest.hexdigest() == lease['payload_sha'],
             'Confidence payload changed or checksum differs')


def _rewrite_index(lease, output, global_digest, payload_base, chunk_records):
    source_digest = hashlib.sha256()
    previous_end, previous_key = 0, None
    count, area = 0, 0
    shape, block = lease['shape'], lease['block_size']
    with lease['index_path'].open('rb') as stream:
        while raw := stream.read(chunk_records*BLOCK_DTYPE.itemsize):
            _require(len(raw) % BLOCK_DTYPE.itemsize == 0, 'Truncated confidence index record')
            source_digest.update(raw)
            rows = np.frombuffer(raw, dtype=BLOCK_DTYPE)
            count += len(rows)
            _require(count <= lease['count'], 'Confidence index grew during copy')
            z, y, x, h, w, length = (rows[k].astype(np.uint64) for k in ('z', 'y', 'x', 'h', 'w', 'length'))
            offsets = rows['offset']
            _require(int(offsets[0]) == previous_end and not np.any(offsets[1:] != offsets[:-1]+length[:-1]),
                     'Confidence index payload offsets are not contiguous')
            _require(not np.any(z >= shape[0]) and not np.any(h == 0) and not np.any(w == 0)
                     and not np.any(length == 0) and not np.any(y+h > shape[1]) and not np.any(x+w > shape[2]),
                     'Confidence block lies outside its native lease')
            _require(not np.any(y//block != (y+h-1)//block) and not np.any(x//block != (x+w-1)//block),
                     'Confidence block crosses the declared grid')
            # The writer emits one crop per grid block in frame/row/column order.
            row_block, column_block = y//block, x//block
            first_key = (int(z[0]), int(row_block[0]), int(column_block[0]))
            repeated = ((z[1:] < z[:-1])
                        | ((z[1:] == z[:-1]) & (row_block[1:] < row_block[:-1]))
                        | ((z[1:] == z[:-1]) & (row_block[1:] == row_block[:-1])
                           & (column_block[1:] <= column_block[:-1])))
            _require((previous_key is None or first_key > previous_key) and not np.any(repeated),
                     'Confidence blocks repeat or are out of order')
            previous_key = (int(z[-1]), int(row_block[-1]), int(column_block[-1]))
            previous_end = int(offsets[-1])+int(length[-1])
            _require(previous_end <= lease['payload_bytes'], 'Confidence block exceeds compressed payload')
            area += int(np.sum(h*w, dtype=np.uint64))
            rewritten = rows.copy()
            rewritten['z'] += lease['first']
            rewritten['offset'] += payload_base
            data = rewritten.tobytes()
            output.write(data)
            global_digest.update(data)
    _require(count == lease['count'] and previous_end == lease['payload_bytes'],
             'Confidence index does not cover payload')
    _require(source_digest.hexdigest() == lease['index_sha'], 'Confidence index changed or checksum differs')
    _require(lease['known'] <= area, 'Confidence known count exceeds block area')


def write_disjoint_leases(path, pieces, *, native_shape, source_shape, layer_key, model_name,
                          provenance, chunk_bytes=1024*1024, chunk_records=16384):
    """Publish three files directly from immutable, full-frame native leases.

    Payload bytes are copied verbatim, with bounded index rewriting and checksum
    validation. The destination must be new; metadata is published last and any
    failed attempt removes only files owned by that attempt. Inputs are retained.
    """
    from .confidence_evidence import ConfidenceEvidenceRef, SCORE_SEMANTICS, _write_json_atomic, _json_value
    destination = Path(path).resolve()
    native_shape, source_shape = checked_shape(native_shape), checked_shape(source_shape)
    chunk_bytes = _integer(chunk_bytes, 'copy chunk bytes', 1, 16*1024*1024)
    chunk_records = _integer(chunk_records, 'index chunk records', 1, 65536)
    if destination.exists():
        raise FileExistsError(f'Confidence destination already exists: {destination}')
    leases = _preflight(pieces, native_shape, source_shape, layer_key, model_name, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    owned = []
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix='.staging-', dir=destination) as raw_staging:
            staging = Path(raw_staging)
            payload_digest, index_digest = hashlib.sha256(), hashlib.sha256()
            with (staging/'scores.u8.zlib').open('wb') as payload, (staging/'index.bin').open('wb') as index:
                for lease in leases:
                    _rewrite_index(lease, index, index_digest, payload.tell(), chunk_records)
                    _copy_payload(lease, payload, payload_digest, chunk_bytes)
            for lease in leases:
                _require(_metadata(lease['directory']/'metadata.json')[1] == lease['metadata_sha'],
                         'Confidence lease metadata changed during copy')
            known = sum(lease['known'] for lease in leases)
            metadata = dict(schema=BLOCK_SCHEMA, layout=BLOCK_LAYOUT, coordinate_space='native_view_processing',
                stored_shape_tyx=list(native_shape), source_shape_tyx=list(source_shape),
                output_shape_tyx=list(source_shape), model_name=str(model_name), layer_key=str(layer_key),
                dtype='uint8', unknown='score_zero', score_semantics=SCORE_SEMANTICS,
                quantization=leases[0]['metadata']['quantization'], reduction='maximum',
                payload='scores.u8.zlib', index='index.bin',
                index_layout='z:u32,y:u32,x:u32,h:u16,w:u16,offset:u64,length:u32; little-endian',
                index_record_bytes=BLOCK_DTYPE.itemsize, block_size=leases[0]['block_size'],
                block_count=sum(lease['count'] for lease in leases),
                payload_bytes=sum(lease['payload_bytes'] for lease in leases),
                payload_sha256=payload_digest.hexdigest(), index_sha256=index_digest.hexdigest(),
                known_voxels=known, known_contributions=known, known_voxels_coordinate_space='native_view_processing',
                count_semantics='unique native known voxels',
                stored_axes='(view_frame,view_row,view_column)', exported_axes=None,
                provenance=_json_value(provenance),
                consolidation=dict(source_piece_contract='disjoint_frame_leases', compressed_blocks_preserved=True,
                    leases=[dict(offset_tyx=[lease['first'], 0, 0], metadata=lease['metadata'],
                                 metadata_sha256=lease['metadata_sha']) for lease in leases]))
            _write_json_atomic(staging/'metadata.json', metadata)
            for name in ('scores.u8.zlib', 'index.bin', 'metadata.json'):
                target = destination/name
                (staging/name).replace(target)
                owned.append(target)
            result = ConfidenceEvidenceRef.open(destination)
            completed = True
            return result
    finally:
        if not completed:
            for target in reversed(owned):
                target.unlink(missing_ok=True)
            try:
                destination.rmdir()
            except OSError:
                pass


__all__ = ['write_disjoint_leases']
