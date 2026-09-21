"""Persistent, source-aligned instance-confidence evidence for reconciliation.

Scores are uint8 maxima of surviving instance scores. Zero explicitly means
unknown: background, unobserved support, and scores quantized to zero are not
negative evidence. These sidecars never modify the prediction or bridge masks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Callable
import zlib

import numpy as np


SCHEMA = 'xta.confidence_evidence/1'
SCORE_SEMANTICS = 'maximum_surviving_instance_confidence_u8'
_LOCK = threading.RLock()
_OUTPUT_DIR: Path | None = None
_ENABLED = False
_DEFER_PROJECTION = False
_REGISTRY: dict[tuple[str, str], 'ConfidenceEvidenceRef'] = {}


def configure_confidence_evidence(output_dir=None, *, enabled=False, min_conf=0.0,
                                  defer_projection=False):
    """Start a run-owned collection; output_dir is the run's output directory."""
    global _OUTPUT_DIR, _ENABLED, _DEFER_PROJECTION
    with _LOCK:
        _ENABLED = bool(enabled)
        _DEFER_PROJECTION = bool(defer_projection)
        _OUTPUT_DIR = None if output_dir is None else Path(output_dir) / 'reconciliation_evidence'
        _REGISTRY.clear()
    from .confidence_tiles import reset_tile_confidence
    reset_tile_confidence()


def configure_confidence_evidence_worker(*, enabled=False, min_conf=0.0):
    """Configure score-only inference semantics in an independent worker."""
    global _ENABLED
    _ENABLED = bool(enabled)


def confidence_evidence_enabled():
    return bool(_ENABLED)


def confidence_projection_deferred():
    return bool(_ENABLED and _DEFER_PROJECTION)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {str(k): _json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(_json_value(value), sort_keys=True, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@dataclass(frozen=True)
class ConfidenceEvidenceRef:
    path: Path
    shape: tuple[int, int, int]
    layer_key: str
    model_name: str
    metadata: dict

    def reader(self):
        return ConfidenceEvidenceReader(self)

    def read(self, z0, z1):
        with self.reader() as reader:
            return reader(z0, z1)

    @property
    def coordinate_space(self):
        raw = self.metadata.get('coordinate_space', self.metadata.get('provenance',{}).get('coordinate_space','source'))
        return 'native_view_processing' if raw in ('native_view_processing','tile_native_processing') else raw

    @property
    def storage_shape(self):
        return tuple(map(int,self.metadata.get('stored_shape_tyx') or self.metadata['output_shape_tyx']))

    @property
    def source_shape(self):
        value = self.metadata.get('source_shape_tyx')
        if value is None and self.coordinate_space == 'source':
            value = self.metadata.get('output_shape_tyx')
        return None if value is None else tuple(map(int,value))

    def native_reader(self):
        """Explicit access in the declared payload grid, never source projection."""
        return ConfidenceEvidenceReader(self, native=True)

    def source_reader(self, workspace=None, *, memory_mib=512, max_staging_mib=32768):
        """Explicit, bounded one-layer conversion; native readers never do this implicitly."""
        from .confidence_native import source_reader
        return source_reader(self, workspace, memory_mib=memory_mib, max_staging_mib=max_staging_mib)

    @classmethod
    def open(cls, path):
        directory = Path(path)
        data = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        from .confidence_storage import BLOCK_SCHEMA, checked_shape
        if data.get('schema') not in (SCHEMA,BLOCK_SCHEMA):
            raise ValueError('Unsupported confidence evidence schema')
        storage = checked_shape(data.get('stored_shape_tyx') or data['output_shape_tyx'])
        raw_shape = data.get('source_shape_tyx') or data.get('output_shape_tyx') or storage
        shape = checked_shape(raw_shape)
        if data.get('unknown') != 'score_zero':
            raise ValueError('Invalid confidence evidence geometry or support semantics')
        if data.get('dtype') != 'uint8' or data.get('score_semantics') != SCORE_SEMANTICS:
            raise ValueError('Invalid confidence evidence score semantics')
        result = cls(directory, shape, str(data['layer_key']), str(data['model_name']), data)
        if result.coordinate_space not in ('source','native_view_processing'):
            raise ValueError('Unsupported confidence evidence coordinate space')
        if result.coordinate_space=='source' and result.storage_shape!=result.shape:
            raise ValueError('Source confidence payload and source grids differ')
        return result


class ConfidenceEvidenceReader:
    """Bounded random-slice reader; no decompressed whole-volume cache."""
    def __init__(self, reference, *, native=False):
        self.reference = reference
        self._closed=False
        self._delegate = None
        if reference.coordinate_space != 'source' and not native:
            raise ValueError('Native confidence requires explicit source_reader(workspace, ...) conversion')
        self.shape = reference.storage_shape if native else reference.shape
        from .confidence_storage import BLOCK_SCHEMA, BLOCK_LAYOUT, PIECE_LAYOUT, BlockScoreReader
        if reference.metadata.get('schema') == BLOCK_SCHEMA:
            layout = reference.metadata.get('layout')
            if layout == BLOCK_LAYOUT:
                self._delegate = BlockScoreReader(reference)
            elif layout == PIECE_LAYOUT:
                from .confidence_native import NativePieceReader
                self._delegate = NativePieceReader(reference)
            else:
                raise ValueError('Unsupported confidence evidence storage layout')
            self.records = None
            return
        self.records = json.loads((reference.path / 'index.json').read_text(encoding='utf-8'))
        if len(self.records) != self.shape[0]:
            raise ValueError('Confidence evidence index depth differs from its output grid')
        payload_bytes = (reference.path / 'scores.u8.zlib').stat().st_size
        previous_end = 0
        for record in self.records:
            y0, y1, x0, x1, offset, length = map(int, record)
            if (not 0 <= y0 <= y1 <= self.shape[1] or not 0 <= x0 <= x1 <= self.shape[2]
                    or offset != previous_end or length < 0 or offset + length > payload_bytes
                    or (length == 0) != (y0 == y1 or x0 == x1)):
                raise ValueError('Malformed confidence evidence slice index')
            previous_end = offset + length
        if previous_end != payload_bytes:
            raise ValueError('Confidence payload has unindexed data')

    def __call__(self, z0, z1):
        if self._closed:
            raise RuntimeError('Confidence reader is closed')
        first, stop = int(z0), int(z1)
        if not 0 <= first <= stop <= self.shape[0]:
            raise IndexError('Confidence slab is outside its output grid')
        scores = np.zeros((stop - first, *self.shape[1:]), dtype=np.uint8)
        if self._delegate is not None:
            for z in range(first,stop):
                for y0,y1,x0,x1,crop in self.iter_crops(z):
                    target = scores[z-first,y0:y1,x0:x1]
                    np.maximum(target,crop,out=target)
            return scores,scores>0
        with (self.reference.path / 'scores.u8.zlib').open('rb') as stream:
            for z in range(first, stop):
                y0, y1, x0, x1, offset, length = map(int, self.records[z])
                if not length:
                    continue
                stream.seek(offset)
                compressed = stream.read(length)
                expected = (y1 - y0) * (x1 - x0)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed, expected + 1)
                if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError('Confidence score crop is truncated or malformed')
                scores[z - first, y0:y1, x0:x1] = np.frombuffer(raw, dtype=np.uint8).reshape(y1-y0, x1-x0)
        return scores, scores > 0

    def read_crop(self, z):
        """Return a known-support crop without allocating a native full plane."""
        if self._closed:
            raise RuntimeError('Confidence reader is closed')
        z = int(z)
        if not 0 <= z < self.shape[0]:
            raise IndexError('Confidence crop is outside its grid')
        if self._delegate is not None:
            from .confidence_storage import assembled_crop
            return assembled_crop(self.iter_crops(z))
        y0, y1, x0, x1, offset, length = map(int, self.records[z])
        if not length:
            return None
        with (self.reference.path / 'scores.u8.zlib').open('rb') as stream:
            stream.seek(offset)
            compressed = stream.read(length)
        expected = (y1-y0) * (x1-x0)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, expected + 1)
        if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Confidence score crop is truncated or malformed')
        return y0, y1, x0, x1, np.frombuffer(raw, dtype=np.uint8).reshape(y1-y0, x1-x0)

    def iter_crops(self,z):
        if self._closed:
            raise RuntimeError('Confidence reader is closed')
        if self._delegate is not None:
            yield from self._delegate.iter_crops(z)
        else:
            crop = self.read_crop(z)
            if crop is not None:
                yield crop

    def close(self):
        self._closed=True
        delegate = getattr(self,'_delegate',None)
        if delegate is not None:
            delegate.close()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def __del__(self):
        self.close()


def write_confidence_evidence(path, shape, slice_reader: Callable, *, layer_key, model_name,
                              provenance=None):
    """Write one source-space score sidecar using a single XY plane at a time."""
    directory = Path(path)
    shape = tuple(map(int, shape))
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Confidence output grid must contain three positive dimensions')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'metadata.json').exists():
        raise FileExistsError(f'Confidence evidence already exists: {directory}')
    payload = directory / 'scores.u8.zlib'
    temporary = directory / 'scores.u8.zlib.partial'
    records, digest, known_count = [], hashlib.sha256(), 0
    z_first, z_stop = map(int, getattr(slice_reader, 'known_z_bounds', (0, shape[0])))
    if not 0 <= z_first <= z_stop <= shape[0]:
        raise ValueError('Confidence known-support bounds are outside the output grid')
    try:
        with temporary.open('wb') as stream:
            for z in range(shape[0]):
                if not z_first <= z < z_stop:
                    records.append([0, 0, 0, 0, stream.tell(), 0])
                    continue
                value = slice_reader(z)
                if value is None:
                    records.append([0, 0, 0, 0, stream.tell(), 0])
                    continue
                plane = np.asarray(value)
                if plane.dtype != np.uint8 or plane.shape != shape[1:]:
                    raise ValueError('Confidence slice reader returned an invalid shape or dtype')
                rows = np.flatnonzero(np.any(plane, axis=1))
                offset = stream.tell()
                if not len(rows):
                    records.append([0, 0, 0, 0, offset, 0])
                    continue
                y0, y1 = int(rows[0]), int(rows[-1]) + 1
                columns = np.flatnonzero(np.any(plane[y0:y1], axis=0))
                x0, x1 = int(columns[0]), int(columns[-1]) + 1
                crop = np.ascontiguousarray(plane[y0:y1, x0:x1])
                known_count += int(np.count_nonzero(crop))
                encoded = zlib.compress(crop.tobytes(), level=3)
                stream.write(encoded)
                digest.update(encoded)
                records.append([y0, y1, x0, x1, offset, len(encoded)])
        temporary.replace(payload)
        _write_json_atomic(directory / 'index.json', records)
        metadata = dict(
            schema=SCHEMA, layer_key=str(layer_key), model_name=str(model_name),
            output_shape_tyx=list(shape), exported_axes='(X, Y, t)', dtype='uint8',
            score_semantics=SCORE_SEMANTICS, unknown='score_zero',
            quantization='round-half-even(clip(instance_score,0,1)*255); quantized zero is unknown',
            projection='maximum over categorical source-address support',
            payload='scores.u8.zlib', index='index.json', payload_sha256=digest.hexdigest(),
            known_voxels=known_count, provenance=_json_value(provenance or {}))
        _write_json_atomic(directory / 'metadata.json', metadata)
        return ConfidenceEvidenceRef(directory, shape, str(layer_key), str(model_name), metadata)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_block_confidence_evidence(path, shape, slice_reader: Callable, *, layer_key, model_name,
                                    provenance=None, coordinate_space=None, source_shape_tyx=None,
                                    block_size=128):
    """Write schema2 numeric blocks without a whole-plane bounding rectangle."""
    from .confidence_storage import write_blocks
    provenance = provenance or {}
    space = coordinate_space or provenance.get('coordinate_space','source')
    if space == 'tile_native_processing':
        space = 'native_view_processing'
    write_blocks(path,shape,slice_reader,layer_key=layer_key,model_name=model_name,
        provenance=provenance,coordinate_space=space,source_shape=source_shape_tyx,block_size=block_size)
    return ConfidenceEvidenceRef.open(path)


def _register_confidence_reference(reference):
    with _LOCK:
        output = _OUTPUT_DIR
        if output is None:
            raise RuntimeError('Confidence evidence has no configured output directory')
        token = (reference.model_name,reference.layer_key)
        if token in _REGISTRY:
            raise RuntimeError(f'Duplicate confidence evidence for {token!r}')
        _REGISTRY[token] = reference
        _write_json_atomic(output/'manifest.json',dict(schema=SCHEMA,layers=[
            dict(model_name=m,layer_key=k,directory=ref.path.name,output_shape_tyx=list(ref.shape),
                 storage_schema=ref.metadata['schema'],coordinate_space=ref.coordinate_space,
                 stored_shape_tyx=list(ref.storage_shape),score_semantics=SCORE_SEMANTICS,
                 unknown='score_zero') for (m,k),ref in sorted(_REGISTRY.items())]))
    return reference


def _confidence_destination(model_name,layer_key):
    with _LOCK:
        output = _OUTPUT_DIR
    if output is None:
        raise RuntimeError('Confidence evidence publication has no configured run output directory')
    identity = hashlib.sha256((str(model_name)+'\0'+str(layer_key)).encode()).hexdigest()
    return output/identity


def publish_confidence_scores(scores, *, view, model_name, temp_dir, output_shape=None,
                               source='fullframe', tile_config_id='', tile_acceptance='',
                               stage='pre_interpolation', layer_key=None):
    """Project immutable native-view scores and persist one prediction companion."""
    from .assembly import final_source_output_shape
    from .confidence_projection import score_projection_reader
    from .cuda_d1 import _nrrd_layer_key
    if not confidence_evidence_enabled():
        return None
    with _LOCK:
        output = _OUTPUT_DIR
    if output is None:
        raise RuntimeError('Confidence evidence publication has no configured run output directory')
    key = str(layer_key or _nrrd_layer_key(
        view_name=view.name, source=source, mask_kind='yolo', pass_index=0,
        tile_config_id=tile_config_id, tile_acceptance=tile_acceptance, stage=stage))
    identity = hashlib.sha256((str(model_name) + '\0' + key).encode('utf-8')).hexdigest()
    shape = tuple(output_shape or final_source_output_shape() or
                  (int(view.full_t), int(view.full_h), int(view.full_w)))
    work = Path(temp_dir) / 'confidence_evidence' / identity
    provenance = dict(view=_json_value(view), processing_shape_tyx=list(scores.shape),
                      source=source, mask_kind='yolo', stage=stage,
                      tile_config_id=tile_config_id, tile_acceptance=tile_acceptance,
                      confidence_scope='surviving observed prediction support before interpolation')
    started = time.perf_counter()
    operation = 'native retention' if confidence_projection_deferred() else 'source projection'
    print(f'Confidence {operation} start {model_name}/{view.name}: '
          f'native_shape={tuple(scores.shape)}, source_shape={shape}.', flush=True)
    if confidence_projection_deferred():
        reference = write_block_confidence_evidence(output/identity,scores.shape,lambda z:scores[z],
            layer_key=key,model_name=model_name,provenance=provenance,
            coordinate_space='native_view_processing',source_shape_tyx=shape)
    elif not any(bool(np.any(scores[index])) for index in range(scores.shape[0])):
        reference = write_block_confidence_evidence(output / identity, shape, lambda _z: None,
            layer_key=key, model_name=model_name, provenance=provenance)
    else:
        with score_projection_reader(scores, view, shape, work) as reader:
            reference = write_block_confidence_evidence(output / identity, shape, reader,
                layer_key=key, model_name=model_name, provenance=provenance)
    result = _register_confidence_reference(reference)
    print(f'Confidence {operation} complete {model_name}/{view.name}: '
          f'elapsed_s={time.perf_counter()-started:.3f}.', flush=True)
    return result


def capture_prediction_confidence(mask, scores, *, view, model_name, temp_dir, **kwargs):
    """Mask a retiring score workspace, then publish before it is closed/deleted."""
    if not confidence_evidence_enabled():
        return None
    if scores is None:
        raise RuntimeError(f'Missing retained confidence for {model_name}/{view.name}')
    values, retained = np.asarray(scores), np.asarray(mask)
    if values.dtype != np.uint8 or values.shape != retained.shape or values.ndim != 3:
        raise ValueError('Retained confidence and prediction mask geometry differ')
    for index in range(values.shape[0]):
        values[index][np.asarray(retained[index]) == 0] = np.uint8(0)
    return publish_confidence_scores(values, view=view, model_name=model_name, temp_dir=temp_dir, **kwargs)


def lookup_confidence_evidence(layer, model_name=None):
    if hasattr(layer, 'key'):
        key = str(layer.key)
        model_name = str(layer.model_name)
    else:
        key = str(layer)
    with _LOCK:
        if model_name is not None:
            return _REGISTRY.get((str(model_name), key))
        matches = [ref for (model, stored_key), ref in _REGISTRY.items() if stored_key == key]
    if len(matches) > 1:
        raise ValueError('Confidence layer key requires a model identity')
    return matches[0] if matches else None


def publish_native_confidence_pieces(pieces, *, native_shape, view, model_name, output_shape=None,
                                     source='fullframe',tile_config_id='',tile_acceptance='',
                                     stage='pre_interpolation',layer_key=None,disjoint=False):
    """Persist native pieces and complete geometry without creating a dense map."""
    from .assembly import final_source_output_shape
    from .cuda_d1 import _nrrd_layer_key
    from .confidence_native import write_native_pieces
    key=str(layer_key or _nrrd_layer_key(view_name=view.name,source=source,mask_kind='yolo',
        pass_index=0,tile_config_id=tile_config_id,tile_acceptance=tile_acceptance,stage=stage))
    shape=tuple(output_shape or final_source_output_shape() or (view.full_t,view.full_h,view.full_w))
    provenance=dict(view=_json_value(view),processing_shape_tyx=list(native_shape),source=source,
        mask_kind='yolo',stage=stage,tile_config_id=tile_config_id,tile_acceptance=tile_acceptance,
        confidence_scope='surviving observed prediction support before interpolation')
    ref=write_native_pieces(_confidence_destination(model_name,key),pieces,native_shape=native_shape,
        source_shape=shape,layer_key=key,model_name=model_name,provenance=provenance,disjoint=disjoint)
    return _register_confidence_reference(ref)


def publish_confidence_shards(shards, *, view, model_name, temp_dir, output_shape=None):
    """Join exact disjoint D1 frame leases in one disk-backed native score map."""
    from .runtime import close_memmap_array_without_flush
    records = sorted(shards, key=lambda item: int(item['slice_start']))
    if not records:
        raise ValueError(f'No D1 confidence shards for {model_name}/{view.name}')
    shape = tuple(map(int, records[0]['view_shape_tyx']))
    if len(shape) != 3 or shape[0] != int(view.num_slices) or min(shape) <= 0:
        raise ValueError('D1 confidence native view geometry is invalid')
    expected_start = 0
    layer_key = str(records[0]['layer_key'])
    for record in records:
        count = int(record['slice_count'])
        if (str(record['model_name']) != str(model_name) or str(record['view_name']) != str(view.name)
                or tuple(map(int, record['view_shape_tyx'])) != shape
                or str(record['layer_key']) != layer_key
                or int(record['slice_start']) != expected_start or count <= 0
                or tuple(map(int, record['shape_tyx'])) != (count, *shape[1:])):
            raise ValueError('D1 confidence leases overlap, have gaps, or change identity/geometry')
        expected_start += count
    if expected_start != shape[0]:
        raise ValueError('D1 confidence leases do not cover the complete native view')
    if confidence_projection_deferred():
        pieces=[]
        for record in records:
            ref=ConfidenceEvidenceRef.open(record['path'])
            if (ref.storage_shape!=tuple(map(int,record['shape_tyx']))
                    or ref.model_name!=str(model_name) or ref.layer_key!=layer_key):
                raise ValueError('D1 confidence shard descriptor differs from its stored identity or grid')
            pieces.append(dict(reference=ref,offset_tyx=(int(record['slice_start']),0,0)))
        return publish_native_confidence_pieces(pieces,native_shape=shape,view=view,model_name=model_name,
            output_shape=output_shape,layer_key=layer_key,disjoint=True)
    identity = hashlib.sha256((str(model_name) + '\0' + str(view.name)).encode()).hexdigest()
    path = Path(temp_dir) / 'confidence_evidence' / identity / 'native_score_leases.u8.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = np.memmap(path, dtype=np.uint8, mode='w+', shape=shape)
    try:
        for record in records:
            ref = ConfidenceEvidenceRef.open(record['path'])
            if (ref.storage_shape != tuple(map(int, record['shape_tyx']))
                    or ref.model_name != str(model_name) or ref.layer_key != layer_key):
                raise ValueError('D1 confidence shard descriptor differs from its stored identity or grid')
            start = int(record['slice_start'])
            with ref.native_reader() as reader:
                for local in range(ref.storage_shape[0]):
                    for y0,y1,x0,x1,crop in reader.iter_crops(local):
                        merged[start + local, y0:y1, x0:x1] = crop
        return publish_confidence_scores(merged, view=view, model_name=model_name,
            temp_dir=temp_dir, output_shape=output_shape, layer_key=layer_key)
    finally:
        close_memmap_array_without_flush(merged)
        path.unlink(missing_ok=True)


__all__ = [
    'ConfidenceEvidenceRef', 'ConfidenceEvidenceReader', 'configure_confidence_evidence',
    'configure_confidence_evidence_worker', 'confidence_evidence_enabled',
    'confidence_projection_deferred',
    'capture_prediction_confidence',
    'publish_confidence_scores', 'lookup_confidence_evidence', 'write_confidence_evidence',
    'publish_confidence_shards',
    'write_block_confidence_evidence','publish_native_confidence_pieces',
]
