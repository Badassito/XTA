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
from typing import Callable
import zlib

import numpy as np


SCHEMA = 'xta.confidence_evidence/1'
SCORE_SEMANTICS = 'maximum_surviving_instance_confidence_u8'
_LOCK = threading.RLock()
_OUTPUT_DIR: Path | None = None
_ENABLED = False
_REGISTRY: dict[tuple[str, str], 'ConfidenceEvidenceRef'] = {}


def configure_confidence_evidence(output_dir=None, *, enabled=False, min_conf=0.0):
    """Start a run-owned collection; output_dir is the run's output directory."""
    global _OUTPUT_DIR, _ENABLED
    with _LOCK:
        _ENABLED = bool(enabled)
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
        return self.reader()(z0, z1)

    @classmethod
    def open(cls, path):
        directory = Path(path)
        data = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if data.get('schema') != SCHEMA:
            raise ValueError('Unsupported confidence evidence schema')
        shape = tuple(int(v) for v in data['output_shape_tyx'])
        if len(shape) != 3 or min(shape) <= 0 or data.get('unknown') != 'score_zero':
            raise ValueError('Invalid confidence evidence geometry or support semantics')
        if data.get('dtype') != 'uint8' or data.get('score_semantics') != SCORE_SEMANTICS:
            raise ValueError('Invalid confidence evidence score semantics')
        return cls(directory, shape, str(data['layer_key']), str(data['model_name']), data)


class ConfidenceEvidenceReader:
    """Bounded random-slice reader; no decompressed whole-volume cache."""
    def __init__(self, reference):
        self.reference = reference
        self.shape = reference.shape
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
        first, stop = int(z0), int(z1)
        if not 0 <= first <= stop <= self.shape[0]:
            raise IndexError('Confidence slab is outside its output grid')
        scores = np.zeros((stop - first, *self.shape[1:]), dtype=np.uint8)
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
        z = int(z)
        if not 0 <= z < self.shape[0]:
            raise IndexError('Confidence crop is outside its grid')
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
    if not any(bool(np.any(scores[index])) for index in range(scores.shape[0])):
        reference = write_confidence_evidence(output / identity, shape, lambda _z: None,
            layer_key=key, model_name=model_name, provenance=provenance)
    else:
        with score_projection_reader(scores, view, shape, work) as reader:
            reference = write_confidence_evidence(output / identity, shape, reader,
                layer_key=key, model_name=model_name, provenance=provenance)
    with _LOCK:
        token = (str(model_name), key)
        if token in _REGISTRY:
            raise RuntimeError(f'Duplicate confidence evidence for {token!r}')
        _REGISTRY[token] = reference
        _write_json_atomic(output / 'manifest.json', dict(schema=SCHEMA, layers=[
            dict(model_name=m, layer_key=k, directory=ref.path.name,
                 output_shape_tyx=list(ref.shape), score_semantics=SCORE_SEMANTICS,
                 unknown='score_zero') for (m, k), ref in sorted(_REGISTRY.items())]))
    return reference


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
    identity = hashlib.sha256((str(model_name) + '\0' + str(view.name)).encode()).hexdigest()
    path = Path(temp_dir) / 'confidence_evidence' / identity / 'native_score_leases.u8.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = np.memmap(path, dtype=np.uint8, mode='w+', shape=shape)
    try:
        for record in records:
            ref = ConfidenceEvidenceRef.open(record['path'])
            if (ref.shape != tuple(map(int, record['shape_tyx']))
                    or ref.model_name != str(model_name) or ref.layer_key != layer_key):
                raise ValueError('D1 confidence shard descriptor differs from its stored identity or grid')
            reader = ref.reader()
            start = int(record['slice_start'])
            for local, index in enumerate(reader.records):
                y0, y1, x0, x1, offset, length = map(int, index)
                if length:
                    crop = reader.read_crop(local)
                    merged[start + local, y0:y1, x0:x1] = crop[4]
        return publish_confidence_scores(merged, view=view, model_name=model_name,
            temp_dir=temp_dir, output_shape=output_shape, layer_key=layer_key)
    finally:
        close_memmap_array_without_flush(merged)
        path.unlink(missing_ok=True)


__all__ = [
    'ConfidenceEvidenceRef', 'ConfidenceEvidenceReader', 'configure_confidence_evidence',
    'configure_confidence_evidence_worker', 'confidence_evidence_enabled',
    'capture_prediction_confidence',
    'publish_confidence_scores', 'lookup_confidence_evidence', 'write_confidence_evidence',
    'publish_confidence_shards',
]
