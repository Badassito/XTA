"""Retain original tile scores through immutable parent/component support gates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import threading

import numpy as np

from .confidence_evidence import (
    ConfidenceEvidenceRef, capture_prediction_confidence, confidence_evidence_enabled,
    write_block_confidence_evidence, confidence_projection_deferred, publish_native_confidence_pieces,
    _can_stream_eager_planes, _publish_eager_transverse_pieces,
)


@dataclass(frozen=True)
class _TileScores:
    reference: ConfidenceEvidenceRef
    parent_crop: tuple[int, int, int, int]
    work_dir: Path


_LOCK = threading.RLock()
_RAW: dict[tuple[str, str, str, str], _TileScores] = {}
_ACCEPTED: dict[tuple[str, str, str, str], list[_TileScores]] = {}


def reset_tile_confidence():
    with _LOCK:
        _RAW.clear()
        _ACCEPTED.clear()


def _tile_key(result):
    return tuple(str(getattr(result, field)) for field in ('model_name', 'view_name', 'config_id', 'tile_id'))


def capture_tile_confidence(task, *, view, work_dir, destinations=None):
    """Preserve crop-local scores before the inference workspace is retired."""
    if not confidence_evidence_enabled():
        return
    if task.tile_confmap_mm is None:
        raise RuntimeError(f'Missing confidence for tile {task.view_name}/{task.tile_id}')
    scores, mask = np.asarray(task.tile_confmap_mm), np.asarray(task.tile_mask_mm)
    if scores.dtype != np.uint8 or scores.shape != mask.shape or scores.ndim != 3:
        raise ValueError('Tile confidence and mask geometry differ')
    key = _tile_key(task)
    digest = hashlib.sha256('\0'.join(key).encode()).hexdigest()
    root = Path(work_dir) / 'tile_confidence' / digest
    def read(z):
        if destinations is not None:
            z = destinations.get(z)
            if z is None:
                return None
        # A new slice owns its zeros; the original mask is never changed.
        return np.where(mask[z] != 0, scores[z], np.uint8(0)).astype(np.uint8, copy=False)
    shape = scores.shape if destinations is None else (int(view.num_slices), *scores.shape[1:])
    root.mkdir(parents=True)
    try:
        ref = write_block_confidence_evidence(root / 'native', shape, read,
            layer_key=digest, model_name=task.model_name,
            provenance=dict(coordinate_space='tile_native_processing', view_name=view.name,
                            tile_id=task.tile_id, config_id=task.config_id, parent_crop=list(task.parent_crop)))
        with _LOCK:
            if key in _RAW:
                raise RuntimeError(f'Duplicate original tile confidence: {key}')
            _RAW[key] = _TileScores(ref, tuple(map(int, task.parent_crop)), root)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def record_tile_confidence_gate(result, category, gate_slice, *, workers):
    """Consume the gate's accepted crops in slice order while encoding scores."""
    if category not in ('parent_mask', 'parent_bridge'):
        raise ValueError(f'Unknown tile confidence support category {category!r}')
    from contextlib import closing
    from .runtime import parallel_map_in_order
    key = _tile_key(result)
    with _LOCK:
        retained = _RAW.get(key)
    if retained is None:
        raise RuntimeError(f'Tile gate lost original confidence: {key}')
    shape = retained.reference.storage_shape
    accepted_crops = parallel_map_in_order(gate_slice, range(shape[0]),
        max_workers=workers, max_pending=2 * workers)
    with closing(accepted_crops), retained.reference.native_reader() as scores:
        def read(z):
            crop = next(accepted_crops)
            if crop is None:
                return None
            y0, x0, accepted = crop
            if not accepted.any():
                return None
            y1, x1 = y0 + accepted.shape[0], x0 + accepted.shape[1]
            plane = np.zeros(shape[1:], np.uint8)
            for ry0, ry1, rx0, rx1, values in scores.iter_crops(z):
                a, b = max(y0, ry0), min(y1, ry1)
                c, d = max(x0, rx0), min(x1, rx1)
                if a < b and c < d:
                    plane[a:b, c:d] = np.where(accepted[a-y0:b-y0, c-x0:d-x0],
                        values[a-ry0:b-ry0, c-rx0:d-rx0], np.uint8(0))
            return plane
        reference = write_block_confidence_evidence(
            retained.work_dir / category, shape, read,
            layer_key=retained.reference.layer_key + '/' + category, model_name=result.model_name,
            provenance=dict(coordinate_space='tile_native_processing', tile_acceptance=category,
                            original_tile_id=result.tile_id, parent_crop=list(retained.parent_crop)))
    if reference.metadata['known_voxels']:
        token = (str(result.model_name), str(result.view_name), str(result.config_id), str(category))
        with _LOCK:
            _ACCEPTED.setdefault(token, []).append(_TileScores(reference, retained.parent_crop, retained.work_dir))


def retire_tile_confidence(model_name, view_name, config_id):
    """Drop tile-local scores after all gate and publication consumers settle."""
    if not confidence_evidence_enabled():
        return
    prefix = (str(model_name), str(view_name), str(config_id))
    with _LOCK:
        roots = {_RAW.pop(key).work_dir for key in tuple(_RAW) if key[:3] == prefix}
        for category in ('parent_mask', 'parent_bridge'):
            _ACCEPTED.pop((*prefix, category), None)
    for root in roots:
        shutil.rmtree(root)


def discard_unpublished_tile_confidence(result):
    """Retire an original tile before it reaches either component gate."""
    with _LOCK:
        tile = _RAW.pop(_tile_key(result), None)
    if tile is not None:
        shutil.rmtree(tile.work_dir, ignore_errors=True)


def capture_consolidated_tile_confidence(mask, *, view, model_name, config_id,
                                         category, stage, temp_dir):
    """Max original accepted tile scores, then project the matching category layer."""
    if not confidence_evidence_enabled():
        return None
    from .runtime import close_memmap_array_without_flush
    prefix = (str(model_name), str(view.name), str(config_id))
    categories = ('parent_mask', 'parent_bridge') if category == 'parent_support' else (str(category),)
    with _LOCK:
        tiles = [entry for kind in categories for entry in _ACCEPTED.get((*prefix, kind), ())]
    shape = tuple(map(int, mask.shape))
    pieces = []
    for tile in tiles:
        py0, py1, px0, px1 = tile.parent_crop
        if tile.reference.storage_shape != (shape[0], py1-py0, px1-px0):
            raise ValueError('Accepted tile confidence changed parent-grid geometry')
        pieces.append(dict(reference=tile.reference, offset_tyx=(0, py0, px0)))
    if confidence_projection_deferred():
        return publish_native_confidence_pieces(pieces,native_shape=shape,view=view,model_name=model_name,
            source='tile',tile_config_id=config_id,tile_acceptance=category,stage=stage,disjoint=False)
    if _can_stream_eager_planes(view):
        return _publish_eager_transverse_pieces(pieces, shape, mask=mask, view=view,
            model_name=model_name, temp_dir=temp_dir, source='tile', tile_config_id=config_id,
            tile_acceptance=category, stage=stage)
    digest = hashlib.sha256('\0'.join((*prefix, str(category))).encode()).hexdigest()
    path = Path(temp_dir) / 'confidence_evidence' / digest / 'accepted_tiles.u8.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = np.memmap(path, dtype=np.uint8, mode='w+', shape=shape)
    try:
        for piece in pieces:
            ref, (_, py0, px0) = piece['reference'], piece['offset_tyx']
            with ref.native_reader() as reader:
                for z in range(shape[0]):
                    for y0,y1,x0,x1,values in reader.iter_crops(z):
                        destination = merged[z, py0+y0:py0+y1, px0+x0:px0+x1]
                        np.maximum(destination, values, out=destination)
        reference = capture_prediction_confidence(mask, merged, view=view, model_name=model_name,
            temp_dir=temp_dir, source='tile', tile_config_id=config_id,
            tile_acceptance=category, stage=stage)
    finally:
        close_memmap_array_without_flush(merged, unlink_path=path)
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass
    return reference


__all__ = ['capture_tile_confidence', 'record_tile_confidence_gate',
           'capture_consolidated_tile_confidence', 'retire_tile_confidence', 'reset_tile_confidence']
