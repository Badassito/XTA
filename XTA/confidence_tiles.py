"""Retain original tile scores through immutable parent/component support gates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading

import numpy as np

from .confidence_evidence import (
    ConfidenceEvidenceRef, capture_prediction_confidence, confidence_evidence_enabled,
    write_confidence_evidence,
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


def capture_tile_confidence(task, *, view, work_dir):
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
        # A new slice owns its zeros; the original mask is never changed.
        return np.where(mask[z] != 0, scores[z], np.uint8(0)).astype(np.uint8, copy=False)
    ref = write_confidence_evidence(root / 'native', scores.shape, read,
        layer_key=digest, model_name=task.model_name,
        provenance=dict(coordinate_space='tile_native_processing', view_name=view.name,
                        tile_id=task.tile_id, config_id=task.config_id, parent_crop=list(task.parent_crop)))
    with _LOCK:
        if key in _RAW:
            raise RuntimeError(f'Duplicate original tile confidence: {key}')
        _RAW[key] = _TileScores(ref, tuple(map(int, task.parent_crop)), root)


def record_tile_confidence_gate(result, support, category):
    """Apply the existing whole-component gate to scores before masks mutate."""
    if not confidence_evidence_enabled():
        return
    if category not in ('parent_mask', 'parent_bridge'):
        raise ValueError(f'Unknown tile confidence support category {category!r}')
    from .assembly import _partition_tile_components_2d
    from .cuda_d1 import _read_binary_volume_slice_crop_bool
    key = _tile_key(result)
    with _LOCK:
        retained = _RAW.get(key)
    if retained is None:
        raise RuntimeError(f'Tile gate lost original confidence: {key}')
    py0, py1, px0, px1 = retained.parent_crop
    source = result.tile_mask_mm if result.tile_mask_mm is not None else result.tile_mask_store
    if source is None:
        raise ValueError('Tile confidence gate requires the original or residual tile mask')
    scores = retained.reference.reader()
    def read(z):
        plane = _read_binary_volume_slice_crop_bool(source, z, 0, py1-py0, 0, px1-px0)
        parent = _read_binary_volume_slice_crop_bool(support, z, py0, py1, px0, px1)
        accepted, _, _ = _partition_tile_components_2d(plane, parent)
        value, _known = scores(z, z+1)
        return np.where(accepted != 0, value[0], np.uint8(0)).astype(np.uint8, copy=False)
    reference = write_confidence_evidence(retained.work_dir / category, retained.reference.shape, read,
        layer_key=retained.reference.layer_key + '/' + category, model_name=result.model_name,
        provenance=dict(coordinate_space='tile_native_processing', tile_acceptance=category,
                        original_tile_id=result.tile_id, parent_crop=list(retained.parent_crop)))
    if reference.metadata['known_voxels']:
        token = (str(result.model_name), str(result.view_name), str(result.config_id), str(category))
        with _LOCK:
            _ACCEPTED.setdefault(token, []).append(_TileScores(reference, retained.parent_crop, retained.work_dir))


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
    shape = tuple(map(int, np.asarray(mask).shape))
    digest = hashlib.sha256('\0'.join((*prefix, str(category))).encode()).hexdigest()
    path = Path(temp_dir) / 'confidence_evidence' / digest / 'accepted_tiles.u8.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = np.memmap(path, dtype=np.uint8, mode='w+', shape=shape)
    try:
        for tile in tiles:
            py0, py1, px0, px1 = tile.parent_crop
            if tile.reference.shape != (shape[0], py1-py0, px1-px0):
                raise ValueError('Accepted tile confidence changed parent-grid geometry')
            reader = tile.reference.reader()
            for z in range(shape[0]):
                crop = reader.read_crop(z)
                if crop is None:
                    continue
                y0, y1, x0, x1, values = crop
                destination = merged[z, py0+y0:py0+y1, px0+x0:px0+x1]
                np.maximum(destination, values, out=destination)
        reference = capture_prediction_confidence(mask, merged, view=view, model_name=model_name,
            temp_dir=temp_dir, source='tile', tile_config_id=config_id,
            tile_acceptance=category, stage=stage)
    finally:
        close_memmap_array_without_flush(merged)
        path.unlink(missing_ok=True)
    return reference


__all__ = ['capture_tile_confidence', 'record_tile_confidence_gate',
           'capture_consolidated_tile_confidence', 'reset_tile_confidence']
