"""Selection and provenance for observed numeric prediction confidence."""
from __future__ import annotations

import json
import math
from pathlib import Path


def confidence_retention_requested(save_options=(), *, reconciliation_mode=None):
    """Capture scores needed for voting or for an explicitly requested output."""
    return reconciliation_mode == 'confidence' or 'confidence' in save_options


def observed_confidence_contract(*, prediction_task=None, confidence_floor=None,
                                 min_conf=0.0, confidence_floor_by_backend=None):
    """Describe captured values without assigning probability to generated support."""
    task = None if prediction_task is None else str(prediction_task)
    if task not in (None, 'segment', 'semantic'):
        raise ValueError('Observed confidence requires segment or semantic prediction task')
    floor = None if confidence_floor is None else float(confidence_floor)
    component = float(min_conf)
    if floor is not None and (not math.isfinite(floor) or not 0 <= floor <= 1):
        raise ValueError('Confidence capture floor must be finite and in [0,1]')
    if not math.isfinite(component) or not 0 <= component <= 1:
        raise ValueError('Confidence component threshold must be finite and in [0,1]')
    floors = {str(backend): float(value)
              for backend, value in (confidence_floor_by_backend or {}).items()}
    if any(not backend or not math.isfinite(value) or not 0 <= value <= 1
           for backend, value in floors.items()):
        raise ValueError('Backend confidence floors must be finite and in [0,1]')
    return dict(
        schema='xta.observed_confidence/1', prediction_task=task,
        score_kind=('semantic_foreground_probability' if task == 'semantic' else
                    'instance_detection_confidence' if task == 'segment' else
                    'prediction_score'),
        support_origin='observed_prediction',
        support_scope='surviving prediction support before interpolation',
        generated_only_support='unknown', unknown='score_zero',
        prediction_confidence_threshold=floor,
        capture_confidence_floor=min(floors.values()) if floors else floor,
        capture_confidence_floor_by_backend=floors, component_min_conf=component,
        reduction='maximum over covering observed predictions',
        quantization=dict(dtype='uint8', scale=255, rounding='half_even',
                          zero='unknown',
                          intermediate_precision='backend-specific floating point'))


def published_confidence_manifest(output_dir, *, run=None):
    """Resolve the run's public score catalogue, including legacy receipts."""
    if run is not None:
        processing = run.get('prediction_processing') or {}
        if 'observed_confidence' in processing:
            observed = processing['observed_confidence']
            if observed is None or observed.get('retained') is False:
                return None
    path = Path(output_dir) / 'reconciliation_evidence' / 'manifest.json'
    if not path.is_file():
        return None
    if run is not None and run.get('run_id') is not None:
        if json.loads(path.read_bytes()).get('run_id') != run['run_id']:
            raise ValueError('Confidence catalogue and run manifest come from different runs')
    return path
