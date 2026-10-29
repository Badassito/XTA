"""Source and requested-runtime provenance for TTA execution receipts."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

from .cylindrical_owner import radial_runtime_provenance
from .file_hashing import sha256_file
from .workspace import _env_flag, _env_int


def _source_record(package: Path, name: str) -> Dict[str, object]:
    """Fingerprint without importing optional runtimes; retain read failures."""
    path = package.joinpath(*name.split('.')).with_suffix('.py')
    entry = {'path': str(path), 'loaded_in_parent': 'XTA.' + name in sys.modules}
    try:
        entry['sha256'] = sha256_file(path)
    except OSError as exc:
        entry.update(sha256=None, read_error=type(exc).__name__)
    return entry


def _execution_runtime_provenance() -> Dict[str, object]:
    """Include nonlinear geometry and scheduling sources in the run receipt."""
    from .geometry_quality import geometry_quality_request_record
    from .cuda_backend import native_trt_ring_enabled, native_trt_ring_mode

    result = radial_runtime_provenance()
    sources = {}
    package = Path(__file__).resolve().parent
    for name in ('geometry', 'cylindrical_geometry', 'spherical_sampling', 'azimuthal_coverage',
                 'unification.runtime', 'qsc', 'spherical_geometry', 'spherical_cuda', 'spherical_sampling_cuda',
                 'cuda_backend', 'runtime', 'startup_io', 'spherical_projection',
                 'spherical_projection_bounds', 'spherical_projection_cuda',
                 'spherical_preflight', 'spherical_projection_cpu', 'geometry_quality',
                 'unification.sampling', 'cylindrical_cuda_projection',
                 'tilted_azimuthal_projection', 'tilted_azimuthal_projection_cuda',
                 'tta_scheduler', 'backprojection', 'd1_orthogonal_coverage',
                 'projection_coverage', 'projection_coverage_cpu', 'azimuthal_trajectory_cpu',
                 'sparse_projection', 'sparse_bounds_cpu', 'publication_memory',
                 'native_pull_scheduling', 'native_pull_cuda', 'native_encoded_cuda', 'packed_tilted_projection',
                 'packed_dense_source',
                 'confidence_evidence', 'confidence_storage',
                 'confidence_capture', 'confidence_capture_cpu', 'confidence_native_codec', 'confidence_orchestration',
                 'confidence_publication', 'tta_prepare_planning', 'tta_provenance',
                 'runtime_observability'):
        sources[name] = _source_record(package, name)
    result['spherical_sources'] = sources
    # A package version can describe several development snapshots. Retain the
    # actual SAM preparation/publication bytes for next-run timing attribution
    # without importing optional model runtimes merely to fingerprint them.
    for name in ('assembly', 'view_prepare', 'projection_queue', 'reconciliation_runtime', 'sam_integration',
                 'sam_sparse_shell_projection', 'sam_bridge_planning',
                 'sam_observation_masks', 'sam_interpolation_contracts',
                 'sam_extrapolation_planning', 'sam_gpu_rendering', 'sam_azimuthal_batch', 'workspace', 'sam_retry_attempt_bounds',
                 'sam_interpolation', 'sam_extrapolation', 'sam_extrapolation_policy', 'sam_retry_groups',
                 'sam_tracker_runtime', 'sam_device_pool', 'sam_resources', 'sam_image_prefetch', 'sam_crop_retry',
                 'lta_worker_adapter', 'lta_sam', 'sam_sdk_compat',
                 'sam_parent_staging', 'tta_background', 'detector_workspaces', 'yolo_cpu_input', 'gray_cache', 'sam_yolo_capture', 'sam_view_geometry',
                 'sam_policy', 'sam_mask_reader', 'sam_planning_outcomes',
                 'sam_branch_selection', 'sam_evidence', 'sam_canvas_rendering', 'tta_outputs',
                 'nrrd_compression', '_prediction_cleanup'):
        result['modules'][name] = _source_record(package, name)
    result['geometry_quality_requests'] = geometry_quality_request_record()
    result['native_trt_ring_requested'] = native_trt_ring_enabled()
    result['native_trt_ring_mode'] = native_trt_ring_mode()
    result['task_trace_requested'] = _env_flag('YOLO_TTA_TASK_TRACE', False)
    result['sam_gray_capture_requested'] = _env_flag('YOLO_TTA_SAM_GRAY_CACHE', True)
    result['cpu_input_prefetch_requested'] = _env_flag('YOLO_TTA_CPU_INPUT_PREFETCH', True)
    result['tilted_cpu_input_prefetch_requested'] = _env_flag('YOLO_TTA_CPU_INPUT_TILTED', False)
    result['terminal_confidence_score_only_requested'] = _env_flag('YOLO_TTA_TERMINAL_SCORE_ONLY', True)
    result['cpu_input_prefetch_policy_requests'] = dict(
        default_builders_per_gpu=1, default_packets_per_gpu=2,
        builders_override=os.environ.get('YOLO_TTA_CPU_INPUT_BUILDERS'),
        packets_override=os.environ.get('YOLO_TTA_CPU_INPUT_PACKETS'))
    result['cropped_upload_pipeline_requested'] = True
    result['spherical_cpu_compact_requested'] = True
    result['tilted_azimuthal_cuda_projection_requested'] = _env_flag(
        'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT', True)
    result['native_projection_requests'] = {
        'packed_parent': _env_flag('YOLO_TTA_PACKED_PARENT', True),
        'packed_parent_min_mib': max(0, _env_int('YOLO_TTA_PACKED_PARENT_MIN_MIB', 64)),
        'exact_cuda': _env_flag('YOLO_TTA_NATIVE_PULL_CUDA', True),
        'cuda_min_remaining_voxels': max(0, _env_int('YOLO_TTA_NATIVE_PULL_GPU_MIN_VOXELS', 64 * 1024**2)),
        'shared_cpu_budget': True,
    }
    result['spherical_retirement_requests'] = {
        'enabled': _env_flag('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT', True),
        'age_enabled': _env_flag('YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT', True),
        'continuous_wait_seconds': 30.0,
    }
    result['spherical_locality_requested'] = bool(_env_int('YOLO_TTA_GPU_SPHERICAL_LOCALITY', 1))
    return result
