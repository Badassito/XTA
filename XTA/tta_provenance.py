"""Source and requested-runtime provenance for TTA execution receipts."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

from .cylindrical_owner import radial_runtime_provenance
from .file_hashing import sha256_file
from .lta_telemetry import _SOURCE_FINGERPRINT_SCOPE, package_source_digest
from .workspace import _env_flag, _env_int


def _source_record(package: Path, name: str) -> Dict[str, object]:
    """Fingerprint without importing optional runtimes; retain read failures."""
    path = package.joinpath(*name.split('.')).with_suffix('.py')
    module = 'XTA' if name == '__init__' else 'XTA.' + name.removesuffix('.__init__')
    entry = {'path': str(path), 'loaded_in_parent': module in sys.modules}
    try:
        entry['sha256'] = sha256_file(path)
    except OSError as exc:
        entry.update(sha256=None, read_error=type(exc).__name__)
    return entry


def _execution_runtime_provenance() -> Dict[str, object]:
    """Fingerprint every package Python source in the run receipt, without importing it.

    A package version can describe several development snapshots, so each file's
    actual bytes are recorded, plus the aggregate in the shared package source scope.
    """
    from .geometry_quality import geometry_quality_request_record
    from .cuda_backend import native_trt_ring_enabled, native_trt_ring_mode

    result = radial_runtime_provenance()
    package = Path(__file__).resolve().parent
    files = {}
    for path in sorted(package.rglob('*.py')):
        if path.is_file():
            relative = path.relative_to(package).as_posix()
            name = relative[:-3].replace('/', '.')
            result['modules'][name] = entry = _source_record(package, name)
            files[relative] = entry['sha256']
    result['package_python_sources'] = dict(scope=_SOURCE_FINGERPRINT_SCOPE,
        sha256=package_source_digest(files) if all(isinstance(value, str) for value in files.values()) else None,
        file_count=len(files))
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
