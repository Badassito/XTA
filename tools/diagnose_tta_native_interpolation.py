#!/usr/bin/env python3
"""Replay a recorded TTA invocation while capturing one native interpolation input.

This diagnostic changes no numerical settings. It records arrays before parent
cleanup, immediately before interpolation, and immediately after interpolation.
Use separate processes/output directories for predecessor and current sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def _arguments(recorded, source_root, case, subset):
    arguments = list(recorded[recorded.index('--mode'):])
    groups = []
    for value in arguments:
        if value.startswith('--'):
            groups.append([value])
        elif not groups:
            raise ValueError('Malformed recorded command line')
        else:
            groups[-1].append(value)
    result = []
    for group in groups:
        flag = group[0]
        if flag in ('--output', '--temp'):
            group = [flag, str(case / ('outputs' if flag == '--output' else 'runtime'))]
        elif flag == '--augmentation':
            group = [flag, str(source_root / 'XTA/examples/external_augmentations' / Path(group[1]).name)]
        elif subset and flag in ('--enable_tilted', '--enable_spherical', '--enable_radial', '--enable_tile'):
            continue
        elif subset and flag in ('--enable_cartesian', '--enable_azimuthal'):
            group = [flag, 'transverse']
        result.extend(group)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--invocation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--view', default='transverse__tta_a0')
    parser.add_argument('--transverse-azimuthal-only', action='store_true')
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    case = args.output.resolve()
    if case.exists():
        parser.error('Use a fresh --output directory')
    case.mkdir(parents=True)
    captures = case / 'captures'
    captures.mkdir()
    # Reproduce qualify_tta_augmentation's runtime environment, including its
    # four-core worker allocation and disabled optional policy compilation.
    os.environ.update(PYTHONPATH=os.pathsep.join([str(source_root), os.environ.get('PYTHONPATH', '')]),
        PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
        CUPY_CACHE_DIR=str(case / 'cupy-cache'), NUMBA_CACHE_DIR=str(case / 'numba-cache'),
        YOLO_CONFIG_DIR=str(case / 'ultralytics-config'), YOLO_AUTOINSTALL='false',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        SLURM_CPUS_PER_TASK='4', YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND='0',
        YOLO_TTA_TELEMETRY='0', PTA_GPU_TORCH_COMPILE='0')
    sys.path.insert(0, str(source_root))
    import numpy as np
    from XTA import assembly, pipeline
    from XTA.cli import run

    recorded = json.loads(args.invocation.read_text())['argv']
    arguments = _arguments(recorded, source_root, case, args.transverse_azimuthal_only)
    record = dict(source_root=str(source_root), invocation=str(args.invocation), argv=arguments,
        view=args.view, captures={}, complete=False)
    def save_array(name, value):
        array = np.asarray(value)
        np.save(captures / f'{name}.npy', array, allow_pickle=False)
        record['captures'][name] = dict(shape=list(array.shape), dtype=str(array.dtype),
            nonzero=int(np.count_nonzero(array)),
            sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
    def save_metadata(name, values):
        descriptor = {}
        for key, value in dict(values or {}).items():
            if isinstance(value, np.ndarray):
                save_array(f'{name}_{key}', value)
                descriptor[key] = 'array'
            elif value is None or isinstance(value, (str, int, float, bool)):
                descriptor[key] = value
            else:
                descriptor[key] = repr(value)
        (captures / f'{name}.json').write_text(json.dumps(descriptor, indent=2) + '\n')

    prepare = pipeline.prepare_view_volume_after_fullframe
    def capture_prepare(**kwargs):
        if str(kwargs['view'].name) == args.view:
            save_array('before_parent_cleanup', kwargs['union_mm'])
            save_metadata('parent_metadata', kwargs.get('slice_meta'))
            save_metadata('parent_options', {name: value for name, value in kwargs.items()
                if name in ('hole_fill_done_on_device', 'precleaned_slice_cleanup', 'min_conf',
                    'min_radius', 'interpolate', 'interpolation_search_angle', 'slice_workers',
                    'interpolation_task_workers')})
        return prepare(**kwargs)
    pipeline.prepare_view_volume_after_fullframe = capture_prepare

    interpolate = assembly.interpolate_view_volume_pass_maybe_process
    def capture_interpolate(**kwargs):
        active = str(kwargs['view'].name) == args.view
        if active:
            save_array('before_interpolation', kwargs['mask_mm'])
            save_metadata('interpolation_metadata', {name: kwargs.get(name) for name in
                ('known_slice_any', 'known_slice_bboxes', 'max_slice_distance', 'search_angle_deg',
                 'interpolation_walk_back', 'interpolation_candidates', 'interpolate_min_radius', 'workers')})
        result = interpolate(**kwargs)
        if active:
            save_array('after_interpolation', result[0])
            (captures / 'interpolation_stats.json').write_text(
                json.dumps(result[1], default=str, indent=2) + '\n')
        return result
    assembly.interpolate_view_volume_pass_maybe_process = capture_interpolate
    sys.argv = ['diagnostic-tta', *arguments]
    try:
        run()
        record['complete'] = True
    finally:
        (case / 'capture_record.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
