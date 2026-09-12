#!/usr/bin/env python3
"""Run a small real CUDA policy pipeline and decode its independent NRRDs.

Requires a local compatible segmentation model and pynrrd for header parsing.
This checks orchestration and publication using synthetic input, not model accuracy.
All generated inputs, caches, logs and outputs stay beneath --output.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_outputs(output: Path) -> dict:
    import nrrd
    import numpy as np
    from tools.check_tta_augmentation_run import check

    checked = check(output)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'
    manifest = next((output / 'nrrd').glob('*_nrrd_manifest.json'))
    layers = json.loads(manifest.read_text())['layers']
    decoded = {}
    for layer in layers:
        path = manifest.parent / layer['filename']
        header = nrrd.read_header(str(path))
        raw = path.read_bytes()
        boundary = re.search(b'\r?\n\r?\n', raw)
        assert boundary and header['encoding'] == 'gzip' and header['type'] == 'unsigned char'
        # The production writer uses concatenated gzip members; decode every member.
        values = np.frombuffer(gzip.decompress(raw[boundary.end():]), dtype=np.uint8)
        values = values.reshape(tuple(header['sizes']), order='F')
        assert np.all((values == 0) | (values == 1))
        decoded[layer['filename']] = (values.astype(bool), header)
    final_layers = [layer for layer in layers if layer['source'] == 'global' and layer['mask_kind'] == 'union']
    assert len(final_layers) == 1
    final, final_header = decoded[final_layers[0]['filename']]
    inverse_directions = np.linalg.inv(np.asarray(final_header['space directions']))
    origin = np.asarray(final_header['space origin'])
    union, base, augmented = (np.zeros(final.shape, dtype=bool) for _ in range(3))
    groups = defaultdict(dict)
    rows = []
    for layer in layers:
        data, header = decoded[layer['filename']]
        indices = np.column_stack(np.nonzero(data))
        target = (np.asarray(header['space origin']) + indices @ np.asarray(header['space directions']) - origin) @ inverse_directions
        rounded = np.rint(target).astype(np.int64)
        assert np.allclose(rounded, target, atol=1e-6)
        assert np.all(rounded >= 0) and np.all(rounded < np.asarray(final.shape))
        native = np.zeros(final.shape, dtype=bool)
        native[tuple(rounded.T)] = True
        match = re.search(r'__policy_(\d+)$', layer['view_name'])
        policy_pass = int(match.group(1)) if match else 0
        if layer['recomposition_op'] == 'union':
            assert not np.any(native & ~final), layer['filename']
            union |= native
            if policy_pass:
                augmented |= native
            else:
                base |= native
        if layer['source'] == 'fullframe' and layer['mask_kind'] == 'yolo':
            name = layer['view_name'][:match.start()] if match else layer['view_name']
            groups[name][str(policy_pass)] = int(native.sum())
        if layer['mask_kind'] == 'bridge':
            assert policy_pass == 0
        rows.append({'filename': layer['filename'], 'source': layer['source'],
                     'mask_kind': layer['mask_kind'], 'policy_pass': policy_pass,
                     'foreground': int(native.sum())})
    assert np.array_equal(union, final), 'Saved component OR differs from the final union'
    return {'checker': checked, 'final_shape_x_y_t': list(final.shape),
            'final_foreground': int(final.sum()), 'component_union_equals_final': True,
            'augmented_voxels_outside_base': int(np.count_nonzero(augmented & ~base)),
            'fullframe_pass_foreground': dict(groups), 'layers': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--channel_format', default='gray')
    parser.add_argument('--imgsz', type=int, default=128)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--augmentation_ratio', type=int, default=3)
    parser.add_argument('--augmentation_granularity', choices=('slice', 'slab', 'lease', 'view'), default='slice')
    parser.add_argument('--projection_sampling', choices=('coverage', 'dense'), default='coverage')
    parser.add_argument('--enable_spherical', nargs='+', default=[])
    parser.add_argument('--enable_radial', nargs='+', default=[])
    parser.add_argument('--enable_azimuthal', nargs='+', default=['transverse:30'])
    parser.add_argument('--no-tiles', action='store_true')
    args = parser.parse_args()
    from XTA.config import resolve_channel_format
    layout = resolve_channel_format(args.channel_format)
    if args.device < 0 or args.imgsz < 32 or args.batch < 1 or args.augmentation_ratio < 2:
        parser.error('Require device >= 0, imgsz >= 32, batch >= 1, augmentation_ratio >= 2')
    if not args.model.is_file():
        parser.error('--model must be an existing local model')
    case = args.output.resolve()
    case.mkdir(parents=True, exist_ok=True)
    output = case / 'outputs'
    if output.exists():
        parser.error('Use a fresh --output directory; an outputs subdirectory already exists')
    import cv2
    import numpy as np
    video = case / 'synthetic.mkv'
    yy, xx = np.mgrid[:64, :64]
    rng = np.random.default_rng(20260912)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 12.0, (64, 64), False)
    if not writer.isOpened():
        raise RuntimeError('FFV1 fixture writer unavailable')
    try:
        for index in range(16):
            plane = 15 + rng.normal(0, 3, (64, 64))
            plane += 205 * np.exp(-((xx - (24 + 7 * np.sin(index * .27))) ** 2 + (yy - (30 + 5 * np.cos(index * .31))) ** 2) / 90)
            plane += 100 * np.exp(-((xx - (46 - index * .3)) ** 2 + (yy - 45) ** 2) / 35)
            plane[30:34, 8:57] += 25
            writer.write(np.uint8(np.clip(plane, 0, 255)))
    finally:
        writer.release()
    argv = [sys.executable, '-B', '-u', '-m', 'XTA', '--mode', 'tta',
            '--input', str(video), '--model', 'gpu:' + str(args.model.resolve()),
            '--output', str(output), '--temp', str(case / 'runtime'), '--device', str(args.device),
            '--channel_format', layout.token, '--imgsz', str(args.imgsz), '--batch', str(args.batch),
            '--quantize', 'gpu:fp32', '--conf', '0.00001', '--min_conf', '0', '--min_radius', '0',
            '--angle', '0', '--enable_cartesian', 'transverse',
            '--projection_sampling', args.projection_sampling,
            '--interpolation_distance', '2', '--interpolation_walk_back', '1', '--interpolation_candidates', '1',
            '--interpolation_passes', '1', '--interpolation_min_radius', '0',
            '--augmentation', str(ROOT / 'XTA/examples/external_augmentations/GPU_light.py'),
            '--augmentation_ratio', str(args.augmentation_ratio),
            '--augmentation_granularity', args.augmentation_granularity,
            '--augmentation_coverage', 'packed', '--save', 'nrrd', 'summary', 'voxel_volume']
    argv.extend(['--enable_azimuthal', *args.enable_azimuthal])
    if args.enable_spherical:
        argv.extend(['--enable_spherical', *args.enable_spherical])
    if args.enable_radial:
        argv.extend(['--enable_radial', *args.enable_radial])
    if not args.no_tiles:
        argv.extend(['--enable_tile', '32:32'])
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT), env.get('PYTHONPATH', '')]),
               PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
               CUPY_CACHE_DIR=str(case / 'cupy-cache'), NUMBA_CACHE_DIR=str(case / 'numba-cache'),
               YOLO_CONFIG_DIR=str(case / 'ultralytics-config'), YOLO_AUTOINSTALL='false',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               SLURM_CPUS_PER_TASK='4', YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND='0',
               YOLO_TTA_TELEMETRY='0', PTA_GPU_TORCH_COMPILE='0')
    (case / 'invocation.json').write_text(json.dumps({'argv': argv}, indent=2))
    with (case / 'pipeline.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f'Pipeline failed; see {case / "pipeline.log"}', flush=True)
        return result.returncode
    verification = verify_outputs(output)
    (case / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps({key: verification[key] for key in ('checker', 'component_union_equals_final', 'augmented_voxels_outside_base')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
