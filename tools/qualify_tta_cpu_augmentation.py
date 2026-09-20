"""Run a complete CPU TTA policy pipeline with a tiny OpenVINO segmentation model.

All generated model, video, logs and output evidence go under --output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--tiles', action='store_true')
    args = parser.parse_args()
    case = args.output.resolve()
    case.mkdir(parents=True, exist_ok=True)
    if (case / 'outputs').exists():
        parser.error('Use a fresh --output directory')
    import cv2
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops

    image = ops.parameter([2, 1, 32, 32], np.float32, name='images')
    proto = ops.subtract(image, ops.constant(np.float32(.35)))
    head = np.zeros((2, 6, 8), np.float32)
    head[:, :4, 0] = [16., 16., 32., 32.]
    head[:, 4:, 0] = [.9, 1.]
    model = ov.Model([ops.constant(head), proto], [image], 'tta_cpu_policy_fixture')
    model_path = case / 'model.xml'
    ov.save_model(model, model_path)
    video = case / 'input.mkv'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 8., (32, 32), False)
    if not writer.isOpened():
        raise RuntimeError('FFV1 fixture writer unavailable')
    try:
        yy, xx = np.mgrid[:32, :32]
        for index in range(8):
            frame = np.zeros((32, 32), np.uint8)
            frame[(xx - 15 - index % 3)**2 + (yy - 15)**2 < 70] = 220
            writer.write(frame)
    finally:
        writer.release()
    argv = [sys.executable, '-B', '-u', '-m', 'XTA', '--mode', 'tta', '--input', str(video),
            '--model', f'cpu:{model_path}', '--device', 'cpu', '--quantize', 'cpu:fp32',
            '--batch', 'cpu:2', '--channel_format', 'gray', '--imgsz', '32', '--cpu_threads', '4',
            '--cpu_streams', '1', '--cpu_infer_requests', '2', '--enable_cartesian', 'transverse',
            '--angle', '0', '--conf', '.1', '--min_conf', '0', '--min_radius', '0',
            '--interpolation_distance', '0', '--augmentation_ratio', '3',
            '--augmentation', f'cpu:{ROOT / "XTA/examples/external_augmentations/CPU_light.py"}',
            '--augmentation_coverage', 'packed', '--output', str(case / 'outputs'),
            '--temp', str(case / 'runtime'), '--save', 'nrrd', 'summary']
    if args.tiles:
        argv.extend(['--enable_tile', '16:16'])
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               SLURM_CPUS_PER_TASK='4', YOLO_TTA_CPU_SOCKET_RESERVE_CORES='0',
               YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND='0', YOLO_TTA_TELEMETRY='0',
               YOLO_CONFIG_DIR=str(case / 'ultralytics-config'),
               NUMBA_CACHE_DIR=str(case / 'numba-cache'), NO_ALBUMENTATIONS_UPDATE='1')
    (case / 'invocation.json').write_text(json.dumps({'argv': argv}, indent=2))
    with (case / 'pipeline.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        print(f'CPU pipeline failed; see {case / "pipeline.log"}')
        return completed.returncode
    from tools.qualify_tta_augmentation import verify_outputs
    verification = verify_outputs(case / 'outputs')
    manifest = json.loads((case / 'outputs/augmentation_manifest.json').read_text())
    assert manifest['policies']['cpu'] and not manifest['policies'].get('gpu')
    assert all(record['backend'] == 'cpu' and record['pass_count'] == 3
               for record in manifest['execution_records'])
    assert verification['final_foreground'] > 0
    (case / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
