"""Exercise GPU and hybrid policy TTA with tiny real TensorRT/OpenVINO models."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixtures(case: Path):
    import cv2
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import openvino as ov
    import torch
    import tensorrt as trt
    size = 64
    head = np.zeros((1, 6, 8), np.float32)
    head[:, :4, 0] = [size / 2, size / 2, size, size]
    head[:, 4:, 0] = [.9, 1.]
    graph = helper.make_graph([
        helper.make_node('Constant', [], ['output0'], value=numpy_helper.from_array(head)),
        helper.make_node('Sub', ['images', 'offset'], ['output1']),
    ], 'policy_fixture', [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 1, size, size])],
        [helper.make_tensor_value_info('output0', TensorProto.FLOAT, list(head.shape)),
         helper.make_tensor_value_info('output1', TensorProto.FLOAT, [1, 1, size, size])],
        [numpy_helper.from_array(np.asarray(.35, np.float32), name='offset')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx_path = case / 'model.onnx'
    onnx.save(model, onnx_path)
    ov.save_model(ov.convert_model(onnx_path), case / 'model.xml')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024**2)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('Could not build TensorRT fixture')
    metadata = json.dumps({'task': 'segment', 'batch': 1, 'imgsz': [size, size],
                           'stride': 32, 'names': {0: 'object'}, 'channels': 1}).encode()
    (case / 'model.engine').write_bytes(len(metadata).to_bytes(4, 'little', signed=True) + metadata + bytes(engine))
    video = case / 'input.mkv'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 8., (size, size), False)
    if not writer.isOpened():
        raise RuntimeError('FFV1 fixture writer unavailable')
    try:
        yy, xx = np.mgrid[:size, :size]
        for index in range(16):
            image = np.zeros((size, size), np.uint8)
            image[(xx - 30 - index % 4)**2 + (yy - 29)**2 < 220] = 220
            image[28:34, 29:35] = 0
            writer.write(image)
    finally:
        writer.release()
    (case / 'runtime_versions.json').write_text(json.dumps({'python': sys.executable,
        'tensorrt': trt.__version__, 'torch': torch.__version__, 'openvino': ov.__version__,
        'gpu': torch.cuda.get_device_name(0)}, indent=2))


def run_case(case: Path, hybrid: bool):
    mode = 'hybrid' if hybrid else 'gpu'
    output = case / mode
    argv = [sys.executable, '-B', '-u', '-m', 'XTA', '--mode', 'tta', '--input', str(case / 'input.mkv'),
        '--model', f'gpu:{case / "model.engine"}', *([f'cpu:{case / "model.xml"}'] if hybrid else []),
        '--device', '0:cpu' if hybrid else '0', '--quantize', 'gpu:fp32', *(['cpu:fp32'] if hybrid else []),
        '--batch', 'gpu:1', *(['cpu:1'] if hybrid else []), '--channel_format', 'gray', '--imgsz', '64',
        '--cpu_threads', '2', '--cpu_streams', '1', '--cpu_infer_requests', '2',
        '--enable_cartesian', 'transverse', 'sagittal', 'coronal', '--enable_tile', '32:32',
        '--angle', '0', '--conf', '.1', '--min_conf', '0', '--min_radius', '0',
        '--interpolation_distance', '0', '--augmentation_ratio', '3',
        '--augmentation', f'gpu:{ROOT / "XTA/examples/external_augmentations/GPU_light.py"}',
        *([f'cpu:{ROOT / "XTA/examples/external_augmentations/CPU_light.py"}'] if hybrid else []),
        '--augmentation_coverage', 'packed', '--output', str(output), '--temp', str(case / f'{mode}_runtime'),
        '--save', 'nrrd', 'summary']
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', SLURM_CPUS_PER_TASK='8',
        YOLO_TTA_CPU_SOCKET_RESERVE_CORES='0', YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND='0',
        YOLO_TTA_TELEMETRY='0', YOLO_CONFIG_DIR=str(case / 'ultralytics-config'),
        NUMBA_CACHE_DIR=str(case / 'numba-cache'), CUPY_CACHE_DIR=str(case / 'cupy-cache'),
        TORCHINDUCTOR_CACHE_DIR=str(case / 'torchinductor-cache'), TRITON_CACHE_DIR=str(case / 'triton-cache'),
        NO_ALBUMENTATIONS_UPDATE='1', PTA_GPU_TORCH_COMPILE='0', YOLO_AUTOINSTALL='false')
    (case / f'{mode}_invocation.json').write_text(json.dumps(argv, indent=2))
    with (case / f'{mode}.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{mode} CLI failed: see {case / (mode + ".log")}')
    from tools.qualify_tta_augmentation import verify_outputs
    from tools.check_tta_augmentation_run import check
    verification = verify_outputs(output)
    receipt = check(output)
    manifest = json.loads((output / 'augmentation_manifest.json').read_text())
    backends = sorted({row['backend'] for row in manifest['execution_records']})
    expected = ['cpu', 'gpu'] if hybrid else ['gpu']
    if backends != expected:
        raise RuntimeError(f'{mode} execution used {backends}, expected {expected}')
    assert verification['final_foreground'] > 0
    data = {'verification': verification, 'receipt': receipt, 'executed_backends': backends}
    (case / f'{mode}_verification.json').write_text(json.dumps(data, indent=2))
    print(json.dumps({mode: data}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-lock', required=True, type=Path)
    args = parser.parse_args()
    case = args.output.resolve()
    if case.is_relative_to(ROOT) or case.exists():
        parser.error('Use a fresh output directory in Scratch')
    case.mkdir(parents=True)
    lock = args.gpu_lock.resolve()
    while True:
        try:
            with lock.open('x') as stream:
                stream.write(json.dumps({'task': 'v22.2.0 TensorRT and hybrid qualification', 'pid': os.getpid(),
                    'start_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}))
            break
        except FileExistsError:
            print('Waiting for GPU_LOCK', flush=True)
            time.sleep(5)
    try:
        fixtures(case)
        print('Real TensorRT and OpenVINO fixtures built.', flush=True)
        run_case(case, hybrid=False)
        run_case(case, hybrid=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
