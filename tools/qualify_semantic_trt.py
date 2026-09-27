"""Qualify semantic TensorRT accumulation with the same engine and frames in three modes.

Modes are legacy CPU cleanup, GPU cleanup, and GPU cleanup plus the TensorRT
ring. Direct-source timings include preprocess, inference, semantic decoding,
warp, and union. Raw AutoBackend forward timings are a separate calibration.
Optional full TTA runs measure pipeline wall time and compare published PNGs.
All evidence is written below sibling Scratch/Experiments/v24-semantic-ring.
The local 4090 mobile eGPU verifies CUDA behavior; its throughput is not an
estimate of H100 performance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRATCH = ROOT.parent / 'Scratch'
DEFAULT_INPUT = SCRATCH / 'Experiments' / 'v24-semantic' / 'input.mkv'
DEFAULT_OUTPUT_PARENT = SCRATCH / 'Experiments' / 'v24-semantic-ring'
LOCK = SCRATCH / 'Temp' / 'GPU_LOCK'
MODES = (
    ('legacy', '0', '0'),
    ('gpu_cleanup', '1', '0'),
    ('ring', '1', '1'),
)

MASK_FRACTION_TOLERANCE = 1e-4
CONFIDENCE_U8_TOLERANCE = 1


def _plane_difference(
    reference_mask: object, candidate_mask: object,
    reference_confidence: object | None = None, candidate_confidence: object | None = None,
    *, confidence_threshold: float | None = None,
) -> dict[str, object]:
    """Describe exact and bounded differences without suppressing spatial evidence."""
    import numpy as np

    before = np.asarray(reference_mask)
    after = np.asarray(candidate_mask)
    if before.shape != after.shape:
        return {'shape_match': False, 'reference_shape': list(before.shape),
                'candidate_shape': list(after.shape), 'within_tolerance': False}
    mismatch = before != after
    mismatch_count = int(np.count_nonzero(mismatch))
    mismatch_fraction = mismatch_count / max(1, int(before.size))
    coordinates = np.argwhere(mismatch)
    bbox = (
        [[int(value) for value in coordinates.min(axis=0)],
         [int(value) for value in coordinates.max(axis=0)]]
        if mismatch_count else None
    )
    result: dict[str, object] = {
        'shape_match': True,
        'mask_mismatch_pixels': mismatch_count,
        'mask_mismatch_fraction': mismatch_fraction,
        'mask_disagreement_bbox_inclusive': bbox,
        'mask_exact': mismatch_count == 0,
    }
    confidence_ok = True
    if reference_confidence is not None and candidate_confidence is not None:
        conf_before = np.asarray(reference_confidence)
        conf_after = np.asarray(candidate_confidence)
        if conf_before.shape != conf_after.shape or conf_before.shape != before.shape:
            result.update(confidence_shape_match=False,
                          reference_confidence_shape=list(conf_before.shape),
                          candidate_confidence_shape=list(conf_after.shape))
            confidence_ok = False
        else:
            absolute = np.abs(conf_before.astype(np.int16) - conf_after.astype(np.int16))
            max_absolute = int(absolute.max()) if absolute.size else 0
            result.update(
                confidence_shape_match=True,
                confidence_mismatch_pixels=int(np.count_nonzero(absolute)),
                confidence_max_abs_u8=max_absolute,
                confidence_exact=max_absolute == 0,
            )
            confidence_ok = max_absolute <= CONFIDENCE_U8_TOLERANCE
            if confidence_threshold is not None and mismatch_count:
                threshold_u8 = int(np.rint(float(confidence_threshold) * 255.0))
                near = ((np.abs(conf_before.astype(np.int16) - threshold_u8) <= 1)
                        | (np.abs(conf_after.astype(np.int16) - threshold_u8) <= 1))
                result['mask_disagreement_near_conf_threshold_pixels'] = int(
                    np.count_nonzero(mismatch & near)
                )
    result['within_tolerance'] = bool(
        mismatch_fraction <= MASK_FRACTION_TOLERANCE and confidence_ok
    )
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    staged.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(staged, path)


def _case_spec(raw: str) -> dict[str, object]:
    try:
        size_text, batch_text, path_text = str(raw).split(':', 2)
        size, batch = int(size_text), int(batch_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('Use SIZE:BATCH:PATH for --engine') from exc
    path = Path(path_text).expanduser().resolve()
    if size < 32 or batch not in (1, 2) or not path.is_file() or path.suffix.lower() != '.engine':
        raise argparse.ArgumentTypeError(
            'Each --engine needs SIZE>=32, BATCH=1 or 2, and an existing .engine file'
        )
    return {'size': size, 'batch': batch, 'engine': str(path)}


def _claim_gpu_lock(wait_seconds: float) -> dict[str, object]:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    claim = {
        'task': 'v24 semantic TensorRT qualification',
        'pid': os.getpid(),
        'start_time': datetime.now(timezone.utc).isoformat(),
        'token': uuid.uuid4().hex,
    }
    deadline = time.monotonic() + float(wait_seconds)
    while True:
        try:
            with LOCK.open('x', encoding='utf-8') as handle:
                json.dump(claim, handle)
            return claim
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f'GPU_LOCK is held; inspect {LOCK}')
            print(f'Waiting for {LOCK}', flush=True)
            time.sleep(min(15.0, max(0.1, deadline - time.monotonic())))


def _release_gpu_lock(claim: dict[str, object]) -> None:
    try:
        found = json.loads(LOCK.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return
    if found.get('token') == claim.get('token') and found.get('pid') == os.getpid():
        LOCK.unlink(missing_ok=True)


class _GpuMemorySampler:
    """Sample whole-device VRAM, including allocations outside PyTorch's cache."""

    def __init__(self, device: int) -> None:
        self.device = int(device)
        self.peak_used_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        except Exception:
            self._nvml = None

    def _sample(self) -> None:
        if self._nvml is None:
            return
        while not self._stop.is_set():
            try:
                used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                self.peak_used_bytes = max(int(self.peak_used_bytes or 0), used)
            except Exception:
                break
            self._stop.wait(0.02)

    def __enter__(self) -> '_GpuMemorySampler':
        if self._nvml is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def _heatsoak(torch: object, device: int, seconds: float, cpu_workers: int) -> dict[str, object]:
    import cv2  # type: ignore
    import numpy as np

    if seconds <= 0:
        return {'gpu_seconds': 0.0, 'cpu_seconds': 0.0, 'cpu_workers': 0,
                'cpu_connected_component_calls': 0}
    index = int(device)
    cv2.setNumThreads(1)
    y, x = np.mgrid[:512, :512]
    seeded = np.random.default_rng(2400)
    component_mask = np.uint8(
        (((x - 180) ** 2 + (y - 210) ** 2) < 115 ** 2)
        | (((x - 355) ** 2 + (y - 335) ** 2) < 62 ** 2)
        | (seeded.random((512, 512)) < 0.012)
    )
    stop = threading.Event()
    cpu_counts = [0] * int(cpu_workers)
    cpu_errors: list[BaseException] = []

    def cpu_heat(worker_index: int) -> None:
        try:
            local_mask = np.roll(component_mask, worker_index * 7, axis=0)
            while not stop.is_set():
                cv2.connectedComponentsWithStats(local_mask, connectivity=8)
                cpu_counts[worker_index] += 1
        except BaseException as exc:
            cpu_errors.append(exc)
            stop.set()

    cpu_started = time.perf_counter()
    workers = [
        threading.Thread(target=cpu_heat, args=(worker_index,), daemon=True)
        for worker_index in range(int(cpu_workers))
    ]
    for worker in workers:
        worker.start()
    side = 1024
    gpu_started = time.perf_counter()
    iterations = 0
    next_report = 10.0
    try:
        a = torch.randn((side, side), device=f'cuda:{index}', dtype=torch.float16)
        b = torch.randn((side, side), device=f'cuda:{index}', dtype=torch.float16)
        while time.perf_counter() - gpu_started < float(seconds):
            a = torch.matmul(a, b).clamp_(-2.0, 2.0)
            iterations += 1
            if iterations % 16 == 0:
                torch.cuda.synchronize(index)
            elapsed = time.perf_counter() - gpu_started
            if elapsed >= next_report:
                print(f'GPU+CPU heatsoak {elapsed:.0f}/{seconds:.0f}s, '
                      f'{iterations} matmuls, {sum(cpu_counts)} CC calls', flush=True)
                next_report += 10.0
        torch.cuda.synchronize(index)
        gpu_seconds = float(time.perf_counter() - gpu_started)
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=10.0)
    if cpu_errors:
        raise RuntimeError(f'CPU connected-component heatsoak failed: {cpu_errors[0]}')
    if any(worker.is_alive() for worker in workers):
        raise RuntimeError('CPU connected-component heatsoak worker did not stop')
    return {
        'gpu_seconds': gpu_seconds, 'cpu_seconds': float(time.perf_counter() - cpu_started),
        'cpu_workers': int(cpu_workers),
        'cpu_connected_component_calls': int(sum(cpu_counts)),
        'gpu_matmul_calls': iterations,
        'cpu_task': '512x512 OpenCV 8-connected components, cv2 internal threads=1',
    }


def _frames(case: dict[str, object]) -> object:
    import cv2  # type: ignore
    import numpy as np

    size, count = int(case['size']), int(case['frames'])
    input_path = case.get('input')
    collected: list[object] = []
    if input_path:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f'Unable to open representative input {input_path}')
        try:
            while len(collected) < count:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                collected.append(cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR))
        finally:
            capture.release()
        if not collected:
            raise RuntimeError(f'No frames decoded from {input_path}')
        original_count = len(collected)
        while len(collected) < count:
            collected.append(np.asarray(collected[len(collected) % original_count]).copy())
        return np.ascontiguousarray(np.stack(collected[:count]), dtype=np.uint8)

    yy, xx = np.mgrid[:size, :size]
    rng = np.random.default_rng(2400 + size)
    for index in range(count):
        center_x = size * (0.38 + 0.13 * math.sin(index * 0.21))
        center_y = size * (0.52 + 0.11 * math.cos(index * 0.19))
        radius = max(3.0, size * 0.11)
        glow = 215.0 * np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * radius * radius))
        texture = 35.0 * np.sin(xx * 0.07) * np.cos(yy * 0.06)
        plane = 24.0 + glow + texture + rng.normal(0, 4, (size, size))
        collected.append(np.uint8(np.clip(plane, 0, 255)))
    return np.ascontiguousarray(np.stack(collected))


def _run_direct_child(case: dict[str, object]) -> dict[str, object]:
    import numpy as np
    import torch  # type: ignore
    from XTA.geometry import InMemoryYoloVolumeSource
    from XTA.inference import (
        PredictConfig, _ensure_predictor_for_direct_predict,
        load_ultralytics_model, predict_source_and_accumulate,
        set_retina_mask_processor,
    )

    size, batch, count = int(case['size']), int(case['batch']), int(case['frames'])
    device = int(case['device'])
    torch.cuda.set_device(device)
    set_retina_mask_processor('gpu')
    import cv2  # type: ignore
    cv2.setNumThreads(1)
    frames = _frames(case)
    cfg = PredictConfig(
        imgsz=size, conf=float(case['conf']), device=f'cuda:{device}',
        quantize=str(case['quantize']), batch=batch, input_channels=1,
        channel_token='gray', task='semantic',
    )
    model = load_ultralytics_model(str(case['engine']), task='semantic')
    affine = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    def one_run() -> tuple[dict[str, object], object, object, float, int, int, int | None]:
        union = np.zeros((count, size, size), dtype=np.uint8)
        confidence = np.zeros_like(union)
        source = InMemoryYoloVolumeSource(frames, 'semantic-qualification', batch_size=batch)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with _GpuMemorySampler(device) as memory:
            started = time.perf_counter()
            stats = predict_source_and_accumulate(
                model, source, source_label='semantic-trt-qualification',
                num_frames=count, out_size=size, cfg=cfg, view_union_mm=union,
                view_confmap_mm=confidence, M_out_to_native=affine,
                native_h=size, native_w=size,
                postprocess_workers=int(case['postprocess_workers']),
                streaming_cleanup_enabled=bool(float(case['min_conf']) > 0),
                streaming_cleanup_min_conf=float(case['min_conf']),
                streaming_cleanup_min_radius=0.0, retain_confidence=True,
            )
            torch.cuda.synchronize(device)
            seconds = float(time.perf_counter() - started)
        return (
            stats, union, confidence, seconds,
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
            memory.peak_used_bytes,
        )

    first = one_run()
    if case['mode'] == 'ring' and int(first[0].get('semantic_trt_ring_used', 0)) != 1:
        raise RuntimeError('Semantic TensorRT ring silently declined the requested engine/source')
    for _ in range(max(0, int(case['warmups']) - 1)):
        warmed = one_run()
        if case['mode'] == 'ring' and int(warmed[0].get('semantic_trt_ring_used', 0)) != 1:
            raise RuntimeError('Semantic TensorRT ring declined during warmup')

    rows: list[dict[str, object]] = []
    repeatability: list[dict[str, object]] = []
    reference_mask = reference_conf = None
    reference_counts = None
    for _ in range(int(case['repeats'])):
        stats, union, confidence, seconds, allocated, reserved, device_used = one_run()
        if case['mode'] == 'ring' and int(stats.get('semantic_trt_ring_used', 0)) != 1:
            raise RuntimeError('Semantic TensorRT ring declined during measurement')
        counts = (int(stats.get('prediction_count', -1)), int(stats.get('frames_with_predictions', -1)))
        if reference_mask is None:
            reference_mask, reference_conf, reference_counts = union, confidence, counts
        else:
            comparison = _plane_difference(
                reference_mask, union, reference_conf, confidence,
                confidence_threshold=float(case['conf']),
            )
            comparison['count_difference'] = [
                counts[0] - reference_counts[0], counts[1] - reference_counts[1],
            ]
            repeatability.append(comparison)
        rows.append({
            'seconds': seconds, 'frames_per_second': count / seconds,
            'torch_peak_allocated_bytes': allocated, 'torch_peak_reserved_bytes': reserved,
            'device_peak_used_bytes': device_used,
            'prediction_count': counts[0], 'frames_with_predictions': counts[1],
            'semantic_trt_ring_used': int(stats.get('semantic_trt_ring_used', 0)),
            'semantic_trt_ring_batches': int(stats.get('semantic_trt_ring_batches', 0)),
            'semantic_trt_ring_frames': int(stats.get('semantic_trt_ring_frames', 0)),
            'semantic_trt_ring_infer_graphs': int(stats.get('semantic_trt_ring_infer_graphs', 0)),
        })

    output = Path(str(case['output']))
    np.savez_compressed(output / 'direct-planes.npz', mask=reference_mask, confidence=reference_conf)
    # Forward-only calibration. This is AutoBackend.forward on an already-preprocessed
    # batch; it is not claimed as a separately measured ring stage.
    predictor = _ensure_predictor_for_direct_predict(model, cfg)
    if predictor is None:
        raise RuntimeError('Unable to initialize AutoBackend for raw forward calibration')
    source = InMemoryYoloVolumeSource(frames, 'semantic-forward-calibration', batch_size=batch)
    _paths, images, _info = next(iter(source))
    model_input = predictor.preprocess(images)
    backend = predictor.model
    for _ in range(4):
        _ = backend(model_input)
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(int(case['backend_iterations'])):
        _ = backend(model_input)
    end_event.record()
    end_event.synchronize()
    forward_seconds = float(start_event.elapsed_time(end_event)) / 1000.0

    result = {
        'case': {key: case[key] for key in (
            'size', 'batch', 'engine', 'mode', 'frames', 'conf', 'min_conf',
            'postprocess_workers',
        )},
        'first_call_seconds': first[3],
        'first_call_stats': {key: value for key, value in first[0].items() if isinstance(value, (int, float, str, bool))},
        'timed': rows,
        'repeatability': repeatability,
        'repeatability_within_tolerance': all(
            item['within_tolerance'] for item in repeatability
        ),
        'median_seconds': statistics.median(float(row['seconds']) for row in rows),
        'median_frames_per_second': statistics.median(float(row['frames_per_second']) for row in rows),
        'raw_autobackend_forward_seconds_per_batch': forward_seconds / int(case['backend_iterations']),
        'raw_autobackend_forward_scope': 'Already-preprocessed fixed batch; calibration, not an isolated ring stage',
        'foreground_voxels': int(np.count_nonzero(reference_mask)),
        'nonzero_confidence_voxels': int(np.count_nonzero(reference_conf)),
        'mask_unique_values': [int(value) for value in np.unique(reference_mask)],
        'confidence_unique_count': int(len(np.unique(reference_conf))),
    }
    _write_json(output / 'direct-result.json', result)
    return result


def _controlled_logit_qualification(torch: object, device: int) -> dict[str, object]:
    """Exercise foreground-rich one/two-channel logits independently of model quality."""
    import numpy as np
    from XTA.semantic_cuda import semantic_native_cuda, semantic_native_reference

    height = width = 96
    yy, xx = np.mgrid[:height, :width]
    strong = (xx - 29) ** 2 + (yy - 43) ** 2 <= 15 ** 2
    weak = (xx - 72) ** 2 + (yy - 44) ** 2 <= 12 ** 2
    base = np.full((height, width), -7.0, dtype=np.float32)
    base[strong] = 7.0
    base[weak] = -0.2
    affine = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    cases: list[dict[str, object]] = []
    for channels in (1, 2):
        logits = (base[None] if channels == 1 else
                  np.stack((np.zeros_like(base), base), axis=0))
        for threshold in (0, 200):
            print(f'Controlled logits C={channels}, min_conf_u8={threshold}', flush=True)
            common = dict(
                output_size=width, M_out_to_native=affine,
                native_h=height, native_w=width,
                conf_threshold=0.25, min_conf_u8=threshold,
            )
            reference_mask, reference_conf = semantic_native_reference(logits, **common)
            logits_gpu = torch.from_numpy(np.ascontiguousarray(logits)).to(f'cuda:{int(device)}')
            mask_gpu, conf_gpu = semantic_native_cuda(logits_gpu, **common)
            torch.cuda.synchronize(int(device))
            mask = mask_gpu.cpu().numpy()
            confidence = conf_gpu.cpu().numpy()
            difference = _plane_difference(
                reference_mask, mask, reference_conf, confidence,
                confidence_threshold=0.25,
            )
            foreground = int(np.count_nonzero(mask))
            if foreground <= 0:
                raise RuntimeError('Controlled semantic logits did not exercise foreground decoding')
            cases.append({
                'channels': channels, 'min_conf_u8': threshold,
                'foreground_voxels': foreground,
                'confidence_values': int(len(np.unique(confidence))),
                'difference': difference,
            })
    if not all(
        next(row['foreground_voxels'] for row in cases
             if row['channels'] == channels and row['min_conf_u8'] == 200)
        < next(row['foreground_voxels'] for row in cases
               if row['channels'] == channels and row['min_conf_u8'] == 0)
        for channels in (1, 2)
    ):
        raise RuntimeError('Controlled semantic component threshold did not remove the weak island')
    return {
        'all_exact': all(row['difference']['mask_exact'] and row['difference']['confidence_exact']
                         for row in cases),
        'all_within_tolerance': all(row['difference']['within_tolerance'] for row in cases),
        'cases': cases,
    }


def _child_environment(output: Path, mode: tuple[str, str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8',
        'PYTHONPATH': os.pathsep.join((str(ROOT), environment.get('PYTHONPATH', ''))),
        'YOLO_AUTOINSTALL': 'false', 'YOLO_CONFIG_DIR': str(output / 'ultralytics-config'),
        'YOLO_TTA_SEMANTIC_GPU_CLEANUP': mode[1],
        'YOLO_TTA_SEMANTIC_TRT_RING': mode[2],
        'YOLO_TTA_SEMANTIC_TRT_TRACE': '1' if mode[0] == 'ring' else '0',
        'CUPY_CACHE_DIR': str(output / 'cupy-cache'),
        'NUMBA_CACHE_DIR': str(output / 'numba-cache'),
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        'SLURM_CPUS_PER_TASK': '4', 'YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND': '0',
    })
    return environment


def _run_command(command: list[str], *, output: Path, log_name: str,
                 environment: dict[str, str], device: int) -> tuple[float, int | None]:
    output.mkdir(parents=True, exist_ok=True)
    with _GpuMemorySampler(device) as memory:
        started = time.perf_counter()
        with (output / log_name).open('w', encoding='utf-8') as handle:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=handle, stderr=subprocess.STDOUT)
        seconds = float(time.perf_counter() - started)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed ({result.returncode}); see {output / log_name}')
    return seconds, memory.peak_used_bytes


def _pipeline_case(case: dict[str, object], mode: tuple[str, str, str], output: Path,
                   environment: dict[str, str]) -> dict[str, object]:
    command = [
        sys.executable, '-B', '-u', '-m', 'XTA', '--mode', 'tta', '--task', 'semantic',
        '--input', str(case['input']), '--model', 'gpu:' + str(case['engine']),
        '--device', str(case['device']), '--output', str(output / 'outputs'),
        '--temp', str(output / 'runtime'), '--imgsz', str(case['size']),
        '--batch', f'gpu:{case["batch"]}', '--quantize', 'gpu:' + str(case['quantize']),
        '--channel_format', 'gray', '--angle', '0', '--conf', str(case['conf']),
        '--min_conf', str(case['min_conf']), '--min_radius', '0',
        '--interpolation_distance', '0', '--enable_cartesian', 'transverse',
        '--save', 'semantic,summary',
    ]
    _write_json(output / 'pipeline-invocation.json', {'argv': command, 'mode': mode[0]})
    seconds, peak = _run_command(command, output=output, log_name='pipeline.log',
                                 environment=environment, device=int(case['device']))
    log = (output / 'pipeline.log').read_text(encoding='utf-8', errors='replace')
    activated = log.count('Semantic TensorRT ring active:')
    declined = log.count('Semantic TensorRT ring declined:')
    if mode[0] == 'ring' and (activated < 1 or declined > 0):
        raise RuntimeError(
            f'Full TTA semantic ring was not active for every attempted task '
            f'(active={activated}, declined={declined}); see {output / "pipeline.log"}'
        )
    if mode[0] != 'ring' and activated:
        raise RuntimeError(f'Full TTA {mode[0]} unexpectedly activated semantic ring')
    paths = sorted((output / 'outputs' / 'semantic_masks').glob('*.png'))
    if not paths:
        raise RuntimeError(f'No semantic PNGs were published by {mode[0]}; see {output / "pipeline.log"}')
    return {
        'seconds': seconds, 'device_peak_used_bytes': peak,
        'semantic_png_count': len(paths), 'semantic_png_dir': str(paths[0].parent),
        'ring_active_task_count': activated, 'ring_decline_count': declined,
    }


def _compare_planes(rows: list[dict[str, object]], *, pipeline: bool) -> dict[str, object]:
    import cv2  # type: ignore
    import numpy as np

    baseline = rows[0]
    with np.load(Path(str(baseline['output'])) / 'direct-planes.npz') as base_direct:
        mask = np.array(base_direct['mask'], copy=True)
        confidence = np.array(base_direct['confidence'], copy=True)
    base_counts = [
        (int(item['prediction_count']), int(item['frames_with_predictions']))
        for item in baseline['direct']['timed']
    ]
    comparisons: list[dict[str, object]] = []
    pipeline_reference: list[object] | None = None
    if pipeline:
        baseline_pipeline = next((row for row in rows if row['mode'] == 'legacy' and 'pipeline' in row), None)
        if baseline_pipeline is None:
            raise RuntimeError('Pipeline comparison requires a legacy pipeline arm')
        base_dir = Path(str(baseline_pipeline['pipeline']['semantic_png_dir']))
        pipeline_reference = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                              for path in sorted(base_dir.glob('*.png'))]
        if any(frame is None for frame in pipeline_reference):
            raise RuntimeError('Unable to decode baseline pipeline semantic PNG')
    for row in rows:
        with np.load(Path(str(row['output'])) / 'direct-planes.npz') as candidate:
            difference = _plane_difference(
                mask, candidate['mask'], confidence, candidate['confidence'],
                confidence_threshold=float(row['direct']['case']['conf']),
            )
        counts = [(int(item['prediction_count']), int(item['frames_with_predictions']))
                  for item in row['direct']['timed']]
        counts_match = bool(counts == base_counts)
        pipeline_difference: dict[str, object] | None = None
        if pipeline and pipeline_reference is not None and 'pipeline' in row:
            current_dir = Path(str(row['pipeline']['semantic_png_dir']))
            current_paths = sorted(current_dir.glob('*.png'))
            current = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in current_paths]
            if len(current) != len(pipeline_reference) or any(frame is None for frame in current):
                pipeline_difference = {
                    'frame_count_match': len(current) == len(pipeline_reference),
                    'reference_frames': len(pipeline_reference), 'candidate_frames': len(current),
                    'within_tolerance': False,
                }
            else:
                pipeline_difference = _plane_difference(
                    np.stack(pipeline_reference), np.stack(current),
                )
                pipeline_difference['frame_count_match'] = True
        comparisons.append({
            'mode': row['mode'], 'direct': difference,
            'counts_exact': counts_match,
            'count_differences_by_repeat': [
                [counts[index][0] - base_counts[index][0],
                 counts[index][1] - base_counts[index][1]]
                for index in range(min(len(counts), len(base_counts)))
            ],
            'pipeline_semantic_png': pipeline_difference,
        })
    return {
        'all_exact': all(
            item['direct']['mask_exact'] and item['direct'].get('confidence_exact', False)
            and (item['pipeline_semantic_png'] is None
                 or item['pipeline_semantic_png'].get('mask_exact', False))
            for item in comparisons
        ),
        'all_counts_exact': all(item['counts_exact'] for item in comparisons),
        'all_within_tolerance': all(
            item['direct']['within_tolerance']
            and (item['pipeline_semantic_png'] is None
                 or item['pipeline_semantic_png']['within_tolerance'])
            for item in comparisons
        ),
        'comparisons': comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=_case_spec, action='append', default=[],
                        help='Repeat SIZE:BATCH:PATH, e.g. 512:2:C:\\models\\semantic.engine')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT if DEFAULT_INPUT.is_file() else None,
                        help='Representative video; omit for deterministic synthetic frames')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--min-conf', type=float, default=0.35)
    parser.add_argument('--quantize', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--backend-iterations', type=int, default=20)
    parser.add_argument('--postprocess-workers', type=int, default=4)
    parser.add_argument('--heatsoak-seconds', type=float, default=90.0)
    parser.add_argument('--cpu-heat-workers', type=int, default=4)
    parser.add_argument('--lock-wait-seconds', type=float, default=3600.0)
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Measure direct source only (full pipeline needs --input)')
    parser.add_argument('--pipeline-size', type=int, action='append', default=[],
                        help='Run full TTA only for this engine size; repeat for more sizes')
    parser.add_argument('--pipeline-mode', choices=tuple(mode[0] for mode in MODES),
                        action='append', default=[],
                        help='Run full TTA only for this mode; repeat (include legacy for comparison)')
    parser.add_argument('--_case-json', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._case_json is not None:
        case = json.loads(args._case_json.read_text(encoding='utf-8'))
        _run_direct_child(case)
        return 0
    if not args.engine:
        parser.error('At least one --engine SIZE:BATCH:PATH is required')
    if (args.device < 0 or args.frames < 2 or args.warmups < 1 or args.repeats < 2
            or args.backend_iterations < 1 or args.postprocess_workers < 1
            or not 0 < args.conf <= args.min_conf <= 1
            or args.cpu_heat_workers < 1 or args.heatsoak_seconds < 30
            or args.lock_wait_seconds < 0):
        parser.error('Require frames>=2, warmups>=1, repeats>=2, 0<conf<=min-conf<=1, heatsoak>=30s')
    if args.input is not None and not args.input.is_file():
        parser.error(f'Input video does not exist: {args.input}')
    if not args.skip_pipeline and args.input is None:
        parser.error('Full TTA pipeline comparison requires --input; or pass --skip-pipeline')
    if args.pipeline_mode and 'legacy' not in args.pipeline_mode and not args.skip_pipeline:
        parser.error('Pipeline comparison must include --pipeline-mode legacy')
    output = (args.output or DEFAULT_OUTPUT_PARENT / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if not output.is_relative_to(DEFAULT_OUTPUT_PARENT.resolve()) or output.exists():
        parser.error(f'Use a new --output beneath {DEFAULT_OUTPUT_PARENT}')
    if any(int(case['batch']) > int(args.frames) for case in args.engine):
        parser.error('--frames must be at least as large as every requested batch')
    output.mkdir(parents=True)
    report: dict[str, object] = {
        'status': 'running', 'scope': __doc__, 'device': args.device,
        'performance_scope': 'Local 4090 mobile eGPU CUDA sanity check; not representative H100 throughput',
        'input': str(args.input) if args.input else None,
        'engines': args.engine, 'cases': [], 'heatsoak_seconds': None,
        'tolerance': {
            'mask_mismatch_fraction_max': MASK_FRACTION_TOLERANCE,
            'confidence_max_abs_u8': CONFIDENCE_U8_TOLERANCE,
        },
    }
    receipt = output / 'qualification.json'
    _write_json(receipt, report)
    claim: dict[str, object] | None = None
    try:
        claim = _claim_gpu_lock(float(args.lock_wait_seconds))
        faulthandler.enable()
        faulthandler.dump_traceback_later(60, repeat=True)
        os.environ['CUPY_CACHE_DIR'] = str(output / 'cupy-cache')
        os.environ['NUMBA_CACHE_DIR'] = str(output / 'numba-cache')
        os.environ['TEMP'] = str(output / 'temp')
        os.environ['TMP'] = str(output / 'temp')
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        (output / 'cupy-cache').mkdir()
        (output / 'numba-cache').mkdir()
        (output / 'temp').mkdir()
        print('GPU lock acquired; importing Torch and initializing CUDA', flush=True)
        import torch  # type: ignore
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(int(args.device))
        report['gpu'] = torch.cuda.get_device_name(int(args.device))
        report['gpu_lock'] = claim
        report['phase'] = 'heatsoak'
        _write_json(receipt, report)
        print(f'Heatsoaking {report["gpu"]} and {args.cpu_heat_workers} CPU CC workers '
              f'for {args.heatsoak_seconds:.0f}s', flush=True)
        heatsoak = _heatsoak(
            torch, int(args.device), float(args.heatsoak_seconds), int(args.cpu_heat_workers),
        )
        report['heatsoak_seconds'] = heatsoak['gpu_seconds']
        report['heatsoak'] = heatsoak
        report['phase'] = 'controlled_logits'
        _write_json(receipt, report)
        torch.cuda.empty_cache()
        print('Qualifying controlled foreground-rich logits', flush=True)
        report['controlled_logits'] = _controlled_logit_qualification(torch, int(args.device))
        report['phase'] = 'direct_and_pipeline_cases'
        _write_json(receipt, report)
        for engine_case in args.engine:
            spec = dict(engine_case)
            spec.update({
                'device': int(args.device), 'frames': int(args.frames),
                'conf': float(args.conf), 'min_conf': float(args.min_conf),
                'quantize': str(args.quantize), 'warmups': int(args.warmups),
                'repeats': int(args.repeats), 'backend_iterations': int(args.backend_iterations),
                'postprocess_workers': int(args.postprocess_workers),
                'input': str(args.input) if args.input else None,
            })
            case_root = output / f'{spec["size"]}-b{spec["batch"]}'
            rows: list[dict[str, object]] = []
            for mode in MODES:
                mode_root = case_root / mode[0]
                mode_root.mkdir(parents=True)
                (mode_root / 'ultralytics-config').mkdir()
                child = {**spec, 'mode': mode[0], 'output': str(mode_root)}
                child_path = mode_root / 'direct-invocation.json'
                environment = _child_environment(mode_root, mode)
                _write_json(child_path, {
                    **child,
                    'environment': {
                        key: environment[key] for key in (
                            'YOLO_TTA_SEMANTIC_GPU_CLEANUP', 'YOLO_TTA_SEMANTIC_TRT_RING',
                            'YOLO_TTA_SEMANTIC_TRT_TRACE',
                            'YOLO_CONFIG_DIR', 'CUPY_CACHE_DIR', 'NUMBA_CACHE_DIR',
                        )
                    },
                })
                command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                           '--_case-json', str(child_path)]
                process_seconds, peak = _run_command(
                    command, output=mode_root, log_name='direct.log',
                    environment=environment, device=int(args.device),
                )
                direct = json.loads((mode_root / 'direct-result.json').read_text(encoding='utf-8'))
                row: dict[str, object] = {
                    'mode': mode[0], 'output': str(mode_root), 'direct': direct,
                    'direct_process_seconds': process_seconds,
                    'direct_process_device_peak_used_bytes': peak,
                }
                pipeline_selected = bool(
                    not args.skip_pipeline
                    and (not args.pipeline_size or int(spec['size']) in args.pipeline_size)
                    and (not args.pipeline_mode or mode[0] in args.pipeline_mode)
                )
                if pipeline_selected:
                    pipeline_root = mode_root / 'pipeline'
                    pipeline_root.mkdir()
                    row['pipeline'] = _pipeline_case(spec, mode, pipeline_root, environment)
                rows.append(row)
                print(f'{spec["size"]} B{spec["batch"]} {mode[0]}: '
                      f'{direct["median_frames_per_second"]:.2f} frames/s', flush=True)
                _write_json(case_root / 'progress.json', rows)
            comparison = _compare_planes(rows, pipeline=any('pipeline' in row for row in rows))
            legacy = float(rows[0]['direct']['median_seconds'])
            gpu_cleanup = float(rows[1]['direct']['median_seconds'])
            ring = float(rows[2]['direct']['median_seconds'])
            report['cases'].append({
                'size': spec['size'], 'batch': spec['batch'], 'runs': rows,
                'exactness': comparison,
                'gpu_cleanup_vs_legacy_speed_ratio': legacy / gpu_cleanup,
                'ring_vs_gpu_cleanup_speed_ratio': gpu_cleanup / ring,
                'ring_vs_legacy_speed_ratio': legacy / ring,
                'foreground_rich_model_output': bool(rows[0]['direct']['foreground_voxels'] > 0),
            })
            _write_json(receipt, report)
        within_tolerance = bool(
            report['controlled_logits']['all_within_tolerance']
            and all(
                case['exactness']['all_within_tolerance']
                and all(run['direct']['repeatability_within_tolerance'] for run in case['runs'])
                for case in report['cases']
            )
        )
        report['all_within_tolerance'] = within_tolerance
        report['status'] = 'complete' if within_tolerance else 'discrepant'
        _write_json(receipt, report)
        return 0 if within_tolerance else 2
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        _write_json(receipt, report)
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if claim is not None:
            _release_gpu_lock(claim)


if __name__ == '__main__':
    raise SystemExit(main())
