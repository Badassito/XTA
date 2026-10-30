"""Compare exact OpenVINO preparation against an immutable source snapshot.

This measures input preparation only, not model or whole-pipeline throughput.
No OpenVINO model or CUDA device is initialized. Run timing only in a coordinated
CPU slot; generated output and isolated runtime directories belong in Scratch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _reference(baseline: Path, formatter: object):
    # Both arms intentionally share the unchanged channel formatter. Refuse a
    # baseline with different helper semantics rather than silently mixing eras.
    helper = ('InMemoryYoloVolumeSource', '_frame_to_model_channels')
    old = _method(baseline / 'XTA/geometry.py', *helper)
    current = _method(ROOT / 'XTA/geometry.py', *helper)
    if ast.dump(old) != ast.dump(current):
        raise ValueError('Baseline channel formatter differs from current source')
    method = _method(baseline / 'XTA/workers.py', '_OpenVinoCpuSegmenter', '_prepare_input')
    namespace = dict(np=np, Sequence=Sequence, InMemoryYoloVolumeSource=formatter)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(baseline / 'XTA/workers.py'), 'exec'), namespace)
    return namespace['_prepare_input']


def _runner(cls, batch, channels, layout, element_type):
    result = cls.__new__(cls)
    result.batch, result.input_channels = batch, channels
    result.input_layout, result.input_element_type = layout, element_type
    return result


def _parity(cls, baseline, candidate):
    count = 0
    rng = np.random.default_rng(7321)
    for batch in (1, 3):
        for channels in (1, 3, 5):
            for layout in ('NCHW', 'NHWC'):
                for element_type in ('u8', 'uint8', 'f16', 'float16', 'f32', 'bf16'):
                    obj = _runner(cls, batch, channels, layout, element_type)
                    for storage in ('contiguous', 'strided', 'reverse', 'float64'):
                        images = [rng.integers(0, 256, (32,24,channels), dtype=np.uint8) for _ in range(batch)]
                        if storage == 'strided':
                            images = [image[::2, ::2] for image in images]
                        elif storage == 'reverse':
                            images = [image[::-1, ::-1] for image in images]
                        elif storage == 'float64':
                            images = [image.astype(np.float64) + .5 for image in images]
                        before = [image.copy() for image in images]
                        old, new = baseline(obj, images), candidate(obj, images)
                        assert old.dtype == new.dtype and old.shape == new.shape
                        assert new.flags.c_contiguous and np.array_equal(old, new)
                        for source, snapshot in zip(images, before):
                            assert np.array_equal(source, snapshot)
                            assert not np.shares_memory(new, source)
                        count += 1
    return count


def _measure(cls, baseline, candidate, seconds):
    rows = []
    rng = np.random.default_rng(9432)
    for batch, side, channels in ((1,640,3), (8,640,3), (16,640,3), (8,1024,1)):
        images = [rng.integers(0,256,(side,side,channels),dtype=np.uint8) for _ in range(batch)]
        for element_type in ('u8', 'f16', 'f32'):
            for layout in ('NCHW', 'NHWC'):
                obj = _runner(cls, batch, channels, layout, element_type)
                functions = dict(baseline=baseline, candidate=candidate)
                times = {name: [] for name in functions}
                until = time.perf_counter() + seconds
                iteration = 0
                while time.perf_counter() < until:
                    order = list(functions) if iteration % 2 == 0 else list(reversed(functions))
                    for name in order:
                        started = time.perf_counter()
                        result = functions[name](obj, images)
                        times[name].append(time.perf_counter() - started)
                        del result
                    iteration += 1
                peaks = {}
                for name, fn in functions.items():
                    tracemalloc.start()
                    try:
                        result = fn(obj, images)
                        _, peaks[name] = tracemalloc.get_traced_memory()
                        del result
                    finally:
                        tracemalloc.stop()
                rows.append(dict(batch=batch,side=side,channels=channels,dtype=element_type,
                    layout=layout,iterations=iteration,median_ms={k:statistics.median(v)*1000 for k,v in times.items()},
                    peak_bytes=peaks))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--seconds', type=float, default=.5)
    parser.add_argument('--warmup', type=float, default=20)
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.warmup < 0:
        parser.error('seconds must be positive and warmup must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = args.output.parent / f'openvino-preparation-runtime-{os.getpid()}'
    for name, subdir in (('YOLO_CONFIG_DIR','yolo'), ('TEMP','temp'), ('TMP','temp'),
                         ('XDG_CACHE_HOME','cache'), ('TORCH_HOME','torch'), ('HF_HOME','huggingface')):
        directory = runtime / subdir
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, str(ROOT))
    from XTA.workers import _OpenVinoCpuSegmenter
    from XTA.geometry import InMemoryYoloVolumeSource
    baseline_root = args.baseline_source.resolve()
    baseline = _reference(baseline_root, InMemoryYoloVolumeSource)
    candidate = _OpenVinoCpuSegmenter._prepare_input
    output = dict(baseline=str(baseline_root), candidate=str(ROOT),
        baseline_workers_sha256=hashlib.sha256((baseline_root/'XTA/workers.py').read_bytes()).hexdigest(),
        candidate_workers_sha256=hashlib.sha256((ROOT/'XTA/workers.py').read_bytes()).hexdigest(),
        numpy_version=np.__version__, scope='input preparation only; no model or GPU execution',
        parity_cases=_parity(_OpenVinoCpuSegmenter, baseline, candidate))
    if args.timing:
        obj = _runner(_OpenVinoCpuSegmenter,8,3,'NCHW','f32')
        images = [np.full((640,640,3),128,np.uint8) for _ in range(8)]
        until = time.perf_counter() + args.warmup
        while time.perf_counter() < until:
            baseline(obj,images)
            candidate(obj,images)
        output['heatsoak_seconds'] = args.warmup
        output['timings'] = _measure(_OpenVinoCpuSegmenter,baseline,candidate,args.seconds)
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
