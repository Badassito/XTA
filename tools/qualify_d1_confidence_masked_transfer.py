"""Paired CUDA qualification for D1 confidence's one-copy crop capture.

Run only while the caller holds Scratch/Temp/GPU_LOCK. Each ABBA pair compares
the production CUDA-masked capture with the previous two-copy host masking on
the same source tensors. Exact compressed score and index bytes are checked
for every shape and layout before timing. These are local path timings, not a
prediction of H100 pipeline walltime.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from XTA import cuda_d1
from XTA.confidence_evidence import write_block_confidence_evidence


@contextmanager
def _mode(mask_on: bool):
    with mock.patch.dict(os.environ, {'YOLO_TTA_D1_GPU_MASK_CONFIDENCE':
                                      '1' if mask_on else '0',
                                      'YOLO_TTA_D1_GPU_MASK_MIN_PIXELS': '0'}):
        yield


def _capture(scores, masks, box, *, mask_on: bool):
    pixels = (box[1] - box[0]) * (box[3] - box[2])
    metrics = cuda_d1._d1_confidence_metrics((1, *scores.shape[1:]), {0: box})
    with _mode(mask_on):
        started = time.perf_counter()
        captured = cuda_d1._d1_copy_confidence_crop(scores, masks, 0, box, metrics)
        elapsed = time.perf_counter() - started
    expected_bytes = pixels if mask_on else 2 * pixels
    expected_calls = 1 if mask_on else 2
    if metrics['d2h_bytes'] != expected_bytes or metrics['d2h_calls'] != expected_calls:
        raise AssertionError(f'unexpected copy metrics: {metrics}')
    if metrics['gpu_mask_oom_fallbacks']:
        raise AssertionError('masked path ran out of device memory')
    return captured[4], elapsed, metrics


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(torch, *, size: int, layout: str, root: Path, repeats: int):
    offset = 0 if layout == 'contiguous' else 5
    height = size + (0 if layout == 'contiguous' else 11)
    width = size + (0 if layout == 'contiguous' else 13)
    box = (offset, offset + size, offset, offset + size)
    rng = np.random.default_rng(2048 + size + offset)
    scores_np = rng.integers(0, 256, (1, height, width), dtype=np.uint8)
    masks_np = rng.choice(np.asarray((0, 0, 0, 1, 2, 255), np.uint8),
                          size=(1, height, width))
    scores_np[0, offset, offset] = 0
    masks_np[0, offset, offset] = 1
    scores = torch.as_tensor(scores_np, device='cuda')
    masks = torch.as_tensor(masks_np, device='cuda')
    torch.cuda.synchronize()
    if bool(scores[0, box[0]:box[1], box[2]:box[3]].is_contiguous()) != (
            layout == 'contiguous'):
        raise AssertionError(f'{layout} fixture has wrong crop contiguity')

    masked, _, masked_metrics = _capture(scores, masks, box, mask_on=True)
    host, _, host_metrics = _capture(scores, masks, box, mask_on=False)
    oracle = np.where(masks_np[0, box[0]:box[1], box[2]:box[3]] != 0,
                      scores_np[0, box[0]:box[1], box[2]:box[3]], np.uint8(0))
    if not np.array_equal(masked, host) or not np.array_equal(masked, oracle):
        raise AssertionError(f'{layout} {size}: captured values differ')
    hashes = {}
    for label, values in (('masked', masked), ('host', host)):
        path = root / label
        write_block_confidence_evidence(
            path, (1, size, size), lambda _z, values=values: values,
            model_name='qualification', layer_key='confidence',
        )
        hashes[label] = {name: _hash(path / name)
                         for name in ('index.bin', 'scores.u8.zlib')}
    if hashes['masked'] != hashes['host']:
        raise AssertionError(f'{layout} {size}: encoded evidence differs')

    durations = {'masked': [], 'host': []}
    for _ in range(repeats):
        for label, mode in (('masked', True), ('host', False),
                            ('host', False), ('masked', True)):
            values, elapsed, _ = _capture(scores, masks, box, mask_on=mode)
            if not np.array_equal(values, oracle):
                raise AssertionError(f'{layout} {size}: timed capture differs')
            durations[label].append(elapsed)
    def summarize(samples):
        sorted_values = sorted(samples)
        return dict(median_ms=1000 * statistics.median(samples),
                    p90_ms=1000 * sorted_values[min(len(samples)-1,
                                                     int(.9 * len(samples)))],
                    min_ms=1000 * sorted_values[0], n=len(samples))
    return dict(size=size, layout=layout, source_shape=[height, width],
                crop_contiguous=(layout == 'contiguous'),
                evidence_sha256=hashes['masked'],
                copy_metrics=dict(masked=masked_metrics, host=host_metrics),
                timing={name: summarize(times) for name, times in durations.items()},
                paired_ratio_host_over_masked=(statistics.median(durations['host']) /
                                               statistics.median(durations['masked'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sizes', default='32,128,512,2048')
    parser.add_argument('--layouts', default='contiguous,offset')
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--warm-seconds', type=float, default=20)
    args = parser.parse_args()
    if args.repeats < 2 or args.warm_seconds < 0:
        parser.error('repeats must be at least 2 and warm-seconds must be nonnegative')
    sizes = tuple(int(part) for part in args.sizes.split(','))
    layouts = tuple(part.strip() for part in args.layouts.split(','))
    if not sizes or min(sizes) < 1 or not set(layouts) <= {'contiguous', 'offset'}:
        parser.error('sizes must be positive and layouts must be contiguous or offset')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA qualification requested but CUDA is unavailable')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warm_size = max(sizes)
    warm_scores = torch.full((1, warm_size+1, warm_size+1), 177,
                             dtype=torch.uint8, device='cuda')
    warm_masks = torch.ones_like(warm_scores)
    warm_box = (0, warm_size, 0, warm_size)
    torch.cuda.synchronize()
    warm_count = 0
    deadline = time.perf_counter() + args.warm_seconds
    while time.perf_counter() < deadline:
        _capture(warm_scores, warm_masks, warm_box, mask_on=bool(warm_count % 2))
        warm_count += 1
    torch.cuda.synchronize()
    del warm_scores, warm_masks

    results = []
    for size in sizes:
        for layout in layouts:
            case = _case(torch, size=size, layout=layout,
                         root=args.output_dir / f'{size}-{layout}', repeats=args.repeats)
            results.append(case)
            print(f"{size:4d} {layout:10s}: masked {case['timing']['masked']['median_ms']:.3f} ms; "
                  f"host {case['timing']['host']['median_ms']:.3f} ms; "
                  f"ratio {case['paired_ratio_host_over_masked']:.3f}", flush=True)
    report = dict(qualification='d1-confidence-gpu-masked-crop-abba',
                  local_host='4090 Laptop over Oculink',
                  torch_version=torch.__version__,
                  cuda_device=torch.cuda.get_device_name(),
                  source_sha256=_hash(Path(cuda_d1.__file__)),
                  warm_seconds=args.warm_seconds, warm_captures=warm_count,
                  repeats=args.repeats, cases=results)
    target = args.output_dir / 'result.json'
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
