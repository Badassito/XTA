"""Bounded no-interpolation bridge-support prototype; never a pipeline speed claim.

The candidate is valid only when assembly proves no interpolation pass ran.
It deliberately does not accept zero-added statistics as proof. Generated
stores and measurement evidence belong in the explicit Scratch output path.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def empty_support_prototype(after, before, destination, *, desc, workers=1,
                            format_name=None):
    from XTA.cuda_d1 import _volume_shape_tuple
    from XTA.interpolation import (CVOL_FORMAT, RawBBoxSlicePayload,
                                   _write_raw_bbox_payload_store)
    shape = _volume_shape_tuple(after)
    before_shape = _volume_shape_tuple(before)
    if shape != before_shape:
        raise ValueError(f'{desc}: shape mismatch {shape} vs {before_shape}')
    return _write_raw_bbox_payload_store(
        shape=shape, store_dir=Path(destination),
        encode_slice=lambda idx: RawBBoxSlicePayload(idx=int(idx), is_empty=True),
        format_name=CVOL_FORMAT if format_name is None else format_name,
        desc=desc, workers=workers, extra_meta={'operation': 'after_and_not_before'})


def _store_bytes(root):
    return {name: (Path(root) / name).read_bytes()
            for name in ('index.bin', 'chunks.bin', 'meta.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape', default='32,2048,2048')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--trials', type=int, default=6)
    parser.add_argument('--heatsoak-seconds', type=float, default=20.)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import numpy as np
    import psutil
    from XTA.cuda_d1 import subtract_volume_to_raw_bbox_store
    from XTA.interpolation import CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
    from XTA.sam_integration import publish_sam_gate_identity

    shape = tuple(map(int, args.shape.split(',')))
    if len(shape) != 3 or min(shape) <= 0 or np.prod(shape) > 1024**3:
        raise ValueError('Use a positive TYX fixture no larger than 1 GiB')
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output / 'operations.log').open('w', encoding='utf-8')
    report = dict(shape=shape, fixture='Synthetic deterministic sparse blocks, immutable no-pass parent',
                  logical_bytes=int(np.prod(shape)), workers=args.workers,
                  cache_regime='warm source and store pages; complete synchronous output stores',
                  source_hashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                 for name in ('XTA/assembly.py', 'XTA/cuda_d1.py', 'XTA/interpolation.py')},
                  samples=[], exact_store_bytes=True, exact_sam_identity=True)
    with log, contextlib.redirect_stdout(log):
        source = np.zeros(shape, dtype=np.uint8)
        source[:, shape[1]//3:shape[1]//3+19, shape[2]//4:shape[2]//4+23] = 1
        support_path = args.output / 'parent.cvol'
        write_raw_bbox_mask_store(source, support_path, format_name=CVOL_FORMAT,
                                  desc='Immutable synthetic parent', workers=args.workers)
        support = RawBBoxMaskStore.open(support_path, mmap_payload=True)
        try:
            until = time.perf_counter() + max(0., args.heatsoak_seconds)
            while time.perf_counter() < until:
                subtract_volume_to_raw_bbox_store(source, support, args.output / 'heatsoak.cvol',
                    desc='Representative CPU heatsoak', workers=args.workers, format_name=CVOL_FORMAT)
            reference = None
            identity = None
            functions = {'baseline': subtract_volume_to_raw_bbox_store,
                         'candidate': empty_support_prototype}
            for trial in range(args.trials):
                order = ('baseline', 'candidate') if trial % 2 == 0 else ('candidate', 'baseline')
                for arm in order:
                    destination = args.output / f'{trial:02d}-{arm}.cvol'
                    started = time.perf_counter()
                    stats = functions[arm](source, support, destination,
                        desc='Exact no-pass bridge support', workers=args.workers,
                        format_name=CVOL_FORMAT)
                    elapsed = time.perf_counter() - started
                    content = _store_bytes(destination)
                    if reference is None:
                        reference = content
                    if content != reference:
                        raise AssertionError(f'Artifact byte mismatch: {trial}/{arm}')
                    with contextlib.closing(RawBBoxMaskStore.open(destination)) as empty:
                        current = publish_sam_gate_identity(empty, policy_identity='', evidence_path='')
                    if identity is None:
                        identity = current
                    if current != identity:
                        raise AssertionError('SAM empty-support identity changed')
                    mem = psutil.Process().memory_info()
                    report['samples'].append(dict(trial=trial, arm=arm, seconds=elapsed,
                        rss=mem.rss, peak_wset=getattr(mem, 'peak_wset', None), stats=stats))
            # Preserve the original shape failure rather than certifying an empty wrong grid.
            errors = []
            for arm, operation in functions.items():
                try:
                    operation(source[:1], support, args.output / f'wrong-{arm}',
                              desc='Mismatch', workers=1, format_name=CVOL_FORMAT)
                except ValueError as error:
                    errors.append(str(error))
            if len(errors) != 2 or errors[0] != errors[1]:
                raise AssertionError(f'Shape error mismatch: {errors}')
            report['exact_shape_error'] = errors[0]
        finally:
            support.close()
    report['medians'] = {arm: statistics.median(sample['seconds'] for sample in report['samples']
                         if sample['arm'] == arm) for arm in ('baseline', 'candidate')}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['medians']))


if __name__ == '__main__':
    main()
