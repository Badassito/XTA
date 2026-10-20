"""Compare dense and sparse component processing on a window of a real NRRD."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np

import xta_native_filter as current
from xta_native_sparse import SparseLabels, sparse_label_stream


def window(source, start, count):
    result = np.empty((count, *source.shape[1:]), np.uint8)
    copied = 0
    for z, block in source.slabs(8):
        lo, hi = max(z, start), min(z+len(block), start+count)
        if hi > lo:
            result[lo-start:hi-start] = block[lo-z:hi-z]
            copied += hi-lo
        if z+len(block) >= start+count:
            break
    if copied != count:
        raise ValueError('Benchmark window extends beyond the native volume.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--global-mask', type=Path, required=True)
    p.add_argument('--low-dir', type=Path, required=True)
    p.add_argument('--edits', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--planes', type=int, default=64)
    args = p.parse_args()
    if args.start < 0 or args.planes < 1:
        p.error('start must be nonnegative and planes must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('dense_reference', args.reference)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    native = current.Nrrd(args.native)
    data = window(native, args.start, args.planes)
    clip = window(current.Nrrd(args.global_mask), args.start, args.planes)
    low_src = current.Nrrd(args.low_dir/args.native.name)
    original = low_src.array()
    bundle = current.load_json(args.edits/'edits.json')
    entry = next(e for e in bundle['segments'] if e['source_filename'] == args.native.name)
    after = current.load_edit(args.edits, entry, low_src)
    low_global = current.Nrrd(args.low_dir/args.global_mask.name).array()
    removed = (original != 0) & (after == 0)
    if np.any(low_global):
        removed &= low_global != 0
    # Warm the CPU on the same projection workload before measuring either path.
    warm = current.PreviewSupportCheck(native.shape, original)
    deadline = time.perf_counter()+45
    while time.perf_counter() < deadline:
        warm.observe(args.start, data[:1])
    rows, results, projections = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='benchmark-', dir=args.output) as tmp:
        for name, module in (('dense', old), ('sparse', current)):
            support = module.Support(native.shape, original.shape)
            support.indices[0] = tuple(a[args.start:args.start+args.planes] for a in support.indices[0])
            check = module.PreviewSupportCheck(native.shape, original)
            check.inverse = tuple(a[args.start:args.start+args.planes] for a in check.inverse)
            path = Path(tmp)/name
            context = (old.mapped(path, data.shape, np.int32) if name == 'dense'
                       else SparseLabels(data.shape, path))
            with context as labels, open(args.output/(name+'.log'), 'w') as log, redirect_stdout(log):
                started = time.perf_counter()
                fn = old.label_stream if name == 'dense' else sparse_label_stream
                result = fn(current.array_slabs(data, 1), labels, maximum=2_000_000,
                            keep=after, removed=removed, support=support,
                            clip=clip, observer=check.observe)
                elapsed = time.perf_counter()-started
                rows[name] = {'label_seconds': elapsed, 'label_bytes': path.stat().st_size}
                # Component identity/count/evidence and source projection must agree.
                results[name] = result
                projections[name] = check.seen.copy()
        for a, b in zip(results['dense'], results['sparse']):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(projections['dense'], projections['sparse'])
    record = {'native_file': str(args.native), 'native_sha256': current.digest(args.native),
              'window_start': args.start, 'window_shape': list(data.shape),
              'foreground_voxels': int(np.count_nonzero(data)), 'cpu_warmup_seconds': 45,
              'component_and_evidence_equality': True, 'source_projection_equality': True,
              'includes_decompression': False, 'results': rows,
              'label_speedup': rows['dense']['label_seconds']/rows['sparse']['label_seconds'],
              'label_storage_reduction': rows['dense']['label_bytes']/max(1, rows['sparse']['label_bytes'])}
    current.save_json(args.output/'benchmark.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
