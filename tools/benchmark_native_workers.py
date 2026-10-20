"""Measure native-filter process scaling with repeated copies of a real source window."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, redirect_stdout
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy import ndimage as ndi

import xta_native_filter as nf
from benchmark_native_filter import window


@contextmanager
def thread_control_executor():
    """Run the same shared-map worker entry point in threads for a backend control."""
    previous_executor = nf.native_executor
    previous_context = nf._native_worker_context
    @contextmanager
    def executor(workers, context):
        nf.initialize_native_worker(context)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                yield pool
        finally:
            worker_context = nf._native_worker_context
            for key in ('clip', 'union'):
                mapped = worker_context.get(key)
                if isinstance(mapped, np.memmap):
                    mapped._mmap.close()
            nf._native_worker_context = previous_context
    nf.native_executor = executor
    try:
        yield
    finally:
        nf.native_executor = previous_executor
        nf._native_worker_context = previous_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--global-mask', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--planes', type=int, default=32)
    parser.add_argument('--copies', type=int, default=8)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 8])
    args = parser.parse_args()
    if args.start < 0 or args.planes < 1 or args.copies < 1 or min(args.workers) < 1:
        parser.error('start must be nonnegative; planes, copies and workers must be positive')
    if len(set(args.workers)) != len(args.workers):
        parser.error('worker counts must be unique')
    if args.output.exists():
        parser.error('output must be a new directory')
    args.output.mkdir(parents=True)
    native = nf.Nrrd(args.native)
    data = window(native, args.start, args.planes)
    clip = window(nf.Nrrd(args.global_mask, native.shape), args.start, args.planes)
    shape = data.shape
    low_shape = (max(1, args.planes//4), max(1, shape[1]//5), max(1, shape[2]//5))

    cache = {}
    def pooled(a, requested):
        key = (id(a), tuple(requested))
        if key not in cache:
            check = nf.PreviewSupportCheck(a.shape, np.zeros(requested, np.uint8))
            for z, block in nf.array_slabs(a, 1):
                check.observe(z, block)
            cache[key] = check.seen.astype(np.uint8)
        return cache[key]

    original_low = pooled(data, low_shape)
    low_clip = pooled(clip, low_shape)
    labels, count = ndi.label(original_low & low_clip, structure=nf.FACE)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if not count or not np.any(sizes):
        raise ValueError('The selected source window has no foreground inside Global.')
    edited = (labels == int(np.argmax(sizes))).astype(np.uint8)
    if np.array_equal(edited, original_low):
        raise ValueError('The selected window needs more than one preview island to exercise deletion processing.')
    # Reuse the dev fixture writer while replacing its slow scalar pooling helper.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests import test_native_filter as fixtures
    previous_pool = fixtures.pool
    fixtures.pool = pooled
    try:
        base_args = fixtures.fixture(args.output/'fixture',
            {f'copy{i:03d}': data for i in range(args.copies)},
            {f'copy{i:03d}': edited for i in range(args.copies)},
            global_mask=clip, low_shape=low_shape)
    finally:
        fixtures.pool = previous_pool
    metadata = {'native_file': str(args.native.resolve()), 'native_sha256': nf.digest(args.native),
        'global_file': str(args.global_mask.resolve()), 'global_sha256': nf.digest(args.global_mask),
        'window_start': args.start, 'shape_tyx': list(shape), 'preview_shape_tyx': list(low_shape),
        'foreground_voxels_per_copy': int(np.count_nonzero(data)),
        'original_preview_voxels': int(np.count_nonzero(original_low)),
        'edited_preview_voxels': int(np.count_nonzero(edited)),
        'repeated_copies_of_one_source': args.copies, 'distinct_sources': 1,
        'includes_fixture_build_and_source_window_extraction_in_timings': False,
        'includes_full_filter_run_preflight_decode_and_publication': True,
        'cluster_throughput_benchmark': False, 'results': {}}
    del data, clip, original_low, low_clip, labels, sizes, edited
    cache.clear()
    gc.collect()
    # Warm the CPU with the same full-width reduction used by support validation.
    warm_frame = np.ones(shape[1:], np.uint8)
    warm = nf.PreviewSupportCheck(shape, np.zeros(low_shape, np.uint8))
    started = time.perf_counter()
    while time.perf_counter()-started < 45:
        warm.observe(0, warm_frame[None])
    metadata['cpu_warmup_seconds'] = time.perf_counter()-started
    del warm_frame, warm
    gc.collect()

    try:
        import psutil
    except ImportError:
        psutil = None
    expected_digest = expected_support = None
    cases = [(str(workers), workers, False) for workers in args.workers]
    if max(args.workers) > 1:
        cases.append((f'{max(args.workers)}-threads', max(args.workers), True))
    for case, workers, thread_control in cases:
        run_args = argparse.Namespace(**vars(base_args))
        run_args.output = str(args.output/f'workers-{case}')
        run_args.scratch = str(args.output/'scratch')
        run_args.workers = workers
        run_args.conflicts = 'keep'
        stop = threading.Event()
        memory = {'psutil_available': psutil is not None,
                  'peak_summed_process_rss_bytes': 0, 'minimum_system_available_bytes': None,
                  'rss_includes_double_counted_shared_file_pages': True}
        def monitor():
            if psutil is None:
                return
            parent = psutil.Process(os.getpid())
            while not stop.is_set():
                total = 0
                for process in [parent, *parent.children(recursive=True)]:
                    try:
                        total += process.memory_info().rss
                    except psutil.Error:
                        pass
                memory['peak_summed_process_rss_bytes'] = max(memory['peak_summed_process_rss_bytes'], total)
                available = psutil.virtual_memory().available
                before = memory['minimum_system_available_bytes']
                memory['minimum_system_available_bytes'] = available if before is None else min(before, available)
                stop.wait(.2)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        started = time.perf_counter()
        try:
            with open(args.output/f'workers-{case}.log', 'w', encoding='utf-8') as log, redirect_stdout(log), \
                    (thread_control_executor() if thread_control else nullcontext()):
                code = nf.run(run_args)
            assert code == 0
            elapsed = time.perf_counter()-started
        finally:
            stop.set()
            watcher.join()
        report = nf.load_json(Path(run_args.output)/'report.json')
        result_digest = nf.digest(Path(run_args.output)/report['pta_label_file'])
        source_checks = {row['source_filename']: row['source_support_check'] for row in report['layers']}
        if expected_digest is None:
            expected_digest, expected_support = result_digest, source_checks
        else:
            assert result_digest == expected_digest, 'Worker counts produced different output NRRDs'
            assert source_checks == expected_support, 'Worker counts produced different source support checks'
        metadata['results'][case] = {'wall_seconds': elapsed,
            'worker_backend': 'thread_control' if thread_control else report.get('worker_backend', 'processes' if workers > 1 else 'thread'),
            'production_report_worker_backend': report.get('worker_backend'),
            'final_global_voxels': report['final_global_voxels'], 'output_sha256': result_digest,
            'unresolved_components': report['unresolved_components'], 'memory': memory,
            'layers': [{key: row.get(key) for key in ('name', 'label_seconds', 'auto_seconds',
                'merge_seconds', 'elapsed_seconds', 'native_components', 'sparse_label_bytes')}
                for row in report['layers']]}
        nf.save_json(args.output/'benchmark.json', metadata)
        print(f'{case} worker(s): {elapsed:.3f}s', flush=True)
    baseline = metadata['results'][str(args.workers[0])]['wall_seconds']
    for result in metadata['results'].values():
        result['speedup_vs_first'] = baseline/result['wall_seconds']
    metadata['all_output_nrrds_byte_identical'] = True
    metadata['all_source_support_checks_equal'] = True
    nf.save_json(args.output/'benchmark.json', metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
