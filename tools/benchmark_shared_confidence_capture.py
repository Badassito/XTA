"""Paired four-parent confidence capture with unchanged numeric staging limits.

The legacy witness uses the existing standalone path with one worker per parent;
the candidate uses the run's shared CPU scheduler. This is a bounded local CPU
sanity check, not a full-cluster or scientific quality qualification.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MIB = 1024**2


def source_pins():
    result = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'XTA/confidence_capture.py', 'XTA/confidence_capture_cpu.py',
        'XTA/confidence_evidence.py', 'XTA/confidence_storage.py',
        'XTA/confidence_publication.py', 'XTA/native_pull_scheduling.py',
        'tools/benchmark_shared_confidence_capture.py')}
    text = (ROOT/'XTA/pipeline.py').read_text()
    node = next(node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name == '_parent_confidence_capture_plan')
    result['XTA/pipeline.py::_parent_confidence_capture_plan'] = hashlib.sha256(
        ast.get_source_segment(text, node).encode()).hexdigest()
    return result


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(MIB):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, choices=(16, 32, 64), default=32)
    parser.add_argument('--repetitions', type=int, choices=(1, 2), default=2)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--reference-storage-source', type=Path,
        help='Compare a frozen predecessor writer against current bulk storage with the same shared CPU budget')
    parser.add_argument('--reference-storage-sha256')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT.parent/'Scratch').resolve()):
        parser.error('Generated artifacts must be under sibling Scratch')
    if args.heatsoak_seconds < 60:
        parser.error('Measurements require at least 60 seconds CPU heatsoak')
    if bool(args.reference_storage_source) != bool(args.reference_storage_sha256):
        parser.error('Frozen writer source requires its exact SHA256')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'benchmark.json').exists():
        parser.error('Use a fresh output directory')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    import numpy as np
    import psutil
    from XTA import confidence_evidence as evidence, geometry, pipeline
    from XTA import confidence_storage as storage
    from XTA.confidence_capture import confidence_capture_resources
    from XTA.confidence_capture_cpu import warm_confidence_capture_kernels
    from XTA.confidence_publication import ConfidencePublicationQueue
    from XTA.native_pull_scheduling import configure_native_pull_scheduler, shutdown_native_pull_scheduler
    from tools.benchmark_native_pull_projection import heatsoak, PeakRss
    workers = min(32, os.cpu_count() or 1)
    shape = (args.frames, 2048, 2048)
    source_bytes = 4*2*int(np.prod(shape))
    if psutil.virtual_memory().available < source_bytes + 2*1024**3:
        raise RuntimeError('Insufficient physical headroom for independent source buffers and capture windows')
    pins = source_pins()
    current_writer = storage.write_blocks
    predecessor_writer = None
    if args.reference_storage_source:
        if digest(args.reference_storage_source) != args.reference_storage_sha256:
            parser.error('Frozen confidence writer source hash mismatch')
        spec = importlib.util.spec_from_file_location('XTA._confidence_writer_reference',
            args.reference_storage_source)
        reference_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference_module
        spec.loader.exec_module(reference_module)
        # The queue catches this public exception from the production module.
        # A private import must share that identity to reproduce the old
        # writer's normal64MiB staging fallback rather than inventing a failure.
        reference_module.ConfidenceStageLimit = storage.ConfidenceStageLimit
        predecessor_writer = reference_module.write_blocks
    report = dict(source_pins=pins, shape=shape, parents=4, workers=workers,
        python=sys.version, numpy=np.__version__, logical_cpus=os.cpu_count(),
        quiet_window='Explicit task authorization; preceding GPU/CPU probe finished before this probe',
        source_physical_bytes=source_bytes, repetitions=args.repetitions,
        legacy_scope='Current standalone path, no configured scheduler, explicit one worker per parent',
        candidate_scope='Configured shared run scheduler; charged per-parent window, unchanged four numeric slots',
        limitations='Local synthetic CPU capture; no model, projection, GPU or cluster wall-time claim',
        measurements=[])
    if predecessor_writer is not None:
        report.update(reference_storage_source=str(args.reference_storage_source),
            reference_storage_sha256=args.reference_storage_sha256,
            reference_exception_binding='Frozen writer uses production ConfidenceStageLimit identity for unchanged queue fallback',
            legacy_scope='Frozen pre-bulk writer using the same configured shared CPU scheduler',
            candidate_scope='Current bulk writer using the same configured shared CPU scheduler')
    warm_confidence_capture_kernels()
    print('Confidence kernels warm; starting CPU heatsoak', flush=True)
    report['heatsoak'] = heatsoak(args.heatsoak_seconds, min(8, workers))
    rng = np.random.default_rng(160587)
    for pattern in ('sparse', 'noisy'):
        fixtures = []
        for parent in range(4):
            mask = np.zeros(shape, np.uint8) if pattern == 'sparse' else np.ones(shape, np.uint8)
            scores = np.full(shape, 173, np.uint8)
            active = np.ones(shape[0], bool)
            boxes = np.empty((shape[0], 4), np.int64)
            for z in range(shape[0]):
                if pattern == 'sparse':
                    y, x = 17+(z*13+parent*7)%1300, 19+(z*17+parent*11)%1300
                    mask[z, y:y+256, x:x+256] = 1
                    scores[z, y:y+256, x:x+256] = rng.integers(0, 256, (256, 256), np.uint8)
                    boxes[z] = (y, y+256, x, x+256)
                else:
                    scores[z] = rng.integers(0, 256, shape[1:], np.uint8)
                    boxes[z] = (0, shape[1], 0, shape[2])
            for values in (mask, scores, active, boxes):
                values.flags.writeable = False
            fixtures.append((mask, scores, active, boxes))
        input_hashes = [[hashlib.sha256(memoryview(values).cast('B')).hexdigest()
                         for values in fixture] for fixture in fixtures]
        base_view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
        expected_hashes = None
        timings = dict(legacy=[], shared=[])
        for repetition in range(args.repetitions):
            for backend in (('legacy', 'shared') if repetition%2 == 0 else ('shared', 'legacy')):
                target = output/pattern/f'{backend}-{repetition}'
                scheduler = None
                if backend == 'shared' or predecessor_writer is not None:
                    scheduler = configure_native_pull_scheduler(workers=workers, max_workers=128)
                storage.write_blocks = (predecessor_writer if backend == 'legacy'
                                        and predecessor_writer is not None else current_writer)
                plan = pipeline._parent_confidence_capture_plan(shape, 1, enabled=True)
                if plan.workspace_bytes > 256*MIB:
                    raise RuntimeError('Capture plan exceeded unchanged per-parent admission')
                evidence.configure_confidence_evidence(target/'out', enabled=True, defer_projection=True)
                publication = ConfidencePublicationQueue()  # Four64MiB slots, two immutable-file copies.
                evidence._PUBLICATION = publication
                def capture(parent):
                    mask, scores, active, boxes = fixtures[parent]
                    with confidence_capture_resources(plan):
                        return evidence.capture_prediction_confidence(mask, scores,
                            view=replace(base_view, name=f'parent_{parent}'), model_name='m',
                            temp_dir=target/'temporary', known_slice_any=active,
                            known_slice_bboxes=boxes)
                try:
                    with PeakRss() as rss:
                        started = time.perf_counter()
                        with ThreadPoolExecutor(max_workers=4) as parents:
                            results = list(parents.map(capture, range(4)))
                        references = [value.result() if hasattr(value, 'result') else value for value in results]
                        publication_state = publication.drain()
                        seconds = time.perf_counter()-started
                    timings[backend].append(seconds)
                    hashes = [[digest(ref.path/name) for name in ('scores.u8.zlib', 'index.bin', 'metadata.json')]
                              for ref in references]
                    if expected_hashes is None:
                        expected_hashes = hashes
                    if hashes != expected_hashes:
                        raise AssertionError('Legacy/shared confidence payload, index or metadata bytes differ')
                    state = None if scheduler is None else scheduler.snapshot()
                    if state is not None and (state['active_jobs'] or state['retained_operations']
                                              or state['peak_active_operations'] > workers):
                        raise AssertionError('Shared CPU work/credits did not settle inside its budget')
                    record = dict(pattern=pattern, backend=backend, repetition=repetition,
                        seconds=seconds, exact_payload_index_parity=True, output_sha256=hashes,
                        plan_workers=plan.workers, plan_requested_workers=plan.requested_workers,
                        plan_workspace_bytes=plan.workspace_bytes,
                        numeric_publication=publication_state, shared_scheduler=state, **rss.record())
                    report['measurements'].append(record)
                    print(json.dumps(record), flush=True)
                finally:
                    storage.write_blocks = current_writer
                    evidence.shutdown_confidence_publication(raise_errors=False)
                    evidence.configure_confidence_evidence(None, enabled=False)
                    shutdown_native_pull_scheduler(scheduler)
        current_hashes = [[hashlib.sha256(memoryview(values).cast('B')).hexdigest()
                          for values in fixture] for fixture in fixtures]
        if current_hashes != input_hashes:
            raise AssertionError('Borrowed input bytes changed during capture')
        old, new = (statistics.median(timings[name]) for name in ('legacy', 'shared'))
        report.setdefault('summary', []).append(dict(pattern=pattern, seconds=timings,
            median_legacy_seconds=old, median_shared_seconds=new, speedup=old/new,
            inputs_unchanged=True, input_sha256=input_hashes))
        del fixtures, mask, scores, active, boxes
    report['source_unchanged'] = source_pins() == pins
    if not report['source_unchanged']:
        raise RuntimeError('Capture source changed during paired measurements')
    (output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
