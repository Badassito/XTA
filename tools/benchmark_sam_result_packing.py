"""CPU-only actual SAM evidence writer sanity with live packing admission.

Use only in a coordinated CPU window. Retained SAM masks and one explicitly
synthetic stress case are bounded below1GiB. No SDK/model or GPU is started.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from XTA import sam_resources as resources
from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, _plain

GIB = 1024**3
HISTORY = ROOT.parent/'Scratch'/'Data'/'XTA'/'History'
REAL_BUNDLE = HISTORY/'fixtures'/'sam-smoke'/'SAM_Outer_Crop_20261001'/'runs'/'source_590_599'/'whole'/'B1'/'evidence'


def retained_cases():
    bundle = SamEvidenceBundle.open(REAL_BUNDLE)
    eligible = [run for run in bundle.runs.values()
        if run.get('complete') and bundle.groups[run['group_id']].get('complete', True)
        and len(run['observed_frames']) >= 5]
    def pixels(run):
        y0, x0, y1, x1 = bundle.groups[run['group_id']]['context_bbox_yx']
        return (y1-y0)*(x1-x0)
    large = max(eligible, key=pixels)
    cheap = max((run for run in eligible if pixels(run) <= 64*1024), key=pixels)
    cases = []
    for name, selected in (('retained-real-large', large), ('retained-real-cheap', cheap)):
        group = _plain(bundle.groups[selected['group_id']])
        masks = {name: bundle.group_mask(group['group_id'], name) for name in group['mask_keys']}
        raw = {frame: bundle.raw_mask(selected['run_id'], frame) for frame in selected['observed_frames']}
        cases.append(dict(name=name, scope=_plain(bundle.scope), group=group, masks=masks,
            run=_plain(selected), raw=raw, provenance=dict(kind='retained_actual_sam31_masks',
                source_bundle=str(REAL_BUNDLE), source_evidence_fingerprint=bundle.evidence_fingerprint,
                source_payload_sha256=bundle.manifest['files']['masks.bin']['sha256'],
                source_run_id=selected['run_id'], source_frame_start=bundle.scope.get('source_frame_start'),
                scope_shape=bundle.scope.get('shape_tyx'))))
    return cases


def synthetic_case():
    side, frames = 2048, 16
    y, x = np.ogrid[:side, :side]
    blocks = np.random.default_rng(155423).random((side//8, side//8)) < .18
    speckled = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)
    base = (((x-side*.42)**2/(side*.28)**2+(y-side*.54)**2/(side*.34)**2) < 1)
    base |= speckled & ((x % 193) < 117) & ((y % 157) < 96)
    raw = {frame: np.roll(base, frame*3, axis=1) for frame in range(frames)}
    allowed = np.ones((side, side), bool)
    masks = {f'acceptance:{frame}': allowed for frame in range(frames)}
    masks.update({f'write:{frame}': allowed for frame in range(frames)})
    masks.update({'endpoint:A': raw[0], 'endpoint:B': raw[frames-1],
        'evaluation:A': allowed, 'evaluation:B': allowed})
    masks['write:0'] = allowed & ~raw[0]
    masks[f'write:{frames-1}'] = allowed & ~raw[frames-1]
    group = dict(group_id='structured-family', context_bbox_yx=[0, 0, side, side],
        frame_indices=list(range(frames)), complete=True,
        endpoints=[dict(observation_id='A', frame_index=0, canonical_label=1),
                   dict(observation_id='B', frame_index=frames-1, canonical_label=1)],
        edges=[dict(edge_id='E', source_id='A', target_id='B')])
    run = dict(run_id='structured-run', group_id=group['group_id'], direction=1,
        seed_ids=['A'], held_out_ids=['B'], expected_frames=list(range(frames)),
        injected_frames=[0], complete=True, pass_index=1)
    return dict(name='structured-synthetic-2048', scope=dict(shape_tyx=[frames, side, side],
        fixture='structured synthetic shifting binary contours and8pixel blocks'),
        group=group, masks=masks, run=run, raw=raw,
        provenance=dict(kind='structured_synthetic_not_model_output', seed=155423,
            source_shape=[frames, side, side]))


@contextmanager
def actual_packing_scope(case):
    shape = next(iter(case['raw'].values())).shape
    pixels = int(np.prod(shape))
    frames = len(case['run']['expected_frames'])
    pool = SimpleNamespace(capacity=4*GIB, in_use=0,
        condition=threading.Condition(threading.RLock()))
    if resources.physical_sam_headroom() < 4*GIB:
        raise RuntimeError('Actual physical headroom cannot fund this benchmark parent')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'packing-benchmark',
            worker_count=8, base_allowance_bytes=4*GIB) as profile:
        session = resources.cpu_session_bytes(frames, pixels)['estimated_peak_bytes']
        wave = resources.cpu_wave_admission(session, frames*pixels,
            profile.assigned_cpu_wave_bytes, 8)
        with resources.admit_sam_tracker_scope(profile, wave, max_seed_pixels=pixels,
                max_frame_count=frames, evidence_contract_bytes=0) as scope:
            limits = scope.acquire_scope()
            try:
                permit = scope.admit_mask_packing()
                admission = dict(parent_profile=profile.metadata(), wave=wave,
                    tracker_limits=dict(limits), packing_scratch_bytes=permit.scratch_bytes(pixels),
                    packing_cpu_worker_limit=permit.cpu_worker_limit(),
                    direct_masks_have_no_live_planner_contract=True,
                    scheduler_and_sdk_invocation_emulated=True)
                yield permit, admission
            finally:
                scope.release_scope()
    if pool.in_use:
        raise RuntimeError('Packing scope failed to return its real parent/bank credit')


def write_case(case, directory, *, parallel, commit=True):
    from XTA.runtime import runtime_telemetry
    telemetry = runtime_telemetry()
    before = telemetry.snapshot().get('counters', {})
    total_started = time.perf_counter()
    with actual_packing_scope(case) as (permit, admission):
        with SamEvidenceWriter(directory, case['scope']) as writer:
            group_started = time.perf_counter()
            writer.add_group(case['group'], case['masks'])
            group_seconds = time.perf_counter()-group_started
            body_started = time.perf_counter()
            with writer.parallel_packing(permit if parallel else None):
                writer.add_run(case['run'], case['raw'])
            result_seconds = time.perf_counter()-body_started
            commit_started = time.perf_counter()
            bundle = writer.commit() if commit else None
            commit_seconds = time.perf_counter()-commit_started
        # Warm-up uses the actual writer/admission too; its private staging is
        # aborted by the writer's context after all helper work has drained.
    total_seconds = time.perf_counter()-total_started
    after = telemetry.snapshot().get('counters', {})
    counters = {key: value-before.get(key, 0) for key, value in after.items()
        if key.startswith('sam.consumer.')}
    result = dict(mode='parallel' if parallel else 'serial', group_seconds=group_seconds,
        result_packing_and_drain_seconds=result_seconds, commit_seconds=commit_seconds,
        total_writer_and_admission_seconds=total_seconds, counters=counters,
        admission=admission)
    if bundle is not None:
        result.update(evidence_fingerprint=bundle.evidence_fingerprint,
            payload_sha256=bundle.manifest['files']['masks.bin']['sha256'],
            index_sha256=bundle.manifest['files']['index.json']['sha256'],
            payload_bytes=bundle.manifest['files']['masks.bin']['bytes'],
            record_count=len(bundle.records), directory=str(bundle.directory))
    return result, bundle


def verify_decoded(case, bundle):
    for frame, original in case['raw'].items():
        if not np.array_equal(bundle.raw_mask(case['run']['run_id'], frame), original):
            raise RuntimeError('Published raw mask differs from actual input')
        edge_ids = case['run'].get('edge_ids', ())
        if edge_ids:
            write = np.zeros(original.shape, bool)
            for edge in edge_ids:
                write |= case['masks'][f'edge_write:{edge}:{frame}']
        else:
            write = case['masks'][f'write:{frame}']
        if not np.array_equal(bundle.candidate_mask(case['run']['run_id'], frame), original & write):
            raise RuntimeError('Published candidate differs from fixed write domain')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    scratch = (ROOT.parent/'Scratch').resolve(strict=True)
    if not destination.is_relative_to(scratch) or destination == scratch:
        parser.error('Put benchmark artifacts beneath task Scratch')
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'result.json').exists():
        parser.error('This benchmark output already has a retained result')
    cases = retained_cases()+[synthetic_case()]
    unique_arrays = {id(value): value for case in cases for value in
        (*case['masks'].values(), *case['raw'].values())}
    input_bytes = sum(value.nbytes for value in unique_arrays.values())
    if input_bytes >= GIB:
        raise RuntimeError('Benchmark input exceeds its bounded1GiB corpus')
    for value in unique_arrays.values():
        value.setflags(write=False)
    print(json.dumps(dict(stage='input', source_bytes=input_bytes,
        physical_headroom_bytes=resources.physical_sam_headroom(),
        cases=[dict(name=case['name'], shape=list(next(iter(case['raw'].values())).shape),
                    frames=len(case['raw'])) for case in cases])), flush=True)

    # Heat CPU using the same real writer/grants; leave no warm-up bundles.
    deadline = time.perf_counter()+60.
    warmed = 0
    while time.perf_counter() < deadline:
        write_case(cases[-1], destination/'warmup', parallel=bool(warmed%2), commit=False)
        warmed += 1
    print(json.dumps(dict(stage='warmup_complete', seconds=60, writer_runs=warmed)), flush=True)

    report = dict(schema='xta.cpu_result_packing_benchmark/1', cpu_only=True,
        sdk_invocation_emulated=True, actual_live_resource_profiles=True,
        corpus_bytes=input_bytes, warmup_seconds=60, repeats=5, cases=[],
        implementation_sha256={name: hashlib.sha256((ROOT/'XTA'/name).read_bytes()).hexdigest()
            for name in ('sam_evidence.py', 'sam_resources.py')})
    for case in cases:
        rows = []
        expected = None
        for repeat in range(5):
            # Alternate order to avoid always assigning the first run to serial.
            for parallel in ((False, True) if repeat%2 == 0 else (True, False)):
                mode = 'parallel' if parallel else 'serial'
                row, bundle = write_case(case, destination/f'{case["name"]}-{mode}-{repeat}', parallel=parallel)
                identity = tuple(row[key] for key in ('evidence_fingerprint', 'payload_sha256', 'index_sha256'))
                if expected is None:
                    expected = identity
                    verify_decoded(case, bundle)
                elif identity != expected:
                    raise RuntimeError('Serial/parallel packet, metadata or output fingerprint changed')
                if repeat == 0:
                    verify_decoded(case, bundle)
                rows.append(dict(repeat=repeat, **row))
                print(json.dumps(dict(stage='measured', case=case['name'], repeat=repeat,
                    mode=mode, result_seconds=row['result_packing_and_drain_seconds'],
                    total_seconds=row['total_writer_and_admission_seconds'],
                    parallel_submissions=row['counters'].get('sam.consumer.parallel_mask_submissions', 0))), flush=True)
        serial = statistics.median(row['result_packing_and_drain_seconds'] for row in rows if row['mode']=='serial')
        parallel = statistics.median(row['result_packing_and_drain_seconds'] for row in rows if row['mode']=='parallel')
        report['cases'].append(dict(name=case['name'], provenance=case['provenance'], rows=rows,
            median_serial_result_seconds=serial, median_parallel_result_seconds=parallel,
            serial_to_parallel_result_ratio=serial/parallel,
            exact_packets_metadata_fingerprints=True))
        (destination/'result.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    if not any(row['counters'].get('sam.consumer.parallel_mask_submissions', 0) > 0
               for case in report['cases'] for row in case['rows']):
        raise RuntimeError('No measured case exercised parallel mask packing')
    print(json.dumps(dict(stage='done', results=[dict(name=case['name'],
        ratio=case['serial_to_parallel_result_ratio']) for case in report['cases']])), flush=True)


if __name__ == '__main__':
    main()
