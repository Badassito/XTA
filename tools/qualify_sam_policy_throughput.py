"""Qualify full fixed-evidence SAM policy equivalence and bounded CPU throughput.

Default execution only inspects the input and admission plan. ``--check-only``
runs complete serial/parallel correctness checks without a throughput claim.
``--benchmark`` additionally requires a coordinated quiet window and at least
60 seconds of CPU heatsoak. This tool never loads a model or uses a GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GIB = 1024**3
BASE_BYTES = 4 * GIB
EXTRA_BYTES = 3 * GIB
CACHE_BYTES = 32 * 1024**2
SOURCE_FILES = (
    'XTA/sam_policy.py', 'XTA/sam_mask_reader.py', 'XTA/sam_filtering.py', 'XTA/sam_branch_selection.py',
    'XTA/sam_evidence.py', 'XTA/sam_cyclic.py', 'XTA/sam_resources.py',
    'XTA/interpolation.py', 'tools/qualify_sam_policy_throughput.py',
)
AUDIT_ONLY_FILES = ('XTA/sam_interpolation.py', 'XTA/sam_tracker_runtime.py')
IDENTITY_PATHS = {
    ('policy_hash',), ('policy_implementation_sha256',),
    ('component_filter_implementation_sha256',), ('reader_implementation_sha256',),
    ('cyclic_quality_implementation_sha256',), ('selection_identity',),
    ('mask_filter', 'implementation_sha256'), ('mask_filter', 'sha256'),
    ('reader_cache', 'implementation_sha256'),
    ('branch_selection', 'implementation_sha256'), ('branch_selection', 'sha256'),
}
READER_COUNTERS = frozenset(('max_cache_bytes', 'cache_bytes', 'peak_cache_bytes',
    'cache_hits', 'cache_misses', 'cache_evictions', 'oversized_products',
    'mask_decodes', 'filter_computations', 'effective_candidate_computations',
    'filter_spec_validations', 'integrity_checks', 'packed_boundary_contact_scans',
    'compact_filter_expansions', 'compact_filter_parent_hits', 'compact_filter_parent_exports',
    'compact_cache_bytes', 'peak_compact_cache_bytes'))
PROFILE_FIELDS = frozenset(('scope_id', 'status', 'base_requested_bytes',
    'base_charged_bytes', 'reserved_extra_bytes', 'pool_capacity_bytes',
    'physical_headroom_bytes', 'worker_count', 'assigned_contract_bytes',
    'assigned_live_contract_bytes', 'assigned_topology_bytes', 'assigned_plane_bytes',
    'assigned_session_cpu_bytes', 'assigned_cpu_wave_bytes', 'assigned_cpu_wave_base_bytes',
    'assigned_cpu_wave_extra_bytes', 'base_fixed_allowance_bytes', 'base_non_cpu_allowance_bytes',
    'base_cpu_wave_nominal_bytes', 'base_cpu_wave_physical_clamp_bytes',
    'other_promised_bytes_at_admission', 'cpu_wave_physical_residual_bytes',
    'base_known_dense_and_other_work_bytes', 'lease_id', 'resource_implementation_sha256',
    'profile_id'))
EFFECTIVE_FIELDS = frozenset(('assigned_contract_bytes', 'assigned_live_contract_bytes',
    'assigned_topology_bytes', 'assigned_plane_bytes', 'assigned_session_cpu_bytes',
    'assigned_cpu_wave_bytes', 'resource_implementation_sha256'))
EXECUTION_FIELDS = frozenset(('schema', 'requested_workers', 'parallel_credit_bytes',
    'lane_cache_bytes', 'workspace_rule', 'fallback_reason', 'peak_pending_runs',
    'peak_charged_bytes', 'parallel_group_count', 'serial_group_count', 'parallel_run_count',
    'oversized_serial_runs', 'maximum_run_charge_bytes', 'wall_seconds', 'serial_reasons',
    'reader_totals', 'branch_metadata'))
BRANCH_METADATA_FIELDS = frozenset(('index_admission', 'parallel_credit_rule',
    'peak_retained_index_bytes', 'peak_simultaneous_index_bytes',
    'minimum_effective_parallel_credit_bytes', 'overlay_snapshot_count', 'full_merge_fallback_count'))
_MISSING = object()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(names=SOURCE_FILES):
    return {name: file_digest(REPO / name) for name in names}


def assert_source_hashes(expected):
    actual = source_hashes(tuple(expected))
    if actual != expected:
        raise RuntimeError('Qualification source changed during fixed-evidence replay')


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _valid_execution(value):
    return (isinstance(value, dict) and set(value) <= EXECUTION_FIELDS
        and set(value.get('reader_totals', {})) <= READER_COUNTERS
        and set(value.get('serial_reasons', {})) <= {
            'worker_hint_serial', 'single_run_group', 'insufficient_parallel_credit'}
        and ('branch_metadata' not in value or _valid_branch_metadata(value['branch_metadata'])))


def _valid_branch_metadata(value):
    return (isinstance(value, dict) and set(value) == BRANCH_METADATA_FIELDS
        and value['index_admission'] == 'simultaneous_prefix_chunk_candidate_indexes_within_unused_topology_credit'
        and value['parallel_credit_rule'] == 'parallel_credit_bytes_minus_retained_prefix_index_bytes'
        and all(type(value[key]) is int and value[key] >= 0 for key in BRANCH_METADATA_FIELDS
                if key not in ('index_admission', 'parallel_credit_rule')))


def allowed_difference(path, before, after, allow_implementation_changes=True):
    """Exact operational/source-identity paths; never blanket SHA/reason removal."""
    if path in IDENTITY_PATHS:
        if not allow_implementation_changes and path != ('selection_identity',):
            return False
        return all(value is _MISSING or _sha(value) for value in (before, after))
    if len(path) == 2 and path[0] == 'reader_cache' and path[1] in READER_COUNTERS:
        return True
    if path == ('selection_resources', 'effective_resource_identity'):
        return _sha(before) and _sha(after)
    if path == ('branch_selection', 'max_group_bytes'):
        return all(type(value) is int and value > 0 for value in (before, after))
    if len(path) == 3 and path[:2] == ('selection_resources', 'effective_budgets'):
        return path[2] in {'topology_bytes', 'plane_bytes'}
    if len(path) == 3 and path[:2] == ('selection_resources', 'live_profile'):
        return path[2] in PROFILE_FIELDS
    if len(path) == 4 and path[:3] == ('selection_resources', 'live_profile', 'effective_budgets'):
        return path[3] in EFFECTIVE_FIELDS
    if path[:2] == ('selection_resources', 'intrinsic_measurements'):
        if len(path) == 2:
            value = after if before is _MISSING else before if after is _MISSING else None
            return _valid_execution(value)
        if path[2] not in EXECUTION_FIELDS:
            return False
        if path[2] == 'branch_metadata':
            return len(path) == 3 and all(value is _MISSING or _valid_branch_metadata(value)
                                         for value in (before, after))
        if path[2] == 'reader_totals':
            return len(path) == 4 and path[3] in READER_COUNTERS
        if path[2] == 'serial_reasons':
            return len(path) == 4 and path[3] in {
                'worker_hint_serial', 'single_run_group', 'insufficient_parallel_credit'}
        return len(path) == 3
    return False


def compare_selections(reference, candidate, *, allow_implementation_changes=True):
    """Compare every receipt value, recording each individually allowed delta."""
    # Compare JSON types, not NumPy/Python scalar implementation classes.
    reference = json.loads(json.dumps(reference, allow_nan=False))
    candidate = json.loads(json.dumps(candidate, allow_nan=False))
    mismatches, operational = [], []

    def visit(before, after, path=()):
        if (not isinstance(before, (dict, list)) and not isinstance(after, (dict, list))
                and before is not _MISSING and after is not _MISSING
                and type(before) is type(after) and before == after):
            return
        if allowed_difference(path, before, after, allow_implementation_changes):
            operational.append({'path': list(path),
                'before': '<missing>' if before is _MISSING else before,
                'after': '<missing>' if after is _MISSING else after})
            return
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                visit(before.get(key, _MISSING), after.get(key, _MISSING), path + (key,))
        elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
            for index, (left, right) in enumerate(zip(before, after)):
                visit(left, right, path + (str(index),))
        else:
            mismatches.append({'path': list(path),
                'before': '<missing>' if before is _MISSING else before,
                'after': '<missing>' if after is _MISSING else after})

    visit(reference, candidate)
    return dict(exact_quality=not mismatches, mismatch_count=len(mismatches),
                mismatches=mismatches, allowed_differences=operational)


def policy_for_reference(selection):
    settings = dict(selection['resolved_policy'])
    # These were inherited production defaults, not intentional source-policy
    # caps. Preserve that declaration and mint fresh operational replay credit.
    resource = selection['selection_resources']
    for name, flag in (('max_group_bytes', 'explicit_topology_cap'),
                       ('rescue_max_plane_bytes', 'explicit_plane_cap')):
        if not resource.get(flag, False):
            settings.pop(name, None)
    return {'sam_bridge_policy': settings}


def frozen_mode_for_reference(selection):
    dependencies = selection.get('dependencies', {})
    status = dependencies.get('status')
    if status not in {'fixed_proposal_replay', 'frozen_evidence_diagnostic'}:
        raise ValueError('Reference dependency replay mode is not supported')
    if dependencies.get('current_snapshot_verified') or dependencies.get('fresh_pipeline_equivalent'):
        raise ValueError('Offline qualification cannot invent current upstream snapshot verification')
    return status == 'frozen_evidence_diagnostic'


def inspect_demands(bundle, selection, workers=8, cache_bytes=CACHE_BYTES):
    from XTA.sam_policy import _measurement_charge, resolve_sam_bridge_policy

    if (set(selection['run_receipts']) != set(bundle.runs)
            or set(selection['group_receipts']) != set(bundle.groups)
            or selection['evidence_fingerprint'] != bundle.evidence_fingerprint
            or set(selection['selected_run_ids']) - set(bundle.runs)):
        raise ValueError('Reference selection does not cover the complete evidence inventory')
    resolved = resolve_sam_bridge_policy(policy_for_reference(selection),
                                        generation_mode=bundle.scope.get('sam_crop_mode', 'whole'))
    frozen_mode_for_reference(selection)
    if resolved != selection['resolved_policy']:
        raise ValueError('Reference quality settings cannot be reproduced exactly')
    topologies, charges = {}, {}
    for key, group in bundle.groups.items():
        if not group.get('complete', True) or group.get('status') in {'incomplete', 'unresolved', 'invalid'}:
            continue
        y0, x0, y1, x1 = map(int, group['context_bbox_yx'])
        shape = (len(group['frame_indices']), y1-y0, x1-x0)
        if resolved['branch_aware_selection']:
            from XTA.sam_branch_selection import branch_workspace_bytes
            topologies[key] = branch_workspace_bytes(shape)
        else:
            topologies[key] = math.prod(shape)*16
    for key, run in bundle.runs.items():
        charges[key] = _measurement_charge(bundle, bundle.groups[run['group_id']], key, cache_bytes)
    return dict(group_count=len(bundle.groups), run_count=len(bundle.runs),
        selected_run_count=len(selection['selected_run_ids']), assessed_group_count=len(topologies),
        required_topology_bytes=max(topologies.values(), default=0),
        largest_topology_group=max(topologies, key=topologies.get) if topologies else None,
        maximum_run_charge_bytes=max(charges.values(), default=0), requested_workers=int(workers),
        reader_cache_bytes=int(cache_bytes), base_bytes=BASE_BYTES, proposed_extra_bytes=EXTRA_BYTES,
        proposed_pool_capacity_bytes=BASE_BYTES+EXTRA_BYTES,
        physical_bytes_needed_for_full_grant=BASE_BYTES+2*EXTRA_BYTES,
        saved_cluster_resource_receipt_is_permission=False)


def input_inventory(evidence, selection_path):
    parent = evidence.parent
    files = [evidence / name for name in ('manifest.json', 'index.json', 'masks.bin')]
    files.append(selection_path)
    if (parent / 'generation.json').exists():
        files.append(parent / 'generation.json')
    for store in sorted(parent.glob('sam_bridge_pass*_*.cvol')):
        files.extend(store / name for name in ('meta.json', 'index.bin', 'chunks.bin'))
    return {str(path.resolve()): dict(bytes=path.stat().st_size, sha256=file_digest(path)) for path in files}


def assert_inputs_unchanged(expected):
    for name, record in expected.items():
        path = Path(name)
        if path.stat().st_size != record['bytes'] or file_digest(path) != record['sha256']:
            raise ValueError(f'Immutable qualification input changed: {name}')


class _ReplayPool:
    def __init__(self, capacity=BASE_BYTES+EXTRA_BYTES):
        self.capacity, self.in_use = int(capacity), 0
        self.condition = threading.Condition()


@contextmanager
def admitted_profile(required_bytes):
    from XTA.sam_resources import admit_sam_parent_resources

    pool = _ReplayPool()
    # worker_count describes an SDK allowance; this replay has no SDK workers.
    # Intrinsic measurement threads receive a separate bounded hint.
    with admit_sam_parent_resources(pool, BASE_BYTES, 'fixed_evidence_policy_replay',
                                    worker_count=1) as profile:
        if profile.assigned_topology_bytes < required_bytes or not profile.has_extra_credit:
            raise MemoryError('Actual local admission cannot assess every saved group; no partial qualification')
        yield profile


def validate_execution(selection, requested_workers):
    resources = selection['selection_resources']
    execution = resources['intrinsic_measurements']
    if not _valid_execution(execution):
        raise ValueError('Unknown intrinsic measurement operational fields')
    if (execution['requested_workers'] != requested_workers
            or execution['peak_pending_runs'] > requested_workers
            or execution['peak_charged_bytes'] > execution['parallel_credit_bytes']):
        raise ValueError('Parallel measurement exceeded its admitted worker/byte bounds')
    metadata = execution.get('branch_metadata')
    if metadata is not None and (metadata['minimum_effective_parallel_credit_bytes'] > execution['parallel_credit_bytes']
            or metadata['peak_simultaneous_index_bytes'] > resources['effective_budgets']['topology_bytes']):
        raise ValueError('Branch metadata exceeded its admitted index/measurement bounds')
    refused = [key for key, row in selection['group_receipts'].items()
               if row['status'] == 'not_assessed_resource_refused']
    if refused:
        raise MemoryError('Replay refused saved groups instead of assessing the full inventory')


def compare_saved_bridges(bundle, selection, parent, cache_bytes=CACHE_BYTES):
    import numpy as np
    from XTA.interpolation import RawBBoxMaskStore
    from XTA.sam_evidence import selected_native_plane, evidence_frame_geometry

    geometry = evidence_frame_geometry(bundle)
    pairs = {(int(run['pass_index']), str(run['direction'])) for run in bundle.runs.values()}
    records = []
    for pass_index, direction in sorted(pairs):
        path = parent / f'sam_bridge_pass{pass_index:02d}_{direction}.cvol'
        saved = RawBBoxMaskStore.open(path, mmap_payload=True)
        try:
            shape = tuple(saved.shape)
            if shape != tuple(geometry['native_shape_tyx']):
                raise ValueError('Saved directional bridge shape differs from native evidence geometry')
            active = set(np.flatnonzero(saved.index['kind']).tolist())
            for key in selection['selected_run_ids']:
                run = bundle.runs[key]
                if int(run['pass_index']) != pass_index or run['direction'] != direction:
                    continue
                addresses = geometry['groups'][run['group_id']]['addresses']
                for frame, mask_key in run['candidate_mask_keys'].items():
                    # Expanded branch owners can write on a plane whose old
                    # clipped candidate packet is empty. Compare every selected
                    # owner's covered plane for branch-aware receipts.
                    if selection.get('branch_selection') is not None or bundle.records[mask_key]['foreground']:
                        frame = int(frame)
                        active.add(int(addresses[frame]['native_index']) if addresses is not None else frame)
            old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
            header = json.dumps(dict(shape=shape, pass_index=pass_index, direction=direction), sort_keys=True).encode()
            old_hash.update(header); new_hash.update(header)
            changed = old_foreground = new_foreground = 0
            with bundle.reader(max_cache_bytes=cache_bytes) as reader:
                bound_selection = selection
                if hasattr(reader, 'filter_snapshot'):
                    snapshot = reader.filter_snapshot(selection)
                    bound_selection = dict(selection, mask_filter=snapshot)
                    bound_selection.pop('branch_selection', None)
                for frame in sorted(active):
                    old = saved.decode_slice(frame, dtype=bool)
                    new = selected_native_plane(reader, bound_selection, frame, direction=direction,
                                                pass_index=pass_index, shape_yx=shape[1:])
                    changed += int(np.count_nonzero(old != new))
                    old_foreground += int(np.count_nonzero(old)); new_foreground += int(np.count_nonzero(new))
                    for digest, mask in ((old_hash, old), (new_hash, new)):
                        digest.update(struct.pack('<Q', frame))
                        digest.update(np.packbits(mask.reshape(-1), bitorder='little').tobytes())
            records.append(dict(pass_index=pass_index, direction=direction, shape=list(shape),
                compared_active_planes=len(active), proven_both_empty_planes=shape[0]-len(active),
                changed_pixel_bytes=changed, saved_foreground=old_foreground, replay_foreground=new_foreground,
                saved_canonical_pixel_sha256=old_hash.hexdigest(), replay_canonical_pixel_sha256=new_hash.hexdigest(),
                exact=changed == 0 and old_hash.digest() == new_hash.digest(),
                digest_domain='native binary pixel bytes, packed little; shape/owners/active indices bound; other planes proven empty'))
        finally:
            saved.close()
    return dict(exact=all(row['exact'] for row in records), directional_bridges=records)


@contextmanager
def rss_monitor():
    import psutil

    process = psutil.Process()
    stop = threading.Event()
    record = dict(start_rss_bytes=int(process.memory_info().rss), peak_rss_bytes=0)
    def sample():
        while not stop.is_set():
            record['peak_rss_bytes'] = max(record['peak_rss_bytes'], int(process.memory_info().rss))
            stop.wait(.1)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield record
    finally:
        record['end_rss_bytes'] = int(process.memory_info().rss)
        record['peak_rss_bytes'] = max(record['peak_rss_bytes'], record['end_rss_bytes'])
        stop.set(); thread.join()


def cpu_heatsoak(seconds, workers):
    import numpy as np

    stop, counts = threading.Event(), [0] * workers
    def heat(index):
        a = np.full((512, 512), 0.001, np.float32)
        b, out = a.copy(), np.empty_like(a)
        while not stop.is_set():
            np.matmul(a, b, out=out)
            counts[index] += 1
    threads = [threading.Thread(target=heat, args=(index,)) for index in range(workers)]
    started = time.perf_counter()
    for thread in threads: thread.start()
    try:
        while time.perf_counter()-started < seconds:
            time.sleep(min(1., max(0., seconds-(time.perf_counter()-started))))
    finally:
        stop.set()
        for thread in threads: thread.join()
    return dict(seconds=time.perf_counter()-started, cpu_workers=workers, matmuls_by_worker=counts,
                gpu_used=False, method='CPU float32 matmul, BLAS threads=1 per heatsoak worker')


def environment_receipt():
    import numpy
    import scipy
    import psutil

    process = psutil.Process()
    try:
        affinity = process.cpu_affinity()
    except (AttributeError, NotImplementedError):
        affinity = None
    return dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
        logical_cpu_count=os.cpu_count(), affinity=affinity, memory_total_bytes=psutil.virtual_memory().total,
        numpy=numpy.__version__, scipy=scipy.__version__, psutil=psutil.__version__,
        blas_environment={name: os.environ.get(name) for name in
                          ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')})


def _write_json(path, value):
    if path.exists():
        raise FileExistsError(f'Preserve prior qualification artifact; choose a fresh output: {path}')
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', dest='mode', action='store_const', const='check')
    mode.add_argument('--benchmark', dest='mode', action='store_const', const='benchmark')
    parser.set_defaults(mode='plan')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--quiet-window-confirmed', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.mode == 'benchmark' and (
            not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds < 60
            or not args.quiet_window_confirmed)):
        parser.error('Benchmark needs positive workers, >=60s heatsoak and a root-coordinated quiet window')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '1'
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_policy import select_sam_proposals
    from XTA.sam_resources import physical_sam_headroom

    evidence, output = args.evidence.resolve(), args.output.resolve()
    selection_path = (args.selection or evidence.parent / 'selection.json').resolve()
    if output.is_relative_to(evidence.parent) or evidence.parent.is_relative_to(output):
        raise ValueError('Qualification outputs must not overlap immutable input receipts')
    output.mkdir(parents=True, exist_ok=True)
    sources, producer_before = source_hashes(), source_hashes(AUDIT_ONLY_FILES)
    inputs = input_inventory(evidence, selection_path)
    if selection_path.stat().st_size > 64 * 1024**2:
        raise MemoryError('Reference selection exceeds its bounded receipt budget')
    reference = json.loads(selection_path.read_text(encoding='utf-8'))
    bundle = SamEvidenceBundle.open(evidence)
    plan = inspect_demands(bundle, reference, args.workers)
    plan.update(mode=args.mode, evidence=str(evidence), selection=str(selection_path),
                physical_headroom_bytes=physical_sam_headroom(), source_sha256=sources,
                inputs=inputs, gpu_used=False, allocation_permission='fresh live local lease only',
                started_utc=datetime.now(timezone.utc).isoformat(), environment=environment_receipt(),
                quiet_window_confirmed=bool(args.quiet_window_confirmed))
    _write_json(output / ('plan.json' if args.mode == 'plan' else 'execution_plan.json'), plan)
    print(json.dumps({key: value for key, value in plan.items() if key not in ('inputs', 'source_sha256')}, indent=2), flush=True)
    if args.mode == 'plan':
        assert_inputs_unchanged(inputs); assert_source_hashes(sources)
        return 0
    if plan['required_topology_bytes'] > EXTRA_BYTES:
        raise MemoryError('Proposed local credit cannot cover full saved topology demand')
    heat = None
    if args.mode == 'benchmark':
        heat = cpu_heatsoak(args.heatsoak_seconds, min(args.workers, os.cpu_count() or 1))
    phases = []
    serial = parallel = None
    policy = policy_for_reference(reference)
    frozen = frozen_mode_for_reference(reference)
    try:
        with admitted_profile(plan['required_topology_bytes']) as profile:
            for name, workers in (('serial', 1), ('parallel', args.workers)):
                assert_source_hashes(sources); assert_inputs_unchanged(inputs)
                with rss_monitor() as memory:
                    started = time.perf_counter()
                    selection = select_sam_proposals(bundle, policy, workers=workers,
                        resource_profile=profile, frozen_evidence=frozen, reader_cache_bytes=CACHE_BYTES)
                    elapsed = time.perf_counter()-started
                validate_execution(selection, workers)
                if memory['peak_rss_bytes'] > profile.base_charged_bytes + profile.reserved_extra_bytes:
                    raise MemoryError('Process RSS exceeded the conservatively owned base+extra replay allowance')
                _write_json(output / f'selection_{name}.json', selection)
                comparison = compare_selections(reference, selection)
                bridges = compare_saved_bridges(bundle, selection, evidence.parent)
                _write_json(output / f'comparison_{name}.json', comparison)
                _write_json(output / f'bridges_{name}.json', bridges)
                phases.append(dict(name=name, requested_workers=workers, selection_wall_seconds=elapsed,
                    full_group_count=len(selection['group_receipts']), full_run_count=len(selection['run_receipts']),
                    selected_run_count=len(selection['selected_run_ids']), quality_exact=comparison['exact_quality'],
                    bridge_pixel_bytes_exact=bridges['exact'], live_resource_profile=profile.metadata(),
                    memory=memory,
                    intrinsic_measurements=selection['selection_resources']['intrinsic_measurements']))
                if name == 'serial': serial = selection
                else: parallel = selection
        same = compare_selections(serial, parallel, allow_implementation_changes=False)
        _write_json(output / 'comparison_serial_parallel.json', same)
        assert_inputs_unchanged(inputs); assert_source_hashes(sources)
        exact = same['exact_quality'] and all(row['quality_exact'] and row['bridge_pixel_bytes_exact'] for row in phases)
        result = dict(schema='xta.sam_policy_throughput/1', status='qualified' if exact else 'equivalence_failed',
            mode=args.mode, full_inventory=True, group_count=plan['group_count'], run_count=plan['run_count'],
            gpu_used=False, input_receipts_unchanged=True, executing_sources_unchanged=True,
            source_sha256=sources, producer_before=producer_before, producer_after=source_hashes(AUDIT_ONLY_FILES),
            producer_changes_are_audit_only='Neither generation/runtime file executes in fixed-evidence policy/bridge replay',
            heatsoak=heat, phases=phases, exact_quality_and_bridge_bytes=exact,
            started_utc=plan['started_utc'], finished_utc=datetime.now(timezone.utc).isoformat(),
            environment=plan['environment'], quiet_window_confirmed=plan['quiet_window_confirmed'],
            serial_parallel_quality_exact=same['exact_quality'],
            timing_claim='Coordinated heatsoaked local CPU sanity only' if args.mode == 'benchmark' else 'Correctness-only; not formal throughput',
            parallel_speedup=phases[0]['selection_wall_seconds']/phases[1]['selection_wall_seconds'] if args.mode == 'benchmark' else None)
        _write_json(output / 'qualification.json', result)
        print(json.dumps({key: value for key, value in result.items() if key not in ('phases', 'source_sha256')}, indent=2), flush=True)
        return 0 if exact else 1
    except BaseException as error:
        _write_json(output / 'failure.json', dict(status='infrastructure_or_validation_failure',
            exception=type(error).__name__, message=str(error), completed_phases=phases,
            source_before=sources, source_after=source_hashes(), input_inventory=inputs))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
