"""Bounded real-GPU crop-retry smoke from one predeclared original family.

Retained raw contacts select the fixture before new inference. They are not an
accuracy reference. Live model outputs alone trigger the production retry;
absence of a trigger or admission refusal is reported without another search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import smoke_sam_extrapolation as shared


def pins():
    return {**shared.source_pins(), str(Path(__file__).resolve().relative_to(REPO)): shared.sha(__file__)}


def prepare(args):
    import numpy as np
    from XTA.sam_evidence import SamEvidenceBundle, fingerprint
    from XTA.sam_bridge_planning import SamPlanningLimits
    from XTA.sam_interpolation import prepare_sam_interpolation_pass, _prepare_sam_group_retry, _retry_work
    from XTA.sam_crop_retry import SamCropRetryPolicy, SamCropRetryController, raw_crop_boundary_contacts, merge_crop_contacts
    from XTA.geometry import get_view_infos
    from XTA.lta_sam import resolve_local_sam_bundle
    bundle = SamEvidenceBundle.open(args.evidence)
    group = bundle.groups[args.group_id]
    if not group.get('complete', True) or len(group['endpoints']) != 2:
        raise ValueError('Retry smoke requires one complete original two-anchor family')
    shape = tuple(bundle.scope['shape_tyx'])
    if args.images.stat().st_size != int(np.prod(shape)):
        raise ValueError('Retained native images differ from the declared uint8 TYX shape')
    image_sha = shared.sha(args.images)
    if bundle.scope.get('source_image_sha256') != image_sha:
        raise ValueError('Retained family evidence and native image identity disagree')
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_path = args.output / 'original_detector_family.npy'
    observations = np.lib.format.open_memmap(baseline_path, mode='w+', dtype='uint8', shape=shape)
    observations[:] = 0
    y0, x0, y1, x1 = group['context_bbox_yx']
    contacts = []
    with bundle.reader() as reader:
        for node in group['endpoints']:
            observations[node['frame_index'], y0:y1, x0:x1] |= reader.group_mask(
                group['group_id'], 'endpoint:' + node['observation_id'])
        for run in bundle.runs.values():
            if run['group_id'] == group['group_id']:
                for frame in run['expected_frames']:
                    contacts.append(raw_crop_boundary_contacts(reader.raw_mask(run['run_id'], frame),
                        group['context_bbox_yx'], shape[1:]))
    observations.flush()
    view = get_view_infos(*shape, cartesian_views=('transverse',))[0]
    source_policy = {'sam_bridge_policy': {'max_group_bytes': args.allowance_mib * 1024**2}}
    limits = SamPlanningLimits(max_group_bytes=args.allowance_mib * 1024**2,
                              max_total_contract_bytes=args.allowance_mib * 1024**2)
    settings = dict(gap_distance=15, min_radius=3., search_angle_deg=30.,
                    interpolation_walk_back=0, interpolation_candidates=1, interpolation_passes=1)
    default_diagnostic = None
    try:
        prepare_sam_interpolation_pass(observations, view=view, scope='adaptive-smoke/default-diagnostic',
                                      crop_mode='whole', **settings)
    except Exception as error:
        default_diagnostic = dict(status='refused', error_type=type(error).__name__, reason=str(error))
    prepared = prepare_sam_interpolation_pass(observations, view=view, scope='adaptive-smoke',
        crop_mode='whole', policy=source_policy, planner_limits=limits, **settings)
    if len(prepared.runs) != 2 or len(prepared.groups) != 1:
        raise ValueError('Predeclared family did not yield exactly two independent original-seed runs')
    initial = prepared.groups[0]
    if tuple(initial.context_bbox_yx) != tuple(group['context_bbox_yx']):
        raise ValueError('Current original crop differs from the predeclared retained contact fixture')
    work = _retry_work(prepared)[initial.group_id]
    if work[1] != 18:
        raise ValueError('Approved smoke is bounded to two complete nine-frame original sessions')
    retry_policy = SamCropRetryPolicy(enabled=True)
    controller = SamCropRetryController(retry_policy, baseline_pixel_frames=work[0],
        baseline_tracker_frames=work[1], largest_original_group_pixel_frames=work[0],
        largest_original_group_tracker_frames=work[1])
    candidates = {}
    def candidate(box):
        if box not in candidates:
            candidates[box] = _prepare_sam_group_retry(prepared, initial, box,
                retry_policy=retry_policy, policy=source_policy, resource_profile=None, worker_count=1)
        return candidates[box]
    seed_identity = fingerprint(dict(snapshot=prepared.observation_snapshot_sha256,
        seeds=[(run.run_id, run.seed_ids) for run in prepared.runs]))
    interval_identity = fingerprint([(run.run_id, run.expected_frames, run.direction) for run in prepared.runs])
    decision = controller.reserve_retry(initial.group_id, crop_bbox_yx=initial.context_bbox_yx,
        canvas_shape_yx=shape[1:], frame_count=work[1], seed_identity=seed_identity,
        interval_identity=interval_identity, contacts=merge_crop_contacts(contacts),
        available_memory_bytes=2*1024**3, memory_estimator=lambda box: candidate(box)[1],
        work_estimator=lambda box: sum(value[0] for value in _retry_work(candidate(box)[0]).values()),
        tracker_frame_estimator=lambda box: sum(value[1] for value in _retry_work(candidate(box)[0]).values()))
    if not decision.retry:
        raise ValueError('Predeclared genuine-contact fixture was not admitted by the bounded retry controller')
    model = resolve_local_sam_bundle(args.model)
    protocol = dict(schema='xta.sam_crop_retry_real_smoke/1', functional_smoke=True,
        benchmark=False, accuracy_claim=False, production_default_quality_claim=False,
        scope='Only the predeclared original two-anchor family; other detector families are outside this functional fixture',
        retained_evidence_path=str(bundle.directory), retained_evidence_fingerprint=bundle.evidence_fingerprint,
        retained_group_id=group['group_id'], retained_crop_bbox_yx=list(group['context_bbox_yx']),
        original_seed_descriptors=[dict(observation_id=node['observation_id'], frame_index=node['frame_index'])
                                  for node in group['endpoints']],
        observations_path=str(baseline_path.resolve()), observations_sha256=shared.sha(baseline_path),
        image_path=str(args.images.resolve()), image_sha256=image_sha, shape_tyx=list(shape),
        source_frame_start=bundle.scope.get('source_frame_start'), settings=settings,
        source_policy=source_policy, planner_allowance_mib=args.allowance_mib,
        explicit_research_resource_allowance=True, default_allowance_diagnostic=default_diagnostic,
        cpu_contact_admission=decision.record, retry_policy=retry_policy.to_dict(),
        maximum_total_tracker_frames=36, maximum_retries_per_original_family=1,
        model_path=str(args.model.resolve()), checkpoint_path=str(model.checkpoint_path),
        checkpoint_sha256=shared.sha(model.checkpoint_path), model_version=model.model_version,
        gpu_lock_path=str(args.gpu_lock.resolve()), device=args.device,
        feature_cache_mib=args.feature_cache_mib, source_sha256=pins(), invocation=sys.argv,
        stop_rule='If live initial masks have no retry trigger, report no_trigger without another fixture or forced contact')
    protocol['protocol_sha256'] = shared.identity(protocol)
    shared.write(args.output / 'protocol.json', protocol)
    print(json.dumps(dict(protocol=str(args.output / 'protocol.json'), admission=decision.record)), flush=True)


def load_protocol(path):
    protocol = json.loads(Path(path).read_text('utf-8'))
    if shared.identity({key: value for key, value in protocol.items() if key != 'protocol_sha256'}) != protocol['protocol_sha256']:
        raise ValueError('Predeclared retry protocol changed')
    for path_key, hash_key in (('observations_path', 'observations_sha256'), ('image_path', 'image_sha256'),
                              ('checkpoint_path', 'checkpoint_sha256')):
        if shared.sha(protocol[path_key]) != protocol[hash_key]:
            raise ValueError('Retry input/model changed: ' + path_key)
    if pins() != protocol['source_sha256']:
        raise ValueError('Source changed; prepare again after final source freeze')
    return protocol


def verify_attempt(bundle, observations, protocol):
    import numpy as np
    if len(bundle.runs) != 2:
        raise AssertionError('Original family attempt changed its independent two-seed inventory')
    records = []
    with bundle.reader() as reader:
        for run in bundle.runs.values():
            if len(run['seed_ids']) != 1 or len(run['expected_frames']) != 9:
                raise AssertionError('Retry changed its frozen single seed or full interval')
            group = bundle.groups[run['group_id']]
            node = next(node for node in group['endpoints'] if node['observation_id'] == run['seed_ids'][0])
            y0, x0, y1, x1 = group['context_bbox_yx']
            seed = np.zeros(observations.shape[1:], bool)
            seed[y0:y1, x0:x1] = reader.group_mask(group['group_id'], 'endpoint:' + run['seed_ids'][0])
            np.testing.assert_array_equal(seed, observations[node['frame_index']] != 0,
                                          err_msg='Retry conditioning seed differs from the frozen original detector mask')
            adapter = run['runtime_receipt'].get('adapter_receipt', {})
            if adapter.get('seed_roundtrip_exact') is not True or adapter.get('raw_observation_complete') is not True:
                raise AssertionError('Real SDK seed/raw callback integrity is not complete')
            if (run['runtime_receipt']['sam_model']['checkpoint_sha256'] != protocol['checkpoint_sha256']
                    or run['runtime_receipt']['sam_model']['model_version'] != protocol['model_version']):
                raise AssertionError('Real runtime checkpoint differs from its sealed identity')
            records.append(dict(run_id=run['run_id'], seed_ids=list(run['seed_ids']),
                seed_frame=node['frame_index'], expected_frames=list(run['expected_frames']),
                injected_frames=list(run['injected_frames']), context_bbox_yx=list(group['context_bbox_yx']),
                seed_roundtrip_exact=True, raw_callback_complete=True))
    return records


def run(args):
    import numpy as np
    from XTA.geometry import get_view_infos
    from XTA.sam_bridge_planning import SamPlanningLimits
    from XTA.sam_integration import SamInterpolationContext
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.runtime import close_memmap_array_without_flush
    from tools.diagnose_sam_interpolation import resource_monitor
    protocol = load_protocol(args.protocol)
    extrapolation = shared.load_protocol(args.extrapolation_protocol) if args.extrapolation_protocol else None
    if extrapolation and (extrapolation['gpu_lock_path'] != protocol['gpu_lock_path']
                          or extrapolation['device'] != protocol['device']):
        raise ValueError('Combined smokes must share the same explicitly owned GPU and lock')
    args.output.mkdir(parents=True, exist_ok=False)
    shared.write(args.output / 'protocol.json', protocol)
    observations = np.load(protocol['observations_path'], mmap_mode='r', allow_pickle=False)
    images = np.memmap(protocol['image_path'], dtype='uint8', mode='r', shape=tuple(protocol['shape_tyx']))
    view = get_view_infos(*observations.shape, cartesian_views=('transverse',))[0]
    state = dict(contexts_settled=True)
    report = dict(schema='xta.sam_crop_retry_real_smoke_result/1', status='running',
        functional_smoke=True, benchmark=False, accuracy_claim=False,
        explicit_research_resource_allowance=True, protocol_sha256=protocol['protocol_sha256'],
        source_sha256_before=pins(), observations_sha256_before=shared.sha(protocol['observations_path']))
    try:
        with shared.gpu_lock(protocol['gpu_lock_path'], state, 'sam-bounded-adaptive-and-extrapolation-smokes') as owner, \
                resource_monitor(args.output / 'resources.json', protocol['device']):
            report['gpu_lock_owner'] = owner
            if extrapolation:
                extrapolation_result = shared.run(SimpleNamespace(protocol=args.extrapolation_protocol,
                    output=args.output / 'extrapolation'), held_gpu_lock=owner, lock_state=state)
                report['default_off_extrapolation_result'] = str(args.output / 'extrapolation' / 'result.json')
                if extrapolation_result['status'] != 'passed':
                    raise AssertionError('Combined default-off extrapolation smoke failed')
            context = SamInterpolationContext(model_path=protocol['model_path'],
                device_ids=(protocol['device'],), detector_device_ids=(),
                temp_dir=args.output / 'runtime', evidence_root=args.output / 'evidence',
                source_volume=images, source_identity=protocol['image_sha256'],
                source_grid_shape=images.shape, crop_mode='whole', adaptive_crop=True,
                policy=protocol['source_policy'], bundle_identity=protocol['checkpoint_sha256'],
                feature_cache_mib=protocol['feature_cache_mib'])
            state['contexts_settled'] = False
            merged = None
            try:
                merged, stats, _ = context.interpolate(observations, view=view, scope='adaptive-smoke',
                    work_dir=args.output / 'retained',
                    planner_limits=SamPlanningLimits(max_group_bytes=protocol['planner_allowance_mib']*1024**2,
                        max_total_contract_bytes=protocol['planner_allowance_mib']*1024**2),
                    **protocol['settings'])
                ledger = stats.get('sam_crop_retry')
                if not ledger or len(ledger['attempts']) != 1:
                    raise AssertionError('Production orchestration did not retain the bounded family retry ledger')
                original = SamEvidenceBundle.open(ledger['initial_evidence_path'])
                if list(next(iter(original.groups.values()))['context_bbox_yx']) != protocol['retained_crop_bbox_yx']:
                    raise AssertionError('Live original crop differs from its predeclared retained geometry')
                original_records = verify_attempt(original, observations, protocol)
                attempt = next(iter(ledger['attempts'].values()))
                report.update(stats=stats, retry_ledger=ledger, original_runtime_records=original_records)
                if attempt.get('status') == 'succeeded':
                    retry = SamEvidenceBundle.open(attempt['completion_detail']['evidence_path'])
                    retry_records = verify_attempt(retry, observations, protocol)
                    original_by_seed = {tuple(record['seed_ids']): record for record in original_records}
                    for record in retry_records:
                        old = original_by_seed[tuple(record['seed_ids'])]
                        if old['expected_frames'] != record['expected_frames'] or old['injected_frames'] != record['injected_frames']:
                            raise AssertionError('Live retry changed its original seed frame or complete interval')
                    if ledger['charged_tracker_frames'] != 18:
                        raise AssertionError('Live retry work was not charged for both full original sessions')
                    report.update(status='passed', retry_runtime_records=retry_records,
                                  genuine_live_retry_triggered=True)
                else:
                    report.update(status='no_trigger' if attempt['reason'] == 'no_internal_crop_contact' else 'refused',
                                  genuine_live_retry_triggered=False)
                report['total_tracker_frames_charged'] = 18 + ledger['charged_tracker_frames']
                if shared.sha(protocol['observations_path']) != protocol['observations_sha256']:
                    raise AssertionError('Frozen original detector masks changed during adaptive tracking')
            finally:
                try:
                    if merged is not None and merged is not observations:
                        close_memmap_array_without_flush(merged)
                        merged = None
                finally:
                    try:
                        context.close()
                    finally:
                        state['contexts_settled'] = context._runtime is None and not context._leases
            report.update(source_sha256_after=pins(), observations_sha256_after=shared.sha(protocol['observations_path']),
                          gpu_used=True, worker_model_owners_retired=state['contexts_settled'])
            if report['source_sha256_before'] != report['source_sha256_after']:
                raise AssertionError('Production Python source changed during live retry proof')
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      worker_model_owners_retired=state['contexts_settled'])
        shared.write(args.output / 'result.json', report)
        raise
    shared.write(args.output / 'result.json', report)
    print(json.dumps(dict(status=report['status'], total_tracker_frames_charged=report['total_tracker_frames_charged'],
                          benchmark=False, accuracy_claim=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    p = stages.add_parser('prepare')
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--group-id', required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--gpu-lock', type=Path, required=True)
    p.add_argument('--allowance-mib', type=int, default=512)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--feature-cache-mib', type=int, default=512)
    p.add_argument('--output', type=Path, required=True)
    p = stages.add_parser('run')
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--extrapolation-protocol', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.stage](args)


if __name__ == '__main__':
    main()
