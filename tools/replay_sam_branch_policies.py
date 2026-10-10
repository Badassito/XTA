"""Replay focused SAM bridge-quality variants on immutable raw evidence.

This CPU-only tool does not infer, consume labels, or add SDF pixels. Each arm
recomputes measurements and publishes a fresh selection receipt. Allocation
permission comes only from the explicit current workspace argument.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from XTA.sam_evidence import SamEvidenceBundle
from XTA.artifact_archive import read_artifact
from XTA.sam_policy import resolve_sam_bridge_policy, select_sam_proposals

VARIANTS = ('legacy', 'branches', 'paired_context', 'anchor_context', 'anchor_context_censored',
            'anchor_context_radius0')


@contextlib.contextmanager
def trace_selection_phases(path, *, enabled=False):
    """Observe scalar phase timing without changing any selection function."""
    if not enabled:
        yield None
        return
    from XTA import sam_policy, sam_branch_selection
    audit = dict(schema='xta.sam_selection_phase_trace/1',
        wrapper_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), phases={})
    current = dict(group_id=None, contact_calls=0, contact_seconds=0., contact_positive_calls=0)
    wrapped = []
    trace_started = time.perf_counter()
    with Path(path).open('w', encoding='utf-8') as stream:
        def emit(row):
            line = json.dumps(dict(trace_elapsed_seconds=time.perf_counter()-trace_started, **row),
                              sort_keys=True, allow_nan=False)
            stream.write(line+'\n')
            stream.flush()
            print('trace '+line, flush=True)

        def flush_contacts():
            if current['contact_calls']:
                emit(dict(event='contact_group_summary', **current))
            current.update(contact_calls=0, contact_seconds=0., contact_positive_calls=0)

        def observe(module, attribute, phase):
            original = getattr(module, attribute)
            @functools.wraps(original)
            def traced(*args, **kwargs):
                if phase == 'intrinsic':
                    group_id = str(args[1]['group_id'])
                    flush_contacts()
                    current['group_id'] = group_id
                elif phase == 'certificate':
                    group_id = str(args[2])
                elif phase == 'contact':
                    group_id = current['group_id']
                else:
                    group_id = str(args[1])
                record = audit['phases'].setdefault(phase, dict(calls=0, wall_seconds=0., maximum_seconds=0.))
                record['calls'] += 1
                if phase == 'contact' and not current['contact_calls']:
                    emit(dict(event='contact_group_begin', group_id=group_id))
                if phase != 'contact':
                    detail = dict(phase=phase, group_id=group_id, call=record['calls'])
                    if phase == 'intrinsic':
                        detail['run_count'] = len(args[2])
                    elif phase == 'certificate':
                        detail['eligible_edge_count'] = len(args[3])
                    else:
                        detail['run_count'] = len(args[2])
                        detail['edge_subset_count'] = None if kwargs.get('edge_ids') is None else len(kwargs['edge_ids'])
                    emit(dict(event='begin', **detail))
                start = time.perf_counter()
                try:
                    result = original(*args, **kwargs)
                except BaseException:
                    elapsed = time.perf_counter()-start
                    record['wall_seconds'] += elapsed
                    record['maximum_seconds'] = max(record['maximum_seconds'], elapsed)
                    if phase != 'contact':
                        emit(dict(event='failed', elapsed_seconds=elapsed, **detail))
                    raise
                elapsed = time.perf_counter()-start
                record['wall_seconds'] += elapsed
                record['maximum_seconds'] = max(record['maximum_seconds'], elapsed)
                if phase == 'contact':
                    current['contact_calls'] += 1
                    current['contact_seconds'] += elapsed
                    current['contact_positive_calls'] += bool(result)
                    if elapsed >= .25:
                        emit(dict(event='slow_contact', group_id=group_id, elapsed_seconds=elapsed,
                            positive=bool(result), group_contact_call=current['contact_calls']))
                else:
                    extra = {}
                    if phase == 'certificate':
                        extra['certified_edge_count'] = len(result[0]['edges'])
                    elif phase == 'topology':
                        extra['connected_edge_count'] = sum(edge['connected'] for edge in result['edges'])
                    emit(dict(event='end', elapsed_seconds=elapsed, **detail, **extra))
                return result
            setattr(module, attribute, traced)
            wrapped.append((module, attribute, original))
        try:
            observe(sam_policy, '_measure_group_intrinsic', 'intrinsic')
            observe(sam_branch_selection, 'build_connected_edge_selection', 'certificate')
            observe(sam_policy, 'measure_group_topology', 'topology')
            observe(sam_policy, '_pair_contact', 'contact')
            observe(sam_policy, 'measure_family_agreement', 'agreement')
            yield audit
        finally:
            flush_contacts()
            for module, attribute, original in reversed(wrapped):
                setattr(module, attribute, original)
            (Path(path).with_suffix('.summary.json')).write_text(json.dumps(audit, indent=2), encoding='utf-8')


def variant_policy(name, *, generation_mode, max_group_bytes, original=None):
    if name not in VARIANTS:
        raise ValueError(f'Unknown SAM branch replay variant: {name}')
    legacy_version = (int(original['version']) if original is not None
                      else 5 if generation_mode == 'tiled' else 4)
    branch_version = 7 if generation_mode == 'tiled' else 6
    settings = dict(original or {}) if name == 'legacy' else {}
    settings.update(version=legacy_version if name == 'legacy' else branch_version,
                    max_group_bytes=int(max_group_bytes))
    if name != 'legacy':
        settings.update(branch_aware_selection=True, allow_paired_seed_tracks=True,
            guarded_rescue=False, strict_containment=name == 'branches',
            branch_write_domain='edge_write' if name == 'branches' else 'fixed_context',
            min_endpoint_recall=0. if name.startswith('anchor_context') else .5)
        if name == 'anchor_context_radius0':
            settings['component_min_radius'] = 0.
        if name == 'anchor_context_censored':
            settings['branch_crop_boundary_policy'] = 'retain_censored'
    # Freeze all quality settings so the current environment cannot change an arm.
    return resolve_sam_bridge_policy({'sam_bridge_policy': settings}, generation_mode=generation_mode,
        environ={})


def run_replays(evidence, output, *, original_selection=None, variants=VARIANTS[:-1], max_group_mib=256,
                trace_phases=False):
    if isinstance(max_group_mib, bool) or int(max_group_mib) <= 0:
        raise ValueError('The current topology workspace MiB must be positive')
    variants = list(variants)
    if not variants or len(variants) != len(set(variants)) or set(variants)-set(VARIANTS):
        raise ValueError('Replay needs unique supported variants')
    evidence, output = Path(evidence).resolve(), Path(output).resolve()
    bundle = SamEvidenceBundle.open(evidence)
    generation_mode = bundle.scope.get('sam_crop_mode') or (
        'tiled' if any(run.get('generation_mode') == 'tiled' for run in bundle.runs.values()) else 'whole')
    original = None
    if original_selection is not None:
        original = json.loads(read_artifact(original_selection))
        if original.get('schema') != 'xta.sam_selection/1' or original.get('evidence_fingerprint') != bundle.evidence_fingerprint:
            raise ValueError('Original selection receipt does not belong to the evidence')
    cap = int(max_group_mib)*1024**2
    summary = dict(schema='xta.sam_branch_policy_replays/1', evidence=str(evidence),
        evidence_fingerprint=bundle.evidence_fingerprint, generation_mode=generation_mode,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inference_performed=False, labels_consumed=False, sdf_pixels_injected=False,
        fresh_allocation_bound_bytes=cap, saved_allocation_permission=False,
        group_count=len(bundle.groups), run_count=len(bundle.runs), arms={})
    output.mkdir(parents=True, exist_ok=False)
    for name in variants:
        policy = variant_policy(name, generation_mode=generation_mode, max_group_bytes=cap,
                                original=None if original is None else original['resolved_policy'])
        if policy['branch_aware_selection']:
            from XTA.sam_branch_selection import branch_workspace_bytes
            estimate = branch_workspace_bytes
            workspace_rule = 'XTA.sam_branch_selection.branch_workspace_bytes'
        else:
            estimate = lambda shape: shape[0]*shape[1]*shape[2]*16
            workspace_rule = 'legacy_full_group_16_bytes_per_voxel'
        estimated_refused = sorted(group_id for group_id, group in bundle.groups.items()
            if estimate((len(group['frame_indices']), group['context_bbox_yx'][2]-group['context_bbox_yx'][0],
                         group['context_bbox_yx'][3]-group['context_bbox_yx'][1])) > cap)
        start = time.perf_counter()
        with trace_selection_phases(output/(name+'_phase_trace.jsonl'), enabled=trace_phases) as phase_audit:
            receipt = select_sam_proposals(bundle, {'sam_bridge_policy': policy}, frozen_evidence=True)
        arm_path = output/name
        arm_path.mkdir()
        (arm_path/'selection.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        selected_groups = [row for row in receipt['group_receipts'].values() if row['selected_run_ids']]
        refused = sorted(group_id for group_id, group in receipt['group_receipts'].items()
            if group['status'] == 'not_assessed_resource_refused')
        selected_edges = (set(receipt['branch_selection']['edges']) if 'branch_selection' in receipt else
            {edge['edge_id'] for row in selected_groups for edge in row['topology'].get('edges', ()) if edge['connected']})
        record = dict(selection=str(arm_path/'selection.json'), policy_hash=receipt['policy_hash'],
            policy_name=receipt['policy_name'], selected_runs=len(receipt['selected_run_ids']),
            selected_groups=len(selected_groups), selected_edges=len(selected_edges),
            selected_addition_voxels=sum(row['topology'].get('selected_addition_voxels', 0) for row in selected_groups),
            run_reason_counts=dict(collections.Counter(reason for row in receipt['run_receipts'].values() for reason in row['reasons'])),
            selected_unintended_contact_voxels=sum(row['topology'].get('unintended_contact_voxels', 0) for row in selected_groups),
            resource_abstention_group_ids=refused, estimated_resource_abstention_group_ids=estimated_refused,
            workspace_rule=workspace_rule,
            wall_seconds=time.perf_counter()-start)
        if 'branch_selection_summary' in receipt:
            record['branch_selection_summary'] = receipt['branch_selection_summary']
        if phase_audit is not None:
            record['phase_trace'] = dict(path=str(output/(name+'_phase_trace.jsonl')), **phase_audit)
        if name == 'legacy' and original is not None:
            expected = {run_id for run_id in original['selected_run_ids'] if bundle.runs[run_id]['group_id'] not in refused}
            record['original_admitted_selection_match'] = expected == set(receipt['selected_run_ids'])
        summary['arms'][name] = record
        (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(name, json.dumps(record), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--original-selection', type=Path)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS[:-1]))
    parser.add_argument('--max-group-mib', type=int, default=256)
    parser.add_argument('--trace-phases', action='store_true', help='Record scalar per-group phase timings; no numerical changes')
    run_replays(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
