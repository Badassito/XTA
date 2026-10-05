"""Compare guard-on/off selection on immutable SAM evidence without inference.

Default replay recomputes intrinsic measurements. The explicit
--reuse-original-measurements shortcut reuses the matching original receipt's
measurements while recomputing topology, contacts and deterministic conflicts.
Both arms apply new bounded workspace caps, never the saved live allocation.
Baseline selection is checked against the original within admitted groups.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from XTA.sam_evidence import SamEvidenceBundle
from XTA.sam_filtering import IMPLEMENTATION_SHA256 as FILTER_IMPLEMENTATION_SHA256
from XTA.sam_mask_reader import IMPLEMENTATION_SHA256 as READER_IMPLEMENTATION_SHA256
from XTA.sam_policy import select_sam_proposals


def run_ablation(evidence, selection, output, *, max_group_mib=256,
                 max_rescue_plane_mib=128, reader_cache_mib=32,
                 reuse_original_measurements=False):
    """Write two diagnostic receipts and an attributed, bounded comparison."""
    if max_group_mib <= 0 or max_rescue_plane_mib <= 0 or reader_cache_mib < 0:
        raise ValueError('Workspace bounds must be positive; reader cache must be nonnegative')
    evidence, selection, output = Path(evidence).resolve(), Path(selection).resolve(), Path(output).resolve()
    original = json.loads(selection.read_text(encoding='utf-8'))
    raw_bundle = SamEvidenceBundle.open(evidence)
    if (original.get('schema') != 'xta.sam_selection/1'
            or original.get('evidence_fingerprint') != raw_bundle.evidence_fingerprint):
        raise ValueError('Original selection receipt does not belong to this evidence bundle')
    policy = dict(original['resolved_policy'])
    if policy['kind'] != 'conservative' or not policy['strict_containment']:
        raise ValueError('The original receipt must use conservative guard-on selection')
    policy.update(max_group_bytes=int(max_group_mib*1024**2))
    if 'rescue_max_plane_bytes' in policy:
        policy['rescue_max_plane_bytes'] = int(max_rescue_plane_mib*1024**2)
    def topology_bytes(group):
        shape = (len(group['frame_indices']), group['context_bbox_yx'][2]-group['context_bbox_yx'][0],
                 group['context_bbox_yx'][3]-group['context_bbox_yx'][1])
        if policy.get('branch_aware_selection', False):
            from XTA.sam_branch_selection import branch_workspace_bytes
            return branch_workspace_bytes(shape)
        return shape[0]*shape[1]*shape[2]*16
    refused = sorted(group_id for group_id, group in raw_bundle.groups.items()
        if topology_bytes(group) > policy['max_group_bytes'])
    if reuse_original_measurements:
        for field, identity in [('component_filter_implementation_sha256', FILTER_IMPLEMENTATION_SHA256),
                                ('reader_implementation_sha256', READER_IMPLEMENTATION_SHA256)]:
            if original.get(field) != identity:
                raise ValueError(f'Original {field} changed; replay without the measurement reuse shortcut')
        required = {'infrastructure_errors', 'endpoint_agreement', 'first_observed_violation'}
        for run_id, run in raw_bundle.runs.items():
            if run['group_id'] not in refused and not required.issubset(
                    original.get('run_receipts', {}).get(run_id, {}).get('measurements', {})):
                raise ValueError(f'Original intrinsic measurements unavailable for admitted run {run_id}')

    def frozen_measurements(bundle, group, run_ids, mask_filter, execution, *, retained_index_bytes=0):
        for field in ('schema', 'enabled', 'connectivity', 'measurement_domain', 'comparison',
                      'threshold_source', 'thresholds_by_group'):
            if mask_filter[field] != original['mask_filter'][field]:
                raise ValueError(f'Original intrinsic measurement reuse requires unchanged filter {field}')
        # Reusing measurements starts no worker lanes, but the accepted branch
        # prefix still occupies its admitted indexes throughout this phase.
        # Preserve the same effective-credit accounting as the real adapter.
        credit = max(0, int(execution['parallel_credit_bytes'])-int(retained_index_bytes))
        if 'branch_metadata' in execution:
            metadata = execution['branch_metadata']
            metadata['peak_retained_index_bytes'] = max(metadata['peak_retained_index_bytes'], int(retained_index_bytes))
            metadata['minimum_effective_parallel_credit_bytes'] = min(
                metadata['minimum_effective_parallel_credit_bytes'], credit)
        return {run_id: copy.deepcopy(original['run_receipts'][run_id]['measurements']) for run_id in run_ids}

    method = ('Frozen original intrinsic measurements, recomputed topology/contact/conflict selection'
              if reuse_original_measurements else 'Full fixed-evidence policy replay from immutable masks')
    summary = dict(schema='xta.sam_tight_crop_guard_ablation/1', source_evidence=str(evidence),
        source_selection=str(selection), source_selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
        evidence_fingerprint=raw_bundle.evidence_fingerprint, method=method,
        original_policy_hash=original['policy_hash'],
        allocation_bound_bytes=policy['max_group_bytes'], saved_runtime_allocation_permission=False,
        inference_performed=False, annotation_accuracy_assessed=False,
        original_selected_runs=len(original['selected_run_ids']), arms={},
        resource_abstention_group_ids=refused, assessed_group_count=len(raw_bundle.groups)-len(refused),
        group_count=len(raw_bundle.groups), run_count=len(raw_bundle.runs),
        abstained_original_selected_run_ids=sorted(run_id for run_id in original['selected_run_ids']
            if raw_bundle.runs[run_id]['group_id'] in refused),
        interpretation_limits=[
            'Bounded diagnostic; resource-abstained original selections are absent from intergroup conflicts and contacts',
            'Baseline mismatch prevents attributing selection differences solely to guard disablement',
            'Selection counts do not assess annotation accuracy or enlarge output write contracts'])
    output.mkdir(parents=True, exist_ok=False)
    receipts = {}
    with raw_bundle.reader(max_cache_bytes=int(reader_cache_mib*1024**2)) as bundle:
        for arm, enabled in [('guard_on', True), ('guard_off', False)]:
            selected_policy = {**policy, 'strict_containment': enabled,
                'guarded_rescue': policy.get('guarded_rescue', False) if enabled else False,
                'name': policy['name'] if enabled else policy['name']+'_tight_crop_guard_off'}
            start = time.perf_counter()
            shortcut = (mock.patch('XTA.sam_policy._measure_group_intrinsic', frozen_measurements)
                        if reuse_original_measurements else contextlib.nullcontext())
            with shortcut:
                receipt = select_sam_proposals(bundle, {'sam_bridge_policy': selected_policy}, frozen_evidence=True)
            receipt['diagnostic_method'] = method
            receipts[arm] = receipt
            (output/(arm+'_selection.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            summary['arms'][arm] = dict(selected_runs=len(receipt['selected_run_ids']),
                selected_groups=sum(bool(row['selected_run_ids']) for row in receipt['group_receipts'].values()),
                run_reason_counts=dict(collections.Counter(reason for row in receipt['run_receipts'].values()
                    for reason in row['reasons'])), wall_seconds=time.perf_counter()-start)
        expected = {run_id for run_id in original['selected_run_ids'] if bundle.runs[run_id]['group_id'] not in refused}
        baseline, off = (set(receipts[arm]['selected_run_ids']) for arm in ('guard_on', 'guard_off'))
        summary.update(baseline_original_admitted_selection_match=baseline == expected,
            baseline_vs_original_extra_run_ids=sorted(baseline-expected),
            baseline_vs_original_missing_run_ids=sorted(expected-baseline),
            guard_off_added_run_ids=sorted(off-baseline), guard_off_lost_run_ids=sorted(baseline-off),
            guard_off_added_group_ids=sorted({bundle.runs[run_id]['group_id'] for run_id in off} -
                {bundle.runs[run_id]['group_id'] for run_id in baseline}),
            guard_off_lost_group_ids=sorted({bundle.runs[run_id]['group_id'] for run_id in baseline} -
                {bundle.runs[run_id]['group_id'] for run_id in off}))
    (output/'guard_ablation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Fresh diagnostic output directory')
    parser.add_argument('--max-group-mib', type=int, default=256)
    parser.add_argument('--max-rescue-plane-mib', type=int, default=128)
    parser.add_argument('--reader-cache-mib', type=int, default=32)
    parser.add_argument('--reuse-original-measurements', action='store_true',
        help='Reuse matching original receipt intrinsic measurements; topology/contact decisions are recomputed')
    arguments = parser.parse_args()
    summary = run_ablation(**vars(arguments))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
