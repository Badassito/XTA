"""Profile retained branch-prefix metadata work without decoding or selecting masks.

This reproduces the original ordered prefix snapshot pattern using already
selected retained chunks, including empty rejected groups. It does not qualify
quality decisions or claim production throughput. Use the full fixed-evidence
policy qualifier separately for masks, attribution and conflict parity.
"""
from __future__ import annotations

import argparse
import cProfile
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import pstats
import sys
import time
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from XTA.artifact_archive import read_artifact, physical_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true',
        help='Unprofiled paired metadata-only microbenchmark; no full-selection or cluster timing claim')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--heatsoak-seconds', type=float, default=60.)
    parser.add_argument('--quiet-window-confirmed', action='store_true')
    args = parser.parse_args(argv)
    if args.benchmark and (args.repeats < 2 or not args.quiet_window_confirmed
            or not math.isfinite(args.heatsoak_seconds) or args.heatsoak_seconds < 60.):
        parser.error('Metadata benchmark requires >=2 paired repeats, >=60s heatsoak and a coordinated quiet window')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ[name] = '1'
    from XTA import sam_branch_selection as branch
    from XTA.sam_evidence import SamEvidenceBundle, _plain, fingerprint
    from XTA.sam_policy import _restrict_branch_recipe
    from tools.qualify_sam_policy_throughput import (input_inventory, assert_inputs_unchanged,
        rss_monitor, cpu_heatsoak, environment_receipt)

    evidence = args.evidence.resolve()
    selection_path = (args.selection or evidence.parent/'selection.json').resolve()
    output = args.output.resolve()
    input_parent = physical_path(evidence).parent
    if output.is_relative_to(input_parent) or input_parent.is_relative_to(output):
        raise ValueError('Profile output must not overlap immutable evidence')
    output.mkdir(parents=True, exist_ok=False)
    names = ('XTA/sam_policy.py', 'XTA/sam_branch_selection.py', 'XTA/sam_mask_reader.py',
             'XTA/sam_evidence.py', 'XTA/artifact_archive.py', 'XTA/sam_filtering.py', 'tools/profile_sam_branch_metadata.py',
             'tools/qualify_sam_policy_throughput.py')
    sources = {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in names}
    inputs = input_inventory(evidence, selection_path)
    saved = json.loads(read_artifact(selection_path))
    bundle = SamEvidenceBundle.open(evidence)
    recipe = saved['branch_selection']
    with bundle.reader(max_cache_bytes=0) as reader:
        reader.filter_snapshot(saved)
    chunks = []
    for group_id in sorted(bundle.groups):
        group_edges = [edge_id for edge_id, edge in recipe['edges'].items() if edge['group_id'] == group_id]
        group = _restrict_branch_recipe(recipe, group_edges)
        trials = [_restrict_branch_recipe(group, [edge_id]) for edge_id in sorted(group_edges)]
        chunks.append((group_id, group, trials))

    results = []
    heat = cpu_heatsoak(args.heatsoak_seconds, min(8, os.cpu_count() or 1)) if args.benchmark else None
    orders = [('full_merge', 'immutable_prefix') if repeat % 2 == 0 else ('immutable_prefix', 'full_merge')
              for repeat in range(args.repeats if args.benchmark else 1)]
    for repeat, mode in ((repeat, mode) for repeat, order in enumerate(orders) for mode in order):
        counts = dict(validation_calls=0, validated_edge_records=0, validated_owner_records=0,
            validated_support_plane_records=0, peak_simultaneous_index_bytes=0,
            trial_snapshots=0, group_snapshots=0)
        validate = branch.validate_branch_selection
        def counted(value, *positional, **keywords):
            incoming = branch.branch_selection_from_value(value)
            counts['validation_calls'] += 1
            if incoming is not None:
                counts['validated_edge_records'] += len(incoming['edges'])
                counts['validated_owner_records'] += len(incoming['selected_edge_ids_by_run'])
                counts['validated_support_plane_records'] += sum(len(planes)
                    for edge in incoming['edges'].values() for planes in edge['owner_support'].values())
            return validate(value, *positional, **keywords)
        profiler = cProfile.Profile()
        instrumentation = nullcontext() if args.benchmark else mock.patch.object(branch, 'validate_branch_selection', counted)
        with bundle.reader(max_cache_bytes=0) as reader, instrumentation:
            prefix = reader.filter_snapshot(saved['mask_filter'])
            retained = []
            started = time.perf_counter()
            with rss_monitor() as memory:
                if not args.benchmark:
                    profiler.enable()
                for _group_id, group, trials in chunks:
                    for trial in trials:
                        if mode == 'full_merge':
                            combined = branch.merge_connected_edge_selections([*retained, trial])
                            snapshot = reader.filter_snapshot(dict(mask_filter=prefix, branch_selection=combined))
                            del combined
                        else:
                            snapshot = reader._branch_filter_overlay(prefix, trial, max_index_bytes=64*1024**2)
                            counts['peak_simultaneous_index_bytes'] = max(counts['peak_simultaneous_index_bytes'],
                                3*reader._branch_index_bytes(snapshot))
                        counts['trial_snapshots'] += 1
                        del snapshot
                    if mode == 'full_merge':
                        combined = branch.merge_connected_edge_selections([*retained, group])
                        snapshot = reader.filter_snapshot(dict(mask_filter=prefix, branch_selection=combined))
                        del combined
                    else:
                        snapshot = reader._branch_filter_overlay(prefix, group, max_index_bytes=64*1024**2)
                        counts['peak_simultaneous_index_bytes'] = max(counts['peak_simultaneous_index_bytes'],
                            3*reader._branch_index_bytes(snapshot))
                    counts['group_snapshots'] += 1
                    if group['edges']:
                        if mode == 'full_merge':
                            retained.append(group)
                        else:
                            prefix = snapshot
                    del snapshot
                full = branch.merge_connected_edge_selections(retained if mode == 'full_merge' else [prefix.branch_selection])
                reader.filter_snapshot(dict(mask_filter=prefix, branch_selection=full))
                if not args.benchmark:
                    profiler.disable()
            elapsed = time.perf_counter()-started
            assert reader.stats['mask_decodes'] == 0
            assert full == recipe
            if memory['peak_rss_bytes'] > 1024**3:
                raise MemoryError('Retained metadata exercise exceeded its 1 GiB process envelope')
            stats = dict(reader.stats)
        functions = []
        if not args.benchmark:
            profiler.dump_stats(str(output/f'{mode}.prof'))
            profile = pstats.Stats(profiler)
            selected_names = {'_plain', '_freeze', 'fingerprint', 'validate_branch_selection',
                              'merge_connected_edge_selections', '_validated_branch_overlay'}
            functions = [dict(file=name, line=line, function=function, primitive_calls=primitive,
                calls=calls, self_seconds=self_seconds, cumulative_seconds=cumulative)
                for (name, line, function), (primitive, calls, self_seconds, cumulative, _callers)
                in profile.stats.items() if function in selected_names]
        else:
            counts = {key: value for key, value in counts.items() if not key.startswith(('validation_', 'validated_'))}
        results.append(dict(mode=mode, repeat=repeat, counts=counts, functions=functions, reader_stats=stats,
            wall_seconds=elapsed, cprofile_enabled=not args.benchmark, memory=memory,
            final_serialized_bytes=len(json.dumps(_plain(full), separators=(',', ':')).encode()),
            final_sha256=full['sha256']))
        print(json.dumps(dict(mode=mode, repeat=repeat, counts=counts, wall_seconds=elapsed)), flush=True)
        del full
    assert_inputs_unchanged(inputs)
    assert sources == {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in names}
    report = dict(schema='xta.sam_branch_prefix_metadata_profile/1', status='exact_retained_metadata',
        full_retained_scope=True, group_count=len(bundle.groups), run_count=len(bundle.runs),
        selected_edge_count=len(recipe['edges']), selected_owner_count=len(recipe['selected_edge_ids_by_run']),
        evidence_fingerprint=bundle.evidence_fingerprint, source_sha256=sources, inputs=inputs,
        gpu_used=False, final_portable_receipts_exact=True, results=results, heatsoak=heat,
        environment=environment_receipt(), mode='microbenchmark' if args.benchmark else 'profile',
        quiet_window_confirmed=bool(args.quiet_window_confirmed), process_rss_limit_bytes=1024**3,
        limitation=('Heatsoaked CPU metadata-only microbenchmark; no mask decodes, recomputed quality decisions, full-selection or cluster wall-time claim.'
            if args.benchmark else 'Metadata-only cProfile diagnostic; no mask decodes, recomputed quality decisions, or production throughput claim.'))
    name = 'metadata_benchmark.json' if args.benchmark else 'metadata_profile.json'
    (output/name).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
