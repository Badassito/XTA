#!/usr/bin/env python3
"""Inspect an external-TTA run's receipts, support, and independent NRRD passes.

Reads manifests, not loose/stale filenames. It does not judge segmentation quality
or independently prove that the binary final union contains every component.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def _output_group(row: dict) -> tuple[str, str, str]:
    kind = row.get('kind')
    view = row.get('view')
    tile_config = row.get('tile_config_id') or ''
    assert kind in ('fullframe', 'tile'), f'missing or invalid task kind: {row}'
    assert isinstance(view, str) and view, f'missing task view: {row}'
    assert not re.search(r'__policy_\d+$', view), f'expected base task view: {view}'
    assert isinstance(tile_config, str), f'invalid tile configuration: {row}'
    assert bool(tile_config) == (kind == 'tile'), f'task kind/configuration mismatch: {row}'
    return kind, view, tile_config


def check(root: Path, *, require_nrrd: bool = True) -> dict:
    root=root.resolve()
    manifest=json.loads((root/'augmentation_manifest.json').read_text())
    ratio=int(manifest['ratio'])
    assert ratio>=2, 'not an augmented run'
    policies = manifest.get('policies')
    if policies:
        snapshots = manifest.get('policy_snapshots', {})
        assert set(snapshots) == set(policies), 'policy snapshot backend mismatch'
        for backend, policy in policies.items():
            snapshot = root / 'augmentation_support' / Path(snapshots[backend]).name
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == policy['sha256'], 'policy snapshot hash mismatch'
    else:
        snapshot=root/'augmentation_support'/'policy.py'
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==manifest['content_sha256'], 'policy snapshot hash mismatch'
    execution=manifest['execution_records']
    assert execution, 'no completed policy inference tasks'
    planned_rows = manifest.get('planned_output_groups')
    assert isinstance(planned_rows, list) and planned_rows, (
        'missing planned_output_groups; rerun with updated augmentation manifests to verify complete output groups')
    planned = {_output_group(row) for row in planned_rows}
    assert len(planned) == len(planned_rows), 'duplicate planned output groups'
    executed = {_output_group(row) for row in execution}
    assert executed == planned, (
        f'execution/planning group mismatch; missing={sorted(planned-executed)}, unexpected={sorted(executed-planned)}')
    for row in execution:
        assert row['pass_count']==ratio, f'bad pass count: {row}'
        assert row['source_render_replays']==0, f'geometry was replayed: {row}'
        assert row['model_batches']==row['rendered_batches']*ratio, f'bad inference count: {row}'
    coverage=manifest['coverage_records']
    if manifest['coverage']=='packed':
        assert len(coverage)==len(execution)*(ratio-1), 'missing per-task/pass support'
        import numpy as np
        for row in coverage:
            # The basename allows relocating a completed output directory.
            path=root/'augmentation_support'/Path(row['path']).name
            with np.load(path,allow_pickle=False) as packed:
                meta=json.loads(str(packed['metadata'].item()))
                assert meta['view']==row['view'], 'support view mismatch'
                h,w=meta['raster_shape']
                n=len(packed['seeds'])
                assert packed['validity_bits'].shape==(n,h,(w+7)//8), 'invalid packed support shape'
                assert len(packed['global_destinations'])==n, 'missing support destinations'
    report={'pass_count':ratio,'inference_tasks':len(execution),
            'rendered_batches':sum(r['rendered_batches'] for r in execution),
            'model_batches':sum(r['model_batches'] for r in execution),
            'coverage_files':len(coverage),'fullframe_groups':0,'fullframe_nrrds':0,'tile_groups':0}
    nrrd_manifests=list((root/'nrrd').glob('*_nrrd_manifest.json'))
    if require_nrrd:
        assert len(nrrd_manifests)==1, 'expected one primary NRRD manifest; use a fresh output directory and --save nrrd'
    if nrrd_manifests:
        assert len(nrrd_manifests)==1, 'ambiguous NRRD manifests in reused output directory'
        data=json.loads(nrrd_manifests[0].read_text())
        full=defaultdict(set);tiles=defaultdict(set);filenames=[]
        for layer in data['layers']:
            filename=layer['filename'];filenames.append(filename)
            assert (nrrd_manifests[0].parent/filename).is_file(), f'missing NRRD {filename}'
            name=layer.get('view_name','')
            match=re.search(r'__policy_(\d+)$',name)
            p=int(match.group(1)) if match else 0
            base=name[:match.start()] if match else name
            assert 0<=p<ratio, f'unexpected policy pass: {name}'
            if p:
                assert layer.get('mask_kind')!='bridge', f'augmented interpolation layer: {filename}'
                assert int(layer.get('pass_index',0))==0, f'augmented interpolation pass: {filename}'
            if layer.get('source')=='fullframe' and layer.get('mask_kind')=='yolo':
                assert p not in full[base], f'duplicate full-frame pass: {name}'
                full[base].add(p)
            if layer.get('source')=='tile' and layer.get('mask_kind')=='yolo':
                tiles[(base,layer.get('tile_config_id',''))].add(p)
        assert len(filenames)==len(set(filenames)), 'duplicate output filenames'
        expected=set(range(ratio))
        for base,passes in full.items():
            assert passes==expected, f'missing full-frame passes for {base}: {passes}'
        for key,passes in tiles.items():
            assert passes==expected, f'missing tile passes for {key}: {passes}'
        # Plan entries exist independently of execution and publication, including
        # empty detections. Checking encountered groups alone misses an entire
        # absent tile configuration (or an absent view and all its receipts).
        published = {('fullframe', view, '') for view in full}
        published.update(('tile', view, config) for view, config in tiles)
        assert published == planned, (
            f'published/planned output group mismatch; missing={sorted(planned-published)}, '
            f'unexpected={sorted(published-planned)}')
        report.update(fullframe_groups=len(full),fullframe_nrrds=sum(map(len,full.values())),tile_groups=len(tiles),
                      total_nrrds=len(filenames))
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('--no-nrrd',action='store_true',help='Check a run that deliberately omitted --save nrrd')
    args=parser.parse_args()
    print(json.dumps(check(args.output_dir,require_nrrd=not args.no_nrrd),indent=2,sort_keys=True))


if __name__=='__main__':
    main()
