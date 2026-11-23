"""Reconciliation consumes immutable component backings in source geometry."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.interpolation import NrrdLayerRef
from XTA.reconciliation_policy import resolve_reconciliation
from XTA.reconciliation_runtime import reconcile_tta_layers


def settings():
    path = Path(__file__).resolve().parents[1] / 'XTA/examples/external_reconciliation/union.py'
    return resolve_reconciliation(SimpleNamespace(reconciliation=path, reconciliation_memory_mib=8))


def test_raw_layer_union_and_metadata_preserve_sources(tmp_path):
    a = np.zeros((3, 5, 7), np.uint8)
    b = a.copy()
    a[:, 1:3, 2:4] = 1
    b[1:, 2:4, 3:5] = 1
    refs = []
    for name, value in (('a', a), ('b', b)):
        path = tmp_path / f'{name}.dat'
        value.tofile(path)
        refs.append(NrrdLayerRef(key=name, name=name, path=path, shape=value.shape,
            model_name='model', view_name=name, physical_view_name=name, view_family='orthogonal',
            source='fullframe', mask_kind='yolo'))
    before = [ref.path.read_bytes() for ref in refs]
    output, report = reconcile_tta_layers(refs, views=[], source_shape_tyx=a.shape,
        processing_shape_tyx=a.shape, settings=settings(), output_dir=tmp_path / 'published', workspace=tmp_path / 'work')
    try:
        np.testing.assert_array_equal(output, a | b)
        assert report['source_layers_preserved']
        assert report['counts']['retained_voxels'] == int((a | b).sum())
        saved = json.loads((tmp_path / 'published/manifest.json').read_text())
        assert saved['layers']['model/a']['metadata']['layer_key'] == 'a'
    finally:
        output._mmap.close()
    assert [ref.path.read_bytes() for ref in refs] == before


def test_conflicting_layer_identity_is_rejected(tmp_path):
    path = tmp_path / 'a.dat'
    np.zeros((2, 3, 4), np.uint8).tofile(path)
    a = NrrdLayerRef(key='same', name='a', path=path, shape=(2, 3, 4),
                     model_name='m', source='fullframe', mask_kind='yolo')
    other = tmp_path / 'b.dat'
    path.replace(other)
    other.write_bytes(other.read_bytes())
    path.write_bytes(other.read_bytes())
    b = replace(a, path=other)
    with pytest.raises(ValueError, match='Conflicting'):
        reconcile_tta_layers([a, b], views=[], source_shape_tyx=a.shape, processing_shape_tyx=a.shape,
            settings=settings(), output_dir=tmp_path / 'published', workspace=tmp_path / 'work')
    assert not (tmp_path / 'work/reconciled_union.u8.dat').exists()


def test_empty_run_produces_empty_result(tmp_path):
    output, report = reconcile_tta_layers([], views=[], source_shape_tyx=(2, 3, 4),
        processing_shape_tyx=(2, 3, 4), settings=settings(),
        output_dir=tmp_path / 'published', workspace=tmp_path / 'work')
    try:
        assert not output.any()
        assert report['layer_count'] == 0
    finally:
        output._mmap.close()


def test_native_view_budget_rejects_before_any_data_reader():
    from XTA.geometry import ViewInfo
    from XTA.reconciliation_policy import validate_policy
    from XTA.reconciliation_runtime import preflight_reconciliation
    from dataclasses import replace
    views = [ViewInfo(name=f'view{i}', num_slices=1931, src_h=3064, src_w=3022,
                       pad_mode='clamp', physical_view_name=f'view{i}') for i in range(300)]
    cfg = replace(settings(), memory_mib=4096)
    with pytest.raises(ValueError, match='at least'):
        preflight_reconciliation(views, source_shape_tyx=(1931,3064,3022),
            processing_shape_tyx=(2911,3064,3022), settings=cfg,
            policy=validate_policy(dict(grouping='views')))
    plan = preflight_reconciliation(views, source_shape_tyx=(1931,3064,3022),
        processing_shape_tyx=(2911,3064,3022), settings=cfg,
        policy=validate_policy(dict(mode='union', grouping='views')))
    assert plan['minimum_working_bytes'] < 100 * 1024**2


def test_section_preflight_pools_repeated_spherical_patches():
    from XTA.geometry import ViewInfo
    from XTA.reconciliation_policy import validate_policy
    from XTA.reconciliation_runtime import preflight_reconciliation
    from dataclasses import replace
    views = [ViewInfo(name=f'spherical_patch{i}', num_slices=1931, src_h=3064, src_w=3022,
                       pad_mode='clamp', physical_view_name=f'spherical_patch{i}', family='spherical') for i in range(300)]
    plan = preflight_reconciliation(views, source_shape_tyx=(1931,3064,3022),
        processing_shape_tyx=(2911,3064,3022), settings=replace(settings(), memory_mib=4096),
        policy=validate_policy(dict(grouping='sections')))
    assert plan['planned_group_count'] == 1


def test_sweep_reconciliation_receipts_share_policy_and_bind_accepted_reports(tmp_path):
    from XTA.reconciliation_policy import (
        accept_sweep_reconciliation_policy, load_reconciliation_report, stage_sweep_reconciliation_report,
    )

    settings_value = settings()
    source = tmp_path / 'work' / 'policy-records'
    family = {'schema': 'xta.confidence_sweep/1', 'sweep_id': 'family', 'status': 'failed',
              'branches': [], 'reconciliation_policies': {}}
    shape = (2, 3, 4)
    raw = np.ones(shape, dtype=np.uint8)
    for cutoff in (50, 60):
        output = tmp_path / 'confidence_cutoffs' / f'conf_{cutoff:03d}'
        _, report = reconcile_tta_layers([], views=[], source_shape_tyx=shape, processing_shape_tyx=shape,
            settings=settings_value, output_dir=output, workspace=tmp_path / 'work',
            assembled_union=raw if cutoff == 50 else np.zeros_like(raw), publish_report=False)
        output.mkdir(parents=True)
        assert not list(output.iterdir())
        compact = stage_sweep_reconciliation_report(report, source)
        assert {'policy', 'policy_path', 'policy_sha256'}.isdisjoint(compact)
        receipt = {'confidence_sweep': {'sweep_id': 'family', 'cutoff_hundredths': cutoff,
                                        'reconciliation_policy_sha256': settings_value.sha256},
                   'status': 'complete', 'reconciliation': compact}
        branch_path = output / 'manifest.json'
        branch_bytes = json.dumps(receipt).encode()
        branch_path.write_bytes(branch_bytes)
        digest, policy_record = accept_sweep_reconciliation_policy(receipt, family['reconciliation_policies'], source)
        family['reconciliation_policies'][digest] = policy_record
        family['branches'].append({'cutoff_hundredths': cutoff, 'status': 'complete',
            'manifest': branch_path.relative_to(tmp_path).as_posix(),
            'manifest_sha256': hashlib.sha256(branch_bytes).hexdigest()})
    assert len(list(source.iterdir())) == len(family['reconciliation_policies']) == 1
    family_path = tmp_path / 'manifest.json'
    family_path.write_text(json.dumps(family), encoding='utf-8')
    low = load_reconciliation_report(tmp_path / family['branches'][0]['manifest'])
    high = load_reconciliation_report(branch_path)
    assert low['policy'] == high['policy'] == report['policy']
    assert low['policy_path'] == high['policy_path'] == settings_value.path
    assert low['counts']['retained_voxels'] == raw.size
    assert high['counts']['retained_voxels'] == 0

    family['reconciliation_policies'][digest]['policy']['name'] = 'tampered'
    family_path.write_text(json.dumps(family), encoding='utf-8')
    with pytest.raises(ValueError, match='differs from the cutoff receipt'):
        load_reconciliation_report(branch_path)
    branch_path.write_bytes(branch_bytes + b'\n')
    with pytest.raises(ValueError, match='differs from its family manifest'):
        load_reconciliation_report(branch_path)
