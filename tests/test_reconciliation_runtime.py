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


def test_same_grid_packed_confidence_reconciliation_matches_dense_vote(tmp_path, monkeypatch):
    from XTA import confidence_evidence
    from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, write_raw_bbox_mask_store
    from XTA.reconciliation_runtime import RuntimeLayer

    shape = (3, 16, 145)
    dense, packed, scores_by_key = [], [], {}
    for index, view in enumerate(('transverse', 'tilted_transverse_horizontal_p0p1', 'sagittal')):
        values = np.zeros(shape, np.uint8)
        values[index % 2:, 4+index:11+index, 119+index:139-index] = 1
        values[:, 6, 132] = 0
        scores = np.zeros(shape, np.uint8)
        scores[:, 3:14, 117:135+index] = 210 - 4*index
        scores[0, 7, 130] = 0
        scores_by_key[view] = confidence_evidence.write_block_confidence_evidence(
            tmp_path / f'{view}.scores', shape, scores.__getitem__, model_name='m', layer_key=view)
        raw_path = tmp_path / f'{view}.dat'
        values.tofile(raw_path)
        ref = NrrdLayerRef(key=view, name=view, path=raw_path, shape=shape, model_name='m',
                           view_name=view, physical_view_name=view, view_family='orthogonal',
                           source='fullframe', mask_kind='yolo')
        dense.append(ref)
        packed_path = tmp_path / f'{view}.cvol'
        write_raw_bbox_mask_store(values, packed_path, format_name=INTERNAL_PACKED_CVOL_FORMAT)
        packed.append(replace(ref, path=packed_path, storage_format=INTERNAL_PACKED_CVOL_FORMAT))
    monkeypatch.setattr(confidence_evidence, 'lookup_confidence_evidence',
                        lambda ref: scores_by_key[ref.key])

    calls = []

    def decide(block):
        calls.append({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in block.items()})
        return block['anchored'] & (block['score'] >= 1.5)

    def execute(refs, directory):
        output, report = reconcile_tta_layers(
            refs, views=[], source_shape_tyx=shape, processing_shape_tyx=shape, settings=settings(),
            policy=dict(mode='confidence', decide=decide), output_dir=tmp_path / directory,
            workspace=tmp_path / (directory + '-workspace'))
        try:
            return output.copy(), report['counts']
        finally:
            output._mmap.close()

    expected_mask, expected_counts = execute(dense, 'dense')
    dense_calls = calls.copy()
    calls.clear()

    def unexpected_slab(owner, z0, z1):
        pytest.fail('Same-grid packed voting expanded a complete dense mask slab')

    monkeypatch.setattr(RuntimeLayer, 'read_slab', unexpected_slab)
    actual_mask, actual_counts = execute(packed, 'packed')
    assert expected_counts['duplicate_group_support_removed'] > 0
    assert expected_counts['confidence_unknown_prediction_voxels'] > 0
    assert actual_counts == expected_counts
    np.testing.assert_array_equal(actual_mask, expected_mask)
    for actual, expected in zip(calls, dense_calls, strict=True):
        assert actual.keys() == expected.keys()
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])


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


def test_largest_island_uses_indexed_bounds_on_the_managed_entrypoint(tmp_path):
    from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, write_raw_bbox_mask_store
    cfg = resolve_reconciliation(SimpleNamespace(
        reconciliation=Path('XTA/examples/external_reconciliation/largest_island.py'),
        reconciliation_memory_mib=8))
    values = np.zeros((6, 9, 13), np.uint8)
    values[2:5, 3:6, 4:8] = 1
    refs = []
    for view in ('transverse', 'sagittal'):
        path = tmp_path / f'{view}.cvol'
        write_raw_bbox_mask_store(values, path, format_name=INTERNAL_PACKED_CVOL_FORMAT)
        refs.append(NrrdLayerRef(key=view, name=view, path=path, shape=values.shape,
            model_name='m', view_name=view, physical_view_name=view, view_family='orthogonal',
            source='fullframe', mask_kind='yolo', storage_format=INTERNAL_PACKED_CVOL_FORMAT))
    output, report = reconcile_tta_layers(refs, views=[], source_shape_tyx=values.shape,
        processing_shape_tyx=values.shape, settings=cfg,
        output_dir=tmp_path / 'published', workspace=tmp_path / 'workspace')
    try:
        np.testing.assert_array_equal(output, values)
        for record in report['components'].values():
            assert record['component_working_bounds_tyx'] == (2, 5, 3, 6, 4, 8)
            assert record['component_count'] == 1
            assert record['largest_component_voxels'] == 36
    finally:
        output._mmap.close()


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
