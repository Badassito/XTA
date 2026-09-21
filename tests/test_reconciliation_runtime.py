"""Reconciliation consumes immutable component backings in source geometry."""
from dataclasses import replace
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
