"""Occupied-tile reductions preserve source-grid votes and policy callbacks."""

from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import numpy as np
import pytest

import XTA
from XTA import reconciliation
from XTA.reconciliation_policy import load_reconciliation_policy, resolve_reconciliation


def test_single_voxel_edge_tile_keeps_full_grid_float32_order():
    shape = (22, 2, 129, 129)
    scores = np.zeros(shape, np.float32)
    values = np.random.default_rng(1234).random((4, shape[0]), dtype=np.float32)
    scores[:, 0, -1, -1] = values[-1]
    direct = scores > .2
    extrapolated = (scores > 0) & ~direct
    tiles = np.zeros((shape[0], shape[1], 2, 2), bool)
    tiles[:, 0, -1, -1] = True
    expected = reconciliation._reduce_vote_groups(scores, direct, extrapolated, None)
    result = reconciliation._reduce_vote_groups(scores, direct, extrapolated, tiles)
    for actual, reference in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual.view(np.uint8), reference.view(np.uint8))
    assert result[0][0, -1, -1] == 22
    assert not result[0][1].any()


def test_complete_package_does_not_fallback_on_native_build_errors(monkeypatch):
    from XTA import native_cpu
    from XTA.native_cpu_build import NativeCpuBuildError

    def unavailable():
        raise NativeCpuBuildError('test native artifact verification failure')

    monkeypatch.setattr(native_cpu, 'load_native_cpu', unavailable)
    scores = np.zeros((2, 1, 129, 129), np.float32)
    direct = np.zeros_like(scores, dtype=bool)
    tiles = np.zeros((2, 1, 2, 2), dtype=bool)
    tiles[0, 0, 0, 0] = True
    with pytest.raises(NativeCpuBuildError, match='test native artifact verification failure'):
        reconciliation._reduce_vote_groups(scores, direct, None, tiles)


class _Scores:
    def __init__(self, data):
        self.data = data

    def iter_crops(self, z):
        plane = self.data[z]
        yield 3, 13, 4, 14, plane[3:13, 4:14]
        yield 8, 18, 9, 19, plane[8:18, 9:19]
        yield 256, 257, 256, 257, plane[-1:, -1:]


def _layer(name, view, kind, foreground, score=None):
    def read_crop(z):
        y, x = np.nonzero(foreground[z])
        if y.size:
            top, bottom, left, right = int(y.min()), int(y.max())+1, int(x.min()), int(x.max())+1
            return top, left, bottom, right, foreground[z, top:bottom, left:right]

    def read_slab(z0, z1):
        return foreground[z0:z1]

    bounds = (0, 2, 4, 257, 5, 257)
    family = 'azimuthal' if view.startswith('azimuthal') else 'orthogonal'
    return reconciliation.EvidenceLayer(
        name, foreground.shape,
        dict(view_name=view, physical_view_name=view, view_family=family, mask_kind=kind, azimuths_deg=(0.,)),
        read_slab, confidence_reader=None if score is None else _Scores(score),
        read_mask_crop=read_crop, mask_crop_bounds=bounds,
    )


def _fixture():
    shape = (3, 257, 257)
    mask = np.zeros(shape, np.uint8)
    mask[:2, 4:16, 5:17] = 1
    mask[1, -1, -1] = 1
    mask[0, 7, 8] = 0
    predictions = []
    for index, view in enumerate(('transverse', 'sagittal', 'azimuthal_transverse')):
        score = mask * (206 + 17 * index)
        score[0, 5, 6] = 0
        score[0, 7, 8] = 255
        predictions.append(_layer(str(index), view, 'yolo', mask, score))
    component = np.zeros_like(mask)
    component[:2, 7:15, 7:14] = 1
    component[:2, -1, -1] = 1
    return shape, [*predictions, _layer('3', 'coronal', 'bridge', component),
                   _layer('4', 'coronal', 'extrapolation', component)]


@pytest.mark.parametrize('name', [
    'union', 'confidence_core_rescue', 'confidence_anchored', 'quorum3', 'largest_island',
    'hybrid_with_fill', 'sam_conservative', 'sam_strict', 'sam_raw_candidates', 'external',
])
def test_sparse_source_voting_matches_dense(name, monkeypatch):
    shape, layers = _fixture()
    context = dict(source_shape_tyx=shape, processing_shape_tyx=shape)
    if name == 'external':
        policy = dict(mode='confidence', anchor_bonus=.35,
                      decide=lambda block: block['candidate'] & (block['score'] >= .5))
    else:
        path = Path(XTA.__file__).parent / 'examples' / 'external_reconciliation' / f'{name}.py'
        policy = load_reconciliation_policy(resolve_reconciliation(
            SimpleNamespace(reconciliation=path, reconciliation_memory_mib=26)))
    owner = get_ident()

    def execute():
        calls, writes = [], []
        base_decide = policy.get('decide')

        def decide(block):
            assert get_ident() == owner
            assert block['candidate'].shape == (block['z1'] - block['z0'], *shape[1:])
            assert all(not member.flags.writeable for member in block.values() if isinstance(member, np.ndarray))
            calls.append(block)
            return base_decide(block)

        report = reconciliation.reconcile(
            layers, shape_tyx=shape, policy={**policy, 'decide': decide if base_decide else None},
            memory_mib=26, geometry_context=context,
            write_slab=lambda a, b, output: writes.append((a, b, output)),
        )
        return report, writes, calls

    reduce_groups = reconciliation._reduce_vote_groups
    reductions = []

    def reduce_and_record(scores, direct, extrapolated, tiles):
        reductions.append(tiles is not None and np.count_nonzero(tiles) * 2 < tiles.size)
        return reduce_groups(scores, direct, extrapolated, tiles)

    with monkeypatch.context() as dense:
        dense.setattr(reconciliation, '_voting_tile_grid', lambda *args: None)
        expected_report, expected_writes, expected_calls = execute()

    monkeypatch.setattr(reconciliation, '_reduce_vote_groups', reduce_and_record)
    report, writes, calls = execute()
    assert report == expected_report
    if policy['mode'] == 'union':
        assert not reductions
    else:
        assert reductions and all(reductions)
    for (a, b, output), (start, stop, reference) in zip(writes, expected_writes, strict=True):
        assert (a, b) == (start, stop)
        np.testing.assert_array_equal(output, reference)
    for actual, reference in zip(calls, expected_calls, strict=True):
        assert actual.keys() == reference.keys()
        for key in actual:
            if isinstance(reference[key], np.ndarray):
                assert actual[key].dtype == reference[key].dtype
                np.testing.assert_array_equal(actual[key].tobytes(), reference[key].tobytes())
            else:
                assert actual[key] == reference[key]
    if name == 'external':
        assert report['counts']['duplicate_group_support_removed'] > 0
        assert report['counts']['confidence_unknown_prediction_voxels'] > 0
        assert [(call['z0'], call['z1']) for call in calls] == [(0, 2), (2, 3)]
        assert calls[0]['extrapolation_support'].max() == 1
        assert not calls[1]['candidate'].any()


def test_unusual_weights_keep_original_dense_reduction(monkeypatch):
    shape, layers = _fixture()
    reductions = []
    dense = reconciliation._reduce_vote_groups

    def record(scores, direct, extrapolated, tiles):
        reductions.append(tiles)
        return dense(scores, direct, extrapolated, tiles)

    monkeypatch.setattr(reconciliation, '_reduce_vote_groups', record)
    with np.errstate(over='ignore', invalid='ignore'):
        reconciliation.reconcile(
            layers, shape_tyx=shape, write_slab=lambda *_: None, memory_mib=26,
            policy=dict(grouping='views', provenance_weights=dict(prediction=1e100, bridge=-0., mixed=1.)),
        )
    assert reductions and all(tiles is None for tiles in reductions)
