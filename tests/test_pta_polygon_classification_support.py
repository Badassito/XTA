"""Polygon-only classification reuses prepared mask support without rereading the cube."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta
from tests.test_pta_multi_input import cpu_only, sequence, run


SUPPORT = 'guaranteed_output_foreground_transverse_slices'


@pytest.mark.parametrize('polygon', [
    '0 .1 .1 .9 .1 .9 .9 .1 .9\n',
    '',
    '0 2 2 3 2 3 3\n',
    '0 .1 .1 .5 .5 .9 .9\n',
])
@pytest.mark.parametrize('fraction', ['1', '.5'])
def test_prepared_partial_polygons_match_dense_support_without_cube_scan(
        tmp_path, monkeypatch, polygon, fraction):
    source = sequence(tmp_path / 'source', labels={1, 3})
    (source / 'sample_0001.txt').write_text(polygon)
    original = pta.classify_original_foregrounds_for_volume
    np_any = np.any
    compared = []

    def classify(prep, **kwargs):
        support = prep.foreground_preservation_stats.pop(SUPPORT)
        try:
            expected = original(prep, **kwargs)
        finally:
            prep.foreground_preservation_stats[SUPPORT] = support

        def no_cube_read(value, *args, **kw):
            assert value is not prep.mask_for_render, 'reread the full prepared mask'
            return np_any(value, *args, **kw)

        with mock.patch.object(pta.np, 'any', side_effect=no_cube_read):
            actual = original(prep, **kwargs)
        assert actual == expected
        assert set(key[1] for key in actual).issubset({0, 2})
        compared.append((support, actual))
        return actual

    monkeypatch.setattr(pta, 'classify_original_foregrounds_for_volume', classify)
    run([source], tmp_path / 'output', extra=[
        '--fraction', fraction, '--background_percent', '1', '--enable_tile', '2:2',
    ])
    assert compared
    assert compared[0][0] == int(bool(polygon) and polygon.startswith('0 .1 .1 .9'))


@pytest.mark.parametrize('change', ['missing_count', 'gaussian', 'void_fill', 'semantic', 'other_view'])
def test_ineligible_or_unknown_cached_support_keeps_dense_empty_check(change):
    mask = np.zeros((2, 3, 4), np.uint8)
    prep = SimpleNamespace(
        label_enabled=True, semantic_foreground=False, semantic_coverage_for_render=None,
        mask_for_render=mask, plans=[SimpleNamespace(
            tag='Transverse', view=SimpleNamespace(family='transverse', num_slices=2),
            eligible_frame_indices=None, tile_layout=())],
        smoothing_stats=[], void_fill_stats=None,
        foreground_preservation_stats={SUPPORT: 1},
        src=SimpleNamespace(stem='sample', label_source='yolo',
                            yolo_polygons_by_frame={0: ()}, labels_by_frame={0: 'sample.txt'}),
    )
    if change == 'missing_count':
        prep.foreground_preservation_stats.clear()
    elif change == 'gaussian':
        prep.smoothing_stats = [{'sigma': 1}]
    elif change == 'void_fill':
        prep.void_fill_stats = {'added_voxels': 1}
    elif change == 'semantic':
        prep.semantic_foreground = True
    else:
        prep.plans[0].view.family = 'sagittal'
    with mock.patch.object(pta.np, 'any', wraps=np.any) as scan:
        result = pta.classify_original_foregrounds_for_volume(prep, workers=1)
    assert scan.call_count == 1
    assert scan.call_args.args[0] is mask
    assert result and not any(result.values())


def test_empty_fraction_selection_returns_before_support_or_mask_access():
    assert pta.classify_original_foregrounds_for_volume(
        SimpleNamespace(label_enabled=True), workers=1, selected_candidates=[]) == {}
