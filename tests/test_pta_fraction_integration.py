"""Fraction selection must precede mask classification and image publication."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options
from tests.test_pta_multi_input import cpu_only, sequence, run, artifacts


@pytest.mark.parametrize('value', ['0', '-0.1', '1.1', 'nan', 'inf'])
def test_fraction_cli_rejects_invalid_values(value, capsys):
    with pytest.raises(SystemExit) as error:
        parse_pta_args(['--input', 'dataset', '--fraction', value])
    assert error.value.code == 2
    assert '--fraction' in capsys.readouterr().err


def test_fraction_default_and_runtime_option():
    assert build_runtime_options(parse_pta_args(['--input', 'dataset'])).fraction == 1.0
    assert build_runtime_options(parse_pta_args(['--input', 'dataset', '--fraction', '.25'])).fraction == .25


def prepared(tmp_path, *, semantic):
    config = parse_pta_args(['--input', 'dataset', '--imgsz', '8', '--enable_cartesian', 'transverse'])
    views, _ = pta.compile_v18_pta_views(t_dim=4, h=8, w=8, config=config, azimuthal_native_raster=8)
    plans = pta.build_plans_for_views(
        views=views, out_dir=tmp_path, stem='sample',
        tile_configs=(pta.TileConfig(4, 4, 's4_st4'),),
        save_overlay=False, imgsz=8, label_enabled=True, image_format='png',
        channel_variants=(pta.DEFAULT_CHANNEL_VARIANT,), publish_images=False, publish_labels=False,
    )
    return SimpleNamespace(
        label_enabled=True, semantic_foreground=semantic, semantic_coverage_for_render=None,
        mask_for_render=np.ones((4, 8, 8), np.uint8), plans=plans, smoothing_stats=[],
        src=SimpleNamespace(stem='sample', label_source='nrrd', yolo_polygons_by_frame={}, labels_by_frame={}),
    )


@pytest.mark.parametrize('semantic', [False, True])
def test_classification_skips_rejected_frames_full_output_and_other_tiles(tmp_path, semantic):
    prep = prepared(tmp_path, semantic=semantic)
    selected = next(c for c in pta.enumerate_candidates_for_volume(prep) if c.is_tile and c.frame_idx == 1)
    with mock.patch.object(pta, 'render_plan_frame_mask_source', side_effect=AssertionError('rendered rejected full output')):
        result = pta.classify_original_foregrounds_for_volume(prep, workers=1, selected_candidates=[selected])
    assert result == {(selected.parent_view_tag, 1, selected.item_key, 0): True}


def test_empty_selection_does_not_scan_source_mask():
    assert pta.classify_original_foregrounds_for_volume(
        SimpleNamespace(label_enabled=True), workers=1, selected_candidates=[],
    ) == {}


def test_candidate_enumeration_is_lazy_before_fraction_selection(tmp_path):
    prep = prepared(tmp_path, semantic=False)
    def first_only(candidates, *, fraction):
        assert iter(candidates) is candidates
        return [next(candidates)]
    with mock.patch.object(pta, 'select_fraction_candidates', side_effect=first_only), \
            mock.patch.object(pta, 'OutputCandidate', wraps=pta.OutputCandidate) as constructed:
        candidates = pta.enumerate_candidates_for_volume(prep, fraction=.1)
    assert len(candidates) == constructed.call_count == 1


def test_fraction_one_preserves_files_and_fraction_filters_before_classification(tmp_path, monkeypatch):
    source = sequence(tmp_path / 'input', labels=set(range(1, 9)), indices=range(1, 9))
    for label in source.glob('*.txt'):
        label.write_text('0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n')
    baseline, explicit, sampled = (tmp_path / name for name in ('baseline', 'explicit', 'sampled'))
    run([source], baseline)
    run([source], explicit, extra=['--fraction', '1'])
    for folder in ('images', 'labels'):
        assert artifacts(baseline, folder) == artifacts(explicit, folder)
    classified = []
    original = pta.classify_original_foregrounds_for_volume
    def classify(prep, **kwargs):
        selected = kwargs['selected_candidates']
        assert selected is not None
        classified.append(len(selected))
        return original(prep, **kwargs)
    monkeypatch.setattr(pta, 'classify_original_foregrounds_for_volume', classify)
    manifest = run([source], sampled, extra=['--fraction', '.5', '--background_percent', '.25'])
    images = artifacts(sampled, 'images')
    assert len(images) < len(artifacts(baseline, 'images'))
    assert classified == [len(images)]
    for folder in ('images', 'labels'):
        full = artifacts(baseline, folder)
        assert all(full[name] == content for name, content in artifacts(sampled, folder).items())
    assert manifest['resolved_configuration']['fraction_sampling']['fraction'] == .5
    assert 'original_candidates_before_fraction: 8' in (sampled / 'summary.txt').read_text()


def test_tiny_fraction_publishes_complete_empty_dataset(tmp_path):
    source = sequence(tmp_path / 'input', labels={1, 2, 3, 4})
    output = tmp_path / 'empty'
    manifest = run([source], output, extra=['--fraction', '1e-300', '--background_percent', '.5'])
    assert manifest['status'] == 'complete'
    assert artifacts(output, 'images') == artifacts(output, 'labels') == {}
    assert 'original_candidates_total_before_filtering: 0' in (output / 'summary.txt').read_text()


def test_tiles_can_be_published_when_their_full_frame_was_not_selected(tmp_path, monkeypatch):
    source = sequence(tmp_path / 'input', labels={1, 2, 3, 4})
    output = tmp_path / 'sampled-tiles'
    selected = []
    original = pta.enumerate_candidates_for_volume
    def enumerate_selected(*args, **kwargs):
        candidates = original(*args, **kwargs)
        selected.extend(candidates)
        return candidates
    monkeypatch.setattr(pta, 'enumerate_candidates_for_volume', enumerate_selected)
    run([source], output, extra=['--fraction', '.35', '--enable_tile', '2:2'])
    full_frames = {(c.parent_view_tag, c.frame_idx) for c in selected if not c.is_tile}
    assert any(c.is_tile and (c.parent_view_tag, c.frame_idx) not in full_frames for c in selected)
    expected = {str(pta.candidate_output_paths(output, c, split_active=False, image_format='png')[0].relative_to(output / 'images')) for c in selected}
    assert set(artifacts(output, 'images')) == expected


def test_offline_ratio_only_copies_selected_originals(tmp_path, monkeypatch):
    monkeypatch.setenv('NO_ALBUMENTATIONS_UPDATE', '1')
    source = sequence(tmp_path / 'input', labels=set(range(1, 9)), indices=range(1, 9))
    for label in source.glob('*.txt'):
        label.write_text('0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n')
    policy = tmp_path / 'identity.py'
    policy.write_text('import albumentations as A\ndef build_augmentation():\n    return A.Compose([A.NoOp(p=1)])\n')
    selected = []
    original = pta.enumerate_candidates_for_volume
    def enumerate_selected(*args, **kwargs):
        candidates = original(*args, **kwargs)
        selected.extend(candidates)
        return candidates
    monkeypatch.setattr(pta, 'enumerate_candidates_for_volume', enumerate_selected)
    output = tmp_path / 'offline'
    run([source], output, extra=['--fraction', '.5', '--background_percent', '.25',
        '--augmentation', f'cpu:{policy}', '--augmentation_execution', 'offline', '--augmentation_ratio', '5'])
    assert 0 < len(selected) < 8
    assert len(artifacts(output, 'images')) == len(artifacts(output, 'labels')) == 5 * len(selected)
