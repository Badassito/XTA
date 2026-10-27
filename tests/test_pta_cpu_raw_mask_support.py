"""Polygon budgeting is not proof that a source or output mask is empty."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import tifffile

from XTA import pta_publication as publication
from XTA.pta_dataset import OutputCandidate


def candidate(*, augmented=False, foreground=False):
    return OutputCandidate(
        order=0, volume_name='sample', parent_view_tag='Transverse', output_tag='Transverse',
        item_key='full', frame_idx=0, is_tile=False, label_enabled=True, foreground=foreground,
        augmentation_index=int(augmented), augmentation_seed=17 if augmented else None,
        augmentation_tag='AbCd1234EfGh5678' if augmented else None,
    )


def publish(tmp_path, cand, mask, **kwargs):
    return publication.write_selected_candidate_version(
        cand=cand, image=np.zeros_like(mask), mask=mask, out_dir=tmp_path,
        split_active=False, image_format='png', png_compression=1, jpeg_quality=95,
        warnings=SimpleNamespace(add=mock.Mock()),
        augmentation=object() if cand.augmentation_index else None,
        save_images=False, **kwargs,
    )


@pytest.mark.parametrize('kind', ['point', 'line'])
def test_unaugmented_nonpolygon_source_keeps_binary_and_semantic_pixels(tmp_path, kind):
    mask = np.zeros((8, 8), np.uint8)
    mask[3, 2:6 if kind == 'line' else 3] = 1
    assert not publication.mask_has_yolo_polygon(mask)
    cand = candidate()
    assert publish(tmp_path, cand, mask, save_binary=True, save_semantic=True) == 'written'
    _, label = publication.candidate_output_paths(tmp_path, cand, split_active=False, image_format='png')
    assert label.read_text() == ''
    binary = publication.candidate_binary_output_path(tmp_path, cand, split_active=False)
    np.testing.assert_array_equal(tifffile.imread(binary), mask)
    semantic = publication.candidate_semantic_output_path(tmp_path, cand, split_active=False)
    np.testing.assert_array_equal(
        publication.cv2.imread(str(semantic), publication.cv2.IMREAD_UNCHANGED), mask,
    )


def test_nonpolygon_source_can_grow_into_exportable_augmented_polygon(tmp_path, monkeypatch):
    source = np.zeros((8, 8), np.uint8)
    source[3, 3] = 1
    output = np.zeros_like(source)
    output[2:6, 2:6] = 1
    cand = candidate(augmented=True)
    monkeypatch.setattr(publication, 'apply_augmentation_pair', lambda *a, **kw: (source, output))
    assert publish(tmp_path, cand, source) == 'written'
    _, label = publication.candidate_output_paths(tmp_path, cand, split_active=False, image_format='png')
    assert label.read_text().startswith('0 ')


@pytest.mark.parametrize('planned_foreground', [False, True])
def test_actually_empty_source_cannot_synthesize_foreground(tmp_path, monkeypatch, planned_foreground):
    source = np.zeros((8, 8), np.uint8)
    monkeypatch.setattr(publication, 'apply_augmentation_pair',
                        lambda *a, **kw: (source, np.ones_like(source)))
    with pytest.raises(RuntimeError, match='truly empty mask'):
        publish(tmp_path, candidate(augmented=True, foreground=planned_foreground), source)


@pytest.mark.parametrize('initially_empty', [False, True])
def test_private_inplace_policy_uses_source_truth_from_before_replay(tmp_path, monkeypatch, initially_empty):
    source = np.zeros((8, 8), np.uint8)
    source[3, 3] = int(not initially_empty)
    output = np.zeros_like(source)
    output[2:6, 2:6] = 1

    def mutate(_policy, image, mask, **kwargs):
        assert not kwargs['copy_inputs']
        mask[:] = int(initially_empty)
        return image, output

    monkeypatch.setattr(publication, 'apply_augmentation_pair', mutate)
    if initially_empty:
        with pytest.raises(RuntimeError, match='truly empty mask'):
            publish(tmp_path, candidate(augmented=True), source, inputs_are_private=True)
    else:
        assert publish(tmp_path, candidate(augmented=True), source, inputs_are_private=True) == 'written'


@pytest.mark.parametrize('outputs', [dict(save_labels=True), dict(save_labels=False, save_binary=True),
                                     dict(save_labels=False, save_semantic=True)])
def test_planned_foreground_copies_that_render_empty_still_drop(tmp_path, monkeypatch, outputs):
    source = np.ones((8, 8), np.uint8)
    monkeypatch.setattr(publication, 'apply_augmentation_pair',
                        lambda *a, **kw: (source, np.zeros_like(source)))
    assert publish(tmp_path, candidate(augmented=True, foreground=True), source, **outputs) == 'flip_dropped'
    assert not list(tmp_path.rglob('*.png')) and not list(tmp_path.rglob('*.txt'))


def test_real_background_augmented_copy_is_retained(tmp_path, monkeypatch):
    source = np.zeros((8, 8), np.uint8)
    monkeypatch.setattr(publication, 'apply_augmentation_pair', lambda *a, **kw: (source, source))
    assert publish(tmp_path, candidate(augmented=True), source) == 'written'
    assert len(list((tmp_path / 'labels').glob('*.txt'))) == 1


def test_unknown_semantic_pixels_are_not_source_known_foreground(tmp_path, monkeypatch):
    source = np.ones((8, 8), np.uint8)
    coverage = np.zeros_like(source)
    monkeypatch.setattr(publication, 'apply_augmentation_pair_with_coverage',
                        lambda *a, **kw: (source, source, np.ones_like(source)))
    with pytest.raises(RuntimeError, match='truly empty mask'):
        publish(tmp_path, candidate(augmented=True), source,
                save_semantic=True, semantic_coverage=coverage)
