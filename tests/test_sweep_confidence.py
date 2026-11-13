"""View-major score publication matches scalar retained evidence per cutoff."""
from __future__ import annotations

import hashlib
import json
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.confidence_evidence import (
    capture_prediction_confidence,
    configure_confidence_evidence,
    lookup_confidence_evidence,
    publish_native_confidence_pieces,
)
from XTA.sweep_confidence import SweepObservedConfidenceWriter


@pytest.fixture(autouse=True)
def reset_collection():
    source_shape = assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(None)
    yield
    configure_confidence_evidence(None, enabled=False)
    assembly.set_final_source_output_shape(source_shape)


def inputs():
    shape = (3, 5, 7)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    mask = np.zeros(shape, np.uint8)
    mask[0, 1:4, 1:6] = 1
    mask[1, 2, 2] = 1
    scores = np.zeros_like(mask)
    scores[0, 1, 1] = 173
    scores[0, 2, 2] = 240
    scores[0, 0, 0] = 250  # Discarded support cannot leak into evidence.
    return shape, view, mask, scores


def read(ref):
    with ref.native_reader() as reader:
        return reader(0, ref.storage_shape[0])


@pytest.mark.parametrize('task', ['segment', 'semantic'])
def test_branch_scores_equal_scalar_native_capture(tmp_path, task):
    shape, view, mask, scores = inputs()
    floors = {'gpu': .50}
    configure_confidence_evidence(tmp_path / 'scalar', enabled=True, defer_projection=True,
        run_id='scalar', prediction_task=task, confidence_floor=.50,
        min_conf=0, confidence_floor_by_backend=floors)
    baseline = capture_prediction_confidence(mask, scores, view=view,
        model_name='model', temp_dir=tmp_path / 'scalar_work')
    writer = SweepObservedConfidenceWriter(tmp_path / 'branch', cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='branch', task=task,
        confidence_floor_by_backend=floors)
    before_mask, before_scores = mask.copy(), scores.copy()
    actual = writer.publish_view(view, 'model', native_shape=shape,
        read_mask=lambda z: mask[z], read_scores=lambda z: scores[z])
    np.testing.assert_array_equal(mask, before_mask)
    np.testing.assert_array_equal(scores, before_scores)
    for expected, got in zip(read(baseline), read(actual)):
        np.testing.assert_array_equal(expected, got)
    assert actual.metadata['payload_sha256'] == baseline.metadata['payload_sha256']
    assert actual.metadata['index_sha256'] == baseline.metadata['index_sha256']
    assert actual.metadata['provenance']['run_id'] == 'branch'
    assert actual.metadata['provenance']['observed_confidence']['score_kind'] == (
        'instance_detection_confidence' if task == 'segment' else 'semantic_foreground_probability')
    # The explicit publisher cannot replace or add to the active scalar registry.
    assert lookup_confidence_evidence(actual.layer_key, 'model').path == baseline.path


def test_two_cutoff_catalogues_do_not_mix_and_empty_scores_are_retained(tmp_path):
    shape, view, mask, scores = inputs()
    refs = []
    for cent in (50, 80):
        branch_scores = scores if cent == 50 else np.zeros_like(scores)
        writer = SweepObservedConfidenceWriter(tmp_path / str(cent), cutoff_hundredths=cent,
            source_shape_tyx=shape, run_id=f'run-{cent}')
        refs.append(writer.publish_view(view, 'same-model', native_shape=shape,
            read_mask=lambda z: mask[z], read_scores=lambda z: branch_scores[z]))
        catalogue = json.loads(writer.finish().read_text())
        assert catalogue['run_id'] == f'run-{cent}'
        assert len(catalogue['layers']) == 1
        assert catalogue['observed_confidence']['capture_confidence_floor'] == cent / 100.
        assert catalogue['layers'][0]['directory'] == refs[-1].path.name
    assert refs[0].metadata['known_voxels'] == 2
    assert refs[1].metadata['known_voxels'] == 0
    assert not read(refs[1])[1].any()
    assert refs[0].path.parent != refs[1].path.parent


def test_empty_masks_skip_score_reads_but_publish_an_empty_layer(tmp_path):
    shape, view, mask, scores = inputs()
    writer = SweepObservedConfidenceWriter(tmp_path, cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='empty')
    callback = mock.Mock(side_effect=AssertionError('no score read for empty support'))
    ref = writer.publish_view(view, 'model', native_shape=shape,
        read_mask=lambda z: None, read_scores=callback)
    assert ref.metadata['known_voxels'] == ref.metadata['block_count'] == 0
    callback.assert_not_called()
    assert len(json.loads(writer.manifest_path.read_text())['layers']) == 1


def test_consumer_adopts_exact_branch_and_skips_recapture(tmp_path):
    shape, view, mask, scores = inputs()
    writer = SweepObservedConfidenceWriter(tmp_path, cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='branch')
    ref = writer.publish_view(view, 'model', native_shape=shape,
        read_mask=lambda z: mask[z], read_scores=lambda z: scores[z])
    before = hashlib.sha256(writer.manifest_path.read_bytes()).hexdigest()
    configure_confidence_evidence(tmp_path, enabled=True, defer_projection=True,
        prediction_task='segment', confidence_floor=.50, run_id='branch', adopt_existing=True)
    actual = lookup_confidence_evidence(ref.layer_key, 'model')
    assert actual.path == ref.path.resolve()
    assert capture_prediction_confidence(mask, None, view=view, model_name='model',
        temp_dir=tmp_path / 'unused').path == actual.path
    assert publish_native_confidence_pieces([], native_shape=shape, view=view,
        model_name='model').path == actual.path
    assert hashlib.sha256(writer.manifest_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('wrong', ['run_id', 'confidence_floor'])
def test_consumer_rejects_another_identity_or_cutoff(tmp_path, wrong):
    shape, view, mask, scores = inputs()
    writer = SweepObservedConfidenceWriter(tmp_path, cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='branch')
    writer.publish_view(view, 'model', native_shape=shape,
        read_mask=lambda z: mask[z], read_scores=lambda z: scores[z])
    config = dict(prediction_task='segment', confidence_floor=.50, run_id='branch')
    config[wrong] = 'different' if wrong == 'run_id' else .60
    with pytest.raises(ValueError, match='identity or score contract'):
        configure_confidence_evidence(tmp_path, enabled=True, defer_projection=True,
            adopt_existing=True, **config)


def test_partial_adoption_can_capture_other_views(tmp_path):
    shape, view, mask, scores = inputs()
    writer = SweepObservedConfidenceWriter(tmp_path, cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='branch')
    first = writer.publish_view(view, 'model', native_shape=shape,
        read_mask=lambda z: mask[z], read_scores=lambda z: scores[z])
    configure_confidence_evidence(tmp_path, enabled=True, defer_projection=True,
        prediction_task='segment', confidence_floor=.50, run_id='branch', adopt_existing=True)
    second = capture_prediction_confidence(mask, scores, view=view, model_name='model',
        temp_dir=tmp_path / 'work', layer_key='other-view')
    assert first.layer_key != second.layer_key
    catalogue = json.loads(writer.manifest_path.read_text())
    assert len(catalogue['layers']) == 2
    configure_confidence_evidence(tmp_path, enabled=True, defer_projection=True,
        prediction_task='segment', confidence_floor=.50, run_id='branch', adopt_existing=True)
    assert lookup_confidence_evidence(second.layer_key, 'model') is not None


def test_invalid_reader_fails_without_a_catalogue_entry(tmp_path):
    shape, view, mask, scores = inputs()
    writer = SweepObservedConfidenceWriter(tmp_path, cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='branch')
    with pytest.raises(ValueError, match='score plane differs'):
        writer.publish_view(view, 'model', native_shape=shape,
            read_mask=lambda z: mask[z], read_scores=lambda z: scores[z].astype(np.float32))
    assert not writer.manifest_path.exists()


def test_sparse_overlapping_strips_equal_masked_max_without_dense_reads(tmp_path):
    shape = (2, 260, 270)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    first_mask = np.ones((142, 155), np.uint8)
    first_mask[:4, :7] = 0
    first_scores = np.full(first_mask.shape, 173, np.uint8)
    second_mask = np.ones((131, 141), np.uint8)
    second_mask[3:8, 5:9] = 0
    second_scores = np.full(second_mask.shape, 231, np.uint8)
    crops = [(11, 153, 19, 174, first_mask, first_scores),
             (110, 241, 120, 261, second_mask, second_scores)]
    expected = np.zeros(shape, np.uint8)
    for y0, y1, x0, x1, mask, values in crops:
        target = expected[0, y0:y1, x0:x1]
        np.maximum(target, np.where(mask, values, 0), out=target)
    writer = SweepObservedConfidenceWriter(tmp_path, cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='sparse')
    dense = mock.Mock(side_effect=AssertionError('dense plane should never be read'))
    ref = writer.publish_view(view, 'model', native_shape=shape,
        read_mask=dense, read_scores=dense,
        read_native_crops=lambda z: reversed(crops) if z == 0 else ())
    values, known = read(ref)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(known, expected > 0)
    dense.assert_not_called()
    # Sorted global blocks preserve exactly the ordinary scalar file encoding.
    configure_confidence_evidence(tmp_path / 'scalar', enabled=True, defer_projection=True,
        prediction_task='segment', confidence_floor=.50)
    scalar = capture_prediction_confidence((expected != 0).astype(np.uint8), expected,
        view=view, model_name='model', temp_dir=tmp_path / 'work')
    assert ref.metadata['payload_sha256'] == scalar.metadata['payload_sha256']
    assert ref.metadata['index_sha256'] == scalar.metadata['index_sha256']


def test_sparse_unknown_score_crops_do_not_invent_observations(tmp_path):
    shape, view, mask, scores = inputs()
    writer = SweepObservedConfidenceWriter(tmp_path, cutoff_hundredths=50,
        source_shape_tyx=shape, run_id='unknown')
    ref = writer.publish_view(view, 'model', native_shape=shape,
        read_native_crops=lambda z: [(0, shape[1], 0, shape[2], mask[z], None)])
    assert ref.metadata['known_voxels'] == 0
    assert not read(ref)[1].any()
