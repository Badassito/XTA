"""Observed score saves work without reconciliation and never score generated support."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.config import resolve_save_request
from XTA.confidence_evidence import (
    ConfidenceEvidenceRef, capture_prediction_confidence, configure_confidence_evidence,
    publish_native_confidence_pieces,
)
from XTA.confidence_retention import (
    confidence_retention_requested, observed_confidence_contract, published_confidence_manifest,
)


@pytest.fixture(autouse=True)
def reset_confidence():
    source_shape = assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(None)
    yield
    configure_confidence_evidence(None, enabled=False)
    assembly.set_final_source_output_shape(source_shape)


def test_scalar_confidence_save_is_optional_and_composes_with_existing_outputs():
    assert resolve_save_request(None).options == ()
    request = resolve_save_request(['confidence,labels', 'low_quality:0.5:video', 'confidence'])
    assert request.options == ('confidence', 'labels', 'low_quality:0.5:video')
    assert request.low_quality[0].formats == ('video',)
    assert not confidence_retention_requested(())
    assert confidence_retention_requested(request.options)
    assert confidence_retention_requested((), reconciliation_mode='confidence')
    for mode in ('union', 'weighted'):
        assert not confidence_retention_requested((), reconciliation_mode=mode)
        assert confidence_retention_requested(request.options, reconciliation_mode=mode)


@pytest.mark.parametrize('task,kind', [
    ('segment', 'instance_detection_confidence'),
    ('semantic', 'semantic_foreground_probability'),
])
def test_observed_native_publication_is_independent_of_reconciliation(tmp_path, task, kind):
    shape = (2, 5, 7)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    output = tmp_path / 'output'
    configure_confidence_evidence(output, enabled=True, defer_projection=True,
        prediction_task=task, confidence_floor=.21, min_conf=.30)
    mask = np.zeros(shape, np.uint8)
    mask[0, 1:4, 1:5] = 1
    scores = np.zeros_like(mask)
    scores[0, 1, 1] = 173
    scores[0, 0, 0] = 231  # Removed prediction support must not be resurrected.
    before_mask, before_scores = mask.copy(), scores.copy()
    with redirect_stdout(io.StringIO()):
        ref = capture_prediction_confidence(mask, scores, view=view,
            model_name='model', temp_dir=tmp_path / 'work')
    np.testing.assert_array_equal(mask, before_mask)
    np.testing.assert_array_equal(scores, before_scores)
    assert ref.coordinate_space == 'native_view_processing'
    with ref.native_reader() as reader:
        values, known = reader(0, 2)
    assert values[0, 1, 1] == 173
    assert not known[0, 0, 0]
    assert not known[0, 2, 2]  # Foreground added by reconstruction has no observed score.
    manifest = published_confidence_manifest(output)
    assert manifest == output / 'reconciliation_evidence' / 'manifest.json'
    published = json.loads(manifest.read_text())
    row = published['layers'][0]
    contract = row['observed_confidence']
    assert published['purpose'] == 'observed_prediction_confidence'
    assert row['score_semantics'] == 'maximum_surviving_observed_prediction_score_u8'
    assert contract['score_kind'] == kind
    assert contract['capture_confidence_floor'] == .21
    assert contract['component_min_conf'] == .30
    assert contract['generated_only_support'] == 'unknown'
    assert contract['quantization']['scale'] == 255
    assert contract == ref.metadata['provenance']['observed_confidence']
    # Native piece publication keeps the same observation contract without decoding.
    with redirect_stdout(io.StringIO()):
        joined = publish_native_confidence_pieces(
            [dict(reference=ref, offset_tyx=(0, 0, 0))], native_shape=shape,
            view=view, model_name='model', layer_key='joined')
    assert joined.metadata['provenance']['observed_confidence'] == contract


def test_new_retention_run_does_not_reuse_stale_manifest_layers(tmp_path):
    output = tmp_path / 'output'
    folder = output / 'reconciliation_evidence'
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text(json.dumps({'layers': [{'directory': 'old-run'}]}))
    configure_confidence_evidence(output, enabled=True, defer_projection=True,
        prediction_task='semantic', confidence_floor=.17)
    assert published_confidence_manifest(output) is None


@pytest.mark.parametrize('options', [
    {'confidence_floor': float('nan')}, {'confidence_floor': -1},
    {'min_conf': 2}, {'prediction_task': 'detector-unknown'},
])
def test_capture_contract_rejects_unrecordable_floors_or_task(options):
    with pytest.raises(ValueError):
        observed_confidence_contract(**options)


def test_missing_evidence_manifest_is_not_reported(tmp_path):
    assert published_confidence_manifest(tmp_path) is None


def test_transient_score_owner_releases_only_its_private_scratch(tmp_path):
    from XTA.confidence_evidence import current_published_confidence_manifest

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    marker = scratch / 'keep'
    marker.write_bytes(b'other scratch')
    out = tmp_path / 'out'
    catalog = out / 'reconciliation_evidence' / 'manifest.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(b'{"run_id":"old","layers":[]}')
    configure_confidence_evidence(out, enabled=True, transient_parent=scratch,
                                  run_id='current', defer_projection=True)
    mask = np.ones((2,5,7), np.uint8)
    view = geometry.get_view_infos(*mask.shape, cartesian_views=('transverse',))[0]
    ref = capture_prediction_confidence(mask, mask * 212, view=view, model_name='m', temp_dir=scratch)
    assert ref.path.is_relative_to(scratch) and ref.path.is_dir()
    assert current_published_confidence_manifest(out) is None
    configure_confidence_evidence(None, enabled=False)
    assert not ref.path.exists()
    assert marker.read_bytes() == b'other scratch'
    assert catalog.read_bytes() == b'{"run_id":"old","layers":[]}'


def test_receipt_owns_saved_scores_and_legacy_catalogues_remain_readable(tmp_path):
    directory = tmp_path / 'reconciliation_evidence'
    directory.mkdir()
    catalog = directory / 'manifest.json'
    catalog.write_bytes(b'{"run_id":"old","layers":[]}')
    assert published_confidence_manifest(tmp_path, run={}) == catalog
    assert published_confidence_manifest(tmp_path, run={'run_id':'old'}) == catalog
    with pytest.raises(ValueError, match='different runs'):
        published_confidence_manifest(tmp_path, run={'run_id':'new'})
    for observed in (None, {'retained':False}):
        receipt = {'run_id':'new', 'prediction_processing':{'observed_confidence':observed}}
        assert published_confidence_manifest(tmp_path, run=receipt) is None


def test_backend_capture_floors_do_not_hide_gpu_instance_prefilter():
    contract = observed_confidence_contract(prediction_task='segment', confidence_floor=.15,
        min_conf=.30, confidence_floor_by_backend={'cpu': .15, 'gpu': .30})
    assert contract['prediction_confidence_threshold'] == .15
    assert contract['capture_confidence_floor'] == .15
    assert contract['capture_confidence_floor_by_backend'] == {'cpu': .15, 'gpu': .30}
    gpu = observed_confidence_contract(prediction_task='segment', confidence_floor=.15,
        min_conf=.30, confidence_floor_by_backend={'gpu': .30})
    assert gpu['capture_confidence_floor'] == .30
