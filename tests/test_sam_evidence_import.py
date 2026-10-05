"""Retry composition copies immutable encoded evidence without reprocessing it."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from unittest import mock

import numpy as np
import pytest

from XTA import sam_evidence
from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, fingerprint
from tests.test_sam_evidence_policy import fixture_group, fixture_run


def _scope(**overrides):
    return dict(shape_tyx=[5, 12, 16], observation_snapshot_sha256='a' * 64,
                view_name='transverse', spacing_zyx=[1., 1., 1.], **overrides)


def _bundle(path, *, tiled=False, scope=None, group_id='family', extra_group=False):
    group, masks, raw = fixture_group(group_id, x=2)
    run = fixture_run(group_id + ':F', group)
    with SamEvidenceWriter(path, scope or _scope()) as writer:
        if extra_group:
            unused, unused_masks, _ = fixture_group('unused', x=8)
            writer.add_group(unused, unused_masks)
        writer.add_group(group, masks)
        if tiled:
            for tile_id, crop, owner, attempted in (
                    ('left', [0, 0, 12, 10], [0, 0, 12, 8], True),
                    ('right', [0, 6, 12, 16], [0, 8, 12, 16], False)):
                tile_raw = {frame: plane[:, crop[1]:crop[3]].copy() for frame, plane in raw.items()} if attempted else {}
                if attempted:
                    tile_raw[2][3, 9] = True  # Retained halo outside this tile's owned core.
                writer.add_run_tile(run['run_id'], dict(tile_id=tile_id, group_id=group_id,
                    crop_bbox_yx=crop, ownership_bbox_yx=owner, expected_frames=run['expected_frames'],
                    seed_ids=run['seed_ids'], attempted=attempted, tracker_scores={'2': .5}), tile_raw)
            available = {frame: np.indices(plane.shape)[1] < 8 for frame, plane in raw.items()}
        else:
            available = None
        writer.add_run(run, raw, availability_masks=available)
        return writer.commit()


def _encoded(bundle, key):
    record = bundle.records[key]
    with (bundle.directory / 'masks.bin').open('rb') as stream:
        stream.seek(record['offset'])
        return stream.read(record['bytes'])


def _rewrite_index(bundle, change):
    """Re-authenticate a corrupted descriptor to exercise per-record validation."""
    index_path = bundle.directory / 'index.json'
    index = json.loads(index_path.read_text())
    change(index)
    index_path.write_bytes(sam_evidence._json_bytes(index))
    manifest_path = bundle.directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name in ('index.json', 'masks.bin'):
        payload = (bundle.directory / name).read_bytes()
        manifest['files'][name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    manifest.pop('evidence_fingerprint')
    manifest['evidence_fingerprint'] = fingerprint(manifest)
    manifest_path.write_bytes(sam_evidence._json_bytes(manifest))
    return SamEvidenceBundle.open(bundle.directory)


@pytest.mark.parametrize('tiled', [False, True])
def test_encoded_import_preserves_all_packets_and_tiled_availability_without_codec(tmp_path, tiled):
    source = _bundle(tmp_path / 'source', tiled=tiled, extra_group=True)
    target_scope = {**sam_evidence._plain(source.scope), 'retry_policy': {'attempt': 1}}
    with SamEvidenceWriter(tmp_path / 'target', target_scope) as writer:
        with (mock.patch.object(source, 'assert_unchanged', wraps=source.assert_unchanged) as unchanged,
              mock.patch.object(sam_evidence, '_decode_mask', side_effect=AssertionError('decoded')),
              mock.patch.object(sam_evidence.zlib, 'compress', side_effect=AssertionError('recompressed'))):
            with writer.import_transaction(source):
                assert writer.import_group(source, 'family') == 'family'
                assert writer.import_run(source, 'family:F') == 'family:F'
            copied = writer.commit()
        assert unchanged.call_count == 2
    assert sam_evidence._plain(copied.groups['family']) == sam_evidence._plain(source.groups['family'])
    assert sam_evidence._plain(copied.runs['family:F']) == sam_evidence._plain(source.runs['family:F'])
    assert 'unused' not in copied.groups
    assert copied.evidence_fingerprint != source.evidence_fingerprint
    assert copied.scope['retry_policy']['attempt'] == 1
    assert any(copied.records[key]['offset'] != source.records[key]['offset'] for key in copied.records)
    for key in copied.records:
        assert _encoded(copied, key) == _encoded(source, key)
        assert {k: v for k, v in copied.records[key].items() if k != 'offset'} == {
            k: v for k, v in source.records[key].items() if k != 'offset'}
        np.testing.assert_array_equal(copied.mask(key), source.mask(key))
    if tiled:
        assert copied.tile_raw_mask('family:F', 'left', 2)[3, 9]
        assert not copied.raw_mask('family:F', 2)[3, 9]
        assert copied.availability_mask('family:F', 2)[:, :8].all()
        assert not copied.availability_mask('family:F', 2)[:, 8:].any()
        assert not copied.runs['family:F']['tile_evidence'][1]['attempted']
    expected = copied.raw_mask('family:F', 2)
    shutil.rmtree(source.directory)
    np.testing.assert_array_equal(copied.raw_mask('family:F', 2), expected)


def test_single_calls_are_safe_and_duplicates_unknowns_or_group_changes_do_not_copy(tmp_path):
    source = _bundle(tmp_path / 'source')
    with SamEvidenceWriter(tmp_path / 'target', _scope()) as writer:
        with pytest.raises(ValueError, match='unchanged imported group'):
            writer.import_run(source, 'family:F')
        writer.import_group(source, 'family')
        offset = writer._stream.tell()
        for identifier in ('family', 'missing'):
            with pytest.raises(ValueError, match='group'):
                writer.import_group(source, identifier)
        with pytest.raises(ValueError, match='run'):
            writer.import_run(source, 'missing')
        writer.groups['family']['complete'] = False
        with pytest.raises(ValueError, match='unchanged imported group'):
            writer.import_run(source, 'family:F')
        writer.groups['family']['complete'] = True
        assert writer._stream.tell() == offset
        writer.import_run(source, 'family:F')
        offset = writer._stream.tell()
        with pytest.raises(ValueError, match='run'):
            writer.import_run(source, 'family:F')
        assert writer._stream.tell() == offset
        assert set(writer.commit().runs) == {'family:F'}


def test_two_source_retry_composition_preserves_unaffected_ids_and_fresh_retry_ids(tmp_path):
    original = _bundle(tmp_path / 'original', group_id='unaffected')
    retry = _bundle(tmp_path / 'retry', group_id='affected.retry1',
                    scope={**_scope(), 'retry_attempt': 1})
    with SamEvidenceWriter(tmp_path / 'final', {**_scope(), 'retry_controller': 'final'}) as writer:
        for source, group_id in ((original, 'unaffected'), (retry, 'affected.retry1')):
            with mock.patch.object(source, 'assert_unchanged', wraps=source.assert_unchanged) as unchanged:
                with writer.import_transaction(source):
                    writer.import_group(source, group_id)
                    writer.import_run(source, group_id + ':F')
            assert unchanged.call_count == 2
        final = writer.commit()
    assert set(final.groups) == {'unaffected', 'affected.retry1'}
    assert set(final.runs) == {'unaffected:F', 'affected.retry1:F'}
    for source in (original, retry):
        for key in source.records:
            assert _encoded(final, key) == _encoded(source, key)


@pytest.mark.parametrize('changed', [
    {'observation_snapshot_sha256': 'b' * 64},
    {'observation_snapshot_sha256': ''},
    {'shape_tyx': [6, 12, 16]},
    {'view_name': 'azimuthal'},
    {'spacing_zyx': [2., 1., 1.]},
    {'canvas_transform': {'mirror': True}},
    {'evidence_purpose': 'sam_extrapolation'},
    {'input_fingerprints': {'original_snapshot': 'b' * 64}},
    {'frozen_observation_snapshot_sha256': 'b' * 64},
])
def test_foreign_or_unpinned_scope_rejected_before_payload_copy(tmp_path, changed):
    source = _bundle(tmp_path / 'source')
    with SamEvidenceWriter(tmp_path / 'target', {**_scope(), **changed}) as writer:
        with pytest.raises(ValueError, match='snapshot|observation|geometry|purpose'):
            writer.import_group(source, 'family')
        assert writer._stream.tell() == 0 and not writer.groups


@pytest.mark.parametrize('changed', [
    {'scope_id': 'another-view'},
    {'image_snapshot_sha256': 'b' * 64},
    {'sam_model': {'checkpoint_sha256': 'b' * 64}},
    {'sam_runtime': {'package_tree_sha256': 'b' * 64}},
    {'sam_bundle_identity': 'different-bundle'},
    {'model_identity': 'different-model'},
    {'source_identity': 'another-source'},
    {'sampler_source_sha256': 'b' * 64},
    {'input_fingerprints': {'image_snapshot': 'b' * 64}},
    {'input_fingerprints': None},
    {'gate_support_fingerprints': {'upstream': 'different'}},
])
def test_different_source_images_models_or_upstream_identities_cannot_be_relabeled(tmp_path, changed):
    source = _bundle(tmp_path / 'source')
    with SamEvidenceWriter(tmp_path / 'target', {**_scope(), **changed}) as writer:
        with pytest.raises(ValueError, match='source|identity|fingerprints'):
            writer.import_group(source, 'family')
        assert writer._stream.tell() == 0 and not writer.groups


def test_retry_image_demand_metadata_does_not_change_the_canonical_image_identity(tmp_path):
    canonical = {**_scope(), 'image_snapshot_sha256': 'c' * 64,
                 'input_fingerprints': {'image_snapshot': 'c' * 64}}
    source = _bundle(tmp_path / 'source', scope=canonical)
    retry_scope = {**canonical, 'retry_image_snapshot_sha256': 'd' * 64,
                   'retry_image_frame_crops': {'2': [0, 0, 12, 16]},
                   'input_fingerprints': {**canonical['input_fingerprints'], 'retry_tiling_plan': 'attempt1'}}
    with SamEvidenceWriter(tmp_path / 'target', retry_scope) as writer:
        with writer.import_transaction(source):
            writer.import_group(source, 'family')
            writer.import_run(source, 'family:F')
        result = writer.commit()
    assert result.scope['image_snapshot_sha256'] == 'c' * 64
    assert result.scope['retry_image_snapshot_sha256'] == 'd' * 64


def test_unverified_bundle_and_mismatched_transaction_source_are_rejected(tmp_path):
    source = _bundle(tmp_path / 'source')
    unverified = SamEvidenceBundle.open(source.directory, verify=False)
    other = SamEvidenceBundle.open(source.directory)
    with SamEvidenceWriter(tmp_path / 'target', _scope()) as writer:
        with pytest.raises(ValueError, match='verified'):
            writer.import_group(unverified, 'family')
        with writer.import_transaction(source):
            with pytest.raises(ValueError, match='switch source'):
                writer.import_group(other, 'family')
            with pytest.raises(ValueError, match='nested'):
                with writer.import_transaction(source):
                    pass
            writer.import_group(source, 'family')
        writer.commit()


@pytest.mark.parametrize('limit', ['mask', 'payload'])
def test_import_respects_budgets_before_any_packet_copy(tmp_path, limit):
    source = _bundle(tmp_path / 'source')
    budgets = {'max_mask_bytes': 12 * 16 - 1} if limit == 'mask' else {'max_payload_bytes': 1}
    writer = SamEvidenceWriter(tmp_path / 'target', _scope(), **budgets)
    with pytest.raises(MemoryError if limit == 'mask' else OSError, match='limit'):
        writer.import_group(source, 'family')
    assert not writer.directory.exists() and not writer.staging.exists()


@pytest.mark.parametrize('mutation', [
    lambda record: record.update(shape=[12, 15]),
    lambda record: record.update(packed_bytes=999),
    lambda record: record.update(foreground=1000),
    lambda record: record.update(compressed_sha256='0' * 64),
    lambda record: record.update(sha256='not-a-sha'),
])
def test_malformed_or_corrupt_encoded_record_aborts_private_stage(tmp_path, mutation):
    source = _bundle(tmp_path / 'source')
    key = source.groups['family']['mask_keys']['write:2']
    source = _rewrite_index(source, lambda index: mutation(index['masks'][key]))
    writer = SamEvidenceWriter(tmp_path / 'target', _scope())
    with pytest.raises(ValueError, match='Malformed|checksum'):
        writer.import_group(source, 'family')
    assert not writer.directory.exists() and not writer.staging.exists()


@pytest.mark.parametrize('mutation', [
    lambda run: run['candidate_mask_keys'].pop('2'),
    lambda run: run.update(availability_mask_keys={}),
    lambda run: run['tile_evidence'][0].update(parent_run_id='other'),
    lambda run: run['tile_evidence'][0].update(raw_mask_keys={'2': 'missing'}),
    lambda run: run.update(tile_evidence_schema='unknown'),
])
def test_malformed_nested_tile_or_frame_inventory_cannot_publish(tmp_path, mutation):
    source = _bundle(tmp_path / 'source', tiled=True)
    source = _rewrite_index(source, lambda index: mutation(index['runs']['family:F']))
    writer = SamEvidenceWriter(tmp_path / 'target', _scope())
    with pytest.raises(ValueError, match='Malformed'):
        with writer.import_transaction(source):
            writer.import_group(source, 'family')
            writer.import_run(source, 'family:F')
    assert not writer.directory.exists() and not writer.staging.exists()


def test_truncation_after_boundary_check_aborts_and_checks_source_only_twice(tmp_path, monkeypatch):
    source = _bundle(tmp_path / 'source')
    original_check = source.assert_unchanged
    checks = []

    def truncate_after_check():
        checks.append(True)
        original_check()
        if len(checks) == 1:
            (source.directory / 'masks.bin').write_bytes(b'')

    monkeypatch.setattr(source, 'assert_unchanged', truncate_after_check)
    writer = SamEvidenceWriter(tmp_path / 'target', _scope())
    with pytest.raises(ValueError, match='Truncated'):
        writer.import_group(source, 'family')
    assert len(checks) == 1  # Failure stops before the final whole-file check.
    assert not writer.directory.exists() and not writer.staging.exists()


def test_final_boundary_check_rejects_source_mutation_and_in_transaction_publication(tmp_path):
    source = _bundle(tmp_path / 'source')
    writer = SamEvidenceWriter(tmp_path / 'target', _scope())
    with pytest.raises(ValueError, match='changed'):
        with writer.import_transaction(source):
            writer.import_group(source, 'family')
            writer.import_run(source, 'family:F')
            with pytest.raises(RuntimeError, match='finish before publication'):
                writer.commit()
            with (source.directory / 'masks.bin').open('ab') as stream:
                stream.write(b'changed')
    assert not writer.directory.exists() and not writer.staging.exists()


def test_initial_boundary_failure_aborts_private_stage(tmp_path):
    source = _bundle(tmp_path / 'source')
    with (source.directory / 'masks.bin').open('ab') as stream:
        stream.write(b'changed')
    writer = SamEvidenceWriter(tmp_path / 'target', _scope())
    with pytest.raises(ValueError, match='changed'):
        writer.import_group(source, 'family')
    assert not writer.directory.exists() and not writer.staging.exists()


def test_retry_policy_metadata_allowed_but_scope_mutation_during_import_aborts(tmp_path):
    source = _bundle(tmp_path / 'source')
    writer = SamEvidenceWriter(tmp_path / 'target', {**_scope(), 'retry_policy': {'attempt': 1}})
    with pytest.raises(ValueError, match='scope changed'):
        with writer.import_transaction(source):
            writer.import_group(source, 'family')
            writer.scope['retry_policy']['attempt'] = 2
    assert not writer.directory.exists() and not writer.staging.exists()


def test_import_reads_large_encoded_packet_in_bounded_chunks(tmp_path, monkeypatch):
    shape = (1024, 1024)
    scope = {**_scope(), 'shape_tyx': [1, *shape]}
    group = dict(group_id='unresolved', context_bbox_yx=[0, 0, *shape], frame_indices=[0],
                 endpoints=[], complete=False, status='unresolved')
    mask = np.random.default_rng(2).integers(0, 2, size=shape, dtype=np.uint8)
    with SamEvidenceWriter(tmp_path / 'source', scope) as source_writer:
        source_writer.add_group(group, {'endpoint_local:original': mask})
        source = source_writer.commit()
    requests, original_open = [], Path.open

    class RecordingReader:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def seek(self, offset):
            return self.stream.seek(offset)
        def read(self, size):
            requests.append(size)
            return self.stream.read(size)

    def open_recording(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == source.directory / 'masks.bin' and args == ('rb',):
            return RecordingReader(stream)
        return stream

    with SamEvidenceWriter(tmp_path / 'target', scope) as writer:
        with writer.import_transaction(source):
            with monkeypatch.context() as patch:
                patch.setattr(Path, 'open', open_recording)
                writer.import_group(source, 'unresolved')
        copied = writer.commit()
    assert len(requests) >= 3 and max(requests) <= 64 * 1024
    np.testing.assert_array_equal(copied.group_mask('unresolved', 'endpoint_local:original'), mask)


def test_cyclic_frame_addresses_survive_import_and_foreign_period_rejected(tmp_path):
    from tests.test_sam_cyclic_evidence_replay import fixture
    scope, group, masks, run, raw = fixture()
    scope['observation_snapshot_sha256'] = 'a' * 64
    with SamEvidenceWriter(tmp_path / 'source', scope) as writer:
        writer.add_group(group, masks)
        writer.add_run(run, raw)
        source = writer.commit()
    with SamEvidenceWriter(tmp_path / 'target', scope) as writer:
        with writer.import_transaction(source):
            writer.import_group(source, 'seam')
            writer.import_run(source, 'forward')
        copied = writer.commit()
    assert sam_evidence._plain(copied.scope) == sam_evidence._plain(source.scope)
    assert sam_evidence._plain(copied.groups) == sam_evidence._plain(source.groups)
    foreign = deepcopy(scope)
    foreign['frame_addressing']['period_degrees'] = 360.
    # Recreate a valid foreign closure rather than corrupting the existing map.
    from XTA.sam_cyclic import build_cyclic_frame_addressing
    foreign['frame_addressing'] = dict(build_cyclic_frame_addressing((4, 12, 16), 2, period_degrees=360.))
    with SamEvidenceWriter(tmp_path / 'foreign', foreign) as writer:
        with pytest.raises(ValueError, match='frame geometry'):
            writer.import_group(source, 'seam')
