"""Canonical bounded indexes release writer ownership before immutable reload."""
from collections.abc import Mapping
import json
from pathlib import Path
from types import MappingProxyType
import weakref

import numpy as np
import pytest

from XTA import sam_evidence as evidence
from XTA.artifact_archive import read_artifact, reference
from tests.test_sam_evidence_policy import fixture_group, fixture_run


def populate(writer, tmp_path):
    group, masks, raw = fixture_group()
    group['endpoints'][0]['lineage'] = dict(source=tmp_path/'delta-δ',
        label=np.int64(8), confidence=np.float32(.5), flags=(np.bool_(True), False))
    writer.add_group(group, masks)
    run = fixture_run('forward', group)
    run['runtime_receipt'] = dict(score=np.float64(.25), source=tmp_path/'source')
    writer.add_run(run, raw)
    return raw


@pytest.mark.parametrize('archived', [False, True])
def test_streamed_index_is_byte_identical_to_legacy_normalized_serializer(tmp_path, monkeypatch, archived):
    path = Path(reference(tmp_path/'old.tar', 'scope/evidence')) if archived else tmp_path/'bundle'
    with evidence.SamEvidenceWriter(path, dict(source=tmp_path/'input', rank=np.int64(3))) as writer:
        raw = populate(writer, tmp_path)
        expected = evidence._json_bytes(dict(groups=writer.groups, runs=writer.runs, masks=writer.records))
        original = evidence._json_bytes
        def no_full_index_copy(value):
            if isinstance(value, Mapping) and {'groups', 'runs', 'masks'} <= set(value):
                pytest.fail('index encoding must not copy the complete metadata tree')
            return original(value)
        monkeypatch.setattr(evidence, '_json_bytes', no_full_index_copy)
        bundle = writer.commit()
    assert read_artifact(bundle.directory/'index.json') == expected
    assert '\\u03b4' in expected.decode('ascii')
    assert writer.groups is bundle.groups and writer.runs is bundle.runs and writer.records is bundle.records
    assert isinstance(writer.groups, MappingProxyType)
    np.testing.assert_array_equal(bundle.raw_mask('forward', 2), raw[2])


def test_writer_tree_owners_are_gone_before_reloading_index(tmp_path, monkeypatch):
    class OwnedDict(dict):
        pass
    with evidence.SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        populate(writer, tmp_path)
        for name in ('groups', 'runs', 'records', '_pending_tiles'):
            setattr(writer, name, OwnedDict(getattr(writer, name)))
        owners = [weakref.ref(getattr(writer, name)) for name in ('groups', 'runs', 'records', '_pending_tiles')]
        original = evidence.SamEvidenceBundle.open
        def checked(directory, **kwargs):
            assert all(owner() is None for owner in owners)
            return original(directory, **kwargs)
        monkeypatch.setattr(evidence.SamEvidenceBundle, 'open', checked)
        bundle = writer.commit()
    assert writer.groups is bundle.groups and writer._pending_tiles is bundle.unfinalized_tile_runs


def test_owned_freeze_seals_every_container_and_global_freeze_preserves_borrowed_input():
    owned = evidence._freeze_owned(json.loads('{"group":{"frames":[1,2],"nested":[{"flag":true}]}}'))
    assert isinstance(owned, MappingProxyType) and isinstance(owned['group'], MappingProxyType)
    assert owned['group']['frames'] == (1, 2)
    with pytest.raises(TypeError):
        owned['group']['nested'][0]['flag'] = False
    with pytest.raises(TypeError):
        owned['group']['frames'][0] = 3
    borrowed = {'nested': [{'frames': [1, 2]}]}
    sealed = evidence._freeze(borrowed)
    borrowed['nested'][0]['frames'][0] = 99
    assert sealed['nested'][0]['frames'] == (1, 2)


def test_index_writer_enforces_the_exact_byte_limit(tmp_path, monkeypatch):
    value = dict(groups={}, runs={}, masks={}, unicode='δ', escaped='a\nb')
    expected = evidence._json_bytes(value)
    monkeypatch.setattr(evidence, 'INDEX_MAX_BYTES', len(expected))
    evidence._write_index(tmp_path/'exact.json', value)
    assert (tmp_path/'exact.json').read_bytes() == expected
    monkeypatch.setattr(evidence, 'INDEX_MAX_BYTES', len(expected)-1)
    with pytest.raises(MemoryError, match='index exceeds bounded'):
        evidence._write_index(tmp_path/'too-large.json', value)
    assert (tmp_path/'too-large.json').stat().st_size <= len(expected)-1


@pytest.mark.parametrize('archived', [False, True])
def test_oversized_index_never_publishes_complete_evidence(tmp_path, monkeypatch, archived):
    from XTA.artifact_archive import artifact_exists
    path = Path(reference(tmp_path/'failed.tar', 'scope/evidence')) if archived else tmp_path/'failed'
    with pytest.raises(MemoryError, match='index exceeds bounded'):
        with evidence.SamEvidenceWriter(path, {}) as writer:
            populate(writer, tmp_path)
            stage = writer.staging
            monkeypatch.setattr(evidence, 'INDEX_MAX_BYTES', 32)
            writer.commit()
    assert writer._closed and writer._stream.closed and not stage.exists()
    assert not artifact_exists(path)


def test_index_limit_is_separate_from_manifest_and_mask_bounds(tmp_path, monkeypatch):
    assert evidence.INDEX_MAX_BYTES == 128 * 1024**2
    calls = []
    original = evidence.read_artifact
    def bounded(path, **kwargs):
        calls.append((Path(path).name, kwargs.get('max_bytes')))
        return original(path, **kwargs)
    monkeypatch.setattr(evidence, 'read_artifact', bounded)
    with evidence.SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        populate(writer, tmp_path)
        bundle = writer.commit()
    limits = dict(calls)
    assert limits['index.json'] == 128 * 1024**2
    assert limits['manifest.json'] == bundle.max_mask_bytes == 64 * 1024**2


def test_reader_rejects_declared_index_over_limit(tmp_path, monkeypatch):
    with evidence.SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        populate(writer, tmp_path)
        bundle = writer.commit()
    monkeypatch.setattr(evidence, 'INDEX_MAX_BYTES', bundle.manifest['files']['index.json']['bytes']-1)
    with pytest.raises(MemoryError, match='index exceeds bounded'):
        evidence.SamEvidenceBundle.open(bundle.directory)


def test_actual_read_bound_remains_when_reported_size_is_stale(tmp_path, monkeypatch):
    with evidence.SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        populate(writer, tmp_path)
        bundle = writer.commit()
    size = bundle.manifest['files']['index.json']['bytes']
    with (bundle.directory/'index.json').open('ab') as stream:
        stream.write(b' ' * 32)
    monkeypatch.setattr(evidence, 'INDEX_MAX_BYTES', size+16)
    original = evidence.artifact_size
    monkeypatch.setattr(evidence, 'artifact_size',
        lambda path: size if Path(path).name == 'index.json' else original(path))
    with pytest.raises(ValueError, match='bounded read budget'):
        evidence.SamEvidenceBundle.open(bundle.directory, verify=False)


@pytest.mark.parametrize('name', ['index.json', 'masks.bin'])
def test_index_reload_does_not_skip_payload_source_integrity(tmp_path, name):
    with evidence.SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        populate(writer, tmp_path)
        bundle = writer.commit()
    path = bundle.directory/name
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match='checksum mismatch'):
        evidence.SamEvidenceBundle.open(bundle.directory)


def test_nonfinite_index_metadata_still_cannot_publish(tmp_path):
    with pytest.raises(ValueError, match='Out of range float'):
        with evidence.SamEvidenceWriter(tmp_path/'failed', {}) as writer:
            populate(writer, tmp_path)
            writer.runs['forward']['runtime_receipt']['invalid'] = float('nan')
            writer.commit()
    assert not (tmp_path/'failed').exists()


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le',
                                    'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be'])
@pytest.mark.parametrize('archived', [False, True])
def test_index_decode_preserves_legacy_encoding_and_surrogate_semantics(tmp_path, encoding, archived):
    from XTA.artifact_archive import publish_directory
    with evidence.SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        populate(writer, tmp_path)
        bundle = writer.commit()
    path = bundle.directory/'index.json'
    index = json.loads(path.read_bytes())
    index['runs']['forward']['runtime_receipt'].update(unicode='delta-δ', surrogate='lone-\ud800')
    text = json.dumps(index, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    encoded = text.encode(encoding, 'surrogatepass')
    path.write_bytes(encoded)
    manifest_path = bundle.directory/'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['files']['index.json'] = dict(bytes=len(encoded), sha256=evidence._file_hash(path))
    manifest.pop('evidence_fingerprint')
    manifest['evidence_fingerprint'] = evidence.fingerprint(manifest)
    manifest_path.write_bytes(evidence._json_bytes(manifest))
    directory = bundle.directory
    if archived:
        directory = Path(reference(tmp_path/'encoded.tar', 'legacy/evidence'))
        publish_directory(bundle.directory, directory)
    actual = evidence.SamEvidenceBundle.open(directory)
    assert evidence._plain(actual.runs) == json.loads(encoded)['runs']


def test_encoded_bytes_and_decoded_text_leave_scope_before_next_large_allocation(tmp_path, monkeypatch):
    with evidence.SamEvidenceWriter(tmp_path/'bundle', {}) as writer:
        populate(writer, tmp_path)
        bundle = writer.commit()
    events = []
    class Text(str):
        def __del__(self):
            events.append('text_released')
    class Encoded(bytes):
        def decode(self, *args, **kwargs):
            events.append('decode')
            return Text(super().decode(*args, **kwargs))
        def __del__(self):
            events.append('bytes_released')
    original_read, original_loads, original_freeze = evidence.read_artifact, json.loads, evidence._freeze_owned
    def read(path, **kwargs):
        value = original_read(path, **kwargs)
        return Encoded(value) if Path(path).name == 'index.json' else value
    def loads(value, *args, **kwargs):
        if isinstance(value, Text):
            assert 'bytes_released' in events
            events.append('parse')
        return original_loads(value, *args, **kwargs)
    def freeze(value):
        if isinstance(value, dict) and {'groups', 'runs', 'masks'} <= set(value):
            assert 'text_released' in events
            events.append('freeze')
        return original_freeze(value)
    monkeypatch.setattr(evidence, 'read_artifact', read)
    monkeypatch.setattr(json, 'loads', loads)
    monkeypatch.setattr(evidence, '_freeze_owned', freeze)
    evidence.SamEvidenceBundle.open(bundle.directory)
    assert events == ['decode', 'bytes_released', 'parse', 'text_released', 'freeze']
