"""Overlapping confidence copies publish last and release failed attempt ownership."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import confidence_evidence as evidence
from XTA import confidence_native as subject


def fixture(root, *, legacy=False):
    shape = (3, 7, 9)
    expected = np.zeros(shape, np.uint8)
    pieces = []
    for number, (offset, score) in enumerate((((0, 0, 1), 173), ((1, 2, 3), 241))):
        values = np.full((2, 4, 5), score, np.uint8)
        values[:, 1, 1] = 0
        values[0, 0, 0] = 1
        writer = evidence.write_confidence_evidence if legacy else evidence.write_block_confidence_evidence
        reference = writer(root/f'input-{number}', values.shape, lambda z: values[z],
            model_name='m', layer_key=f'tile-{number}',
            provenance={'coordinate_space': 'tile_native_processing'})
        pieces.append(dict(reference=reference, offset_tyx=offset))
        z, y, x = offset
        target = expected[z:z+2, y:y+4, x:x+5]
        np.maximum(target, values, out=target)
    kwargs = dict(native_shape=shape, source_shape=(5, 11, 13), model_name='m',
                  layer_key='parent', provenance={'view': 'geometry-preserved'}, disjoint=False)
    return pieces, expected, kwargs


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def assert_roundtrip(destination, pieces, expected, kwargs):
    ref = subject.write_native_pieces(destination, pieces, **kwargs)
    assert ref.layer_key == 'parent'
    assert ref.model_name == 'm'
    assert ref.storage_shape == expected.shape
    assert ref.source_shape == kwargs['source_shape']
    assert ref.metadata['known_voxels'] is None
    assert ref.metadata['known_contributions'] == sum(p['reference'].metadata['known_voxels'] for p in pieces)
    assert ref.metadata['provenance'] == kwargs['provenance']
    assert {p.name for p in destination.iterdir()} == {'pieces', 'metadata.json'}
    with ref.native_reader() as reader:
        actual, known = reader(0, expected.shape[0])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(known, expected != 0)
    for record, piece in zip(ref.metadata['pieces'], pieces):
        for name, digest in record['copied_sha256'].items():
            data = (destination/record['directory']/name).read_bytes()
            assert data == (piece['reference'].path/name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
    return ref


@pytest.mark.parametrize('legacy', [False, True])
def test_overlapping_pieces_preserve_maximum_unknown_identity_and_inputs(tmp_path, legacy):
    pieces, expected, kwargs = fixture(tmp_path/'inputs', legacy=legacy)
    before = snapshot(tmp_path/'inputs')
    with mock.patch('zlib.decompressobj', side_effect=AssertionError('copy decoded scores')):
        subject.write_native_pieces(tmp_path/'no-decode', pieces, **kwargs)
    assert_roundtrip(tmp_path/'output', pieces, expected, kwargs)
    assert snapshot(tmp_path/'inputs') == before


@pytest.mark.parametrize('number', [0, 1])
def test_checksum_failure_removes_every_attempt_file_and_same_path_retry_succeeds(tmp_path, number):
    pieces, expected, kwargs = fixture(tmp_path/'inputs')
    payload = pieces[number]['reference'].path/'scores.u8.zlib'
    original = payload.read_bytes()
    payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    before = snapshot(tmp_path/'inputs')
    destination = tmp_path/'output'
    with pytest.raises(ValueError, match='changed before/during persistent copy'):
        subject.write_native_pieces(destination, pieces, **kwargs)
    assert not destination.exists()
    assert snapshot(tmp_path/'inputs') == before
    payload.write_bytes(original)
    assert_roundtrip(destination, pieces, expected, kwargs)


def test_midcopy_io_error_preserves_original_exception_and_retries(tmp_path):
    pieces, expected, kwargs = fixture(tmp_path/'inputs')
    before = snapshot(tmp_path/'inputs')
    destination = tmp_path/'output'
    failure = OSError('injected partial numeric write')
    original = Path.open

    class PartialWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:3])
            raise failure

    def open_file(path, mode='r', *args, **kw):
        stream = original(path, mode, *args, **kw)
        if mode == 'xb' and path.name == 'scores.u8.zlib' and path.is_relative_to(destination):
            return PartialWriter(stream)
        return stream

    with mock.patch.object(Path, 'open', open_file), pytest.raises(OSError) as caught:
        subject.write_native_pieces(destination, pieces, **kwargs)
    assert caught.value is failure
    assert not destination.exists()
    assert snapshot(tmp_path/'inputs') == before
    assert_roundtrip(destination, pieces, expected, kwargs)


@pytest.mark.parametrize('stage', ['later_piece', 'child_metadata', 'parent_metadata', 'final_marker'])
def test_late_and_metadata_failures_leave_no_commit_and_retry_same_destination(tmp_path, stage):
    pieces, expected, kwargs = fixture(tmp_path/'inputs')
    before = snapshot(tmp_path/'inputs')
    destination = tmp_path/'output'
    failure = OSError(f'injected {stage}')
    copy, write, replace = subject._copy_numeric_file, evidence._write_json_atomic, Path.replace

    def copy_file(source, target, expected=None):
        result = copy(source, target, expected)
        if Path(target).parent.name == '000001':
            raise failure
        return result

    def write_json(path, value):
        is_parent = value.get('layout') == 'native_pieces'
        if (stage == 'parent_metadata') == is_parent:
            path.with_name(path.name+'.partial').write_text('partial')
            raise failure
        return write(path, value)

    def replace_file(path, target):
        target = Path(target)
        if target == destination/'metadata.json':
            assert (destination/'pieces'/'000001'/'metadata.json').exists()
            assert not target.exists()
            raise failure
        return replace(path, target)

    patch = (mock.patch.object(subject, '_copy_numeric_file', copy_file) if stage == 'later_piece' else
             mock.patch.object(Path, 'replace', replace_file) if stage == 'final_marker' else
             mock.patch.object(evidence, '_write_json_atomic', write_json))
    with patch, pytest.raises(OSError) as caught:
        subject.write_native_pieces(destination, pieces, **kwargs)
    assert caught.value is failure
    assert not destination.exists()
    assert snapshot(tmp_path/'inputs') == before
    assert_roundtrip(destination, pieces, expected, kwargs)


@pytest.mark.parametrize('committed', [False, True])
def test_preexisting_destination_and_source_are_never_modified(tmp_path, committed):
    pieces, expected, kwargs = fixture(tmp_path/'inputs')
    destination = tmp_path/'output'
    if committed:
        assert_roundtrip(destination, pieces, expected, kwargs)
    else:
        destination.mkdir()
        (destination/'user-file').write_text('keep')
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        subject.write_native_pieces(destination, pieces, **kwargs)
    with pytest.raises(ValueError, match='separate'):
        subject.write_native_pieces(pieces[0]['reference'].path/'child', pieces, **kwargs)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('change', ['identity', 'source', 'offset', 'stale', 'count', 'index_checksum'])
def test_invalid_identity_hash_or_geometry_never_publishes(tmp_path, change):
    pieces, _, kwargs = fixture(tmp_path/'inputs')
    if change == 'identity':
        kwargs['model_name'] = 'wrong-model'
    elif change == 'source':
        path = pieces[0]['reference'].path/'metadata.json'
        metadata = json.loads(path.read_text())
        metadata['source_shape_tyx'] = [5, 11, 12]
        metadata['output_shape_tyx'] = [5, 11, 12]
        path.write_text(json.dumps(metadata))
        pieces[0]['reference'] = evidence.ConfidenceEvidenceRef.open(path.parent)
    elif change == 'offset':
        pieces[0]['offset_tyx'] = (0, 0, 1.5)
    else:
        path = pieces[0]['reference'].path/'metadata.json'
        metadata = json.loads(path.read_text())
        metadata[{'stale': 'layer_key', 'count': 'known_voxels', 'index_checksum': 'index_sha256'}[change]] = {
            'stale': 'changed-identity', 'count': -1, 'index_checksum': '0'*64}[change]
        path.write_text(json.dumps(metadata))
        if change != 'stale':
            pieces[0]['reference'] = evidence.ConfidenceEvidenceRef.open(path.parent)
    before = snapshot(tmp_path/'inputs')
    with pytest.raises(ValueError):
        subject.write_native_pieces(tmp_path/'output', pieces, **kwargs)
    assert not (tmp_path/'output').exists()
    assert snapshot(tmp_path/'inputs') == before


def test_input_metadata_change_during_copy_is_rejected_and_retry_succeeds(tmp_path):
    pieces, expected, kwargs = fixture(tmp_path/'inputs')
    path = pieces[0]['reference'].path/'metadata.json'
    original_metadata, copy = path.read_bytes(), subject._copy_numeric_file

    def mutate(source, target, expected=None):
        result = copy(source, target, expected)
        if Path(target).parent.name == '000001':
            path.write_bytes(original_metadata+b' ')
        return result

    with mock.patch.object(subject, '_copy_numeric_file', mutate), pytest.raises(ValueError, match='metadata changed during'):
        subject.write_native_pieces(tmp_path/'output', pieces, **kwargs)
    assert not (tmp_path/'output').exists()
    path.write_bytes(original_metadata)
    assert_roundtrip(tmp_path/'output', pieces, expected, kwargs)


def test_cleanup_error_never_masks_copy_error_or_removes_unowned_files(tmp_path):
    pieces, _, kwargs = fixture(tmp_path/'inputs')
    destination = tmp_path/'output'
    failure = OSError('primary copy failure')

    def fail_copy(*args):
        (destination/'foreign-file').write_text('preserve')
        raise failure

    with mock.patch.object(subject, '_copy_numeric_file', fail_copy), mock.patch.object(
            subject.shutil, 'rmtree', side_effect=OSError('cleanup denied')), pytest.raises(OSError) as caught:
        subject.write_native_pieces(destination, pieces, **kwargs)
    assert caught.value is failure
    assert (destination/'foreign-file').read_text() == 'preserve'
    assert not (destination/'metadata.json').exists()


def test_empty_piece_set_roundtrips_without_dense_copy(tmp_path):
    values = np.zeros((2, 3, 4), np.uint8)
    kwargs = dict(native_shape=values.shape, source_shape=values.shape, layer_key='parent',
                  model_name='m', provenance={})
    assert_roundtrip(tmp_path/'output', [], values, kwargs)
