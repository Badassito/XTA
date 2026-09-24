"""Direct confidence lease publication preserves payloads and fails atomically."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import confidence_consolidation as subject
from XTA.confidence_evidence import ConfidenceEvidenceRef, write_block_confidence_evidence
from XTA.confidence_native import write_native_disjoint_leases, write_native_pieces
from XTA.confidence_storage import BLOCK_DTYPE, BLOCK_LAYOUT
from XTA.geometry import get_view_infos


def fixture(root, *, empty=False):
    values = np.zeros((7,9,17), np.uint8)
    if not empty:
        values[0,0,0] = 1
        values[1,:,8:] = 128
        values[4,1:7,1:16] = 127
        values[6,-1,-1] = 255
    source_shape = (6,13,19)
    pieces = []
    for start, stop in ((0,2),(2,4),(4,7)):
        array = values[start:stop]
        reference = write_block_confidence_evidence(root/f'raw-{start}', array.shape, lambda z: array[z],
            layer_key='k', model_name='m', coordinate_space='native_view_processing',
            source_shape_tyx=source_shape, provenance={'slice_start': start, 'slice_count': stop-start})
        pieces.append(dict(reference=reference, offset_tyx=(start,0,0)))
    kwargs = dict(native_shape=values.shape, source_shape=source_shape, layer_key='k', model_name='m',
        provenance={'view': asdict(get_view_infos(7,9,17, cartesian_views=('transverse',))[0]),
                    'source':'fullframe', 'geometry':[1.2,3,4]})
    return pieces, values, kwargs


def change_metadata(piece, callback):
    path = piece['reference'].path/'metadata.json'
    metadata = json.loads(path.read_text())
    callback(metadata)
    path.write_text(json.dumps(metadata))
    piece['reference'] = ConfidenceEvidenceRef.open(path.parent)


def assert_rejected(root, pieces, kwargs, message):
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        subject.write_disjoint_leases(root/'output', pieces, chunk_records=1, chunk_bytes=7, **kwargs)
    assert not (root/'output').exists()


def test_exact_scores_support_geometry_and_compressed_bytes_without_dense_copy(tmp_path):
    pieces, values, kwargs = fixture(tmp_path)
    combined = b''.join((p['reference'].path/'scores.u8.zlib').read_bytes() for p in pieces)
    with mock.patch('zlib.compress', side_effect=AssertionError('recompressed')), mock.patch(
            'zlib.decompressobj', side_effect=AssertionError('decoded')), mock.patch(
            'numpy.zeros', side_effect=AssertionError('dense allocation')):
        result = subject.write_disjoint_leases(tmp_path/'output', pieces, chunk_records=1, chunk_bytes=7, **kwargs)
    assert result.metadata['layout'] == BLOCK_LAYOUT
    assert result.storage_shape == values.shape
    assert result.source_shape == (6,13,19)
    assert result.metadata['provenance'] == json.loads(json.dumps(kwargs['provenance']))
    assert result.metadata['known_voxels'] == result.metadata['known_contributions'] == np.count_nonzero(values)
    assert result.metadata['quantization'] == pieces[0]['reference'].metadata['quantization']
    with result.native_reader() as reader:
        actual, known = reader(0,len(values))
    np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(known, values>0)
    assert (result.path/'scores.u8.zlib').read_bytes() == combined
    assert result.metadata['payload_sha256'] == hashlib.sha256(combined).hexdigest()
    assert sorted(p.name for p in result.path.iterdir()) == ['index.bin', 'metadata.json', 'scores.u8.zlib']
    assert [entry['metadata'] for entry in result.metadata['consolidation']['leases']] == [
        p['reference'].metadata for p in pieces]
    with pytest.raises(ValueError, match='explicit source_reader'):
        result.reader()


def test_empty_leases_remain_valid_and_wrapper_accepts_paths(tmp_path):
    pieces, values, kwargs = fixture(tmp_path, empty=True)
    for piece in pieces:
        piece['reference'] = piece['reference'].path
    result = write_native_disjoint_leases(tmp_path/'output', pieces, **kwargs)
    assert result.metadata['block_count'] == result.metadata['payload_bytes'] == 0
    with result.native_reader() as reader:
        actual, known = reader(0,7)
    np.testing.assert_array_equal(actual, values)
    assert not known.any()


def test_explicit_source_projection_matches_piece_layout(tmp_path):
    pieces, _, kwargs = fixture(tmp_path)
    original = write_native_pieces(tmp_path/'pieces', pieces, disjoint=True, **kwargs)
    result = write_native_disjoint_leases(tmp_path/'output', list(reversed(pieces)), **kwargs)
    with original.source_reader(tmp_path/'before', memory_mib=4, max_staging_mib=4) as reader:
        expected, known = reader(0,6)
    with result.source_reader(tmp_path/'after', memory_mib=4, max_staging_mib=4) as reader:
        actual, actual_known = reader(0,6)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_known, known)
    assert list((tmp_path/'before').iterdir()) == list((tmp_path/'after').iterdir()) == []


@pytest.mark.parametrize('case,message', [
    ('overlap','overlap|gaps'), ('gap','overlap|gaps'), ('short','complete native view'),
    ('xy','full native XY'), ('identity','identity'), ('source','source geometry'),
    ('empty','No disjoint'), ('native','exceeds native'), ('quantization','quantization'),
    ('block_size','block sizes'), ('known','known count'), ('negative','offset'),
])
def test_invalid_lease_contract_is_rejected_before_output(tmp_path, case, message):
    pieces, _, kwargs = fixture(tmp_path)
    if case == 'overlap': pieces[1]['offset_tyx'] = (1,0,0)
    elif case == 'gap': pieces.pop(1)
    elif case == 'short': pieces.pop()
    elif case == 'xy': pieces[1]['offset_tyx'] = (2,1,0)
    elif case == 'identity': kwargs['model_name'] = 'other'
    elif case == 'source': kwargs['source_shape'] = (6,13,18)
    elif case == 'empty': pieces.clear()
    elif case == 'native': kwargs['native_shape'] = (6,9,17)
    elif case == 'quantization': change_metadata(pieces[0], lambda m:m.__setitem__('quantization', 'different'))
    elif case == 'block_size': change_metadata(pieces[0], lambda m:m.__setitem__('block_size', 64))
    elif case == 'known': change_metadata(pieces[0], lambda m:m.__setitem__('known_voxels', 100000))
    elif case == 'negative': pieces[0]['offset_tyx'] = (-1,0,0)
    assert_rejected(tmp_path, pieces, kwargs, message)


@pytest.mark.parametrize('kind', ['truncated_payload', 'truncated_index', 'payload_checksum', 'index_checksum',
                                  'index_bounds', 'index_duplicate', 'index_offset'])
def test_corrupt_numeric_files_are_rejected_and_incomplete_output_removed(tmp_path, kind):
    pieces, _, kwargs = fixture(tmp_path)
    directory = pieces[0]['reference'].path
    payload, index = directory/'scores.u8.zlib', directory/'index.bin'
    if kind == 'truncated_payload':
        payload.write_bytes(payload.read_bytes()[:-1]); message = 'payload size'
    elif kind == 'truncated_index':
        index.write_bytes(index.read_bytes()[:-1]); message = 'index size'
    elif kind == 'payload_checksum':
        data = bytearray(payload.read_bytes()); data[-1] ^= 1; payload.write_bytes(data); message = 'checksum'
    elif kind == 'index_checksum':
        change_metadata(pieces[0], lambda m:m.__setitem__('index_sha256', '0'*64)); message = 'checksum'
    else:
        rows = np.fromfile(index, dtype=BLOCK_DTYPE)
        if kind == 'index_bounds': rows['z'][0] = 999; message = 'outside its native lease'
        elif kind == 'index_duplicate': rows['z'][1] = rows['z'][0]; message = 'repeat'
        else: rows['offset'][1] += 1; message = 'contiguous'
        index.write_bytes(rows.tobytes())
        digest = hashlib.sha256(index.read_bytes()).hexdigest()
        change_metadata(pieces[0], lambda m:m.__setitem__('index_sha256', digest))
    assert_rejected(tmp_path, pieces, kwargs, message)


def test_input_metadata_mutation_is_rejected_and_inputs_are_preserved(tmp_path):
    pieces, _, kwargs = fixture(tmp_path)
    original = subject._copy_payload
    def mutate(*args):
        original(*args)
        change_metadata(pieces[0], lambda m:m.__setitem__('changed', True))
    with mock.patch.object(subject, '_copy_payload', side_effect=mutate):
        assert_rejected(tmp_path, pieces, kwargs, 'metadata changed during')
    assert all((piece['reference'].path/'scores.u8.zlib').is_file() for piece in pieces)


def test_stale_reference_is_rejected(tmp_path):
    pieces, _, kwargs = fixture(tmp_path)
    path = pieces[0]['reference'].path/'metadata.json'
    data = json.loads(path.read_text()); data['changed'] = True; path.write_text(json.dumps(data))
    assert_rejected(tmp_path, pieces, kwargs, 'metadata changed before')


@pytest.mark.parametrize('failure', ['copy', 'index_publish', 'metadata_publish'])
def test_io_failures_leave_no_output_and_metadata_is_last(tmp_path, failure):
    pieces, _, kwargs = fixture(tmp_path)
    destination = tmp_path/'output'
    original = Path.replace
    def fail_publish(path, target):
        target = Path(target)
        if target.parent == destination and target.name == (
                'index.bin' if failure == 'index_publish' else 'metadata.json'):
            assert not (destination/'metadata.json').exists()
            assert (destination/'scores.u8.zlib').is_file()
            raise OSError('injected publication failure')
        return original(path, target)
    patch = (mock.patch.object(subject, '_copy_payload', side_effect=OSError('injected copy failure'))
             if failure == 'copy' else mock.patch.object(Path, 'replace', fail_publish))
    with patch, pytest.raises(OSError, match='injected'):
        write_native_disjoint_leases(destination, pieces, **kwargs)
    assert not destination.exists()


def test_existing_output_and_ancestor_or_descendant_paths_are_never_modified(tmp_path):
    pieces, _, kwargs = fixture(tmp_path)
    destination = tmp_path/'output'; destination.mkdir()
    marker = destination/'user.txt'; marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        write_native_disjoint_leases(destination, pieces, **kwargs)
    assert marker.read_text() == 'preserve'
    with pytest.raises(ValueError, match='separate'):
        write_native_disjoint_leases(pieces[0]['reference'].path/'nested', pieces, **kwargs)
    with pytest.raises(FileExistsError):
        write_native_disjoint_leases(tmp_path, pieces, **kwargs)
