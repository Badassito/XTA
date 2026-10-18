"""Endpoint publication respects bounded frame-local contract caches."""
from collections.abc import Mapping

import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceWriter


class LazyMasks(Mapping):
    def __init__(self, values):
        self.values = values
        self.accessed = []

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, name):
        self.accessed.append(name)
        # Mirror a producer that owns only the plane currently requested.
        value = self.values[name]
        return np.frombuffer(value.tobytes(), dtype=bool).reshape(value.shape)


def fixture():
    shape = (5, 7)
    endpoints = [dict(observation_id=identifier, frame_index=frame)
                 for identifier, frame in (('a', 2), ('b', 0), ('c', 2), ('d', 0))]
    group = dict(group_id='group', context_bbox_yx=(0, 0, *shape),
        frame_indices=(0, 1, 2), endpoints=endpoints, edges=[], complete=True,
        endpoint_identity_basis='slice_connected_components')
    values = {f'{role}:{frame}': np.zeros(shape, bool)
              for role in ('acceptance', 'write') for frame in range(3)}
    for index, endpoint in enumerate(endpoints):
        mask = np.zeros(shape, bool)
        mask[index, index] = True
        for role in ('endpoint', 'evaluation', 'permitted'):
            values[f'{role}:{endpoint["observation_id"]}'] = mask.copy()
    return group, values


def test_writer_groups_hash_named_endpoint_roles_by_frame_and_preserves_exact_evidence(tmp_path):
    group, values = fixture()
    lazy = LazyMasks(values)
    with SamEvidenceWriter(tmp_path/'streamed', {'shape_tyx': (3, 5, 7)}) as writer:
        writer.add_group(group, lazy)
        bundle = writer.commit()
    for role in ('evaluation', 'permitted'):
        actual = [name for name in lazy.accessed if name.startswith(role + ':')]
        assert actual == [f'{role}:{identifier}' for identifier in ('b', 'd', 'a', 'c')]
    with SamEvidenceWriter(tmp_path/'eager', {'shape_tyx': (3, 5, 7)}) as writer:
        writer.add_group(group, values)
        writer.commit()
    for name in ('masks.bin', 'index.json', 'manifest.json'):
        assert (tmp_path/'streamed'/name).read_bytes() == (tmp_path/'eager'/name).read_bytes()
    assert set(bundle.groups['group']['mask_keys']) == set(values)
    for name, value in values.items():
        np.testing.assert_array_equal(bundle.group_mask('group', name), value)


def test_endpoint_metadata_order_does_not_change_packed_publication_order(tmp_path):
    group, values = fixture()
    for name, endpoints in (('forward', group['endpoints']), ('reverse', group['endpoints'][::-1])):
        with SamEvidenceWriter(tmp_path/name, {'shape_tyx': (3, 5, 7)}) as writer:
            writer.add_group({**group, 'endpoints': endpoints}, values)
            writer.commit()
    assert (tmp_path/'forward'/'masks.bin').read_bytes() == (tmp_path/'reverse'/'masks.bin').read_bytes()


def test_grouped_publication_keeps_strict_mask_shape_and_endpoint_identity_checks(tmp_path):
    group, values = fixture()
    bad = {**values, 'permitted:a': np.zeros((1, 1), bool)}
    with SamEvidenceWriter(tmp_path/'shape', {'shape_tyx': (3, 5, 7)}) as writer:
        with pytest.raises(ValueError, match='shape'):
            writer.add_group(group, bad)
    with SamEvidenceWriter(tmp_path/'duplicate', {'shape_tyx': (3, 5, 7)}) as writer:
        with pytest.raises(ValueError, match='Duplicate endpoint identity'):
            writer.add_group({**group, 'endpoints': group['endpoints'] + group['endpoints'][:1]}, values)


def test_all_declared_roles_share_frame_order_and_extensions_keep_lexical_fallback(tmp_path, monkeypatch):
    group, values = fixture()
    group['edges'] = [dict(edge_id='edge:alpha', source_id='b', target_id='a')]
    for frame in range(3):
        for role in ('unrelated', 'known_foreground'):
            values[f'{role}:{frame}'] = np.zeros((5, 7), bool)
        for role in ('edge_write', 'edge_contract'):
            values[f'{role}:edge:alpha:{frame}'] = np.zeros((5, 7), bool)
    extensions = ('custom:z', 'custom:a', 'acceptance:01', 'write:99',
                  'evaluation:unlisted', 'edge_write:unknown:1', 'edge_contract:edge:alpha:+1')
    for name in extensions:
        values[name] = np.zeros((5, 7), bool)
    lazy = LazyMasks(values)
    published, validation = [], []
    original = SamEvidenceWriter._put
    def put(writer, key, value, shape):
        if not published:
            validation.extend(lazy.accessed[:-1])  # The current plane is already fetched for _put.
        published.append(key.removeprefix('g/group/'))
        return original(writer, key, value, shape)
    monkeypatch.setattr(SamEvidenceWriter, '_put', put)
    with SamEvidenceWriter(tmp_path/'roles', {'shape_tyx': (3, 5, 7)}) as writer:
        writer.add_group(group, lazy)
        bundle = writer.commit()
    expected = []
    for frame, identifiers in ((0, ('b', 'd')), (1, ()), (2, ('a', 'c'))):
        expected += [f'acceptance:{frame}', f'edge_contract:edge:alpha:{frame}',
                     f'edge_write:edge:alpha:{frame}']
        expected += [f'endpoint:{identifier}' for identifier in identifiers]
        expected += [f'evaluation:{identifier}' for identifier in identifiers]
        expected += [f'known_foreground:{frame}']
        expected += [f'permitted:{identifier}' for identifier in identifiers]
        expected += [f'unrelated:{frame}', f'write:{frame}']
    assert published == expected + sorted(extensions)
    # The first validation pass must not revisit older frames, even when the
    # endpoint metadata arrives in a different order. Every edge is still read.
    write_frames = [int(name.partition(':')[2]) for name in validation if name.startswith('write:')]
    assert write_frames == sorted(write_frames)
    assert {f'edge_write:edge:alpha:{frame}' for frame in range(3)} <= set(validation)
    for name, value in values.items():
        np.testing.assert_array_equal(bundle.group_mask('group', name), value)


@pytest.mark.parametrize('invalid', ['late_empty_endpoint', 'edge_outside_write', 'write_outside_acceptance'])
def test_late_geometry_failure_never_publishes_earlier_valid_frames(tmp_path, monkeypatch, invalid):
    group, values = fixture()
    group['edges'] = [dict(edge_id='edge', source_id='b', target_id='a')]
    for frame in range(3):
        values[f'edge_write:edge:{frame}'] = np.zeros((5, 7), bool)
    if invalid == 'late_empty_endpoint':
        values['endpoint:a'][:] = False
    elif invalid == 'edge_outside_write':
        values['edge_write:edge:2'][4, 6] = True
    else:
        values['write:2'][4, 6] = True
    monkeypatch.setattr(SamEvidenceWriter, '_put',
        lambda *args, **kwargs: pytest.fail('Group geometry must validate before any plane is published'))
    with SamEvidenceWriter(tmp_path/invalid, {'shape_tyx': (3, 5, 7)}) as writer:
        with pytest.raises(ValueError):
            writer.add_group(group, values)


@pytest.mark.parametrize('contract', [
    {},
    {'endpoint_identity_basis': 'custom'},
    {'crop_contract': {'contract_materialization': 'streamed_planes'}},
    {'endpoint_identity_basis': 'frozen_baseline_slice_components', 'evidence_purpose': 'sam_extrapolation'},
    {'endpoint_identity_basis': 'slice_connected_components', 'evidence_purpose': 'sam_extrapolation'},
])
def test_noninterpolation_contracts_keep_lexical_roles_independently_of_mapping_type(tmp_path, contract):
    group, values = fixture()
    group.pop('endpoint_identity_basis')
    group.update(contract)
    for name, masks in (('eager', values), ('lazy', LazyMasks(values))):
        with SamEvidenceWriter(tmp_path/name, {'shape_tyx': (3, 5, 7)}) as writer:
            writer.add_group(group, masks)
            assert list(writer.groups['group']['mask_keys']) == sorted(values)
            writer.commit()
    for name in ('masks.bin', 'index.json', 'manifest.json'):
        assert (tmp_path/'eager'/name).read_bytes() == (tmp_path/'lazy'/name).read_bytes()
