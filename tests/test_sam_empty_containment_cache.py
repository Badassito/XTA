"""Containment reuse requires immutable identity, not merely a readonly flag."""
from collections.abc import Mapping

import numpy as np
import pytest

from XTA import sam_evidence as evidence


def fixture(empty, *, edge_count=8):
    shape = empty.shape
    reference = np.zeros(shape, bool)
    reference[0, 0] = True
    group = dict(group_id='group', context_bbox_yx=(0, 0, *shape), frame_indices=(0,),
        endpoints=[dict(observation_id=key, frame_index=0) for key in ('a', 'b')],
        edges=[dict(edge_id=f'e{index}', source_id='a', target_id='b') for index in range(edge_count)],
        complete=True, endpoint_identity_basis='slice_connected_components')
    values = {'write:0': np.zeros(shape, bool), 'acceptance:0': np.ones(shape, bool)}
    values.update({f'{role}:{key}': reference.copy()
                   for role in ('endpoint', 'evaluation') for key in ('a', 'b')})
    values.update({f'edge_write:e{index}:0': empty for index in range(edge_count)})
    return group, values


def immutable_empty(shape=(2, 4)):
    return np.frombuffer(bytes(int(np.prod(shape))), bool).reshape(shape)


@pytest.mark.parametrize('backing', ['bytes', 'mutable', 'readonly_mutable', 'readonly_memoryview', 'oversized_bytes'])
def test_only_exact_bytes_owner_reuses_empty_containment_scan(tmp_path, monkeypatch, backing):
    shape = (2, 4)
    if backing == 'bytes':
        empty = immutable_empty(shape)
    elif backing == 'oversized_bytes':
        empty = np.frombuffer(bytes(16), bool)[:8].reshape(shape)
    elif backing == 'readonly_memoryview':
        empty = np.frombuffer(memoryview(bytearray(8)).toreadonly(), bool).reshape(shape)
    else:
        empty = np.zeros(shape, bool)
        if backing == 'readonly_mutable':
            empty = empty.view()
            empty.setflags(write=False)
    group, values = fixture(empty)
    calls = []
    original = np.any
    def any_value(value, *args, **kwargs):
        calls.append(value is empty)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(evidence.np, 'any', any_value)
    with evidence.SamEvidenceWriter(tmp_path/backing, {}) as writer:
        writer.add_group(group, values)
    # Two endpoint repaint checks and one write/acceptance check always remain.
    assert len(calls) == (4 if backing == 'bytes' else 11)
    assert sum(calls) == (1 if backing == 'bytes' else 0)


class AdversarialMasks(Mapping):
    def __init__(self, values, on_second_edge_write):
        self.values = values
        self.hook = on_second_edge_write
        self.edge_reads = 0
        self.write_reads = 0
    def __iter__(self):
        return iter(self.values)
    def __len__(self):
        return len(self.values)
    def __getitem__(self, key):
        if key.startswith('edge_write:'):
            self.edge_reads += 1
        if key == 'write:0':
            self.write_reads += 1
            if self.edge_reads == 2:
                return self.hook(self.values[key])
        return self.values[key]


@pytest.mark.parametrize('failure', ['shape', 'binary', 'raise'])
def test_cached_empty_never_skips_custom_write_lookup_or_validation(tmp_path, failure):
    group, values = fixture(immutable_empty())
    def hook(write):
        if failure == 'shape':
            return np.zeros((1, 1), bool)
        if failure == 'binary':
            return np.full(write.shape, 2, np.uint8)
        raise RuntimeError('controlled custom mapping failure')
    masks = AdversarialMasks(values, hook)
    with evidence.SamEvidenceWriter(tmp_path/failure, {}) as writer:
        with pytest.raises((ValueError, RuntimeError)):
            writer.add_group(group, masks)
        assert not writer.records and not writer.groups
    assert masks.edge_reads == 2 and masks.write_reads == 5


@pytest.mark.parametrize('backing', ['writable', 'readonly_view', 'readonly_memoryview'])
def test_mutation_by_custom_write_lookup_is_still_checked(tmp_path, backing):
    source = bytearray(8) if backing == 'readonly_memoryview' else np.zeros((2, 4), bool)
    if backing == 'readonly_memoryview':
        empty = np.frombuffer(memoryview(source).toreadonly(), bool).reshape(2, 4)
    else:
        empty = source.view() if backing == 'readonly_view' else source
        if backing == 'readonly_view':
            empty.setflags(write=False)
    group, values = fixture(empty)
    def hook(write):
        if isinstance(source, bytearray):
            source[7] = 1
        else:
            source[1, 3] = True
        return write
    with evidence.SamEvidenceWriter(tmp_path/backing, {}) as writer:
        with pytest.raises(ValueError, match='subset of group write'):
            writer.add_group(group, AdversarialMasks(values, hook))
        assert not writer.records


def test_cached_array_shape_rebinding_is_not_hidden(tmp_path):
    empty = immutable_empty()
    group, values = fixture(empty)
    def hook(write):
        empty.shape = (4, 2)
        return write
    with evidence.SamEvidenceWriter(tmp_path/'shape-rebind', {}) as writer:
        with pytest.raises(ValueError):
            writer.add_group(group, AdversarialMasks(values, hook))
        assert not writer.records


def test_readonly_stride_change_cannot_expose_unvalidated_foreground(tmp_path):
    root = np.frombuffer(bytes(8)+bytes([1])*8, bool)
    empty = root[:8].reshape(2, 4)
    group, values = fixture(empty)
    def hook(write):
        empty.strides = (8, 1)
        return write
    with evidence.SamEvidenceWriter(tmp_path/'stride-rebind', {}) as writer:
        with pytest.raises(ValueError, match='subset of group write'):
            writer.add_group(group, AdversarialMasks(values, hook))
        assert not writer.records


def test_immutable_replacement_is_checked_again_and_rejects_foreground(tmp_path):
    empty = immutable_empty()
    group, values = fixture(empty)
    values['edge_write:e1:0'] = np.frombuffer(bytes([1])*8, bool).reshape(2, 4)
    with evidence.SamEvidenceWriter(tmp_path/'replacement', {}) as writer:
        with pytest.raises(ValueError, match='subset of group write'):
            writer.add_group(group, values)
        assert not writer.records


def test_cached_and_uncached_validation_produce_identical_evidence(tmp_path):
    empty = immutable_empty()
    group, values = fixture(empty)
    for name, masks in (('cached', values), ('uncached', {key:value.copy() for key,value in values.items()})):
        with evidence.SamEvidenceWriter(tmp_path/name, {'shape_tyx': (1, 2, 4)}) as writer:
            writer.add_group(group, masks)
            writer.commit()
    for name in ('masks.bin', 'index.json', 'manifest.json'):
        assert (tmp_path/'cached'/name).read_bytes() == (tmp_path/'uncached'/name).read_bytes()
