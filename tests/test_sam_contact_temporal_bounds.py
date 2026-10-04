"""Temporal contact fast paths preserve the exact 6/18/26 voxel oracle."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage

from XTA import sam_policy


def _group(identifier, frames):
    return dict(group_id=identifier, frame_indices=list(frames), context_bbox_yx=[0, 0, 8, 8],
                endpoints=[dict(observation_id=identifier+':endpoint')])


def test_disjoint_temporal_neighborhoods_decode_no_masks(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError('Disjoint time intervals decoded a mask')
    monkeypatch.setattr(sam_policy, '_support_plane', forbid)
    bundle = SimpleNamespace(scope={})
    assert sam_policy._pair_contact(bundle, _group('A', [0, 1]), ['A'],
                                   _group('B', [3, 4]), ['B'], 26) is False


@pytest.mark.parametrize('connectivity', [6, 18, 26])
@pytest.mark.parametrize('delta_t,delta_y,delta_x', [(0, 0, 1), (0, 1, 1),
    (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (-1, 0, 0)])
def test_temporal_boundary_matches_full_3d_dilation_oracle(monkeypatch, connectivity, delta_t, delta_y, delta_x):
    a_frame, b_frame = 2, 2+delta_t
    first = np.zeros((8, 8), bool)
    second = np.zeros_like(first)
    first[3, 3] = True
    second[3+delta_y, 3+delta_x] = True
    def support(bundle, group, run_ids, frame, **kwargs):
        return first if group['group_id'] == 'A' else second
    monkeypatch.setattr(sam_policy, '_support_plane', support)
    actual = sam_policy._pair_contact(SimpleNamespace(scope={}), _group('A', [a_frame]), ['A'],
        _group('B', [b_frame]), ['B'], connectivity)
    source, target = np.zeros((6, 8, 8), bool), np.zeros((6, 8, 8), bool)
    source[a_frame] = first
    target[b_frame] = second
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    expected = bool(np.any(ndimage.binary_dilation(source, structure=structure) & target))
    assert actual is expected


def test_frames_without_neighbors_skip_support_before_decoding(monkeypatch):
    calls = []
    def support(bundle, group, run_ids, frame, **kwargs):
        calls.append((group['group_id'], frame))
        plane = np.zeros((8, 8), bool)
        if group['group_id'] == 'A':
            plane[3, 3] = True
        return plane
    monkeypatch.setattr(sam_policy, '_support_plane', support)
    assert not sam_policy._pair_contact(SimpleNamespace(scope={}), _group('A', range(20)), ['A'],
        _group('B', [10]), ['B'], 26)
    assert [frame for group, frame in calls if group == 'A'] == [9, 10, 11]
